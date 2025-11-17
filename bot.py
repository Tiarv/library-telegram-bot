#!/usr/bin/env python3
import os
import logging
import configparser
from pathlib import Path
import zipfile
import asyncio
import time
import tempfile
import subprocess


from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

KNOWN_COMMANDS = {
    # search / lookup aliases
    "lookup", "look", "search", "find", "l", "s", "f",
    # pick/get aliases
    "pick", "get", "p", "g",
    # info
    "info",
    # export / catalog / dump aliases
    "export", "catalog", "dump",
}

# Populated from bot.conf
ALLOWED_USER_IDS: set[int] = set()
INPX_FILES: list[str] = []
BOT_TOKEN: str | None = None

# Cache of last search results per (chat_id, user_id)
# key: (chat_id, user_id) -> list of match dicts
MATCH_CACHE: dict[tuple[int, int], list[dict]] = {}
INPX_SCHEMA_CACHE: dict[str, dict[str, int]] = {}
INPX_FIELD_NAMES_CACHE: dict[str, list[str]] = {}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
CATALOG_META_PATH = os.path.join(CACHE_DIR, "catalog_meta.json")

MAX_MATCH_COLLECT = 9999
MAX_MATCH_DISPLAY = 9999
TELEGRAM_MAX_MESSAGE_LEN = 3900
MAX_CAPTION_LEN = 3001
CHECK_CONFIRM_THRESHOLD = 20  # how many matches to list without asking
# Delay between sending consecutive search-result messages (seconds).
# Set to 0 to disable throttling.
SEARCH_RESULTS_MESSAGE_DELAY_SECONDS = 2.0

SEPARATORS = ("\x04", "\t", ";", "|")


def format_mb(bytes_size: int) -> str:
    """
    Format bytes as a human-readable string in megabytes with 2 decimals.
    """
    mb = bytes_size / (1024 * 1024)
    return f"{mb:.2f} MB"


def get_book_size_for_match(match: dict) -> int | None:
    """
    Use extract_book_for_match to get a temp file and return its size in bytes.
    The temp file is deleted afterwards.
    """
    tmp_book_path, _ = extract_book_for_match(match)
    if not tmp_book_path:
        return None

    try:
        size_bytes = os.path.getsize(tmp_book_path)
    except OSError:
        size_bytes = None
    finally:
        try:
            os.remove(tmp_book_path)
        except OSError:
            pass

    return size_bytes


def build_safe_caption(prefix: str, match: dict) -> str:
    """
    Build a Telegram-safe caption from match fields:
      - prefix (e.g. 'found' or 'found (converted to EPUB)')
      - author, title, id, ext

    It trims very long fields so the final caption stays under MAX_CAPTION_LEN.
    """
    author = (match.get("author") or "").strip()
    title = (match.get("title") or "").strip()
    ext = (match.get("ext") or "").strip()

    def shorten(s: str, max_len: int) -> str:
        if len(s) <= max_len:
            return s
        return s[: max_len - 1] + "…"

    # Reasonable per-field limits
    author = shorten(author, 1500)
    title = shorten(title, 1500)

    caption = f"{prefix}\n(author: {author}, title: {title}, ext: {ext})"

    # Extra safety: if somehow still too long, drop title completely
    if len(caption) > MAX_CAPTION_LEN:
        caption = f"{prefix}\n(ext: {ext})"

    return caption


def split_text_for_telegram(text: str, limit: int = TELEGRAM_MAX_MESSAGE_LEN) -> list[str]:
    """
    Split text into chunks that fit within Telegram's message size limit.

    Tries to split on newline boundaries; if a single line is still too long,
    it will hard-split that line.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""

    for line in text.splitlines(keepends=True):
        # If one line is longer than the limit, hard-split it
        while len(line) > limit:
            part = line[:limit]
            line = line[limit:]
            if current:
                chunks.append(current.rstrip())
                current = ""
            chunks.append(part.rstrip())

        if len(current) + len(line) > limit:
            chunks.append(current.rstrip())
            current = line
        else:
            current += line

    if current.strip():
        chunks.append(current.rstrip())

    return chunks


def split_record(line: str):
    """
    Try to detect a separator and split the line.
    Returns (sep, parts) or (None, None) if no separator found.
    """
    line = line.rstrip("\r\n")
    for sep in SEPARATORS:
        if sep in line:
            return sep, line.split(sep)
    return None, None


def load_config() -> None:
    """Load config from bot.conf in the same directory as this script."""
    global ALLOWED_USER_IDS, INPX_FILES, BOT_TOKEN

    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "bot.conf"

    if not config_path.is_file():
        raise RuntimeError(f"Config file not found: {config_path}")

    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    if "bot" not in cfg:
        raise RuntimeError("[bot] section missing in config file")

    section = cfg["bot"]

    # Token
    BOT_TOKEN = section.get("token", "").strip()
    if not BOT_TOKEN:
        raise RuntimeError("token is missing or empty in [bot] section")

    # Allowed user IDs: comma-separated
    raw_ids = section.get("allowed_user_ids", "").strip()
    if not raw_ids:
        raise RuntimeError("allowed_user_ids is missing or empty in [bot] section")

    ids: list[int] = []
    for part in raw_ids.replace("\n", " ").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            raise RuntimeError(f"Invalid user id in allowed_user_ids: {part!r}")

    ALLOWED_USER_IDS = set(ids)

    # INPX files: comma-separated, optional quotes, can contain spaces
    raw_inpx = section.get("inpx_files", "").strip()
    files: list[str] = []
    if raw_inpx:
        for part in raw_inpx.split(","):
            part = part.strip()
            if not part:
                continue

            # Strip optional surrounding quotes
            if (part.startswith('"') and part.endswith('"')) or (
                part.startswith("'") and part.endswith("'")
            ):
                part = part[1:-1].strip()

            if part:
                files.append(part)

    INPX_FILES = files

    logger.info(
        "Config loaded: %d allowed users, %d INPX files",
        len(ALLOWED_USER_IDS),
        len(INPX_FILES),
    )


def load_inpx_schema(inpx_path: str) -> dict[str, int]:
    """
    Try to load field positions from structure.info inside the INPX file.

    If structure.info is missing or cannot be parsed, fall back to the
    legacy hardcoded schema:

        1: author
        3: title
        6: filename (no ext)
        10: ext
        13: container filename (relative)

    Returns a dict with 0-based indexes:
      {
        "author_idx": int,
        "title_idx": int,
        "filename_idx": int,
        "ext_idx": int,
        "container_idx": int,
      }
    """

    # Return cached version if already loaded
    if inpx_path in INPX_SCHEMA_CACHE:
        return INPX_SCHEMA_CACHE[inpx_path]

    # Default / legacy schema (0-based indices)
    default_schema: dict[str, int] = {
        "author_idx": 0,      # field 1
        "title_idx": 2,       # field 3
        "filename_idx": 5,    # field 6
        "ext_idx": 9,         # field 10
        "container_idx": 12,  # field 13
    }

    def use_default(reason: str):
        logger.info(
            "Using default INPX schema for %s (%s)",
            inpx_path,
            reason,
        )
        INPX_SCHEMA_CACHE[inpx_path] = default_schema
        return default_schema

    try:
        with zipfile.ZipFile(inpx_path, "r") as zf:
            # structure.info might be at root or in some subdir; pick first match
            struct_name = None
            for name in zf.namelist():
                if name.lower().endswith("structure.info"):
                    struct_name = name
                    break

            if not struct_name:
                return use_default("structure.info not found")

            try:
                raw = zf.read(struct_name)
            except Exception as e:
                logger.warning(
                    "Failed to read structure.info from %s: %s",
                    inpx_path,
                    e,
                )
                return use_default("read error")

            text = raw.decode("utf-8", errors="ignore")
            header_line = None
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#") or line.startswith("//"):
                    continue
                header_line = line
                break

            if not header_line:
                return use_default("empty structure.info")

            # Detect separator in header_line
            sep = None
            parts = None
            for candidate in ("\x04", ";", "|", "\t", ","):
                if candidate in header_line:
                    sep = candidate
                    parts = [p.strip().lower() for p in header_line.split(candidate)]
                    break

            if parts is None:
                return use_default("no recognizable separator in structure.info")

            def find_index(candidates: list[str]) -> int | None:
                for i, name in enumerate(parts):
                    for cand in candidates:
                        if cand in name:
                            return i
                return None

            author_idx = find_index(["author"])
            title_idx = find_index(["title"])
            filename_idx = find_index(["file", "filename"])
            ext_idx = find_index(["ext", "extension"])
            container_idx = find_index(["container", "arc", "zip", "archive", "folder"])

            # We need at least filename, ext, container to work. Others can fall back.
            if filename_idx is None or ext_idx is None or container_idx is None:
                return use_default("required fields not found in structure.info")

            # For optional ones, fall back to defaults if missing
            if author_idx is None:
                author_idx = default_schema["author_idx"]
            if title_idx is None:
                title_idx = default_schema["title_idx"]

            schema = {
                "author_idx": author_idx,
                "title_idx": title_idx,
                "filename_idx": filename_idx,
                "ext_idx": ext_idx,
                "container_idx": container_idx,
            }

            logger.info(
                "Loaded INPX schema from structure.info for %s: %s",
                inpx_path,
                schema,
            )
            INPX_SCHEMA_CACHE[inpx_path] = schema
            return schema

    except Exception as e:
        logger.warning(
            "Error while trying to load structure.info from %s: %s",
            inpx_path,
            e,
        )
        return use_default("exception")


def _read_inpx_field_names(inpx_path: str) -> list[str] | None:
    """
    Try to read structure.info from the given INPX archive and return
    a list of field names (in order). Returns None if not present or
    cannot be parsed.
    """
    # Cached?
    if inpx_path in INPX_FIELD_NAMES_CACHE:
        return INPX_FIELD_NAMES_CACHE[inpx_path]

    names: list[str] | None = None

    try:
        with zipfile.ZipFile(inpx_path, "r") as zf:
            # structure.info is the conventional name
            if "structure.info" not in zf.namelist():
                INPX_FIELD_NAMES_CACHE[inpx_path] = None
                return None

            with zf.open("structure.info", "r") as f:
                raw = f.read()

        # Try UTF-8 first, fall back to cp1251 (common for Russian INPX files)
        text: str
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("cp1251")
            except UnicodeDecodeError:
                logger.warning(
                    "Failed to decode structure.info in %s as utf-8 or cp1251",
                    inpx_path,
                )
                INPX_FIELD_NAMES_CACHE[inpx_path] = None
                return None

        # Find the first non-empty, non-comment line that looks like a structure
        # (semicolon-separated field names)
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Treat lines starting with # or ; as comments
            if line.startswith("#") or line.startswith(";"):
                continue
            if ";" not in line:
                continue

            # This is probably the structure line
            parts = [p.strip() for p in line.split(";") if p.strip()]
            if parts:
                names = parts
                break

    except Exception as e:
        logger.warning(
            "Failed to read structure.info from %s: %s",
            inpx_path,
            e,
        )

    INPX_FIELD_NAMES_CACHE[inpx_path] = names
    return names


def parse_inpx_record(line: str) -> dict | None:
    """
    Fixed schema:

      1: author(s)
      3: title
      6: filename (no ext)
      8: library id
      10: extension
      13: container path

    Returns dict with:
      {
        "author": str,
        "title": str,
        "lib_id": str,
        "container_relpath": str,
        "inner_book_name": str,
        "ext": str,
        "fields": list[str],
      }
    or None on failure.
    """
    _, parts = split_record(line)
    if parts is None:
        return None

    # Need at least 13 fields (1-based -> index 12)
    if len(parts) < 13:
        return None

    # Fixed positions (0-based indices)
    author = parts[0].strip()              # field 1
    title = parts[2].strip()               # field 3
    filename = parts[5].strip()            # field 6
    lib_id = parts[7].strip()              # field 8
    ext = parts[9].strip()                 # field 10
    container_relpath = parts[12].strip()  # field 13

    if not filename or not ext or not container_relpath:
        return None

    # Normalize ext and build inner_book_name
    ext_clean = ext.lstrip(".")
    if not ext_clean:
        return None

    lower_fname = filename.lower()
    lower_suffix = "." + ext_clean.lower()
    if not lower_fname.endswith(lower_suffix):
        inner_book_name = f"{filename}.{ext_clean}"
    else:
        inner_book_name = filename

    return {
        "author": author,
        "title": title,
        "lib_id": lib_id,
        "container_relpath": container_relpath,
        "inner_book_name": inner_book_name,
        "ext": ext,
        "fields": parts,  # full raw fields for /info
    }


def resolve_container_path(inpx_path: str, relpath: str) -> str | None:
    """
    Given an INPX file path and a relative container path from a record, try to resolve
    the real container path:

    1) <inpx_dir>/<relpath>
    2) <parent_of_inpx_dir>/<relpath>

    Returns absolute path if the file exists, otherwise None.
    Does NOT scan the filesystem recursively.
    """
    inpx_abs = os.path.abspath(inpx_path)
    inpx_dir = os.path.dirname(inpx_abs)
    parent_dir = os.path.dirname(inpx_dir)

    relpath_clean = relpath.lstrip("/\\")

    candidates: list[str] = []

    # Same directory as INPX
    candidates.append(os.path.abspath(os.path.join(inpx_dir, relpath_clean)))

    # One level above
    if parent_dir and parent_dir != inpx_dir:
        candidates.append(os.path.abspath(os.path.join(parent_dir, relpath_clean)))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    return None


def search_in_inpx_records(pattern: str, max_collect: int = 100):
    """
    Plain case-insensitive substring search in raw INPX lines.

    - Multiple words separated by spaces form an AND filter:
        all of them must be present.
    - Words prefixed with '-' form a NOT filter:
        none of them may be present.

    Examples:
      "lem epub"       -> line must contain "lem" AND "epub"
      "lem epub -fb2"  -> line must contain "lem" AND "epub"
                          and must NOT contain "fb2".
    """
    matches: list[dict] = []
    truncated = False

    if not INPX_FILES:
        logger.warning("No INPX files configured")
        return matches, truncated

    # Split query into tokens, ignore empty bits
    tokens = [t for t in pattern.casefold().split() if t]

    if not tokens:
        return matches, truncated

    positive_needles: list[str] = []
    negative_needles: list[str] = []

    for tok in tokens:
        # tokens starting with '-' are treated as NOT filters
        if tok.startswith("-") and len(tok) > 1:
            negative_needles.append(tok[1:])
        else:
            positive_needles.append(tok)


    for inpx_path in INPX_FILES:
        inpx_path = inpx_path.strip()
        if not inpx_path:
            continue

        if not os.path.isfile(inpx_path):
            logger.warning("INPX file not found: %s", inpx_path)
            continue

        try:
            with zipfile.ZipFile(inpx_path, "r") as zf:
                for index_inner_name in zf.namelist():
                    try:
                        with zf.open(index_inner_name, "r") as f:
                            for raw_line in f:
                                try:
                                    line = raw_line.decode("utf-8", errors="ignore")
                                except Exception:
                                    continue

                                line_stripped = line.strip()
                                if not line_stripped:
                                    continue

                                line_cf = line.casefold()

                                # AND filter: all positive needles must be present
                                if positive_needles and not all(n in line_cf for n in positive_needles):
                                    continue

                                # NOT filter: none of the negative needles may be present
                                if negative_needles and any(n in line_cf for n in negative_needles):
                                    continue

                                parsed = parse_inpx_record(line)
                                if not parsed:
                                    # you can keep this quiet if it’s too chatty
                                    # logger.warning("Could not parse INPX record in %s -> %s",
                                    #                inpx_path, index_inner_name)
                                    continue

                                record = {
                                    **parsed,
                                    "inpx_path": inpx_path,
                                    "index_inner_name": index_inner_name,
                                }
                                matches.append(record)

                                if len(matches) >= max_collect:
                                    truncated = True
                                    return matches, truncated
                    except Exception as e:
                        logger.warning(
                            "Failed to read inner file %s in %s: %s",
                            index_inner_name,
                            inpx_path,
                            e,
                        )
                        continue
        except Exception as e:
            logger.warning("Failed to open INPX file %s: %s", inpx_path, e)
            continue

    return matches, truncated


def extract_and_convert_to_format(match: dict, target_format: str):
    """
    For a single INPX match:
      1) Extract the original book file (using extract_book_for_match)
      2) Convert it to the given format via `ebook-convert`
      3) Return (converted_path, converted_filename_for_telegram) on success
         or (None, None) on failure.

    Cleans up temp files appropriately.
    """
    target_format = (target_format or "").strip().lstrip(".").lower()
    if not target_format:
        return None, None

    tmp_book_path, send_name = extract_book_for_match(match)
    if not tmp_book_path:
        return None, None

    # Derive a nice output filename: base + .<format>
    base_name = os.path.splitext(send_name or os.path.basename(tmp_book_path))[0]
    out_filename = f"{base_name}.{target_format}"

    # Create a temp path for the converted file
    fd, tmp_out_path = tempfile.mkstemp(suffix=f".{target_format}")
    os.close(fd)

    try:
        result = subprocess.run(
            ["ebook-convert", tmp_book_path, tmp_out_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            logger.error(
                "ebook-convert failed for %s -> %s: %s",
                tmp_book_path,
                tmp_out_path,
                result.stderr,
            )
            try:
                os.remove(tmp_out_path)
            except OSError:
                pass
            try:
                os.remove(tmp_book_path)
            except OSError:
                pass
            return None, None

    except FileNotFoundError:
        logger.error("ebook-convert binary not found in PATH")
        try:
            os.remove(tmp_out_path)
        except OSError:
            pass
        try:
            os.remove(tmp_book_path)
        except OSError:
            pass
        return None, None
    except Exception as e:
        logger.error(
            "Error running ebook-convert on %s -> %s: %s",
            tmp_book_path,
            tmp_out_path,
            e,
        )
        try:
            os.remove(tmp_out_path)
        except OSError:
            pass
        try:
            os.remove(tmp_book_path)
        except OSError:
            pass
        return None, None

    # Conversion succeeded; we can delete the original temp file
    try:
        os.remove(tmp_book_path)
    except OSError:
        pass

    return tmp_out_path, out_filename


def extract_book_for_match(match: dict):
    """
    For a single INPX match dict, resolve the container and extract the inner book
    into a temporary file.

    Returns (tmp_path, send_name) on success, or (None, None) on failure.
    """
    inpx_path = match["inpx_path"]
    container_relpath = match["container_relpath"]
    inner_book_name = match["inner_book_name"]

    container_path = resolve_container_path(inpx_path, container_relpath)
    if not container_path:
        logger.warning(
            "Container not found for relpath=%s (from %s -> %s)",
            container_relpath,
            inpx_path,
            match["index_inner_name"],
        )
        return None, None

    try:
        with zipfile.ZipFile(container_path, "r") as cf:
            member_name = None

            # Exact match first
            try:
                cf.getinfo(inner_book_name)
                member_name = inner_book_name
            except KeyError:
                # Fallback by basename
                for m in cf.namelist():
                    if os.path.basename(m) == inner_book_name:
                        member_name = m
                        break

            if not member_name:
                logger.warning(
                    "Inner book %s not found in container %s",
                    inner_book_name,
                    container_path,
                )
                return None, None

            data = cf.read(member_name)

            suffix = os.path.splitext(inner_book_name)[1] or ".bin"
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(fd, "wb") as out:
                out.write(data)

            logger.info(
                "Extracted book: %s (from container %s; INPX %s -> %s)",
                tmp_path,
                container_path,
                inpx_path,
                match["index_inner_name"],
            )

            return tmp_path, inner_book_name
    except Exception as e:
        logger.warning(
            "Failed to open or read container %s: %s",
            container_path,
            e,
        )
        return None, None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Just show the help text when user hits /start
    await help_cmd(update, context)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    await message.reply_text(
        "/find <pattern> – search inside INPX (multiple words = AND filter, "
        "-word = NOT filter, +all to show all matches).\n"
        "/get <n> [format] – send n-th result from the last search; "
        "optional format like epub, pdf, etc.\n"
        "  Examples:\n"
        "    /get 3         – send original file\n"
        "    /get 3 epub    – convert to EPUB\n"
        "    /get 3 pdf     – convert to PDF\n"
        "/info <n> – show all INPX fields and file size for the n-th result.\n"
        "/dump – send full catalog export as a zipped CSV, "
        "if it has been generated by the background job.\n"
    )


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or not message.text:
        return

    # First token is the command (e.g. "/find@MyBot", "/foo")
    cmd_token = message.text.strip().split()[0]
    if not cmd_token.startswith("/"):
        # Not a command, ignore
        return

    # Strip leading '/' and optional @botname
    cmd = cmd_token[1:]
    if "@" in cmd:
        cmd = cmd.split("@", 1)[0]

    # If this is one of our known commands, do nothing
    if cmd in KNOWN_COMMANDS:
        return

    # Otherwise, treat as an unknown command and show help
    await help_cmd(update, context)
    

def _cache_key_from_update(update: Update) -> tuple[int, int] | None:
    chat = update.effective_chat
    user = update.effective_user
    if chat is None or user is None:
        return None
    return (chat.id, user.id)


async def send_matches_list(
    message,
    matches: list[dict],
    truncated: bool,
    header_prefix: str = "Multiple matches",
) -> None:
    total_matches = len(matches)
    if total_matches == 0:
        await message.reply_text("not found")
        return

    lines: list[str] = []
    for idx, rec in enumerate(matches[:MAX_MATCH_DISPLAY], start=1):
        author = rec.get("author") or "<?>"
        title = rec.get("title") or "<?>"
        ext = rec.get("ext") or "<?>"
        lines.append(f"{idx}) {author} — {title} {ext}")

    shown = min(total_matches, MAX_MATCH_DISPLAY)

    header = f"{header_prefix} ({total_matches}"
    if truncated:
        header += f"+, search truncated at {MAX_MATCH_COLLECT}"
    header += ").\n"
    header += (
        f"Showing first {shown} result(s). "
        "Please refine your query or choose one with /get <number>.\n\n"
    )

    full_text = header + "\n".join(lines)

    chunks = split_text_for_telegram(full_text, TELEGRAM_MAX_MESSAGE_LEN)

    for i, chunk in enumerate(chunks):
        text = chunk if i == 0 else "…continued…\n\n" + chunk
        await message.reply_text(text)
        if i < len(chunks) - 1 and SEARCH_RESULTS_MESSAGE_DELAY_SECONDS > 0:
            await asyncio.sleep(SEARCH_RESULTS_MESSAGE_DELAY_SECONDS)



async def check_inpx(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Search for a pattern inside the contents of all configured .inpx archives.

    - Case-insensitive substring search on raw INPX lines
    - Multiple words act as an AND filter (all must be present)
    - If 0 matches: reply 'not found'
    - If ≥1 matches: list author/title/ext, do NOT send any book
      and store matches in cache for /get
    """
    message = update.effective_message
    if message is None:
        return

    if not context.args:
        await message.reply_text("Usage: /lookup <pattern>")
        return

    # Detect optional --all / +all at the end
    show_all = False
    args = list(context.args)
    if args and args[-1].lower() in ("--all", "+all"):
        show_all = True
        args = args[:-1]

    if not args:
        await message.reply_text("Usage: /lookup <pattern>")
        return

    # Keep this string so we can show the user how to re-run with --all
    original_pattern_for_echo = " ".join(args).strip()

    pattern = original_pattern_for_echo
    if not pattern:
        await message.reply_text("Pattern must not be empty.")
        return

    # Run heavy search in a thread
    matches, truncated = await asyncio.to_thread(
        search_in_inpx_records,
        pattern,
        MAX_MATCH_COLLECT,
    )
    
    if not matches:
        await message.reply_text("not found")
        return

    total_matches = len(matches)

    # Cache results per chat+user
    key = _cache_key_from_update(update)
    if key is not None:
        MATCH_CACHE[key] = matches

    # If many results and no explicit +all, ask for confirmation
    if total_matches > CHECK_CONFIRM_THRESHOLD and not show_all:
        header_lines = [
            f"Found {total_matches} matching record(s). "
            "Refine your query or press the button below:"
        ]

        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        text="Show all results (+all)",
                        callback_data="show_all_results",
                    )
                ]
            ]
        )

        await message.reply_text(
            "\n".join(header_lines),
            reply_markup=keyboard,
        )
        return

    # One or more matches: show list, no file yet
    await send_matches_list(
        message=message,
        matches=matches,
        truncated=truncated,
        header_prefix="Matches",
    )
    return

    if not tmp_book_path:
        await message.reply_text(
            "A single match was found in the index, but the book could not be extracted."
        )
        return

    try:
        with open(tmp_book_path, "rb") as f:
            await message.reply_document(
                document=f,
                filename=send_name or os.path.basename(tmp_book_path),
                caption=build_safe_caption("found", match),
            )

    except Exception as e:
        logger.error("Failed to send book file %s: %s", tmp_book_path, e)
        await message.reply_text(
            "Match was found and extracted, but failed to send the book file."
        )
    finally:
        try:
            os.remove(tmp_book_path)
        except OSError:
            pass


async def show_all_results_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    query = update.callback_query
    await query.answer()

    key = _cache_key_from_update(update)
    if key is None:
        await query.message.reply_text("No cached search results available.")
        return

    matches = MATCH_CACHE.get(key)
    if not matches:
        await query.message.reply_text("No cached search results available.")
        return

    # We don't know if the original search was truncated; assume False here.
    # If you store `truncated` per key as well, you can pass the real value.
    await send_matches_list(
        message=query.message,
        matches=matches,
        truncated=False,
        header_prefix="All matches",
    )


async def pickfmt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /pickfmt <n> [format] or /pf <n> [format] –
    send the n-th result from the last /checkinpx search.

    - If format is omitted: send original file (like /pick).
    - If format equals the original extension: send original file.
    - Otherwise: convert via ebook-convert to the requested format.
    """
    message = update.effective_message
    if message is None:
        return

    if not context.args:
        await message.reply_text(
            "Usage: /pick <number> [format]\n"
            "Examples:\n"
            "  /pickup 1        – send original file\n"
            "  /pickup 1 epub   – convert to EPUB\n"
            "  /pickup 1 pdf    – convert to PDF"
        )
        return
        
    # Parse index
    try:
        index = int(context.args[0])
    except ValueError:
        await message.reply_text(
            "First argument must be a number, e.g. /pick 1 epub"
        )
        return

    # Optional format
    target_format_arg = context.args[1] if len(context.args) >= 2 else ""
    target_format = target_format_arg.strip().lstrip(".").lower()

    key = _cache_key_from_update(update)
    if key is None or key not in MATCH_CACHE:
        await message.reply_text(
            "I don’t have any recent search results for you. "
            "Run /lookup <pattern> first."
        )
        return

    matches = MATCH_CACHE[key]
    if not 1 <= index <= len(matches):
        await message.reply_text(
            f"Choice out of range. You have {len(matches)} stored result(s)."
        )
        return

    match = matches[index - 1]

    # Determine original format from match["ext"]
    orig_ext_raw = (match.get("ext") or "").strip()
    orig_format = orig_ext_raw.lstrip(".").lower()

    # Decide whether to convert or send as-is:
    # - no target_format given -> send original
    # - target_format == original -> send original
    # - else -> convert
    send_original = False
    if not target_format:
        send_original = True
    elif orig_format and target_format == orig_format:
        send_original = True

    if send_original:
        # Behave like /pick: extract and send original file
        tmp_book_path, send_name = await asyncio.to_thread(
            extract_book_for_match,
            match,
        )

        if not tmp_book_path:
            await message.reply_text(
                "This record was found, but the book could not be extracted."
            )
            return

        try:
            with open(tmp_book_path, "rb") as f:
                await message.reply_document(
                    document=f,
                    filename=send_name or os.path.basename(tmp_book_path),
                    caption=build_safe_caption(
                        "found (by /pick, original file)", match
                    ),
                )
        except Exception as e:
            logger.error(
                "Failed to send original book file %s (pickfmt): %s",
                tmp_book_path,
                e,
            )
            await message.reply_text(
                "Book was extracted, but I failed to send the file."
            )
        finally:
            try:
                os.remove(tmp_book_path)
            except OSError:
                pass

        return

    # Otherwise: convert to requested format
    tmp_out_path, send_name = await asyncio.to_thread(
        extract_and_convert_to_format,
        match,
        target_format,
    )

    if not tmp_out_path:
        await message.reply_text(
            f"This record was found, but I couldn't extract or convert the book to {target_format!r}."
        )
        return

    try:
        with open(tmp_out_path, "rb") as f:
            await message.reply_document(
                document=f,
                filename=send_name or os.path.basename(tmp_out_path),
                caption=build_safe_caption(
                    f"found (converted to {target_format.upper()})", match
                ),
            )
    except Exception as e:
        logger.error(
            "Failed to send converted file %s (pickfmt, %s): %s",
            tmp_out_path,
            target_format,
            e,
        )
        await message.reply_text(
            f"Book was converted to {target_format.upper()}, but I failed to send the file."
        )
    finally:
        try:
            os.remove(tmp_out_path)
        except OSError:
            pass


async def info_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /info <n> – show all INPX fields for the n-th result from the last search
    and the actual book file size in MB.
    """
    message = update.effective_message
    if message is None:
        return

    if not context.args:
        await message.reply_text("Usage: /info <number>")
        return

    try:
        index = int(context.args[0])
    except ValueError:
        await message.reply_text("First argument must be a number, e.g. /info 1")
        return

    key = _cache_key_from_update(update)
    if key is None or key not in MATCH_CACHE:
        await message.reply_text(
            "I don’t have any recent search results for you. "
            "Run /lookup <pattern> first."
        )
        return

    matches = MATCH_CACHE[key]
    if not 1 <= index <= len(matches):
        await message.reply_text(
            f"Choice out of range. You have {len(matches)} stored result(s)."
        )
        return

    match = matches[index - 1]

    # Get fields from INPX
    fields = match.get("fields")
    if not fields:
        # Shouldn't happen after we changed parse_inpx_record, but be defensive
        await message.reply_text(
            "Detailed INPX fields were not stored for this result."
        )
        return

    # Compute size in background (extraction is I/O heavy)
    size_bytes = await asyncio.to_thread(get_book_size_for_match, match)
    size_str = (
        format_mb(size_bytes) if isinstance(size_bytes, int) else "unknown (could not extract)"
    )

    # Build human-readable info text
    lines = []
    lines.append("INPX record details:")
    lines.append("")

    inpx_path = match.get("inpx_path") or ""
    field_names = _read_inpx_field_names(inpx_path) if inpx_path else None

    for i, value in enumerate(fields, start=1):
        value = value.strip()
        if not value:
            value = "«empty»"

        if field_names and i <= len(field_names):
            label = field_names[i - 1]
        else:
            label = f"Field {i}"

        # You can normalize label if you want: label = label.lower()
        lines.append(f"{label}: {value}")

    lines.append("")
    lines.append(f"Container INPX: {os.path.basename(match.get('inpx_path') or '<?>')}")
    lines.append(f"Index file inside INPX: {match.get('index_inner_name') or '<?>'}")
    lines.append(f"Container path (relative): {match.get('container_relpath') or '<?>'}")
    lines.append(f"Inner book name: {match.get('inner_book_name') or '<?>'}")
    lines.append(f"Detected extension: {match.get('ext') or('<?>' )}")
    lines.append("")
    lines.append(f"Actual book size: {size_str}")

    text = "\n".join(lines)
    await message.reply_text(text)


async def export_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /export, /catalog, /dump

    Send all catalog-part*.csv.zip files from cache as separate documents.
    The files are generated by the external generate_catalog.py script.
    """
    message = update.effective_message
    if message is None:
        return

    if not os.path.isdir(CACHE_DIR):
        await message.reply_text(
            "Catalog export is not available right now.\n"
            "The background job has not generated any parts yet."
        )
        return

    # Try to load metadata to get generated_at and part names
    meta = None
    try:
        if os.path.isfile(CATALOG_META_PATH):
            with open(CATALOG_META_PATH, "r", encoding="utf-8") as f:
                import json
                meta = json.load(f)
    except Exception as e:
        logger.warning("Failed to load catalog metadata in bot: %s", e)

    # List all catalog-part*.csv.zip in cache
    all_files = sorted(
        name
        for name in os.listdir(CACHE_DIR)
        if name.startswith("catalog-part") and name.endswith(".csv.zip")
    )

    if not all_files:
        await message.reply_text(
            "Catalog export is not available right now.\n"
            "The background job has not generated any parts yet."
        )
        return

    total_parts = len(all_files)
    generated_at = None
    if meta and isinstance(meta, dict):
        generated_at = meta.get("generated_at")

    header_msg = f"Sending catalog export in {total_parts} part(s)."
    if generated_at:
        header_msg += f"\nGenerated at: {generated_at}"
    await message.reply_text(header_msg)

    # Send each part as a separate document
    for idx, name in enumerate(all_files, start=1):
        path = os.path.join(CACHE_DIR, name)
        try:
            st = os.stat(path)
            size_mb = st.st_size / (1024 * 1024)
            caption = (
                f"Catalog export – part {idx} of {total_parts}\n"
                f"Size: {size_mb:.2f} MB"
            )
        except OSError:
            caption = f"Catalog export – part {idx} of {total_parts}"

        try:
            with open(path, "rb") as f:
                await message.reply_document(
                    document=f,
                    filename=name,
                    caption=caption,
                )
        except Exception as e:
            logger.error("Failed to send catalog part %s: %s", path, e)
            await message.reply_text(
                f"Failed to send catalog part {idx} ({name})."
            )


async def log_unauthorized(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    chat = update.effective_chat
    message = update.effective_message

    user_id = user.id if user else None
    username = user.username if user else None
    first_name = getattr(user, "first_name", None) if user else None
    last_name = getattr(user, "last_name", None) if user else None

    chat_id = chat.id if chat else None
    chat_type = chat.type if chat else None

    text = None
    if message:
        if message.text:
            text = message.text
        elif message.caption:
            text = message.caption

    logger.warning(
        "Unauthorized update: user_id=%s username=%r first_name=%r last_name=%r "
        "chat_id=%s chat_type=%s text=%r",
        user_id,
        username,
        first_name,
        last_name,
        chat_id,
        chat_type,
        text,
    )

    return


async def log_any_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    chat = update.effective_chat
    message = update.effective_message

    user_id = user.id if user else None
    username = user.username if user else None
    first_name = getattr(user, "first_name", None) if user else None
    last_name = getattr(user, "last_name", None) if user else None

    chat_id = chat.id if chat else None
    chat_type = chat.type if chat else None

    text = None
    content_type = None

    if message:
        if message.text:
            text = message.text
            content_type = "text"
        elif message.caption:
            text = message.caption
            # some kind of media with caption
            if message.document:
                content_type = "document"
            elif message.photo:
                content_type = "photo"
            elif message.audio:
                content_type = "audio"
            elif message.video:
                content_type = "video"
            else:
                content_type = "media"
        else:
            # non-text, non-caption content
            if message.document:
                content_type = "document"
            elif message.photo:
                content_type = "photo"
            elif message.audio:
                content_type = "audio"
            elif message.video:
                content_type = "video"
            elif message.sticker:
                content_type = "sticker"
            else:
                content_type = "other"

    logger.info(
        "Incoming update: user_id=%s username=%r first_name=%r last_name=%r "
        "chat_id=%s chat_type=%s content_type=%r text=%r",
        user_id,
        username,
        first_name,
        last_name,
        chat_id,
        chat_type,
        content_type,
        text,
    )

    # IMPORTANT: do not reply from here; this is logging-only
    return


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling an update:", exc_info=context.error)


def main() -> None:
    load_config()

    application = ApplicationBuilder().token(BOT_TOKEN).build()

    allowed_users_filter = filters.User(user_id=ALLOWED_USER_IDS)

    application.add_handler(
        CommandHandler("start", start, filters=allowed_users_filter)
    )
    application.add_handler(
        CommandHandler(
            ["lookup", "look", "search", "find", "l", "s", "f"],
            check_inpx,
            filters=allowed_users_filter,
        )
    )
    application.add_handler(
        CommandHandler(
            ["pick", "get", "p", "g"],
            pickfmt,
            filters=allowed_users_filter,
        )
    )
    application.add_handler(
        CommandHandler("info", info_cmd, filters=allowed_users_filter)
    )
    application.add_handler(
        CommandHandler(
            ["export", "catalog", "dump"],
            export_cmd,
            filters=allowed_users_filter,
        )
    )
    application.add_handler(
        CallbackQueryHandler(
            show_all_results_callback,
            pattern="^show_all_results$",
            block=False,
        )
    )
    # Log every update (allowed and unauthorized users)
    application.add_handler(
        MessageHandler(filters.ALL, log_any_update),
        group=-1,
    )
    application.add_handler(
        MessageHandler(
            filters.COMMAND & allowed_users_filter,
            unknown_command,
        ),
        group=1,
    )
    application.add_handler(
    MessageHandler(
        filters.TEXT & ~filters.COMMAND & allowed_users_filter,
        help_cmd,
    ),
    group=1,
    )
    
    application.add_error_handler(error_handler)

    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
