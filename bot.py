#!/usr/bin/env python3
import os
import logging
import configparser
from pathlib import Path
import zipfile
import asyncio
import tempfile
import subprocess


from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Populated from bot.conf
ALLOWED_USER_IDS: set[int] = set()
INPX_FILES: list[str] = []
BOT_TOKEN: str | None = None

# Cache of last search results per (chat_id, user_id)
# key: (chat_id, user_id) -> list of match dicts
MATCH_CACHE: dict[tuple[int, int], list[dict]] = {}
INPX_SCHEMA_CACHE: dict[str, dict[str, int]] = {}
MAX_MATCH_COLLECT = 9999
MAX_MATCH_DISPLAY = 9999
TELEGRAM_MAX_MESSAGE_LEN = 3900
MAX_CAPTION_LEN = 3001

SEPARATORS = ("\x04", "\t", ";", "|")

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
            parts = line.split(sep)
            return sep, parts
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


def parse_inpx_record(line: str) -> dict | None:
    """
    Fixed schema:

      1: author(s)
      3: title
      6: filename (no ext)
      8: library id
      10: extension
      13: container path

    Returns dict or None.
    """
    _, parts = split_record(line)
    if parts is None or len(parts) < 13:
        return None

    author = parts[0].strip()
    title = parts[2].strip()
    filename = parts[5].strip()
    ext = parts[9].strip()
    container_relpath = parts[12].strip()

    if not filename or not ext or not container_relpath:
        return None

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
        "container_relpath": container_relpath,
        "inner_book_name": inner_book_name,
        "ext": ext,
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

    If pattern has multiple words separated by spaces, ALL of them must
    be present (logical AND), e.g.:

      "lem epub" -> line must contain "lem" AND "epub".
    """
    matches: list[dict] = []
    truncated = False

    if not INPX_FILES:
        logger.warning("No INPX files configured")
        return matches, truncated

    # Split query into tokens, ignore empty bits
    needles = [t for t in pattern.casefold().split() if t]

    if not needles:
        return matches, truncated

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

                                # AND filter: all needles must be present
                                if not all(n in line_cf for n in needles):
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
    message = update.effective_message
    if message is None:
        return

    user = update.effective_user
    await message.reply_text(
        f"Hi, {user.first_name or 'there'}! 👋\n"
        f"You are on the whitelist, so I will talk to you."
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    await message.reply_text(
        "/start – say hi\n"
        "/help or /h – this help\n"
        "/lookup, /search, /find (or /l, /s, /f) <pattern> – "
        "search inside INPX (multiple words = AND filter).\n"
        "/pick, /get (or /p, /g) <n> [format] – send n-th result from the last search.\n"
        "  Examples:\n"
        "    /p 3         – send original file\n"
        "    /p 3 epub    – convert to EPUB\n"
        "    /p 3 pdf     – convert to PDF\n"
        "Send any text and I'll echo it back."
    )

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return

    text = message.text or ""
    await message.reply_text(f"You said:\n{text}")


def _cache_key_from_update(update: Update) -> tuple[int, int] | None:
    chat = update.effective_chat
    user = update.effective_user
    if chat is None or user is None:
        return None
    return (chat.id, user.id)


async def check_inpx(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Search for a pattern inside the contents of all configured .inpx archives.

    - Case-insensitive substring search on raw INPX lines
    - Multiple words act as an AND filter (all must be present)
    - If 0 matches: reply 'not found'
    - If >1 matches: list author/title/id/ext + INPX file, do NOT send any book
      and store matches in cache for /pick
    - If exactly 1 match: extract and send that book
    """
    message = update.effective_message
    if message is None:
        return

    if not context.args:
        await message.reply_text("Usage: /lookup <pattern>")
        return

    pattern = " ".join(context.args).strip()
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

    # Cache results per chat+user for /pick
    key = _cache_key_from_update(update)
    if key is not None:
        MATCH_CACHE[key] = matches

    # Multiple matches: show list, no file yet
    if len(matches) > 1:
        lines: list[str] = []
        for idx, rec in enumerate(matches[:MAX_MATCH_DISPLAY], start=1):
            author = rec.get("author") or "<?>"
            title = rec.get("title") or "<?>"
            ext = rec.get("ext") or "<?>"
            inpx_name = os.path.basename(rec.get("inpx_path") or "") or "<?>"
            lines.append(
                f"{idx}) {author} — {title} [ext: {ext}] (in {inpx_name})"
            )

        shown = min(len(matches), MAX_MATCH_DISPLAY)

        header = f"Multiple matches ({len(matches)}"
        if truncated:
            header += f"+, search truncated at {MAX_MATCH_COLLECT}"
        header += ").\n"
        header += (
            f"Showing first {shown} result(s). "
            "Please refine your query or choose one with /pick <number>.\n\n"
        )

        full_text = header + "\n".join(lines)

        # Split into safe chunks for Telegram
        chunks = split_text_for_telegram(full_text, TELEGRAM_MAX_MESSAGE_LEN)

        for i, chunk in enumerate(chunks):
            if i == 0:
                await message.reply_text(chunk)
            else:
                await message.reply_text("…continued…\n\n" + chunk)

        return

    # Exactly one match: extract and send
    match = matches[0]

    tmp_book_path, send_name = await asyncio.to_thread(
        extract_book_for_match,
        match,
    )

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
        CommandHandler(["help", "h"], help_cmd, filters=allowed_users_filter)
    )
    application.add_handler(
        CommandHandler(
            ["lookup", "look", "search", "find", "l", "s", "f"],
            check_inpx,
            filters=allowed_filter,
        )
    )
    application.add_handler(
        CommandHandler(
            ["pick", "get", "p", "g"],
            pickfmt,
            filters=allowed_filter,
        )
    )
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND & allowed_users_filter,
            echo,
        )
    )

    application.add_error_handler(error_handler)

    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
