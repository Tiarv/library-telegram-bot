#!/usr/bin/env python3
import os
import logging
import configparser
from pathlib import Path
import zipfile
import asyncio
import tempfile
import subprocess
import html
import json
import hashlib
import threading

from telegram.error import TelegramError
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    Application,
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
    "start",
    "lookup", "look", "search", "find", "l", "s", "f",
    "pick", "get", "p", "g",
    "info",
    "compare",
    "export", "catalog", "dump",
}

# Populated from bot.conf
ALLOWED_USER_IDS: set[int] = set()
INPX_FILES: list[str] = []
BOT_TOKEN: str | None = None
BOT_USERNAME: str | None = None

# Cache of last search results per (chat_id, user_id)
# key: (chat_id, user_id) -> list of match dicts
MATCH_CACHE: dict[tuple[int, int], list[dict]] = {}
INPX_FIELD_NAMES_CACHE: dict[str, list[str] | None] = {}
INPX_FIELD_NAMES_CACHE_LOCK = threading.Lock()

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
CATALOG_META_PATH = os.path.join(CACHE_DIR, "catalog_meta.json")

SEARCH_CACHE_PATH = os.path.join(CACHE_DIR, "search_cache.json")

# Global search cache: shared between all chats, keyed by (normalized pattern, max_matches)
SEARCH_CACHE: dict[str, dict] = {}
SEARCH_CACHE_GENERATION: str | None = None
SEARCH_CACHE_LOCK = threading.Lock()

MAX_MATCH_COLLECT = 9999
MAX_MATCH_DISPLAY = 9999

TELEGRAM_MAX_MESSAGE_LEN = 3900
TELEGRAM_HARD_LIMIT = 4096

MAX_CAPTION_LEN = 1024
CAPTION_HARD_LIMIT = 1024

CHECK_CONFIRM_THRESHOLD = 20
SEARCH_RESULTS_MESSAGE_DELAY_SECONDS = 2.0

SEPARATORS = ("\x04",)


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

    if len(caption) > MAX_CAPTION_LEN:
        caption = f"{prefix}\n(title: {title}, ext: {ext})"

        if len(caption) > MAX_CAPTION_LEN:
            caption = f"{prefix}\n(ext: {ext})"

    return caption


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
    global MAX_MATCH_COLLECT, MAX_MATCH_DISPLAY
    global TELEGRAM_MAX_MESSAGE_LEN, MAX_CAPTION_LEN
    global CHECK_CONFIRM_THRESHOLD, SEARCH_RESULTS_MESSAGE_DELAY_SECONDS

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

    # Optional tunable parameters (override defaults if present)
    try:
        MAX_MATCH_COLLECT = section.getint(
            "max_match_collect", fallback=MAX_MATCH_COLLECT
        )
        MAX_MATCH_DISPLAY = section.getint(
            "max_match_display", fallback=MAX_MATCH_DISPLAY
        )

        raw_msg_len = section.getint(
            "telegram_max_message_len", fallback=TELEGRAM_MAX_MESSAGE_LEN
        )
        # Guardrail: never let this exceed Telegram's hard limit
        if raw_msg_len > TELEGRAM_HARD_LIMIT:
            logger.warning(
                "telegram_max_message_len=%d is above TELEGRAM_HARD_LIMIT=%d; "
                "clamping to hard limit.",
                raw_msg_len,
                TELEGRAM_HARD_LIMIT,
            )
            TELEGRAM_MAX_MESSAGE_LEN = TELEGRAM_HARD_LIMIT
        else:
            TELEGRAM_MAX_MESSAGE_LEN = raw_msg_len

        raw_caption_len = section.getint(
            "max_caption_len", fallback=MAX_CAPTION_LEN
        )
        
        if raw_caption_len > CAPTION_HARD_LIMIT:
            logger.warning(
                "max_caption_len=%d is above CAPTION_HARD_LIMIT=%d; "
                "clamping to hard limit.",
                raw_caption_len,
                CAPTION_HARD_LIMIT,
            )
            MAX_CAPTION_LEN = CAPTION_HARD_LIMIT
        else:
            MAX_CAPTION_LEN = raw_caption_len

        CHECK_CONFIRM_THRESHOLD = section.getint(
            "check_confirm_threshold", fallback=CHECK_CONFIRM_THRESHOLD
        )
        SEARCH_RESULTS_MESSAGE_DELAY_SECONDS = section.getfloat(
            "search_results_message_delay_seconds",
            fallback=SEARCH_RESULTS_MESSAGE_DELAY_SECONDS,
        )
    except ValueError as e:
        raise RuntimeError(f"Invalid numeric value in [bot] section: {e}") from e
    
    logger.info(
        "Config loaded: %d allowed users, %d INPX files",
        len(ALLOWED_USER_IDS),
        len(INPX_FILES),
    )
    

def _read_inpx_field_names(inpx_path: str) -> list[str] | None:
    """
    Try to read structure.info from the given INPX archive and return
    a list of field names (in order). Returns None if not present or
    cannot be parsed.
    """
    # Fast path: cached? (protected by lock)
    with INPX_FIELD_NAMES_CACHE_LOCK:
        if inpx_path in INPX_FIELD_NAMES_CACHE:
            return INPX_FIELD_NAMES_CACHE[inpx_path]

    names: list[str] | None = None

    try:
        with zipfile.ZipFile(inpx_path, "r") as zf:
            struct_name: str | None = None
            for name in zf.namelist():
                if name.lower().endswith("structure.info"):
                    struct_name = name
                    break

            if not struct_name:
                names = None
            else:
                try:
                    raw = zf.read(struct_name)
                except Exception as e:
                    logger.warning(
                        "Failed to read structure.info from %s: %s",
                        inpx_path,
                        e,
                    )
                    names = None
                else:
                    text: str | None = None
                    for enc in ("utf-8", "cp1251"):
                        try:
                            text = raw.decode(enc)
                            break
                        except UnicodeDecodeError:
                            continue

                    if text is None:
                        logger.warning(
                            "Failed to decode structure.info in %s as utf-8 or cp1251",
                            inpx_path,
                        )
                        names = None
                    else:
                        header_line: str | None = None
                        for line in text.splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            if line.startswith("#") or line.startswith(";") or line.startswith("//"):
                                continue
                            header_line = line
                            break

                    if text is None:
                        logger.warning(
                            "Failed to decode structure.info in %s as utf-8 or cp1251",
                            inpx_path,
                        )
                        names = None
                    else:
                        header_line: str | None = None
                        for line in text.splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            if line.startswith("#") or line.startswith(";") or line.startswith("//"):
                                continue
                            header_line = line
                            break

                        if not header_line:
                            names = None
                        else:
                            # structure.info uses ';' as field separator
                            if ";" not in header_line:
                                logger.warning(
                                    "structure.info in %s does not contain ';' separator; "
                                    "field names will not be available",
                                    inpx_path,
                                )
                                INPX_FIELD_NAMES_CACHE[inpx_path] = None
                                return None

                            sep = ";"

                            # Preserve original casing but strip whitespace
                            parts_raw = [p.strip() for p in header_line.split(sep)]
                            if not any(parts_raw):
                                names = None
                            else:
                                names = [
                                    p if p else f"Field {i + 1}"
                                    for i, p in enumerate(parts_raw)
                                ]
    except Exception as e:
        logger.warning(
            "Failed to parse structure.info from %s: %s",
            inpx_path,
            e,
        )
        names = None

    # Store result under lock (including None)
    with INPX_FIELD_NAMES_CACHE_LOCK:
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

    candidates.append(os.path.abspath(os.path.join(inpx_dir, relpath_clean)))

    if parent_dir and parent_dir != inpx_dir:
        candidates.append(os.path.abspath(os.path.join(parent_dir, relpath_clean)))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    return None


def _check_deleted_and_maybe_parse(
    inpx_path: str,
    line: str,
    line_stripped: str,
    del_index: int | None,
) -> tuple[bool, dict | None]:
    """
    Deletion logic for a single INPX record line.

    Returns (is_deleted, parsed_record_or_none).

    - If del_index is not None:
        * Interpret parts[del_index] == "1" as a logical delete and skip.
    - If del_index is None (no structure.info / no DEL field):
        * Heuristic:
            1) Check field #9 (1-based; index 8). If it's "1":
                 - parse record, resolve container path, and if the book file
                   cannot be resolved -> treat as deleted.
            2) Otherwise, check the last field. If it's "1":
                 - same existence check.
        * We only skip if the flag is "1" AND the underlying file cannot be
          resolved. If file exists, we keep the record.
    """
    # We always need to be able to split the line into fields
    sep, parts = split_record(line_stripped)
    if parts is None:
        # Malformed line -> behave like "unusable record"
        return True, None

    # --- Case 1: proper DEL field from structure.info ---
    if del_index is not None:
        if len(parts) > del_index and parts[del_index].strip() == "1":
            # Catalog explicitly says this record is deleted
            return True, None
        # Not marked as deleted
        return False, None

    # --- Case 2: no structure.info / no DEL field: heuristic ---
    candidate_idx: int | None = None

    # 1-based field 9 -> index 8
    CANDIDATE_9TH = 8
    if len(parts) > CANDIDATE_9TH and parts[CANDIDATE_9TH].strip() == "1":
        candidate_idx = CANDIDATE_9TH
    else:
        # If 9th is not "1" (or doesn't exist), try last field
        last_idx = len(parts) - 1
        if last_idx >= 0 and parts[last_idx].strip() == "1":
            candidate_idx = last_idx

    if candidate_idx is None:
        # No candidate DEL bit we recognize -> keep record
        return False, None

    # We saw a "1" in a candidate DEL field – verify by checking book file existence
    parsed = parse_inpx_record(line)
    if not parsed:
        # Malformed record with DEL-like flag -> treat as deleted
        return True, None

    container_relpath = parsed["container_relpath"]
    container_abs = resolve_container_path(inpx_path, container_relpath)

    if not container_abs:
        # Flag is 1 and book file cannot be found -> treat as deleted
        return True, None

    # File exists, so ignore the flag; reuse parsed
    return False, parsed
    

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

    Deletion handling:

      1) If structure.info is available:
           - find field named DEL (case-insensitive);
           - if that field == "1" -> record is treated as deleted and skipped.

      2) If structure.info is NOT available:
           - look at field #9 (1-based) and then at the last field:
             * if such a field == "1", parse the record, resolve the book file;
             * if the file cannot be resolved, treat the record as deleted.
           - if the file exists, keep the record (even if the flag is "1").
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

        # Try to discover DEL field index from structure.info
        del_index: int | None = None
        field_names = _read_inpx_field_names(inpx_path)
        if field_names:
            for i, name in enumerate(field_names):
                if name.strip().upper() == "DEL":
                    del_index = i
                    break

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
                                if positive_needles and not all(
                                    n in line_cf for n in positive_needles
                                ):
                                    continue

                                # NOT filter: none of the negative needles may be present
                                if negative_needles and any(
                                    n in line_cf for n in negative_needles
                                ):
                                    continue

                                # --- deletion handling in helper ---
                                is_deleted, parsed_for_reuse = _check_deleted_and_maybe_parse(
                                    inpx_path=inpx_path,
                                    line=line,
                                    line_stripped=line_stripped,
                                    del_index=del_index,
                                )
                                if is_deleted:
                                    continue

                                # --- normal parse / add to results ---
                                parsed = parsed_for_reuse or parse_inpx_record(line)
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


def _get_catalog_generation_for_search_cache() -> str | None:
    """
    Read catalog_meta.json and return its 'generated_at' field as a string.
    Used to invalidate the search cache when catalog changes.

    If metadata is missing or unreadable, returns None.
    """
    try:
        if os.path.isfile(CATALOG_META_PATH):
            with open(CATALOG_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if isinstance(meta, dict):
                gen = meta.get("generated_at")
                if gen is not None:
                    return str(gen)
    except Exception as e:
        logger.warning("Failed to read catalog metadata for search cache: %s", e)
    return None


def _search_cache_key(pattern: str, max_matches: int) -> str:
    """
    Build a stable, order-insensitive key for the search cache.

    - Normalize whitespace and case.
    - Split into tokens and separate positive vs negative (prefixed with '-').
    - Deduplicate and sort tokens, so "lem epub" and "epub lem" hit the same key.
    """
    # Normalize case + whitespace
    tokens = [t for t in pattern.casefold().split() if t]

    positive: list[str] = []
    negative: list[str] = []

    for tok in tokens:
        if tok.startswith("-") and len(tok) > 1:
            negative.append(tok[1:])  # strip leading '-'
        else:
            positive.append(tok)

    # Deduplicate and sort so order doesn't matter
    positive_norm = sorted(set(positive))
    negative_norm = sorted(set(negative))

    # Rebuild canonical query form: all positives first, then negatives with '-'
    canonical_query = " ".join(
        positive_norm + [f"-{t}" for t in negative_norm]
    )

    raw_key = f"{canonical_query}||{max_matches}"
    h = hashlib.sha1(raw_key.encode("utf-8"), usedforsecurity=False).hexdigest()
    return h


def _load_search_cache_from_disk(current_generation: str | None) -> None:
    """
    Load search cache from SEARCH_CACHE_PATH if it matches current_generation.
    Otherwise, clear in-memory cache.
    """
    global SEARCH_CACHE, SEARCH_CACHE_GENERATION

    SEARCH_CACHE = {}
    SEARCH_CACHE_GENERATION = current_generation

    if not os.path.isfile(SEARCH_CACHE_PATH):
        return

    try:
        with open(SEARCH_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("Failed to load search cache: %s", e)
        return

    if not isinstance(data, dict):
        return

    file_gen = data.get("catalog_generation")
    entries = data.get("entries")

    # Invalidate if generations differ
    if file_gen != current_generation or not isinstance(entries, dict):
        return

    SEARCH_CACHE = entries


def _save_search_cache_to_disk() -> None:
    """
    Persist SEARCH_CACHE to SEARCH_CACHE_PATH together with the catalog generation.
    Ensure the file ends up with permissions 0640.
    """
    global SEARCH_CACHE_GENERATION

    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
    except OSError:
        # If we can't create cache dir, just skip persisting
        return

    payload = {
        "catalog_generation": SEARCH_CACHE_GENERATION,
        "entries": SEARCH_CACHE,
    }

    try:
        # Write JSON payload
        with open(SEARCH_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        # Enforce permissions 0640 (rw-r-----)
        try:
            os.chmod(SEARCH_CACHE_PATH, 0o640)
        except OSError as e:
            logger.warning(
                "Failed to chmod search cache file %s to 0640: %s",
                SEARCH_CACHE_PATH,
                e,
            )
    except Exception as e:
        logger.warning("Failed to save search cache: %s", e)


def run_search_with_persistent_cache(
    pattern: str,
    max_matches: int,
) -> tuple[list[dict], bool]:
    """
    Synchronous helper for use via asyncio.to_thread.

    - Uses a process-wide cache (SEARCH_CACHE) that is persisted to disk.
    - Cache is invalidated whenever catalog_meta.json 'generated_at' changes.
    - Falls back to search_in_inpx_records(...) on cache miss.
    """
    global SEARCH_CACHE, SEARCH_CACHE_GENERATION

    cache_key = _search_cache_key(pattern, max_matches)

    # --- fast path: generation check + cache lookup under lock ---
    with SEARCH_CACHE_LOCK:
        current_gen = _get_catalog_generation_for_search_cache()

        if SEARCH_CACHE_GENERATION != current_gen:
            _load_search_cache_from_disk(current_gen)

        entry = SEARCH_CACHE.get(cache_key)
        if isinstance(entry, dict):
            matches = entry.get("matches")
            truncated = bool(entry.get("truncated", False))
            if isinstance(matches, list):
                return matches, truncated

    # --- cache miss: run real search WITHOUT holding the lock ---
    matches, truncated = search_in_inpx_records(pattern, max_matches)

    # --- store result back under lock ---
    with SEARCH_CACHE_LOCK:
        # Generation might have changed while we were searching; re-check
        current_gen2 = _get_catalog_generation_for_search_cache()
        if SEARCH_CACHE_GENERATION != current_gen2:
            _load_search_cache_from_disk(current_gen2)

        SEARCH_CACHE[cache_key] = {
            "pattern": pattern,
            "matches": matches,
            "truncated": truncated,
        }
        SEARCH_CACHE_GENERATION = current_gen2
        _save_search_cache_to_disk()

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

    delete_trigger_message = False

    # Deep-link handling: /start get_<n>
    if context.args:
        arg = context.args[0]
        if arg.startswith("get_"):
            delete_trigger_message = True
            try:
                index = int(arg.split("_", 1)[1])
            except ValueError:
                if message:
                    await message.reply_text("Неверный параметр ссылки.")
                return

            # Pretend user ran "/get <index>"
            context.args = [str(index)]
            await pickfmt(update, context)  # your /get handler

            # Try to delete the triggering /start message
            if delete_trigger_message and message:
                try:
                    await message.delete()
                except TelegramError:
                    # no rights / too old / etc. – ignore quietly
                    pass

            return

    # Normal /start without deep-link: keep behaviour as-is
    await help_cmd(update, context)
    

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return

    help_text = (
        "/find <слова>      Поиск книги в каталогах по ключевым словам. Пробел = И, -слово = НЕ\n"        
        "/get <n> [формат]  Получить книгу по номеру #n из поиска.\n"
        "  Примеры:\n"
        "    /get 3         – отправит книгу как есть\n"
        "    /get 3 epub    – конвертирует книгу в EPUB\n"
        "  Поддерживаемые форматы включают: epub, fb2, pdf, txt, html, doc, mobi и другие\n"        
        "/info <n>          Показывает данные о книге по номеру #n из поиска.\n"
        "/compare a b       Сравнить данные о книгах #n1 и #n2 из поиска.\n"
        "/dump              Полный каталог всех книг (CSV в zip).\n"    )

    await message.reply_text(
        f"<pre>{html.escape(help_text)}</pre>",
        parse_mode=ParseMode.HTML,
    )


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or not message.text:
        return

    cmd_token = message.text.strip().split()[0]
    if not cmd_token.startswith("/"):
        return

    cmd = cmd_token[1:]
    if "@" in cmd:
        cmd = cmd.split("@", 1)[0]

    if cmd in KNOWN_COMMANDS:
        return

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
    max_line_len = max(100, TELEGRAM_MAX_MESSAGE_LEN - 200)

    for idx, rec in enumerate(matches[:MAX_MATCH_DISPLAY], start=1):
        author = rec.get("author") or "<?>"
        title = rec.get("title") or "<?>"
        ext = rec.get("ext") or "<?>"

        base_plain = f"{idx}) {author} — {title}"

        if len(base_plain) > max_line_len:
            base_plain = base_plain[: max_line_len - 1] + "…"

        if BOT_USERNAME and ext != "<?>":
            deeplink = f"https://t.me/{BOT_USERNAME}?start=get_{idx}"
            line_html = (
                f"{html.escape(base_plain)} "
                f"[<a href=\"{html.escape(deeplink)}\">{html.escape(ext)}</a>]"
            )
        else:
            line_html = f"{html.escape(base_plain)} [{html.escape(ext)}]"

        lines.append(line_html)

    shown = min(total_matches, MAX_MATCH_DISPLAY)

    header_plain = f"{header_prefix} ({total_matches}"
    if truncated:
        header_plain += f"+, search truncated at {MAX_MATCH_COLLECT}"
    header_plain += ").\n"

    header = html.escape(header_plain)

    first_chunk = True
    current = header
    cont_prefix = html.escape("…продолжение…") + "\n\n"

    for line in lines:
        candidate = current + line + "\n"

        if len(candidate) > TELEGRAM_MAX_MESSAGE_LEN and current.strip():
            text_to_send = current if first_chunk else cont_prefix + current

            # Hard safety: never send more than Telegram allows
            if len(text_to_send) > TELEGRAM_HARD_LIMIT:
                text_to_send = text_to_send[: TELEGRAM_HARD_LIMIT - 1]

            await message.reply_text(
                text_to_send,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
            if SEARCH_RESULTS_MESSAGE_DELAY_SECONDS > 0:
                await asyncio.sleep(SEARCH_RESULTS_MESSAGE_DELAY_SECONDS)
            first_chunk = False
            current = line + "\n"
        else:
            current = candidate

    if current.strip():
        text_to_send = current if first_chunk else cont_prefix + current

        if len(text_to_send) > TELEGRAM_HARD_LIMIT:
            text_to_send = text_to_send[: TELEGRAM_HARD_LIMIT - 1]

        await message.reply_text(
            text_to_send,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )


def dedupe_and_sort_matches(matches: list[dict]) -> list[dict]:
    """
    Collapse duplicates and sort for nicer display.

    Stage 1: collapse only *truly identical* records:
      - same INPX file
      - same index file inside INPX
      - same full list of fields

    Stage 2: if multiple records effectively point to the same physical
    book file (same FOLDER/its numeric analog + same LIBID/its numeric analog),
    keep only the first one.

    Then sort results to make similar items cluster together.
    """
    # --- Stage 1: exact duplicates within the same INPX/index ---
    by_key: dict[tuple, dict] = {}

    for rec in matches:
        fields = tuple(rec.get("fields") or [])
        key = (
            rec.get("inpx_path"),
            rec.get("index_inner_name"),
            fields,
        )
        if key in by_key:
            # exact duplicate of a record we've already seen in the same INPX/index
            continue
        by_key[key] = rec

    deduped = list(by_key.values())

    # --- Stage 2: duplicates by physical location (FOLDER + LIBID) ---
    # FOLDER's analog: container_relpath
    # LIBID's analog: lib_id
    seen_physical: set[tuple[str, str]] = set()
    deduped_physical: list[dict] = []

    for rec in deduped:
        folder = (rec.get("container_relpath") or "").strip()
        lib_id = (rec.get("lib_id") or "").strip()

        # Only consider as "same physical book" if we know BOTH
        if folder and lib_id:
            key2 = (folder.casefold(), lib_id.casefold())
            if key2 in seen_physical:
                # Another record that points to the same book file in the library structure
                # -> drop it, keep only the first one we saw
                continue
            seen_physical.add(key2)

        deduped_physical.append(rec)

    # --- Final sort: author → title → ext → container path → INPX name → lib_id ---
    def sort_key(r: dict) -> tuple:
        author = (r.get("author") or "").casefold()
        title = (r.get("title") or "").casefold()
        ext = (r.get("ext") or "").casefold()
        rel = (r.get("container_relpath") or "").casefold()
        inpx_name = os.path.basename(r.get("inpx_path") or "").casefold()
        lib_id = (r.get("lib_id") or "").casefold()
        return (author, title, ext, rel, inpx_name, lib_id)

    deduped_physical.sort(key=sort_key)
    return deduped_physical


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

    if context.args:
        args = list(context.args)
    else:
        text = (message.text or "").strip()
        args = text.split() if text else []

    if not args:
        await message.reply_text(
            "Please provide a search query, e.g.:\n"
            "  Asimov Robots\n"
            "  Asimov Robots epub\n"
        )
        return

    show_all = False
    if args and args[-1].lower() in ("--all", "+all"):
        show_all = True
        args = args[:-1]

    if not args:
        await message.reply_text("Использование: /find <ключевые слова>")
        return

    pattern = " ".join(args).strip()
    if not pattern:
        await message.reply_text("Необходимо указать ключевые слова.")
        return

    await message.reply_text(
        f"⌛ Ищу: {pattern}\n"
    )

    matches, truncated = await asyncio.to_thread(
        run_search_with_persistent_cache,
        pattern,
        MAX_MATCH_COLLECT,
    )

    if not matches:
        await message.reply_text("не найдено")
        return

    matches = dedupe_and_sort_matches(matches)
    total_matches = len(matches)

    key = _cache_key_from_update(update)
    if key is not None:
        MATCH_CACHE[key] = matches

    if total_matches > CHECK_CONFIRM_THRESHOLD and not show_all:
        header_lines = [
            f"Найдено {total_matches} подходящих записей. "
            "Уточните запрос или нажмите кнопку:"
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

    await send_matches_list(
        message=message,
        matches=matches,
        truncated=truncated,
        header_prefix="Matches",
    )
    

async def show_all_results_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    query = update.callback_query
    user = query.from_user if query else None
    if not user or user.id not in ALLOWED_USER_IDS:
        if query:
            await query.answer("Нет доступа", show_alert=True)
        return

    await query.answer()

    key = _cache_key_from_update(update)
    if key is None:
        await query.message.reply_text("Результаты поиска недоступны. Повторите поиск еще раз.")
        return

    matches = MATCH_CACHE.get(key)
    if not matches:
        await query.message.reply_text("Результаты поиска недоступны. Повторите поиск еще раз.")
        return

    await send_matches_list(query.message, matches, truncated=False, header_prefix="Все совпадения")


async def pickfmt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return

    if not context.args:
        await message.reply_text(
            "Использование: /get <номер> [формат]\n"
            "Пример:\n"
            "  /get 1        – отправит оригинальный файл\n"
            "  /get 1 epub   – конвертирует первую книгу в EPUB\n"
            "  /get 1 pdf    – конвертирует первую книгу в PDF"
        )
        return
        
    # Parse index
    try:
        index = int(context.args[0])
    except ValueError:
        await message.reply_text(
            "Первый аргумент должен быть числом, напр. /get 1 epub"
        )
        return

    # Optional format
    target_format_arg = context.args[1] if len(context.args) >= 2 else ""
    target_format = target_format_arg.strip().lstrip(".").lower()

    key = _cache_key_from_update(update)
    if key is None or key not in MATCH_CACHE:
        await message.reply_text(
            "Сначала выполните поиск\n"
        )
        return

    matches = MATCH_CACHE[key]
    if not 1 <= index <= len(matches):
        await message.reply_text(
            f"Поиск не содержит результата с таким номером: всего было найдено {len(matches)} книг"
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
        tmp_book_path, send_name = await asyncio.to_thread(
            extract_book_for_match,
            match,
        )

        if not tmp_book_path:
            await message.reply_text(
                "Запись была найдена, но книгу не получилось извлечь. Запросите другую книгу."
            )
            return

        try:
            with open(tmp_book_path, "rb") as f:
                await message.reply_document(
                    document=f,
                    filename=send_name or os.path.basename(tmp_book_path),
                    caption=build_safe_caption(
                        "found (original file)", match
                    ),
                )
        except Exception as e:
            logger.error(
                "Failed to send original book file %s (pickfmt): %s",
                tmp_book_path,
                e,
            )
            await message.reply_text(
                "Книга извлечена, но ее не получилось отправить :("
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
            f"Запись найдена, но извлечь или преобразовать книгу в формат {target_format!r} не получилось :("
        )
        return

    try:
        with open(tmp_out_path, "rb") as f:
            await message.reply_document(
                document=f,
                filename=send_name or os.path.basename(tmp_out_path),
                caption=build_safe_caption(
                    f"найдено (конвертировано в {target_format.upper()})", match
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
            f"Книга конвертирована в {target_format.upper()}, но ее не получилось отправить."
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
        await message.reply_text("Использование: /info <номер в поиске>")
        return

    try:
        index = int(context.args[0])
    except ValueError:
        await message.reply_text("Первый аргумент должен быть числом, напр. /info 1")
        return

    key = _cache_key_from_update(update)
    if key is None or key not in MATCH_CACHE:
        await message.reply_text(
            "Результаты поиска неизвестны. Выполните поиск."
        )
        return

    matches = MATCH_CACHE[key]
    if not 1 <= index <= len(matches):
        await message.reply_text(
            f"Поиск не содержит результата с таким номером: всего было найдено {len(matches)} книг"
        )
        return

    match = matches[index - 1]

    # Get fields from INPX
    fields = match.get("fields")
    if not fields:
        # Shouldn't happen after we changed parse_inpx_record, but be defensive
        await message.reply_text(
            "Метаданные отсутствуют в INPX-каталоге"
        )
        return

    # Compute size in background (extraction is I/O heavy)
    size_bytes = await asyncio.to_thread(get_book_size_for_match, match)
    size_str = (
        format_mb(size_bytes) if isinstance(size_bytes, int) else "неизвестно (не удалось извлечь)"
    )

    # Build human-readable info text
    lines: list[str] = []
    lines.append("Метаданные книги:")
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

        lines.append(f"{label}: {value}")

    lines.append("")
    lines.append(f"Контейнер каталога: {os.path.basename(match.get('inpx_path') or '<?>')}")
    lines.append(f"Файл индекса внутри контейнера: {match.get('index_inner_name') or '<?>'}")
    lines.append(f"Путь к контейнеру (относительный): {match.get('container_relpath') or '<?>'}")
    lines.append(f"Внутреннее имя книги: {match.get('inner_book_name') or '<?>'}")
    lines.append(f"Выявленное расширение: {match.get('ext') or('<?>' )}")
    lines.append("")
    lines.append(f"Действительный объем: {size_str}")

    text = "\n".join(lines)

    # --- NEW: chunk long /info output safely ---
    if len(text) > TELEGRAM_MAX_MESSAGE_LEN:
        for i in range(0, len(text), TELEGRAM_MAX_MESSAGE_LEN):
            chunk = text[i : i + TELEGRAM_MAX_MESSAGE_LEN]
            if i == 0:
                await message.reply_text(chunk)
            else:
                await message.reply_text("…продолжение…\n\n" + chunk)
    else:
        await message.reply_text(text)


async def compare_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /compare a b – compare INPX metadata for two results from the last search.
    """
    message = update.effective_message
    if message is None:
        return

    if len(context.args) < 2:
        await message.reply_text("Использование: /compare <номер первой книги из поиска> <номер второй книги из поиска>")
        return

    try:
        idx1 = int(context.args[0])
        idx2 = int(context.args[1])
    except ValueError:
        await message.reply_text(
            "Оба аргумента должны быть числами, напр. /compare 1 2"
        )
        return

    key = _cache_key_from_update(update)
    if key is None or key not in MATCH_CACHE:
        await message.reply_text(
            "Сначала выполните поиск."
        )
        return

    matches = MATCH_CACHE[key]
    total = len(matches)
    if not (1 <= idx1 <= total) or not (1 <= idx2 <= total):
        await message.reply_text(
            f"Поиск не содержит результата с таким номером: всего было найдено {len(matches)} книг."
        )
        return

    match1 = matches[idx1 - 1]
    match2 = matches[idx2 - 1]

    fields1 = match1.get("fields") or []
    fields2 = match2.get("fields") or []

    if not fields1 and not fields2:
        await message.reply_text("Метаданные для данных книг не найдены.")
        return

    inpx1 = match1.get("inpx_path") or ""
    inpx2 = match2.get("inpx_path") or ""
    names1 = _read_inpx_field_names(inpx1) if inpx1 else None
    names2 = _read_inpx_field_names(inpx2) if inpx2 else None

    max_len = max(len(fields1), len(fields2))
    diff_lines: list[str] = []

    for i in range(1, max_len + 1):
        v1 = fields1[i - 1].strip() if i <= len(fields1) else ""
        v2 = fields2[i - 1].strip() if i <= len(fields2) else ""
        if v1 == v2:
            continue

        label = None
        if names1 and i <= len(names1) and names1[i - 1].strip():
            label = names1[i - 1].strip()
        if not label and names2 and i <= len(names2) and names2[i - 1].strip():
            label = names2[i - 1].strip()
        if not label:
            label = f"Field {i}"

        if not v1:
            v1 = "«empty»"
        if not v2:
            v2 = "«empty»"

        diff_lines.append(f"{label}:")
        diff_lines.append(f"  #{idx1}: {v1}")
        diff_lines.append(f"  #{idx2}: {v2}")
        diff_lines.append("")

    header_lines: list[str] = []
    header_lines.append(
        f"Сравнение книг #{idx1} и #{idx2} из последнего поиска:"
    )

    def summarize(rec: dict, idx: int) -> None:
        author = rec.get("author") or "<?>"
        title = rec.get("title") or "<?>"
        ext = rec.get("ext") or "<?>"
        inpx_name = os.path.basename(rec.get("inpx_path") or "") or "<?>"
        rel = rec.get("container_relpath") or "<?>"
        inner = rec.get("inner_book_name") or "<?>"
        header_lines.append(f"  #{idx}: {author} — {title} {ext}")
        header_lines.append(f"       Индекс: {inpx_name}")
        header_lines.append(f"       Путь: {rel}")
        header_lines.append(f"       Файл: {inner}")

    summarize(match1, idx1)
    summarize(match2, idx2)

    header_lines.append("")

    if not diff_lines:
        header_lines.append(
            "Различий не обнаружено"
        )
        text = "\n".join(header_lines)
    else:
        header_lines.append("Различающиеся поля:")
        header_lines.append("")
        text = "\n".join(header_lines + diff_lines)

    if len(text) > TELEGRAM_MAX_MESSAGE_LEN:
        chunks = [
            text[i : i + TELEGRAM_MAX_MESSAGE_LEN]
            for i in range(0, len(text), TELEGRAM_MAX_MESSAGE_LEN)
        ]
        for i, chunk in enumerate(chunks):
            if i == 0:
                await message.reply_text(chunk)
            else:
                await message.reply_text("…продолжение…\n\n" + chunk)
    else:
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
            "Экспорт из каталога сейчас недоступен.\n"
            "Повторите попытку завтра."
        )
        return

    meta = None
    try:
        if os.path.isfile(CATALOG_META_PATH):
            with open(CATALOG_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
    except Exception as e:
        logger.warning("Failed to load catalog metadata in bot: %s", e)

    all_files = sorted(
        name
        for name in os.listdir(CACHE_DIR)
        if name.startswith("catalog-part") and name.endswith(".csv.zip")
    )

    if not all_files:
        await message.reply_text(
            "Экспорт каталога сейчас недоступен.\n"
            "Повторите попытку завтра."
        )
        return

    total_parts = len(all_files)
    generated_at = None
    if meta and isinstance(meta, dict):
        generated_at = meta.get("generated_at")

    header_msg = f"Отправляю каталог в {total_parts} частях."
    if generated_at:
        header_msg += f"\nGenerated at: {generated_at}"
    await message.reply_text(header_msg)

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
                f"Не получилось послать том каталога {idx} ({name})."
            )


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


async def post_init(application: Application) -> None:
    # Only the "canonical" commands; aliases still work but won't be suggested
    commands = [
        BotCommand("find", "Искать книгу по ключевым словам. Пробел = И, -слово = НЕ"),
        BotCommand("get", "Получить книгу по номеру в поиске"),
        BotCommand("info", "Получить информацию о книге по номеру в поиске"),
        BotCommand("compare", "Сравнить данные двух книг по их номерам в поиске"),
        BotCommand("dump", "Получить полный каталог"),
    ]
    await application.bot.set_my_commands(commands)

    global BOT_USERNAME
    me = await application.bot.get_me()
    BOT_USERNAME = me.username

def main() -> None:
    load_config()

    application = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    # Only allow whitelisted users AND only in private chats
    private_filter = filters.ChatType.PRIVATE
    allowed_users_filter = filters.User(user_id=ALLOWED_USER_IDS) & private_filter

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
        CommandHandler("compare", compare_cmd, filters=allowed_users_filter)
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

    # Log every update
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
            check_inpx,
        ),
    )

    application.add_error_handler(error_handler)

    application.run_polling(drop_pending_updates=True)



if __name__ == "__main__":
    main()
