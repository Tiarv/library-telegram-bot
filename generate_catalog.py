#!/usr/bin/env python3
import csv
import json
import os
import stat
import sys
import time
import zipfile
import tempfile
import logging
import configparser
from typing import Any, Dict, List, Tuple
from collections import Counter

os.umask(0o027)

ROWS_PER_PART = 800_000

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("catalog_generator")

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
CATALOG_META_PATH = os.path.join(CACHE_DIR, "catalog_meta.json")
CATALOG_VERSION = 1  # bump if CSV schema changes

# --- INPX config loading (similar to bot.conf logic) ---

def load_inpx_files_from_config(config_path: str) -> List[str]:
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    read_files = cfg.read(config_path)
    if not read_files:
        logger.error("Could not read config file: %s", config_path)
        sys.exit(1)

    if "bot" not in cfg:
        logger.error("[bot] section missing in config.")
        sys.exit(1)

    section = cfg["bot"]

    raw_inpx = section.get("inpx_files", "").strip()
    if not raw_inpx:
        logger.error("inpx_files is missing or empty in [bot] section.")
        sys.exit(1)

    inpx_files: List[str] = []

    # Each line may contain one or more comma-separated entries,
    # optionally quoted, optionally with trailing commas and comments.
    for line in raw_inpx.splitlines():
        line = line.strip()
        if not line:
            continue

        # Strip inline comments
        for comment_char in ("#", ";"):
            if comment_char in line:
                line = line.split(comment_char, 1)[0].strip()
        if not line:
            continue

        # Now split by commas in case the user used comma-separated style
        for part in line.split(","):
            part = part.strip()
            if not part:
                continue

            # Strip optional surrounding quotes
            if (part.startswith('"') and part.endswith('"')) or (
                part.startswith("'") and part.endswith("'")
            ):
                part = part[1:-1].strip()

            if not part:
                continue

            inpx_files.append(os.path.expanduser(part))

    return inpx_files

# --- Simple INPX record splitting (same as in bot) ---

SEPARATORS = ("\x04", "\t", ";", "|")

def split_record(line: str) -> Tuple[str | None, List[str] | None]:
    line = line.rstrip("\r\n")
    for sep in SEPARATORS:
        if sep in line:
            return sep, line.split(sep)
    return None, None

# --- Metadata helpers ---

def get_inpx_signatures(inpx_files: List[str]) -> List[Dict[str, Any]]:
    sigs: List[Dict[str, Any]] = []
    for path in inpx_files:
        path = path.strip()
        if not path:
            continue
        if not os.path.isfile(path):
            logger.warning("INPX file not found while scanning: %s", path)
            continue
        try:
            st = os.stat(path)
        except OSError as e:
            logger.warning("Failed to stat INPX file %s: %s", path, e)
            continue
        sigs.append(
            {
                "path": path,
                "size": st.st_size,
                "mtime": st.st_mtime,
            }
        )
    return sigs

def load_catalog_meta() -> Dict[str, Any] | None:
    if not os.path.isfile(CATALOG_META_PATH):
        return None
    try:
        with open(CATALOG_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            return None
        return meta
    except Exception as e:
        logger.warning("Failed to load catalog metadata: %s", e)
        return None

def save_catalog_meta(meta: Dict[str, Any]) -> None:
    os.makedirs(CACHE_DIR, mode=0o770, exist_ok=True)
    tmp_path = CATALOG_META_PATH + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, CATALOG_META_PATH)
    except Exception as e:
        logger.error("Failed to save catalog metadata: %s", e)
        try:
            os.remove(tmp_path)
        except OSError:
            pass

def catalog_cache_is_valid(current_sigs: List[Dict[str, Any]]) -> bool:
    meta = load_catalog_meta()
    if not meta:
        return False

    if meta.get("version") != CATALOG_VERSION:
        return False

    prev_sigs = meta.get("inpx_files")
    if not isinstance(prev_sigs, list):
        return False

    prev_map = {e["path"]: (e["size"], e["mtime"]) for e in prev_sigs if "path" in e}
    curr_map = {e["path"]: (e["size"], e["mtime"]) for e in current_sigs if "path" in e}

    if prev_map.keys() != curr_map.keys():
        return False

    for path, sig in curr_map.items():
        if prev_map.get(path) != sig:
            return False

    # Optional: require at least some parts info
    parts = meta.get("parts")
    if not isinstance(parts, list) or not parts:
        return False

    return True

def read_inpx_field_names_from_structure(inpx_path: str) -> list[str] | None:
    """
    Try to read field names from structure.info inside a given INPX archive.
    Returns a list of names (in order) or None if not present/parsable.
    """
    try:
        with zipfile.ZipFile(inpx_path, "r") as zf:
            if "structure.info" not in zf.namelist():
                return None
            with zf.open("structure.info", "r") as f:
                raw = f.read()
    except Exception as e:
        logger.warning("Failed to read structure.info from %s: %s", inpx_path, e)
        return None

    text = ""
    # Try a couple of encodings; utf-8 first, cp1251 common for Russian INPX
    for enc in ("utf-8", "cp1251", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            text = ""
    if not text:
        return None

    # Take the first non-empty, non-comment line that looks like "A;B;C;..."
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith(";"):
            continue
        if ";" not in line:
            continue
        parts = [p.strip() for p in line.split(";")]
        parts = [p for p in parts if p]
        if parts:
            return parts

    return None


def build_global_field_names(inpx_files: list[str], max_fields: int) -> list[str]:
    """
    Build a global list of column names for positions 1..max_fields.

    For each field index i, look at all structure.info definitions across
    all INPX that have a name at that position, and pick the most frequent
    name. If no name is known for that index, fall back to 'FieldN'.
    """
    if max_fields <= 0:
        return []

    counters: list[Counter] = [Counter() for _ in range(max_fields)]

    for path in inpx_files:
        path = path.strip()
        if not path or not os.path.isfile(path):
            continue
        names = read_inpx_field_names_from_structure(path)
        if not names:
            continue

        for idx, raw_name in enumerate(names):
            if idx >= max_fields:
                break
            name = raw_name.strip()
            if not name:
                continue
            # You can normalize case here if you like, e.g. name.upper()
            counters[idx][name] += 1

    columns: list[str] = []
    for i in range(max_fields):
        if counters[i]:
            # Most common name for this position
            name, _ = counters[i].most_common(1)[0]
            columns.append(name)
        else:
            columns.append(f"Field{i+1}")

    return columns


# --- Catalog generation ---

def discover_max_fields(inpx_files: List[str]) -> int:
    import zipfile

    max_fields = 0
    for inpx_path in inpx_files:
        inpx_path = inpx_path.strip()
        if not inpx_path or not os.path.isfile(inpx_path):
            continue
        try:
            with zipfile.ZipFile(inpx_path, "r") as zf:
                for inner_name in zf.namelist():
                    try:
                        with zf.open(inner_name, "r") as f:
                            for raw_line in f:
                                try:
                                    line = raw_line.decode("utf-8", errors="ignore")
                                except Exception:
                                    continue
                                if not line.strip():
                                    continue
                                _, parts = split_record(line)
                                if parts is None:
                                    continue
                                if len(parts) > max_fields:
                                    max_fields = len(parts)
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(
                "Failed to open INPX file %s during discovery: %s",
                inpx_path,
                e,
            )
            continue
    return max_fields


def generate_catalog_parts(inpx_files: List[str]) -> List[str]:
    """
    Generate catalog in multiple parts:
      cache/catalog-part01.csv.zip
      cache/catalog-part02.csv.zip
      ...

    Returns a list of absolute paths to the zip files.
    """
    import zipfile

    os.makedirs(CACHE_DIR, exist_ok=True)

    logger.info("Discovering max number of fields across all INPX...")
    max_fields = discover_max_fields(inpx_files)
    if max_fields <= 0:
        logger.warning("No INPX records found; catalog will be empty.")
        max_fields = 0

    # Build smarter column names using structure.info where available
    field_columns = build_global_field_names(inpx_files, max_fields)
    meta_columns = ["source_inpx", "index_inner_name"]
    header = field_columns + meta_columns

    # Clean old parts
    for name in os.listdir(CACHE_DIR):
        if name.startswith("catalog-part") and name.endswith(".csv.zip"):
            try:
                os.remove(os.path.join(CACHE_DIR, name))
            except OSError:
                pass

    part_paths: List[str] = []

    part_index = 1
    rows_in_part = 0
    csv_file = None
    writer = None
    tmp_csv_path = None

    def start_new_part():
        nonlocal part_index, rows_in_part, csv_file, writer, tmp_csv_path
        # Close/remove leftover temp from previous part
        if csv_file:
            csv_file.close()
            csv_file = None
        if tmp_csv_path:
            try:
                os.remove(tmp_csv_path)
            except OSError:
                pass
            tmp_csv_path = None

        fd, tmp_csv_local = tempfile.mkstemp(
            suffix=f"-part{part_index:02d}.csv"
        )
        os.close(fd)
        csv_file_local = open(
            tmp_csv_local, "w", encoding="utf-8", newline=""
        )
        csv_writer = csv.writer(csv_file_local)
        csv_writer.writerow(header)

        csv_file = csv_file_local
        writer = csv_writer
        tmp_csv_path = tmp_csv_local
        rows_in_part = 0
        logger.info("Started new CSV part %d", part_index)

    def finalize_part():
        nonlocal part_index, rows_in_part, csv_file, writer, tmp_csv_path, part_paths
        if not csv_file or not tmp_csv_path:
            return

        csv_file.close()
        csv_file = None

        part_zip_name = f"catalog-part{part_index:02d}.csv.zip"
        part_zip_path = os.path.join(CACHE_DIR, part_zip_name)
        tmp_zip_path = part_zip_path + ".tmp"

        try:
            with zipfile.ZipFile(
                tmp_zip_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                zf.write(tmp_csv_path, arcname=f"catalog-part{part_index:02d}.csv")
            os.replace(tmp_zip_path, part_zip_path)
            part_paths.append(part_zip_path)
            logger.info(
                "Finalized part %d -> %s (rows: %d)",
                part_index,
                part_zip_path,
                rows_in_part,
            )
        except Exception as e:
            logger.error("Failed to create catalog zip for part %d: %s", part_index, e)
            try:
                os.remove(tmp_zip_path)
            except OSError:
                pass
        finally:
            try:
                os.remove(tmp_csv_path)
            except OSError:
                pass
            tmp_csv_path = None
            rows_in_part = 0
            part_index += 1

    start_new_part()
    total_records = 0

    try:
        for inpx_path in inpx_files:
            inpx_path = inpx_path.strip()
            if not inpx_path or not os.path.isfile(inpx_path):
                continue

            logger.info("Processing INPX: %s", inpx_path)
            try:
                with zipfile.ZipFile(inpx_path, "r") as zf:
                    for inner_name in zf.namelist():
                        try:
                            with zf.open(inner_name, "r") as f:
                                for raw_line in f:
                                    try:
                                        line = raw_line.decode("utf-8", errors="ignore")
                                    except Exception:
                                        continue
                                    if not line.strip():
                                        continue

                                    _, parts = split_record(line)
                                    if parts is None:
                                        continue

                                    row_fields = [p.strip() for p in parts]
                                    if len(row_fields) < max_fields:
                                        row_fields.extend(
                                            [""] * (max_fields - len(row_fields))
                                        )
                                    elif len(row_fields) > max_fields:
                                        row_fields = row_fields[:max_fields]

                                    meta = [inpx_path, inner_name]
                                    writer.writerow(row_fields + meta)
                                    rows_in_part += 1
                                    total_records += 1

                                    if rows_in_part >= ROWS_PER_PART:
                                        finalize_part()
                                        start_new_part()
                        except Exception as e:
                            logger.warning(
                                "Failed to read inner file %s in %s: %s",
                                inner_name,
                                inpx_path,
                                e,
                            )
                            continue
            except Exception as e:
                logger.warning(
                    "Failed to open INPX file %s during catalog generation: %s",
                    inpx_path,
                    e,
                )
                continue

        # Finalize last part if it has any rows
        if rows_in_part > 0 and csv_file:
            finalize_part()
    finally:
        if csv_file:
            csv_file.close()
        if tmp_csv_path:
            try:
                os.remove(tmp_csv_path)
            except OSError:
                pass

    logger.info(
        "Catalog generation finished, total records: %d, parts: %d",
        total_records,
        len(part_paths),
    )
    return part_paths



# --- Main entry point ---

def main() -> None:
    config_path = os.path.join(BASE_DIR, "bot.conf")
    inpx_files = load_inpx_files_from_config(config_path)

    sigs = get_inpx_signatures(inpx_files)
    if not sigs:
        logger.error("No valid INPX files found. Nothing to do.")
        sys.exit(1)

    meta = load_catalog_meta()
    if meta and catalog_cache_is_valid(sigs):
        logger.info("Catalog cache is already valid. Nothing to do.")
        return

    logger.info("Catalog cache is missing or outdated; regenerating...")
    part_paths = generate_catalog_parts(inpx_files)
    if not part_paths:
        logger.error("Failed to generate any catalog parts.")
        sys.exit(1)

    meta = {
        "version": CATALOG_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inpx_files": sigs,
        "parts": [os.path.basename(p) for p in part_paths],
    }
    save_catalog_meta(meta)
    logger.info("Done.")


if __name__ == "__main__":
    main()
