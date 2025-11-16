#!/usr/bin/env python3
import csv
import json
import os
import sys
import time
import zipfile
import tempfile
import logging
import configparser
from typing import Any, Dict, List, Tuple

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("catalog_generator")

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
CATALOG_ZIP_PATH = os.path.join(CACHE_DIR, "catalog.csv.zip")
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
    os.makedirs(CACHE_DIR, exist_ok=True)
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

def catalog_cache_is_valid(
    current_sigs: List[Dict[str, Any]],
) -> bool:
    if not os.path.isfile(CATALOG_ZIP_PATH):
        return False

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

    return True

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

def generate_catalog_zip(inpx_files: List[str]) -> str | None:
    import zipfile

    os.makedirs(CACHE_DIR, exist_ok=True)

    logger.info("Discovering max number of fields across all INPX...")
    max_fields = discover_max_fields(inpx_files)
    if max_fields <= 0:
        logger.warning("No INPX records found; catalog will be empty.")
        max_fields = 0

    field_columns = [f"Field{i}" for i in range(1, max_fields + 1)]
    meta_columns = [
        "source_inpx",
        "index_inner_name",
    ]
    header = field_columns + meta_columns

    fd, tmp_csv_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)

    total_records = 0

    try:
        with open(tmp_csv_path, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)

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

                                        meta = [
                                            inpx_path,
                                            inner_name,
                                        ]
                                        writer.writerow(row_fields + meta)
                                        total_records += 1
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

        logger.info("CSV generation finished, total records: %d", total_records)

        tmp_zip_path = CATALOG_ZIP_PATH + ".tmp"
        try:
            with zipfile.ZipFile(tmp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(tmp_csv_path, arcname="catalog.csv")
            os.replace(tmp_zip_path, CATALOG_ZIP_PATH)
        except Exception as e:
            logger.error("Failed to create catalog zip: %s", e)
            try:
                os.remove(tmp_zip_path)
            except OSError:
                pass
            return None
    finally:
        try:
            os.remove(tmp_csv_path)
        except OSError:
            pass

    logger.info("Catalog zip created at %s", CATALOG_ZIP_PATH)
    return CATALOG_ZIP_PATH

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
    zip_path = generate_catalog_zip(inpx_files)
    if not zip_path:
        logger.error("Failed to generate catalog.")
        sys.exit(1)

    meta = {
        "version": CATALOG_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inpx_files": sigs,
    }
    save_catalog_meta(meta)
    logger.info("Done.")

if __name__ == "__main__":
    main()
