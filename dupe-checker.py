#!/usr/bin/env python3
import os
import zipfile

# Adjust this list or read it from bot.conf like in your bot
INPX_FILES = [
    "/path/to/your1.inpx",
    "/path/to/your2.inpx",
]

SEPARATORS = ("\x04", "\t", ";", "|")


def split_record(line: str):
    line = line.rstrip("\r\n")
    for sep in SEPARATORS:
        if sep in line:
            return sep, line.split(sep)
    return None, None


def parse_inpx_record(line: str):
    _, parts = split_record(line)
    if parts is None or len(parts) < 13:
        return None

    author = parts[0].strip()
    title = parts[2].strip()
    filename = parts[5].strip()
    lib_id = parts[7].strip()
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
        "lib_id": lib_id,
        "container_relpath": container_relpath,
        "inner_book_name": inner_book_name,
        "ext": ext,
        "fields": parts,
    }


def check_duplicates():
    seen = {}
    dup_count = 0

    for inpx_path in INPX_FILES:
        if not os.path.isfile(inpx_path):
            print(f"INPX not found: {inpx_path}")
            continue

        print(f"Scanning {inpx_path}...")
        with zipfile.ZipFile(inpx_path, "r") as zf:
            for inner_name in zf.namelist():
                with zf.open(inner_name, "r") as f:
                    for lineno, raw_line in enumerate(f, start=1):
                        try:
                            line = raw_line.decode("utf-8", errors="ignore")
                        except Exception:
                            continue
                        line = line.strip()
                        if not line:
                            continue

                        parsed = parse_inpx_record(line)
                        if not parsed:
                            continue

                        # Key by full fields tuple to detect exact duplicates
                        key = tuple(parsed["fields"])
                        where = (inpx_path, inner_name, lineno)

                        if key in seen:
                            first_where = seen[key]
                            dup_count += 1
                            print("DUPLICATE RECORD:")
                            print(f"  First: {first_where[0]} -> {first_where[1]} (line {first_where[2]})")
                            print(f"  Again: {where[0]} -> {where[1]} (line {where[2]})")
                        else:
                            seen[key] = where

    print(f"Total exact duplicate records: {dup_count}")


if __name__ == "__main__":
    check_duplicates()
