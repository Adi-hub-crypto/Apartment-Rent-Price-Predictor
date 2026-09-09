"""Check a rent submission against data/sample_submission.csv - and fix it.

    python3 validate_submission.py submission.csv
    python3 validate_submission.py "~/Downloads/submission (1).csv" --fix

Checks the things a strict grader rejects: header, row count, id order, genre
values, line endings, BOM, trailing newline.  --fix rewrites the file in the
sample's exact format; values are read and written as text, so no value is
ever altered.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "data", "sample_submission.csv")
COLUMNS = ["id", "rent_price"]


def read_rows(path):
    """-> (header, rows, raw bytes). Tolerates LF or CRLF, strips a BOM."""
    raw = open(path, "rb").read()
    text = raw.decode("utf-8-sig")
    lines = [ln for ln in text.replace("\r\n", "\n").split("\n") if ln.strip()]
    split = [ln.split(",") for ln in lines]
    return split[0], split[1:], raw


def check(path, sample_path=SAMPLE):
    header, rows, raw = read_rows(path)
    problems, notes = [], []

    if header != COLUMNS:
        problems.append(f"header is {header}, expected {COLUMNS}")
    for i, row in enumerate(rows, start=2):
        if len(row) != 2:
            problems.append(f"line {i} has {len(row)} fields, expected 2")
            break
        try:
            value = float(row[1])
        except ValueError:
            problems.append(f"line {i} has rent_price {row[1]!r}, which is not a number")
            break
        if value != value or value in (float("inf"), float("-inf")):
            problems.append(f"line {i} has a non-finite rent_price")
            break
        if value < 0:
            problems.append(f"line {i} has a negative rent_price ({value})")
            break

    if raw[:3] == b"\xef\xbb\xbf":
        problems.append("file starts with a UTF-8 BOM")
    if not raw.endswith(b"\n"):
        problems.append("file does not end with a newline")

    crlf = raw.count(b"\r\n")
    expected_crlf = 0
    if os.path.exists(sample_path):
        s_header, s_rows, s_raw = read_rows(sample_path)
        expected_crlf = s_raw.count(b"\r\n")
        if len(rows) != len(s_rows):
            problems.append(f"{len(rows)} rows, sample has {len(s_rows)}")
        elif [r[0] for r in rows] != [r[0] for r in s_rows]:
            problems.append("ids differ from the sample, or are in a different order")
        if expected_crlf and crlf == 0:
            problems.append("line endings are LF; the sample uses CRLF")
        notes.append(f"sample: {len(s_rows)} rows, "
                     f"{'CRLF' if expected_crlf else 'LF'} line endings")
    else:
        notes.append(f"no sample at {sample_path} - checked structure only")

    notes.append(f"this file: {len(rows)} rows, {'CRLF' if crlf else 'LF'} line endings, "
                 f"{len(raw)} bytes")
    return problems, notes, header, rows


def fix(path, sample_path=SAMPLE):
    """Rewrite in the sample's exact format. Values are copied verbatim."""
    header, rows, _ = read_rows(path)
    before = [tuple(r) for r in rows]

    terminator = "\r\n"
    if os.path.exists(sample_path) and read_rows(sample_path)[2].count(b"\r\n") == 0:
        terminator = "\n"

    body = terminator.join([",".join(header)] + [",".join(r) for r in rows]) + terminator
    with open(path, "wb") as fh:
        fh.write(body.encode("utf-8"))

    after = [tuple(r) for r in read_rows(path)[1]]
    if before != after:                       # never silently alter a prediction
        raise SystemExit("aborted: rewriting changed a value")
    return len(rows), terminator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--fix", action="store_true",
                    help="rewrite the file in the sample's exact format")
    args = ap.parse_args()
    path = os.path.expanduser(args.path)

    problems, notes, _, _ = check(path)
    for note in notes:
        print("  " + note)

    if not problems:
        print(f"\nOK - {os.path.basename(path)} matches the sample format.")
        return

    print(f"\n{len(problems)} problem(s):")
    for problem in problems:
        print("  - " + problem)

    if not args.fix:
        print("\nRe-run with --fix to rewrite it (values are left untouched).")
        sys.exit(1)

    rows, terminator = fix(path)
    print(f"\nRewrote {rows} rows with "
          f"{'CRLF' if terminator == chr(13) + chr(10) else 'LF'} endings; "
          f"every value unchanged.")
    problems, notes, _, _ = check(path)
    for note in notes:
        print("  " + note)
    print("\nOK - matches the sample format." if not problems
          else "\nstill failing: " + "; ".join(problems))


if __name__ == "__main__":
    main()
