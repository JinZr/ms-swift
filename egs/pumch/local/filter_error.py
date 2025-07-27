#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract entries whose 'response' field truly contains articulation-error annotations
and write them to a new JSON file.

The script treats the following as *no error* and skips them:
  - Empty string / only whitespace
  - "没有构音错误"  (exact match)

Everything else is kept.
"""
import argparse
import json
import os
from typing import Dict, List


def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_actual_error(resp: str) -> bool:
    """Return True if `resp` really lists some error."""
    if resp is None:
        return False
    text = resp.strip()
    return text and text != "没有构音错误"


def extract_errors(records: List[Dict]) -> List[Dict]:
    return [entry for entry in records if is_actual_error(entry.get("response", ""))]


def dump_json(data: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Filter entries with real articulation errors from a JSON file."
    )
    parser.add_argument("input", help="Input JSON file.")
    parser.add_argument(
        "output",
        nargs="?",
        help="Output JSON file (default: <input>_errors.json)."
    )
    args = parser.parse_args()

    in_path = args.input
    out_path = args.output or f"{os.path.splitext(in_path)[0]}_errors.json"

    records = load_json(in_path)
    error_records = extract_errors(records)
    dump_json(error_records, out_path)

    kept = len(error_records)
    total = len(records)
    print(f"✔  Done. Kept {kept} / {total} entries with actual errors → {out_path}")


if __name__ == "__main__":
    main()