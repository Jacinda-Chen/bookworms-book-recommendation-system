#!/usr/bin/env python3
"""
scripts/filter_count_books_jsonl.py

Goal
----
Implement the same filtering approach agreed by your classmates:
- Keep a book only if EVERY required field is populated ("EMPTY" rules).
- Do NOT require ASIN.

EMPTY definition
----------------
A value is EMPTY if:
- key does not exist, OR
- value is null, OR
- value is "" (empty string after stripping)

Filter rules (must all pass)
----------------------------
author_id               NOT EMPTY
author_name             NOT EMPTY
average_rating           NOT EMPTY; if 0 then ratings_count must be > 1
description             NOT EMPTY
id                      NOT EMPTY
isbn                    NOT EMPTY
isbn13                  NOT EMPTY
language                == "eng"
num_pages               NOT EMPTY and > 0
original_publication_date NOT EMPTY
publication_date        NOT EMPTY
publisher               NOT EMPTY
rating_dist             NOT EMPTY
ratings_count           NOT EMPTY and > 1
shelves                 NOT EMPTY and non-empty array
title                   NOT EMPTY

Notes
-----
- This script is for verifying your kept count matches (e.g., 744,283).
- It does NOT write Parquet; it just counts + explains rejections.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any, Dict, Iterator, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--books", required=True, help="Path to JSONL books.json")
    p.add_argument("--progress-every", type=int, default=0, help="Print progress every N seen rows (0=off).")
    p.add_argument("--max-seen", type=int, default=0, help="Debug: stop after N seen rows (0=all).")
    return p.parse_args()


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def safe_int(x: Any) -> Optional[int]:
    try:
        return None if is_empty(x) else int(x)
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    try:
        return None if is_empty(x) else float(x)
    except Exception:
        return None


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj


def first_author_from_authors_array(authors_val: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Some dumps include both:
      - top-level author_id/author_name
      - authors: [{author_id,name}, ...]
    We accept either, but your rules require author_id + author_name to be present somehow.
    """
    if not isinstance(authors_val, list) or not authors_val:
        return (None, None)
    for a in authors_val:
        if not isinstance(a, dict):
            continue
        aid = a.get("author_id") or a.get("id")
        aname = a.get("name")
        if not is_empty(aid) and not is_empty(aname):
            return (str(aid), str(aname))
    return (None, None)


def passes_all_required_fields(rec: Dict[str, Any], rejects: Counter) -> bool:
    def reject(reason: str) -> bool:
        rejects[reason] += 1
        return False

    # Required scalar-ish fields
    if is_empty(rec.get("id")):
        return reject("missing_id")
    if is_empty(rec.get("title")):
        return reject("missing_title")
    if is_empty(rec.get("description")):
        return reject("missing_description")
    if is_empty(rec.get("isbn")):
        return reject("missing_isbn")
    if is_empty(rec.get("isbn13")):
        return reject("missing_isbn13")
    if rec.get("language") != "eng":
        return reject("language_not_eng")
    if is_empty(rec.get("publication_date")):
        return reject("missing_publication_date")
    if is_empty(rec.get("original_publication_date")):
        return reject("missing_original_publication_date")
    if is_empty(rec.get("publisher")):
        return reject("missing_publisher")
    if is_empty(rec.get("rating_dist")):
        return reject("missing_rating_dist")

    # Required numeric constraints
    ratings_count = safe_int(rec.get("ratings_count"))
    if ratings_count is None:
        return reject("missing_ratings_count")
    if ratings_count <= 1:
        return reject("ratings_count_le_1")

    avg_rating = safe_float(rec.get("average_rating"))
    if avg_rating is None:
        return reject("missing_average_rating")
    if avg_rating == 0.0 and ratings_count <= 1:
        return reject("avg_rating_0_and_low_ratings")

    num_pages = safe_int(rec.get("num_pages"))
    if num_pages is None:
        return reject("missing_num_pages")
    if num_pages <= 0:
        return reject("num_pages_le_0")

    # Shelves must be non-empty array
    shelves_val = rec.get("shelves")
    if not isinstance(shelves_val, list):
        return reject("shelves_not_list")
    if len(shelves_val) == 0:
        return reject("shelves_empty")

    # Author fields must exist (top-level OR first author in authors[])
    author_id = rec.get("author_id")
    author_name = rec.get("author_name")
    if not is_empty(author_id) and not is_empty(author_name):
        return True

    a_id, a_name = first_author_from_authors_array(rec.get("authors"))
    if not is_empty(a_id) and not is_empty(a_name):
        return True

    return reject("missing_author_id_or_name")


def main() -> None:
    args = parse_args()

    seen = 0
    kept = 0
    rejects: Counter = Counter()

    for rec in iter_jsonl(args.books):
        seen += 1
        if passes_all_required_fields(rec, rejects):
            kept += 1

        if args.progress_every and seen % args.progress_every == 0:
            print(f"[progress] seen={seen:,} kept={kept:,} keep_rate={kept/seen:.2%}")

        if args.max_seen and seen >= args.max_seen:
            break

    print(f"\n[done] seen={seen:,} kept={kept:,} keep_rate={kept/seen:.2%}\n")
    print("[top reject reasons]")
    for reason, count in rejects.most_common(20):
        print(f"  {reason}: {count:,}")


if __name__ == "__main__":
    main()