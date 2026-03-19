#!/usr/bin/env python3
"""
scripts/4_preprocess_work_descriptions.py

Goal
----
Clean description_raw at work-level and create modeling-ready text fields.

Why this exists
--------------
- description_raw often contains HTML and entities that hurt NLP quality.
- TF-IDF / LDA / NMF expect plain text, not markup.
- Extremely long descriptions slow vectorization and can dominate similarity,
  so a capped modeling field is created.

4b_inspect_work_parquets
--------------
Row counts match: work_books_model and work_books_text 
both have 498,776 rows → Step 4 didn’t unexpectedly drop 
anything beyond the model filter (min len already 
enforced in Step 3, and cleaning preserved length ≥80).

Input
-----
data/processed_v2/work_books_model.parquet
  Required columns:
    - work_id
    - description_raw

Output
------
data/processed_v2/work_books_text.parquet
  Adds:
    - description_clean  (plain text)
    - description_len    (quality/QA metric)
    - description_model  (capped plain text for modeling)

Filtering
---------
- Rows with cleaned description length < --min-len are dropped (default 80).
  This keeps only works with enough text to support NLP similarity and topic modeling.
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import pandas as pd

# HTML tags appear in Goodreads descriptions (e.g., <br />, <p>).
# Removing tags avoids modeling markup instead of content.
_TAG_RE = re.compile(r"<[^>]+>")

# Whitespace normalization keeps downstream tokenization consistent.
_SPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--infile", default="data/processed_v2/work_books_model.parquet")
    p.add_argument("--out", default="data/processed_v2/work_books_text.parquet")
    p.add_argument("--min-len", type=int, default=80)
    p.add_argument("--cap-chars", type=int, default=4000)
    return p.parse_args()


def clean_description(raw: str) -> str:
    # Entities like &amp; and &quot; are common; decoding improves readability and tokens.
    s = html.unescape(raw or "")
    # Tag stripping removes markup without trying to preserve HTML structure.
    s = _TAG_RE.sub(" ", s)
    # Whitespace normalization prevents accidental token boundaries and empty runs.
    s = _SPACE_RE.sub(" ", s).strip()
    return s


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.infile)

    if "description_raw" not in df.columns:
        raise ValueError("Input parquet must contain description_raw")

    # Missing descriptions are converted to empty strings so cleaning logic stays simple.
    df["description_raw"] = df["description_raw"].fillna("").astype(str)

    df["description_clean"] = df["description_raw"].map(clean_description)
    df["description_len"] = df["description_clean"].str.len()

    # Minimum length ensures the remaining texts contain enough signal for similarity/topic models.
    df = df[df["description_len"] >= args.min_len].copy()

    # Capping keeps compute predictable and reduces the impact of extreme outliers.
    df["description_model"] = df["description_clean"].str.slice(0, args.cap_chars)

    df.to_parquet(out_path, index=False)
    print(f"[done] wrote -> {out_path} rows={len(df):,}")
    print(f"[info] min_len={args.min_len} cap_chars={args.cap_chars}")


if __name__ == "__main__":
    main()