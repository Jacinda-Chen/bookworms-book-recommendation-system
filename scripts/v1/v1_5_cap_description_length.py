# scripts/cap_description_length.py
"""
Cap very long descriptions before TF-IDF / topic modeling.

Why cap?

2000 23006
4000 625
8000 54
16000 3
--------
A small number of descriptions can be extremely long (you saw up to ~43k chars).
Those outliers can:
- slow down vectorization (TF-IDF / CountVectorizer),
- introduce lots of rare tokens (noisy vocabulary),
- and sometimes disproportionately influence topic models.

Capping is a cheap, low-risk “stability” step:
- We only cap at 4000 chars (affects ~625 rows in your dataset).
- We keep the original description_clean unchanged.
- We create a new column description_model (the capped version) for modeling.

Input:
------
data/processed/books_text.parquet

Output:
-------
data/processed/books_text_capped.parquet
Adds:
- description_model (capped text for modeling)
- description_model_len
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """CLI options so you can change the cap without editing code."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="data/processed/books_text.parquet")
    ap.add_argument("--outfile", default="data/processed/books_text_capped.parquet")
    ap.add_argument("--max-chars", type=int, default=4000, help="Maximum characters to keep in description_model")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Load the dataset that already has description_clean.
    df = pd.read_parquet(args.infile)

    # Ensure we always have strings (no NaNs) before slicing.
    # We cap based on characters because it’s fast and good enough as a guardrail.
    s = df["description_clean"].fillna("").astype(str)

    # Create the modeling text column (capped).
    df["description_model"] = s.str.slice(0, args.max_chars)

    # Track the capped length (handy for QA / debugging).
    df["description_model_len"] = df["description_model"].str.len()

    # Count how many rows were actually truncated (useful sanity check).
    # Uses the original description_len from your pipeline.
    truncated_rows = int((df["description_len"] > args.max_chars).sum())

    # Ensure output folder exists.
    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)

    # Write a new parquet so the original file stays untouched.
    df.to_parquet(args.outfile, index=False)

    print(f"[done] wrote {len(df):,} rows -> {args.outfile}")
    print(f"[info] truncated_rows={truncated_rows:,} max_chars={args.max_chars}")


if __name__ == "__main__":
    main()