#!/usr/bin/env python3
"""
scripts/4b_inspect_work_parquets.py

Goal
----
Inspect work-level Parquet outputs in a beginner-friendly way:
- show row counts
- show column names
- show description length stats
- print a few sample rows (different slices)

Why this exists
--------------
Parquet is not human-readable. This script provides quick quality checks before
running expensive modeling steps.

Inputs
------
data/processed_v2/work_books_model.parquet
data/processed_v2/work_books_text.parquet

Output
------
Printed summary to the terminal.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/processed_v2/work_books_model.parquet")
    p.add_argument("--text", default="data/processed_v2/work_books_text.parquet")
    p.add_argument("--n", type=int, default=5, help="Rows to print per sample slice")
    p.add_argument("--skip", type=int, default=50_000, help="Offset for the second slice")
    p.add_argument("--cols-model", default="work_id,rep_book_id,title,author_name,ratings_count,average_rating,genres_top,badges_top,has_desc80")
    p.add_argument("--cols-text", default="work_id,title,genres_top,description_len,description_clean")
    p.add_argument("--max-clean-chars", type=int, default=200, help="Preview length for description_clean")
    return p.parse_args()


def count_rows(path: str) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def list_cols(path: str) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def parse_cols(s: str) -> list[str]:
    cols = [c.strip() for c in s.split(",") if c.strip()]
    return cols


def print_len_stats(name: str, series: pd.Series) -> None:
    s = series.fillna("").astype(str).str.len()
    q = np.quantile(s, [0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    print(f"\n{name} length stats (chars)")
    print(f"  min={int(q[0])} p25={int(q[1])} median={int(q[2])} p75={int(q[3])} p95={int(q[4])} p99={int(q[5])} max={int(q[6])}")


def main() -> None:
    args = parse_args()

    model_rows = count_rows(args.model)
    text_rows = count_rows(args.text)

    print("=== PARQUET SUMMARY ===")
    print(f"model: {args.model} rows={model_rows:,}")
    print(f"text : {args.text} rows={text_rows:,}")

    print("\n=== COLUMNS ===")
    print(f"model cols ({len(list_cols(args.model))}): {list_cols(args.model)}")
    print(f"text  cols ({len(list_cols(args.text))}): {list_cols(args.text)}")

    # Model preview
    model_cols = parse_cols(args.cols_model)
    dfm_a = pd.read_parquet(args.model, columns=[c for c in model_cols if c in list_cols(args.model)]).head(args.n)
    dfm_b = pd.read_parquet(args.model, columns=[c for c in model_cols if c in list_cols(args.model)]).iloc[args.skip : args.skip + args.n]

    print("\n=== MODEL SAMPLE (first slice) ===")
    print(dfm_a.to_string(index=False))

    print(f"\n=== MODEL SAMPLE (skip={args.skip}) ===")
    print(dfm_b.to_string(index=False))

    # Text preview
    text_cols = parse_cols(args.cols_text)
    dft_a = pd.read_parquet(args.text, columns=[c for c in text_cols if c in list_cols(args.text)]).head(args.n)
    dft_b = pd.read_parquet(args.text, columns=[c for c in text_cols if c in list_cols(args.text)]).iloc[args.skip : args.skip + args.n]

    # Trim long clean text for display
    if "description_clean" in dft_a.columns:
        dft_a["description_clean"] = dft_a["description_clean"].astype(str).str.slice(0, args.max_clean_chars)
    if "description_clean" in dft_b.columns:
        dft_b["description_clean"] = dft_b["description_clean"].astype(str).str.slice(0, args.max_clean_chars)

    print("\n=== TEXT SAMPLE (first slice) ===")
    print(dft_a.to_string(index=False))

    print(f"\n=== TEXT SAMPLE (skip={args.skip}) ===")
    print(dft_b.to_string(index=False))

    # Length stats
    if "description" in list_cols(args.model):
        dfm_desc = pd.read_parquet(args.model, columns=["description"])
        print_len_stats("model.description (raw-ish proxy)", dfm_desc["description"])

    if "description_clean" in list_cols(args.text):
        dft_desc = pd.read_parquet(args.text, columns=["description_clean"])
        print_len_stats("text.description_clean", dft_desc["description_clean"])


if __name__ == "__main__":
    main()