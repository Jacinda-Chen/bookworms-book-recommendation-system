#!/usr/bin/env python3
"""
scripts/extract_candidate_ids.py

Goal
----
Extract the unique set of book IDs needed for embeddings from a candidate table:
- book_id
- neighbor_book_id

Why this exists
--------------
Embeddings are only required for IDs that appear in candidate pairs.
Computing the unique ID set prevents embedding unrelated books.

Input
-----
data/processed/tableau_candidates_tfidf_lda_nmf.parquet

Output
------
data/processed/embed_candidate_ids.npy
data/processed/embed_candidate_ids.txt  (optional, for quick inspection)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", default="data/processed/tableau_candidates_tfidf_lda_nmf.parquet")
    p.add_argument("--out", default="data/processed/embed_candidate_ids.npy")
    p.add_argument("--also-txt", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(args.candidates)

    ids = set()
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["book_id", "neighbor_book_id"])
        ids.update(map(int, t["book_id"].to_pylist()))
        ids.update(map(int, t["neighbor_book_id"].to_pylist()))
        if (rg + 1) % 5 == 0:
            print(f"[progress] row_groups={rg+1}/{pf.num_row_groups} unique_ids={len(ids):,}")

    arr = np.array(sorted(ids), dtype=np.int64)
    np.save(out_path, arr)

    if args.also_txt:
        txt_path = out_path.with_suffix(".txt")
        txt_path.write_text("\n".join(map(str, arr.tolist())), encoding="utf-8")

    print(f"[done] unique_ids={len(arr):,} -> {out_path}")


if __name__ == "__main__":
    main()