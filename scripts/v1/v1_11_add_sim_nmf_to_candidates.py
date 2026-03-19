#!/usr/bin/env python3
"""
scripts/add_sim_nmf_to_candidates.py

Goal
----
Add NMF-based similarity (sim_nmf) to an existing candidate table.

Why this exists
--------------
- TF-IDF candidates provide a restricted candidate set per book.
- NMF topic vectors provide low-dimensional, interpretable features.
- Cosine similarity on NMF topic vectors provides sim_nmf for Pattern A sliders.

Implementation detail
---------------------
Streaming is implemented using PyArrow Parquet row groups to avoid loading
tens of millions of candidate rows into memory.

Inputs
------
data/processed/tableau_candidates_tfidf_lda.parquet
  Columns used:
    - book_id
    - neighbor_book_id
  Other columns are preserved.

data/processed/book_ids_nmf.npy
data/processed/topics_nmf.npy
  Alignment:
    topics_nmf[i] corresponds to book_ids_nmf[i]

Output
------
data/processed/tableau_candidates_tfidf_lda_nmf.parquet
  Same rows as input plus:
    - sim_nmf (cosine similarity on NMF topic vectors)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", default="data/processed/tableau_candidates_tfidf_lda.parquet")
    p.add_argument("--book-ids", default="data/processed/book_ids_nmf.npy")
    p.add_argument("--topics", default="data/processed/topics_nmf.npy")
    p.add_argument("--out", default="data/processed/tableau_candidates_tfidf_lda_nmf.parquet")
    p.add_argument("--progress-every", type=int, default=10, help="Print progress every N row groups")
    return p.parse_args()


def fmt_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return mat / denom


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    book_ids = np.load(args.book_ids)
    topics = np.load(args.topics).astype(np.float32)
    topics = l2_normalize(topics)

    index: Dict[int, int] = {int(b): i for i, b in enumerate(book_ids)}

    pf = pq.ParquetFile(args.candidates)
    total_rgs = pf.num_row_groups

    writer: pq.ParquetWriter | None = None
    rows_written = 0

    for rg in range(total_rgs):
        t_rg = pf.read_row_group(rg)

        b = t_rg["book_id"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        n = t_rg["neighbor_book_id"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)

        bi = np.fromiter((index.get(int(x), -1) for x in b), dtype=np.int64, count=len(b))
        ni = np.fromiter((index.get(int(x), -1) for x in n), dtype=np.int64, count=len(n))

        valid = (bi >= 0) & (ni >= 0)
        sim = np.zeros(len(b), dtype=np.float32)
        if valid.any():
            sim[valid] = (topics[bi[valid]] * topics[ni[valid]]).sum(axis=1)

        t_out = t_rg.append_column("sim_nmf", pa.array(sim, type=pa.float32()))

        if writer is None:
            writer = pq.ParquetWriter(out_path.as_posix(), t_out.schema, compression="zstd")

        writer.write_table(t_out)
        rows_written += t_out.num_rows

        if args.progress_every and (rg + 1) % args.progress_every == 0:
            elapsed = time.time() - t0
            rate = rows_written / max(elapsed, 1e-9)
            print(
                f"[progress] row_groups={rg+1}/{total_rgs} rows_written={rows_written:,} "
                f"elapsed={fmt_seconds(elapsed)} rate={rate:,.0f} rows/s"
            )

    if writer is not None:
        writer.close()

    total = time.time() - t0
    print(f"[done] rows={rows_written:,} -> {out_path}")
    print(f"[time] total={fmt_seconds(total)}")


if __name__ == "__main__":
    main()