#!/usr/bin/env python3
"""
scripts/8_add_sim_lda_nmf_to_tfidf_pairs.py

Goal
----
Add LDA and NMF cosine similarity to TF-IDF candidate pairs (work_id -> neighbor_work_id).

Why this exists
--------------
- TF-IDF provides a candidate list and lexical similarity (sim_tfidf).
- LDA and NMF provide thematic similarity signals for re-ranking and explanation.

Inputs
------
data/processed_v2/neighbors_work_tfidf_k100.parquet
  Columns:
    - work_id
    - neighbor_work_id
    - sim_tfidf
    - rank_tfidf

data/processed_v2/work_ids_lda.npy
data/processed_v2/topics_lda.npy
  topics_lda shape: (n_works, n_topics)

data/processed_v2/work_ids_nmf.npy
data/processed_v2/topics_nmf.npy
  topics_nmf shape: (n_works, n_topics)

Output
------
data/processed_v2/neighbors_work_tfidf_k100_lda_nmf.parquet
  Adds:
    - sim_lda
    - sim_nmf

Notes
-----
- Cosine similarity is computed as dot product after L2-normalizing vectors.
- Parquet is processed in row groups to keep memory stable.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default="data/processed_v2/neighbors_work_tfidf_k100.parquet")
    p.add_argument("--work-ids-lda", default="data/processed_v2/work_ids_lda.npy")
    p.add_argument("--topics-lda", default="data/processed_v2/topics_lda.npy")
    p.add_argument("--work-ids-nmf", default="data/processed_v2/work_ids_nmf.npy")
    p.add_argument("--topics-nmf", default="data/processed_v2/topics_nmf.npy")
    p.add_argument("--out", default="data/processed_v2/neighbors_work_tfidf_k100_lda_nmf.parquet")
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
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (mat / norms).astype(np.float32)


def build_index(work_ids: np.ndarray) -> dict[int, int]:
    # work_id -> row index in topics matrix
    return {int(w): int(i) for i, w in enumerate(work_ids)}


def cosine_from_rows(mat: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    # Dot product of L2-normalized rows gives cosine similarity.
    a = mat[idx_a]
    b = mat[idx_b]
    return np.einsum("ij,ij->i", a, b).astype(np.float32)


def main() -> None:
    args = parse_args()
    t0 = time.time()

    work_ids_lda = np.load(args.work_ids_lda)
    topics_lda = l2_normalize(np.load(args.topics_lda).astype(np.float32))
    idx_lda = build_index(work_ids_lda)

    work_ids_nmf = np.load(args.work_ids_nmf)
    topics_nmf = l2_normalize(np.load(args.topics_nmf).astype(np.float32))
    idx_nmf = build_index(work_ids_nmf)

    pf = pq.ParquetFile(args.pairs)
    writer: pq.ParquetWriter | None = None
    rows_written = 0

    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg)
        work_id = tbl["work_id"].to_numpy(zero_copy_only=False).astype(np.int64)
        nbr_id = tbl["neighbor_work_id"].to_numpy(zero_copy_only=False).astype(np.int64)

        # Map work_ids to topic row indices.
        # Mapping should be complete if ids were consistent across steps.
        src_i_lda = np.fromiter((idx_lda[int(w)] for w in work_id), dtype=np.int64, count=len(work_id))
        nbr_i_lda = np.fromiter((idx_lda[int(w)] for w in nbr_id), dtype=np.int64, count=len(nbr_id))

        src_i_nmf = np.fromiter((idx_nmf[int(w)] for w in work_id), dtype=np.int64, count=len(work_id))
        nbr_i_nmf = np.fromiter((idx_nmf[int(w)] for w in nbr_id), dtype=np.int64, count=len(nbr_id))

        sim_lda = cosine_from_rows(topics_lda, src_i_lda, nbr_i_lda)
        sim_nmf = cosine_from_rows(topics_nmf, src_i_nmf, nbr_i_nmf)

        out_tbl = tbl.append_column("sim_lda", pa.array(sim_lda, type=pa.float32()))
        out_tbl = out_tbl.append_column("sim_nmf", pa.array(sim_nmf, type=pa.float32()))

        if writer is None:
            writer = pq.ParquetWriter(args.out, out_tbl.schema, compression="zstd")

        writer.write_table(out_tbl)
        rows_written += out_tbl.num_rows

        if (rg + 1) % 10 == 0 or (rg + 1) == pf.num_row_groups:
            elapsed = time.time() - t0
            rate = rows_written / max(elapsed, 1e-9)
            print(f"[progress] row_groups={rg+1}/{pf.num_row_groups} rows_written={rows_written:,} elapsed={fmt_seconds(elapsed)} rate={rate:,.0f} rows/s")

    if writer is not None:
        writer.close()

    total = time.time() - t0
    print(f"[done] rows={rows_written:,} -> {args.out}")
    print(f"[time] total={fmt_seconds(total)}")


if __name__ == "__main__":
    main()