#!/usr/bin/env python3
"""
scripts/5_tfidf_cosine_neighbors_work.py

Goal
----
Compute TF-IDF cosine nearest neighbors at work-level.

Why this exists
--------------
- Candidate generation reduces the search space for later re-ranking signals (LDA/NMF/genre overlap).
- Work-level candidates avoid duplicate editions.

Input
-----
data/processed_v2/work_books_text.parquet
  Required columns:
    - work_id
    - description_model

Output
------
data/processed_v2/neighbors_work_tfidf_k100.parquet
  Columns:
    - work_id
    - neighbor_work_id
    - sim_tfidf
    - rank_tfidf

Notes
-----
- K=100 provides enough candidates after later filtering (e.g., series blocking).
- Cosine similarity is computed in sparse space.
- If a work has too few candidates, fewer than K may appear.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--infile", default="data/processed_v2/work_books_text.parquet")
    p.add_argument("--outfile", default="data/processed_v2/neighbors_work_tfidf_k100.parquet")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--max-features", type=int, default=80_000)
    p.add_argument("--min-df", type=int, default=10)
    p.add_argument("--max-df", type=float, default=0.6)
    p.add_argument("--ngram-max", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2_000, help="Query batch size for neighbor search")
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


def main() -> None:
    args = parse_args()
    t0 = time.time()

    df = pd.read_parquet(args.infile, columns=["work_id", "description_model"]).dropna(subset=["description_model"])
    df["description_model"] = df["description_model"].astype(str)
    work_ids = df["work_id"].to_numpy(dtype=np.int64)
    texts = df["description_model"].tolist()
    n = len(work_ids)

    # TF-IDF vectorization provides a sparse high-dimensional representation.
    t_vec0 = time.time()
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=False,
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        dtype=np.float32,
    )
    X = vectorizer.fit_transform(texts)
    t_vec = time.time() - t_vec0

    # NearestNeighbors with cosine uses 1 - cosine_similarity as distance.
    t_nn0 = time.time()
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=args.k + 1)
    nn.fit(X)

    out_path = args.outfile
    writer: pq.ParquetWriter | None = None
    rows_written = 0

    # Query in batches to keep peak memory stable.
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        distances, indices = nn.kneighbors(X[start:end], return_distance=True)

        src_ids = work_ids[start:end]

        # Drop self-neighbor (distance 0 at index 0), then keep top K.
        indices = indices[:, 1 : args.k + 1]
        distances = distances[:, 1 : args.k + 1]

        # Convert cosine distance -> cosine similarity.
        sims = (1.0 - distances).astype(np.float32)

        # Flatten into columns
        src_col = np.repeat(src_ids, args.k).astype(np.int64)
        nbr_col = work_ids[indices.reshape(-1)].astype(np.int64)
        sim_col = sims.reshape(-1)
        rank_col = (np.tile(np.arange(1, args.k + 1, dtype=np.int32), end - start)).astype(np.int32)

        # Remove any accidental self-pairs (defensive)
        mask = src_col != nbr_col
        src_col = src_col[mask]
        nbr_col = nbr_col[mask]
        sim_col = sim_col[mask]
        rank_col = rank_col[mask]

        table = pa.table(
            {
                "work_id": pa.array(src_col, type=pa.int64()),
                "neighbor_work_id": pa.array(nbr_col, type=pa.int64()),
                "sim_tfidf": pa.array(sim_col, type=pa.float32()),
                "rank_tfidf": pa.array(rank_col, type=pa.int32()),
            }
        )

        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")

        writer.write_table(table)
        rows_written += table.num_rows

        if (start // args.batch_size) % 10 == 0 and start > 0:
            elapsed = time.time() - t0
            rate = rows_written / max(elapsed, 1e-9)
            print(
                f"[progress] batch_start={start:,} rows_written={rows_written:,} "
                f"elapsed={fmt_seconds(elapsed)} rate={rate:,.0f} rows/s"
            )

    if writer is not None:
        writer.close()

    t_nn = time.time() - t_nn0
    total = time.time() - t0

    print(f"[done] works={n:,} tfidf_shape={X.shape} neighbors_rows={rows_written:,}")
    print(f"[time] tfidf={fmt_seconds(t_vec)} neighbors={fmt_seconds(t_nn)} total={fmt_seconds(total)}")
    print(f"[out] {out_path}")


if __name__ == "__main__":
    main()