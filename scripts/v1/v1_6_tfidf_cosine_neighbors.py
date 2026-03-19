#!/usr/bin/env python3
"""
scripts/tfidf_cosine_neighbors.py

Goal
----
Compute cosine similarity between books using TF-IDF text vectors, and save TOP-K
similar neighbors per book.

Timing 
[done] books=735,602 tfidf_shape=(735602, 80000) neighbors_rows=36,676,154
[time] tfidf=2m 13s neighbors=2h 17m 30s total=2h 20m 36s

self_pairs 0

books_in_table 735602
min_neighbors 49
median_neighbors 50
p01 49
p99 50

------
This version prints:
- total elapsed time
- TF-IDF build time
- neighbor search time
- throughput (books/sec)

Input
-----
data/processed/books_text_capped.parquet
  Columns used:
    - book_id
    - description_model

Outputs
-------
data/processed/neighbors_tfidf.parquet
  Columns:
    - book_id
    - neighbor_book_id
    - cosine_sim

data/processed/tfidf_vectorizer.joblib
  Fitted TF-IDF vectorizer for reuse

data/processed/tfidf_matrix.npz
  Sparse TF-IDF matrix (row order aligns with book_ids_tfidf.npy)

data/processed/book_ids_tfidf.npy
  book_id for each TF-IDF row index

Notes
-----
All-pairs similarity is infeasible at this scale. TOP-K neighbors per book are computed instead.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def parse_args() -> argparse.Namespace:
    """
    CLI arguments control speed/quality trade-offs.
    Smaller settings are recommended for quick validation runs.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--infile", default="data/processed/books_text_capped.parquet")
    p.add_argument("--outdir", default="data/processed")
    p.add_argument("--top-k", type=int, default=20, help="Neighbors per book")
    p.add_argument("--max-features", type=int, default=80_000, help="Vocabulary size cap (larger is slower)")
    p.add_argument("--min-df", type=int, default=10, help="Ignore tokens appearing in fewer than min_df documents")
    p.add_argument("--max-df", type=float, default=0.6, help="Ignore tokens appearing in more than max_df fraction of documents")
    p.add_argument("--ngram-max", type=int, default=2, help="Use 1..ngram-max grams (2 includes bigrams)")
    p.add_argument("--batch", type=int, default=5000, help="Query batch size (memory control)")
    p.add_argument("--progress-every", type=int, default=25, help="Print progress every N batches (0 disables)")
    return p.parse_args()


def fmt_seconds(seconds: float) -> str:
    """Pretty-print seconds as h/m/s."""
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
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # 1) Load minimal columns to reduce memory usage.
    df = pd.read_parquet(args.infile, columns=["book_id", "description_model"])
    df = df.dropna(subset=["description_model"])
    df["description_model"] = df["description_model"].astype(str)

    # Stable alignment: book_ids[i] corresponds to row i in the TF-IDF matrix.
    book_ids = df["book_id"].to_numpy(dtype=np.int64)
    texts = df["description_model"].tolist()
    n_books = len(book_ids)

    # 2) Build TF-IDF vectors.
    # Stopwords are removed at vectorization time to preserve the stored cleaned text.
    t_tfidf0 = time.time()
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=False,  # preprocessing already lowercased text
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        dtype=np.float32,
    )
    X = vectorizer.fit_transform(texts)
    t_tfidf = time.time() - t_tfidf0

    # 3) Nearest neighbors using cosine distance on a sparse TF-IDF matrix.
    # cosine distance = 1 - cosine similarity
    t_nn0 = time.time()
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(X)

    # Top-k + 1 is requested because the closest neighbor of each item is itself (distance=0).
    k = args.top_k + 1
    rows = []

    n_batches = (n_books + args.batch - 1) // args.batch
    for b in range(n_batches):
        start = b * args.batch
        end = min(start + args.batch, n_books)

        distances, indices = nn.kneighbors(X[start:end], n_neighbors=k)
        sims = 1.0 - distances  # cosine similarity

        # Output rows: (book_id, neighbor_book_id, cosine_sim)
        for i_local, (idxs, ss) in enumerate(zip(indices, sims)):
            src_row = start + i_local
            src_id = int(book_ids[src_row])

            # Self neighbor is expected at position 0; skip it.
            # Extra guard: skip any neighbor whose ID equals the source ID.
            for j in range(1, len(idxs)):
                nb_row = int(idxs[j])
                nb_id = int(book_ids[nb_row])
                if nb_id == src_id:
                    continue
                rows.append((src_id, nb_id, float(ss[j])))

        if args.progress_every and (b + 1) % args.progress_every == 0:
            done = end
            elapsed = time.time() - t_nn0
            rate = done / max(elapsed, 1e-9)
            print(
                f"[progress] batches={b+1}/{n_batches} "
                f"books_done={done:,}/{n_books:,} "
                f"nn_time={fmt_seconds(elapsed)} rate={rate:,.0f} books/s"
            )

    t_nn = time.time() - t_nn0

    # 4) Save neighbors and reusable artifacts.
    neighbors = pd.DataFrame(rows, columns=["book_id", "neighbor_book_id", "cosine_sim"])
    out_neighbors = outdir / "neighbors_tfidf.parquet"
    neighbors.to_parquet(out_neighbors, index=False)

    joblib.dump(vectorizer, outdir / "tfidf_vectorizer.joblib")
    sparse.save_npz(outdir / "tfidf_matrix.npz", X)
    np.save(outdir / "book_ids_tfidf.npy", book_ids)

    total = time.time() - t0
    print(f"[done] books={n_books:,} tfidf_shape={X.shape} neighbors_rows={len(neighbors):,}")
    print(f"[time] tfidf={fmt_seconds(t_tfidf)} neighbors={fmt_seconds(t_nn)} total={fmt_seconds(total)}")
    print(f"[out] {out_neighbors}")


if __name__ == "__main__":
    main()