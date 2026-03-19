#!/usr/bin/env python3
"""
scripts/7_train_nmf_work.py

Goal
----
Train an NMF topic model on work-level descriptions and compute topic vectors.

Why this exists
--------------
- NMF produces nonnegative topic weights per work (50-dim by default).
- Cosine similarity between NMF vectors becomes a thematic similarity signal.
- Topic keywords are saved as Parquet for visualization/tooltips downstream.

Input
-----
data/processed_v2/work_books_text.parquet
  Required columns:
    - work_id
    - description_model

Outputs (v2)
------------
data/processed_v2/work_ids_nmf.npy
data/processed_v2/topics_nmf.npy
data/processed_v2/topic_keywords_nmf.parquet

Notes
-----
- TF-IDF vectorization is used because NMF factorizes TF-IDF well.
- stop_words='english' is applied in the vectorizer to reduce noise.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--infile", default="data/processed_v2/work_books_text.parquet")
    p.add_argument("--n-topics", type=int, default=50)
    p.add_argument("--max-features", type=int, default=80_000)
    p.add_argument("--min-df", type=int, default=10)
    p.add_argument("--max-df", type=float, default=0.6)
    p.add_argument("--ngram-max", type=int, default=2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--outdir", default="data/processed_v2")
    p.add_argument("--top-words", type=int, default=15)
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
    return f"{m}m {s}s" if m else f"{s}s"


def main() -> None:
    args = parse_args()
    t0 = time.time()

    df = pd.read_parquet(args.infile, columns=["work_id", "description_model"]).dropna(subset=["description_model"])
    df["description_model"] = df["description_model"].astype(str)

    work_ids = df["work_id"].to_numpy(dtype=np.int64)
    texts = df["description_model"].tolist()
    n = len(work_ids)

    # TF-IDF converts text into a sparse matrix suitable for matrix factorization.
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

    # NMF learns topic weights W (docs x topics) and topic-word weights H (topics x terms).
    t_nmf0 = time.time()
    nmf = NMF(
        n_components=args.n_topics,
        init="nndsvda",
        random_state=args.random_state,
        max_iter=300,
    )
    W = nmf.fit_transform(X).astype(np.float32)
    t_nmf = time.time() - t_nmf0

    out_ids = f"{args.outdir}/work_ids_nmf.npy"
    out_topics = f"{args.outdir}/topics_nmf.npy"
    out_kw = f"{args.outdir}/topic_keywords_nmf.parquet"

    np.save(out_ids, work_ids)
    np.save(out_topics, W)

    # Topic keywords are stored as Parquet for easy DuckDB/Tableau joins.
    vocab = np.array(vectorizer.get_feature_names_out())
    H = nmf.components_
    rows = []
    for k in range(args.n_topics):
        top_idx = np.argsort(H[k])[::-1][: args.top_words]
        rows.append({"topic": int(k), "top_words": "|".join(vocab[top_idx].tolist())})
    pd.DataFrame(rows).to_parquet(out_kw, index=False)

    total = time.time() - t0
    print(f"[done] docs={n:,} X_shape={X.shape} topics_shape={W.shape} topics={args.n_topics}")
    print(f"[time] vectorize={fmt_seconds(t_vec)} nmf_train={fmt_seconds(t_nmf)} total={fmt_seconds(total)}")
    print(f"[out] {out_ids}")
    print(f"[out] {out_topics}")
    print(f"[out] {out_kw}")


if __name__ == "__main__":
    main()