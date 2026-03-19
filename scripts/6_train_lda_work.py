#!/usr/bin/env python3
"""
scripts/6_train_lda_work.py

Goal
----
Train an LDA topic model on work-level descriptions and compute topic vectors.

Why this exists
--------------
- LDA produces an interpretable topic distribution per work (50-dim by default).
- Cosine similarity between topic vectors becomes a thematic similarity signal.
- Topic keywords are saved for visualization/tooltips downstream.

Input
-----
data/processed_v2/work_books_text.parquet
  Required columns:
    - work_id
    - description_model

Outputs (v2)
------------
data/processed_v2/work_ids_lda.npy
data/processed_v2/topics_lda.npy
data/processed_v2/topic_keywords_lda.parquet

Notes
-----
- LDA is trained on a sample (default 100,000 works) for speed and topic quality.
- The fitted vectorizer+model are used to transform all works into topic vectors.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--infile", default="data/processed_v2/work_books_text.parquet")
    p.add_argument("--n-topics", type=int, default=50)
    p.add_argument("--fit-docs", type=int, default=100_000)
    p.add_argument("--max-features", type=int, default=50_000)
    p.add_argument("--min-df", type=int, default=10)
    p.add_argument("--max-df", type=float, default=0.6)
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
    return f"{s}s"


def main() -> None:
    args = parse_args()
    t0 = time.time()

    df = pd.read_parquet(args.infile, columns=["work_id", "description_model"]).dropna(subset=["description_model"])
    df["description_model"] = df["description_model"].astype(str)

    work_ids = df["work_id"].to_numpy(dtype=np.int64)
    texts = df["description_model"].tolist()
    n = len(work_ids)

    # Fit on a deterministic head slice to keep runs reproducible.
    fit_n = min(args.fit_docs, n)
    fit_texts = texts[:fit_n]

    t_vec0 = time.time()
    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
    )
    X_fit = vectorizer.fit_transform(fit_texts)
    t_vec_fit = time.time() - t_vec0

    t_lda0 = time.time()
    lda = LatentDirichletAllocation(
        n_components=args.n_topics,
        random_state=args.random_state,
        learning_method="batch",
        max_iter=10,
        evaluate_every=-1,
        n_jobs=-1,
    )
    lda.fit(X_fit)
    t_lda_fit = time.time() - t_lda0

    # Transform all works into topic distributions.
    t_tr0 = time.time()
    X_all = vectorizer.transform(texts)
    topics = lda.transform(X_all).astype(np.float32)
    t_tr = time.time() - t_tr0

    out_ids = f"{args.outdir}/work_ids_lda.npy"
    out_topics = f"{args.outdir}/topics_lda.npy"
    out_kw = f"{args.outdir}/topic_keywords_lda.parquet"

    np.save(out_ids, work_ids)
    np.save(out_topics, topics)

    # Topic keywords saved as Parquet for easy DuckDB/Tableau joins.
    vocab = np.array(vectorizer.get_feature_names_out())
    comps = lda.components_
    rows = []
    for k in range(args.n_topics):
        top_idx = np.argsort(comps[k])[::-1][: args.top_words]
        rows.append({"topic": int(k), "top_words": "|".join(vocab[top_idx].tolist())})
    pd.DataFrame(rows).to_parquet(out_kw, index=False)

    total = time.time() - t0
    print(f"[done] fit_docs={fit_n:,} all_docs={n:,} vocab={len(vocab):,} topics={args.n_topics}")
    print(
        f"[time] vectorize_fit={fmt_seconds(t_vec_fit)} "
        f"lda_fit={fmt_seconds(t_lda_fit)} transform_all={fmt_seconds(t_tr)} total={fmt_seconds(total)}"
    )
    print(f"[out] {out_ids}")
    print(f"[out] {out_topics}")
    print(f"[out] {out_kw}")


if __name__ == "__main__":
    main()