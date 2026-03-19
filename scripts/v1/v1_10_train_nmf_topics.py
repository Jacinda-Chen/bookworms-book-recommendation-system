#!/usr/bin/env python3
"""
scripts/train_nmf_topics.py

Goal
----
Train an NMF topic model (50 topics) on TF-IDF features and save:
- per-book topic weights (topics_nmf.npy)
- top words per topic (topic_keywords_nmf.csv)
- model/vectorizer artifacts for reuse

TIME
------
(.venv) PS C:\Users\Jacinda Chen\Documents\Georgia Tech\4-2026 CSE 6242 Data and Visual Analytics\Project\goodreads-pipeline> python scripts\train_nmf_topics.py --n-topics 50
[done] docs=735,602 X_shape=(735602, 80000) topics_shape=(735602, 50)
[time] vectorize=2m 30s nmf_train=5m 53s total=8m 25s
[out] data\processed

Why NMF
-------
- NMF on TF-IDF frequently produces cleaner topics than LDA on short descriptions.
- The topic weights per book can be used for cosine similarity and explanation.

Input
-----
data/processed/books_text_capped.parquet
  Columns used:
    - book_id
    - description_model

Outputs (data/processed/)
------------------------
book_ids_nmf.npy
topics_nmf.npy
topic_keywords_nmf.csv
tfidf_vectorizer_nmf.joblib
nmf_model.joblib

Notes
-----
- TF-IDF settings mirror the cosine baseline to keep feature space consistent.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--infile", default="data/processed/books_text_capped.parquet")
    p.add_argument("--outdir", default="data/processed")
    p.add_argument("--n-topics", type=int, default=50)
    p.add_argument("--max-features", type=int, default=80_000)
    p.add_argument("--min-df", type=int, default=10)
    p.add_argument("--max-df", type=float, default=0.6)
    p.add_argument("--ngram-max", type=int, default=2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=300)
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
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    df = pd.read_parquet(args.infile, columns=["book_id", "description_model"]).dropna(subset=["description_model"])
    df["description_model"] = df["description_model"].astype(str)

    book_ids = df["book_id"].to_numpy(dtype=np.int64)
    texts = df["description_model"].tolist()

    # TF-IDF for NMF
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

    # NMF training
    t_nmf0 = time.time()
    nmf = NMF(
        n_components=args.n_topics,
        init="nndsvda",
        random_state=args.random_state,
        max_iter=args.max_iter,
    )
    W = nmf.fit_transform(X).astype(np.float32)
    t_nmf = time.time() - t_nmf0

    # Save artifacts
    np.save(outdir / "book_ids_nmf.npy", book_ids)
    np.save(outdir / "topics_nmf.npy", W)
    joblib.dump(vectorizer, outdir / "tfidf_vectorizer_nmf.joblib")
    joblib.dump(nmf, outdir / "nmf_model.joblib")

    # Save top words per topic
    vocab = np.array(vectorizer.get_feature_names_out())
    H = nmf.components_
    rows = []
    for topic_id, weights in enumerate(H):
        top_idx = np.argsort(weights)[-args.top_words:][::-1]
        rows.append({"topic_id": topic_id, "top_terms": "|".join(vocab[top_idx].tolist())})
    pd.DataFrame(rows).to_csv(outdir / "topic_keywords_nmf.csv", index=False)

    total = time.time() - t0
    print(f"[done] docs={len(book_ids):,} X_shape={X.shape} topics_shape={W.shape}")
    print(f"[time] vectorize={fmt_seconds(t_vec)} nmf_train={fmt_seconds(t_nmf)} total={fmt_seconds(total)}")
    print(f"[out] {outdir}")


if __name__ == "__main__":
    main()