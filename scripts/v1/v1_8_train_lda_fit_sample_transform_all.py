#!/usr/bin/env python3
"""
scripts/train_lda_fit_sample_transform_all.py

Goal
----
Train an LDA topic model on a sample (for better topics and faster training),
then compute topic distributions for the full dataset using lda.transform().

Why this approach
-----------------
- LDA training on the full corpus can be slow.
- Fitting on a representative sample is faster and often produces good topics.
- Transforming the full corpus produces topic vectors for every book, enabling:
  - cosine similarity on topic vectors
  - Tableau sliders that apply to the full dataset

Input
-----
data/processed/books_text_capped.parquet
  Columns used:
    - book_id
    - description_model

Outputs (data/processed/)
------------------------
book_ids_lda.npy
  book_id aligned to rows of topics_lda.npy (full dataset order)

topics_lda.npy
  Topic distribution per book (n_books x n_topics) for full dataset

topic_keywords_lda.csv
  Top words per topic for transparency

count_vectorizer_lda.joblib
lda_model.joblib
  Saved artifacts for reuse

Notes
-----
- CountVectorizer is used (recommended for LDA).
- Unigrams are used by default for stability and speed.
- learning_method="batch" tends to produce better topics but is slower than "online".
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--infile", default="data/processed/books_text_capped.parquet")
    p.add_argument("--outdir", default="data/processed")
    p.add_argument("--n-topics", type=int, default=50)
    p.add_argument("--sample", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-features", type=int, default=80_000)
    p.add_argument("--min-df", type=int, default=10)
    p.add_argument("--max-df", type=float, default=0.6)
    p.add_argument("--max-iter", type=int, default=10)
    p.add_argument("--learning-method", choices=["batch", "online"], default="batch")
    p.add_argument("--random-state", type=int, default=42)
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

    # 1) Load full dataset (book_id + capped text)
    df_all = pd.read_parquet(args.infile, columns=["book_id", "description_model"]).dropna(subset=["description_model"])
    df_all["description_model"] = df_all["description_model"].astype(str)

    # 2) Sample for LDA training
    if args.sample <= 0 or args.sample >= len(df_all):
        df_fit = df_all.copy()
    else:
        df_fit = df_all.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)

    # 3) CountVectorizer vocabulary (fit on sample)
    t_vec0 = time.time()
    vectorizer = CountVectorizer(
        stop_words="english",
        lowercase=False,  # preprocessing already lowercased
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
        ngram_range=(1, 1),  # unigrams for LDA stability
    )
    X_fit = vectorizer.fit_transform(df_fit["description_model"].tolist())
    t_vec_fit = time.time() - t_vec0

    # 4) Train LDA on sample counts
    t_lda0 = time.time()
    lda = LatentDirichletAllocation(
        n_components=args.n_topics,
        learning_method=args.learning_method,
        max_iter=args.max_iter,
        random_state=args.random_state,
        n_jobs=-1,
    )
    lda.fit(X_fit)
    t_lda_fit = time.time() - t_lda0

    # 5) Transform full dataset into topic distributions
    # Count matrix for full dataset uses the sample-fitted vocabulary.
    t_full0 = time.time()
    X_all = vectorizer.transform(df_all["description_model"].tolist())
    topics_all = lda.transform(X_all).astype(np.float32)
    t_full = time.time() - t_full0

    # 6) Save aligned outputs for full dataset
    book_ids = df_all["book_id"].to_numpy(dtype=np.int64)
    np.save(outdir / "book_ids_lda.npy", book_ids)
    np.save(outdir / "topics_lda.npy", topics_all)

    joblib.dump(vectorizer, outdir / "count_vectorizer_lda.joblib")
    joblib.dump(lda, outdir / "lda_model.joblib")

    # 7) Save topic keywords for interpretability
    vocab = np.array(vectorizer.get_feature_names_out())
    rows = []
    for topic_id, topic_weights in enumerate(lda.components_):
        top_idx = np.argsort(topic_weights)[-args.top_words :][::-1]
        rows.append({"topic_id": topic_id, "top_terms": "|".join(vocab[top_idx].tolist())})
    pd.DataFrame(rows).to_csv(outdir / "topic_keywords_lda.csv", index=False)

    total = time.time() - t0
    print(f"[done] fit_docs={len(df_fit):,} all_docs={len(df_all):,} vocab={len(vocab):,} topics={args.n_topics}")
    print(f"[time] vectorize_fit={fmt_seconds(t_vec_fit)} lda_fit={fmt_seconds(t_lda_fit)} transform_all={fmt_seconds(t_full)} total={fmt_seconds(total)}")
    print(f"[out] {outdir}")


if __name__ == "__main__":
    main()