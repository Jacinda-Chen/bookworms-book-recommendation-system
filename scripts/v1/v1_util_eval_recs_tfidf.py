#!/usr/bin/env python3
"""
scripts/eval_recs_tfidf.py

Goal
----
Evaluate recommendation quality using proxies available in the dataset.

Metrics (no user labels needed)
-------------------------------
- Genre overlap (Jaccard on genres_top)
- Author diversity in top-K
- Popularity bias (ratings_count of recommended books)
- Human-readable samples (titles + scores)

Inputs
------
data/processed/neighbors_tfidf.parquet
data/processed/books_with_tags.parquet

Output
------
Printed summary + a few sampled recommendation lists.
"""
from __future__ import annotations

import argparse
import random
from typing import Dict, List, Set, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--neighbors", default="data/processed/neighbors_tfidf.parquet")
    p.add_argument("--books", default="data/processed/books_with_tags.parquet")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--sample", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def split_tags(s: str) -> Set[str]:
    s = (s or "").strip()
    return {t for t in s.split("|") if t} if s else set()


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    return (len(a & b) / u) if u else 0.0


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # Load minimal book metadata used for evaluation.
    cols = ["book_id", "title", "author_name", "genres_top", "ratings_count"]
    books = pd.read_parquet(args.books, columns=cols)
    books["title"] = books["title"].fillna("").astype(str)
    books["author_name"] = books["author_name"].fillna("").astype(str)
    books["genres_top"] = books["genres_top"].fillna("").astype(str)
    books["ratings_count"] = books["ratings_count"].fillna(0).astype(int)

    # Build lookups for fast access.
    title: Dict[int, str] = dict(zip(books["book_id"].astype(int), books["title"]))
    author: Dict[int, str] = dict(zip(books["book_id"].astype(int), books["author_name"]))
    genres: Dict[int, Set[str]] = {int(b): split_tags(g) for b, g in zip(books["book_id"], books["genres_top"])}
    pop: Dict[int, int] = dict(zip(books["book_id"].astype(int), books["ratings_count"]))

    # Neighbors table is large; load and keep only needed columns.
    nbr = pd.read_parquet(args.neighbors, columns=["book_id", "neighbor_book_id", "cosine_sim"])

    # Keep top-K per book by cosine_sim.
    nbr = nbr.sort_values(["book_id", "cosine_sim"], ascending=[True, False]).groupby("book_id").head(args.k)

    # Pick sample source books that exist in neighbors.
    source_ids = nbr["book_id"].drop_duplicates().tolist()
    sample_ids = random.sample(source_ids, k=min(args.sample, len(source_ids)))

    # --- Human-readable samples ---
    print("\n=== SAMPLE RECOMMENDATIONS ===")
    for sid in sample_ids:
        s_title = title.get(int(sid), "")
        s_author = author.get(int(sid), "")
        s_genres = "|".join(sorted(genres.get(int(sid), set())))

        print(f"\nSOURCE: {sid} | {s_title} | {s_author}")
        print(f"GENRES: {s_genres}")

        block = nbr[nbr["book_id"] == sid].copy()
        for _, r in block.iterrows():
            nid = int(r["neighbor_book_id"])
            print(
                f"  -> {nid} | {title.get(nid,'')} | {author.get(nid,'')} "
                f"| sim={float(r['cosine_sim']):.3f} | pop={pop.get(nid,0)}"
            )

    # --- Aggregate proxy metrics ---
    overlaps: List[float] = []
    unique_authors: List[int] = []
    neighbor_pops: List[int] = []

    for sid, group in nbr.groupby("book_id"):
        s_set = genres.get(int(sid), set())
        n_auth = set()
        for _, r in group.iterrows():
            nid = int(r["neighbor_book_id"])
            overlaps.append(jaccard(s_set, genres.get(nid, set())))
            n_auth.add(author.get(nid, ""))
            neighbor_pops.append(pop.get(nid, 0))
        unique_authors.append(len([a for a in n_auth if a]))

    print("\n=== METRICS (PROXIES) ===")
    print(f"pairs_evaluated: {len(overlaps):,}")
    if overlaps:
        s = pd.Series(overlaps)
        print(f"genre_overlap_jaccard: mean={s.mean():.3f} median={s.median():.3f} p90={s.quantile(0.9):.3f}")
    if unique_authors:
        s = pd.Series(unique_authors)
        print(f"unique_authors_in_topk: mean={s.mean():.2f} median={s.median():.2f} (k={args.k})")
    if neighbor_pops:
        s = pd.Series(neighbor_pops)
        print(f"neighbor_ratings_count: median={int(s.median())} p90={int(s.quantile(0.9))} p99={int(s.quantile(0.99))}")


if __name__ == "__main__":
    main()