#!/usr/bin/env python3
"""
scripts/build_tableau_candidates_tfidf.py

Goal
----
Create a Tableau-friendly candidate table for Pattern A (transparent scoring).

Inputs
------
data/processed/neighbors_tfidf.parquet
  Columns: book_id, neighbor_book_id, cosine_sim
  Contains ~50 neighbors per book.

data/processed/books_with_tags.parquet
  One row per book with metadata for display and scoring features.

Output
------
data/processed/tableau_candidates_tfidf.parquet
  One row per candidate pair with component columns:
    - sim_tfidf
    - rank_tfidf
    - genre_overlap (Jaccard on genres_top)
    - neighbor_average_rating
    - neighbor_ratings_count
    - titles/authors/genres/badges for display
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Set

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--neighbors", default="data/processed/neighbors_tfidf.parquet")
    p.add_argument("--books", default="data/processed/books_with_tags.parquet")
    p.add_argument("--out", default="data/processed/tableau_candidates_tfidf.parquet")
    return p.parse_args()


def split_tags(s: str) -> Set[str]:
    s = (s or "").strip()
    if not s:
        return set()
    return {t for t in s.split("|") if t}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Book metadata used for display and for computed components.
    cols = [
        "book_id",
        "title",
        "author_name",
        "genres_top",
        "badges_top",
        "average_rating",
        "ratings_count",
    ]
    books = pd.read_parquet(args.books, columns=cols).copy()
    books["book_id"] = books["book_id"].astype("int64")
    books["title"] = books["title"].fillna("").astype(str)
    books["author_name"] = books["author_name"].fillna("").astype(str)
    books["genres_top"] = books["genres_top"].fillna("").astype(str)
    books["badges_top"] = books["badges_top"].fillna("").astype(str)
    books["average_rating"] = books["average_rating"].fillna(0.0).astype(float)
    books["ratings_count"] = books["ratings_count"].fillna(0).astype(int)

    # Lookups keyed by book_id for fast mapping.
    title_map: Dict[int, str] = dict(zip(books["book_id"], books["title"]))
    author_map: Dict[int, str] = dict(zip(books["book_id"], books["author_name"]))
    genres_str_map: Dict[int, str] = dict(zip(books["book_id"], books["genres_top"]))
    badges_str_map: Dict[int, str] = dict(zip(books["book_id"], books["badges_top"]))
    avg_map: Dict[int, float] = dict(zip(books["book_id"], books["average_rating"]))
    rc_map: Dict[int, int] = dict(zip(books["book_id"], books["ratings_count"]))
    genres_set_map: Dict[int, Set[str]] = {bid: split_tags(g) for bid, g in genres_str_map.items()}

    # Neighbors table already contains a restricted candidate set per book.
    nbr = pd.read_parquet(args.neighbors, columns=["book_id", "neighbor_book_id", "cosine_sim"]).copy()
    nbr["book_id"] = nbr["book_id"].astype("int64")
    nbr["neighbor_book_id"] = nbr["neighbor_book_id"].astype("int64")
    nbr = nbr.rename(columns={"cosine_sim": "sim_tfidf"})

    # Rank within each source book by similarity (1 = best).
    nbr = nbr.sort_values(["book_id", "sim_tfidf"], ascending=[True, False])
    nbr["rank_tfidf"] = nbr.groupby("book_id")["sim_tfidf"].rank(method="first", ascending=False).astype(int)

    # Attach display fields.
    b_ids = nbr["book_id"].tolist()
    n_ids = nbr["neighbor_book_id"].tolist()

    nbr["book_title"] = [title_map.get(int(b), "") for b in b_ids]
    nbr["neighbor_title"] = [title_map.get(int(n), "") for n in n_ids]
    nbr["neighbor_author"] = [author_map.get(int(n), "") for n in n_ids]
    nbr["neighbor_genres_top"] = [genres_str_map.get(int(n), "") for n in n_ids]
    nbr["neighbor_badges_top"] = [badges_str_map.get(int(n), "") for n in n_ids]
    nbr["neighbor_average_rating"] = [float(avg_map.get(int(n), 0.0)) for n in n_ids]
    nbr["neighbor_ratings_count"] = [int(rc_map.get(int(n), 0)) for n in n_ids]

    # Genre overlap as a proxy relevance component.
    nbr["genre_overlap"] = [
        jaccard(genres_set_map.get(int(b), set()), genres_set_map.get(int(n), set()))
        for b, n in zip(b_ids, n_ids)
    ]

    out_cols = [
        "book_id",
        "book_title",
        "neighbor_book_id",
        "neighbor_title",
        "neighbor_author",
        "sim_tfidf",
        "rank_tfidf",
        "genre_overlap",
        "neighbor_average_rating",
        "neighbor_ratings_count",
        "neighbor_genres_top",
        "neighbor_badges_top",
    ]
    out = nbr[out_cols]
    out.to_parquet(out_path, index=False)
    print(f"[done] wrote {len(out):,} rows -> {out_path}")


if __name__ == "__main__":
    main()