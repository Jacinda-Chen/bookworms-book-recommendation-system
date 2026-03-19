#!/usr/bin/env python3
"""
scripts/merge_top_genres_into_books.py

Merge top-N genres per book into books.parquet as compact columns.

Inputs:
- data/interim/books.parquet
- data/processed/book_genres.parquet

Output:
- data/processed/books_with_genres.parquet

Adds:
- genres_top: pipe-delimited string of top genres per book (already top-N in book_genres)
- genres_top_weights: "genre:weight" pipe-delimited string (optional, useful for QA)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--books", default="data/interim/books.parquet")
    p.add_argument("--genres", default="data/processed/book_genres.parquet")
    p.add_argument("--out", default="data/processed/books_with_genres.parquet")
    p.add_argument("--max-genres", type=int, default=15, help="How many genres to merge per book")
    p.add_argument("--sep", default="|", help="Separator for merged genre string")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    books_path = Path(args.books)
    genres_path = Path(args.genres)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load books (744k rows) — OK to load at once on most machines.
    books = pq.read_table(books_path.as_posix())
    book_ids = books["book_id"].to_pylist()

    # Build mapping book_id -> list[(genre, weight)] from the genres table (already cleaned).
    pf = pq.ParquetFile(genres_path.as_posix())
    mapping: Dict[int, List[Tuple[str, int]]] = {}

    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["book_id", "genre", "weight"])
        b = t["book_id"].to_pylist()
        g = t["genre"].to_pylist()
        w = t["weight"].to_pylist()
        for bi, gi, wi in zip(b, g, w):
            if bi is None or gi is None:
                continue
            bi = int(bi)
            if bi not in mapping:
                mapping[bi] = []
            mapping[bi].append((str(gi), int(wi) if wi is not None else 0))

    # Create merged columns aligned to books row order.
    genres_top: List[str] = []
    genres_top_weights: List[str] = []

    for bid in book_ids:
        if bid is None:
            genres_top.append("")
            genres_top_weights.append("")
            continue

        lst = mapping.get(int(bid), [])
        # Ensure order is by weight desc, then trim.
        lst.sort(key=lambda x: x[1], reverse=True)
        lst = lst[: args.max_genres]

        genres_top.append(args.sep.join([g for g, _ in lst]))
        genres_top_weights.append(args.sep.join([f"{g}:{w}" for g, w in lst]))

    books = books.append_column("genres_top", pa.array(genres_top, type=pa.string()))
    books = books.append_column("genres_top_weights", pa.array(genres_top_weights, type=pa.string()))

    pq.write_table(books, out_path.as_posix(), compression="zstd")
    print(f"[done] wrote {books.num_rows:,} rows -> {out_path}")


if __name__ == "__main__":
    main()