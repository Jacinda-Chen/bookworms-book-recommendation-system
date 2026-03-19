#!/usr/bin/env python3
"""
scripts/merge_tags_into_books.py

Merge top-N genres and top-N badges into the books table as compact strings.

Inputs:
- data/interim/books.parquet
- data/processed/book_genres.parquet
- data/processed/book_badges.parquet

Output (default):
- data/processed/books_with_tags.parquet

Adds columns:
- genres_top: pipe-delimited genres
- badges_top: pipe-delimited badges
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
    p.add_argument("--badges", default="data/processed/book_badges.parquet")
    p.add_argument("--out", default="data/processed/books_with_tags.parquet")
    p.add_argument("--max-tags", type=int, default=15)
    p.add_argument("--sep", default="|")
    return p.parse_args()


def load_mapping(path: Path, label_col: str) -> Dict[int, List[Tuple[str, int]]]:
    pf = pq.ParquetFile(path.as_posix())
    mapping: Dict[int, List[Tuple[str, int]]] = {}
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["book_id", label_col, "weight"])
        b = t["book_id"].to_pylist()
        lab = t[label_col].to_pylist()
        w = t["weight"].to_pylist()
        for bi, li, wi in zip(b, lab, w):
            if bi is None or li is None:
                continue
            bi = int(bi)
            if bi not in mapping:
                mapping[bi] = []
            mapping[bi].append((str(li), int(wi) if wi is not None else 0))
    return mapping


def merge_top(book_ids: List[int], mapping: Dict[int, List[Tuple[str, int]]], max_tags: int, sep: str) -> List[str]:
    out: List[str] = []
    for bid in book_ids:
        lst = mapping.get(int(bid), [])
        lst.sort(key=lambda x: x[1], reverse=True)
        lst = lst[:max_tags]
        out.append(sep.join([g for g, _ in lst]))
    return out


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    books = pq.read_table(Path(args.books).as_posix())
    book_ids = books["book_id"].to_pylist()

    genres_map = load_mapping(Path(args.genres), "genre")
    badges_map = load_mapping(Path(args.badges), "badge")

    books = books.append_column("genres_top", pa.array(merge_top(book_ids, genres_map, args.max_tags, args.sep), type=pa.string()))
    books = books.append_column("badges_top", pa.array(merge_top(book_ids, badges_map, args.max_tags, args.sep), type=pa.string()))

    pq.write_table(books, out_path.as_posix(), compression="zstd")
    print(f"[done] wrote {books.num_rows:,} rows -> {out_path}")


if __name__ == "__main__":
    main()