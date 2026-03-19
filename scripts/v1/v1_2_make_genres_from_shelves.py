#!/usr/bin/env python3
"""
scripts/make_genres_from_shelves.py

Build two clean tag tables from Goodreads shelf data:

1) Genres (genre-ish shelves):
   data/processed/book_genres.parquet
   Columns: book_id, genre, weight

2) Badges (lists/awards/meta shelves):
   data/processed/book_badges.parquet
   Columns: book_id, badge, weight

Why split?
- Genres are used for modeling and "what kind of book is this?"
- Badges are used for dashboard filtering (reddit-top-200, awards, shelfari-favorites, etc.)
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


# Common status/ownership/format shelves that are not useful as genres.
DROP_EXACT = {
    "to-read",
    "uk0-lib",
    "currently-reading",
    "read",
    "tbr",
    "owned",
    "books-i-own",
    "owned-books",
    "my-books",
    "my-library",
    "library",
    "wishlist",
    "wish-list",
    "want-to-read",
    "default",
    "reviewed",
    "recommendations",
    "kindle",
    "ebook",
    "e-book",
    "ebooks",
    "audiobook",
    "audiobooks",
    "audio",
    "audio_wanted",
    "books",
}

# Patterns that indicate "junk shelves" (drop entirely).
DROP_PATTERNS = [
    re.compile(r"(^| )dnf( |$)"),
    re.compile(r"did not finish"),
    re.compile(r"(^| )not read( |$)"),
    re.compile(r"(^| )not-read( |$)"),
    re.compile(r"(^| )not interested( |$)"),
    re.compile(r"(^| )not-interested( |$)"),
    re.compile(r"\bowned\b"),
    re.compile(r"\bwishlist\b"),
    re.compile(r"\btbr\b"),
    re.compile(r"\bkindle\b|\bebook\b|e-book|\baudio\b|\baudiobook\b"),
    re.compile(r"\bbooks we own\b|books-we-own"),
    re.compile(r"\buk0-lib\b"),
    re.compile(r"\blibrary\b"),          # catches things like "the-lewis-library"
    re.compile(r"\bmy[- ]?library\b"),
    re.compile(r"\b\d{4}[- ]books[- ]read\b"),   # 2019-books-read, 2020 books read
    re.compile(r"\bbooks[- ]read\b"),            # books-read variants
    re.compile(r"\bcomics[- ]to[- ]read\b"),     # comics-to-read
    re.compile(r"\bto[- ]read\b"),               # *-to-read variants
    re.compile(r"\b\d[- ]stars?\b"),             # 3-stars, 4 stars
    re.compile(r"\bstar[- ]rating\b"),           # star rating shelves
    re.compile(r"\bdid[- ]not[- ]finish\b"),     # did-not-finish
    re.compile(r"\bdnf\b"),                      # dnf
    re.compile(r"\babandoned\b"),                # abandoned
    re.compile(r"\bgeo\b|\blocation\b|\busa\b"), # catches geo/location style shelves like aaa-geo-...
    re.compile(r"\baaa-geo\b"),                  # explicit prefix
    re.compile(r"\bto[- ]buy\b"),
    re.compile(r"\bto[- ]own\b"),
    re.compile(r"\bbooks[- ]i[- ]have\b"),
    re.compile(r"\bi[- ]own\b"),
    ]

# Badges/lists/awards: keep, but *not* in genres_top.
BADGE_PATTERNS = [
    re.compile(r"\breddit\b"),
    re.compile(r"shelfari"),
    re.compile(r"goodreads[- ]choice"),
    re.compile(r"\btop[- ]\d+\b"),

    # Awards / prizes (keep as badges)
    re.compile(r"\bnational[- ]book[- ]award\b"),
    re.compile(r"\bbooker\b"),
    re.compile(r"\bhugo\b"),
    re.compile(r"\bnebula\b"),
    re.compile(r"\barth?ur[- ]c[- ]clarke\b"),
    re.compile(r"\baward\b|\bprize\b"),

    # Favorites (keep as badges)
    re.compile(r"(^|-)favorites($|-)"),   # childhood-favorites, all-time-favorites
    re.compile(r"\bfavorites?\b"),        # favorites / favourite(s) (US spelling)
    re.compile(r"\bfavourites?\b"),       # UK spelling

    # Classics (keep as badge)
    re.compile(r"\bclassic(s)?\b"),
]

_space_re = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--infile", default="data/interim/book_shelves.parquet")
    p.add_argument("--outdir", default="data/processed")
    p.add_argument("--top-n", type=int, default=15, help="Keep top N per book for each table")
    return p.parse_args()


def normalize_shelf(s: str) -> str:
    s = s.strip().lower()
    s = _space_re.sub(" ", s)
    return s


def keep_top_n(items: List[Tuple[str, int]], n: int) -> List[Tuple[str, int]]:
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:n]


def is_badge(shelf: str) -> bool:
    return any(p.search(shelf) for p in BADGE_PATTERNS)

def force_badge(shelf: str) -> bool:
    # Hard rule: these should never be treated as genres
    return "favorite" in shelf or "favourite" in shelf


def is_drop(shelf: str) -> bool:
    if shelf in DROP_EXACT:
        return True
    return any(p.search(shelf) for p in DROP_PATTERNS)


def write_table(path: Path, key: str, rows: List[Tuple[int, str, int]]) -> None:
    book_id, label, weight = zip(*rows) if rows else ([], [], [])
    tbl = pa.table(
        {
            "book_id": pa.array(book_id, type=pa.int64()),
            key: pa.array(label, type=pa.string()),
            "weight": pa.array(weight, type=pa.int64()),
        }
    )
    pq.write_table(tbl, path.as_posix(), compression="zstd")


def main() -> None:
    args = parse_args()
    infile = Path(args.infile)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    genres_out = outdir / "book_genres.parquet"
    badges_out = outdir / "book_badges.parquet"

    pf = pq.ParquetFile(infile.as_posix())

    # Output rows (we write at the end; OK size for top-N per book).
    genres_rows: List[Tuple[int, str, int]] = []
    badges_rows: List[Tuple[int, str, int]] = []

    current_book_id: int | None = None
    bucket_genres: Dict[str, int] = defaultdict(int)
    bucket_badges: Dict[str, int] = defaultdict(int)

    def flush_current() -> None:
        nonlocal current_book_id, bucket_genres, bucket_badges
        if current_book_id is None:
            bucket_genres.clear()
            bucket_badges.clear()
            return

        # Keep top-N genres and top-N badges for this book.
        for shelf, cnt in keep_top_n(list(bucket_genres.items()), args.top_n):
            genres_rows.append((current_book_id, shelf, int(cnt)))
        for shelf, cnt in keep_top_n(list(bucket_badges.items()), args.top_n):
            badges_rows.append((current_book_id, shelf, int(cnt)))

        bucket_genres.clear()
        bucket_badges.clear()

    seen_rows = 0

    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["book_id", "shelf", "count"])
        b_ids = t["book_id"].to_pylist()
        shelves = t["shelf"].to_pylist()
        counts = t["count"].to_pylist()

        for b, s, c in zip(b_ids, shelves, counts):
            seen_rows += 1
            if b is None or s is None:
                continue

            b_int = int(b)
            shelf = normalize_shelf(str(s))
            if not shelf:
                continue

            if current_book_id is None:
                current_book_id = b_int

            if b_int != current_book_id:
                flush_current()
                current_book_id = b_int

            if is_drop(shelf):
                continue

            cnt = int(c) if c is not None else 0

            # Route to badge vs genre
            if force_badge(shelf) or is_badge(shelf):
                if cnt > bucket_badges[shelf]:
                    bucket_badges[shelf] = cnt
            else:
                if cnt > bucket_genres[shelf]:
                    bucket_genres[shelf] = cnt

    flush_current()

    write_table(genres_out, "genre", genres_rows)
    write_table(badges_out, "badge", badges_rows)

    print(f"[done] scanned_rows={seen_rows:,}")
    print(f"[done] genres_rows={len(genres_rows):,} -> {genres_out}")
    print(f"[done] badges_rows={len(badges_rows):,} -> {badges_out}")


if __name__ == "__main__":
    main()