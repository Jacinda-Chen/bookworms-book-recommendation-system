#!/usr/bin/env python3
"""
scripts/2_make_work_genres_from_shelves.py

Goal
----
Create work-level genres and badges from edition-level shelves.

Step summary
------------
1) Join edition shelves (book_id, shelf, count) to edition metadata (book_id -> work_id)
2) Normalize shelf names in Python (lowercase, trim, whitespace -> hyphen)
3) Aggregate shelves to work-level: (work_id, shelf_norm, count_sum)
4) Classify each shelf_norm into:
   - drop  (workflow/format/noise shelves)
   - badge (awards/lists/provenance/library collections)
   - genre (everything else)
5) Output:
   - data/processed_v2/work_genres.parquet
   - data/processed_v2/work_badges.parquet

Inputs
------
data/interim/books.parquet
  Required columns:
    - book_id
    - work_id

data/interim/book_shelves.parquet
  Required columns:
    - book_id
    - shelf
    - count

Outputs (v2)
------------
data/processed_v2/work_genres.parquet
data/processed_v2/work_badges.parquet

Notes
-----
- Classics is treated as a genre (not a badge).
- uk0-lib and named library collections are treated as badges.
- Library is not dropped broadly; only workflow "library"/"my-library" is dropped.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Pattern

import duckdb


# Exact shelves that are always operational noise (drop entirely).
DROP_EXACT = {
    # reading state
    "to-read",
    "want-to-read",
    "currently-reading",
    "read",
    "tbr",
    # ownership / personal inventory
    "owned",
    "books-i-own",
    "owned-books",
    "my-books",
    "my-library",
    "wishlist",
    "wish-list",
    # platform / format
    "kindle",
    "ebook",
    "e-book",
    "ebooks",
    "audiobook",
    "audiobooks",
    "audio",
    "audio_wanted",
    # generic / ui-ish
    "default",
    "reviewed",
    "recommendations",
    "books",
}

# Patterns that indicate "junk shelves" (drop entirely).
# Tightening change: do NOT drop every shelf containing "library" (too broad).
# Only drop true workflow-style "library" shelves such as "my-library" / "library".
DROP_PATTERNS: list[Pattern[str]] = [
    re.compile(r"(^|-)dnf($|-)"),
    re.compile(r"did[- ]not[- ]finish"),
    re.compile(r"(^|-)not[- ]read($|-)"),
    re.compile(r"(^|-)not[- ]interested($|-)"),
    re.compile(r"\bowned\b"),
    re.compile(r"\bwishlist\b"),
    re.compile(r"\btbr\b"),
    # platform/format
    re.compile(r"\bkindle\b|\bebook\b|e-book|\baudio\b|\baudiobook\b"),
    # ownership variants
    re.compile(r"\bbooks[- ]we[- ]own\b|books-we-own"),
    re.compile(r"\bbooks[- ]i[- ]have\b"),
    re.compile(r"\bi[- ]own\b"),
    re.compile(r"\bto[- ]own\b"),
    re.compile(r"\bto[- ]buy\b"),
    re.compile(r"\bto[- ]read\b"),  # *-to-read variants
    # star-rating shelves
    re.compile(r"\b\d[- ]stars?\b"),
    re.compile(r"\bstar[- ]rating\b"),
    # year tracking shelves
    re.compile(r"\b\d{4}[- ]books[- ]read\b"),
    re.compile(r"\bbooks[- ]read\b"),
    # geo/location style shelves (typically noise for genre modeling)
    re.compile(r"\bgeo\b|\blocation\b"),
    re.compile(r"\baaa-geo\b"),
    # only drop library when clearly operational
    re.compile(r"^(my[- ]?)?library$"),
]

# Badges/lists/awards/provenance: keep, but NOT in genres_top.
# Classics is intentionally NOT included here (kept as genre if present).
BADGE_PATTERNS: list[Pattern[str]] = [
    # provenance / sources
    re.compile(r"\buk0-lib\b"),
    # curated lists
    re.compile(r"\breddit\b"),
    re.compile(r"shelfari"),
    re.compile(r"goodreads[- ]choice"),
    re.compile(r"\btop[- ]\d+\b"),
    # awards / prizes
    re.compile(r"\bnational[- ]book[- ]award\b"),
    re.compile(r"\bbooker\b"),
    re.compile(r"\bhugo\b"),
    re.compile(r"\bnebula\b"),
    re.compile(r"\barth?ur[- ]c[- ]clarke\b"),
    re.compile(r"\baward\b|\bprize\b"),
    re.compile(r"\baward[- ]winner\b|\baward[- ]nominee\b"),
    re.compile(r"\bshortlist\b|\blonglist\b"),
    # favorites
    re.compile(r"(^|-)favorites($|-)"),
    re.compile(r"\bfavorites?\b"),
    re.compile(r"\bfavourites?\b"),
    # library / collection badges (keep as badge, not drop)
    re.compile(r"\bthe[- ].*library\b"),  # e.g., "the-lewis-library"
    re.compile(r"\b.*[- ]library\b"),     # e.g., "lewis-library", "campus library"
]

_space_re = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--books", default="data/interim/books.parquet")
    p.add_argument("--book-shelves", default="data/interim/book_shelves.parquet")
    p.add_argument("--outdir", default="data/processed_v2")
    p.add_argument("--min-count", type=int, default=1, help="Minimum shelf count to keep after aggregation")
    p.add_argument("--max-tags-per-work", type=int, default=25, help="Max genre/badge shelves per work")
    return p.parse_args()


def normalize_shelf(s: str) -> str:
    s = (s or "").strip().lower()
    s = _space_re.sub(" ", s)
    s = s.replace(" ", "-")
    return s


def is_match_any(s: str, pats: Iterable[Pattern[str]]) -> bool:
    return any(p.search(s) for p in pats)


def is_drop(shelf_norm: str) -> bool:
    if shelf_norm in DROP_EXACT:
        return True
    return is_match_any(shelf_norm, DROP_PATTERNS)


def is_badge(shelf_norm: str) -> bool:
    return is_match_any(shelf_norm, BADGE_PATTERNS)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_genres = outdir / "work_genres.parquet"
    out_badges = outdir / "work_badges.parquet"

    con = duckdb.connect()

    # Join shelves -> work_id and normalize shelf strings in SQL lightly (trim only).
    # Full normalization is performed in Python for consistency with regex rules.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE work_shelves_raw AS
        WITH
        b AS (
            SELECT CAST(book_id AS BIGINT) AS book_id, work_id
            FROM read_parquet('{args.books}')
            WHERE work_id IS NOT NULL
        ),
        s AS (
            SELECT
                CAST(book_id AS BIGINT) AS book_id,
                shelf,
                CAST(count AS BIGINT) AS cnt
            FROM read_parquet('{args.book_shelves}')
            WHERE shelf IS NOT NULL AND trim(shelf) <> ''
        )
        SELECT
            b.work_id,
            trim(s.shelf) AS shelf_raw,
            SUM(COALESCE(s.cnt, 0)) AS count
        FROM s
        INNER JOIN b USING (book_id)
        GROUP BY b.work_id, trim(s.shelf)
        HAVING SUM(COALESCE(s.cnt, 0)) >= {int(args.min_count)}
        ;
        """
    )

    # Distinct shelf names are small enough for Python classification.
    shelves = con.execute("SELECT DISTINCT shelf_raw FROM work_shelves_raw;").fetchall()
    shelves = [r[0] for r in shelves if r and isinstance(r[0], str)]

    shelf_kind_rows = []
    for raw in shelves:
        s_norm = normalize_shelf(raw)
        if not s_norm:
            continue

        if is_drop(s_norm):
            kind = "drop"
        elif is_badge(s_norm):
            kind = "badge"
        else:
            kind = "genre"

        shelf_kind_rows.append((raw, s_norm, kind))

    con.execute("DROP TABLE IF EXISTS shelf_kind;")
    con.execute("CREATE TABLE shelf_kind(shelf_raw VARCHAR, shelf VARCHAR, kind VARCHAR);")
    con.executemany("INSERT INTO shelf_kind VALUES (?, ?, ?);", shelf_kind_rows)

    # Normalize + split to genres/badges, cap top tags per work
    con.execute(
        f"""
        CREATE OR REPLACE TABLE work_shelves AS
        SELECT
            ws.work_id,
            sk.shelf,
            ws.count,
            sk.kind
        FROM work_shelves_raw ws
        INNER JOIN shelf_kind sk USING (shelf_raw)
        WHERE sk.kind <> 'drop'
        ;
        """
    )

    con.execute(
        f"""
        COPY (
            SELECT work_id, shelf, count
            FROM (
                SELECT
                    work_id,
                    shelf,
                    count,
                    row_number() OVER (PARTITION BY work_id ORDER BY count DESC, shelf ASC) AS rn
                FROM work_shelves
                WHERE kind = 'genre'
            )
            WHERE rn <= {int(args.max_tags_per_work)}
        )
        TO '{out_genres.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    con.execute(
        f"""
        COPY (
            SELECT work_id, shelf, count
            FROM (
                SELECT
                    work_id,
                    shelf,
                    count,
                    row_number() OVER (PARTITION BY work_id ORDER BY count DESC, shelf ASC) AS rn
                FROM work_shelves
                WHERE kind = 'badge'
            )
            WHERE rn <= {int(args.max_tags_per_work)}
        )
        TO '{out_badges.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    n_genres = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_genres.as_posix()}');").fetchone()[0]
    n_badges = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_badges.as_posix()}');").fetchone()[0]
    con.close()

    print(f"[done] wrote -> {out_genres} rows={n_genres:,}")
    print(f"[done] wrote -> {out_badges} rows={n_badges:,}")


if __name__ == "__main__":
    main()