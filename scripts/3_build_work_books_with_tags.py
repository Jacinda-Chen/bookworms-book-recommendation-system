#!/usr/bin/env python3
"""
scripts/3_build_work_books_with_tags.py

Goal
----
Build work-level books tables (catalog + model) from interim edition metadata and
work-level tags (genres/badges).

Inputs
------
data/interim/books.parquet
  Uses description presence from `description` to pick representative editions.
  Keeps `description_raw` for later cleaning.

data/processed_v2/work_genres.parquet
data/processed_v2/work_badges.parquet

Outputs
-------
data/processed_v2/work_books_catalog.parquet
  One row per work_id; representative edition chosen per work; includes tags.

data/processed_v2/work_books_model.parquet
  Same schema as catalog but filtered to has_desc80 = 1.

Representative edition selection
--------------------------------
Representative edition per work_id uses priority:
1) has_desc80 = 1 where description exists and length(trim(description)) >= 80
2) highest ratings_count
3) tie-break: higher average_rating

Tag merging
-----------
- genres_top and badges_top are pipe-delimited strings of the top 25 shelves per work.
- shelves is combined as genres_top|badges_top.

Notes
-----
- Step 4 should clean `description_raw` into `description_clean`/`description_model`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--books", default="data/interim/books.parquet")
    p.add_argument("--work-genres", default="data/processed_v2/work_genres.parquet")
    p.add_argument("--work-badges", default="data/processed_v2/work_badges.parquet")
    p.add_argument("--outdir", default="data/processed_v2")
    p.add_argument("--topn", type=int, default=25)
    p.add_argument("--min-desc-len", type=int, default=80)
    return p.parse_args()


def q(path: str) -> str:
    return path.replace("'", "''")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_catalog = outdir / "work_books_catalog.parquet"
    out_model = outdir / "work_books_model.parquet"

    b_path = q(args.books)
    g_path = q(args.work_genres)
    bd_path = q(args.work_badges)
    o_cat = q(out_catalog.as_posix())
    o_mod = q(out_model.as_posix())

    con = duckdb.connect()

    # Top-N strings per work_id for genres and badges.
    con.execute(
        f"""
        CREATE OR REPLACE VIEW genres_top AS
        SELECT
            work_id,
            string_agg(shelf, '|' ORDER BY count DESC, shelf ASC) AS genres_top
        FROM (
            SELECT
                work_id, shelf, count,
                row_number() OVER (PARTITION BY work_id ORDER BY count DESC, shelf ASC) AS rn
            FROM read_parquet('{g_path}')
        )
        WHERE rn <= {int(args.topn)}
        GROUP BY work_id;
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE VIEW badges_top AS
        SELECT
            work_id,
            string_agg(shelf, '|' ORDER BY count DESC, shelf ASC) AS badges_top
        FROM (
            SELECT
                work_id, shelf, count,
                row_number() OVER (PARTITION BY work_id ORDER BY count DESC, shelf ASC) AS rn
            FROM read_parquet('{bd_path}')
        )
        WHERE rn <= {int(args.topn)}
        GROUP BY work_id;
        """
    )

    # Build catalog: one row per work_id.
    con.execute(
        f"""
        COPY (
            WITH b AS (
                SELECT
                    *,
                    CASE
                        WHEN description IS NOT NULL
                         AND trim(description) <> ''
                         AND length(trim(description)) >= {int(args.min_desc_len)}
                        THEN 1 ELSE 0
                    END AS has_desc80
                FROM read_parquet('{b_path}')
                WHERE work_id IS NOT NULL
            ),
            rep AS (
                SELECT
                    work_id,

                    arg_max(
                        book_id,
                        has_desc80 * 1000000000000000
                        + coalesce(ratings_count, 0) * 1000000
                        + CAST(coalesce(average_rating, 0) * 1000 AS BIGINT)
                    ) AS rep_book_id,

                    arg_max(title,       has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS title,
                    arg_max(author_id,   has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS author_id,
                    arg_max(author_name, has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS author_name,

                    arg_max(language,    has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS language,
                    arg_max(isbn,        has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS isbn,
                    arg_max(isbn13,      has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS isbn13,
                    arg_max(publisher,   has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS publisher,

                    arg_max(publication_date,          has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS publication_date,
                    arg_max(original_publication_date, has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS original_publication_date,
                    arg_max(publication_year,          has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS publication_year,

                    arg_max(num_pages,   has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS num_pages,

                    arg_max(average_rating, has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS average_rating,
                    max(coalesce(ratings_count, 0)) AS ratings_count,
                    arg_max(text_reviews_count, has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS text_reviews_count,

                    arg_max(rating_dist,  has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS rating_dist,
                    arg_max(rating_5,     has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS rating_5,
                    arg_max(rating_4,     has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS rating_4,
                    arg_max(rating_3,     has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS rating_3,
                    arg_max(rating_2,     has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS rating_2,
                    arg_max(rating_1,     has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS rating_1,
                    arg_max(rating_total, has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS rating_total,

                    arg_max(description_raw, has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS description_raw,
                    arg_max(description,     has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS description,

                    arg_max(format,             has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS format,
                    arg_max(edition_information,has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS edition_information,
                    arg_max(image_url,          has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS image_url,

                    arg_max(series_id,       has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS series_id,
                    arg_max(series_name,     has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS series_name,
                    arg_max(series_position, has_desc80 * 1000000000000000 + coalesce(ratings_count,0)*1000000 + CAST(coalesce(average_rating,0)*1000 AS BIGINT)) AS series_position,

                    max(has_desc80) AS has_desc80
                FROM b
                GROUP BY work_id
            )
            SELECT
                rep.*,
                gt.genres_top,
                bt.badges_top,
                CASE
                    WHEN gt.genres_top IS NULL AND bt.badges_top IS NULL THEN NULL
                    WHEN gt.genres_top IS NULL THEN bt.badges_top
                    WHEN bt.badges_top IS NULL THEN gt.genres_top
                    ELSE gt.genres_top || '|' || bt.badges_top
                END AS shelves_combined
            FROM rep
            LEFT JOIN genres_top gt USING (work_id)
            LEFT JOIN badges_top bt USING (work_id)
        )
        TO '{o_cat}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    # Model table: only works with a usable description proxy
    con.execute(
        f"""
        COPY (
            SELECT *
            FROM read_parquet('{o_cat}')
            WHERE has_desc80 = 1
        )
        TO '{o_mod}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    rows_cat = con.execute(f"SELECT COUNT(*) FROM read_parquet('{o_cat}');").fetchone()[0]
    rows_mod = con.execute(f"SELECT COUNT(*) FROM read_parquet('{o_mod}');").fetchone()[0]
    con.close()

    print(f"[done] wrote -> {out_catalog} rows={rows_cat:,}")
    print(f"[done] wrote -> {out_model} rows={rows_mod:,}")
    print(f"[info] topn={args.topn} min_desc_len={args.min_desc_len}")


if __name__ == "__main__":
    main()