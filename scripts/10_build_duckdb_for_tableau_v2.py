#!/usr/bin/env python3
"""
scripts/10_build_duckdb_for_tableau_v2.py

Goal
----
Build a DuckDB database for Tableau that exposes:
- works metadata (v_works)
- 4 candidate variant tables (Top-50): v_candidates_tfidf/lda/nmf/equal
- 4 enriched recommendation views joining candidates to works twice:
  v_recs_tfidf / v_recs_lda / v_recs_nmf / v_recs_equal

Output
------
data/db/goodreads_work_v2.duckdb 

DuckDB note
-----------
DuckDB does not allow prepared parameters inside CREATE VIEW ... read_parquet(?).
This script embeds paths directly into SQL (single-quoted and escaped).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--works", default="data/processed_v2/work_books_text.parquet")
    p.add_argument("--cand-tfidf", default="data/processed_v2/candidates_top50_by_tfidf.parquet")
    p.add_argument("--cand-lda", default="data/processed_v2/candidates_top50_by_lda.parquet")
    p.add_argument("--cand-nmf", default="data/processed_v2/candidates_top50_by_nmf.parquet")
    p.add_argument("--cand-equal", default="data/processed_v2/candidates_top50_by_equal.parquet")
    p.add_argument("--outdb", default=r"data\db\goodreads_work_v2.duckdb")
    p.add_argument(
        "--materialize",
        action="store_true",
        help="COPY Parquet into DuckDB tables for portability/faster queries (larger DB).",
    )
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args()


def sql_quote_path(p: str) -> str:
    # Single-quote for SQL string literal; escape internal single quotes.
    return p.replace("'", "''")


def main() -> None:
    args = parse_args()
    outdb = Path(args.outdb)
    outdb.parent.mkdir(parents=True, exist_ok=True)

    works_path = sql_quote_path(args.works)
    tfidf_path = sql_quote_path(args.cand_tfidf)
    lda_path = sql_quote_path(args.cand_lda)
    nmf_path = sql_quote_path(args.cand_nmf)
    eq_path = sql_quote_path(args.cand_equal)

    con = duckdb.connect(outdb.as_posix())

    con.execute("SET preserve_insertion_order=false;")
    con.execute(f"SET threads={int(args.threads)};")

    if args.materialize:
        con.execute(f"CREATE OR REPLACE TABLE works AS SELECT * FROM read_parquet('{works_path}');")
        con.execute(f"CREATE OR REPLACE TABLE cand_tfidf AS SELECT * FROM read_parquet('{tfidf_path}');")
        con.execute(f"CREATE OR REPLACE TABLE cand_lda AS SELECT * FROM read_parquet('{lda_path}');")
        con.execute(f"CREATE OR REPLACE TABLE cand_nmf AS SELECT * FROM read_parquet('{nmf_path}');")
        con.execute(f"CREATE OR REPLACE TABLE cand_equal AS SELECT * FROM read_parquet('{eq_path}');")

        con.execute("CREATE OR REPLACE VIEW v_works AS SELECT * FROM works;")
        con.execute("CREATE OR REPLACE VIEW v_candidates_tfidf AS SELECT * FROM cand_tfidf;")
        con.execute("CREATE OR REPLACE VIEW v_candidates_lda AS SELECT * FROM cand_lda;")
        con.execute("CREATE OR REPLACE VIEW v_candidates_nmf AS SELECT * FROM cand_nmf;")
        con.execute("CREATE OR REPLACE VIEW v_candidates_equal AS SELECT * FROM cand_equal;")

        works_ref = "works"
        cand_ref = {
            "tfidf": "cand_tfidf",
            "lda": "cand_lda",
            "nmf": "cand_nmf",
            "equal": "cand_equal",
        }
    else:
        con.execute(f"CREATE OR REPLACE VIEW v_works AS SELECT * FROM read_parquet('{works_path}');")
        con.execute(f"CREATE OR REPLACE VIEW v_candidates_tfidf AS SELECT * FROM read_parquet('{tfidf_path}');")
        con.execute(f"CREATE OR REPLACE VIEW v_candidates_lda AS SELECT * FROM read_parquet('{lda_path}');")
        con.execute(f"CREATE OR REPLACE VIEW v_candidates_nmf AS SELECT * FROM read_parquet('{nmf_path}');")
        con.execute(f"CREATE OR REPLACE VIEW v_candidates_equal AS SELECT * FROM read_parquet('{eq_path}');")

        works_ref = "v_works"
        cand_ref = {
            "tfidf": "v_candidates_tfidf",
            "lda": "v_candidates_lda",
            "nmf": "v_candidates_nmf",
            "equal": "v_candidates_equal",
        }

    def create_recs_view(view_name: str, cand_table: str, include_score_eq: bool) -> None:
        score_expr = "c.score_eq AS score_eq," if include_score_eq else "NULL::DOUBLE AS score_eq,"
        con.execute(
            f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT
                c.work_id AS src_work_id,
                c.neighbor_work_id AS rec_work_id,

                c.sim_tfidf,
                c.rank_tfidf,
                c.sim_lda,
                c.sim_nmf,
                {score_expr}
                c.rank_variant,

                s.title AS src_title,
                s.author_name AS src_author_name,
                s.publisher AS src_publisher,
                s.publication_date AS src_publication_date,
                s.num_pages AS src_num_pages,
                s.average_rating AS src_average_rating,
                s.ratings_count AS src_ratings_count,
                s.rating_dist AS src_rating_dist,
                s.genres_top AS src_genres_top,
                s.badges_top AS src_badges_top,
                s.series_id AS src_series_id,
                s.series_name AS src_series_name,
                s.description_clean AS src_description_clean,

                r.title AS rec_title,
                r.author_name AS rec_author_name,
                r.publisher AS rec_publisher,
                r.publication_date AS rec_publication_date,
                r.num_pages AS rec_num_pages,
                r.average_rating AS rec_average_rating,
                r.ratings_count AS rec_ratings_count,
                r.rating_dist AS rec_rating_dist,
                r.genres_top AS rec_genres_top,
                r.badges_top AS rec_badges_top,
                r.series_id AS rec_series_id,
                r.series_name AS rec_series_name,
                r.description_clean AS rec_description_clean

            FROM {cand_table} c
            INNER JOIN {works_ref} s ON s.work_id = c.work_id
            INNER JOIN {works_ref} r ON r.work_id = c.neighbor_work_id
            WHERE c.work_id <> c.neighbor_work_id
            ;
            """
        )

    create_recs_view("v_recs_tfidf", cand_ref["tfidf"], include_score_eq=False)
    create_recs_view("v_recs_lda", cand_ref["lda"], include_score_eq=False)
    create_recs_view("v_recs_nmf", cand_ref["nmf"], include_score_eq=False)
    create_recs_view("v_recs_equal", cand_ref["equal"], include_score_eq=True)

    con.close()
    print(f"[done] wrote -> {outdb}")
    print("[info] views: v_works, v_candidates_*, v_recs_*")


if __name__ == "__main__":
    main()