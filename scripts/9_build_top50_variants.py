#!/usr/bin/env python3
"""
scripts/9_build_top50_variants.py

Goal
----
Create 4 Top-50 candidate tables (thin) for comparison:
1) Top-50 by TF-IDF similarity (sim_tfidf / rank_tfidf)
2) Top-50 by LDA similarity (sim_lda)
3) Top-50 by NMF similarity (sim_nmf)
4) Top-50 by combined score with equal weights:
   score_eq = (sim_tfidf + sim_lda + sim_nmf) / 3

Why thin
--------
Duplicating work metadata/text into tens of millions of rows is huge.
Tableau can relate these candidate tables to work_books_text.parquet to fetch
publisher/num_pages/description_clean/rating_dist/etc.

Input
-----
data/processed_v2/neighbors_work_tfidf_k100_lda_nmf.parquet

Outputs
-------
data/processed_v2/candidates_top50_by_tfidf.parquet
data/processed_v2/candidates_top50_by_lda.parquet
data/processed_v2/candidates_top50_by_nmf.parquet
data/processed_v2/candidates_top50_by_equal.parquet

Columns in outputs
------------------
work_id
neighbor_work_id
sim_tfidf
rank_tfidf
sim_lda
sim_nmf
score_eq (only in the combined file)
rank_variant (1..50)  (rank within each variant)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pairs",
        default="data/processed_v2/neighbors_work_tfidf_k100_lda_nmf.parquet",
    )
    p.add_argument("--outdir", default="data/processed_v2")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args()


def write_topk(con: duckdb.DuckDBPyConnection, *, pairs: str, out_path: Path, topk: int, order_expr: str, add_score: bool) -> None:
    score_select = ""
    score_cols = ""
    if add_score:
        score_select = ", (sim_tfidf + sim_lda + sim_nmf) / 3.0 AS score_eq"
        score_cols = ", score_eq"

    con.execute(
        f"""
        COPY (
            SELECT
                work_id,
                neighbor_work_id,
                sim_tfidf,
                rank_tfidf,
                sim_lda,
                sim_nmf
                {score_select},
                row_number() OVER (
                    PARTITION BY work_id
                    ORDER BY {order_expr}
                ) AS rank_variant
            FROM read_parquet('{pairs}')
            QUALIFY rank_variant <= {int(topk)}
        )
        TO '{out_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_tfidf = outdir / "candidates_top50_by_tfidf.parquet"
    out_lda = outdir / "candidates_top50_by_lda.parquet"
    out_nmf = outdir / "candidates_top50_by_nmf.parquet"
    out_eq = outdir / "candidates_top50_by_equal.parquet"

    con = duckdb.connect()

    # Reduce memory overhead and avoid extra bookkeeping.
    con.execute("SET preserve_insertion_order=false;")
    con.execute(f"SET threads={int(args.threads)};")

    # Variant 1: TF-IDF
    # rank_tfidf already encodes TF-IDF ordering, but sim_tfidf is used as primary for safety.
    write_topk(
        con,
        pairs=args.pairs,
        out_path=out_tfidf,
        topk=args.topk,
        order_expr="sim_tfidf DESC, rank_tfidf ASC, neighbor_work_id ASC",
        add_score=False,
    )

    # Variant 2: LDA
    write_topk(
        con,
        pairs=args.pairs,
        out_path=out_lda,
        topk=args.topk,
        order_expr="sim_lda DESC, sim_tfidf DESC, neighbor_work_id ASC",
        add_score=False,
    )

    # Variant 3: NMF
    write_topk(
        con,
        pairs=args.pairs,
        out_path=out_nmf,
        topk=args.topk,
        order_expr="sim_nmf DESC, sim_tfidf DESC, neighbor_work_id ASC",
        add_score=False,
    )

    # Variant 4: equal-weight combined
    # score_eq is computed and then used for ordering.
    con.execute(
        f"""
        COPY (
            SELECT
                work_id,
                neighbor_work_id,
                sim_tfidf,
                rank_tfidf,
                sim_lda,
                sim_nmf,
                (sim_tfidf + sim_lda + sim_nmf) / 3.0 AS score_eq,
                row_number() OVER (
                    PARTITION BY work_id
                    ORDER BY ((sim_tfidf + sim_lda + sim_nmf) / 3.0) DESC, sim_tfidf DESC, neighbor_work_id ASC
                ) AS rank_variant
            FROM read_parquet('{args.pairs}')
            QUALIFY rank_variant <= {int(args.topk)}
        )
        TO '{out_eq.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    con.close()

    print(f"[done] wrote -> {out_tfidf}")
    print(f"[done] wrote -> {out_lda}")
    print(f"[done] wrote -> {out_nmf}")
    print(f"[done] wrote -> {out_eq}")
    print(f"[info] topk={args.topk} threads={args.threads}")


if __name__ == "__main__":
    main()