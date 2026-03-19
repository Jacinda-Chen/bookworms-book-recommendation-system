#!/usr/bin/env python3
"""
scripts/build_duckdb_cdb.py

Goal
----
Build a DuckDB database file from short-path Parquet files for Tableau ODBC.

Inputs (C:\\db)
---------------
books_with_tags.parquet
tableau_candidates.parquet

Output (C:\\db)
---------------
goodreads.duckdb
  Tables:
    - books
    - candidates
  Views:
    - v_books
    - v_candidates
"""
from __future__ import annotations

import duckdb
from pathlib import Path

DB = r"C:\db\goodreads.duckdb"
BOOKS = r"C:\db\books_with_tags.parquet"
CAND = r"C:\db\tableau_candidates.parquet"

def main() -> None:
    Path(r"C:\db").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(DB)

    con.execute("DROP TABLE IF EXISTS books;")
    con.execute("CREATE TABLE books AS SELECT * FROM read_parquet(?);", [BOOKS])

    con.execute("DROP TABLE IF EXISTS candidates;")
    con.execute("CREATE TABLE candidates AS SELECT * FROM read_parquet(?);", [CAND])

    con.execute("DROP VIEW IF EXISTS v_books;")
    con.execute("CREATE VIEW v_books AS SELECT * FROM books;")

    con.execute("DROP VIEW IF EXISTS v_candidates;")
    con.execute("CREATE VIEW v_candidates AS SELECT * FROM candidates;")

    con.close()
    print("[done] C:\\db\\goodreads.duckdb updated with v_books and v_candidates")

if __name__ == "__main__":
    main()