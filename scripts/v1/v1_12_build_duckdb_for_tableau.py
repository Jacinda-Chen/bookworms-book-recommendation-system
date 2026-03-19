#!/usr/bin/env python3
"""
scripts/build_duckdb_for_tableau.py

Goal
----
Build a DuckDB database file for Tableau ODBC using short-path Parquet inputs.

Why this exists
--------------
- Tableau ODBC dialogs can hit path-length limits with long project paths.
- Tableau can lock the database file while connected.
- Short paths reduce connection issues and simplify DSN configuration.
- Tableau works well with database-like sources (ODBC).
- DuckDB reads Parquet efficiently and stores everything in one .duckdb file.
- Views can simplify Tableau modeling (books table + candidates table).

Inputs (C:\\db)
---------------
books_with_tags.parquet
candidates.parquet

Output (C:\\db)
---------------
goodreads_tableau.duckdb

Views
-----
v_books
v_candidates
"""
from __future__ import annotations

from pathlib import Path

import duckdb

DB = r"C:\db\goodreads_build.duckdb"
BOOKS = r"C:\db\books_with_tags.parquet"
CANDIDATES = r"C:\db\candidates.parquet"


def main() -> None:
    Path(r"C:\db").mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(DB)

    # Tables are materialized inside DuckDB for faster Tableau queries.
    con.execute("DROP TABLE IF EXISTS books;")
    con.execute("CREATE TABLE books AS SELECT * FROM read_parquet(?);", [BOOKS])

    con.execute("DROP TABLE IF EXISTS candidates;")
    con.execute("CREATE TABLE candidates AS SELECT * FROM read_parquet(?);", [CANDIDATES])

    # Views provide stable names for Tableau.
    con.execute("DROP VIEW IF EXISTS v_books;")
    con.execute("CREATE VIEW v_books AS SELECT * FROM books;")

    con.execute("DROP VIEW IF EXISTS v_candidates;")
    con.execute("CREATE VIEW v_candidates AS SELECT * FROM candidates;")

    con.close()
    print("[done] built C:\\db\\goodreads_tableau.duckdb with v_books + v_candidates")


if __name__ == "__main__":
    main()