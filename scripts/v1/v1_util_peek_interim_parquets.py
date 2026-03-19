# scripts/peek_interim_parquets.py

# get a sneak peak into what is available in each parquet

import pyarrow.parquet as pq

def peek(path: str, cols: list[str], n: int = 3) -> None:
    t = pq.read_table(path, columns=cols).slice(0, n).to_pydict()
    print("\n===", path, "===")
    for i in range(n):
        row = {c: t[c][i] for c in cols}
        print(row)

peek(
    "data/interim/books.parquet",
    ["book_id", "title", "language", "isbn", "isbn13", "ratings_count", "num_pages", "publisher", "description"],
    n=3,
)

peek(
    "data/interim/book_authors.parquet",
    ["book_id", "author_id", "author_order", "author_name"],
    n=3,
)

peek(
    "data/interim/book_shelves.parquet",
    ["book_id", "shelf", "count"],
    n=3,
)