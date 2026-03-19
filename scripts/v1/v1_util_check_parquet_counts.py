from pathlib import Path
import pyarrow.parquet as pq

outdir = Path("data/interim")

files = ["books.parquet", "book_authors.parquet", "book_shelves.parquet"]
for f in files:
    p = outdir / f
    pf = pq.ParquetFile(p)
    print(f"{f}: rows={pf.metadata.num_rows:,} row_groups={pf.num_row_groups}")