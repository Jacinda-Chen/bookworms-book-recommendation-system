#!/usr/bin/env python3
"""
scripts/preprocess_descriptions.py

Create an NLP-ready text column for topic modeling and cosine similarity.

Input:
  data/processed/books_with_genres.parquet

Output:
  data/processed/books_text.parquet

Design choices (good for topic modeling + cosine similarity):
- Remove HTML tags (already mostly handled, but we do it again defensively)
- Unescape HTML entities (&amp;, &quot;, etc.)
- Normalize unicode (e.g., curly quotes)
- Lowercase
- Replace URLs/emails with tokens
- Remove most punctuation but keep intra-word apostrophes
- Collapse whitespace
- Do NOT remove stopwords here (often helpful for topic coherence depending on model)
"""
from __future__ import annotations

import argparse
import html
import re
import unicodedata
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq


TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
# Keep letters/numbers/spaces and a few separators; remove the rest.
PUNCT_RE = re.compile(r"[^a-z0-9\s'\-]+")
WS_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--infile", default="data/processed/books_with_genres.parquet")
    p.add_argument("--outfile", default="data/processed/books_text.parquet")
    p.add_argument("--min-len", type=int, default=80, help="Drop descriptions shorter than this after cleaning")
    p.add_argument("--keep-raw", action="store_true", help="Keep description_raw in output")
    return p.parse_args()


def clean_text(text: Optional[str]) -> str:
    if text is None:
        return ""

    # Defensive: ensure string
    if not isinstance(text, str):
        text = str(text)

    # Decode entities and normalize unicode
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)

    # Strip HTML tags
    text = TAG_RE.sub(" ", text)

    # Replace urls/emails with placeholders to reduce noise
    text = URL_RE.sub(" URL ", text)
    text = EMAIL_RE.sub(" EMAIL ", text)

    # Lowercase
    text = text.lower()

    # Remove punctuation/noise (keep apostrophes and hyphens)
    text = PUNCT_RE.sub(" ", text)

    # Collapse whitespace
    text = WS_RE.sub(" ", text).strip()

    return text


def main() -> None:
    args = parse_args()
    infile = Path(args.infile)
    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # Read only columns we need to keep output smaller/faster
    cols = ["book_id", "title", "genres_top", "description_raw", "description"]
    tbl = pq.read_table(infile.as_posix(), columns=[c for c in cols if c in pq.ParquetFile(infile.as_posix()).schema_arrow.names])

    # Prefer "description_raw" if present, else "description"
    if "description_raw" in tbl.schema.names:
        src = tbl["description_raw"].to_pylist()
    else:
        src = tbl["description"].to_pylist()

    cleaned = []
    lengths = []

    for t in src:
        c = clean_text(t)
        cleaned.append(c)
        lengths.append(len(c))

    out = pa.table(
        {
            "book_id": tbl["book_id"],
            "title": tbl["title"] if "title" in tbl.schema.names else pa.array([None] * tbl.num_rows),
            "genres_top": tbl["genres_top"] if "genres_top" in tbl.schema.names else pa.array([""] * tbl.num_rows),
            "description_clean": pa.array(cleaned, type=pa.string()),
            "description_len": pa.array(lengths, type=pa.int32()),
        }
    )

    # Optional: keep raw for debugging / auditing
    if args.keep_raw and "description_raw" in tbl.schema.names:
        out = out.append_column("description_raw", tbl["description_raw"])

    # Drop very short descriptions (helps topic modeling quality)
    mask = pa.array([l >= args.min_len for l in lengths])
    out = out.filter(mask)

    pq.write_table(out, outfile.as_posix(), compression="zstd")
    print(f"[done] wrote {out.num_rows:,} rows -> {outfile}")


if __name__ == "__main__":
    main()