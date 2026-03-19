#!/usr/bin/env python3
"""
scripts/flatten_books_file.py

Flatten Goodreads JSONL books into Parquet using the SAME strict filter that yields 744,283 kept books.

Filter approach
----------------------------------------
Keep a record only if every required field is populated.
"EMPTY" means:
- key missing OR
- null OR
- "" (after stripping whitespace)

Required (ASIN is NOT required):
- author_id NOT EMPTY
- author_name NOT EMPTY
- average_rating NOT EMPTY (if 0 then ratings_count must be > 1)
- description NOT EMPTY
- id NOT EMPTY
- isbn NOT EMPTY
- isbn13 NOT EMPTY
- language == "eng"
- num_pages NOT EMPTY and > 0
- original_publication_date NOT EMPTY
- publication_date NOT EMPTY
- publisher NOT EMPTY
- rating_dist NOT EMPTY
- ratings_count NOT EMPTY and > 1
- shelves NOT EMPTY non-empty array
- title NOT EMPTY

Outputs
-------
<outdir>/books.parquet
<outdir>/book_authors.parquet
<outdir>/book_shelves.parquet

Run (PowerShell):
  python scripts\\flatten_books_file.py --books "data\\raw\\books.json\\books.json" --outdir "data\\interim" --progress-every 1000000
"""
from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


# ----------------------------- CLI -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--books", required=True, help="Path to JSONL books.json")
    p.add_argument("--outdir", required=True, help="Output directory for Parquet files")
    p.add_argument("--chunk-rows", type=int, default=200_000, help="Flush every N kept book rows")
    p.add_argument("--progress-every", type=int, default=0, help="Print progress every N seen rows (0=off)")
    p.add_argument("--max-seen", type=int, default=0, help="Debug: stop after N seen rows (0=all)")
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ----------------------------- Read JSONL -----------------------------


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


# ----------------------------- Normalization helpers -----------------------------


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def safe_int(x: Any) -> Optional[int]:
    try:
        return None if is_empty(x) else int(x)
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    try:
        return None if is_empty(x) else float(x)
    except Exception:
        return None


def parse_year(date_str: Any) -> Optional[int]:
    if is_empty(date_str) or not isinstance(date_str, str):
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(date_str[: len(fmt)], fmt).year
        except Exception:
            continue
    m = re.search(r"\b(\d{4})\b", date_str)
    return int(m.group(1)) if m else None


_TAG_RE = re.compile(r"<[^>]+>")
_RATING_RE = re.compile(r"(\d):(\d+)")


def clean_description(text: Any) -> Optional[str]:
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def parse_rating_dist(dist: Any) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
    if is_empty(dist) or not isinstance(dist, str):
        return (None, None, None, None, None, None)

    star_5 = star_4 = star_3 = star_2 = star_1 = total = None
    for part in dist.split("|"):
        part = part.strip()
        if part.startswith("total:"):
            total = safe_int(part.split(":", 1)[1])
            continue
        m = _RATING_RE.fullmatch(part)
        if not m:
            continue
        star = int(m.group(1))
        val = safe_int(m.group(2))
        if star == 5:
            star_5 = val
        elif star == 4:
            star_4 = val
        elif star == 3:
            star_3 = val
        elif star == 2:
            star_2 = val
        elif star == 1:
            star_1 = val
    return (star_5, star_4, star_3, star_2, star_1, total)


def page_bucket(pages: Optional[int]) -> Optional[str]:
    if pages is None:
        return None
    if pages < 100:
        return "short"
    if pages < 300:
        return "medium"
    if pages < 600:
        return "long"
    return "epic"


def first_author_from_authors_array(authors_val: Any) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(authors_val, list) or not authors_val:
        return (None, None)
    for a in authors_val:
        if not isinstance(a, dict):
            continue
        aid = a.get("author_id") or a.get("id")
        aname = a.get("name")
        if not is_empty(aid) and not is_empty(aname):
            return (str(aid), str(aname))
    return (None, None)


# ----------------------------- Filter -----------------------------


def passes_all_required_fields(rec: Dict[str, Any], rejects: Counter) -> bool:
    def reject(reason: str) -> bool:
        rejects[reason] += 1
        return False

    if is_empty(rec.get("id")):
        return reject("missing_id")
    if is_empty(rec.get("title")):
        return reject("missing_title")
    if is_empty(rec.get("description")):
        return reject("missing_description")
    if is_empty(rec.get("isbn")):
        return reject("missing_isbn")
    if is_empty(rec.get("isbn13")):
        return reject("missing_isbn13")
    if rec.get("language") != "eng":
        return reject("language_not_eng")
    if is_empty(rec.get("publication_date")):
        return reject("missing_publication_date")
    if is_empty(rec.get("original_publication_date")):
        return reject("missing_original_publication_date")
    if is_empty(rec.get("publisher")):
        return reject("missing_publisher")
    if is_empty(rec.get("rating_dist")):
        return reject("missing_rating_dist")

    ratings_count = safe_int(rec.get("ratings_count"))
    if ratings_count is None:
        return reject("missing_ratings_count")
    if ratings_count <= 1:
        return reject("ratings_count_le_1")

    avg_rating = safe_float(rec.get("average_rating"))
    if avg_rating is None:
        return reject("missing_average_rating")
    if avg_rating == 0.0 and ratings_count <= 1:
        return reject("avg_rating_0_and_low_ratings")

    num_pages = safe_int(rec.get("num_pages"))
    if num_pages is None:
        return reject("missing_num_pages")
    if num_pages <= 0:
        return reject("num_pages_le_0")

    shelves_val = rec.get("shelves")
    if not isinstance(shelves_val, list):
        return reject("shelves_not_list")
    if len(shelves_val) == 0:
        return reject("shelves_empty")

    author_id = rec.get("author_id")
    author_name = rec.get("author_name")
    if not is_empty(author_id) and not is_empty(author_name):
        return True

    a_id, a_name = first_author_from_authors_array(rec.get("authors"))
    if not is_empty(a_id) and not is_empty(a_name):
        return True

    return reject("missing_author_id_or_name")


# ----------------------------- Flatten -----------------------------


def flatten_book(rec: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    book_id = safe_int(rec.get("id"))
    work_id = safe_int(rec.get("work_id"))

    # link rows
    authors_link: List[Dict[str, Any]] = []
    authors_val = rec.get("authors")
    if isinstance(authors_val, list) and authors_val:
        for i, a in enumerate(authors_val):
            if not isinstance(a, dict):
                continue
            aid = safe_int(a.get("author_id") or a.get("id"))
            aname = a.get("name")
            if book_id is not None and aid is not None and not is_empty(aname):
                authors_link.append({"book_id": book_id, "author_id": aid, "author_order": i, "author_name": aname})
    else:
        aid = safe_int(rec.get("author_id"))
        aname = rec.get("author_name")
        if book_id is not None and aid is not None and not is_empty(aname):
            authors_link.append({"book_id": book_id, "author_id": aid, "author_order": 0, "author_name": aname})

    shelves_link: List[Dict[str, Any]] = []
    shelves_val = rec.get("shelves")
    if isinstance(shelves_val, list) and book_id is not None:
        for s in shelves_val:
            if not isinstance(s, dict):
                continue
            name = s.get("name")
            cnt = safe_int(s.get("count"))
            if not is_empty(name):
                shelves_link.append({"book_id": book_id, "shelf": str(name), "count": cnt})

    star_5, star_4, star_3, star_2, star_1, rating_total = parse_rating_dist(rec.get("rating_dist"))
    pub_year = parse_year(rec.get("publication_date")) or parse_year(rec.get("original_publication_date"))
    pages = safe_int(rec.get("num_pages"))

    books_row = {
        "book_id": book_id,
        "work_id": work_id,
        "title": rec.get("title"),
        "author_id": safe_int(rec.get("author_id")),
        "author_name": rec.get("author_name"),
        "language": rec.get("language"),
        "isbn": rec.get("isbn"),
        "isbn13": rec.get("isbn13"),
        "publisher": rec.get("publisher"),
        "publication_date": rec.get("publication_date"),
        "original_publication_date": rec.get("original_publication_date"),
        "publication_year": pub_year,
        "num_pages": pages,
        "page_bucket": page_bucket(pages),
        "average_rating": safe_float(rec.get("average_rating")),
        "ratings_count": safe_int(rec.get("ratings_count")),
        "text_reviews_count": safe_int(rec.get("text_reviews_count")),
        "rating_dist": rec.get("rating_dist"),
        "rating_5": star_5,
        "rating_4": star_4,
        "rating_3": star_3,
        "rating_2": star_2,
        "rating_1": star_1,
        "rating_total": rating_total,
        "description_raw": rec.get("description"),
        "description": clean_description(rec.get("description")),
        # Optional extras (kept if present)
        "format": rec.get("format"),
        "edition_information": rec.get("edition_information"),
        "image_url": rec.get("image_url"),
        "series_id": safe_int(rec.get("series_id")),
        "series_name": rec.get("series_name"),
        "series_position": safe_int(rec.get("series_position")),
    }

    return books_row, authors_link, shelves_link


# ----------------------------- Parquet sink -----------------------------


@dataclass
class ParquetSink:
    path: Path
    writer: Optional[pq.ParquetWriter] = None

    def write_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        table = pa.Table.from_pylist(rows)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path.as_posix(), table.schema, compression="zstd")
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


# ----------------------------- Main -----------------------------


def main() -> None:
    args = parse_args()
    ensure_dir(Path(args.outdir))
    outdir = Path(args.outdir)

    books_out = ParquetSink(outdir / "books.parquet")
    authors_out = ParquetSink(outdir / "book_authors.parquet")
    shelves_out = ParquetSink(outdir / "book_shelves.parquet")

    books_buf: List[Dict[str, Any]] = []
    authors_buf: List[Dict[str, Any]] = []
    shelves_buf: List[Dict[str, Any]] = []

    seen = 0
    kept = 0
    rejects: Counter = Counter()

    for rec in iter_jsonl(args.books):
        seen += 1

        if not passes_all_required_fields(rec, rejects):
            if args.progress_every and seen % args.progress_every == 0:
                rate = kept / seen if seen else 0.0
                print(f"[progress] seen={seen:,} kept={kept:,} keep_rate={rate:.2%}")
            if args.max_seen and seen >= args.max_seen:
                break
            continue

        row, auth_rows, shelf_rows = flatten_book(rec)
        books_buf.append(row)
        authors_buf.extend(auth_rows)
        shelves_buf.extend(shelf_rows)
        kept += 1

        if kept % args.chunk_rows == 0:
            books_out.write_rows(books_buf)
            authors_out.write_rows(authors_buf)
            shelves_out.write_rows(shelves_buf)
            books_buf.clear()
            authors_buf.clear()
            shelves_buf.clear()
            if args.progress_every:
                rate = kept / seen if seen else 0.0
                print(f"[progress] seen={seen:,} kept={kept:,} keep_rate={rate:.2%}")

        if args.max_seen and seen >= args.max_seen:
            break

        if args.progress_every and seen % args.progress_every == 0:
            rate = kept / seen if seen else 0.0
            print(f"[progress] seen={seen:,} kept={kept:,} keep_rate={rate:.2%}")

    books_out.write_rows(books_buf)
    authors_out.write_rows(authors_buf)
    shelves_out.write_rows(shelves_buf)
    books_out.close()
    authors_out.close()
    shelves_out.close()

    rate = kept / seen if seen else 0.0
    print(f"\n[done] seen={seen:,} kept={kept:,} keep_rate={rate:.2%}\n")
    print("[top reject reasons]")
    for reason, count in rejects.most_common(20):
        print(f"  {reason}: {count:,}")


if __name__ == "__main__":
    main()