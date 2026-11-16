"""Corpus readers. Both input formats are converted into one canonical document table.

Canonical columns of documents.parquet:
    doc_id    int64     row number, stable for the run
    year      int32
    venue     string    journal, conference or book series; null if unknown
    title     string
    abstract  string    null if missing
    citations int64     null if unknown
    authors   list<struct<id: int64, name: string>>   id is null when unknown
    source_id string    id in the source database (e.g. OpenAlex W123 or a DOI), null if unknown
"""

from __future__ import annotations

import bz2
import gzip
import io
import logging
import shutil
import subprocess
from pathlib import Path

import polars as pl
import pyarrow as pa

from .config import CsvColumn, InputConfig

log = logging.getLogger("tst.io")

AUTHOR_TYPE = pa.struct([("id", pa.int64()), ("name", pa.string())])
SCHEMA = pa.schema([
    ("doc_id", pa.int64()),
    ("year", pa.int32()),
    ("venue", pa.string()),
    ("title", pa.string()),
    ("abstract", pa.string()),
    ("citations", pa.int64()),
    ("authors", pa.list_(AUTHOR_TYPE)),
    ("source_id", pa.string()),
])
BLOCK_BYTES = 32 * 2**20


def open_binary(path: Path) -> io.BufferedIOBase:
    """Open plain, gzip or bzip2 files as bytes. bzip2 uses lbzip2/pbzip2 (parallel) when available."""
    suffix = path.suffix.lower()
    if suffix == ".bz2":
        for tool in ("lbzip2", "pbzip2"):
            exe = shutil.which(tool)
            if exe:
                return subprocess.Popen([exe, "-dc", str(path)], stdout=subprocess.PIPE,
                                        bufsize=BLOCK_BYTES).stdout
        return bz2.open(path, "rb")
    if suffix == ".gz":
        return gzip.open(path, "rb")
    return open(path, "rb")


def ingest(cfg: InputConfig, out: Path) -> dict:
    """Convert the configured input into the canonical documents.parquet."""
    if cfg.format == "openalex":
        from . import openalex

        o = cfg.openalex
        works = openalex.read(cfg.path, o.fields, o.subfields, o.types, o.languages, o.venue_types)
        if cfg.limit:
            works = works.head(cfg.limit)
        df = works.select("year", "venue", "title", "abstract", "citations", "authors",
                          source_id=pl.lit("W") + pl.col("work_id").cast(pl.String))
        info = {}
    else:
        df, info = read_csv(cfg)
    df = _finalize(df, cfg)
    df.write_parquet(out)
    info.update(documents=df.height, with_abstract=int(df["abstract"].is_not_null().sum()),
                year_range=[df["year"].min(), df["year"].max()] if df.height else None)
    log.info("ingest: %s", info)
    return info


def read_csv(cfg: InputConfig) -> tuple[pl.DataFrame, dict]:
    """A CSV/TSV file with a header row (e.g. a Scopus or Web of Science export) as canonical columns. If the
    path is a folder, all its files are read and concatenated (exports split into several files).

    Years are the first four-digit number of the year cell, so dates work too. Rows without a year are
    dropped. Author ids that are not integers (ORCID, Scopus ids with prefixes, ...) are numbered within
    the corpus; the names stay as they are.
    """
    c = cfg.csv
    names = {k: c.columns.get(k, k) for k in CsvColumn.__args__}
    files = sorted(p for p in cfg.path.iterdir() if p.is_file() and not p.name.startswith(".")) \
        if cfg.path.is_dir() else [cfg.path]
    frames = []
    for path in files:
        with open_binary(path) as f:
            raw = f.read()
        if cfg.encoding.lower().replace("-", "") not in ("utf8", "utf8lossy"):
            raw = raw.decode(cfg.encoding, errors="replace").encode()
        raw = raw.removeprefix(b"\xef\xbb\xbf")  # byte order mark of "UTF-8 (Windows)" exports
        frames.append(pl.read_csv(io.BytesIO(raw), separator=c.separator, infer_schema=False, encoding="utf8-lossy",
                                  quote_char='"', truncate_ragged_lines=True))
    df = pl.concat(frames, how="diagonal")
    if cfg.limit:
        df = df.head(cfg.limit)
    missing = [names[k] for k in ("year", "title") if names[k] not in df.columns]
    if missing:
        raise ValueError(f"CSV input lacks the columns {missing}; it has {df.columns}. Map them with input.csv.columns.")

    def col(key: str) -> pl.Expr:
        if names[key] not in df.columns:
            return pl.lit(None, pl.String)
        e = pl.col(names[key]).str.strip_chars()
        return pl.when(e != "").then(e)

    rows = df.height
    df = df.select(
        year=col("year").str.extract(r"(\d{4})").cast(pl.Int32), venue=col("venue"), title=col("title"),
        abstract=col("abstract"), citations=col("citations").str.extract(r"(\d+)").cast(pl.Int64),
        source_id=col("source_id"), _authors=col("authors"), _ids=col("author_ids"),
    ).filter(pl.col("year").is_not_null())

    numbers: dict[str, int] = {}

    def authors(cell: str | None, ids: str | None) -> list[dict]:
        if cell is None:
            return []
        people = [n.strip() for n in cell.split(c.list_separator)]
        idl = [i.strip() for i in ids.split(c.list_separator)] if ids is not None else []
        if len(idl) != len(people):
            idl = [""] * len(people)
        out = []
        for name, aid in zip(people, idl):
            if not name:
                continue
            if aid.isdigit():
                aid = int(aid)
            elif aid:
                aid = numbers.setdefault(aid, -1 - len(numbers))
            out.append({"id": aid if aid != "" else None, "name": " ".join(name.split())})
        return out

    authors_col = pl.Series("authors", [authors(a, i) for a, i in zip(df["_authors"], df["_ids"])],
                            dtype=pl.List(pl.Struct({"id": pl.Int64, "name": pl.String})))
    df = df.drop("_authors", "_ids").with_columns(authors_col)
    df = df.select([n for n in SCHEMA.names if n != "doc_id"])
    return df, {"rows": rows, "rows_without_year": rows - df.height}


def _finalize(df: pl.DataFrame, cfg: InputConfig) -> pl.DataFrame:
    if cfg.min_year is not None:
        df = df.filter(pl.col("year") >= cfg.min_year)
    if cfg.max_year is not None:
        df = df.filter(pl.col("year") <= cfg.max_year)
    if cfg.venue_aliases:
        df = df.with_columns(pl.col("venue").replace(cfg.venue_aliases))
    return df.with_row_index("doc_id").with_columns(pl.col("doc_id").cast(pl.Int64))
