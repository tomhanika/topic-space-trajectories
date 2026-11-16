"""OpenAlex: a local store of the works snapshot, and the reader for `input.format: openalex`.

`fetch` downloads the columns the pipeline needs from the public parquet snapshot
(s3://openalex/data/parquet/works/, CC0) and writes one compact parquet file per snapshot file:
abstracts are turned back into text, works without an abstract are dropped. It is resumable and runs once
for all fields; `read` then selects the works of some OpenAlex fields or subfields. `fetch` also stores the
source table (journal, conference, repository, ...) so that repositories and ebook platforms, which are no
venue in the usual sense, can be dropped as venues.

Topic ids: https://api.openalex.org/fields and https://api.openalex.org/subfields
(e.g. field 12 = Arts and Humanities, subfield 1211 = Philosophy).
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from .run import limited_threads

log = logging.getLogger("tst.openalex")

BUCKET = "openalex"
MANIFEST_URL = "https://openalex.s3.amazonaws.com/data/parquet/manifest.json"
COLUMNS = [
    "id", "title", "publication_year", "language", "type", "cited_by_count",
    "authorships.list.element.author.id", "authorships.list.element.author.display_name",
    "primary_topic.subfield.id", "primary_topic.field.id",
    "primary_location.source.display_name", "abstract_inverted_index",
]
AUTHOR_TYPE = pa.struct([("id", pa.int64()), ("name", pa.string())])
STORE_SCHEMA = pa.schema([
    ("work_id", pa.int64()),
    ("part", pa.int32()),
    ("year", pa.int32()),
    ("title", pa.string()),
    ("abstract", pa.string()),
    ("language", pa.string()),
    ("type", pa.string()),
    ("venue", pa.string()),
    ("citations", pa.int64()),
    ("authors", pa.list_(AUTHOR_TYPE)),
    ("subfield", pa.int32()),
    ("field", pa.int32()),
])


def _tail_id(url: str | None) -> int | None:
    """https://openalex.org/A5036823227 -> 5036823227, https://openalex.org/subfields/1211 -> 1211."""
    if not url:
        return None
    tail = url.rsplit("/", 1)[-1].lstrip("WAS")
    return int(tail) if tail.isdigit() else None


def abstract_text(inverted: str | None) -> str | None:
    """Rebuild an abstract from OpenAlex's inverted index {"word": [positions]}."""
    if not inverted:
        return None
    try:
        index = json.loads(inverted)
    except ValueError:
        return None
    pos = {p: w for w, ps in index.items() for p in ps if isinstance(p, int) and 0 <= p < 100_000}
    if not pos:
        return None
    return " ".join(pos[p] for p in sorted(pos))


def _convert(tab: pa.Table, part: int) -> pa.Table:
    rows = tab.to_pylist()
    out = {name: [] for name in STORE_SCHEMA.names}
    for r in rows:
        text = abstract_text(r["abstract_inverted_index"])
        if text is None or r["publication_year"] is None:
            continue
        topic = r["primary_topic"] or {}
        source = (r["primary_location"] or {}).get("source") or {}
        out["work_id"].append(_tail_id(r["id"]))
        out["part"].append(part)
        out["year"].append(r["publication_year"])
        out["title"].append(r["title"])
        out["abstract"].append(text)
        out["language"].append(r["language"])
        out["type"].append(r["type"])
        out["venue"].append(source.get("display_name"))
        out["citations"].append(r["cited_by_count"])
        out["authors"].append([{"id": _tail_id((a.get("author") or {}).get("id")),
                                "name": (a.get("author") or {}).get("display_name")}
                               for a in (r["authorships"] or [])])
        out["subfield"].append(_tail_id((topic.get("subfield") or {}).get("id")))
        out["field"].append(_tail_id((topic.get("field") or {}).get("id")))
    return pa.table(out, schema=STORE_SCHEMA)


def fetch_sources(store: Path) -> int:
    """The name and type of every OpenAlex source, as `store/sources.parquet`."""
    import pyarrow.fs as pfs

    manifest = json.loads(urllib.request.urlopen(MANIFEST_URL, timeout=120).read())
    files = next(e for e in manifest["entities"] if e["entity"] == "sources")["files"]
    fs = pfs.S3FileSystem(anonymous=True, region="us-east-1", retry_strategy=pfs.AwsStandardS3RetryStrategy(10))
    tab = pa.concat_tables(pq.read_table(f["url"].removeprefix("s3://"), filesystem=fs,
                                         columns=["id", "display_name", "type"]) for f in files)
    store.mkdir(parents=True, exist_ok=True)
    pq.write_table(tab, store / "sources.parquet")
    log.info("openalex sources: %d", tab.num_rows)
    return tab.num_rows


def _fetch_file(args) -> tuple[str, int, int]:
    import pyarrow.fs as pfs

    key, part, target = args
    target = Path(target)
    fs = pfs.S3FileSystem(anonymous=True, region="us-east-1", retry_strategy=pfs.AwsStandardS3RetryStrategy(10))
    f = pq.ParquetFile(fs.open_input_file(f"{BUCKET}/{key}"), pre_buffer=True)
    tmp = target.with_suffix(".tmp")
    n_in = n_out = 0
    with pq.ParquetWriter(tmp, STORE_SCHEMA, compression="zstd") as w:
        # Row group by row group keeps memory small.
        for rg in range(f.metadata.num_row_groups):
            tab = _convert(f.read_row_group(rg, columns=COLUMNS), part)
            n_in += f.metadata.row_group(rg).num_rows
            n_out += tab.num_rows
            w.write_table(tab)
    tmp.rename(target)
    return key, n_in, n_out


def fetch(store: Path, n_jobs: int = 48, limit: int | None = None) -> dict:
    """Download the works snapshot's needed columns into `store`. Finished files are skipped on rerun."""
    store.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(urllib.request.urlopen(MANIFEST_URL, timeout=120).read())
    files = next(e for e in manifest["entities"] if e["entity"] == "works")["files"]
    keys = sorted(f["url"].removeprefix(f"s3://{BUCKET}/") for f in files)
    (store / "manifest.json").write_text(json.dumps({"date": manifest["date"], "files": keys}, indent=1))
    fetch_sources(store)
    works = store / "works"
    works.mkdir(exist_ok=True)
    tasks = []
    for part, key in enumerate(keys[:limit] if limit else keys):
        # works/updated_date=2026-10-12/part_0065.parquet -> 2026-10-12_part_0065.parquet
        name = key.split("updated_date=")[1].replace("/", "_")
        if not (works / name).exists():
            tasks.append((key, part, str(works / name)))
    log.info("openalex snapshot %s: %d works files, %d to fetch", manifest["date"], len(keys), len(tasks))
    t0, done, n_in, n_out = time.time(), 0, 0, 0
    pa.set_io_thread_count(8)
    with limited_threads(2), ProcessPoolExecutor(max(1, min(n_jobs, len(tasks) or 1)),
                                                 mp_context=mp.get_context("spawn")) as ex:
        futures = [ex.submit(_fetch_file, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                key, a, b = fut.result()
            except Exception as e:  # noqa: BLE001 - keep going; a rerun retries the missing files
                log.warning("failed: %s", e)
                continue
            done, n_in, n_out = done + 1, n_in + a, n_out + b
            if done % 20 == 0 or done == len(tasks):
                log.info("  %d/%d files, %.1fM works, %.1fM with abstract, %.0f min", done, len(tasks),
                         n_in / 1e6, n_out / 1e6, (time.time() - t0) / 60)
    missing = len(tasks) - done
    if missing:
        log.warning("%d files failed; run fetch again to retry them", missing)
    return {"snapshot": manifest["date"], "fetched": done, "failed": missing, "works": n_in, "with_abstract": n_out}


def snapshot_year(store: Path) -> int | None:
    """Year of the snapshot the store was fetched from (None for a store without manifest)."""
    m = store / "manifest.json"
    return int(json.loads(m.read_text())["date"][:4]) if m.exists() else None


def read(store: Path, fields: list[int], subfields: list[int], types: list[str] | None,
         languages: list[str] | None, venue_types: list[str] | None = None) -> pl.DataFrame:
    """Works of the given OpenAlex fields/subfields (primary topic) from the local store, deduplicated."""
    lf = pl.scan_parquet(store / "works" / "*.parquet")
    sel = []
    if fields:
        sel.append(pl.col("field").is_in(fields))
        if 0 in fields:  # works without a primary topic
            sel.append(pl.col("field").is_null())
    if subfields:
        sel.append(pl.col("subfield").is_in(subfields))
    if not sel:
        raise ValueError("input.openalex needs `fields` or `subfields`")
    lf = lf.filter(pl.any_horizontal(sel))
    if types:
        lf = lf.filter(pl.col("type").is_in(types))
    if languages:
        lf = lf.filter(pl.col("language").is_in(languages))
    # A work should be in one snapshot partition only; if not, keep its latest version.
    df = lf.collect(engine="streaming").sort("part").unique("work_id", keep="last", maintain_order=True)
    if venue_types:
        if not (store / "sources.parquet").exists():
            fetch_sources(store)
        # Matched by name, the works store keeps no source id. A name that is a journal somewhere stays a venue;
        # names missing from the source table (newer than it) are kept.
        src = pl.read_parquet(store / "sources.parquet").group_by("display_name").agg(
            ok=pl.col("type").is_in(venue_types).any())
        df = (df.join(src, left_on="venue", right_on="display_name", how="left")
              .with_columns(venue=pl.when(pl.col("ok").fill_null(True)).then(pl.col("venue")))
              .drop("ok"))
    return df.sort("year", "work_id")
