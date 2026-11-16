"""Topic collaboration graph.

Construction (the method of the old network.py / network_accumulate.py / topicflow.reorient):
  1. Every paper keeps its top_k topics (others set to 0).
  2. For each unordered author pair {a, b} on a paper, including a == b, the thresholded topic weights
     are added to edge (a, b, year, topic). Self-pairs measure an author's topic expertise.
  3. Edges are accumulated over a sliding window of `window` years ending in each year.
  4. Each edge (a, b, year, topic) is oriented from the author with the higher expertise in that topic
     to the one with lower expertise; ties keep both directions.

Stored as Parquet tables instead of GML: edges(src, dst, year, topic, weight) without self-loops,
expertise(author, year, topic, weight) and authors(author, name, n_papers).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from .config import NetworkConfig

log = logging.getLogger("tst.network")


def topic_long(doc_ids: np.ndarray, theta: np.ndarray, top_k: int) -> pl.DataFrame:
    """(doc_id, topic, w) for each paper's top_k topics with positive weight."""
    top_k = min(top_k, theta.shape[1])
    thr = np.partition(theta, -top_k, axis=1)[:, -top_k]
    mask = (theta >= thr[:, None]) & (theta > 0)
    rows, topics = np.nonzero(mask)
    return pl.DataFrame({
        "doc_id": doc_ids[rows].astype(np.int64),
        "topic": topics.astype(np.int16),
        "w": theta[rows, topics].astype(np.float32),
    })


def build(documents: Path, doc_ids: np.ndarray, theta: np.ndarray, cfg: NetworkConfig, out: Path) -> dict:
    k = theta.shape[1]
    top_k = max(1, round(k / 8)) if cfg.top_k_topics == "auto" else cfg.top_k_topics

    docs = pl.read_parquet(documents, columns=["doc_id", "year", "authors"]).join(
        pl.DataFrame({"doc_id": doc_ids.astype(np.int64)}), on="doc_id")
    if cfg.min_year is not None:
        docs = docs.filter(pl.col("year") >= cfg.min_year)
    if cfg.max_year is not None:
        docs = docs.filter(pl.col("year") <= cfg.max_year)
    n_papers_in = docs.height
    if cfg.max_authors_per_paper is not None:
        docs = docs.filter(pl.col("authors").list.len() <= cfg.max_authors_per_paper)
    max_year = cfg.max_year or int(docs["year"].max())

    pa_ = (docs.select("doc_id", "year", "authors").explode("authors")
           .select("doc_id", "year", pl.col("authors").struct.field("id").alias("author"),
                   pl.col("authors").struct.field("name").alias("name"))
           .filter(pl.col("author").is_not_null()).unique(["doc_id", "author"]))
    authors = pa_.group_by("author").agg(pl.col("name").first(), pl.len().alias("n_papers"))
    authors = authors.filter(pl.col("n_papers") >= cfg.min_author_papers)
    pa_ = pa_.join(authors.select("author"), on="author").select("doc_id", "year", "author")
    log.info("network: %d papers (%d after author cap), %d authors with >= %d papers",
             n_papers_in, docs.height, authors.height, cfg.min_author_papers)

    sel = np.isin(doc_ids, pa_["doc_id"].unique().to_numpy())
    topics = topic_long(doc_ids[sel], theta[sel], top_k)

    # Yearly pair weights: self-join on paper, a <= b, times the paper's thresholded topics.
    yearly = (
        pa_.lazy()
        .join(pa_.lazy().select("doc_id", pl.col("author").alias("b")), on="doc_id")
        .filter(pl.col("author") <= pl.col("b"))
        .join(topics.lazy(), on="doc_id")
        .group_by(pl.col("author").alias("a"), "b", "year", "topic")
        .agg(pl.col("w").sum())
        .collect(engine="streaming")
    )
    # Sliding window: an entry of year y counts for y, ..., y+window-1.
    acc = (
        yearly.lazy()
        .with_columns(pl.int_ranges(0, cfg.window, dtype=pl.Int32).alias("d"))
        .explode("d")
        .with_columns((pl.col("year") + pl.col("d")).alias("year"))
        .filter(pl.col("year") <= max_year)
        .group_by("a", "b", "year", "topic")
        .agg(pl.col("w").sum())
        .collect(engine="streaming")
    )
    del yearly

    expertise = (acc.filter(pl.col("a") == pl.col("b"))
                 .select(pl.col("a").alias("author"), "year", "topic", pl.col("w").alias("weight"))
                 .sort("author", "topic", "year"))
    if cfg.expertise == "cumulative":
        expertise = expertise.with_columns(
            pl.col("weight").cum_sum().over("author", "topic").alias("level"))
    else:
        expertise = expertise.with_columns(pl.col("weight").alias("level"))

    level = expertise.select("author", "year", "topic", "level")
    pairs = (
        acc.lazy().filter(pl.col("a") != pl.col("b"))
        .join(level.lazy().rename({"author": "a", "level": "ea"}), on=["a", "year", "topic"], how="left")
        .join(level.lazy().rename({"author": "b", "level": "eb"}), on=["b", "year", "topic"], how="left")
        .with_columns(pl.col("ea").fill_null(0.0), pl.col("eb").fill_null(0.0))
    )
    forward = pairs.filter(pl.col("ea") >= pl.col("eb")).select(
        pl.col("a").alias("src"), pl.col("b").alias("dst"), "year", "topic", "w")
    backward = pairs.filter(pl.col("eb") >= pl.col("ea")).select(
        pl.col("b").alias("src"), pl.col("a").alias("dst"), "year", "topic", "w")
    edges = (pl.concat([forward, backward]).rename({"w": "weight"})
             .sort("year", "topic").collect(engine="streaming"))
    del acc

    out.mkdir(parents=True, exist_ok=True)
    edges.write_parquet(out / "edges.parquet", row_group_size=1_000_000, statistics=True)
    expertise.write_parquet(out / "expertise.parquet", statistics=True)
    authors.sort("author").write_parquet(out / "authors.parquet")
    info = {
        "top_k_topics": top_k, "papers": docs.height, "authors": authors.height,
        "edges": edges.height, "expertise_rows": expertise.height,
        "collaborations": edges.select(pl.min_horizontal("src", "dst").alias("u"), pl.max_horizontal("src", "dst").alias("v"))
        .unique().height,
        "years": [int(edges["year"].min()), int(edges["year"].max())] if edges.height else None,
    }
    log.info("network: %s", info)
    return info
