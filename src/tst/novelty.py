"""Paper novelty and citation impact.

Novelty: how unusual a paper's combination of topics is. A paper has a topic if the topic's share in it is at least
`threshold`. For each pair of topics, the papers of the `window` preceding years give the observed number of papers
with both topics and the number expected if topics combined independently; their ratio is the pair's commonness
(Lee, Walsh & Wang 2015, with topics in place of the journals of the cited references; Uzzi et al. 2013). A paper's
novelty is -log10 of the commonness of its rarest pair: 0 = the pair is as common as expected, 1 = ten times rarer,
-1 = ten times more common. Papers with fewer than two topics have no novelty score.

Impact: the percentile of a paper's citation count among the papers of the same publication year in the corpus
(0-100, ties share the mean rank), since older papers had more time to be cited.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np
import polars as pl

from .config import NoveltyConfig

log = logging.getLogger("tst.novelty")

# Pairs among a paper's strongest topics; papers rarely have more than five topics above the threshold.
MAX_TOPICS = 6


def theta_matrix(col: pl.Series) -> np.ndarray:
    """Topic shares as an (n, k) float32 array, from an Array or a List column."""
    a = col.to_numpy()
    return (np.stack(a) if a.ndim == 1 else a).astype(np.float32, copy=False)


def topic_sets(theta: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """The (up to MAX_TOPICS) strongest topics of each paper and whether each is above the threshold."""
    m = min(MAX_TOPICS, theta.shape[1])
    if theta.shape[1] <= m:
        top = np.tile(np.arange(theta.shape[1]), (len(theta), 1))
    else:
        top = np.argpartition(-theta, m - 1, axis=1)[:, :m]
    present = np.take_along_axis(theta, top, axis=1) >= threshold
    return top, present


def pair_counts(top: np.ndarray, present: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Papers per topic (k) and per topic pair (k x k, symmetric) among the given papers."""
    m = top.shape[1]
    ni = np.bincount(top[present], minlength=k).astype(np.int64)
    flat = np.zeros(k * k, np.int64)
    for a, b in itertools.combinations(range(m), 2):
        both = present[:, a] & present[:, b]
        i, j = top[both, a], top[both, b]
        flat += np.bincount(i * k + j, minlength=k * k)
        flat += np.bincount(j * k + i, minlength=k * k)
    return ni, flat.reshape(k, k), len(top)


def rarest_pair(top: np.ndarray, present: np.ndarray, log_common: np.ndarray) -> np.ndarray:
    """-log10 commonness of each paper's rarest topic pair; NaN with fewer than two topics."""
    best = np.full(len(top), np.inf)
    for a, b in itertools.combinations(range(top.shape[1]), 2):
        v = np.where(present[:, a] & present[:, b], log_common[top[:, a], top[:, b]], np.inf)
        np.minimum(best, v, out=best)
    return np.where(np.isfinite(best), -best, np.nan)


def compute(doc_topics: Path, documents: Path, cfg: NoveltyConfig, out: Path) -> dict:
    dt = pl.read_parquet(doc_topics, columns=["doc_id", "theta"])
    docs = pl.read_parquet(documents, columns=["doc_id", "year", "citations"])
    dt = dt.join(docs, on="doc_id", how="left", maintain_order="left")
    theta = theta_matrix(dt["theta"])
    k = theta.shape[1]
    years = dt["year"].to_numpy()
    top, present = topic_sets(theta, cfg.threshold)
    del theta
    n_topics = present.sum(axis=1)

    # Counts per year, then sums over the preceding window.
    uyears = np.unique(years)
    counts = {}
    for y in uyears:
        rows = years == y
        counts[int(y)] = pair_counts(top[rows], present[rows], k)
    novelty = np.full(len(years), np.nan)
    unusual = np.zeros(len(years), bool)
    scored_years = []
    for y in uyears:
        y = int(y)
        prev = [counts[p] for p in range(y - cfg.window, y) if p in counts]
        n = sum(c[2] for c in prev)
        if n < cfg.min_window_papers:
            continue
        ni = sum(c[0] for c in prev).astype(np.float64)
        nij = sum(c[1] for c in prev).astype(np.float64)
        # Add-one smoothing: a pair never seen in the window is rare, not infinitely rare.
        common = (nij + 1.0) * n / ((ni[:, None] + 1.0) * (ni[None, :] + 1.0))
        rows = np.flatnonzero(years == y)
        nov = rarest_pair(top[rows], present[rows], np.log10(common))
        novelty[rows] = nov
        unusual[rows] = nov >= -np.log10(cfg.unusual_ratio)
        scored_years.append(y)

    res = pl.DataFrame({"doc_id": dt["doc_id"], "year": dt["year"], "n_topics": n_topics.astype(np.int8),
                        "novelty": novelty.astype(np.float32), "unusual": unusual,
                        "citations": dt["citations"]})
    res = res.with_columns(pl.col("novelty").fill_nan(None)).with_columns(
        # Percentiles within the year (ties share the mean rank): 0 = lowest, 100 = highest.
        ((pl.col("citations").fill_null(0).rank("average").over("year") - 0.5)
         / pl.len().over("year") * 100).cast(pl.Float32).alias("citation_pct"),
        ((pl.col("novelty").rank("average").over("year") - 0.5)
         / pl.col("novelty").count().over("year") * 100).cast(pl.Float32).alias("novelty_pct"),
    ).with_columns(pl.col("unusual") & pl.col("novelty").is_not_null())
    res.write_parquet(out)
    scored = int(res["novelty"].is_not_null().sum())
    info = {"papers": res.height, "scored": scored, "single_topic": int((n_topics < 2).sum()),
            "unusual": int(unusual.sum()), "years": [min(scored_years), max(scored_years)] if scored_years else None}
    log.info("novelty: %s", info)
    return info
