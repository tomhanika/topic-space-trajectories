"""Papers per year that use each word or frequent two-word phrase, for a term explorer across corpora.

Words are the topic model's vocabulary after curation (plural and spelling variants merged, stop words removed), so
"networks" counts as "network". Phrases are two words that stand next to each other once stop words are removed
("neural network", "climate change"; "theory of mind" becomes "theory mind"), both in the vocabulary, kept if at
least `min_bigram_papers` papers use them. Only the papers of the topic space count (same years as the map).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import scipy.sparse as sp

from .config import TermsConfig
from .run import limited_threads
from .topics.vectorize import Vectorizer

log = logging.getLogger("tst.terms")

# Year offset in the combined (phrase, year) key; years from Y0 to Y0 + 255.
Y0 = 1800
_VEC: Vectorizer | None = None
_YEARS: Path | None = None


def _init(vec_dir: Path, years: Path) -> None:
    global _VEC, _YEARS
    _VEC, _YEARS = Vectorizer.load(vec_dir), years


def _bigram_worker(args: tuple[Path, list[int]]) -> tuple[np.ndarray, np.ndarray]:
    path, groups = args
    t = pl.from_arrow(pq.ParquetFile(path).read_row_groups(groups, columns=["doc_id", "tokens", "reason"]))
    t = t.filter(pl.col("reason") == "ok").join(pl.read_parquet(_YEARS), on="doc_id")
    vec = _VEC
    index, v = vec.index, len(vec.vocabulary)
    keys = []
    for tokens, year in zip(t["tokens"].to_list(), t["year"].to_list()):
        ids = [index.get(w, -1) for w in vec.normalise(tokens)]
        pairs = {a * v + b for a, b in zip(ids, ids[1:]) if a >= 0 and b >= 0 and a != b}
        y = year - Y0
        keys.extend(p * 256 + y for p in pairs)
    u, c = np.unique(np.asarray(keys, dtype=np.int64), return_counts=True)
    return u, c.astype(np.int32)


def compute(tokens: Path, doc_topics: Path, documents: Path, matrix: Path, matrix_index: Path, vec_dir: Path,
            cfg: TermsConfig, n_jobs: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    vec = Vectorizer.load(vec_dir)
    placed = pl.read_parquet(doc_topics, columns=["doc_id"]).join(
        pl.read_parquet(documents, columns=["doc_id", "year"]), on="doc_id")
    years_path = out_dir / "doc_years.parquet"
    placed.write_parquet(years_path)
    totals = placed.group_by("year").len().rename({"len": "papers"}).sort("year")
    totals.write_parquet(out_dir / "years.parquet")

    # Words: column counts of the document-term matrix, per year.
    ids = pl.read_parquet(matrix_index).with_row_index("row").join(placed, on="doc_id")
    X = sp.load_npz(matrix).tocsr()
    parts = []
    for (y,), g in ids.group_by(["year"]):
        df = X[g["row"].to_numpy()].getnnz(axis=0)
        nz = np.flatnonzero(df)
        parts.append(pl.DataFrame({"term_id": nz.astype(np.int32), "year": np.full(len(nz), y, np.int16),
                                   "papers": df[nz].astype(np.int32)}))
    del X
    vocab = pl.DataFrame({"term_id": np.arange(len(vec.vocabulary), dtype=np.int32), "term": vec.vocabulary})
    words = pl.concat(parts).join(vocab, on="term_id").select("term", "year", "papers")
    result = [words]
    n_bigrams = 0
    if cfg.bigrams:
        n = pq.ParquetFile(tokens).metadata.num_row_groups
        bounds = np.linspace(0, n, min(n, max(1, 4 * n_jobs)) + 1).astype(int)
        tasks = [(tokens, list(range(a, b))) for a, b in zip(bounds, bounds[1:]) if b > a]
        keys, counts = [], []
        pool = ProcessPoolExecutor(max(1, min(n_jobs, len(tasks))), mp_context=mp.get_context("spawn"),
                                   initializer=_init, initargs=(vec_dir, years_path))
        with limited_threads(1), pool as ex:
            for u, c in ex.map(_bigram_worker, tasks):
                keys.append(u), counts.append(c)
        kv = (pl.DataFrame({"key": np.concatenate(keys), "papers": np.concatenate(counts)})
              .group_by("key").agg(pl.col("papers").sum())
              .with_columns((pl.col("key") // 256).alias("pair"),
                            (pl.col("key") % 256 + Y0).cast(pl.Int16).alias("year")))
        frequent = kv.group_by("pair").agg(pl.col("papers").sum()).filter(pl.col("papers") >= cfg.min_bigram_papers)
        kv = kv.join(frequent.select("pair"), on="pair")
        v = len(vec.vocabulary)
        voc = np.asarray(vec.vocabulary, dtype=object)
        pairs = kv["pair"].to_numpy()
        names = pl.Series("term", [f"{a} {b}" for a, b in zip(voc[pairs // v], voc[pairs % v])])
        result.append(kv.with_columns(names).select("term", "year", pl.col("papers").cast(pl.Int32)))
        n_bigrams = frequent.height
    terms = pl.concat(result).sort("term", "year")
    terms.write_parquet(out_dir / "terms.parquet")
    years_path.unlink()
    info = {"words": int(words["term"].n_unique()), "phrases": n_bigrams, "rows": terms.height,
            "papers": int(totals["papers"].sum())}
    log.info("terms: %s", info)
    return info
