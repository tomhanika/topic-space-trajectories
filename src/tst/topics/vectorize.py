"""Vocabulary and TF-IDF. The fitted Vectorizer is part of the model bundle."""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import scipy.sparse as sp

from ..config import VectorizeConfig
from ..run import limited_threads
from . import curation

log = logging.getLogger("tst.vectorize")


def terms(tokens: list[str], ngram_max: int) -> list[str]:
    if ngram_max == 1:
        return tokens
    return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]


@dataclass
class Vectorizer:
    vocabulary: list[str]
    idf: np.ndarray
    tf: str = "sublinear"
    ngram_max: int = 1
    idf_kind: str = "smooth"
    # Token -> vocabulary term it is merged into (plural and spelling variants, tst.topics.curation).
    variants: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.index = {t: i for i, t in enumerate(self.vocabulary)}

    def normalise(self, tokens: list[str]) -> list[str]:
        v = self.variants
        return [v.get(t, t) for t in tokens] if v else tokens

    @property
    def n_features(self) -> int:
        return len(self.vocabulary)

    def counts(self, docs: list[list[str]]) -> sp.csr_matrix:
        indptr, indices, data = [0], [], []
        index = self.index
        for tokens in docs:
            c = Counter(index[t] for t in terms(self.normalise(tokens), self.ngram_max) if t in index)
            indices.extend(c.keys())
            data.extend(c.values())
            indptr.append(len(indices))
        m = sp.csr_matrix(
            (np.asarray(data, np.float32), np.asarray(indices, np.int32), np.asarray(indptr, np.int64)),
            shape=(len(docs), self.n_features),
        )
        m.sort_indices()
        return m

    def weight(self, counts: sp.csr_matrix) -> sp.csr_matrix:
        x = counts.astype(np.float32, copy=True)
        if self.tf == "sublinear":
            np.log1p(x.data, out=x.data)
        x = x @ sp.diags(self.idf.astype(np.float32))
        norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        return sp.csr_matrix(sp.diags(1.0 / norms.astype(np.float32)) @ x)

    def transform(self, docs: list[list[str]]) -> sp.csr_matrix:
        return self.weight(self.counts(docs))

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "vocabulary.json").write_text(json.dumps(self.vocabulary, ensure_ascii=False))
        np.save(folder / "idf.npy", self.idf)
        (folder / "vectorizer.json").write_text(
            json.dumps({"tf": self.tf, "ngram_max": self.ngram_max, "idf": self.idf_kind})
        )
        (folder / "variants.json").write_text(json.dumps(self.variants, ensure_ascii=False, sort_keys=True))

    @classmethod
    def load(cls, folder: Path) -> Vectorizer:
        meta = json.loads((folder / "vectorizer.json").read_text())
        variants = folder / "variants.json"  # missing in bundles from before 2026-11
        return cls(
            json.loads((folder / "vocabulary.json").read_text()),
            np.load(folder / "idf.npy"),
            meta["tf"], meta["ngram_max"], meta["idf"],
            json.loads(variants.read_text()) if variants.exists() else {},
        )


def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """Smooth idf as in scikit-learn: ln((1+N)/(1+df)) + 1."""
    df = df.astype(np.float64)
    return np.log((1 + n_docs) / (1 + df)) + 1.0


# Spawned workers read their own row groups of tokens.parquet, so the parent only merges results.
_NGRAM = 1
_VEC: Vectorizer | None = None
_VARIANTS: dict[str, str] = {}


def _init_worker(ngram: int, vec_folder: Path | None, variants_path: Path | None = None) -> None:
    global _NGRAM, _VEC, _VARIANTS
    _NGRAM, _VEC = ngram, None if vec_folder is None else Vectorizer.load(vec_folder)
    _VARIANTS = json.loads(variants_path.read_text()) if variants_path is not None else {}


def _read_usable(tokens_path: Path, groups: list[int]) -> tuple[np.ndarray, list[list[str]]]:
    t = pl.from_arrow(pq.ParquetFile(tokens_path).read_row_groups(groups, columns=["doc_id", "tokens", "reason"]))
    t = t.filter(pl.col("reason") == "ok")
    return t["doc_id"].to_numpy(), t["tokens"].to_list()


def _df_worker(args) -> tuple[int, Counter]:
    _, docs = _read_usable(*args)
    c: Counter = Counter()
    v = _VARIANTS
    for tokens in docs:
        c.update(set(terms([v.get(t, t) for t in tokens] if v else tokens, _NGRAM)))
    return len(docs), c


def _counts_worker(args) -> tuple[np.ndarray, sp.csr_matrix]:
    ids, docs = _read_usable(*args)
    return ids, _VEC.counts(docs)


def _chunks(tokens_path: Path, n_chunks: int) -> list[tuple[Path, list[int]]]:
    """Contiguous runs of row groups, in file order."""
    n = pq.ParquetFile(tokens_path).metadata.num_row_groups
    bounds = np.linspace(0, n, min(n, max(1, n_chunks)) + 1).astype(int)
    return [(tokens_path, list(range(a, b))) for a, b in zip(bounds, bounds[1:]) if b > a]


def _pool(n_jobs: int, ngram: int, vec_folder: Path | None, variants_path: Path | None = None) -> ProcessPoolExecutor:
    # Large initargs make process start-up serial and slow (seconds per process on many-core machines),
    # so workers load the vectorizer from disk.
    return ProcessPoolExecutor(max(1, n_jobs), mp_context=mp.get_context("spawn"),
                               initializer=_init_worker, initargs=(ngram, vec_folder, variants_path))


def _document_frequencies(tokens_path: Path, ngram: int, n_jobs: int,
                          variants: dict[str, str] | None = None) -> tuple[int, Counter]:
    df_total: Counter = Counter()
    n_docs = 0
    # One task per process: few large Counters are cheaper to merge than many small ones.
    chunks = _chunks(tokens_path, n_jobs)
    with tempfile.TemporaryDirectory(dir=tokens_path.parent) as tmp:
        variants_path = None
        if variants:
            variants_path = Path(tmp) / "variants.json"
            variants_path.write_text(json.dumps(variants))
        with limited_threads(1), _pool(min(n_jobs, len(chunks)), ngram, None, variants_path) as ex:
            for n, c in ex.map(_df_worker, chunks):
                n_docs += n
                df_total.update(c)
    return n_docs, df_total


def fit_vectorizer(tokens_path: Path, cfg: VectorizeConfig, n_jobs: int) -> tuple[Vectorizer, dict]:
    """The vocabulary (variants merged, stop words removed, min_df <= df <= max_df) and its idf; plus counts
    of what curation did, for the stage info."""
    n_docs, df_total = _document_frequencies(tokens_path, 1, n_jobs)
    if n_docs == 0:
        raise ValueError("No usable documents after preprocessing.")
    info: dict = {}
    variants = curation.build_variants(df_total, cfg.conflate)
    if variants:
        info["variants_merged"] = len(variants)
        info["variant_groups"] = len(set(variants.values()))
    if variants or cfg.ngram_max > 1:
        # Exact document frequencies of the merged words (and of bigrams of them).
        n_docs, df_total = _document_frequencies(tokens_path, cfg.ngram_max, n_jobs, variants)

    stop = set(cfg.stopwords) | (set(curation.academic_stopwords()) if cfg.academic_stopwords else set())
    stop -= set(cfg.keep)
    removed = curation.stopword_names(stop, variants) - curation.stopword_names(cfg.keep, variants)
    removed &= df_total.keys()
    if removed:
        info["stopwords_removed"] = len(removed)
        top = sorted(removed, key=lambda t: -df_total[t])[:30]
        log.info("removed %d stop words, most frequent: %s", len(removed), ", ".join(top))

    min_df = cfg.min_df if isinstance(cfg.min_df, int) else int(np.ceil(cfg.min_df * n_docs))
    max_df = cfg.max_df if isinstance(cfg.max_df, int) else int(np.floor(cfg.max_df * n_docs))
    kept = [(t, d) for t, d in df_total.items() if min_df <= d <= max_df and t not in removed
            and not (cfg.ngram_max > 1 and "_" in t and any(p in removed for p in t.split("_")))]
    kept.sort(key=lambda x: (-x[1], x[0]))
    if cfg.max_features is not None:
        kept = kept[: cfg.max_features]
    vocab = [t for t, _ in kept]
    dfs = np.array([d for _, d in kept], dtype=np.int64)
    log.info("vocabulary: %d of %d terms kept (min_df=%d, max_df=%d, docs=%d, %d variants merged)",
             len(vocab), len(df_total), min_df, max_df, n_docs, len(variants))
    if cfg.ngram_max == 1:  # bigrams may contain merged words whose unigram is not in the vocabulary
        in_vocab = set(vocab)
        variants = {w: t for w, t in variants.items() if t in in_vocab}
    return Vectorizer(vocab, compute_idf(dfs, n_docs), cfg.tf, cfg.ngram_max, variants=variants), info


def build_matrix(tokens_path: Path, vec: Vectorizer, n_jobs: int) -> tuple[sp.csr_matrix, np.ndarray]:
    """TF-IDF matrix of all usable documents and the doc_id of each row."""
    ids, batches = [], []
    chunks = _chunks(tokens_path, 4 * n_jobs)
    with tempfile.TemporaryDirectory(dir=tokens_path.parent) as tmp:
        vec.save(Path(tmp))
        with limited_threads(1), _pool(min(n_jobs, len(chunks)), vec.ngram_max, Path(tmp)) as ex:
            for doc_ids, m in ex.map(_counts_worker, chunks):
                ids.append(doc_ids)
                batches.append(m)
    counts = sp.vstack(batches, format="csr")
    return vec.weight(counts), np.concatenate(ids)
