"""Citation flow between topics: the Poisson citation blockmodel of notes/citation_flow_theory.md.

A citation from paper p to an earlier paper q has the rate  r_p * g(t_p - t_q) * theta_q' B theta_p : r_p the
number of references of p, g an aging kernel over the citation lag, theta the topic mixes from the text (fixed),
B[i, j] the rate at which topic-j work cites topic-i work per unit of exposure. B and g are fitted by EM; the
expected number of citations carried from topic i to topic j given the observed citations is the flow F[i, j].
Direction is read from B[i, j] against B[j, i] (raw flows confuse it with growth); a Hodge decomposition of
log(B[i, j] / B[j, i]) gives each topic an upstream potential and the flow a hierarchy index.

References cross corpora, so `tst citations configs/*.yaml` runs once for all corpora, in steps:

titles   works without an abstract (types of the corpus, primary subfield in the corpus, within its years) get a
         topic mix from their title, with the corpus's main and fine model;
nodes    the citable works of each corpus: its papers (doc_topics) and its title-only works, each with its
         `top_topics` strongest topics, renormalised;
authors  the author ids of all citable works (one pass over the links store), for self-citations;
resolve  the references of the corpus papers become citation pairs (citing, cited) with self and future flags;
fit      per corpus, resolution and window of citing years: B, g, flows, lift against the configuration null,
         potentials and hierarchy, lags and the bootstrap (main resolution);
global   the same on all main topics of all corpora (`global_window` years): flows between fields; channels
         between twin topics (one subject in two corpora, by their topic-word weights) are flagged; with a
         saved fit and without --force only global.json is rewritten;
papers   knowledge-base surprise of each citing paper, from the global fit.

Outputs: <workdir>/citations/ and <workdir>/fine/citations/ per corpus (bundle.json and papers.parquet for the
export stage), <runs>/_citations/ for all corpora (global.json for `tst collect`).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import scipy.sparse as sp
import scipy.sparse.csgraph  # noqa: F401  (sp.csgraph)
from numba import get_num_threads, njit, prange
from scipy.sparse.linalg import lsqr

from .config import RunConfig, load_config
from .run import RunContext, limited_threads, write_json

log = logging.getLogger("tst.citations")

STEPS = ("titles", "nodes", "authors", "resolve", "fit", "global", "papers")
PAPER, TITLE = 0, 1                     # node kinds
SELF, FUTURE, SAME = 1, 2, 4            # pair flags: shared author, cited after citing, a work citing itself


# ---------------------------------------------------------------------------------------------------------------
# Corpora


@dataclass
class Corpus:
    index: int
    cfg: RunConfig
    main: RunContext
    fine: RunContext | None
    k: int
    k_fine: int

    @property
    def name(self) -> str:
        return self.cfg.name

    def resolutions(self) -> list[str]:
        return ["main"] + (["fine"] if self.fine is not None else [])

    def ctx(self, res: str) -> RunContext:
        return self.main if res == "main" else self.fine

    def k_of(self, res: str) -> int:
        return self.k if res == "main" else self.k_fine

    def dir(self, res: str = "main") -> Path:
        d = self.ctx(res).root / "citations"
        d.mkdir(parents=True, exist_ok=True)
        return d


def load_corpora(configs: list[str | Path]) -> list[Corpus]:
    """The published OpenAlex corpora with topics, in the order given."""
    out: list[Corpus] = []
    for path in configs:
        cfg = load_config(path)
        if not cfg.export.publish or cfg.input.format != "openalex" or not cfg.citations.enabled:
            continue
        main = RunContext(cfg, "main")
        if not main.doc_topics.exists():
            log.warning("%s has no doc_topics; left out", cfg.name)
            continue
        fine = RunContext(cfg, "fine") if cfg.fine.enabled else None
        if fine is not None and not fine.doc_topics.exists():
            fine = None
        k = len(json.loads((main.root / "model_topics.json").read_text()))
        kf = len(json.loads((fine.root / "model_topics.json").read_text())) if fine else 0
        out.append(Corpus(len(out), cfg, main, fine, k, kf))
    return out


def links_store(corpora: list[Corpus]) -> Path:
    stores = {(c.cfg.citations.links or c.cfg.input.path / "links").resolve() for c in corpora}
    if len(stores) != 1:
        raise ValueError(f"the corpora use different links stores: {sorted(map(str, stores))}")
    store = stores.pop()
    if not any(store.glob("*.parquet")):
        raise FileNotFoundError(f"no links store in {store}; run `tst openalex-fetch <store> --links`")
    return store


def global_dir(corpora: list[Corpus]) -> Path:
    d = Path(os.path.commonpath([str(c.cfg.workdir) for c in corpora])) / "_citations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def top_topics(theta: np.ndarray, s: int, block: int = 500_000) -> tuple[np.ndarray, np.ndarray]:
    """The s strongest topics of each row (int16, -1 where the weight is 0), weights renormalised to sum 1."""
    n, k = theta.shape
    s = min(s, k)
    idx = np.full((n, s), -1, np.int16)
    w = np.zeros((n, s), np.float32)
    for a in range(0, n, block):
        t = np.asarray(theta[a:a + block], np.float32)
        part = np.argpartition(-t, s - 1, axis=1)[:, :s] if s < k else np.tile(np.arange(k), (len(t), 1))
        pw = np.take_along_axis(t, part, axis=1)
        order = np.argsort(-pw, axis=1)
        part, pw = np.take_along_axis(part, order, axis=1), np.take_along_axis(pw, order, axis=1)
        tot = pw.sum(axis=1, keepdims=True)
        pw = np.divide(pw, tot, out=np.zeros_like(pw), where=tot > 0)
        idx[a:a + block] = np.where(pw > 0, part, -1)
        w[a:a + block] = pw
    return idx, w


# ---------------------------------------------------------------------------------------------------------------
# Step titles: topic mixes of works without an abstract

_TOK: dict = {}


def _tokenize_titles(args: tuple[str, list[str]]) -> sp.csr_matrix:
    from .text import Tokenizer
    from .topics.vectorize import Vectorizer

    model_dir, titles = args
    if model_dir not in _TOK:
        meta = json.loads((Path(model_dir) / "model.json").read_text())["tokenizer"]
        tok = Tokenizer(**{f.name: meta[f.name] for f in dataclasses.fields(Tokenizer) if f.name in meta})
        _TOK.clear()  # tasks come corpus by corpus; keep one vocabulary per worker
        _TOK[model_dir] = (tok, Vectorizer.load(Path(model_dir)))
    tok, vec = _TOK[model_dir]
    return vec.transform([tok(t) for t in titles])


def _corpus_years(c: Corpus) -> tuple[int, int]:
    y = (pl.scan_parquet(c.main.doc_topics).select("doc_id")
         .join(pl.scan_parquet(c.main.documents).select("doc_id", "year"), on="doc_id")
         .select(pl.col("year").min().alias("lo"), pl.col("year").max().alias("hi")).collect())
    return int(y["lo"][0]), int(y["hi"][0])


def title_candidates(links: Path, corpora: list[Corpus]) -> pl.DataFrame:
    """Works without an abstract that belong to a corpus: its subfields or fields, its work types, its years."""
    sub, fld, any_type, types = {}, {}, [], []
    for c in corpora:
        o = c.cfg.input.openalex
        sub.update({s: c.index for s in o.subfields})
        fld.update({f: c.index for f in o.fields if f != 0})
        if o.types is None:
            any_type.append(c.index)
        else:
            types += [(c.index, t) for t in o.types]
    years = pl.DataFrame([(c.index, *_corpus_years(c)) for c in corpora], schema=["corpus", "lo", "hi"],
                         orient="row").with_columns(pl.col("corpus").cast(pl.Int32))
    allowed = pl.DataFrame(types, schema=["corpus", "type"], orient="row").with_columns(
        pl.col("corpus").cast(pl.Int32), ok=pl.lit(True))
    lf = (pl.scan_parquet(str(links / "*.parquet"))
          .filter(~pl.col("has_abstract") & pl.col("title").is_not_null()
                  & ~pl.col("paratext").fill_null(False) & ~pl.col("retracted").fill_null(False))
          .with_columns(corpus=pl.coalesce(
              pl.col("subfield").replace_strict(sub, default=None, return_dtype=pl.Int32),
              pl.col("field").replace_strict(fld, default=None, return_dtype=pl.Int32)))
          .filter(pl.col("corpus").is_not_null())
          .join(years.lazy(), on="corpus")
          .filter(pl.col("year").is_between(pl.col("lo"), pl.col("hi")))
          .join(allowed.lazy(), on=["corpus", "type"], how="left")
          .filter(pl.col("ok").fill_null(False) | pl.col("corpus").is_in(any_type))
          .select("work_id", "year", "title", "corpus"))
    return lf.collect(engine="streaming").sort("corpus", "work_id")


def step_titles(corpora: list[Corpus], links: Path, n_jobs: int, force: bool) -> dict:
    from .topics.model import TopicModel

    todo = [c for c in corpora if c.cfg.citations.title_only
            and (force or not (c.dir() / "titles.parquet").exists())]
    for c in corpora:  # title-only works switched off: an empty table keeps the later steps uniform
        if not c.cfg.citations.title_only:
            pl.DataFrame(schema={"work_id": pl.Int64, "year": pl.Int16, "strength": pl.Float32}).write_parquet(
                c.dir() / "titles.parquet")
            for res in c.resolutions():
                np.savez(c.dir(res) / "titles_theta.npz", idx=np.zeros((0, 1), np.int16),
                         w=np.zeros((0, 1), np.float32))
    if not todo:
        return {"corpora": 0}
    t0 = time.time()
    cand = title_candidates(links, todo)
    log.info("titles: %d candidate works for %d corpora (%.0f s)", cand.height, len(todo), time.time() - t0)
    chunk = 20_000
    tasks, owner = [], []
    for c in todo:
        titles = cand.filter(pl.col("corpus") == c.index)["title"].to_list()
        for a in range(0, len(titles), chunk):
            tasks.append((str(c.main.model_dir), titles[a:a + chunk]))
            owner.append(c.index)
    mats: dict[int, list] = {c.index: [] for c in todo}
    with limited_threads(1), ProcessPoolExecutor(max(1, min(n_jobs, len(tasks))),
                                                 mp_context=mp.get_context("spawn")) as ex:
        for i, X in enumerate(ex.map(_tokenize_titles, tasks, chunksize=1)):
            mats[owner[i]].append(X)
    log.info("titles: tokenized (%.0f s)", time.time() - t0)
    out = {}
    for c in todo:
        rows = cand.filter(pl.col("corpus") == c.index)
        V = len(json.loads((c.main.model_dir / "vocabulary.json").read_text()))
        X = sp.vstack(mats[c.index]).tocsr() if mats[c.index] else sp.csr_matrix((0, V), dtype=np.float32)
        s = c.cfg.citations.top_topics
        model = TopicModel.load(c.main.model_dir)
        theta, strength = model.transform_matrix(X, n_jobs=n_jobs)
        ok = strength > 0
        rows.filter(pl.Series(ok)).select(pl.col("work_id"), pl.col("year").cast(pl.Int16),
                                          strength=pl.Series(strength[ok], dtype=pl.Float32)).write_parquet(
            c.dir() / "titles.parquet")
        idx, w = top_topics(theta[ok], s)
        np.savez(c.dir() / "titles_theta.npz", idx=idx, w=w)
        if c.fine is not None:
            fine = TopicModel.load(c.fine.model_dir)
            if fine.vectorizer.vocabulary != model.vectorizer.vocabulary:
                raise ValueError(f"{c.name}: fine and main model have different vocabularies")
            ft, _ = fine.transform_matrix(X[ok], n_jobs=n_jobs)
            idx, w = top_topics(ft, s)
            np.savez(c.dir("fine") / "titles_theta.npz", idx=idx, w=w)
        out[c.name] = {"candidates": rows.height, "with_topics": int(ok.sum())}
        log.info("titles %s: %d of %d works have a topic mix", c.name, int(ok.sum()), rows.height)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Step nodes: the citable works of each corpus


def step_nodes(corpora: list[Corpus], force: bool) -> dict:
    out = {}
    for c in corpora:
        if not force and (c.dir() / "nodes.parquet").exists():
            continue
        s = c.cfg.citations.top_topics
        dt = pl.read_parquet(c.main.doc_topics, columns=["doc_id", "strength", "theta"])
        docs = (dt.select("doc_id", "strength")
                .join(pl.read_parquet(c.main.documents, columns=["doc_id", "year", "source_id"]),
                      on="doc_id", how="left", maintain_order="left"))
        titles = pl.read_parquet(c.dir() / "titles.parquet")
        nodes = pl.concat([
            docs.select(pl.col("source_id").str.strip_prefix("W").cast(pl.Int64, strict=False).alias("work_id"),
                        pl.col("year").cast(pl.Int16), pl.lit(PAPER, pl.Int8).alias("kind"),
                        pl.col("doc_id"), pl.col("strength").cast(pl.Float32)),
            titles.select("work_id", "year", pl.lit(TITLE, pl.Int8).alias("kind"),
                          pl.lit(None, pl.Int64).alias("doc_id"), "strength"),
        ])
        nodes.write_parquet(c.dir() / "nodes.parquet")
        ti = np.load(c.dir() / "titles_theta.npz")
        idx, w = top_topics(dt["theta"].to_numpy(), s)
        np.savez(c.dir() / "theta.npz", idx=_stack(idx, ti["idx"], s), w=_stack(w, ti["w"], s))
        if c.fine is not None:
            fd = pl.read_parquet(c.fine.doc_topics, columns=["doc_id", "theta"])
            if not np.array_equal(fd["doc_id"].to_numpy(), dt["doc_id"].to_numpy()):
                fd = dt.select("doc_id").join(fd, on="doc_id", how="left", maintain_order="left")
            idx, w = top_topics(fd["theta"].to_numpy(), s)
            tf = np.load(c.dir("fine") / "titles_theta.npz")
            np.savez(c.dir("fine") / "theta.npz", idx=_stack(idx, tf["idx"], s), w=_stack(w, tf["w"], s))
        out[c.name] = {"papers": dt.height, "title_only": titles.height}
        log.info("nodes %s: %d papers, %d title-only works", c.name, dt.height, titles.height)
    return out


def _stack(a: np.ndarray, b: np.ndarray, s: int) -> np.ndarray:
    if b.shape[1] != a.shape[1]:  # an empty title table
        b = np.zeros((0, a.shape[1]), a.dtype)
    return np.vstack([a, b])


@dataclass
class Nodes:
    """All citable works of all corpora, by global position (corpus offset + row)."""
    offsets: np.ndarray        # corpus c owns positions offsets[c] .. offsets[c+1]-1
    work_id: np.ndarray
    year: np.ndarray           # int16
    kind: np.ndarray           # int8

    def corpus_of(self, pos: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.offsets, pos, side="right") - 1


def load_nodes(corpora: list[Corpus]) -> Nodes:
    parts = [pl.read_parquet(c.dir() / "nodes.parquet", columns=["work_id", "year", "kind"]) for c in corpora]
    offsets = np.concatenate([[0], np.cumsum([p.height for p in parts])]).astype(np.int64)
    allp = pl.concat(parts)
    return Nodes(offsets, allp["work_id"].fill_null(-1).to_numpy(), allp["year"].to_numpy(),
                 allp["kind"].to_numpy())


@njit(parallel=True, cache=True)
def _fill_table(table, ids):
    for i in prange(ids.shape[0]):
        w = ids[i]
        if 0 <= w < table.shape[0]:
            table[w] = i + 1


@njit(parallel=True, cache=True)
def _lookup(table, ids):
    out = np.full(ids.shape[0], -1, np.int64)
    for i in prange(ids.shape[0]):
        w = ids[i]
        if 0 <= w < table.shape[0]:
            out[i] = np.int64(table[w]) - 1
    return out


def lookup_table(nodes: Nodes) -> np.ndarray:
    """work id -> position + 1 (0: not citable), a direct table over all work ids (about 29 GB for OpenAlex)."""
    table = np.zeros(int(nodes.work_id.max()) + 1, np.uint32)
    _fill_table(table, nodes.work_id)
    return table


# ---------------------------------------------------------------------------------------------------------------
# Step authors: author ids of the citable works (for self-citations)


@njit(parallel=True, cache=True)
def _gather(offs, vals, rows, out_off, out):
    for k in prange(rows.shape[0]):
        r = rows[k]
        o = out_off[k]
        for x in range(offs[r], offs[r + 1]):
            out[o] = vals[x]
            o += 1


@njit(parallel=True, cache=True)
def _scatter_sorted(src_off, src, pos, dst_off, dst):
    """Copy segment k of src to the segment of position pos[k] in dst and sort it."""
    for k in prange(pos.shape[0]):
        a, b = src_off[k], src_off[k + 1]
        o = dst_off[pos[k]]
        seg = np.sort(src[a:b])
        for x in range(b - a):
            dst[o + x] = seg[x]


def _list_parts(col) -> tuple[np.ndarray, np.ndarray]:
    """A pyarrow list column -> (offsets int64, flat values)."""
    arr = col.combine_chunks()
    offs = arr.offsets.to_numpy().astype(np.int64)
    return offs - offs[0], arr.flatten()


def step_authors(corpora: list[Corpus], links: Path, gdir: Path, force: bool) -> dict:
    if not force and (gdir / "authors_ids.npy").exists():
        return {}
    t0 = time.time()
    nodes = load_nodes(corpora)
    table = lookup_table(nodes)
    pos_l, cnt_l, ids_l = [], [], []
    for f in sorted(links.glob("*.parquet")):
        t = pq.read_table(f, columns=["work_id", "authors"])
        pos = _lookup(table, t["work_id"].to_numpy())
        rows = np.flatnonzero(pos >= 0)
        if not len(rows):
            continue
        offs, vals = _list_parts(t["authors"])
        ids = vals.field("id").fill_null(-1).to_numpy() if len(vals) else np.zeros(0, np.int64)
        cnt = (offs[1:] - offs[:-1])[rows]
        out_off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
        out = np.empty(out_off[-1], np.int64)
        _gather(offs, ids, rows, out_off, out)
        pos_l.append(pos[rows])
        cnt_l.append(cnt)
        ids_l.append(out)
    pos = np.concatenate(pos_l)
    cnt = np.concatenate(cnt_l)
    src = np.concatenate(ids_l)
    src_off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
    per = np.zeros(len(nodes.work_id), np.int64)
    per[pos] = cnt
    dst_off = np.concatenate([[0], np.cumsum(per)]).astype(np.int64)
    dst = np.empty(dst_off[-1], np.int64)
    _scatter_sorted(src_off, src, pos, dst_off, dst)
    np.save(gdir / "authors_offsets.npy", dst_off)
    np.save(gdir / "authors_ids.npy", dst)
    log.info("authors: %d works, %d authorships (%.0f s)", len(pos), len(dst), time.time() - t0)
    return {"works": int(len(pos)), "authorships": int(len(dst)), "missing": int(len(nodes.work_id) - len(pos))}


# ---------------------------------------------------------------------------------------------------------------
# Step resolve: references -> citation pairs


@njit(cache=True)
def _shared(a, b):
    i = j = 0
    while i < a.shape[0] and j < b.shape[0]:
        if a[i] == b[j]:
            if a[i] >= 0:
                return True
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return False


@njit(parallel=True, cache=True)
def _resolve_count(pos_p, offs, vals, table, kind):
    n = pos_p.shape[0]
    cnt = np.zeros(n, np.int64)
    for r in prange(n):
        p = pos_p[r]
        if p < 0 or kind[p] != 0:
            continue
        c = 0
        for x in range(offs[r], offs[r + 1]):
            v = vals[x]
            if 0 <= v < table.shape[0] and table[v] > 0:
                c += 1
        cnt[r] = c
    return cnt


@njit(parallel=True, cache=True)
def _resolve_fill(pos_p, offs, vals, table, kind, year, a_off, a_ids, start, out_p, out_q, out_f):
    for r in prange(pos_p.shape[0]):
        p = pos_p[r]
        if p < 0 or kind[p] != 0:
            continue
        o = start[r]
        ap = a_ids[a_off[p]:a_off[p + 1]]
        for x in range(offs[r], offs[r + 1]):
            v = vals[x]
            if not (0 <= v < table.shape[0]) or table[v] == 0:
                continue
            q = np.int64(table[v]) - 1
            f = 0
            if q == p:
                f |= 4
            if year[q] > year[p]:
                f |= 2
            if _shared(ap, a_ids[a_off[q]:a_off[q + 1]]):
                f |= 1
            out_p[o] = p
            out_q[o] = q
            out_f[o] = f
            o += 1


def step_resolve(corpora: list[Corpus], links: Path, gdir: Path, force: bool) -> dict:
    if not force and all((c.dir() / "pairs.parquet").exists() for c in corpora):
        return {}
    t0 = time.time()
    nodes = load_nodes(corpora)
    table = lookup_table(nodes)
    a_off = np.load(gdir / "authors_offsets.npy", mmap_mode="r")
    a_ids = np.load(gdir / "authors_ids.npy", mmap_mode="r")
    a_off, a_ids = np.ascontiguousarray(a_off), np.ascontiguousarray(a_ids)
    refs_total = np.zeros(len(nodes.work_id), np.int32)
    P, Q, Fl = [], [], []
    for f in sorted(links.glob("*.parquet")):
        t = pq.read_table(f, columns=["work_id", "refs"])
        pos_p = _lookup(table, t["work_id"].to_numpy())
        if not (pos_p >= 0).any():
            continue
        offs, vals = _list_parts(t["refs"])
        vals = vals.fill_null(-1).to_numpy() if len(vals) else np.zeros(0, np.int64)
        papers = (pos_p >= 0) & (nodes.kind[np.maximum(pos_p, 0)] == PAPER)
        refs_total[pos_p[papers]] = (offs[1:] - offs[:-1])[papers]
        cnt = _resolve_count(pos_p, offs, vals, table, nodes.kind)
        start = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
        out_p = np.empty(start[-1], np.int64)
        out_q = np.empty(start[-1], np.int64)
        out_f = np.empty(start[-1], np.uint8)
        _resolve_fill(pos_p, offs, vals, table, nodes.kind, nodes.year, a_off, a_ids, start, out_p, out_q, out_f)
        P.append(out_p.astype(np.int32))
        Q.append(out_q.astype(np.int32))
        Fl.append(out_f)
    del table
    P, Q, Fl = np.concatenate(P), np.concatenate(Q), np.concatenate(Fl)
    order = np.lexsort((Q, P))
    P, Q, Fl = P[order], Q[order], Fl[order]
    dup = np.zeros(len(P), bool)
    dup[1:] = (P[1:] == P[:-1]) & (Q[1:] == Q[:-1])  # a reference listed twice counts once
    P, Q, Fl = P[~dup], Q[~dup], Fl[~dup]
    cp, cq = nodes.corpus_of(P), nodes.corpus_of(Q)
    bounds = np.searchsorted(cp, np.arange(len(corpora) + 1))
    out = {}
    for c in corpora:
        a, b = bounds[c.index], bounds[c.index + 1]
        off = nodes.offsets[c.index]
        pl.DataFrame({
            "citing": (P[a:b] - off).astype(np.int32),
            "cited_corpus": cq[a:b].astype(np.int16),
            "cited": (Q[a:b] - nodes.offsets[cq[a:b]]).astype(np.int32),
            "flags": Fl[a:b],
        }).write_parquet(c.dir() / "pairs.parquet")
        n_papers = int((nodes.kind[off:nodes.offsets[c.index + 1]] == PAPER).sum())
        pl.DataFrame({"row": np.arange(n_papers, dtype=np.int32),
                      "refs": refs_total[off:off + n_papers]}).write_parquet(c.dir() / "refs.parquet")
        out[c.name] = int(b - a)
    log.info("resolve: %d citation pairs (%.0f s)", len(P), time.time() - t0)
    return {"pairs": int(len(P)), "per_corpus": out}


# ---------------------------------------------------------------------------------------------------------------
# EM kernels


@njit(parallel=True, fastmath=True, cache=True)
def _estep(p, q, w, ti, tw, B, buf):
    """G[i, j] = sum_e w_e theta_qi theta_pj / d_e with d_e = theta_q' B theta_p, and sum_e w_e log d_e."""
    nthr, K, s, E = buf.shape[0], B.shape[0], ti.shape[1], p.shape[0]
    ll = np.zeros(nthr)
    chunk = (E + nthr - 1) // nthr
    for t in prange(nthr):
        acc = buf[t]
        acc[:, :] = 0.0
        for e in range(t * chunk, min(E, (t + 1) * chunk)):
            a, b = q[e], p[e]
            d = 0.0
            for x in range(s):
                i = ti[a, x]
                if i < 0:
                    break
                for y in range(s):
                    j = ti[b, y]
                    if j < 0:
                        break
                    d += tw[a, x] * B[i, j] * tw[b, y]
            if d <= 0.0:
                continue
            ll[t] += w[e] * math.log(d)
            f = w[e] / d
            for x in range(s):
                i = ti[a, x]
                if i < 0:
                    break
                fx = tw[a, x] * f
                for y in range(s):
                    j = ti[b, y]
                    if j < 0:
                        break
                    acc[i, j] += fx * tw[b, y]
    G = np.zeros((K, K))
    for i in prange(K):
        for t in range(nthr):
            for j in range(K):
                G[i, j] += buf[t, i, j]
    return G, ll.sum()


@njit(parallel=True, fastmath=True, cache=True)
def _estep_lags(p, q, w, lag, ti, tw, B, nlag, nthr):
    """Like _estep, by citation lag: G[i, j, lag]."""
    K, s, E = B.shape[0], ti.shape[1], p.shape[0]
    buf = np.zeros((nthr, K, K, nlag))
    chunk = (E + nthr - 1) // nthr
    for t in prange(nthr):
        for e in range(t * chunk, min(E, (t + 1) * chunk)):
            a, b, L = q[e], p[e], lag[e]
            d = 0.0
            for x in range(s):
                i = ti[a, x]
                if i < 0:
                    break
                for y in range(s):
                    j = ti[b, y]
                    if j < 0:
                        break
                    d += tw[a, x] * B[i, j] * tw[b, y]
            if d <= 0.0:
                continue
            f = w[e] / d
            for x in range(s):
                i = ti[a, x]
                if i < 0:
                    break
                for y in range(s):
                    j = ti[b, y]
                    if j < 0:
                        break
                    buf[t, i, j, L] += tw[a, x] * f * tw[b, y]
    G = np.zeros((K, K, nlag))
    for i in prange(K):
        for t in range(nthr):
            G[i] += buf[t, i]
    return G


@njit(parallel=True, cache=True)
def _cells(p, q, w, cell, ti, tw, ncell, K, nthr):
    """Per (citing year, lag) cell: summed topic mixes of the cited and the citing ends, and the weight."""
    s, E = ti.shape[1], p.shape[0]
    A = np.zeros((nthr, ncell, K))
    C = np.zeros((nthr, ncell, K))
    n = np.zeros((nthr, ncell))
    chunk = (E + nthr - 1) // nthr
    for t in prange(nthr):
        for e in range(t * chunk, min(E, (t + 1) * chunk)):
            c = cell[e]
            n[t, c] += w[e]
            for x in range(s):
                i = ti[q[e], x]
                if i < 0:
                    break
                A[t, c, i] += w[e] * tw[q[e], x]
            for y in range(s):
                j = ti[p[e], y]
                if j < 0:
                    break
                C[t, c, j] += w[e] * tw[p[e], y]
    As, Cs, ns = np.zeros((ncell, K)), np.zeros((ncell, K)), np.zeros(ncell)
    for c in prange(ncell):
        for t in range(nthr):
            ns[c] += n[t, c]
            for k in range(K):
                As[c, k] += A[t, c, k]
                Cs[c, k] += C[t, c, k]
    return As, Cs, ns


@njit(parallel=True, cache=True)
def _group_sums(rows, wts, grp, ngrp, ti, tw, K, nthr):
    """S[g, k] = sum over rows r with grp = g of wts * theta_r (topic volume by group, e.g. by year)."""
    s, n = ti.shape[1], rows.shape[0]
    buf = np.zeros((nthr, ngrp, K))
    chunk = (n + nthr - 1) // nthr
    for t in prange(nthr):
        for e in range(t * chunk, min(n, (t + 1) * chunk)):
            r, g = rows[e], grp[e]
            if g < 0:
                continue
            for x in range(s):
                i = ti[r, x]
                if i < 0:
                    break
                buf[t, g, i] += wts[e] * tw[r, x]
    S = np.zeros((ngrp, K))
    for g in prange(ngrp):
        for t in range(nthr):
            for k in range(K):
                S[g, k] += buf[t, g, k]
    return S


@njit(cache=True)
def _self_term(rows, wts, ti, tw, K):
    """sum_r wts_r theta_r theta_r' (pairs of a paper with itself in the exposure)."""
    S = np.zeros((K, K))
    s = ti.shape[1]
    for e in range(rows.shape[0]):
        r = rows[e]
        for x in range(s):
            i = ti[r, x]
            if i < 0:
                break
            for y in range(s):
                j = ti[r, y]
                if j < 0:
                    break
                S[i, j] += wts[e] * tw[r, x] * tw[r, y]
    return S


# ---------------------------------------------------------------------------------------------------------------
# Fit of one window


@dataclass
class Window:
    lo: int
    hi: int
    p: np.ndarray        # citing node (row into ti/tw)
    q: np.ndarray        # cited node
    lag: np.ndarray      # t_p - t_q (int64)
    tp: np.ndarray       # citing year


def _exposure(g: np.ndarray, Nyear: np.ndarray, S: np.ndarray, lo: int, min_year: int, self_t: np.ndarray):
    """X = sum_t (sum_lag g(lag) N(t - lag)) S(t)' - g(0) * self term; also Nbar[t] for each citing year."""
    K = Nyear.shape[1]
    X = np.zeros((K, K))
    Nbar = np.zeros((S.shape[0], K))
    for k in range(S.shape[0]):
        t = lo + k
        m = t - min_year + 1                  # years min_year .. t
        ages = np.arange(m)                   # lag 0 .. t - min_year
        Nbar[k] = g[ages] @ Nyear[t - min_year - ages]
        X += np.outer(Nbar[k], S[k])
    return X - g[0] * self_t, Nbar


class Fitter:
    """EM for B and g of one set of citing years, with the bootstrap replicates sharing the kernels."""

    def __init__(self, win: Window, ti: np.ndarray, tw: np.ndarray, Nyear: np.ndarray, min_year: int, K: int,
                 prior: float, max_iter: int, tol: float):
        self.win, self.ti, self.tw, self.Nyear, self.min_year, self.K = win, ti, tw, Nyear, min_year, K
        self.prior, self.max_iter, self.tol = prior, max_iter, tol
        self.nlag = int(win.tp.max() - min_year + 1) if len(win.tp) else 1
        self.buf = np.zeros((get_num_threads(), K, K))
        # citing papers and their reference counts
        self.cite_rows, self._inv, self.r = np.unique(win.p, return_inverse=True, return_counts=True)

    def _setup(self, wp: np.ndarray):
        """Pair weights, S(t) and the self term for citing-paper weights wp (1 or bootstrap)."""
        w = wp[self._inv]
        lo, hi = self.win.lo, self.win.hi
        tp_rows = np.zeros(len(self.cite_rows), np.int64)
        tp_rows[self._inv] = self.win.tp
        S = _group_sums(self.cite_rows, (wp * self.r).astype(np.float64), tp_rows - lo, hi - lo + 1,
                        self.ti, self.tw, self.K, get_num_threads())
        self_t = _self_term(self.cite_rows, (wp * self.r).astype(np.float64), self.ti, self.tw, self.K)
        nlag_w = np.bincount(self.win.lag, weights=w, minlength=self.nlag)[:self.nlag]
        return w, S, self_t, nlag_w

    def run(self, wp: np.ndarray | None = None, B0: np.ndarray | None = None, g0: np.ndarray | None = None,
            max_iter: int | None = None) -> dict:
        win = self.win
        wp = np.ones(len(self.cite_rows)) if wp is None else wp
        w, S, self_t, nlag_w = self._setup(wp)
        total = w.sum()
        K, lo = self.K, win.lo
        g = (nlag_w + 1e-9) / (nlag_w + 1e-9).sum() if g0 is None else g0.copy()
        B = np.ones((K, K)) if B0 is None else B0.copy()
        ll_prev, it = -np.inf, 0
        for it in range(max_iter or self.max_iter):
            X, Nbar = _exposure(g, self.Nyear, S, lo, self.min_year, self_t)
            G, lld = _estep(win.p, win.q, w, self.ti, self.tw, B, self.buf)
            ll = lld + (nlag_w * np.log(np.maximum(g, 1e-300))).sum() - (B * X).sum()
            if abs(ll - ll_prev) <= self.tol * abs(ll):
                break
            ll_prev = ll
            F = B * G
            mu = total / max(X.sum(), 1e-300)
            B = (F + self.prior) / (np.maximum(X, 0) + self.prior / mu)
            # aging: exact maximisation given B, then rescaled to sum 1 (B takes the scale)
            v = S @ B.T                         # v[k] = B S(t_k), over source topics
            den = np.zeros(self.nlag)
            for k in range(S.shape[0]):
                t = lo + k
                ages = np.arange(t - self.min_year + 1)
                den[ages] += self.Nyear[t - self.min_year - ages] @ v[k]
            graw = np.where(den > 0, nlag_w / np.maximum(den, 1e-300), 0.0)
            c = graw.sum()
            if c > 0:
                g, B = graw / c, B * c
        X, Nbar = _exposure(g, self.Nyear, S, lo, self.min_year, self_t)
        G, lld = _estep(win.p, win.q, w, self.ti, self.tw, B, self.buf)
        return {"B": B, "g": g, "F": B * G, "X": X, "S": S, "Nbar": Nbar, "w": w, "wp": wp, "total": total,
                "iterations": it + 1, "loglik": float(lld + (nlag_w * np.log(np.maximum(g, 1e-300))).sum()
                                                       - (B * X).sum())}

    def null(self, w: np.ndarray) -> np.ndarray:
        """Expected product flow under the time-stratified configuration null (cited ends shuffled per cell)."""
        win = self.win
        cell = (win.tp - win.lo) * self.nlag + win.lag
        A, C, n = _cells(win.p, win.q, w, cell, self.ti, self.tw, (win.hi - win.lo + 1) * self.nlag, self.K,
                         get_num_threads())
        keep = n > 0
        return A[keep].T @ (C[keep] / n[keep, None])

    def product(self, w: np.ndarray) -> np.ndarray:
        G, _ = _estep(self.win.p, self.win.q, w, self.ti, self.tw, np.ones((self.K, self.K)), self.buf)
        return G

    def fractional(self, B: np.ndarray) -> np.ndarray:
        """Deconvolved flow with every citing paper's references weighted 1/r (its knowledge base sums to 1)."""
        G, _ = _estep(self.win.p, self.win.q, (1.0 / self.r)[self._inv], self.ti, self.tw, B, self.buf)
        return B * G

    def lags(self, fit: dict) -> np.ndarray:
        """Median age (citing year - cited year) of the citations from topic i to topic j, from the deconvolved
        flow by lag; NaN with fewer than 20 citations. Growing literatures make recent work look more cited."""
        win = self.win
        Gl = _estep_lags(win.p, win.q, fit["w"], win.lag, self.ti, self.tw, fit["B"], self.nlag, get_num_threads())
        Fl = fit["B"][:, :, None] * Gl
        cum = np.cumsum(Fl, axis=2)
        tot = cum[:, :, -1:]
        med = np.argmax(cum >= 0.5 * tot, axis=2).astype(float)
        med[tot[:, :, 0] < 20] = np.nan
        return med


# ---------------------------------------------------------------------------------------------------------------
# Hodge decomposition


def hodge(B: np.ndarray, F: np.ndarray, min_flow: float, volume: np.ndarray | None = None,
          split: bool = True) -> dict:
    """Upstream potential phi (phi_i - phi_j ~ log B_ij / B_ji) and the shares of gradient, curl and harmonic
    flow in the w-weighted norm, w = F_ij + F_ji, on the pairs with w >= min_flow."""
    K = B.shape[0]
    W = F + F.T
    iu, ju = np.triu_indices(K, 1)
    w = W[iu, ju]
    keep = (w >= min_flow) & (B[iu, ju] > 0) & (B[ju, iu] > 0)
    i, j, w = iu[keep], ju[keep], w[keep]
    phi = np.full(K, np.nan)
    if not len(w):
        return {"phi": phi, "gradient": None, "curl": None, "harmonic": None, "edges": 0, "cycles": 0}
    Y = np.log(B[i, j]) - np.log(B[j, i])
    L = np.zeros((K, K))
    np.add.at(L, (i, i), w)
    np.add.at(L, (j, j), w)
    np.add.at(L, (i, j), -w)
    np.add.at(L, (j, i), -w)
    div = np.zeros(K)
    np.add.at(div, i, w * Y)
    np.add.at(div, j, -w * Y)
    used = np.zeros(K, bool)
    used[i] = used[j] = True
    sol = np.linalg.lstsq(L[np.ix_(used, used)], div[used], rcond=None)[0]
    vol = np.ones(K) if volume is None else volume
    phi[used] = sol - np.average(sol, weights=np.maximum(vol[used], 1e-12))
    grad = phi[i] - phi[j]
    total = float((w * Y * Y).sum())
    # independent cycles of the graph; without any, every edge flow is a gradient and the hierarchy is trivially 1
    n_comp = sp.csgraph.connected_components(sp.coo_matrix((w, (i, j)), shape=(K, K)), directed=False)[1]
    cycles = int(len(w) - used.sum() + len(np.unique(n_comp[used])))
    base = {"phi": phi, "gradient": None, "curl": None, "harmonic": None, "edges": int(len(w)), "cycles": cycles}
    if total <= 0 or cycles == 0:
        return base
    out = {**base, "gradient": float((w * grad ** 2).sum() / total)}
    if split:
        R = Y - grad
        eid = {(a, b): e for e, (a, b) in enumerate(zip(i.tolist(), j.tolist()))}
        nbr = [set() for _ in range(K)]
        for a, b in eid:
            nbr[a].add(b)
        rows, cols, vals, nt = [], [], [], 0
        for a in range(K):
            for b in nbr[a]:
                for c in nbr[a] & nbr[b]:
                    if c > b:  # triangle a < b < c: edges (a,b), (b,c), (a,c) with sign -1
                        rows += [nt, nt, nt]
                        cols += [eid[(a, b)], eid[(b, c)], eid[(a, c)]]
                        vals += [1.0, 1.0, -1.0]
                        nt += 1
        if nt:
            Cm = sp.csr_matrix((vals, (rows, cols)), shape=(nt, len(w)))
            A = sp.diags(1 / np.sqrt(w)) @ Cm.T
            psi = lsqr(A, np.sqrt(w) * R, atol=1e-10, btol=1e-10, iter_lim=5000)[0]
            curl = (Cm.T @ psi) / w
            out["curl"] = float((w * curl ** 2).sum() / total)
        else:
            out["curl"] = 0.0
        out["harmonic"] = max(0.0, 1.0 - out["gradient"] - out["curl"])
    return out


# ---------------------------------------------------------------------------------------------------------------
# Step fit: per corpus and resolution


def windows_of(lo: int, hi: int, size: int) -> list[tuple[int, int]]:
    """Consecutive windows of `size` citing years, the last one ending at hi; the first may be shorter."""
    out, end = [], hi
    while end >= lo:
        out.append((max(lo, end - size + 1), end))
        end -= size
    return out[::-1]


def node_volumes(year: np.ndarray, ti: np.ndarray, tw: np.ndarray, K: int, min_year: int, max_year: int):
    rows = np.arange(len(year), dtype=np.int64)
    grp = np.where((year >= min_year) & (year <= max_year), year.astype(np.int64) - min_year, -1)
    return _group_sums(rows, np.ones(len(year)), grp, max_year - min_year + 1, ti, tw, K, get_num_threads())


def fit_all(p, q, year, ti, tw, K, wins, min_year, ccfg, bootstrap: int, lags: bool, seed: int,
            split: bool = True) -> list[dict]:
    """Fits of all windows; p, q are node rows into year/ti/tw."""
    max_year = int(year.max())
    Nyear = node_volumes(year, ti, tw, K, min_year, max_year)
    tp, tq = year[p].astype(np.int64), year[q].astype(np.int64)
    rng = np.random.default_rng(seed)
    out = []
    for lo, hi in wins:
        sel = (tp >= lo) & (tp <= hi)
        if sel.sum() < ccfg.min_citations:
            out.append({"lo": lo, "hi": hi, "citations": int(sel.sum()), "fitted": False})
            continue
        win = Window(lo, hi, p[sel].astype(np.int64), q[sel].astype(np.int64), (tp - tq)[sel], tp[sel])
        fitter = Fitter(win, ti, tw, Nyear, min_year, K, ccfg.prior, ccfg.max_iter, ccfg.tol)
        t0 = time.time()
        fit = fitter.run()
        E0 = fitter.null(fit["w"])
        F, B = fit["F"], fit["B"]
        vol = Nyear.sum(axis=0)
        hd = hodge(B, F, ccfg.hodge_min_flow, vol, split=split)
        prod = fitter.product(fit["w"])
        res = {"lo": lo, "hi": hi, "fitted": True, "citations": float(fit["total"]),
               "citing_papers": int(len(fitter.cite_rows)), "iterations": fit["iterations"],
               "loglik": fit["loglik"], "B": B, "F": F, "X": fit["X"], "g": fit["g"],
               "mean_rate": float(fit["total"] / max(fit["X"].sum(), 1e-300)),
               "lift": np.divide(F, E0, out=np.full_like(F, np.nan), where=E0 > 0),
               "fractional": fitter.fractional(B),
               "diag_product": float(np.trace(prod) / max(prod.sum(), 1e-300)),
               "diag_model": float(np.trace(F) / max(F.sum(), 1e-300)),
               "phi": hd["phi"], "hierarchy": {k: hd[k] for k in ("gradient", "curl", "harmonic", "edges", "cycles")}}
        if lags:
            res["lag"] = fitter.lags(fit)
        if bootstrap:
            phis, grads, lifts = [], [], []
            for _ in range(bootstrap):
                wp = rng.poisson(1.0, len(fitter.cite_rows)).astype(np.float64)
                fb = fitter.run(wp, B, fit["g"], max_iter=ccfg.bootstrap_iter)
                hb = hodge(fb["B"], fb["F"], ccfg.hodge_min_flow, vol, split=False)
                e0 = fitter.null(fb["w"])
                phis.append(hb["phi"])
                grads.append(hb["gradient"] if hb["gradient"] is not None else np.nan)
                lifts.append(np.divide(fb["F"], e0, out=np.full_like(F, np.nan), where=e0 > 0))
            phis, lifts = np.array(phis), np.array(lifts)
            res["phi_se"] = np.nanstd(phis, axis=0)
            res["hierarchy"]["gradient_se"] = float(np.nanstd(grads))
            res["lift_lo"] = np.nanpercentile(lifts, 5, axis=0)
            res["lift_hi"] = np.nanpercentile(lifts, 95, axis=0)
        log.info("  %d-%d: %d citations, %d iterations, hierarchy %.2f (%.0f s)", lo, hi, int(fit["total"]),
                 fit["iterations"], hd["gradient"] or float("nan"), time.time() - t0)
        out.append(res)
    return out


def coverage(c: Corpus, pairs: pl.DataFrame, nodes_kind: dict[int, np.ndarray]) -> dict:
    refs = pl.read_parquet(c.dir() / "refs.parquet")["refs"].sum()
    own = pairs.filter(pl.col("cited_corpus") == c.index)
    kinds = own.select(pl.col("cited")).to_series().to_numpy()
    title_own = int((nodes_kind[c.index][kinds] == TITLE).sum()) if len(kinds) else 0
    ok = pairs.filter(pl.col("flags") == 0)
    return {
        "references": int(refs),
        "resolved": int(pairs.height),
        "same_corpus": int(own.height),
        "same_corpus_title_only": title_own,
        "other_corpora": int(pairs.height - own.height),
        "self_citations": int(pairs.filter((pl.col("flags") & SELF) > 0).height),
        "cited_later": int(pairs.filter((pl.col("flags") & FUTURE) > 0).height),
        "used": int(ok.filter(pl.col("cited_corpus") == c.index).height),
    }


def step_fit(corpora: list[Corpus], force: bool) -> dict:
    kinds = {c.index: pl.read_parquet(c.dir() / "nodes.parquet", columns=["kind"])["kind"].to_numpy()
             for c in corpora}
    out = {}
    for c in corpora:
        ccfg = c.cfg.citations
        pairs = pl.read_parquet(c.dir() / "pairs.parquet")
        cov = coverage(c, pairs, kinds)
        nodes = pl.read_parquet(c.dir() / "nodes.parquet", columns=["year", "kind"])
        year = nodes["year"].to_numpy()
        own = pairs.filter((pl.col("cited_corpus") == c.index) & (pl.col("flags") == 0)) if \
            ccfg.self_citations == "exclude" else \
            pairs.filter((pl.col("cited_corpus") == c.index) & ((pl.col("flags") & (FUTURE | SAME)) == 0))
        p, q = own["citing"].to_numpy(), own["cited"].to_numpy()
        papers = nodes["kind"].to_numpy() == PAPER
        lo, hi = int(year[papers].min()), int(year[papers].max())
        wins = windows_of(lo, hi, ccfg.window)
        for res in c.resolutions():
            target = c.dir(res) / "fit.npz"
            if not force and target.exists():
                continue
            t0 = time.time()
            th = np.load(c.dir(res) / "theta.npz")
            K = c.k_of(res)
            log.info("fit %s %s: K=%d, %d citations in the corpus", c.name, res, K, len(p))
            fits = fit_all(p, q, year, th["idx"], th["w"], K, wins, lo, ccfg,
                           bootstrap=ccfg.bootstrap if res == "main" else 0, lags=res == "main",
                           seed=c.cfg.seed)
            save_fits(target, fits, {"corpus": c.name, "resolution": res, "k": K, "window": ccfg.window,
                                     "coverage": cov, "min_year": lo, "max_year": hi})
            write_json(c.dir(res) / "bundle.json", corpus_bundle(fits, K, ccfg, cov, res))
            out[f"{c.name}/{res}"] = round(time.time() - t0)
    return out


ARRAYS = ("B", "F", "X", "g", "lift", "fractional", "phi", "lag", "phi_se", "lift_lo", "lift_hi")


def save_fits(path: Path, fits: list[dict], meta: dict) -> None:
    arrays, rows = {}, []
    for n, f in enumerate(fits):
        rows.append({k: v for k, v in f.items() if k not in ARRAYS})
        for k in ARRAYS:
            if k in f:
                arrays[f"{k}_{n}"] = np.asarray(f[k], np.float32 if k != "g" else np.float64)
    np.savez_compressed(path, **arrays)
    write_json(path.with_suffix(".json"), {**meta, "windows": rows})


# ---------------------------------------------------------------------------------------------------------------
# Bundle files


def _r(x, nd=4):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), nd)


def _top(row: np.ndarray, exclude: int, n: int = 5) -> list[list]:
    r = row.astype(float).copy()
    r[exclude] = 0
    tot = row.sum()
    if tot <= 0:
        return []
    order = np.argsort(-r)[:n]
    return [[int(k), _r(row[k] / tot)] for k in order if r[k] > 0]


def corpus_bundle(fits: list[dict], K: int, ccfg, cov: dict, res: str) -> dict:
    wins = []
    for f in fits:
        if not f.get("fitted"):
            wins.append({"start": f["lo"], "end": f["hi"], "citations": f["citations"], "fitted": False})
            continue
        F, B, L, Fr = f["F"], f["B"], f["lift"], f["fractional"]
        mu = f["mean_rate"]
        off = F.copy()
        np.fill_diagonal(off, 0)
        order = np.argsort(-off, axis=None)[:ccfg.top_flows]
        flows = []
        for o in order:
            i, j = divmod(int(o), K)
            if off[i, j] <= 0:
                break
            row = [i, j, _r(F[i, j], 1), _r(B[i, j] / mu, 3), _r(L[i, j], 3)]
            if "lift_lo" in f:
                row += [_r(f["lift_lo"][i, j], 3), _r(f["lift_hi"][i, j], 3)]
            if "lag" in f:
                row.append(None if not math.isfinite(f["lag"][i, j]) else int(f["lag"][i, j]))
            flows.append(row)
        h = f["hierarchy"]
        wins.append({
            "start": f["lo"], "end": f["hi"], "fitted": True, "citations": round(f["citations"]),
            "citing_papers": f["citing_papers"],
            "diagonal_share": {"product": _r(f["diag_product"], 3), "model": _r(f["diag_model"], 3)},
            "hierarchy": {**{k: _r(h.get(k), 3) for k in ("gradient", "curl", "harmonic", "gradient_se")},
                          "edges": h.get("edges"), "cycles": h.get("cycles")},
            "aging": [_r(x, 5) for x in f["g"][:41]],
            "topics": {
                "potential": [_r(x, 3) for x in f["phi"]],
                **({"potential_se": [_r(x, 3) for x in f["phi_se"]]} if "phi_se" in f else {}),
                "cited": [_r(x, 1) for x in F.sum(axis=1)],
                "citing": [_r(x, 1) for x in F.sum(axis=0)],
                "self": [_r(x, 1) for x in np.diag(F)],
                "draws_on": [_top(Fr[:, j], j) for j in range(K)],
                "used_by": [_top(F[i, :], i) for i in range(K)],
            },
            "flows": flows,
        })
    cols = ["from", "to", "citations", "rate", "lift"]
    if res == "main" and ccfg.bootstrap:
        cols += ["lift_lo", "lift_hi"]
    if res == "main":
        cols.append("lag")
    return {"resolution": res, "k": K, "window": ccfg.window, "unit": "citations",
            "self_citations": ccfg.self_citations, "title_only": ccfg.title_only, "top_topics": ccfg.top_topics,
            "flow_columns": cols, "coverage": cov, "windows": wins}


# ---------------------------------------------------------------------------------------------------------------
# Step global: all main topics of all corpora


def global_theta(corpora: list[Corpus]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Node topic mixes on the global topic index (corpus topic + topic offset of the corpus)."""
    toff = np.concatenate([[0], np.cumsum([c.k for c in corpora])]).astype(np.int64)
    idx, w = [], []
    for c in corpora:
        th = np.load(c.dir() / "theta.npz")
        i = th["idx"].astype(np.int16)
        idx.append(np.where(i >= 0, i + toff[c.index], -1).astype(np.int16))
        w.append(th["w"])
    return np.vstack(idx), np.vstack(w), toff


def global_pairs(corpora: list[Corpus], offsets: np.ndarray, exclude_self: bool) -> tuple[np.ndarray, np.ndarray]:
    P, Q = [], []
    for c in corpora:
        mask = SELF | FUTURE | SAME if exclude_self else FUTURE | SAME
        d = pl.read_parquet(c.dir() / "pairs.parquet").filter((pl.col("flags") & mask) == 0)
        P.append(d["citing"].to_numpy().astype(np.int64) + offsets[c.index])
        Q.append(d["cited"].to_numpy().astype(np.int64) + offsets[d["cited_corpus"].to_numpy()])
    return np.concatenate(P), np.concatenate(Q)


def step_global(corpora: list[Corpus], gdir: Path, window: int, force: bool) -> dict:
    ccfg = corpora[0].cfg.citations
    toff = np.concatenate([[0], np.cumsum([c.k for c in corpora])]).astype(np.int64)
    meta = {"k": int(toff[-1]), "window": window,
            "corpora": [{"name": c.name, "k": c.k, "offset": int(toff[c.index])} for c in corpora]}
    saved = gdir / "fit.json"
    if not force and saved.exists() and _same_global(json.loads(saved.read_text()), meta):
        fits, out = load_fits(gdir / "fit.npz"), {"refit": False}
        log.info("global fit: saved fit reused, global.json rewritten")
    else:
        nodes = load_nodes(corpora)
        ti, tw, _ = global_theta(corpora)
        p, q = global_pairs(corpora, nodes.offsets, ccfg.self_citations == "exclude")
        papers = nodes.kind == PAPER
        lo, hi = int(nodes.year[papers].min()), int(nodes.year[papers].max())
        wins = windows_of(lo, hi, window)
        log.info("global fit: K=%d, %d citations, %d windows", meta["k"], len(p), len(wins))
        fits = fit_all(p, q, nodes.year, ti, tw, meta["k"], wins, lo, ccfg, bootstrap=0, lags=False, seed=0,
                       split=False)
        save_fits(gdir / "fit.npz", fits, {**meta, "min_year": lo, "max_year": hi})
        out = {"refit": True, "k": meta["k"], "citations": int(len(p))}
    sim = topic_similarity(corpora)
    write_json(gdir / "global.json", global_bundle(fits, corpora, toff, ccfg, window, sim))
    return out


def _same_global(saved: dict, meta: dict) -> bool:
    return saved.get("window") == meta["window"] and saved.get("corpora") == meta["corpora"]


def load_fits(path: Path) -> list[dict]:
    """The fits written by save_fits: the per-window rows of the .json with their arrays from the .npz."""
    rows = json.loads(path.with_suffix(".json").read_text())["windows"]
    with np.load(path) as z:
        for n, row in enumerate(rows):
            for k in ARRAYS:
                if f"{k}_{n}" in z:
                    row[k] = z[f"{k}_{n}"].astype(np.float64)
    return rows


def topic_similarity(corpora: list[Corpus]) -> np.ndarray:
    """Cosine similarity of the topic-word weights of all main topics (global index), over the union of the
    corpora's vocabularies (terms matched by their string)."""
    terms: dict[str, int] = {}
    rows, cols, vals = [], [], []
    start = 0
    for c in corpora:
        voc = json.loads((c.main.model_dir / "vocabulary.json").read_text())
        comp = np.load(c.main.model_dir / "components.npy")
        ids = np.fromiter((terms.setdefault(t, len(terms)) for t in voc), np.int64, len(voc))
        r, k = np.nonzero(comp)
        rows.append(r + start)
        cols.append(ids[k])
        vals.append(comp[r, k].astype(np.float64))
        start += comp.shape[0]
    X = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(start, len(terms)))
    norm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    X = sp.diags(1.0 / np.maximum(norm, 1e-300)) @ X
    return (X @ X.T).toarray()


def twins(sim: np.ndarray, toff: np.ndarray, threshold: float) -> np.ndarray:
    """twin[i, j]: topics of different corpora that are each other's most similar topic in the other corpus,
    with a similarity of at least `threshold` (one subject in two maps rather than a flow between subjects)."""
    nf = len(toff) - 1
    blocks = np.searchsorted(toff, np.arange(toff[-1]), side="right") - 1
    best = np.stack([sim[:, toff[b]:toff[b + 1]].argmax(axis=1) + toff[b] for b in range(nf)], axis=1)
    idx = np.arange(toff[-1])
    mutual = (best[:, blocks] == idx[None, :]) & (best[:, blocks].T == idx[:, None])
    return mutual & (blocks[:, None] != blocks[None, :]) & (sim >= threshold)


def global_bundle(fits, corpora, toff, ccfg, window, sim: np.ndarray | None = None) -> dict:
    nf = len(corpora)
    blocks = np.searchsorted(toff, np.arange(toff[-1]), side="right") - 1   # topic -> corpus
    twin = twins(sim, toff, ccfg.twin_similarity) if sim is not None else None
    wins = []
    for f in fits:
        if not f.get("fitted"):
            wins.append({"start": f["lo"], "end": f["hi"], "fitted": False, "citations": f["citations"]})
            continue
        F, L = f["F"], f["lift"]
        E0 = np.divide(F, L, out=np.zeros_like(F), where=np.isfinite(L) & (L > 0))
        FF = np.zeros((nf, nf))
        EE = np.zeros((nf, nf))
        np.add.at(FF, (blocks[:, None], blocks[None, :]), F)
        np.add.at(EE, (blocks[:, None], blocks[None, :]), E0)
        field_flows = [[a, b, _r(FF[a, b], 1), _r(FF[a, b] / EE[a, b], 3) if EE[a, b] > 0 else None]
                       for a in range(nf) for b in range(nf) if FF[a, b] > 0]
        # field potentials: rates of the aggregated blocks (flow over exposure, both summed per block)
        XX = np.zeros((nf, nf))
        np.add.at(XX, (blocks[:, None], blocks[None, :]), f["X"])
        BB = np.divide(FF, XX, out=np.zeros_like(FF), where=XX > 0)
        hf = hodge(BB, FF, ccfg.hodge_min_flow, split=True)
        cross = F.copy()
        cross[blocks[:, None] == blocks[None, :]] = 0
        order = np.argsort(-cross, axis=None)[:300]
        channels = []
        for o in order:
            i, j = divmod(int(o), F.shape[0])
            if cross[i, j] <= 0:
                break
            channels.append([int(blocks[i]), int(i - toff[blocks[i]]), int(blocks[j]), int(j - toff[blocks[j]]),
                             _r(F[i, j], 1), _r(L[i, j], 3)]
                            + ([_r(sim[i, j], 3), bool(twin[i, j])] if sim is not None else []))
        wins.append({
            "start": f["lo"], "end": f["hi"], "fitted": True, "citations": round(f["citations"]),
            "citing_papers": f["citing_papers"],
            "diagonal_share": {"product": _r(f["diag_product"], 3), "model": _r(f["diag_model"], 3)},
            "hierarchy": {"gradient": _r(f["hierarchy"]["gradient"], 3), "edges": f["hierarchy"]["edges"],
                          "cycles": f["hierarchy"]["cycles"]},
            "field_hierarchy": {**{k: _r(hf[k], 3) for k in ("gradient", "curl", "harmonic")},
                                "edges": hf["edges"], "cycles": hf["cycles"]},
            "field_potential": [_r(x, 3) for x in hf["phi"]],
            "field_flows": field_flows,
            "potential": [[_r(x, 3) for x in f["phi"][toff[c]:toff[c + 1]]] for c in range(nf)],
            "channels": channels,
        })
    return {"window": window, "unit": "citations", "self_citations": ccfg.self_citations,
            "title_only": ccfg.title_only, "top_topics": ccfg.top_topics,
            "fields": [{"name": c.name, "label": c.cfg.export.label, "group": c.cfg.export.group, "k": c.k}
                       for c in corpora],
            "field_flow_columns": ["from", "to", "citations", "lift"],
            "channel_columns": ["from_field", "from_topic", "to_field", "to_topic", "citations", "lift"]
            + (["similarity", "twin"] if sim is not None else []),
            **({"twin_similarity": ccfg.twin_similarity} if sim is not None else {}),
            "windows": wins}


# ---------------------------------------------------------------------------------------------------------------
# Step papers: knowledge-base surprise


@njit(parallel=True, fastmath=True, cache=True)
def _surprise(starts, p_of, q, ti, tw, wid, Bs, M, U, Zbar, year0, out):
    """Mean over a paper's references of -log [P(q | p) / P(q | average citing paper of the year)]."""
    s = ti.shape[1]
    for g in prange(starts.shape[0] - 1):
        a, b = starts[g], starts[g + 1]
        p = p_of[g]
        w = wid[g]
        if w < 0:
            out[g] = np.nan
            continue
        B = Bs[w]
        t = year0[g]
        zp = 0.0
        for y in range(s):
            j = ti[p, y]
            if j < 0:
                break
            zp += tw[p, y] * U[t, j]
        acc = 0.0
        n = 0
        for e in range(a, b):
            qq = q[e]
            d = 0.0
            m = 0.0
            for x in range(s):
                i = ti[qq, x]
                if i < 0:
                    break
                m += tw[qq, x] * M[t, i]
                for y in range(s):
                    j = ti[p, y]
                    if j < 0:
                        break
                    d += tw[qq, x] * B[i, j] * tw[p, y]
            if d > 0 and m > 0 and zp > 0:
                acc += -(math.log(d / zp) - math.log(m / Zbar[t]))
                n += 1
        out[g] = acc / n if n else np.nan


def step_papers(corpora: list[Corpus], gdir: Path, force: bool) -> dict:
    if not force and all((c.dir() / "papers.parquet").exists() for c in corpora):
        return {}
    ccfg = corpora[0].cfg.citations
    meta = json.loads((gdir / "fit.json").read_text())
    z = np.load(gdir / "fit.npz")
    nodes = load_nodes(corpora)
    ti, tw, toff = global_theta(corpora)
    K = int(toff[-1])
    lo, hi = meta["min_year"], meta["max_year"]
    Nyear = node_volumes(nodes.year, ti, tw, K, lo, hi)
    fitted = [(n, w) for n, w in enumerate(meta["windows"]) if w.get("fitted")]
    Bs = np.stack([z[f"B_{n}"].astype(np.float64) for n, _ in fitted])
    wmap = np.full(hi - lo + 1, -1, np.int64)
    M = np.zeros((hi - lo + 1, K))
    U = np.zeros((hi - lo + 1, K))
    Zbar = np.zeros(hi - lo + 1)
    p_all, q_all = global_pairs(corpora, nodes.offsets, ccfg.self_citations == "exclude")
    order = np.argsort(p_all, kind="stable")
    p_all, q_all = p_all[order], q_all[order]
    # average citing mix per year (weighted by references) and the exposure-weighted volumes
    tp = nodes.year[p_all].astype(np.int64)
    S = _group_sums(p_all, np.ones(len(p_all)), tp - lo, hi - lo + 1, ti, tw, K, get_num_threads())
    for b, (n, w) in enumerate(fitted):
        g = z[f"g_{n}"]
        B = Bs[b]
        for t in range(w["lo"], w["hi"] + 1):
            k = t - lo
            wmap[k] = b
            ages = np.arange(t - lo + 1)
            v = g[ages] @ Nyear[t - lo - ages]
            thbar = S[k] / max(S[k].sum(), 1e-300)
            M[k] = B @ thbar           # M[i]: rate of an average citing paper of year t for source topic i
            U[k] = B.T @ v             # Z_p = theta_p' U
            Zbar[k] = thbar @ U[k]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(p_all)) + 1, [len(p_all)]]).astype(np.int64)
    p_of = p_all[starts[:-1]]
    counts = np.diff(starts)
    out = np.full(len(p_of), np.nan)
    year0 = nodes.year[p_of].astype(np.int64) - lo
    _surprise(starts, p_of, q_all, ti, tw, wmap[year0], Bs, M, U, Zbar, year0, out)
    out[counts < ccfg.min_refs_surprise] = np.nan
    cp = nodes.corpus_of(p_of)
    res = {}
    for c in corpora:
        sel = cp == c.index
        rows = p_of[sel] - nodes.offsets[c.index]
        nd = pl.read_parquet(c.dir() / "nodes.parquet", columns=["doc_id", "year"]).with_row_index("row")
        refs = pl.read_parquet(c.dir() / "refs.parquet")
        df = (refs.join(pl.DataFrame({"row": rows.astype(np.uint32), "references_resolved": counts[sel],
                                      "kb_surprise": out[sel]}).with_columns(pl.col("row").cast(pl.Int32)),
                        on="row", how="left")
              .join(nd.with_columns(pl.col("row").cast(pl.Int32)), on="row")
              .with_columns(pl.col("references_resolved").fill_null(0),
                            pl.col("kb_surprise").fill_nan(None))
              .with_columns(kb_surprise_pct=(pl.col("kb_surprise").rank("average").over("year") * 100
                                             / pl.col("kb_surprise").count().over("year")))
              .select("doc_id", pl.col("refs").alias("references"), "references_resolved",
                      pl.col("kb_surprise").cast(pl.Float32), pl.col("kb_surprise_pct").cast(pl.Float32)))
        df.write_parquet(c.dir() / "papers.parquet")
        res[c.name] = int(df["kb_surprise"].is_not_null().sum())
    return res


# ---------------------------------------------------------------------------------------------------------------


def run(configs: list[str | Path], steps: list[str] | None = None, n_jobs: int = 200, force: bool = False,
        global_window: int = 5) -> dict:
    steps = steps or list(STEPS)
    corpora = load_corpora(configs)
    if not corpora:
        raise ValueError("no corpus with topics and OpenAlex input")
    links = links_store(corpora)
    gdir = global_dir(corpora)
    out = {}
    for step in STEPS:
        if step not in steps:
            continue
        t0 = time.time()
        log.info("step %s", step)
        if step == "titles":
            out[step] = step_titles(corpora, links, n_jobs, force)
        elif step == "nodes":
            out[step] = step_nodes(corpora, force)
        elif step == "authors":
            out[step] = step_authors(corpora, links, gdir, force)
        elif step == "resolve":
            out[step] = step_resolve(corpora, links, gdir, force)
        elif step == "fit":
            out[step] = step_fit(corpora, force)
        elif step == "global":
            out[step] = step_global(corpora, gdir, global_window, force)
        elif step == "papers":
            out[step] = step_papers(corpora, gdir, force)
        log.info("step %s done (%.0f s)", step, time.time() - t0)
    write_json(gdir / "run.json", {"steps": steps, "result": out,
                                   "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return out
