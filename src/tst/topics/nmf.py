"""NMF training and model selection.

For every k of the grid we fit a reference model on the training sample and `runs` models on random
subsamples. Each k is scored by
  * NPMI coherence of the top terms (document co-occurrence; Bouma 2009, Lau et al. 2014),
  * stability: agreement of the subsample models with the reference model measured by Hungarian-matched
    average Jaccard of the ranked top terms (Greene, O'Callaghan & Cunningham 2014),
  * diversity: share of unique terms among all topics' top 25 terms (Dieng et al. 2020).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF, MiniBatchNMF
from threadpoolctl import threadpool_limits

from ..config import TopicsConfig

log = logging.getLogger("tst.nmf")


def fit_components(X: sp.csr_matrix, k: int, cfg: TopicsConfig, seed: int) -> np.ndarray:
    alpha_H = cfg.alpha_W if cfg.alpha_H == "same" else cfg.alpha_H
    common = dict(n_components=k, init="nndsvda", beta_loss=cfg.beta_loss, max_iter=cfg.max_iter,
                  tol=cfg.tol, random_state=seed, alpha_W=cfg.alpha_W, alpha_H=alpha_H, l1_ratio=cfg.l1_ratio)
    if cfg.algorithm == "minibatch":
        model = MiniBatchNMF(batch_size=4096, **common)
    else:
        model = NMF(solver="cd" if cfg.beta_loss == "frobenius" else "mu", **common)
    model.fit(X)
    return model.components_.astype(np.float32)


def top_ids(H: np.ndarray, n: int) -> np.ndarray:
    return np.argsort(-H, axis=1)[:, :n]


def npmi_coherence(B: sp.csc_matrix, tops: np.ndarray) -> np.ndarray:
    """Mean pairwise NPMI of each topic's top terms on the binary doc-term matrix B (n_docs x V)."""
    N = B.shape[0]
    scores = []
    for ids in tops:
        sub = B[:, ids].astype(np.float64)
        co = (sub.T @ sub).toarray()
        p = np.diag(co) / N
        vals = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pij = co[i, j] / N
                if pij == 0 or p[i] == 0 or p[j] == 0:
                    vals.append(-1.0)
                elif pij == 1:
                    vals.append(1.0)
                else:
                    vals.append(np.log(pij / (p[i] * p[j])) / -np.log(pij))
        scores.append(np.mean(vals))
    return np.array(scores)


def average_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    total = 0.0
    for d in range(1, len(a) + 1):
        sa, sb = set(a[:d]), set(b[:d])
        total += len(sa & sb) / len(sa | sb)
    return total / len(a)


def ranking_agreement(ref: np.ndarray, other: np.ndarray) -> float:
    sim = np.array([[average_jaccard(r, o) for o in other] for r in ref])
    rows, cols = linear_sum_assignment(-sim)
    return float(sim[rows, cols].mean())


def diversity(H: np.ndarray, n: int = 25) -> float:
    tops = top_ids(H, n)
    return len(np.unique(tops)) / tops.size


# Sweep workers share X through fork.
_X: sp.csr_matrix | None = None
_CFG: TopicsConfig | None = None


def _task(args) -> tuple[int, int, np.ndarray]:
    k, run, seed, threads = args
    with threadpool_limits(threads):
        X = _X
        if run > 0:
            rng = np.random.default_rng(seed)
            rows = rng.choice(X.shape[0], size=int(_CFG.subsample * X.shape[0]), replace=False)
            X = X[np.sort(rows)]
        return k, run, fit_components(X, k, _CFG, seed)


def sweep(X_train: sp.csr_matrix, B_ref: sp.csc_matrix, cfg: TopicsConfig, seed: int, n_jobs: int,
          ks: list[int], runs: int) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Fit all (k, run) models; return the metrics table and the reference components per k."""
    global _X, _CFG
    _X, _CFG = X_train, cfg
    tasks = [(k, r, seed + r, 1) for k in ks for r in range(runs + 1)]
    n_proc = max(1, min(n_jobs, len(tasks)))
    threads = max(1, n_jobs // n_proc)
    tasks = [(k, r, s, threads) for k, r, s, _ in tasks]
    log.info("sweep: %d fits (k=%s, %d runs each) on %d docs, %d processes x %d threads",
             len(tasks), ks, runs, X_train.shape[0], n_proc, threads)
    results: dict[int, dict[int, np.ndarray]] = {k: {} for k in ks}
    # Fit large k first so the slowest jobs do not end up last.
    tasks.sort(key=lambda t: -t[0])
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    with mp.get_context("fork").Pool(n_proc) as pool:
        for k, r, H in pool.imap_unordered(_task, tasks):
            results[k][r] = H
            log.info("  fitted k=%d run=%d", k, r)

    rows, refs = [], {}
    for k in ks:
        ref = results[k][0]
        refs[k] = ref
        tops = top_ids(ref, cfg.top_n)
        npmi = npmi_coherence(B_ref, tops)
        stab = [ranking_agreement(tops, top_ids(results[k][r], cfg.top_n)) for r in range(1, runs + 1)]
        rows.append({
            "k": k,
            "npmi": float(npmi.mean()),
            "npmi_min": float(npmi.min()),
            "stability": float(np.mean(stab)) if stab else np.nan,
            "stability_std": float(np.std(stab)) if stab else np.nan,
            "diversity": diversity(ref),
        })
    table = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    return table, refs


def select_fine_k(table: pd.DataFrame, k_main: int, min_stability: float = 0.7) -> int:
    """The finer resolution: the largest stable k above `k_main`, else the next larger k of the table."""
    larger = table[table["k"] > k_main]
    if larger.empty:
        raise ValueError(f"no k larger than {k_main} in the grid; extend topics.k_grid")
    stable = larger[larger["stability"] >= min_stability]
    if stable.empty:
        log.warning("no k above %d reaches stability %.2f; taking the next larger k", k_main, min_stability)
        return int(larger["k"].min())
    return int(stable["k"].max())


def select_k(table: pd.DataFrame, criterion: str, min_stability: float = 0.7) -> int:
    def norm(s: pd.Series) -> pd.Series:
        s = s.fillna(s.min())
        span = s.max() - s.min()
        return (s - s.min()) / span if span > 0 else s * 0 + 1

    if criterion == "stable_coherence":
        stable = table["stability"] >= min_stability
        if not stable.any():
            log.warning("no k reaches stability %.2f; taking the most stable k", min_stability)
            score = table["stability"].fillna(-np.inf)
        else:
            score = table["npmi"].where(stable, -np.inf)
    elif criterion == "coherence":
        score = table["npmi"]
    elif criterion == "stability":
        score = table["stability"].fillna(-np.inf)
    else:
        parts = [norm(table["npmi"]), norm(table["diversity"])]
        if table["stability"].notna().any():
            parts.append(norm(table["stability"]))
        score = sum(parts) / len(parts)
    table["score"] = score
    return int(table.loc[score.idxmax(), "k"])
