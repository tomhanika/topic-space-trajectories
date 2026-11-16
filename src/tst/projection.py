"""2D map of topic space. The projector is fitted on a document sample and persisted, so documents and
trajectory points (and later, new documents) are placed on the same map."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

from .config import ProjectionConfig

log = logging.getLogger("tst.projection")


class Projector:
    def __init__(self, cfg: ProjectionConfig, seed: int, n_jobs: int):
        # Neither UMAP nor openTSNE scales beyond a few dozen threads; more only adds contention.
        self.cfg, self.seed, self.n_jobs = cfg, seed, max(1, min(n_jobs, 32))
        self.model = None

    def _prep(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return np.sqrt(np.clip(X, 0, None)) if self.cfg.hellinger else X

    def fit(self, X: np.ndarray) -> Projector:
        X = self._prep(X)
        c = self.cfg
        if c.method == "umap":
            from umap import UMAP

            # A fixed random_state forces UMAP to run single-threaded.
            rs = self.seed if c.deterministic else None
            self.model = UMAP(n_neighbors=c.n_neighbors, min_dist=c.min_dist, metric="euclidean",
                              random_state=rs, n_jobs=1 if rs is not None else self.n_jobs,
                              low_memory=True)
            self.model.fit(X)
        elif c.method == "tsne":
            from openTSNE import TSNE

            # Without FFTW, openTSNE's FFT gradient falls back to single-threaded numpy; Barnes-Hut is
            # multithreaded and faster up to a few hundred thousand points.
            method = "bh" if len(X) <= 200_000 else "fft"
            self.model = TSNE(perplexity=c.perplexity, n_jobs=self.n_jobs, random_state=self.seed,
                              negative_gradient_method=method).fit(X)
        else:
            from sklearn.decomposition import PCA

            self.model = PCA(2, random_state=self.seed).fit(X)
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        c = self.cfg
        if c.method == "umap":
            return self.model.embedding_.astype(np.float32)
        if c.method == "tsne":
            return np.asarray(self.model, dtype=np.float32)
        return self.model.transform(self._prep(X)).astype(np.float32)

    def transform(self, X: np.ndarray, batch: int = 200_000) -> np.ndarray:
        if len(X) == 0:
            return np.zeros((0, 2), np.float32)
        out = [self.model.transform(self._prep(X[i:i + batch])) for i in range(0, len(X), batch)]
        xy = np.vstack(out).astype(np.float32)
        bad = np.flatnonzero(~np.isfinite(xy).all(axis=1))
        if len(bad) and self.cfg.method == "umap":
            # UMAP's transform yields NaN for points that coincide with their fitted neighbours (e.g.
            # one-hot topic vectors); such a point takes the position of its nearest fitted point.
            raw, emb = self.model._raw_data, self.model.embedding_
            for i in bad:
                xy[i] = emb[np.argmin(((raw - self._prep(X[i:i + 1])) ** 2).sum(axis=1))]
            log.warning("projection: %d points placed at their nearest fitted point (UMAP gave NaN)", len(bad))
        return xy

    def save(self, path: Path) -> None:
        joblib.dump({"cfg": self.cfg.model_dump(), "seed": self.seed, "model": self.model}, path)

    @classmethod
    def load(cls, path: Path, n_jobs: int = 1) -> Projector:
        d = joblib.load(path)
        p = cls(ProjectionConfig(**d["cfg"]), d["seed"], n_jobs)
        p.model = d["model"]
        return p
