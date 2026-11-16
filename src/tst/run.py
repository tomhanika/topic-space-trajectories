"""Run directory layout, logging and stage bookkeeping."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .config import RunConfig

log = logging.getLogger("tst")


def setup_logging(logfile: Path | None = None, level: int = logging.INFO) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger("tst")
    root.setLevel(level)
    root.handlers.clear()
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        root.addHandler(fh)


RESOLUTIONS = ("main", "fine")


@dataclass
class RunContext:
    """A run directory. The fine resolution lives in <workdir>/fine and reads the main run's documents,
    tokens, TF-IDF matrix, vectorizer and sweep."""
    cfg: RunConfig
    resolution: str = "main"

    @property
    def root(self) -> Path:
        return self.cfg.workdir / "fine" if self.resolution == "fine" else self.cfg.workdir

    @property
    def main_root(self) -> Path:
        return self.cfg.workdir

    def main_path(self, *parts: str) -> Path:
        p = self.main_root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def path(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def dir(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Canonical artifact locations.
    @property
    def documents(self) -> Path:
        return self.main_path("documents.parquet")

    @property
    def tokens(self) -> Path:
        return self.main_path("tokens.parquet")

    @property
    def matrix(self) -> Path:
        return self.main_path("tfidf.npz")

    @property
    def matrix_index(self) -> Path:
        return self.main_path("tfidf_index.parquet")

    @property
    def vectorizer_dir(self) -> Path:
        return self.main_root / "vectorizer"

    @property
    def sweep_dir(self) -> Path:
        """Where the k sweep is cached: the main run's model folder."""
        return self.main_root / "model"

    @property
    def model_dir(self) -> Path:
        return self.cfg.topics.model_path or self.root / "model"

    @property
    def doc_topics(self) -> Path:
        return self.path("doc_topics.parquet")

    @property
    def trajectories(self) -> Path:
        return self.path("trajectories", "trajectories.parquet")

    @property
    def coords(self) -> Path:
        return self.path("projection", "coords.parquet")

    @property
    def network_dir(self) -> Path:
        return self.dir("network")

    @property
    def analysis_dir(self) -> Path:
        return self.dir("analysis")

    @property
    def bundle_dir(self) -> Path:
        return self.dir("bundle")

    # Stage state: a fingerprint per finished stage, so unchanged stages are skipped.
    @property
    def _state_file(self) -> Path:
        return self.path("state.json")

    def load_state(self) -> dict:
        if self._state_file.exists():
            return json.loads(self._state_file.read_text())
        return {}

    def load_main_state(self) -> dict:
        f = self.main_root / "state.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def save_stage(self, stage: str, fingerprint: str, seconds: float, info: dict | None = None) -> None:
        state = self.load_state()
        state[stage] = {
            "fingerprint": fingerprint,
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seconds": round(seconds, 1),
            "info": info or {},
        }
        self._state_file.write_text(json.dumps(state, indent=2, default=str))


_THREAD_VARS = ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS")


@contextmanager
def limited_threads(n: int):
    """Thread limits inherited by worker processes started inside the block."""
    old = {v: os.environ.get(v) for v in _THREAD_VARS}
    os.environ.update({v: str(n) for v in _THREAD_VARS})
    try:
        yield
    finally:
        for v, val in old.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val


def fingerprint(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default))


def _json_default(o):
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
