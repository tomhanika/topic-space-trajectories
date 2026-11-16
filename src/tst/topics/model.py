"""The topic model bundle: tokenizer + vectorizer + NMF components. No pickles, only JSON and .npy."""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.decomposition import non_negative_factorization
from threadpoolctl import threadpool_limits

from ..text import Tokenizer, document_text
from .vectorize import Vectorizer

BUNDLE_VERSION = 1

log = logging.getLogger("tst.model")


def _nnls_block(X: sp.csr_matrix, H: np.ndarray, beta_loss: str, max_iter: int) -> np.ndarray:
    with threadpool_limits(1):
        W, _, _ = non_negative_factorization(
            X, H=H, n_components=H.shape[0], update_H=False,
            solver="mu" if beta_loss != "frobenius" else "cd", beta_loss=beta_loss,
            max_iter=max_iter, tol=1e-6,
        )
    return W.astype(np.float32)


@dataclass
class TopicModel:
    tokenizer: Tokenizer
    fields: list[str]
    vectorizer: Vectorizer
    components: np.ndarray  # k x V, non-negative
    beta_loss: str = "frobenius"
    labels: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return self.components.shape[0]

    def top_terms(self, n: int = 10) -> list[list[tuple[str, float]]]:
        H = self.components / np.maximum(self.components.sum(axis=1, keepdims=True), 1e-12)
        vocab = self.vectorizer.vocabulary
        return [[(vocab[j], float(row[j])) for j in np.argsort(-row)[:n]] for row in H]

    def default_labels(self, n: int = 3) -> list[str]:
        return [", ".join(t.replace("_", " ") for t, _ in tt) for tt in self.top_terms(n)]

    def label(self, topic: int) -> str:
        return self.labels[topic] if self.labels else self.default_labels()[topic]

    # Embedding ---------------------------------------------------------------
    def transform_matrix(self, X: sp.csr_matrix, n_jobs: int = 1, block: int = 20_000,
                         max_iter: int = 500) -> tuple[np.ndarray, np.ndarray]:
        """Topic distributions (rows sum to 1) and raw strengths (row sums before normalising).

        W is the non-negative least squares solution of X ≈ W H with H fixed, which is deterministic.
        Documents without any known term get an all-zero row and strength 0.
        """
        H = self.components.astype(X.dtype)
        blocks = [X[i:i + block] for i in range(0, X.shape[0], block)]
        if n_jobs == 1 or len(blocks) == 1:
            parts = [_nnls_block(b, H, self.beta_loss, max_iter) for b in blocks]
        else:
            parts = Parallel(n_jobs=n_jobs)(delayed(_nnls_block)(b, H, self.beta_loss, max_iter) for b in blocks)
        W = np.vstack(parts) if parts else np.zeros((0, self.k), np.float32)
        strength = W.sum(axis=1)
        theta = np.divide(W, strength[:, None], out=np.zeros_like(W), where=strength[:, None] > 0)
        return theta, strength

    def transform_texts(self, texts: list[str], n_jobs: int = 1) -> np.ndarray:
        X = self.vectorizer.transform([self.tokenizer(t) for t in texts])
        return self.transform_matrix(X, n_jobs)[0]

    def transform_documents(self, titles: list[str | None], abstracts: list[str | None]) -> np.ndarray:
        tok = self.tokenizer
        return self.transform_texts([document_text(None if tok.ignores_title(t, a, self.fields) else t, a, self.fields)
                                     for t, a in zip(titles, abstracts)])

    # Persistence -------------------------------------------------------------
    def save(self, folder: str | Path, top_n: int = 20) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.vectorizer.save(folder)
        np.save(folder / "components.npy", self.components.astype(np.float32))
        labels = self.labels or self.default_labels()
        meta = {
            "bundle_version": BUNDLE_VERSION,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "k": self.k,
            "beta_loss": self.beta_loss,
            "fields": self.fields,
            "tokenizer": self.tokenizer.to_dict(),
            **self.meta,
        }
        (folder / "model.json").write_text(json.dumps(meta, indent=2, default=str))
        topics = [
            {"id": i, "label": labels[i], "manual_label": bool(self.labels),
             "top_terms": [{"term": t, "weight": round(w, 6)} for t, w in tt]}
            for i, tt in enumerate(self.top_terms(top_n))
        ]
        (folder / "topics.json").write_text(json.dumps(topics, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, folder: str | Path) -> TopicModel:
        folder = Path(folder)
        meta = json.loads((folder / "model.json").read_text())
        if meta.get("bundle_version") != BUNDLE_VERSION:
            raise ValueError(f"Unsupported model bundle version in {folder}")
        topics = json.loads((folder / "topics.json").read_text())
        manual = [t["label"] for t in topics] if topics and topics[0].get("manual_label") else []
        tok = meta.pop("tokenizer")
        return cls(
            # Options missing in older bundles keep the Tokenizer defaults, i.e. the behaviour of that time.
            tokenizer=Tokenizer(**{f.name: tok[f.name] for f in dataclasses.fields(Tokenizer) if f.name in tok}),
            fields=meta.pop("fields"),
            vectorizer=Vectorizer.load(folder),
            components=np.load(folder / "components.npy"),
            beta_loss=meta.pop("beta_loss"),
            labels=manual,
            meta={k: v for k, v in meta.items() if k not in ("bundle_version", "created", "k")},
        )


def read_labels(path: Path | None, k: int, topics: list[dict] | None = None) -> list[str]:
    """Topic names from a labels file: TSV with a header "id<TAB>name<TAB>terms" (terms: the topic's top terms when
    it was named, comma-separated), or one name per line in topic order. A missing file gives no names. With
    `topics` (model_topics.json), the names of a TSV are kept only if each named topic still shares at least half
    of its top five terms with the recorded ones: a retrained model has other topics behind the same ids."""
    if path is None or not path.exists():
        if path is not None:
            log.info("no topic names: %s does not exist", path)
        return []
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if lines and lines[0].lower().startswith("id\tname"):
        rows = [line.split("\t") for line in lines[1:]]
        names = {int(r[0]): r[1].strip() for r in rows}
        if sorted(names) != list(range(k)):
            # Written for a model with another number of topics, e.g. before a recompute.
            log.warning("topic names in %s are for %d topics, the model has %d; not used", path, len(names), k)
            return []
        if topics is not None:
            for r in rows:
                if len(r) < 3:
                    continue
                recorded = [t.strip() for t in r[2].split(",")][:5]
                current = [t["term"] for t in topics[int(r[0])]["top_terms"][:5]]
                if len(set(recorded) & set(current)) < (len(recorded) + 1) // 2:
                    log.warning("topic names in %s are for another model (topic %s: %s vs %s); not used",
                                path, r[0], recorded, current)
                    return []
        return [names[i] for i in range(k)]
    labels = [line.strip() for line in lines]
    if len(labels) != k:
        raise ValueError(f"{path} has {len(labels)} labels, model has {k} topics")
    return labels
