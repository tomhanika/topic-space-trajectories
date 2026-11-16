"""Pipeline stages. Each stage reads earlier artifacts from the run directory and writes its own.

A stage is skipped when its fingerprint (its config sections, its code version and the fingerprints of
the stages it depends on) matches the one recorded in state.json.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp

from . import analysis, export, indicators, io, network, novelty, parameters, terms, text, trajectories
from .projection import Projector
from .references import bibliography
from .run import RunContext, fingerprint, write_json
from .topics import nmf
from .topics.model import TopicModel, read_labels
from .topics.vectorize import Vectorizer, build_matrix, fit_vectorizer

log = logging.getLogger("tst.stages")


@dataclass
class Stage:
    name: str
    func: Callable[[RunContext], dict]
    deps: list[str]
    sections: list[str]
    version: int = 1
    enabled: Callable[[RunContext], bool] = lambda ctx: True


# ---------------------------------------------------------------------------
def ingest(ctx: RunContext) -> dict:
    return io.ingest(ctx.cfg.input, ctx.documents)


def _external_model(ctx: RunContext) -> TopicModel | None:
    return TopicModel.load(ctx.cfg.topics.model_path) if ctx.cfg.topics.model_path else None


def preprocess(ctx: RunContext) -> dict:
    cfg = ctx.cfg.preprocess
    model = _external_model(ctx)
    if model is not None:
        # Documents must be tokenised exactly as the model's training corpus was.
        tok = model.tokenizer
        cfg = cfg.model_copy(update={"fields": model.fields})
    else:
        tok = text.Tokenizer.from_config(cfg)
    return text.preprocess(ctx.documents, ctx.tokens, cfg, tok, ctx.cfg.n_jobs)


def vectorize(ctx: RunContext) -> dict:
    model = _external_model(ctx)
    vec, info = (model.vectorizer, {}) if model else fit_vectorizer(ctx.tokens, ctx.cfg.vectorize, ctx.cfg.n_jobs)
    vec.save(ctx.vectorizer_dir)
    X, ids = build_matrix(ctx.tokens, vec, ctx.cfg.n_jobs)
    sp.save_npz(ctx.matrix, X, compressed=False)
    pl.DataFrame({"doc_id": ids}).write_parquet(ctx.matrix_index)
    return {"documents": X.shape[0], "features": X.shape[1], "nnz": int(X.nnz), **info}


def train(ctx: RunContext) -> dict:
    c = ctx.cfg
    if c.topics.model_path:
        model = TopicModel.load(c.topics.model_path)
        return {"k": model.k, "model": str(c.topics.model_path), "trained": False}
    X = sp.load_npz(ctx.matrix).tocsr()
    rng = np.random.default_rng(c.seed)
    n = X.shape[0]
    train_rows = np.arange(n)
    if c.topics.train_sample is not None and n > c.topics.train_sample:
        train_rows = np.sort(rng.choice(n, c.topics.train_sample, replace=False))
    X_train = X[train_rows]
    ref_rows = np.arange(n)
    if n > c.topics.coherence_sample:
        ref_rows = np.sort(rng.choice(n, c.topics.coherence_sample, replace=False))
    B_ref = (X[ref_rows] > 0).astype(np.float32).tocsc()

    # The sweep is the expensive part; it is cached in the main run's model folder and reused when only the
    # selection settings, a k within the grid or the fine resolution change.
    def key(ks, runs):
        return fingerprint(
            c.topics.model_dump(mode="json", exclude={"k", "selection", "min_stability", "labels_file"}),
            c.seed, ks, runs, ctx.load_main_state().get("vectorize", {}).get("fingerprint"))

    grid_key = key(c.topics.k_grid, c.topics.runs)
    cached = _load_sweep(ctx.sweep_dir, grid_key)
    fine = ctx.resolution == "fine"
    if fine and cached is None:
        raise RuntimeError("The fine resolution needs the k sweep of the main run; run the main resolution first.")
    if cached is not None and (c.topics.k == "auto" or c.topics.k in cached[1]):
        log.info("reusing the cached sweep in %s", ctx.sweep_dir)
        table, refs = cached
        sweep_key = grid_key
    elif c.topics.k == "auto" or (c.fine.enabled and c.topics.k in c.topics.k_grid):
        # A fixed k with a fine resolution still needs the sweep of the whole grid, from which the fine
        # resolution takes its k.
        table, refs = nmf.sweep(X_train, B_ref, c.topics, c.seed, c.n_jobs, c.topics.k_grid, c.topics.runs)
        sweep_key = grid_key
    else:
        sweep_key = key([c.topics.k], 0)
        cached = _load_sweep(ctx.sweep_dir, sweep_key)
        table, refs = cached if cached is not None else nmf.sweep(X_train, B_ref, c.topics, c.seed, c.n_jobs,
                                                                  [c.topics.k], 0)
    k = nmf.select_k(table, c.topics.selection, c.topics.min_stability) if c.topics.k == "auto" else c.topics.k
    if fine:
        k = nmf.select_fine_k(table, k, c.topics.min_stability) if c.fine.k == "auto" else c.fine.k
        if k not in refs:
            raise ValueError(f"fine.k = {k} is not in topics.k_grid")
    log.info("model selection:\n%s\n-> k = %d%s", table.to_string(index=False), k, " (fine)" if fine else "")

    tok = text.Tokenizer.from_config(c.preprocess)
    model = TopicModel(
        tokenizer=tok, fields=list(c.preprocess.fields), vectorizer=Vectorizer.load(ctx.vectorizer_dir),
        components=refs[k], beta_loss=c.topics.beta_loss,
        meta={"source": f"trained by tst on run '{c.name}'", "train_documents": int(X_train.shape[0]),
              "algorithm": c.topics.algorithm, "seed": c.seed,
              "metrics": table.loc[table.k == k].iloc[0].to_dict()},
    )
    if ctx.model_dir.exists():
        shutil.rmtree(ctx.model_dir)
    model.save(ctx.model_dir)
    table.to_csv(ctx.model_dir / "sweep.csv", index=False)
    if not fine:
        # Keep the reference components of all k (small: k x V each); the fine resolution reads them.
        np.savez_compressed(ctx.model_dir / "sweep_components.npz", **{f"k{kk}": H for kk, H in refs.items()})
        (ctx.model_dir / "sweep.json").write_text(json.dumps({"key": sweep_key}))
    return {"k": k, "trained": True, "resolution": ctx.resolution,
            "sweep": table.drop(columns="score", errors="ignore").to_dict(orient="records")}


def _load_sweep(model_dir, key: str):
    import pandas as pd

    try:
        if json.loads((model_dir / "sweep.json").read_text())["key"] != key:
            return None
        table = pd.read_csv(model_dir / "sweep.csv").drop(columns="score", errors="ignore")
        z = np.load(model_dir / "sweep_components.npz")
        return table, {int(name[1:]): z[name] for name in z.files}
    except (FileNotFoundError, KeyError, ValueError):
        return None


def embed(ctx: RunContext) -> dict:
    model = TopicModel.load(ctx.model_dir)
    X = sp.load_npz(ctx.matrix).tocsr()
    ids = pl.read_parquet(ctx.matrix_index)["doc_id"].to_numpy()
    years = (pl.DataFrame({"doc_id": ids}).join(pl.read_parquet(ctx.documents, columns=["doc_id", "year"]),
                                                 on="doc_id", how="left", maintain_order="left")["year"].to_numpy())
    p = ctx.cfg.period
    max_year = trajectories.last_complete_year(years) if p.max_year == "auto" else p.max_year
    if p.max_year == "auto" and p.drop_snapshot_year and ctx.cfg.input.format == "openalex":
        from . import openalex

        if (snap := openalex.snapshot_year(ctx.cfg.input.path)) is not None and max_year is not None:
            max_year = min(max_year, snap - 1)
    keep = np.ones(len(ids), bool)
    if p.min_year is not None:
        keep &= years >= p.min_year
    if max_year is not None:
        keep &= years <= max_year
    X, ids = X[keep], ids[keep]
    theta, strength = model.transform_matrix(X, n_jobs=ctx.cfg.n_jobs)
    ok = strength > 0
    pl.DataFrame({
        "doc_id": ids[ok],
        "main_topic": theta[ok].argmax(axis=1).astype(np.int16),
        "strength": strength[ok],
        "theta": theta[ok],
    }).write_parquet(ctx.doc_topics)
    shutil.copy(ctx.model_dir / "topics.json", ctx.root / "model_topics.json")
    log.info("embed: years %s-%s, %d of %d documents", p.min_year, max_year, int(keep.sum()), len(keep))
    return {"embedded": int(ok.sum()), "empty": int((~ok).sum()), "outside_period": int((~keep).sum()),
            "min_year": p.min_year, "max_year": max_year, "k": model.k}


def load_doc_topics(ctx: RunContext) -> tuple[pl.DataFrame, np.ndarray]:
    """Documents with topic vectors; `row` indexes the returned theta matrix."""
    dt = pl.read_parquet(ctx.doc_topics)
    docs = pl.read_parquet(ctx.documents, columns=["doc_id", "year", "venue", "authors"]).join(
        dt.select("doc_id").with_row_index("row"), on="doc_id").sort("row")
    return docs, dt["theta"].to_numpy()


def build_trajectories(ctx: RunContext) -> dict:
    c = ctx.cfg.trajectories
    docs, theta = load_doc_topics(ctx)
    trends, field = trajectories.field_trends(docs, theta)
    trends.write_parquet(ctx.path("trajectories", "topic_trends.parquet"))

    parts, cents = [], []
    for etype, ecfg in c.entities.items():
        t, C = trajectories.build(docs, theta, etype, ecfg, field)
        if t.height:
            parts.append(t)
            cents.append(C)
    if not parts:
        raise ValueError("No trajectories: check min_papers and the entity columns of the corpus.")
    traj = pl.concat(parts, how="diagonal_relaxed")
    C = np.vstack(cents)
    cols = ["entity_type", "entity_id", "entity_name", "year", "n_papers", "weight"]
    traj = traj.select(cols + [c_ for c_ in traj.columns if c_ not in cols]).with_columns(
        centroid=pl.Series(C))
    traj.write_parquet(ctx.trajectories)
    trajectories.entity_summary(traj, C).write_parquet(ctx.path("trajectories", "entities.parquet"))
    info = {e: int(traj.filter(pl.col("entity_type") == e)["entity_id"].n_unique()) for e in c.entities}
    if c.pairwise.enabled and c.pairwise.entity_type in c.entities:
        pw = trajectories.pairwise(traj, C, c.pairwise)
        pw.write_parquet(ctx.path("trajectories", "pairwise.parquet"))
        info["pairs"] = pw.height
    return info


def project(ctx: RunContext) -> dict:
    c = ctx.cfg.projection
    rng = np.random.default_rng(ctx.cfg.seed)
    dt = pl.read_parquet(ctx.doc_topics, columns=["doc_id", "theta"])
    theta = dt["theta"].to_numpy()
    n = len(theta)
    fit_rows = rng.choice(n, min(n, c.fit_sample), replace=False)
    place_rows = np.arange(n) if c.doc_sample is None or n <= c.doc_sample else \
        np.sort(rng.choice(n, c.doc_sample, replace=False))
    traj = pl.read_parquet(ctx.trajectories, columns=["entity_type", "entity_id", "year", "centroid"])
    C = traj["centroid"].to_numpy()

    proj = Projector(c, ctx.cfg.seed, ctx.cfg.n_jobs)
    if len(place_rows) + len(C) <= c.fit_sample:
        # Small enough to embed everything jointly (better layout, no out-of-sample transform).
        xy = proj.fit_transform(np.vstack([theta[place_rows], C]))
        doc_xy, traj_xy = xy[: len(place_rows)], xy[len(place_rows):]
        fit_rows = place_rows
    else:
        proj.fit(theta[fit_rows])
        doc_xy = proj.transform(theta[place_rows])
        traj_xy = proj.transform(C)
    proj.save(ctx.path("projection", "projector.joblib"))
    coords = pl.concat([
        pl.DataFrame({"kind": "doc", "doc_id": dt["doc_id"].to_numpy()[place_rows],
                      "x": doc_xy[:, 0], "y": doc_xy[:, 1]}),
        traj.select("entity_type", "entity_id", "year").with_columns(
            kind=pl.lit("trajectory"), x=pl.Series(traj_xy[:, 0]), y=pl.Series(traj_xy[:, 1])),
    ], how="diagonal_relaxed")
    coords.write_parquet(ctx.coords)
    return {"fit": len(fit_rows), "documents": len(place_rows), "trajectory_points": len(C)}


def build_network(ctx: RunContext) -> dict:
    dt = pl.read_parquet(ctx.doc_topics, columns=["doc_id", "theta"])
    return network.build(ctx.documents, dt["doc_id"].to_numpy(), dt["theta"].to_numpy(),
                         ctx.cfg.network, ctx.network_dir)


def analyze(ctx: RunContext) -> dict:
    k = TopicModel.load(ctx.model_dir).k
    return analysis.analyze(ctx.network_dir, ctx.analysis_dir, k, ctx.cfg.analysis, ctx.cfg.n_jobs)


def novelty_stage(ctx: RunContext) -> dict:
    return novelty.compute(ctx.doc_topics, ctx.documents, ctx.cfg.novelty, ctx.path("novelty.parquet"))


def indicators_stage(ctx: RunContext) -> dict:
    components = np.load(ctx.model_dir / "components.npy")
    parent = None
    if ctx.resolution == "fine":
        parent = np.load(ctx.main_root / "model" / "components.npy")
    return indicators.compute(ctx.doc_topics, ctx.documents, ctx.path("novelty.parquet"),
                              ctx.path("trajectories", "topic_trends.parquet"), components, parent,
                              ctx.cfg.indicators, ctx.cfg.novelty.threshold, ctx.dir("indicators"), ctx.tokens)


def terms_stage(ctx: RunContext) -> dict:
    return terms.compute(ctx.tokens, ctx.doc_topics, ctx.documents, ctx.matrix, ctx.matrix_index,
                         ctx.vectorizer_dir, ctx.cfg.terms, ctx.cfg.n_jobs, ctx.dir("terms"))


def bundle(ctx: RunContext) -> dict:
    state = ctx.load_state()
    k = TopicModel.load(ctx.model_dir).k
    if ctx.resolution == "fine":
        state = {**{s: v for s, v in ctx.load_main_state().items() if s in SHARED}, **state}
    stages = {s: v.get("info", {}) for s, v in state.items()}
    if ctx.resolution == "fine" and "sweep" in ctx.load_main_state().get("train", {}).get("info", {}):
        stages["train"] = {**stages.get("train", {}), "sweep": ctx.load_main_state()["train"]["info"]["sweep"]}
    # Topic names are per k: the fine resolution reads export.labels_file with "_fine" before the suffix.
    labels_file = ctx.cfg.export.labels_file
    if labels_file is not None and ctx.resolution == "fine":
        labels_file = labels_file.with_name(f"{labels_file.stem}_fine{labels_file.suffix}")
    topics = json.loads((ctx.root / "model_topics.json").read_text())
    labels = read_labels(labels_file, k, topics)
    groups = parameters.describe(ctx.cfg, stages, k, ctx.resolution, named=bool(labels))
    methods = parameters.methods(ctx.cfg, stages, ctx.vectorizer_dir)
    run_info = {"name": ctx.cfg.name, "resolution": ctx.resolution, "input": ctx.cfg.input.path.name,
                "stages": stages, "parameters": groups, "references": bibliography(parameters.cited(groups, methods)),
                "config": parameters.resolved_config(ctx.cfg)}
    return export.write_bundle(ctx.root, ctx.bundle_dir, ctx.cfg.export, k, ctx.cfg.seed, run_info, labels,
                               documents=ctx.documents, methods=methods,
                               vocabulary=parameters.vocabulary(stages, ctx.vectorizer_dir))


_net = lambda ctx: ctx.cfg.network.enabled  # noqa: E731
_terms = lambda ctx: ctx.cfg.terms.enabled and ctx.resolution == "main"  # noqa: E731

STAGES: list[Stage] = [
    Stage("ingest", ingest, [], ["input"]),
    # v2: HTML entities, accents, apostrophes, Latin-only words, script share (the code, not only the config);
    # v3: titles in other languages left out, Indonesian title words, "science fiction" joined
    Stage("preprocess", preprocess, ["ingest"], ["preprocess", "topics.model_path"], version=3),
    # v2: curation; v3: stop words "collected", "stateoftheart" (the stop-word list is in no config section)
    Stage("vectorize", vectorize, ["preprocess"], ["vectorize", "topics.model_path"], version=3),
    Stage("train", train, ["vectorize"], ["topics", "seed"]),
    Stage("embed", embed, ["train"], ["period"]),
    Stage("trajectories", build_trajectories, ["embed"], ["trajectories"]),
    Stage("project", project, ["trajectories"], ["projection", "seed"]),
    Stage("network", build_network, ["embed"], ["network"], enabled=_net),
    Stage("analyze", analyze, ["network"], ["analysis"], enabled=_net),
    Stage("novelty", novelty_stage, ["embed"], ["novelty"]),
    Stage("indicators", indicators_stage, ["novelty", "trajectories"], ["indicators", "novelty.threshold"]),
    # Words per year: corpus-wide, so only in the main resolution.
    Stage("terms", terms_stage, ["embed"], ["terms"], enabled=_terms),
    # v2: methods, vocabulary; v3: topic names, pages, field indicators, novelty and citation percentiles
    Stage("export", bundle, ["project", "analyze", "indicators"], ["export", "seed"], version=3),
]
STAGE_NAMES = [s.name for s in STAGES]
# Stages whose output the fine resolution takes from the main run.
SHARED = ("ingest", "preprocess", "vectorize")


def _section(cfg, dotted: str):
    obj = cfg
    for part in dotted.split("."):
        obj = getattr(obj, part)
    if hasattr(obj, "fingerprint_dump"):
        return obj.fingerprint_dump()
    return obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj


def run(ctx: RunContext, only: list[str] | None = None, force: bool = False, dry_run: bool = False) -> list[str]:
    """Run the outdated stages; returns their names. `dry_run`: only return them, run nothing."""
    by_name = {s.name: s for s in STAGES}
    fine = ctx.resolution == "fine"
    if fine and ctx.cfg.topics.model_path:
        raise ValueError("A fine resolution needs a trained sweep; it does not work with topics.model_path.")
    if fine and only and (bad := set(only) & set(SHARED)):
        raise ValueError(f"The fine resolution takes {sorted(bad)} from the main run; rerun them there.")
    if not dry_run:
        ctx.dir()
        write_json(ctx.path("config.json"), ctx.cfg.model_dump(mode="json"))
    fps: dict[str, str] = {}
    ran: list[str] = []
    main_state = ctx.load_main_state() if fine else {}
    for stage in STAGES:
        if not stage.enabled(ctx):
            fps[stage.name] = "disabled"
            continue
        if fine and stage.name in SHARED:
            if stage.name not in main_state:
                raise RuntimeError(f"The fine resolution needs the main run's '{stage.name}' stage; run it first.")
            fps[stage.name] = main_state[stage.name]["fingerprint"]
            continue
        deps = [d for d in stage.deps if by_name[d].enabled(ctx)]
        if any(fps[d] == "missing" for d in deps):
            missing = [d for d in deps if fps[d] == "missing"]
            if only is not None and stage.name in only:
                raise RuntimeError(f"Stage '{stage.name}' needs {missing}, which have not run yet.")
            fps[stage.name] = "missing"
            continue
        sections = [_section(ctx.cfg, s) for s in stage.sections]
        if fine and stage.name == "train":
            sections.append(_section(ctx.cfg, "fine"))
        fp = fingerprint(stage.name, stage.version, sections, [fps[d] for d in deps])
        fps[stage.name] = fp
        recorded = ctx.load_state().get(stage.name, {}).get("fingerprint")
        selected = only is None or stage.name in only
        if recorded == fp and not (force and selected):
            log.info("stage %-12s up to date", stage.name)
            continue
        if not selected:
            if recorded is None:
                log.info("stage %-12s not run yet and not selected; skipping it", stage.name)
                fps[stage.name] = "missing"
                continue
            log.warning("stage %-12s is outdated but not selected; using existing output", stage.name)
            fps[stage.name] = recorded
            continue
        ran.append(stage.name)
        if dry_run:
            continue
        log.info("stage %-12s running", stage.name)
        t0 = time.time()
        info = stage.func(ctx)
        ctx.save_stage(stage.name, fp, time.time() - t0, info)
        log.info("stage %-12s done in %.1fs", stage.name, time.time() - t0)
    if not dry_run:
        log.info("run '%s'%s complete: %s", ctx.cfg.name, " (fine)" if fine else "", ctx.root)
    return ran


def status(ctx: RunContext) -> str:
    state = ctx.load_state()
    lines = []
    for s in STAGES:
        st = state.get(s.name)
        done = "done {} ({}s)".format(st["finished"], st["seconds"]) if st else "-"
        lines.append(f"{s.name:<13} {done}")
    return "\n".join(lines) + "\n" + json.dumps({s: v.get("info") for s, v in state.items()}, indent=1,
                                                default=str)[:4000]
