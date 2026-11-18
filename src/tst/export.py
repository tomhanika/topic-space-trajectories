"""The visualization bundle: everything a web app needs, independent of this package. See schema/."""

from __future__ import annotations

import colorsys
import json
import logging
import os
import shutil
import time
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from scipy.stats import gaussian_kde

from . import __version__
from .config import ExportConfig
from .run import write_json

log = logging.getLogger("tst.export")

SCHEMA_VERSION = "1.5"


def topic_colors(k: int) -> list[str]:
    from matplotlib import colormaps

    base = [colormaps[name].colors for name in ("tab20", "tab20b", "tab20c")]
    cols = [c for cmap in base for c in cmap]
    if k <= 10:
        cols = list(colormaps["tab10"].colors)
    elif k <= 20:
        cols = list(colormaps["tab20"].colors)
    out = []
    for i in range(k):
        if i < len(cols):
            r, g, b = cols[i][:3]
        else:  # golden-angle hues beyond the qualitative palettes
            r, g, b = colorsys.hsv_to_rgb((i * 0.618033988749895) % 1, 0.65, 0.85)
        out.append("#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255)))
    return out


def density_grids(xy: np.ndarray, topics: np.ndarray, k: int, grid: int, sample: int, seed: int):
    """Per-topic KDE on a common grid over the map. Returns Z[k, grid, grid] (row 0 = smallest y) and bounds."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(xy).all(axis=1)
    xy, topics = xy[ok], topics[ok]
    lo = np.percentile(xy, 0.5, axis=0)
    hi = np.percentile(xy, 99.5, axis=0)
    pad = (hi - lo) * 0.05
    lo, hi = lo - pad, hi + pad
    gx = np.linspace(lo[0], hi[0], grid)
    gy = np.linspace(lo[1], hi[1], grid)
    X, Y = np.meshgrid(gx, gy)
    pos = np.vstack([X.ravel(), Y.ravel()])
    Z = np.zeros((k, grid, grid), np.float32)
    per_topic = max(200, sample // max(k, 1))
    for t in range(k):
        pts = xy[topics == t]
        if len(pts) < 5:
            continue
        if len(pts) > per_topic:
            pts = pts[rng.choice(len(pts), per_topic, replace=False)]
        try:
            Z[t] = gaussian_kde(pts.T)(pos).reshape(grid, grid) * (topics == t).mean()
        except (np.linalg.LinAlgError, ValueError):  # points (nearly) on one spot: no density
            log.warning("density: topic %d skipped, its %d points are degenerate", t, len(pts))
            continue
    return Z, {"x_min": float(lo[0]), "x_max": float(hi[0]), "y_min": float(lo[1]), "y_max": float(hi[1])}


def density_image(Z: np.ndarray, colors: list[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    rgb = np.array([[int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)] for c in colors])
    Z = Z.astype(np.float64)  # Z ** 2 underflows in float32 at the map's edges
    total = Z.sum(axis=0)
    # Squared shares let the locally dominant topic's colour win instead of averaging towards grey.
    sq = Z ** 2
    norm = sq.sum(axis=0)
    share = np.divide(sq, norm, out=np.zeros_like(Z), where=norm > 0)
    img = np.clip(np.einsum("kij,kc->ijc", share, rgb), 0, 1)
    alpha = np.clip(total / np.percentile(total[total > 0], 99) if (total > 0).any() else total, 0, 1)
    plt.imsave(path, np.dstack([img, alpha ** 0.6])[::-1])  # image row 0 is the top of the map


def write_bundle(root: Path, out: Path, cfg: ExportConfig, k: int, seed: int, run_info: dict,
                 labels: list[str] | None = None, documents: Path | None = None, methods: dict | None = None,
                 vocabulary: dict | None = None, citations: Path | None = None, papers: Path | None = None) -> dict:
    documents = documents or root / "documents.parquet"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    files: dict[str, str] = {}

    topics = json.loads((root / "model_topics.json").read_text())
    colors = topic_colors(k)
    trends = pl.read_parquet(root / "trajectories" / "topic_trends.parquet")
    prev = trends["prevalence"].to_numpy()
    overall = (prev * trends["n_docs"].to_numpy()[:, None]).sum(axis=0) / trends["n_docs"].sum()
    pages = root / "indicators" / "topic_pages.json"
    parents = {p["id"]: p.get("parent") for p in json.loads(pages.read_text())} if pages.exists() else {}
    for t in topics:
        # label: what to show; name: the topic's name (null if unnamed); terms_label: its top three terms.
        t["terms_label"] = ", ".join(x["term"] for x in t["top_terms"][:3])
        t["name"] = labels[t["id"]] if labels else None
        t["label"] = t["name"] or t["terms_label"]
        t["manual_label"] = bool(labels)
        t["color"] = colors[t["id"]]
        t["prevalence"] = float(overall[t["id"]])
        t["top_terms"] = t["top_terms"][:20]
        if parents.get(t["id"]):
            t["parent"] = parents[t["id"]]
    write_json(out / "topics.json", topics)
    files["topics"] = "topics.json"

    trends.write_parquet(out / "topic_trends.parquet")
    write_json(out / "topic_trends.json", {
        "years": trends["year"].to_list(), "n_docs": trends["n_docs"].to_list(),
        "prevalence": np.round(prev, 5).tolist()})
    files["topic_trends"] = "topic_trends.json"

    coords = pl.read_parquet(root / "projection" / "coords.parquet")
    doc_xy = coords.filter(pl.col("kind") == "doc").select("doc_id", "x", "y")
    dt = pl.read_parquet(root / "doc_topics.parquet")
    doc_cols = ["doc_id", "year", "venue", "title", "citations", "authors"]
    if "source_id" in pq.read_schema(documents).names:
        doc_cols.append("source_id")
    docs = (pl.read_parquet(documents, columns=doc_cols)
            .join(dt, on="doc_id").join(doc_xy, on="doc_id"))
    novelty = root / "novelty.parquet"
    if novelty.exists():
        docs = docs.join(pl.read_parquet(novelty, columns=["doc_id", "novelty", "novelty_pct", "unusual",
                                                           "citation_pct"]), on="doc_id", how="left")
    if papers is not None and papers.exists():  # tst citations: references and knowledge-base surprise
        docs = docs.join(pl.read_parquet(papers, columns=["doc_id", "references", "references_resolved",
                                                          "kb_surprise", "kb_surprise_pct"]), on="doc_id", how="left")
    if cfg.max_docs is not None and docs.height > cfg.max_docs:
        docs = docs.sample(cfg.max_docs, seed=seed)
    docs = docs.with_columns(
        pl.col("authors").list.eval(pl.element().struct.field("name")).alias("authors"),
    ).sort("doc_id")
    if cfg.url_template:
        docs = docs.with_columns(pl.struct(pl.col("title"), pl.col("source_id") if "source_id" in docs.columns
                                           else pl.lit(None, pl.String).alias("source_id")).map_elements(
            lambda r: paper_url(cfg.url_template, r["title"], r["source_id"]), return_dtype=pl.String).alias("url"))
    docs.write_parquet(out / "documents.parquet")
    files["documents"] = "documents.parquet"

    # Density from all placed documents, not only the exported sample.
    all_docs = doc_xy.join(dt.select("doc_id", "main_topic"), on="doc_id")
    xy = all_docs.select("x", "y").to_numpy()
    if len(xy) > 10:
        Z, bounds = density_grids(xy, all_docs["main_topic"].to_numpy(), k, cfg.density_grid,
                                  cfg.density_sample, seed)
        np.savez_compressed(out / "density.npz", Z=Z.astype(np.float16), **{k_: np.array(v) for k_, v in bounds.items()})
        density_image(Z, colors, out / "density.png")
        write_json(out / "density.json", {"grid": cfg.density_grid, **bounds, "image": "density.png",
                                          "row0": "y_max (image) / y_min (npz)"})
        files["density"] = "density.json"

    traj = pl.read_parquet(root / "trajectories" / "trajectories.parquet")
    tcoords = coords.filter(pl.col("kind") == "trajectory").select("entity_type", "entity_id", "year", "x", "y")
    traj = traj.join(tcoords, on=["entity_type", "entity_id", "year"], how="left")
    traj.write_parquet(out / "trajectories.parquet")
    files["trajectories"] = "trajectories.parquet"
    # JSON for entity types small enough to ship to a browser in one piece.
    for etype, part in traj.partition_by("entity_type", as_dict=True).items():
        etype = etype[0]
        if part["entity_id"].n_unique() > 5000:
            continue
        payload = []
        for (eid,), g in part.sort("year").partition_by("entity_id", as_dict=True, maintain_order=True).items():
            payload.append({
                "id": eid, "name": g["entity_name"][0],
                "points": [{"year": int(r["year"]), "x": r["x"], "y": r["y"], "n": int(r["n_papers"]),
                            "main_topic": int(r["main_topic"]),
                            "centroid": [round(float(v), 5) for v in r["centroid"]]}
                           for r in g.iter_rows(named=True)],
            })
        write_json(out / f"trajectories_{etype}.json", payload)
        files[f"trajectories_{etype}"] = f"trajectories_{etype}.json"

    for name in ("entities", "pairwise"):
        src = root / "trajectories" / f"{name}.parquet"
        if src.exists():
            shutil.copy(src, out / f"{name}.parquet")
            files[name] = f"{name}.parquet"

    analysis = root / "analysis"
    if analysis.exists():
        net = out / "network"
        net.mkdir()
        for f in sorted(analysis.glob("*.parquet")):
            shutil.copy(f, net / f.name)
            files[f"network/{f.stem}"] = f"network/{f.name}"
        if (analysis / "flows.parquet").exists():
            fl = pl.read_parquet(analysis / "flows.parquet").filter(pl.col("source_topic") != pl.col("target_topic"))
            top = (fl.sort("weight", descending=True).group_by("year", maintain_order=True).head(25)
                   .sort("year", "weight", descending=[False, True]))
            write_json(net / "flows_top.json", top.to_dicts())
            files["network/flows_top"] = "network/flows_top.json"
        stats = root / "network" / "authors.parquet"
        if stats.exists():
            shutil.copy(stats, net / "authors.parquet")
            files["network/authors"] = "network/authors.parquet"

    # Field indicators and topic pages (tst.indicators); papers get their links here.
    for name in ("field_indicators", "topic_pages"):
        src = root / "indicators" / f"{name}.json"
        if not src.exists():
            continue
        data = json.loads(src.read_text())
        if name == "topic_pages" and cfg.url_template:
            for page in data:
                for p in page["typical_papers"] + page["cited_papers"]:
                    p["url"] = paper_url(cfg.url_template, p.get("title"), p.get("source_id"))
        (out / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
        files[name] = f"{name}.json"

    # Citation flow between topics (tst citations), when computed for this corpus and resolution.
    if citations is not None and citations.exists():
        (out / "citations.json").write_text(citations.read_text())
        files["citations"] = "citations.json"

    # How the bundle was made: the lists behind run.parameters, and the vocabulary (large, so compact JSON).
    if methods is not None:
        write_json(out / "methods.json", methods)
        files["methods"] = "methods.json"
    if vocabulary is not None:
        (out / "vocabulary.json").write_text(json.dumps(vocabulary, ensure_ascii=False, separators=(",", ":")))
        files["vocabulary"] = "vocabulary.json"

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator": f"topic-space-trajectories {__version__}",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "k": k,
        "resolution": run_info.get("resolution", "main"),
        "label": cfg.label,
        "group": cfg.group,
        "distance": "hellinger",
        "files": files,
        "run": run_info,
    }
    write_json(out / "manifest.json", manifest)
    log.info("bundle written to %s (%d files)", out, len(files))
    return {"files": len(files), "documents": docs.height}


def paper_url(template: str, title: str | None, source_id: str | None) -> str | None:
    if "{source_id}" in template and not source_id:
        return None
    return template.format(title=quote_plus(title or ""), source_id=source_id or "")


def _remove(dest: Path) -> None:
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.exists():
        shutil.rmtree(dest)


def collect(out: Path, runs: list[tuple[str, str, Path]], copy: bool = False, remove: list[str] = ()) -> dict:
    """Gather bundles in one folder: `<out>/<name>` (main), `<out>/<name>_fine` and `<out>/index.json`.

    runs: (name, resolution, bundle folder). Bundles are linked (relative symlinks) unless `copy`; missing
    bundles are skipped. Existing entries of other runs in `out` are kept, and `index.json` lists them all,
    except the runs named in `remove` (corpora not to publish), whose entries are deleted.
    """
    out.mkdir(parents=True, exist_ok=True)
    for name in remove:
        for dest in (out / name, out / f"{name}_fine"):
            if dest.exists() or dest.is_symlink():
                log.info("removing %s (export.publish is false)", dest)
                _remove(dest)
    for name, resolution, src in runs:
        dest = out / (name if resolution == "main" else f"{name}_{resolution}")
        if not (src / "manifest.json").exists():
            log.warning("no bundle for %s (%s) in %s; skipping it", name, resolution, src)
            continue
        _remove(dest)
        if copy:
            shutil.copytree(src, dest)
        else:
            dest.symlink_to(Path(os.path.relpath(src.resolve(), out.resolve())), target_is_directory=True)
    fields: dict[str, dict] = {}
    for m in sorted(out.glob("*/manifest.json")):
        man = json.loads(m.read_text())
        res = man.get("resolution", "main")
        name = m.parent.name.removesuffix(f"_{res}") if res != "main" else m.parent.name
        field = fields.setdefault(name, {"name": name, "resolutions": {}})
        for key in ("label", "group"):
            if man.get(key):
                field[key] = man[key]
        field["resolutions"][res] = {"path": m.parent.name, "k": man["k"], "created": man.get("created")}
    index = {"schema_version": SCHEMA_VERSION, "fields": list(fields.values())}
    write_json(out / "index.json", index)
    log.info("%d fields in %s", len(fields), out)
    return {"fields": len(fields), "bundles": sum(len(f["resolutions"]) for f in fields.values())}
