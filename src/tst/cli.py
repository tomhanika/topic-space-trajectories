"""Command line interface: `tst run config.yaml`, `tst status`, `tst embed-text`,
`tst openalex-fetch`, `tst collect`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import load_config
from .run import RunContext, setup_logging


def main(argv: list[str] | None = None) -> None:
    # Persist numba's compiled code (UMAP) between runs; the first compile takes minutes.
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path.home() / ".cache" / "tst-numba"))
    p = argparse.ArgumentParser(prog="tst", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the pipeline for a config")
    r.add_argument("config")
    r.add_argument("--stages", help="comma separated stages to (re)run, e.g. trajectories,project,export")
    r.add_argument("--force", action="store_true", help="rerun the selected stages even if up to date")
    r.add_argument("--resolution", choices=["main", "fine"],
                   help="only this topic resolution (default: main, then fine if fine.enabled)")
    r.add_argument("--n-jobs", type=int, help="worker processes, overriding the config's n_jobs (default: all cores)")
    r.add_argument("--dry-run", action="store_true", help="only list the stages that would run")

    s = sub.add_parser("status", help="show which stages of a run are done")
    s.add_argument("config")
    s.add_argument("--resolution", choices=["main", "fine"], default="main")

    s = sub.add_parser("stages", help="list the pipeline stages")

    e = sub.add_parser("embed-text", help="topic vector of a text with a model bundle")
    e.add_argument("model", type=Path)
    e.add_argument("text", nargs="?", help="text, or read from stdin")
    e.add_argument("--top", type=int, default=5)

    o = sub.add_parser("openalex-fetch", help="download the needed columns of the OpenAlex works snapshot")
    o.add_argument("store", type=Path, help="local store folder (use it as input.path with format: openalex)")
    o.add_argument("--jobs", type=int, default=48, help="parallel downloads")
    o.add_argument("--limit", type=int, help="only the first N snapshot files (for testing)")
    o.add_argument("--links", action="store_true",
                   help="references, authors' countries and institutions of all works, into <store>/links")

    ci = sub.add_parser("citations", help="citation flow between topics, for all corpora at once")
    ci.add_argument("configs", nargs="+")
    ci.add_argument("--steps", help="comma-separated subset of: titles,nodes,authors,resolve,fit,global,papers")
    ci.add_argument("--jobs", type=int, default=200, help="worker processes for the title topics")
    ci.add_argument("--global-window", type=int, default=5, help="citing years per window of the global fit")
    ci.add_argument("--force", action="store_true", help="recompute steps whose outputs exist")

    c = sub.add_parser("collect", help="gather the bundles of several runs in one folder for a web app")
    c.add_argument("out", type=Path, help="folder: <out>/<name>, <out>/<name>_fine and <out>/index.json")
    c.add_argument("configs", nargs="+")
    c.add_argument("--copy", action="store_true", help="copy the bundles instead of linking to them")
    c.add_argument("--no-global", action="store_true",
                   help="skip the outputs across corpora (term explorer, researchers moving between corpora)")

    a = p.parse_args(argv)
    if a.cmd == "run":
        from .stages import SHARED, STAGE_NAMES, run

        cfg = load_config(a.config)
        if a.n_jobs:
            cfg.n_jobs = a.n_jobs
        only = a.stages.split(",") if a.stages else None
        if only and (bad := set(only) - set(STAGE_NAMES)):
            p.error(f"unknown stages {bad}; choose from {STAGE_NAMES}")
        resolutions = [a.resolution] if a.resolution else ["main"] + (["fine"] if cfg.fine.enabled else [])
        for res in resolutions:
            ctx = RunContext(cfg, res)
            if not a.dry_run:
                setup_logging(ctx.path("logs", "run.log"))
            # The fine resolution takes the shared stages from the main run.
            stages = [s for s in only if s not in SHARED] if only and res == "fine" else only
            if stages != []:
                ran = run(ctx, stages, a.force, a.dry_run)
                if a.dry_run:
                    print(f"{cfg.name} {res}: {', '.join(ran) or 'up to date'}")
    elif a.cmd == "status":
        from .stages import status

        print(status(RunContext(load_config(a.config), a.resolution)))
    elif a.cmd == "stages":
        from .stages import STAGES

        for st in STAGES:
            print(f"{st.name:<13} after: {', '.join(st.deps) or '-':<22} config: {', '.join(st.sections) or '-'}")
    elif a.cmd == "openalex-fetch":
        from .openalex import fetch, fetch_links

        setup_logging(a.store / ("fetch_links.log" if a.links else "fetch.log"))
        print(json.dumps((fetch_links if a.links else fetch)(a.store, a.jobs, a.limit), indent=1))
    elif a.cmd == "citations":
        from .citations import STEPS, run as run_citations

        steps = a.steps.split(",") if a.steps else None
        if steps and (bad := set(steps) - set(STEPS)):
            p.error(f"unknown steps {bad}; choose from {STEPS}")
        setup_logging()
        print(json.dumps(run_citations(a.configs, steps, a.jobs, a.force, a.global_window), indent=1, default=str))
    elif a.cmd == "collect":
        from .export import collect

        setup_logging()
        runs, roots, hidden = [], [], []
        for path in a.configs:
            cfg = load_config(path)
            if not cfg.export.publish:
                hidden.append(cfg.name)
                continue
            runs += [(cfg.name, res, RunContext(cfg, res).bundle_dir) for res in ("main", "fine")]
            roots.append((cfg.name, cfg.workdir, cfg.export.group))
        result = collect(a.out, runs, a.copy, remove=hidden)
        if not a.no_global:
            from . import crossfield

            result["terms"] = crossfield.build_terms(a.out, [(n, r) for n, r, _ in roots])
            # Works without an OpenAlex topic are no research field to move to or from.
            result["mobility"] = crossfield.build_mobility(a.out, [(n, r) for n, r, g in roots if g != "Unclassified"])
            # Citation flow between fields (tst citations writes it next to the run folders).
            gcit = Path(os.path.commonpath([str(r) for _, r, _ in roots])) / "_citations" / "global.json"
            if gcit.exists():
                (a.out / "_global").mkdir(parents=True, exist_ok=True)
                (a.out / "_global" / "citations.json").write_text(gcit.read_text())
                result["citations"] = True
            index_file = a.out / "index.json"
            index = json.loads(index_file.read_text())
            index["global"] = {k: v for k, v in (("terms", "_terms/index.json"), ("mobility", "_global/mobility.json"),
                                                 ("citations", "_global/citations.json"))
                               if (a.out / v).exists()}
            index_file.write_text(json.dumps(index, indent=1, ensure_ascii=False))
        print(json.dumps(result, indent=1))
    elif a.cmd == "embed-text":
        from .topics.model import TopicModel

        m = TopicModel.load(a.model)
        theta = m.transform_texts([a.text if a.text else sys.stdin.read()])[0]
        top = theta.argsort()[::-1][: a.top]
        print(json.dumps([{"topic": int(t), "label": m.label(int(t)), "weight": round(float(theta[t]), 4)}
                          for t in top], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
