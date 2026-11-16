"""Count documents matching named drop-pattern candidates in every run, and keep the hits for inspection.

Patterns see "title\\nabstract", case-insensitively, as preprocess.drop_patterns do (tst.text.corpus_filters).
The patterns file is YAML: {group name: [regex, ...]}. Writes <out>/hits.parquet (run, group, doc_id, year,
venue, title, abstract) and prints a table of hits per run and group, split into documents the preprocess kept
("ok") and the ones it already removed.

    python scripts/pattern_hits.py candidates.yaml out/ [--runs law communication] [--workers 46]
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import yaml


def _scan(job):
    run, groups = job
    pf = pq.ParquetFile(run / "documents.parquet")
    try:
        reason = pl.read_parquet(run / "tokens.parquet", columns=["doc_id", "reason"])
    except Exception as e:  # a run being recomputed rewrites tokens.parquet; count its hits without "kept"
        print(f"{run.name}: tokens.parquet not readable ({e}); kept is unknown", flush=True)
        doc_id = pl.from_arrow(pf.read_row_group(0, columns=["doc_id"])).schema["doc_id"]
        reason = pl.DataFrame(schema={"doc_id": doc_id, "reason": pl.String})
    out = []
    for g in range(pf.metadata.num_row_groups):
        df = pl.from_arrow(pf.read_row_group(g, columns=["doc_id", "year", "venue", "title", "abstract"]))
        df = df.filter(pl.col("abstract").is_not_null())
        text = pl.concat_str([pl.col("title").fill_null(""), pl.col("abstract")], separator="\n")
        for name, pats in groups.items():
            pattern = "(?i)" + "|".join(f"(?:{p})" for p in pats)
            hit = df.filter(text.str.contains(pattern))
            if hit.height:
                out.append(hit.with_columns(run=pl.lit(run.name), group=pl.lit(name),
                                            length=pl.col("abstract").str.len_chars(),
                                            abstract=pl.col("abstract").str.slice(0, 600)))
    if not out:
        return None
    hits = pl.concat(out).join(reason, on="doc_id", how="left")
    return hits.with_columns(kept=pl.col("reason") == "ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patterns", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--runs", nargs="*", help="run names (default: every run with documents.parquet)")
    ap.add_argument("--root", type=Path, default=Path("runs"))
    ap.add_argument("--workers", type=int, default=46)
    args = ap.parse_args()
    groups = yaml.safe_load(args.patterns.read_text())
    runs = [args.root / r for r in args.runs] if args.runs else sorted(
        p.parent for p in args.root.glob("*/documents.parquet"))
    # Largest runs first, so the long ones don't start last.
    runs.sort(key=lambda r: -(r / "documents.parquet").stat().st_size)
    with ProcessPoolExecutor(min(args.workers, len(runs))) as ex:
        parts = [p for p in ex.map(_scan, [(r, groups) for r in runs]) if p is not None]
    args.out.mkdir(parents=True, exist_ok=True)
    hits = pl.concat(parts) if parts else pl.DataFrame()
    hits.write_parquet(args.out / "hits.parquet")
    if not hits.height:
        print("no hits")
        return
    # Runs whose tokens.parquet could not be read (kept is null) count as kept here.
    table = (hits.filter(pl.col("kept").fill_null(True)).group_by("run", "group").len()
             .pivot(on="group", index="run", values="len").fill_null(0).sort("run"))
    with pl.Config(tbl_rows=100, tbl_cols=30, fmt_str_lengths=40, tbl_width_chars=250):
        print("Hits among documents the preprocess kept:")
        print(table)
        print(hits.group_by("group").agg(kept=pl.col("kept").sum(), removed=(~pl.col("kept")).sum(),
                                         unknown=pl.col("kept").is_null().sum()).sort("group"))


if __name__ == "__main__":
    main()
