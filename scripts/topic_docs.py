"""Show what a topic is made of: its size, top venues and the most typical documents.

Used to audit topics that look like junk (scraped pages, newsletters, front matter) before writing drop patterns.

    python scripts/topic_docs.py runs/law 1 [--fine] [-n 12] [--random] [--venues 15]
"""

import argparse
import textwrap
from pathlib import Path

import polars as pl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="run directory, e.g. runs/law")
    ap.add_argument("topics", type=int, nargs="+", help="topic ids (main_topic)")
    ap.add_argument("--fine", action="store_true", help="use the fine resolution")
    ap.add_argument("-n", type=int, default=12, help="documents to show")
    ap.add_argument("--random", action="store_true", help="random documents instead of the strongest")
    ap.add_argument("--venues", type=int, default=15, help="top venues to list")
    ap.add_argument("--chars", type=int, default=300, help="abstract characters to show")
    args = ap.parse_args()

    dt_path = args.run / ("fine" if args.fine else "") / "doc_topics.parquet"
    dt = pl.read_parquet(dt_path, columns=["doc_id", "main_topic", "strength"])
    total = dt.height
    dt = dt.filter(pl.col("main_topic").is_in(args.topics))
    docs = pl.read_parquet(args.run / "documents.parquet",
                           columns=["doc_id", "year", "venue", "source_id", "title", "abstract"])
    df = dt.join(docs, on="doc_id")
    print(f"{df.height} documents ({100 * df.height / total:.2f}% of {total}) in topics {args.topics}")
    for v, n in df.group_by("venue").len().sort("len", descending=True).head(args.venues).iter_rows():
        print(f"  {n:7d}  {v}")
    years = df.group_by("year").len().sort("year")
    peak = years.sort("len", descending=True).row(0)
    print(f"  years {years['year'].min()}-{years['year'].max()}, peak {peak[0]} ({peak[1]})")
    show = df.sample(min(args.n, df.height), seed=0) if args.random else df.sort("strength", descending=True).head(args.n)
    for r in show.iter_rows(named=True):
        print(f"\n[{r['doc_id']}] {r['year']} | {r['venue']} | {r['source_id']} | strength {r['strength']:.2f}")
        print("  T:", (r["title"] or "")[:160])
        print(textwrap.indent(textwrap.fill((r["abstract"] or "")[: args.chars], 110), "  A: "))


if __name__ == "__main__":
    main()
