"""Outputs across corpora, written by `tst collect` next to the bundles:

* `_terms/`: the term explorer. Papers per year that use each word or two-word phrase, per corpus (tst.terms),
  split into files by the first two letters of the term.
* `_global/mobility.json`: researchers moving between corpora. Each author's home corpus per period (the corpus of
  most of their papers in it); counts of authors who stay, move to another corpus or enter.

Both only use the papers of each corpus's topic space. Corpora partition OpenAlex (each subfield is in one
corpus), so a paper counts in one corpus only.
"""

from __future__ import annotations

import collections
import json
import logging
import shutil
from pathlib import Path

import polars as pl

log = logging.getLogger("tst.crossfield")

TERMS_METHOD = {
    "description": (
        "Share of papers per year that use a word or a two-word phrase, in each corpus. Words are those of the "
        "corpus's topic model after curation: plural and spelling variants count as one word ('networks' = "
        "'network'), stop words are left out. Phrases are two words that stand next to each other once stop words "
        "are removed ('climate change', 'neural network'; 'theory of mind' appears as 'theory mind'), both in the "
        "corpus vocabulary, and kept if at least the corpus's minimum number of papers use them. A paper counts once "
        "however often it uses the term. Only papers in the corpus's topic space count, with the years of its map."),
    "share": "papers using the term in a year / papers of the corpus in that year (years.parquet)",
    "limits": (
        "Words below a corpus's minimum document frequency or used in more than its maximum share of papers are "
        "not in its vocabulary (manifest run.parameters, Vocabulary). Early years have few papers with abstracts, "
        "so shares there are noisy."),
}

MOBILITY_METHOD = {
    "description": (
        "For every author (OpenAlex author id) and period, the home corpus is the corpus of most of their papers in "
        "the period; authors with fewer papers than the minimum in a period, or with a tie between corpora, have no "
        "home in it. An author with a home in two consecutive periods stays (same corpus) or moves (other corpus). "
        "Entering authors have a home in a period but none in the period before (first papers, or too few papers "
        "before); leaving authors the other way round. Multi-corpus authors publish in two or more corpora within "
        "a period. Works without an OpenAlex topic (the misc corpus) are no research field and are left out."),
    "limits": (
        "OpenAlex author ids come from automatic name disambiguation, which sometimes splits one person into "
        "several ids or merges namesakes; merged namesakes look like movers. Corpora are groups of OpenAlex "
        "subfields assigned by each paper's primary topic, so a move can also be a change in how a researcher's "
        "papers are classified. Large fields are split into several corpora (engineering, medicine, computer "
        "science), so moves and multi-corpus authors include moves between their parts. Only papers in the topic "
        "spaces count (English, with an abstract)."),
    "references": ["openalex"],
}


def build_terms(out: Path, runs: list[tuple[str, Path]]) -> dict:
    d = out / "_terms"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    terms, years, fields = [], [], []
    for name, root in runs:
        f = root / "terms" / "terms.parquet"
        if not f.exists():
            log.warning("no term counts for %s in %s; skipping it", name, f.parent)
            continue
        fields.append(name)
        terms.append(pl.scan_parquet(f).with_columns(pl.lit(name).alias("field")))
        years.append(pl.read_parquet(root / "terms" / "years.parquet").with_columns(pl.lit(name).alias("field")))
    if not fields:
        return {"fields": 0}
    pl.concat(years).select("field", "year", "papers").sort("field", "year").write_parquet(d / "years.parquet")
    first2 = pl.col("term").str.slice(0, 2)
    allt = (pl.concat(terms).with_columns(pl.when(first2.str.contains("^[a-z]{2}$")).then(first2)
                                          .otherwise(pl.lit("_other")).alias("prefix"))
            .collect(engine="streaming"))
    files = {}
    for (prefix,), part in allt.partition_by(["prefix"], as_dict=True).items():
        part.select("term", "field", pl.col("year").cast(pl.Int16), pl.col("papers").cast(pl.Int32)) \
            .sort("term", "field", "year").write_parquet(d / f"{prefix}.parquet", compression="zstd",
                                                          row_group_size=20_000, statistics=True)
        files[prefix] = f"{prefix}.parquet"
    index = {"fields": fields, "years": "years.parquet", "files": dict(sorted(files.items())),
             "terms": int(allt["term"].n_unique()), "rows": allt.height, "method": TERMS_METHOD}
    (d / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
    log.info("term explorer: %d terms, %d rows, %d files", index["terms"], allt.height, len(files))
    return {"fields": len(fields), "terms": index["terms"], "files": len(files)}


def build_mobility(out: Path, runs: list[tuple[str, Path]], period: int = 5, min_papers: int = 2) -> dict:
    parts, last_years = [], []
    for name, root in runs:
        dt, docs = root / "doc_topics.parquet", root / "documents.parquet"
        if not dt.exists() or not docs.exists():
            log.warning("no topic space for %s; skipping it", name)
            continue
        a = (pl.scan_parquet(docs).select("doc_id", "year", pl.col("authors").list.eval(pl.element().struct.field("id"))
                                          .alias("author"))
             .join(pl.scan_parquet(dt).select("doc_id"), on="doc_id")
             .explode("author").drop_nulls("author")
             .group_by("author", "year").len()
             .with_columns(pl.lit(name).alias("field"))
             .collect(engine="streaming"))
        last_years.append(int(a["year"].max()))
        parts.append(a)
    if not parts:
        return {"fields": 0}
    # Periods end at the last year of most corpora, counting back; a corpus whose map ends earlier has no papers in
    # the last years of the final period.
    last = collections.Counter(last_years).most_common(1)[0][0]
    allp = (pl.concat(parts).filter(pl.col("year") <= last)
            .with_columns(((last - pl.col("year")) // period).cast(pl.Int16).alias("p"))
            .group_by("author", "p", "field").agg(pl.col("len").sum().alias("n")))
    n_periods = int(allp["p"].max()) + 1
    per = (allp.group_by("author", "p").agg(
        pl.col("n").sum().alias("total"), pl.len().alias("fields"),
        pl.col("field").sort_by("n", descending=True).first().alias("home"),
        (pl.col("n") == pl.col("n").max()).sum().alias("ties"))
        .filter(pl.col("total") >= min_papers))
    multi = per.group_by("p").agg((pl.col("fields") >= 2).mean().alias("share"), pl.len().alias("authors"))
    home = per.filter(pl.col("ties") == 1).select("author", "p", "home")
    later = home.rename({"home": "to"})
    earlier = home.rename({"home": "from"}).with_columns((pl.col("p") - 1).cast(pl.Int16))
    trans = later.join(earlier, on=["author", "p"], how="full", coalesce=True)
    flows = (trans.drop_nulls(["from", "to"]).group_by("p", "from", "to").len()
             .sort("p", "len", descending=[False, True]))
    # p = 0 is the latest period; output chronologically.
    def span(p: int) -> dict:
        return {"start": last - period * (p + 1) + 1, "end": last - period * p}
    periods = [span(p) for p in range(n_periods - 1, -1, -1)]
    order = {p: n_periods - 1 - p for p in range(n_periods)}
    transitions = []
    for p in range(n_periods - 2, -1, -1):
        f = flows.filter(pl.col("p") == p)
        if f.height:
            transitions.append({"from_period": order[p + 1], "to_period": order[p],
                                "flows": [{"from": r["from"], "to": r["to"], "authors": r["len"]}
                                          for r in f.iter_rows(named=True)]})
    entered = (trans.filter(pl.col("from").is_null() & pl.col("to").is_not_null() & (pl.col("p") < n_periods - 1))
               .group_by("p", "to").len().rename({"to": "field", "len": "entered"}))
    left = (trans.filter(pl.col("to").is_null() & pl.col("from").is_not_null() & (pl.col("p") >= 0))
            .group_by("p", "from").len().rename({"from": "field", "len": "left_corpora"}))
    homes = home.group_by("p", "home").len().rename({"home": "field", "len": "authors"})
    moves = flows.filter(pl.col("from") != pl.col("to"))
    moved_in = moves.group_by("p", "to").agg(pl.col("len").sum().alias("arrived")).rename({"to": "field"})
    # All counts of a period are about the change since the period before: departed and left count authors whose
    # home was this corpus before and is another one, or none, now.
    moved_out = moves.group_by("p", "from").agg(pl.col("len").sum().alias("departed")).rename({"from": "field"})
    stayed = (flows.filter(pl.col("from") == pl.col("to")).group_by("p", "to").agg(pl.col("len").sum().alias("stayed"))
              .rename({"to": "field"}))
    summary = homes
    for t in (stayed, moved_in, moved_out, entered, left):
        summary = summary.join(t, on=["p", "field"], how="left")
    summary = summary.fill_null(0).sort("p", "field", descending=[True, False])
    fields = sorted({n for n, _ in runs})
    result = {
        "method": {**MOBILITY_METHOD, "period_years": period, "min_papers": min_papers},
        "periods": periods,
        "fields": fields,
        "transitions": transitions,
        "fields_by_period": [{"period": order[r["p"]], "field": r["field"], "authors": r["authors"],
                              "stayed": r["stayed"], "arrived": r["arrived"], "departed": r["departed"],
                              "entered": r["entered"], "left": r["left_corpora"]}
                             for r in summary.iter_rows(named=True)],
        "multi_corpus": [{"period": order[r["p"]], "authors": r["authors"], "share": round(r["share"], 4)}
                         for r in multi.sort("p", descending=True).iter_rows(named=True)],
    }
    g = out / "_global"
    g.mkdir(parents=True, exist_ok=True)
    (g / "mobility.json").write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    log.info("mobility: %d periods, %d fields", n_periods, len(fields))
    return {"periods": n_periods, "fields": len(fields)}
