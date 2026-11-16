"""Field indicators per year and one page of facts per topic.

Field indicators (field_indicators.json), per publication year of the papers in the topic space:
  papers, authors        papers; distinct authors
  effective_topics       exp(Shannon entropy) of the year's topic shares (Hill number of order 1): the number of
                         equally large topics that would give the same diversity (Hill 1973, Jost 2006)
  topics_per_paper       mean number of topics with a share of at least the novelty threshold: how mixed papers are
  change                 Hellinger distance between the topic shares of the year and the year before: how fast the
                         field's topic mix moves
  team_mean, team_median, solo_share   authors per paper
  newcomer_share         share of the year's authors without an earlier paper in the corpus (left out for the first
                         `recent` years, which have no history to compare with)
  novelty_mean, unusual_share   mean novelty and share of papers with an unusual topic pair (tst.novelty)
plus novelty_impact: citation percentile and share of top-10 % cited papers per novelty decile.

Topic pages (topic_pages.json), per topic: papers, life cycle, typical and most cited papers, venues, authors,
team size, citation impact, novelty, related topics (co-occurrence), similar topics (vocabulary) and, in a finer
resolution, the parent topic of the main resolution.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from .config import IndicatorsConfig
from .novelty import theta_matrix

log = logging.getLogger("tst.indicators")


def _entropy_exp(p: np.ndarray) -> np.ndarray:
    """exp(Shannon entropy) of each row of p (rows sum to 1, or are all 0)."""
    p = np.clip(p, 0, None)
    s = p.sum(axis=1, keepdims=True)
    q = np.divide(p, s, out=np.zeros_like(p), where=s > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(q > 0, q * np.log(q), 0.0).sum(axis=1)
    return np.exp(h)


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def life_cycle(years: list[int], shares: np.ndarray, cfg: IndicatorsConfig) -> dict:
    """Peak, rise and recent change of one topic's yearly share (smoothed with a centred moving average)."""
    w = max(1, cfg.smoothing)
    s = np.convolve(np.pad(shares, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w, mode="valid")
    peak = int(np.argmax(s))
    rise = int(np.argmax(s >= 0.5 * s[peak]))
    r = cfg.recent
    out = {"peak_year": years[peak], "peak_share": round(float(s[peak]), 5), "rise_year": years[rise],
           "share_first": round(float(s[0]), 5), "share_last": round(float(s[-1]), 5),
           "recent_change": None, "status": None}
    if len(shares) >= 2 * r:
        now, before = float(shares[-r:].mean()), float(shares[-2 * r:-r].mean())
        change = now / before - 1 if before > 0 else None
        out["recent_change"] = None if change is None else round(change, 4)
        if change is not None:
            out["status"] = "growing" if change >= cfg.change else "declining" if change <= -cfg.change else "stable"
            if out["status"] == "growing" and years[rise] >= years[-1] - 2 * r:
                out["status"] = "emerging"
    return out


def compute(doc_topics: Path, documents: Path, novelty: Path, trends: Path, components: np.ndarray,
            parent_components: np.ndarray | None, cfg: IndicatorsConfig, novelty_threshold: float,
            out_dir: Path, tokens: Path | None = None) -> dict:
    dt = pl.read_parquet(doc_topics, columns=["doc_id", "main_topic", "strength", "theta"])
    docs = pl.read_parquet(documents, columns=["doc_id", "year", "venue", "title", "citations", "authors",
                                               *(["source_id"] if "source_id" in pl.read_parquet_schema(documents)
                                                 else [])])
    nov = pl.read_parquet(novelty, columns=["doc_id", "novelty", "unusual", "citation_pct", "novelty_pct"])
    d = (dt.select("doc_id", "main_topic").join(docs, on="doc_id", how="left", maintain_order="left")
         .join(nov, on="doc_id", how="left", maintain_order="left")
         .with_columns(pl.col("authors").list.len().alias("team")))
    theta = theta_matrix(dt["theta"])
    strength = dt["strength"].to_numpy()
    k = theta.shape[1]
    # Papers whose title was left out as not English (tst.text), or is mostly in other scripts than Latin, are not
    # shown as typical papers.
    title = pl.col("title").fill_null("")
    english_title = d.select((title.str.count_matches(r"[A-Za-z]") >= 0.5 * title.str.count_matches(r"\p{L}"))
                             & (title.str.len_chars() > 0)).to_series().to_numpy()
    if tokens is not None and "title_ignored" in pl.read_parquet_schema(tokens):
        ignored = pl.read_parquet(tokens, columns=["doc_id", "title_ignored"]).filter(pl.col("title_ignored"))
        english_title &= ~d["doc_id"].is_in(ignored["doc_id"].implode()).to_numpy()
    years = d["year"].to_numpy()
    last_year = int(years.max())
    impact_until = last_year - cfg.citation_lag

    # --- field indicators per year ------------------------------------------------------------------------
    topics_per_paper = (theta >= novelty_threshold).sum(axis=1)
    uy = np.unique(years)
    shares = np.stack([theta[years == y].mean(axis=0) for y in uy])
    shares = shares / np.clip(shares.sum(axis=1, keepdims=True), 1e-12, None)
    authors = (d.select("year", pl.col("authors").list.eval(pl.element().struct.field("id")).alias("a"))
               .explode("a").drop_nulls("a"))
    first = authors.group_by("a").agg(pl.col("year").min().alias("first"))
    per_year_authors = (authors.unique().join(first, on="a")
                        .group_by("year").agg(pl.len().alias("authors"),
                                              (pl.col("first") == pl.col("year")).mean().alias("newcomer_share")))
    by_year = (d.group_by("year").agg(
        pl.len().alias("papers"), pl.col("team").mean().alias("team_mean"),
        pl.col("team").median().alias("team_median"), (pl.col("team") == 1).mean().alias("solo_share"),
        pl.col("novelty").mean().alias("novelty_mean"),
        pl.col("unusual").filter(pl.col("novelty").is_not_null()).mean().alias("unusual_share"))
        .join(per_year_authors, on="year", how="left").sort("year"))
    first_year = int(uy.min())
    rows = []
    for i, r in enumerate(by_year.iter_rows(named=True)):
        y = r["year"]
        j = int(np.searchsorted(uy, y))
        rows.append({
            "year": y, "papers": r["papers"], "authors": r["authors"],
            "effective_topics": round(float(_entropy_exp(shares[j:j + 1])[0]), 3),
            "topics_per_paper": round(float(topics_per_paper[years == y].mean()), 3),
            "change": round(hellinger(shares[j], shares[j - 1]), 4) if j > 0 else None,
            "team_mean": _r(r["team_mean"], 3), "team_median": _r(r["team_median"], 1),
            "solo_share": _r(r["solo_share"], 4),
            "newcomer_share": _r(r["newcomer_share"], 4) if y >= first_year + cfg.recent else None,
            "novelty_mean": _r(r["novelty_mean"], 4), "unusual_share": _r(r["unusual_share"], 4),
        })
    # Does novelty pay off? Citation percentile per novelty decile (within year), years with time to be cited.
    imp = d.filter((pl.col("year") <= impact_until) & pl.col("novelty_pct").is_not_null())
    deciles = (imp.with_columns((pl.col("novelty_pct") // 10).clip(0, 9).cast(pl.Int8).alias("decile"))
               .group_by("decile").agg(pl.len().alias("papers"), pl.col("citation_pct").mean().alias("citation_pct"),
                                       (pl.col("citation_pct") >= 90).mean().alias("top10_share"))
               .sort("decile"))
    single = d.filter((pl.col("year") <= impact_until) & pl.col("novelty").is_null())
    field = {
        "years": rows,
        "novelty_impact": {
            "years": [first_year, impact_until],
            "deciles": [{"decile": int(r["decile"]) + 1, "papers": r["papers"],
                         "citation_pct": round(r["citation_pct"], 2), "top10_share": round(r["top10_share"], 4)}
                        for r in deciles.iter_rows(named=True)],
            "single_topic": {"papers": single.height,
                             "citation_pct": _r(single["citation_pct"].mean(), 2) if single.height else None,
                             "top10_share": _r((single["citation_pct"] >= 90).mean(), 4) if single.height else None},
            "spearman": _r(imp.select(pl.corr("novelty", "citation_pct", method="spearman")).item(), 4)
            if imp.height > 2 else None,
        },
    }

    # --- topic pages ------------------------------------------------------------------------------------------
    tr = pl.read_parquet(trends)
    t_years = tr["year"].to_list()
    prev = theta_matrix(tr["prevalence"]).astype(np.float64)
    present = theta >= novelty_threshold
    p_i = present.mean(axis=0)
    co = (present.T.astype(np.float32) @ present.astype(np.float32)) / len(present)
    lift = co / np.clip(np.outer(p_i, p_i), 1e-12, None)
    unit = components / np.clip(np.linalg.norm(components, axis=1, keepdims=True), 1e-12, None)
    cos = unit @ unit.T
    if parent_components is not None:
        punit = parent_components / np.clip(np.linalg.norm(parent_components, axis=1, keepdims=True), 1e-12, None)
        pcos = unit @ punit.T

    papers_cols = ["doc_id", "title", "year", "venue", "citations"] + (["source_id"] if "source_id" in d.columns else [])
    venue_totals = d.drop_nulls("venue").group_by("venue").len().rename({"len": "venue_papers"})
    pages = []
    for t in range(k):
        mine = d.filter(pl.col("main_topic") == t)
        # Typical papers: the largest topic weight (share x strength), so that long abstracts mostly about the
        # topic come before short ones with a share of 1; at least half of the paper on the topic.
        col = theta[:, t]
        weight = np.where((col >= 0.5) & english_title, col * strength, -1.0)
        top_rows = np.argsort(-weight, kind="stable")[: cfg.top_papers * 5]
        top_rows = top_rows[weight[top_rows] > 0]
        typical = _distinct(d[top_rows.tolist()].with_columns(pl.Series("share", col[top_rows])), cfg.top_papers)
        cited = _distinct(mine.sort("citations", descending=True, nulls_last=True).head(cfg.top_papers * 5),
                          cfg.top_papers)
        venues = (mine.drop_nulls("venue").group_by("venue").len().join(venue_totals, on="venue")
                  .with_columns((pl.col("len") / pl.col("venue_papers")).alias("share"))
                  .sort(["len", "venue"], descending=[True, False]).head(cfg.top_entities))
        au = (mine.select(pl.col("authors")).explode("authors").drop_nulls("authors").unnest("authors")
              .drop_nulls("id").group_by("id").agg(pl.len().alias("papers"), pl.col("name").first())
              .sort(["papers", "id"], descending=[True, False]).head(cfg.top_entities))
        recent_impact = mine.filter(pl.col("year") <= impact_until)
        others = [j for j in np.argsort(-lift[t]) if j != t][: cfg.related]
        similar = [j for j in np.argsort(-cos[t]) if j != t][: cfg.related]
        page = {
            "id": t,
            "papers": mine.height,
            "life_cycle": life_cycle(t_years, prev[:, t], cfg),
            "typical_papers": _papers(typical, papers_cols, "share"),
            "cited_papers": _papers(cited, papers_cols, "citation_pct"),
            "venues": [{"name": r["venue"], "papers": r["len"], "share_of_venue": round(r["share"], 4)}
                       for r in venues.iter_rows(named=True)],
            "authors": [{"id": r["id"], "name": r["name"], "papers": r["papers"]} for r in au.iter_rows(named=True)],
            "team_mean": _r(mine["team"].mean(), 3), "team_median": _r(mine["team"].median(), 1),
            "solo_share": _r((mine["team"] == 1).mean(), 4) if mine.height else None,
            "citation_pct": _r(recent_impact["citation_pct"].mean(), 2) if recent_impact.height else None,
            "top10_share": _r((recent_impact["citation_pct"] >= 90).mean(), 4) if recent_impact.height else None,
            "novelty_mean": _r(mine["novelty"].mean(), 4),
            "unusual_share": _r(mine.filter(pl.col("novelty").is_not_null())["unusual"].mean(), 4),
            "related": [{"id": int(j), "lift": round(float(lift[t, j]), 3)} for j in others],
            "similar": [{"id": int(j), "cosine": round(float(cos[t, j]), 4)} for j in similar],
        }
        if parent_components is not None:
            j = int(np.argmax(pcos[t]))
            page["parent"] = {"id": j, "cosine": round(float(pcos[t, j]), 4)}
        pages.append(page)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "field_indicators.json").write_text(_dumps(field))
    (out_dir / "topic_pages.json").write_text(_dumps(pages))
    info = {"years": [int(uy.min()), int(uy.max())], "topics": k, "impact_until": impact_until,
            "novelty_citation_spearman": field["novelty_impact"]["spearman"]}
    log.info("indicators: %s", info)
    return info


def _distinct(df: pl.DataFrame, n: int) -> pl.DataFrame:
    """The first n rows with different titles: OpenAlex has some papers several times, with slightly different
    titles or abstracts (preprint, version of record, conference and journal version)."""
    key = pl.col("title").fill_null("").str.to_lowercase().str.replace_all(r"[^a-z0-9]", "").str.slice(0, 50)
    return df.with_columns(key.alias("_k")).unique("_k", keep="first", maintain_order=True).drop("_k").head(n)


def _r(x, digits: int):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), digits)


def _papers(df: pl.DataFrame, cols: list[str], extra: str) -> list[dict]:
    out = []
    for r in df.select([c for c in cols + [extra] if c in df.columns]).iter_rows(named=True):
        out.append({c: (round(v, 4) if isinstance(v, float) else v) for c, v in r.items()})
    return out


def _dumps(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
