# Derived measures

Part of the bundle schema 1.4 (see [README.md](README.md) for conventions). Facts computed from the topic model,
the papers and their citations: one page per topic, indicators of the whole corpus per year, and a novelty score
and citation percentile per paper. How each one is computed, with the literature behind it, is in
`manifest.json` `run.parameters`, group `Derived measures` ([methods.md](methods.md)); a web app should show
that text next to the numbers. Settings are in `methods.json` `derived` and `run.config` (`novelty`,
`indicators`).

## Definitions

* **Has a topic**: a paper has a topic if the topic's share in it (`theta`) is at least `novelty.threshold`
  (0.1).
* **Novelty** of a paper (Lee, Walsh & Wang 2015; Uzzi et al. 2013, with topics in place of cited journals): for
  each pair of topics, the papers of the `novelty.window` (3) years before the paper's year give the observed
  number of papers having both and the number expected if topics combined independently (`n_i · n_j / n`). Their
  ratio, with one added to each count, is the pair's commonness. Novelty = −log10 of the commonness of the
  paper's rarest pair: 0 = as common as expected, 1 = ten times rarer, −1 = ten times more common. Null for
  papers with fewer than two topics, and in years whose window has fewer than `novelty.min_window_papers`
  papers. **Unusual**: the rarest pair occurs at most `novelty.unusual_ratio` (half) as often as expected.
* **Citation percentile**: the paper's OpenAlex citation count (at the snapshot, counted over all of OpenAlex)
  as a percentile among the corpus's papers of the same publication year; ties share their mean rank, so
  uncited papers sit at about half the share of uncited papers. Averages leave out the last
  `indicators.citation_lag` (2) years, whose papers had little time to be cited.

## documents.parquet (columns added in 1.4)

`novelty`, `novelty_pct` (percentile within the year), `unusual`, `citation_pct`; see
[documents.md](documents.md). For colouring the map by novelty or impact.

## topic_pages.json

A list ordered by topic id, one page per topic:

| field | |
|---|---|
| id | topic id |
| papers | papers whose main topic it is |
| life_cycle | `peak_year`, `peak_share`, `rise_year` (first year the smoothed share reached half its peak), `share_first`, `share_last`, `recent_change` (mean share of the last 5 years / the 5 before − 1), `status`: `emerging` (growing and risen within the last 10 years), `growing`, `stable`, `declining` (±20 %), or null for short series. Shares are smoothed with a 3-year moving average |
| typical_papers | up to 8 papers: the largest topic weight (share × strength) among papers with at least half their topic share on this topic and an English title. Each: `doc_id`, `title`, `year`, `venue`, `citations`, `source_id`, `share`, `url` |
| cited_papers | the 8 most cited papers with this main topic (all time): same fields with `citation_pct` instead of `share` |
| venues | up to 10: `name`, `papers` (with this main topic), `share_of_venue` (of the venue's papers in the corpus) |
| authors | up to 10 by papers with this main topic: `id` (OpenAlex author id), `name`, `papers`; authors without an id are left out |
| team_mean, team_median, solo_share | authors per paper, share of single-author papers |
| citation_pct, top10_share | mean citation percentile and share of top-10 % cited papers (without the last 2 years) |
| novelty_mean, unusual_share | mean novelty and share of unusual papers |
| related | 5 topics most often combined with this one in papers: `id`, `lift` = P(both) / (P(this)·P(other)); above 1 = more often than chance |
| similar | 5 topics with the most similar term weights: `id`, `cosine` |
| parent | fine resolution only: as in `topics.json` |

```json
{"id": 14, "papers": 15425,
 "life_cycle": {"peak_year": 2025, "peak_share": 0.0755, "rise_year": 2019, "share_first": 0.0146,
                "share_last": 0.0755, "recent_change": 0.611, "status": "emerging"},
 "typical_papers": [{"doc_id": 81234, "title": "The Role of Social Media in ...", "year": 2025, "venue": "...",
                     "citations": 0, "source_id": "W...", "share": 0.97, "url": "https://openalex.org/W..."}],
 "venues": [{"name": "Religions", "papers": 129, "share_of_venue": 0.089}],
 "authors": [{"id": 5012345678, "name": "Heidi A. Campbell", "papers": 32}],
 "team_mean": 1.75, "team_median": 1.0, "solo_share": 0.55, "citation_pct": 54.99, "top10_share": 0.149,
 "novelty_mean": 0.155, "unusual_share": 0.25,
 "related": [{"id": 8, "lift": 1.86}], "similar": [{"id": 16, "cosine": 0.153}]}
```

## field_indicators.json

```json
{"years": [{"year": 2024, "papers": 27075, "authors": 27234, "effective_topics": 22.47, "topics_per_paper": 2.95,
            "change": 0.018, "team_mean": 1.38, "team_median": 1.0, "solo_share": 0.76, "newcomer_share": 0.62,
            "novelty_mean": 0.16, "unusual_share": 0.25}, ...],
 "novelty_impact": {"years": [1960, 2023],
                    "deciles": [{"decile": 1, "papers": 40174, "citation_pct": 52.15, "top10_share": 0.129}, ...],
                    "single_topic": {"papers": 27053, "citation_pct": 52.31, "top10_share": 0.134},
                    "spearman": 0.0026}}
```

Per year of the topic space:

| field | |
|---|---|
| papers, authors | papers; distinct authors (with an OpenAlex id) |
| effective_topics | exp(Shannon entropy) of the year's topic shares (Hill number of order 1; Hill 1973, Jost 2006): the number of equally common topics that would be as diverse; at most `k` |
| topics_per_paper | mean number of topics a paper has (share ≥ threshold) |
| change | Hellinger distance between the topic shares of the year and the year before (null in the first year) |
| team_mean, team_median, solo_share | authors per paper |
| newcomer_share | share of the year's authors without an earlier paper in the corpus; null in the first 5 years, which have no history |
| novelty_mean, unusual_share | mean novelty, share of unusual papers among those with a score |

`novelty_impact` relates novelty to later citations: papers of the years shown, grouped into deciles of their
novelty percentile within the year (1 = most conventional, 10 = most novel), with their mean citation
percentile and share of top-10 % cited papers; `single_topic` are the papers without a novelty score;
`spearman` the rank correlation between novelty and citation percentile. Without any relation, every decile has
a mean percentile near 50 and a top-10 % share near 0.1.
