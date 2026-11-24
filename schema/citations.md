# Citation flow between topics

Part of the bundle schema 1.5 (see [README.md](README.md) for conventions). Written by `tst citations
configs/*.yaml`, which runs once for all corpora because references cross corpora; the export stage then puts
`citations.json` into each bundle (main and fine) and `tst collect` puts `_global/citations.json` next to them.
The theory, with the propositions behind each quantity, is in `notes/citation_flow_theory.md`.

## Model in one paragraph

A paper p cites an earlier work q at the rate `r_p · g(t_p − t_q) · θ_q' B θ_p`: `r_p` is the number of p's
references, `g` an aging kernel over the citation lag, `θ` the topic mixes from the text (each work keeps its
`top_topics` strongest topics, renormalised), and `B[i][j]` the rate at which topic-j work cites topic-i work
per reference and per available topic-i paper. B and g are fitted by EM, separately for each window of citing
years. Citable works are the corpus papers and, unless `title_only` is false, works without an abstract of the
same types, subfields and years, whose topic mix comes from their title. Citations between papers that share an
author are left out (`self_citations: "exclude"`).

## Definitions

* **Flow** `F[i][j]`: the expected number of citations carried from topic i (the cited work) to topic j (the
  citing paper), given the observed citations. Unlike splitting each citation over all topic pairs ("product
  attribution"), it does not book citations within one topic as flow between the topics a paper mixes.
* **Rate**: `B[i][j]` divided by the mean rate of the window. Rates do not grow with topic size or growth, so
  direction is read from them: `rate[i][j] > rate[j][i]` means j draws on i more than i draws on j.
* **Lift**: the flow over its expectation when the cited ends of the citations are shuffled among citations
  between the same pair of years (which keeps every paper's number of citations and references). Above 1: j draws
  on i more than chance. Blind to direction.
* **Potential** of a topic (upstream potential): from the Hodge decomposition of `log(B[i][j] / B[j][i])` over
  the topic pairs with at least `hodge_min_flow` citations. Higher = more upstream (drawn on by others). Centred
  to a volume-weighted mean of 0; null for topics without such pairs.
* **Hierarchy**: the share of the (flow-weighted) log-rate asymmetries that the potentials explain (`gradient`,
  0–1), the share that circulates in triangles (`curl`), and the rest (`harmonic`, circulation around holes).
  1 = knowledge flows strictly from upstream to downstream topics.
* **Lag** (main resolution): the median age of the cited works (citing year minus cited year) of the flow from
  i to j. Growing literatures make recent work look more cited; compare lags within a window, not across.
* **Knowledge-base surprise** of a paper: the mean over its resolved references of
  `−log [P(q | p) / P(q | an average citing paper of the year)]` under the global fit. Positive: the paper cites
  works its topics rarely draw on.

## Bundle file `citations.json`

```json
{"resolution": "main", "k": 20, "window": 5, "unit": "citations", "self_citations": "exclude",
 "title_only": true, "top_topics": 6,
 "flow_columns": ["from", "to", "citations", "rate", "lift", "lift_lo", "lift_hi", "lag"],
 "coverage": {"references": 4100000, "resolved": 2300000, "same_corpus": 1500000, "same_corpus_title_only": 90000,
              "other_corpora": 800000, "self_citations": 200000, "cited_later": 5000, "used": 1300000},
 "windows": [
   {"start": 2021, "end": 2025, "fitted": true, "citations": 160000, "citing_papers": 21000,
    "diagonal_share": {"product": 0.31, "model": 0.62},
    "hierarchy": {"gradient": 0.71, "curl": 0.22, "harmonic": 0.07, "gradient_se": 0.03},
    "aging": [0.01, 0.04, ...],
    "topics": {"potential": [...], "potential_se": [...], "cited": [...], "citing": [...], "self": [...],
               "draws_on": [[[3, 0.21], [7, 0.12]], ...], "used_by": [[[5, 0.18]], ...]},
    "flows": [[3, 5, 1234.5, 2.1, 1.8, 1.6, 2.0, 4], ...]}]}
```

* `coverage`: references of the corpus papers in OpenAlex; how many resolve to a citable work; of those, in
  the same corpus (and how many of them title-only), in other corpora; self-citations and citations of works
  published later (version mix-ups), both left out; `used`: citations in the flows of this file.
* `windows`: consecutive windows of `window` citing years (5 by default), the last ending with the corpus's last
  year. Earlier years that do not fill a window are cited but not citing. Windows with fewer than
  `min_citations` citations have `"fitted": false` and nothing else.
* `diagonal_share`: share of citations within one topic, with product attribution and in the model.
* `aging`: g by lag (0 = same year), first 41 lags, summing to 1 over all lags.
* `topics.*`: arrays indexed by topic id. `cited` / `citing`: flow received from / made by the topic within
  the corpus; `self`: within the topic. `draws_on[j]`: the five topics topic j draws on most, with their share of
  j's knowledge base (every citing paper's references weighted to sum 1; own topic left out of the list, not the
  denominator). `used_by[i]`: the five topics that cite topic i most, with their share of i's received flow.
* `flows`: the `top_flows` largest flows between different topics, columns as in `flow_columns`. `lift_lo`,
  `lift_hi`: 5th and 95th percentile over the bootstrap (main resolution), `lag` in years (null with < 20
  citations).

## Global file `_global/citations.json`

All corpora together on one topic index (main resolution, the `--global-window` of `tst citations`, 5 years by
default):

```json
{"window": 5, "fields": [{"name": "philosophy", "label": "Philosophy", "group": "Arts and Humanities", "k": 20}],
 "field_flow_columns": ["from", "to", "citations", "lift"],
 "channel_columns": ["from_field", "from_topic", "to_field", "to_topic", "citations", "lift", "similarity", "twin"],
 "twin_similarity": 0.5,
 "windows": [{"start": 2021, "end": 2025, "fitted": true, "citations": 9.1e7, "citing_papers": 1.2e7,
              "diagonal_share": {...}, "hierarchy": {"gradient": 0.55},
              "field_hierarchy": {"gradient": 0.8, "curl": 0.15, "harmonic": 0.05},
              "field_potential": [...], "field_flows": [[0, 3, 12345.0, 1.4], ...],
              "potential": [[...topics of field 0...], ...], "channels": [[0, 4, 3, 2, 812.0, 5.2, 0.31, false], ...]}]}
```

* Field numbers are positions in `fields`. `field_flows`: flow summed over the topics of both fields, lift
  against the summed null expectation. `field_potential` and `field_hierarchy`: the Hodge decomposition of the
  field-level rates (flow over exposure, both summed per field pair).
* `potential[f][t]`: upstream potential of topic t of field f on the global index.
* `channels`: the 300 largest flows between topics of different fields. `similarity`: cosine similarity of the
  two topics' word weights (over the union of the corpora's vocabularies). `twin`: the two topics are each
  other's most similar topic in the other corpus and their similarity is at least `twin_similarity` (config
  `citations.twin_similarity`): often one subject listed in two maps (corpora overlap) rather than knowledge
  moving between subjects. Read columns by `channel_columns`; files before the twin flag lack the last two.

## documents.parquet columns

| column | type | meaning |
|---|---|---|
| references | int32 \| null | references of the paper in OpenAlex |
| references_resolved | int64 \| null | of those, citations used in the global fit (citable, no self-citation) |
| kb_surprise | float32 \| null | knowledge-base surprise (null with fewer than `min_refs_surprise` references) |
| kb_surprise_pct | float32 \| null | its percentile within the corpus and publication year (0–100) |
