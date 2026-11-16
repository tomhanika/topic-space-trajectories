# Topics

Part of the bundle schema 1.0 (see [README.md](README.md) for conventions). A corpus has `k` topics,
learned by NMF on the abstracts. Each topic is a weighted list of terms.

## topics.json

A list ordered by `id`:

| field | type | |
|---|---|---|
| id | int | `0..k-1` |
| label | string | what to show: `name` if the topic has one, otherwise `terms_label` |
| name | string \| null | the topic's name (schema 1.4), e.g. `"Philosophy of religion"`; null if the topics are unnamed. Names are interpretations: show them with the top terms (see `run.parameters`, Derived measures, Topic names, for how they were made) |
| terms_label | string | the top three terms (`"model, method, analysis"`, schema 1.4) |
| manual_label | bool | whether the topics have names (`name` is set) |
| parent | `{id: int, cosine: float}` | only in a fine resolution (schema 1.4): the topic of the main resolution with the most similar term weights, for nesting fine topics under main ones |
| top_terms | list of `{term: string, weight: float}` | 20 terms, best first; weights are normalised per topic. Bigrams join with `_` (`"machine_learning"`) |
| color | string | `#rrggbb`, a stable categorical colour per topic. Use it everywhere the topic appears so that the map, legends and charts agree |
| prevalence | float | share of the topic over the whole corpus; the shares of all topics sum to 1 |

```json
{"id": 0, "label": "model, method, analysis", "manual_label": false,
 "top_terms": [{"term": "model", "weight": 0.02341}, {"term": "method", "weight": 0.010509}, ...],
 "color": "#1f77b4", "prevalence": 0.061}
```

Some topics are generic (methods, "this book", "this study") rather than subject areas. Names exist only
where a run's `export.labels_file` exists and still fits the model; otherwise `label` is the top terms. More
facts per topic are in `topic_pages.json` ([derived.md](derived.md)).

## topic_trends.json

Corpus-wide topic shares per year, e.g. for stream graphs or area charts.

```json
{"years": [1960, 1961, ...], "n_docs": [3140, 3301, ...], "prevalence": [[float × k], ...]}
```

`prevalence[i]` belongs to `years[i]` and sums to 1. `n_docs[i]` is the number of documents that year.
Early years have few documents, so their shares are noisy; consider hiding years below a minimum
`n_docs`, or weighting by it.

## topic_trends.parquet

The same data as a table: `year` int32, `n_docs` int64, `prevalence` fixed_size_list<float32, k>.
