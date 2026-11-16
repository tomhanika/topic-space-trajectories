# Across corpora

Part of the bundle schema 1.4. `tst collect` writes these next to the bundles and lists them in `index.json`
under `global`. Each file carries a `method` object (`description`, `limits`, settings, `references` as ids of
the bundles' reference lists) to show with the views.

## Term explorer: `_terms/`

How often a word or two-word phrase is used, per corpus and year.

* `_terms/index.json`: `{"fields": [names], "years": "years.parquet", "files": {"ab": "ab.parquet", ...,
  "_other": "_other.parquet"}, "terms": 1234567, "rows": ..., "method": {...}}`. A term is in the file of its
  first two letters (`a`–`z`), other terms in `_other`.
* `_terms/<prefix>.parquet`: `term` (string; phrases have one space), `field` (corpus name), `year` (int16),
  `papers` (int32): papers of the corpus that year using the term. Sorted by term, field, year, in row groups of
  20,000 rows with min/max statistics, so a reader can fetch only the row groups whose term range can hold a
  typed prefix. Zero counts are not stored.
* `_terms/years.parquet`: `field`, `year`, `papers`: all papers of the corpus that year. Share = papers using the
  term / papers.

For autocompletion, load the file of the first two typed letters and list its terms by total papers.

## Researchers moving between corpora: `_global/mobility.json`

```json
{"method": {"description": "...", "limits": "...", "period_years": 5, "min_papers": 2, "references": ["openalex"]},
 "periods": [{"start": 1961, "end": 1965}, ..., {"start": 2021, "end": 2025}],
 "fields": ["agriculture", ...],
 "transitions": [{"from_period": 11, "to_period": 12,
                  "flows": [{"from": "physics", "to": "materials", "authors": 812}, ...]}, ...],
 "fields_by_period": [{"period": 12, "field": "physics", "authors": 90123, "stayed": 51234, "arrived": 4321,
                       "departed": 3987, "entered": 34568, "left": 20123}, ...],
 "multi_corpus": [{"period": 12, "authors": 4567890, "share": 0.21}, ...]}
```

* Periods are 5-year blocks ending with the last year most corpora have; indices into `periods`
  (chronological).
* The corpus `misc` (works without an OpenAlex topic) is no research field and is left out.
* An author's **home** corpus in a period is the corpus of most of their papers in it; authors with fewer than
  `min_papers` papers or a tie have none.
* `transitions`: authors with a home in two consecutive periods, from the home in the earlier one to the home in
  the later one; `from == to` are the authors who stayed.
* `fields_by_period`, for corpus F and period P, all about the change since the period before: `authors` with
  home F in P; `stayed` (home F before), `arrived` (home elsewhere before), `entered` (no home before: first
  papers or too few); `departed` (home F before, elsewhere now) and `left` (home F before, none now). For the
  first period only `authors` is meaningful.
* `multi_corpus`: share of authors (with at least `min_papers` papers in the period) who published in two or
  more corpora.

OpenAlex author ids come from automatic disambiguation; see `method.limits`.
