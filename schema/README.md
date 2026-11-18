# Visualization bundle, schema 1.5

`tst run` writes one self-contained folder `<workdir>/bundle/` per corpus. A web app needs only these files
and no code from this package. This folder describes the bundle, one file per data modality:

| file | modality | bundle files |
|---|---|---|
| [topics.md](topics.md) | topics and their trends over time | `topics.json`, `topic_trends.json`, `topic_trends.parquet` |
| [documents.md](documents.md) | papers as points on the map | `documents.parquet` |
| [map.md](map.md) | the map background (topic density) | `density.json`, `density.png`, `density.npz` |
| [trajectories.md](trajectories.md) | venues and authors moving through topic space | `trajectories.parquet`, `trajectories_<type>.json`, `entities.parquet`, `pairwise.parquet` |
| [network.md](network.md) | the topic collaboration graph of authors | `network/*` |
| [derived.md](derived.md) | topic pages, field indicators per year, paper novelty and citation percentiles | `topic_pages.json`, `field_indicators.json`, columns of `documents.parquet` |
| [citations.md](citations.md) | citation flow between topics: flows, rates, lift, upstream potentials, hierarchy; between fields; knowledge-base surprise of papers | `citations.json`, `_global/citations.json`, columns of `documents.parquet` |
| [global.md](global.md) | across corpora, next to the bundles: term explorer, researchers moving between corpora | `_terms/*`, `_global/mobility.json` |
| [methods.md](methods.md) | how the bundle was made: parameters with reasons and literature, cleaning and vocabulary | `manifest.json` (`run.parameters`, `run.references`), `methods.json`, `vocabulary.json` |

## Formats

Small files are JSON. Large tables are Parquet. Browsers read Parquet with `duckdb-wasm`,
`parquet-wasm` + `apache-arrow`, or `hyparquet`. Parquet strings are `large_string` and lists are
`large_list`, which Arrow JS handles. A precomputed static site can also convert the files at build time.

## Conventions (all files)

* **Topics** are integers `0..k-1`; `k` is in `manifest.json` and differs per corpus (24–128).
* A **topic vector** (`theta`, `centroid`, `prevalence`) is a list of `k` non-negative floats that sum to 1.
* **Distances** are Hellinger distances between topic vectors, in `[0, 1]`.
* **Map coordinates** `x`, `y` share one coordinate system across documents, trajectory points and the
  density background of one bundle. The scale is arbitrary (UMAP units); fit the view to the density
  bounds. Coordinates of different bundles are unrelated.
* **Entities** are identified by `entity_type` (`venue` or `author`) plus `entity_id` (a string).
  Venues use their name as id. Authors use the source database's author id as a string (OpenAlex: e.g.
  `"5000051029"`; CSV input: the integer from the `author_ids` column, or a negative number within the
  corpus for other ids such as ORCIDs); authors without an id use `"name:<name>"`.
* **Years** are integers. Missing values are `null`.
* Integer column widths vary between files (int16/int32/int64); treat them all as numbers.

## manifest.json

Every bundle starts here; read it first.

```json
{
  "schema_version": "1.5",
  "generator": "topic-space-trajectories 0.1.0",
  "created": "2026-10-29T22:59:36",
  "k": 24,
  "resolution": "main",
  "label": "Medicine: surgery",
  "group": "Health Sciences",
  "distance": "hellinger",
  "files": {"topics": "topics.json", "documents": "documents.parquet", "network/flows": "network/flows.parquet", ...},
  "run": {"name": "eng_aerospace", "input": "...", "stages": {"<stage>": {<statistics>}},
          "parameters": [...], "references": [...], "config": {...}}
}
```

* `files` maps a logical name to a path relative to the bundle. **Only listed files exist.** For example,
  `trajectories_author` is missing when a corpus has more than 5000 authors, and a run without a network
  stage has no `network/*` keys.
* `resolution` is `main` or `fine`. The two bundles of a run share documents and their ids, but topic
  ids, map coordinates and trajectories are unrelated between them.
* `label` and `group` (schema 1.4, may be null): display name of the corpus and the group it belongs to in a
  list of corpora (OpenAlex domains, with engineering and computer science and the arts and humanities as
  groups of their own).
* `run.stages` holds per-stage statistics (document counts, chosen `k`, model scores, ...). They are useful
  for an "about this corpus" panel, but their keys are not part of the stable schema.
* `run.parameters` describes how the bundle was made, ready to display ("About this map"): a list of groups
  (`Corpus`, `Text cleaning`, `Vocabulary`, `Topic model`, `Trajectories`, `Map`, `Collaboration network` if
  there is a network, `Derived measures`), each with a list of parameters. Every parameter is one decision: `label`, `value`
  (display string), `description` (what it does), `rationale` (why), `references` (literature, ids into
  `run.references`), `note` (why this corpus deviates), `details` (where `methods.json` has the full list) and
  `config`, the config keys it comes from. Show them in the given order; labels and groups may grow over time.
  See [methods.md](methods.md).

  ```json
  {"group": "Trajectories", "parameters": [
    {"label": "Authors", "value": "at least 3 papers in a year, in at least 3 years (19,522 authors)",
     "description": "An author gets a trajectory point in each year with 3+ papers, ... Papers with more than 50 authors are left out.",
     "rationale": "A yearly position needs enough papers to be reliable; ... (D'Angelo & Abramo 2015; Piro et al. 2013) ...",
     "references": ["schaefermeier2021", "dangelo2015", "piro2013"], "note": "", "details": null,
     "config": ["trajectories.entities.author"]}, ...]}
  ```
* `run.references` lists the literature cited by the parameters: `id`, `citation` ("Greene et al. 2014"),
  `authors`, `year`, `title`, `venue`, `url` (DOI link or null). See [methods.md](methods.md).
* `run.config` is the complete configuration of the run, with all defaults, as nested JSON (file paths are
  reduced to file names). It is for lookups and reproduction; its keys follow `src/tst/config.py`.

## Bundle folder for web apps

`tst collect <out> configs/*.yaml` gathers the bundles of several runs in one folder:

```
<out>/index.json
<out>/philosophy/          main resolution, a complete bundle (manifest.json, topics.json, ...)
<out>/philosophy_fine/     fine resolution of the same corpus
<out>/math/ ...
```

`index.json` lists the fields and their resolutions:

```json
{"schema_version": "1.5", "fields": [
  {"name": "philosophy", "label": "Philosophy", "group": "Arts and Humanities",
   "resolutions": {"main": {"path": "philosophy", "k": 32, "created": "2026-11-02T05:26:21"},
                   "fine": {"path": "philosophy_fine", "k": 40, "created": "..."}}}],
 "global": {"terms": "_terms/index.json", "mobility": "_global/mobility.json"}}
```

`path` is relative to `index.json`. The folders are symlinks to the run folders (`--copy` copies them).
`label` and `group` come from the bundles' manifests (missing if not set). `global` lists the outputs across
corpora that `tst collect` writes next to the bundles ([global.md](global.md)); `--no-global` skips them.

## Corpora

A run writes its bundle to `<workdir>/bundle/`, and a fine topic resolution (`fine.enabled`) a second one to
`<workdir>/fine/bundle/`. Both describe the same documents with different topics; `manifest.json`
says which is which (`resolution`). The 46 configs in `configs/` cover all of OpenAlex (CC0): every OpenAlex
subfield is in exactly one corpus, and `misc` holds the works without an OpenAlex topic. Large fields are split
by subfield (`med_*`, `cs_*`, `eng_*`); each manifest has a display `label` and `group`. They end with the last full year before the snapshot;
incomplete trailing years are cut off. Their documents have a `source_id` and link to openalex.org.

Bundles of your own data (CSV input) have a `source_id` only if you mapped one, and links only if
`export.url_template` is set.

## Versioning

Additive changes (new files, new columns) keep the major version. Renamed or removed fields raise
`schema_version` to 2.0. Web apps should check `schema_version` and ignore unknown fields.

* **1.5** (2026-11-17): citation flow between topics (`citations.json`, `_global/citations.json`,
  [citations.md](citations.md)), `documents.parquet` columns `references`, `references_resolved`, `kb_surprise`,
  `kb_surprise_pct`; `index.json` `global.citations`. 2026-11-18, additive: channel columns `similarity` and
  `twin` in `_global/citations.json`.
* **1.4** (2026-11-07): topic names (`topics.json` `name`, `terms_label`, `parent` in fine resolutions),
  `topic_pages.json`, `field_indicators.json`, `documents.parquet` columns `novelty`, `novelty_pct`, `unusual`,
  `citation_pct`, `sna.parquet` column `avg_shortest_path`, manifest `label` and `group`, parameter group
  `Derived measures`, `methods.json` `derived`; `index.json` `label`, `group` and `global` with the term explorer
  and researcher mobility across corpora ([global.md](global.md)).
* **1.3** (2026-11-07): `methods.json` `cleaning.papers.titles_ignored` and `cleaning.foreign_titles`. Titles
  in other languages over English abstracts are left out, "science fiction" is one word, and OpenAlex is covered
  completely (46 fields instead of 14, `misc` for works without an OpenAlex topic), so topics differ from 1.2.
* **1.2** (2026-11-05): `methods.json` `cleaning.joined_phrases` and `cleaning.words`; new parameter
  `Apostrophes` in `Text cleaning`. Text cleaning changed (old database formatting codes, other scripts, accents,
  apostrophes), so topics differ from 1.1 bundles.
* **1.1** (2026-11-05): `methods.json`, `vocabulary.json`, `run.references`; parameters gain `rationale`,
  `references`, `note` and `details`; new parameter groups `Text cleaning` and `Vocabulary` (the `Text`
  parameter moved from `Topic model` to `Text cleaning`, `Removed` was split into one parameter per filter).
* **1.0**: first version.
