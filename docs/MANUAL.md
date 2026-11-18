# Manual

This manual shows how to get all analyses of `tst` from your own publication data: a topic model of the
abstracts, topic space trajectories of venues and authors, a map of the field, and the topic
collaboration graph of the authors. The [README](../README.md) explains the methods; [schema/](../schema/README.md)
specifies every output column.

1. [What you get](#what-you-get)
2. [Install](#install)
3. [Your data](#your-data)
4. [The config file](#the-config-file)
5. [Run](#run)
6. [Results](#results)
7. [Working with the results in Python](#working-with-the-results-in-python)
8. [Check and improve the topics](#check-and-improve-the-topics)
9. [Whole fields from OpenAlex](#whole-fields-from-openalex)
10. [Troubleshooting](#troubleshooting)

## What you get

From a table of papers (year, title, abstract, and if available venue and authors), one run produces:

| analysis | question it answers | result |
|---|---|---|
| topic model | which topics does the corpus consist of? | `model/`, `bundle/topics.json` |
| topic trends | which topics grow or shrink? | `bundle/topic_trends.parquet` |
| document topics | what is each paper about? | `doc_topics.parquet` |
| map | how do topics and papers relate? | `bundle/documents.parquet`, `bundle/density.*` |
| venue / author trajectories | how does the topic profile of a journal or an author change over time? | `bundle/trajectories.parquet` |
| trajectory measures | who moved most, who is broad, who is unusual? | `bundle/entities.parquet` |
| lead / lag | which venues anticipate the topics of others? | `bundle/pairwise.parquet` |
| topic collaboration graph | who collaborates with whom, oriented by topic expertise | `network/` |
| intertopic flows | how much does expertise flow from one topic into another? | `bundle/network/flows.parquet` |
| central authors | who are the experts per topic and year? | `bundle/network/pagerank.parquet` |
| graph statistics | how dense and clustered is collaboration per topic? | `bundle/network/sna.parquet` |
| communities | how does the collaboration graph split into groups? | `bundle/network/communities.parquet` |

Without author data you get everything except the author trajectories and the network analyses; without
venues, everything except the venue trajectories and lead/lag.

## Install

You need Python ≥ 3.11 and git.

```bash
git clone https://github.com/tomhanika/topic-space-trajectories.git tst
cd tst
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/tst --help
```

Optional extras: `.[tsne]` for t-SNE maps, `.[dev]` for the tests (`.venv/bin/python -m pytest -q`).

Hardware: a laptop handles tens of thousands of papers in minutes. A million papers needs a
many-core machine with ≥ 64 GB RAM and takes hours (see [Run](#run)).

## Your data

`tst` reads your publications as a table: a CSV or TSV file with a header row and one paper per row.
A folder of such files with the same header also works, e.g. several export files. Plain, `.gz` and
`.bz2` files are read. A minimal file:

```csv
year,title,abstract,venue,authors,author_ids
2019,On graphs,"We study graphs, their colourings and ...",Journal of Graphs,"Doe, Jane; Roe, Rick",0000-0001-2345-6789; 0000-0002-3456-7890
2021,More graphs,"Building on earlier work, we ...",Journal of Graphs,"Doe, Jane",0000-0001-2345-6789
```

Format rules:

* **Header row**: the first line names the columns. The names may differ from the ones below; you map
  them in the config ([below](#exports-of-bibliographic-databases)).
* **Separator**: comma by default; set `input.csv.separator` for tab (`"\t"`) or semicolon (`";"`) files.
* **Quoting**: a cell that contains the separator, a quote or a line break is enclosed in double quotes,
  and quotes inside it are doubled (`""`). Spreadsheet programs and `pandas`/`polars` write this
  automatically.
* **Lists**: several authors in one cell are separated by `;` (`input.csv.list_separator`), so names
  may contain commas ("Doe, Jane"). `author_ids` lists the ids in the same order and with the same
  separator.
* **Encoding**: UTF-8 (with or without byte order mark). Other encodings: set `input.encoding`, e.g.
  `latin-1`.
* **Empty cells** count as missing. Every column except `year` and `title` may be missing or empty.

The columns:

| column | required | content |
|---|---|---|
| year | yes | publication year; the first four-digit number counts, so `2019-05-01` works |
| title | yes | |
| abstract | in practice | papers without an abstract are skipped (`preprocess.require_abstract`) |
| venue | no | journal or conference name; needed for venue trajectories |
| authors | no | author names separated by `;` (`input.csv.list_separator`); needed for author trajectories and the network |
| author_ids | no | author ids in the same order as the names; without ids, authors are identified by name |
| citations | no | citation count |
| source_id | no | an id or DOI per paper, used for links in the bundle |

Your columns may have other names; you map them in the config. Rows without a year are dropped and
counted in the log. Author ids that are not integers (ORCID, prefixed ids) are numbered within the corpus,
and the names stay as they are.

Author names are only matched exactly. Without author ids, "J. Smith" and "John Smith" are two authors,
and two different J. Smiths are one. If you can, export author ids.

### Exports of bibliographic databases

The column names below are those of the exports at the time of writing; compare them with the header
of your file.

**Scopus** (Export → CSV, with "Abstract" ticked):

```yaml
input:
  path: scopus.csv
  format: csv
  csv:
    columns: {year: Year, title: Title, abstract: Abstract, venue: Source title, authors: Authors,
              author_ids: "Author(s) ID", citations: Cited by, source_id: DOI}
```

**Web of Science** (Export → Tab delimited file, "Full Record"; put all files into one folder):

```yaml
input:
  path: wos_exports/
  format: csv
  csv:
    separator: "\t"
    columns: {year: PY, title: TI, abstract: AB, venue: SO, authors: AU, citations: TC, source_id: DI}
```

**OpenAlex, Dimensions, Lens, your own database**: export or convert to CSV with these columns and map
the names the same way. For whole fields of science, use the OpenAlex snapshot directly
([below](#whole-fields-from-openalex)).

**Your own table in Python**: write it with `polars` or `pandas`:

```python
df.to_csv("papers.csv", index=False)   # pandas; columns year, title, abstract, venue, authors ("A; B")
```

### Language

The topic model is for English text. Non-English abstracts are detected (stop-word vote against seven
other languages) and skipped. Bilingual abstracts keep their English sentences. Abstracts written mostly in
another script (Cyrillic, Chinese, Japanese, Korean, Arabic, ...) are skipped even when the title is English
(`preprocess.language_filter.max_other_script`), and only words in the Latin alphabet become tokens
(`preprocess.latin_only`). Accents are removed (`preprocess.fold_accents`: "Schrödinger" = "Schrodinger"), and
words are joined at apostrophes (`preprocess.join_apostrophes`: "Kant's" → "kant", "da'wah" → "dawah").
A title in another language over an English abstract (common in Indonesian and Latin American journals) is
left out, and the paper is kept with its abstract (`preprocess.language_filter.ignore_foreign_titles`).

## The config file

One YAML file describes one run. Start from [examples/my_corpus.yaml](../examples/my_corpus.yaml):

```yaml
name: my_corpus
workdir: runs/my_corpus
input:
  path: scopus.csv
  format: csv
  csv:
    columns: {year: Year, title: Title, abstract: Abstract, venue: Source title, authors: Authors}
vectorize:
  min_df: 3
topics:
  k_grid: [5, 8, 10, 12, 15, 20, 25]
trajectories:
  entities:
    venue: {min_papers: 2, window: 3, smoothing: pooled, min_years: 5}
    author: {min_papers: 1, window: 3, smoothing: pooled, min_years: 3}
network:
  min_author_papers: 2
```

Relative paths are resolved against the config's folder. Every option, its default and an explanation are
in [src/tst/config.py](../src/tst/config.py); `tst stages` lists which config section each stage reads.
The settings that matter most:

| setting | meaning | small corpus (≤ 50k papers) | large corpus (millions) |
|---|---|---|---|
| `vectorize.min_df` | a word must occur in at least this many abstracts | 3 | 50 |
| `vectorize.max_df` | drop words in more than this share of abstracts | 0.5 | 0.4 |
| `vectorize.keep` / `vectorize.stopwords` | words of the academic stop list to keep / more words to drop | – | – |
| `topics.k_grid` | candidate numbers of topics | `[5, 8, 10, 12, 15, 20, 25]` | `[24, 32, 40, 48, 56, 64, 80]` |
| `topics.k` | fix the number of topics instead | `auto` | `auto` |
| `topics.train_sample` | papers used for training (all papers are embedded) | all | 400000 |
| `trajectories.entities.<type>.min_papers` | papers **per year** for a trajectory point | 1–2 | see below |
| `trajectories.entities.<type>.min_years` | minimum number of points of a trajectory | 3 | 3 |
| `trajectories.entities.<type>.window` | years pooled into one point | 3 | 3 |
| `network.min_author_papers` | papers an author needs to be in the graph | 2 | 5–10 |
| `period.min_year`, `period.max_year` | years shown from the trajectories on | | |
| `fine.enabled` | also build a finer topic resolution in `<workdir>/fine` | | |

Notes:

* **The number of topics** is chosen automatically from `k_grid`: the most coherent k among those whose
  topics are stable under resampling (`topics.selection`, `topics.min_stability`). The table of all
  candidates is in `model/sweep.csv`. If you prefer another k, set `topics.k` to it; if it was in the
  grid, the model is taken from the cached sweep within seconds.
* **Thresholds count per year.** A venue with 30 papers spread over 30 years has one paper per year and
  needs `min_papers: 1`. `smoothing: pooled` averages all papers in the window and suits sparse data.
* **Author thresholds depend on the field.** Publication rates differ several-fold between fields
  ([D'Angelo & Abramo 2015](https://www.issi-society.org/proceedings/issi_2015/0909.pdf);
  [Piro, Aksnes & Rørstad 2013](https://doi.org/10.1002/asi.22746)), mostly through the size of author teams. In
  OpenAlex, the median author has 1 paper per year in every field. A good `min_papers` for authors is
  the 95th percentile of the papers per author and year in your corpus, i.e. the output of an established
  researcher: 2 in philosophy and geography, 3 in the social sciences, psychology, mathematics and most
  engineering, 4 in biology and aerospace engineering, 5 in electrical engineering, 10 in physics (with
  `min_years: 3`). Papers with more than `trajectories.entities.author.max_authors` (50) authors do not
  count for author trajectories, so members of large collaborations do not look prolific.
* **Two resolutions.** `fine.enabled: true` adds a second run in `<workdir>/fine` with the largest
  stable k of the grid above the chosen one (or `fine.k`). It reuses the documents and the k sweep, so it
  needs no training, and writes its own `fine/bundle/`. `tst run my_corpus.yaml --resolution fine`
  builds only it; topic names go into `labels_fine.txt` next to `export.labels_file`.
* **The last year** is dropped automatically when it has less than half the papers of the busiest of the
  five years before (`period.max_year: auto`): exports are incomplete for the current year, which would
  look like a sudden trend. Set `period.max_year` to a year to override.
* **Paper links** in the bundle: `export.url_template`, e.g. `"https://doi.org/{source_id}"` when
  `source_id` holds DOIs; `{title}` inserts the URL-encoded title. Without a template there are no
  links, and papers without a `source_id` get none.
* **Reasons for your own settings**: `export.notes` maps a config key to a sentence, e.g.

  ```yaml
  export:
    notes:
      vectorize.keep: "analysis and number are subjects in mathematics (functional analysis, number theory)."
      topics.k_grid: "Extended to 128: the main resolution chose the largest k of the old grid."
  ```

  The bundle shows each note with the parameter that uses the key (`run.parameters`, `note`), so readers of
  the map see why this corpus differs from the defaults. Only the export stage reruns.

## Run

```bash
tst run my_corpus.yaml                 # the whole pipeline
tst status my_corpus.yaml              # what is done, with statistics per stage
```

The pipeline has ten stages: ingest → preprocess → vectorize → train → embed → trajectories → project
→ network → analyze → export. Each stage stores a fingerprint of its config section, so running `tst run`
again only reruns what changed: new trajectory thresholds rerun the trajectories and everything after
them, but not the topic model. To rerun selected stages explicitly:

```bash
tst run my_corpus.yaml --stages trajectories,project,export
tst run my_corpus.yaml --stages export --force     # rerun even if up to date
tst run my_corpus.yaml --dry-run                   # only list the stages that would run
```

The log is printed and also written to `<workdir>/logs/run.log`. `n_jobs` (default: all cores) limits
the parallelism.

Time and memory, measured on a 272-thread Xeon Phi (slow cores; a desktop CPU is 3–10× faster per
core):

| corpus | papers after filtering | time | of which topic model |
|---|---|---|---|
| 20k-paper Scopus-style CSV | 10k | 15 min | 1 min |
| a field of science (philosophy) | 0.5M | 1 h | 20 min |
| a large field (biology) | 8M | 6 h | 3.5 h |

The topic model trains one model per candidate k and subsample in parallel. Its time grows with the
largest k and with `topics.train_sample`. Maps (UMAP) need a few minutes the first time for compilation
(cached in `~/.cache/tst-numba`).

## Results

Everything is in the `workdir`:

```
runs/my_corpus/
  documents.parquet      all papers in canonical form
  tokens.parquet         tokens, and why papers were skipped (column reason)
  model/                 the topic model (reusable, see below), sweep.csv with the candidate scores
  doc_topics.parquet     topic vector (theta) and main topic of every paper
  trajectories/          trajectories, entity measures, lead/lag, topic trends
  projection/            map coordinates and the fitted map
  network/               the topic collaboration graph as yearly edge lists
  analysis/              flows, PageRank, graph statistics (incl. average shortest path), communities per year
  novelty.parquet        per paper: novelty (unusual topic pairs), citation percentile within the year
  indicators/            field indicators per year, one page of facts per topic
  terms/                 papers per year using each word and frequent phrase (term explorer, main resolution)
  bundle/                everything above in the form for visualization, see schema/
  state.json, logs/      stage fingerprints and statistics, logs
  fine/                  the fine resolution (fine.enabled): model/ to bundle/ again, with its own k
```

The `bundle/` folder is self-contained and documented column by column in [schema/](../schema/README.md).
Start there for analyses too: it has the topic labels and the map positions joined in. It also documents how
it was made ([schema/methods.md](../schema/methods.md)): `manifest.json` lists every parameter with its value,
what it does, why it was chosen and the literature behind it; `methods.json` has the cleaning patterns, the
stop-word list with its categories and the words deliberately kept, and the rules for merging word forms;
`vocabulary.json` has the vocabulary with document frequencies and all merged forms.

### Derived measures

Beyond the map, a run computes (stages `novelty`, `indicators`, `terms`; schema in
[schema/derived.md](../schema/derived.md)):

* **Topic pages**: per topic its life cycle (peak, rise, recent growth), typical and most cited papers, main
  venues and authors, team size, citation impact, novelty, related topics (combined in papers) and similar ones
  (similar words); in the fine resolution the parent topic of the main resolution.
* **Field indicators** per year: effective number of topics, topics per paper, how fast the topic mix changes,
  team size, newcomers, novelty; and whether novel papers are cited more.
* **Novelty** per paper: how unusual its combination of topics is compared with the preceding years (after Lee,
  Walsh & Wang 2015), and its citation percentile within its publication year.
* **Word counts** per year for a term explorer; `tst collect` combines them across corpora and adds researchers
  moving between corpora ([schema/global.md](../schema/global.md)).

Settings are in the `novelty`, `indicators` and `terms` sections of the config. Every measure is described with
its reasons and literature in the bundle (`run.parameters`, group Derived measures).

### Naming the topics

Topics are labelled by their top three words. Better names make every plot readable: look at
`bundle/topics.json` (top terms) and `bundle/topic_pages.json` (typical papers), and write a tab-separated file
with a header line, the topic id, its name and the top five terms you named it by:

```
id	name	terms
0	Philosophy of religion	god, religion, religious, belief, faith
1	Kant	kant, critique, reason, transcendental, pure
```

```yaml
export:
  labels_file: labels.tsv        # the fine resolution reads labels_fine.tsv
```

and rerun the export stage (`tst run cfg.yaml --stages export --force`). If the topic model is retrained, names
whose terms no longer match are dropped instead of being attached to other topics; `scripts/naming/` has the
procedure and tools to name a retrained map again ([GUIDE.md](../scripts/naming/GUIDE.md)). A plain text file with one
name per line in topic order also works, without that check. The bundle shows both name and top terms.

## Working with the results in Python

The examples use `polars`, which `tst` installs; `pandas.read_parquet` reads the same files.

```python
import json
import polars as pl

b = "runs/my_corpus/bundle/"
topics = json.load(open(b + "topics.json"))
labels = [t["label"] for t in topics]

# Topics and their top words
for t in topics:
    print(t["id"], t["label"], [w["term"] for w in t["top_terms"][:8]])

# Topic shares per year (long format)
trends = pl.read_parquet(b + "topic_trends.parquet")

# The papers of one topic, most typical first
docs = pl.read_parquet(b + "documents.parquet")          # a sample, at most export.max_docs papers
docs.filter(pl.col("main_topic") == 3).sort("strength", descending=True).select("year", "title").head(10)

# All papers with their topic vectors (not sampled)
theta = pl.read_parquet("runs/my_corpus/doc_topics.parquet")    # doc_id, main_topic, strength, theta
papers = pl.read_parquet("runs/my_corpus/documents.parquet").join(theta, on="doc_id")

# Venues that changed most, and how broad they are
ent = pl.read_parquet(b + "entities.parquet")
ent.filter(pl.col("entity_type") == "venue").sort("path_length", descending=True).select(
    "entity_name", "papers", "path_length", "straightness", "mean_entropy", "dominant_topic").head(10)

# The trajectory of one venue, year by year
traj = pl.read_parquet(b + "trajectories.parquet")
traj.filter(pl.col("entity_name") == "Mind").select("year", "n_papers", "main_topic", "step", "novelty")

# Who leads whom (lead_score > 0: a leads b)
pl.read_parquet(b + "pairwise.parquet").sort("lead_score", descending=True).head(10)

# Strongest flows of expertise between topics in 2020
flows = pl.read_parquet(b + "network/flows.parquet")
flows.filter((pl.col("year") == 2020) & (pl.col("source_topic") != pl.col("target_topic"))).sort(
    "weight", descending=True).head(10)

# Most central authors per topic in 2020
pr = pl.read_parquet(b + "network/pagerank.parquet")
pr.filter((pl.col("year") == 2020) & (pl.col("rank") <= 3)).sort("topic", "rank")
```

A plot of venue trajectories on the map:

```python
import matplotlib.pyplot as plt

d = docs.sample(min(50_000, docs.height), seed=0)
plt.scatter(d["x"], d["y"], s=1, c=d["main_topic"], cmap="tab20", alpha=0.3)
for name in ent.filter(pl.col("entity_type") == "venue").sort("papers", descending=True)["entity_name"][:5]:
    t = traj.filter(pl.col("entity_name") == name).sort("year")
    plt.plot(t["x"], t["y"], "-o", ms=3, label=name)
plt.legend(); plt.axis("off"); plt.show()
```

### Topic vectors of new texts

The model in `model/` places any text into the same topic space:

```python
from tst.topics import TopicModel

m = TopicModel.load("runs/my_corpus/model")
theta = m.transform_texts(["We study the reception of Kant's ethics ..."])   # rows sum to 1
print(theta.argmax(axis=1), m.label(int(theta.argmax())))
```

or from the shell: `tst embed-text runs/my_corpus/model "We study ..."`.

## Check and improve the topics

After the first run, look at the topics (`bundle/topics.json`, or `model/topics.json` with 20 terms per
topic) and at `model/sweep.csv`.

* **A topic of boilerplate** ("rights reserved, copyright, elsevier", "file, additional, table",
  "abstract, available, pdf"): such texts are in many exports. Remove whole papers by regex with
  `preprocess.drop_patterns`, or only the phrase with `preprocess.strip_patterns`. Both are matched
  case-insensitively; drop patterns see the title and the abstract. The defaults already cover templates
  found in OpenAlex; when you set the option, include them
  (`tst.config.DEFAULT_DROP_PATTERNS`) or replace them deliberately. Junk of one corpus only goes into
  `preprocess.extra_drop_patterns`, by group name, in addition to the defaults; the group name labels the
  patterns in the bundle:

  ```yaml
  preprocess:
    extra_drop_patterns:
      "Irish parliamentary questions": ['^pq-\d+-\d+\b']
  ```

  `scripts/topic_docs.py` shows the venues and typical papers of a topic, and `scripts/pattern_hits.py` counts
  what candidate patterns would remove in every run before you add them.
* **Words that dominate without meaning**: generic words of academic writing ("study, paper, results,
  propose") are removed by default (`vectorize.academic_stopwords`, list in
  [src/tst/topics/academic_stopwords.yaml](../src/tst/topics/academic_stopwords.yaml)). Add your own with
  `vectorize.stopwords`, keep a listed word that is a subject in your field with `vectorize.keep` (the
  mathematics config keeps "analysis" and "number"), or lower `vectorize.max_df`. A stop word removes all
  its variants ("study" also "studies").
* **The same word twice in a topic** ("model, models", "behaviour, behavior"): `vectorize.conflate`
  merges singular and plural (also "genetics/genetic", "spectra/spectrum") and British and American
  spellings, but only forms that both occur in the corpus; the most frequent form names the word. The
  plural rules are English ones: "-es" is removed only after s, x, z, ch, sh and o ("processes", but not
  "licensees" or the Spanish "animales"), and singulars ending in -s such as "bias", "lens" or "menses"
  stay apart. Real stemming is avoided on purpose: it merges unrelated words and does not improve topic models
  (Schofield & Mimno, TACL 2016). Hyphenated prefixes are joined in `preprocess`
  (`join_prefixes`: "non-linear" = "nonlinear", "e-learning" = "elearning").
* **Topics too broad or too fine**: change `topics.k_grid`, or set `topics.k`. The automatic choice
  favours the lowest k when only few k pass the stability threshold; `topics.min_stability: 0.6` admits
  more.
* **Why papers were skipped**: `tokens.parquet` has a `reason` per paper (`ok`, `language`, `too_short`,
  `spam`, `boilerplate`, `duplicate`), and the log shows the counts.

Changes to `preprocess` or `vectorize` retrain the model; changes to `topics` within the grid only
re-select.

## Whole fields from OpenAlex

[OpenAlex](https://openalex.org) is a free (CC0) catalogue of the scholarly literature. `tst` can download
the parts of its snapshot it needs (about 400 GB transferred, 140 GB stored, resumable) and then build
any field or subfield:

```bash
tst openalex-fetch /data/openalex
```

```yaml
input:
  path: /data/openalex
  format: openalex
  openalex:
    subfields: [1211]        # Philosophy; ids and work counts in configs/openalex_subfields.csv
export:
  url_template: "https://openalex.org/{source_id}"
```

`configs/` has 46 configs that together cover every OpenAlex subfield once (`misc`: works without a topic,
`input.openalex.fields: [0]`; all its topics are junk, so `export.publish: false` keeps it out of
`tst collect`). Books, chapters and non-paper types are left out
(`input.openalex.types`), and only journals, conferences and book series count as venues
(`input.openalex.venue_types`). The snapshot year is dropped because it is incomplete. The data quality
notes in the [README](../README.md#data-quality-notes) list the cleaning that OpenAlex needs.

For a few thousand works, e.g. those of an institution, query the OpenAlex API instead, write the results
as CSV and use the `csv` format.

### Citation flow between topics

References cross maps, so the citation flow is computed once for all OpenAlex maps, after their runs:

```bash
tst openalex-fetch /data/openalex --links      # references, authors, titles of all works (about 33 GB)
tst citations configs/*.yaml                   # all steps; --steps fit,global,... to redo some
tst run configs/philosophy.yaml --stages export --force   # for every map: puts citations.json into the bundle
tst collect bundle configs/*.yaml              # adds _global/citations.json
```

A paper cites an earlier work at a rate given by its number of references, an aging kernel over the citation
lag, and a topic affinity between the topic mixes of the two works (from their text, fixed). The affinity is
fitted per window of citing years. From it come the **flow** between topics (citations within one topic are not
split over the topics a paper mixes), **rates** that do not grow with a topic's size, so they show direction,
the **lift** over a random choice of cited works from the same years, an **upstream potential** per topic and a
**hierarchy** index (a Hodge decomposition of the rate asymmetries), and per paper a **knowledge-base surprise**
(how unusual its references are for its topics). Works without an abstract are placed by their title;
citations between papers that share an author are left out. A global fit on the main topics of all maps gives
the flow between fields; channels between twin topics (one subject in two maps) are flagged. Settings are in
the `citations` section of the config, the files in [schema/citations.md](../schema/citations.md). Runtime for
all of OpenAlex on 272 threads: about 8 hours.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `CSV input lacks the columns [...]` | map your column names under `input.csv.columns`; the message lists the columns found |
| one column holds all fields | wrong separator: `input.csv.separator: "\t"` (or `";"`) |
| garbled umlauts or accents | set `input.encoding` (e.g. `latin-1`, `utf-16`) |
| `venue trajectories: 0 entities` | `min_papers` counts per year: lower it, or set `smoothing: pooled` and `min_papers: 1` |
| many papers skipped as `language` | non-English abstracts; the topic model is English only |
| all papers skipped as `boilerplate` | the abstract column holds the same text everywhere (e.g. "[No abstract available]"): map the right column |
| the last year is missing | `period.max_year: auto` dropped an incomplete year; set it explicitly |
| a stage reruns although the config did not change | a new version of `tst` changed a default of that stage's config section |
| out of memory | lower `topics.train_sample`, `projection.fit_sample`, `n_jobs` or `analysis.max_workers` |

Questions and bugs: please open an issue.
