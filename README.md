# topic-space-trajectories (`tst`)

`tst` turns a set of publications (year, title, abstract, and if available venue and authors) into

* an NMF **topic model**, with an automatic choice of the number of topics, optionally in a second,
  finer resolution,
* **topic space trajectories** of venues and authors, with trajectory measures and lead/lag analysis,
* a 2D **topic map** of papers and trajectories,
* the **topic collaboration graph**, co-authorship oriented by topic expertise, and its analyses:
  intertopic flows, PageRank, SNA measures, communities and coreness,
* a **visualization bundle**: plain JSON/Parquet files for web apps, specified in [schema/](schema/README.md),
  one file per data modality.

The methods are described in two papers (see [Citation](#citation)).

## Data

There are two ways to bring in data:

* **Your own publication data**, e.g. a Scopus or Web of Science export, or any table you build yourself:
  a CSV or TSV file with one paper per row. The [manual](docs/MANUAL.md) says how to format it, with
  ready-made column mappings for common databases, and walks through every analysis. A minimal file:

  ```csv
  year,title,abstract,venue,authors
  2019,On graphs,"We study graphs ...",J. Graphs,"Doe, J.; Roe, R."
  ```

* **Whole fields of science from [OpenAlex](https://openalex.org)** (CC0). `tst openalex-fetch <store>`
  downloads the needed columns of the works snapshot (about 400 GB transferred, 140 GB stored; resumable).
  After that, any OpenAlex field or subfield is a config away (`input.format: openalex`,
  `input.openalex.subfields: [1211]`). The 46 configs in `configs/` cover all of OpenAlex: every subfield is
  in exactly one map, and `misc` holds the works without an OpenAlex topic (computed, but not published: its
  topics are spam, boilerplate and front matter). Large fields are split by subfield
  (engineering into 5 maps, medicine into 7, computer science into 4); small subfields join a related map.
  Field and subfield ids with their work counts are in `configs/openalex_subfields.csv`. The configs expect the store next to the
  repository folder (`tst openalex-fetch ../openalex`); otherwise change their `input.path`.

## Install

Python ≥ 3.11:

```bash
git clone https://github.com/tomhanika/topic-space-trajectories.git tst
cd tst
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'   # extras: dev (tests), tsne (openTSNE for t-SNE maps)
```

## Use

```bash
tst run examples/my_corpus.yaml                  # full pipeline; reruns skip finished stages
tst run configs/philosophy.yaml --stages trajectories,project,export --force
tst status configs/philosophy.yaml
tst stages                                       # list the stages and their config sections
tst embed-text runs/philosophy/model "We discuss Kant's ethics ..."
tst collect bundle configs/*.yaml                # bundle/<name>, bundle/<name>_fine, index.json for a web app
```

One YAML file describes one run. Every option and its default is in `src/tst/config.py`.
`examples/my_corpus.yaml` is a commented template for your own data, and `configs/` has the OpenAlex
fields. Relative paths in a config are resolved against the config's folder.

### Stages

| stage | output in `workdir/` | notes |
|---|---|---|
| ingest | `documents.parquet` | readers: `csv` (your data; any column names, folders of export files, `.gz`/`.bz2`) and `openalex` (local snapshot store) |
| preprocess | `tokens.parquet` | one tokenizer for training *and* embedding. Drops papers with a missing abstract, spam (regex templates), boilerplate (abstracts occurring ≥ 5 times), duplicate abstracts, too-short texts and non-English texts (stop-word share plus a vote against 7 other languages). Foreign function words are stop words |
| vectorize | `tfidf.npz`, `vectorizer/` | plural and British/American variants merged (only forms that both occur; the most frequent names the word), generic words of academic prose removed (`study`, `results`, `propose`; `src/tst/topics/academic_stopwords.yaml`), vocabulary filtered by min/max document frequency; sublinear TF-IDF; optional bigrams |
| train | `model/` | NMF (NNDSVDa init, coordinate descent) for each k of the grid, scored by NPMI coherence, stability and diversity; `sweep.csv` holds the table. Skipped if `topics.model_path` is set |
| embed | `doc_topics.parquet` | non-negative least squares against the fixed topics; deterministic, also for new documents. Only documents within `period` are kept, and all later stages use them |
| trajectories | `trajectories/` | yearly centroids, smoothing (`centroid_mean`, `pooled`, `cumulative`), point and entity measures, pairwise lead/lag, corpus topic trends |
| project | `projection/` | UMAP (default), t-SNE or PCA on Hellinger-transformed topic vectors; the projector is saved |
| network | `network/` | topic collaboration graph as Parquet edge lists |
| analyze | `analysis/` | per-year flows, PageRank, SNA measures, Leiden communities and coreness; years run in parallel |
| export | `bundle/` | see [schema/](schema/README.md); `manifest.json` also records the settings of the run, ready to display |

### The model bundle

`model/` is self-contained: the tokenizer settings, the vocabulary and IDF, the topic-term matrix and the
topic labels, stored as JSON and `.npy` files. Use it from Python:

```python
from tst.topics import TopicModel
m = TopicModel.load("runs/philosophy/model")
theta = m.transform_texts(["some abstract ..."])   # rows sum to 1
```

`topics.model_path` maps another corpus into the topic space of an existing model.

### Two resolutions

With `fine.enabled: true`, `tst run` also builds a finer topic resolution in `<workdir>/fine`: the
largest k of the grid that is still stable (`topics.min_stability`), or `fine.k`. It shares the
documents, tokens, TF-IDF matrix and k sweep with the main run, so it costs no training; all stages
from train on run again and write their own `fine/bundle/`. `--resolution main|fine` runs only one.
Topic names for it go into `<labels_file stem>_fine<suffix>`.

To name the topics, write one label per line, in topic order, and set `export.labels_file`. Labels only
affect the bundle, so renaming reruns the export only.

## Method choices

* **NMF on TF-IDF** is the established choice for short scientific texts. We use scikit-learn's
  coordinate-descent NMF with NNDSVDa initialisation, which is deterministic. `MiniBatchNMF` is
  available for very large corpora (`topics.algorithm: minibatch`).
* **Author trajectories** leave out papers with more than 50 authors
  (`trajectories.entities.author.max_authors`), so that members of large collaborations do not count as
  prolific authors.
* **Number of topics**: for each k we fit a reference model and `runs` models on random 80 % subsamples.
  Each k gets three scores. *Stability* is the Hungarian-matched average Jaccard of the top terms
  (Greene et al. 2014). *NPMI coherence* (Lau et al. 2014) is computed on a held-out corpus sample.
  *Topic diversity* is from Dieng et al. 2020. The default `selection: stable_coherence` takes the most
  coherent k among those with stability ≥ `min_stability` (0.7). Check `model/sweep.csv`, and fix
  `topics.k` if you prefer another. All sweep components are kept in `model/sweep_components.npz`, and the
  sweep is cached: changing only the selection settings or `k` within the grid re-selects in seconds.
* **Embedding** solves non-negative least squares against the fixed topics, so the same text always
  gets the same topic vector.
* **Maps** treat topic vectors as distributions: their square roots make Euclidean distance equal to
  Hellinger distance before UMAP/t-SNE.
* **Topic collaboration graph**: each paper links its authors in its top-k topics; an author's papers in a
  topic within a time window are their expertise, and edges point from higher to lower expertise.
  Self-pairs are kept as the `expertise` table, not as graph edges, so they don't distort degrees and
  PageRank. PageRank is weighted (`analysis.pagerank.weighted`).
* **Graph analyses** use igraph, with Leiden community detection by default.

## Data quality notes

These apply to OpenAlex, and the filters work on any corpus:

* 8–35 % of the works in a field have no English abstract (fewest in engineering), and some abstracts
  are bilingual. The language vote drops non-English works, and
  `preprocess.language_filter.sentence_filter` keeps only the English sentences of bilingual abstracts.
  Some databases strip accents ("nao", "tambem"); the language lists include both spellings.
* Abstracts in other scripts (Japanese, Chinese, Korean, Russian, Arabic) often come with an English title, which
  the function-word vote cannot catch: `preprocess.language_filter.max_other_script` removes abstracts whose
  letters are mostly in other scripts, and `preprocess.latin_only` keeps such words out of the vocabulary (they
  took 3–17 % of it). Names are spelled with and without accents ("Schrödinger", "Schrodinger";
  `preprocess.fold_accents`), and transliterations contain apostrophes ("da'wah"; `preprocess.join_apostrophes`).
* Many English abstracts carry a title in another language: about 10 % in business, 7–9 % in philosophy,
  political science and geography, under 1 % in physics; mostly Indonesian, Spanish, Portuguese, French and
  German. Their words ("kinerja", "pelanggan") ended up in the topics next to their English translations.
  `preprocess.language_filter.ignore_foreign_titles` leaves such titles out and keeps the paper.
* Some "abstracts" are not abstracts:
  * scraped landing pages and repository cover pages;
  * grey literature: legal opinions, audit letters and press releases, catalogue and archival-database
    stubs;
  * the citation of the work itself ("…, Volume 46, Issue 1, 1 January 1970, Pages 170–171");
  * love-spell, SEO and generated lottery spam;
  * generated dataset descriptions (Materials Project) and supplementary files deposited as separate
    works;
  * placeholders such as "[Formula: see text]";
  * copyright and licence notices ("© 2010 Elsevier Ltd. All rights reserved.", Creative Commons
    statements) and the headings of structured abstracts ("Background:", "Methods:");
  * abstracting-service marks such as "(auth)" (OSTI/INIS).

  The default `preprocess.drop_patterns` and `strip_patterns` remove those seen so far. The tokenizer removes
  the formatting codes of records converted from old databases (OSTI/INIS, IEEE until about 2006: "He/sup 4/",
  "E{sub p}", "/spl mu/m", "{minus}"; 3 % of physics abstracts, which formed a topic).
  Check the topics of every new corpus for more (see the [manual](docs/MANUAL.md#check-and-improve-the-topics)).
* Publication years run into the future (typos, announced issues), and the current year is incomplete.
  `period.max_year: auto` cuts trailing years with few works, and `period.drop_snapshot_year` drops the
  snapshot year.

## Performance notes

* All heavy steps run in parallel (`n_jobs`, default all cores). Forked workers share the large matrices,
  and per-process thread limits prevent oversubscription.
* scikit-learn's NMF solver is single-threaded. The sweep runs all fits in parallel, so its wall time is
  roughly that of the largest k. On a 272-thread Xeon Phi (slow cores), a field of 0.5 M papers takes
  1 h, and one of 8 M papers 6 h. The [manual](docs/MANUAL.md#run) has a table.
* UMAP's first run compiles numba code for several minutes; the compiled code is cached in
  `~/.cache/tst-numba`. t-SNE is mostly serial, so prefer UMAP for large corpora.
* Hyper-authored papers create quadratically many author pairs: `network.max_authors_per_paper`
  (default 50) skips them.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Unit tests cover hand-built cases: CSV and OpenAlex reading, junk filters, smoothing modes, trajectory
measures, lead/lag, NNLS, stability, network orientation and windows, and flows. One end-to-end test runs
on a synthetic two-topic corpus.

## Citation

If you use `tst`, please cite the papers that introduced its methods (also in [CITATION.cff](CITATION.cff)):

* B. Schäfermeier, G. Stumme, T. Hanika: *Topic space trajectories: A case study on machine learning
  literature.* Scientometrics 126, 5759–5795 (2021). <https://doi.org/10.1007/s11192-021-03931-0>
* B. Schäfermeier, J. Hirth, T. Hanika: *Research topic flows in co-authorship networks.*
  Scientometrics 128, 5051–5078 (2023). <https://doi.org/10.1007/s11192-022-04529-w>

## License

`tst` is licensed under the [GNU Affero General Public License v3.0](LICENSE). If you run a modified
version as a network service, you must offer its source code to the service's users.
