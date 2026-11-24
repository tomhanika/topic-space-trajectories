"""Run configuration. One YAML file describes one corpus run; every stage reads its own section."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Options added after runs existed: left out of the stage fingerprints while they have their default value, so
    # adding an option does not re-run finished stages of every corpus.
    fingerprint_omit_default: ClassVar[tuple[str, ...]] = ()

    def fingerprint_dump(self) -> dict:
        d = self.model_dump(mode="json")
        for key in self.fingerprint_omit_default:
            if getattr(self, key) == type(self).model_fields[key].get_default(call_default_factory=True):
                d.pop(key, None)
        return d


class OpenAlexInputConfig(_Section):
    # Works whose primary topic is in one of these OpenAlex fields or subfields (ids as in
    # https://api.openalex.org/fields, e.g. subfield 1211 = Philosophy). Field 0: works without a primary topic.
    fields: list[int] = Field(default_factory=list)
    subfields: list[int] = Field(default_factory=list)
    # Paper-like work types; leaves out books and chapters, datasets, paratext, errata, editorials, letters etc.
    types: list[str] | None = Field(default_factory=lambda: ["article", "review", "conference-paper", "preprint",
                                                             "dissertation", "report"])
    # Source types that count as venues; works hosted only by a repository (HAL, Zenodo, arXiv, ...) or an
    # ebook platform keep their place in the corpus but get no venue.
    venue_types: list[str] | None = Field(default_factory=lambda: ["journal", "conference", "book series"])
    # OpenAlex's detected language, e.g. ["en"]; the preprocess language filter runs in any case.
    languages: list[str] | None = None


CsvColumn = Literal["year", "title", "abstract", "venue", "authors", "author_ids", "citations", "source_id"]


class CsvInputConfig(_Section):
    # Field separator; "\t" for tab-separated files (e.g. Web of Science exports).
    separator: str = ","
    # Separator of the names in the authors cell and of the ids in the author_ids cell.
    list_separator: str = ";"
    # Column names in the file where they differ from the canonical ones, e.g. {year: PY, title: TI}.
    # Only year and title are required; author_ids must have as many entries as authors.
    columns: dict[CsvColumn, str] = Field(default_factory=dict)


class InputConfig(_Section):
    # openalex: works from a local OpenAlex store written by `tst openalex-fetch`, selected by `openalex`.
    # csv:      your own publication data, a CSV/TSV file with a header row (plain, .gz or .bz2) or a folder
    #           of such files, see `csv` and docs/MANUAL.md.
    format: Literal["openalex", "csv"]
    # For openalex: the store folder. For csv: a file or a folder of files.
    path: Path
    csv: CsvInputConfig = Field(default_factory=CsvInputConfig)
    openalex: OpenAlexInputConfig = Field(default_factory=OpenAlexInputConfig)
    # Encoding of csv files, e.g. "latin-1" for older spreadsheet exports.
    encoding: str = "utf-8"
    limit: int | None = None
    min_year: int | None = None
    max_year: int | None = None
    # Map raw venue names to canonical ones (e.g. {"NIPS": "NeurIPS"}).
    venue_aliases: dict[str, str] = Field(default_factory=dict)


class LanguageFilterConfig(_Section):
    # stopword_share: the share of English stop words among all words must exceed `threshold`.
    # stopword_vote: additionally, English stop words must be at least as frequent as the stop words of
    #   each other language in tst.text.FOREIGN_STOPWORDS. This removes bilingual abstracts whose
    #   main text is not English.
    method: Literal["stopword_vote", "stopword_share", "none"] = "stopword_vote"
    threshold: float = 0.08
    # stopword_vote only: a document whose foreign function words reach this share of its words is cut down
    #   to its mainly English sentences first (bilingual abstracts, e.g. Indonesian + English); 0 = off.
    sentence_filter: float = 0.02
    # A document whose abstract has more than this share of its letters in words of other scripts than Latin
    # (Cyrillic, Chinese, Japanese, Korean, Arabic, ...) is not English, e.g. a Japanese abstract with an English
    # title, which the function-word vote cannot see. Above one half, so that bilingual abstracts stay; 1 = off.
    max_other_script: float = 0.6
    # stopword_vote only: a title with more function words of another language than English is left out when the
    # paper has an abstract (an Indonesian or Spanish title over an English abstract), so that its words do not
    # enter the topics; the paper stays.
    ignore_foreign_titles: bool = True


# Documents dropped entirely, matched case-insensitively against the abstract. Grouped by what they catch; the
# group names label the patterns in the bundle (methods.json). Order and text of the patterns are part of the
# preprocess fingerprint: changing them re-runs everything.
_DROP_GROUPS: dict[str, list[str]] = {
    # Templated SEO spam in bibliographic databases ("Thank you for reading <book>. People have search
    # hundreds times ...").
    "E-book download spam": [
        r"thank you (very much )?for (reading|downloading)",
        r"people have (look|search)(ed)? (hundreds|numerous)",
        r"infectious (downloads|bugs)",
        r"\bdownload\b.{0,60}\b(pdf|epub|ebook|e-book|kindle)\b",
        r"\b(pdf|epub|ebook)\b.{0,60}\bdownload\b",
        r"\bour (digital )?(book|library) (collection|saves)",
        r"download it instantly",
        r"reading resources",
        r"\*free\*",
        r"\bpdf\b.{0,300}\bpdf\b.{0,300}\bpdf\b",
        r"reading is a hobby to open the knowledge windows",
        r"it's not surprisingly when entering this site",
        r"if you are looking for the (e-?book|book)",
    ],
    # Love-spell / astrology spam with phone numbers (OpenAlex).
    "Love-spell and astrology spam": [
        r"call\s*/\s*whatsapp", r"\bwazifa\b", r"\bvashikaran\b", r"black magic specialist",
        r"love problem solution",
    ],
    # Scraped publisher landing pages instead of abstracts (OpenAlex; Springer, Optica, ACS; since 2026-11-08 also
    # Atypon, Wiley and Wolters Kluwer pages: Journal of Urology, Annals of Internal Medicine, SIAM, SEG, ASHA, ...).
    "Scraped publisher landing pages": [
        r"search for other works by this author on", r"\b(views|share|cite|tools) icon\b",
        r"add to citeulike|share with facebook|get pdf email share", r"return to issue\s*prev",
        r"cite this: .{0,80}publication date",
        r"advertisement (journals books )?search menu menu navbar",
        r"add to favou?rites.{0,40}(citations?|share)", r"(citations?|tools).{0,40}add to favou?rites",
        r"pdf\s?download\s?loading",
    ],
    # Oxford Academic works whose "abstract" is only the citation ("..., Volume 46, Issue 1, 1 January 1970,
    # Pages 1-2"), possibly after "This article comments on:". Title line, at most 300 characters, the citation,
    # little after it; real abstracts that cite a paper are longer.
    "Citation instead of an abstract": [
        r"^[^\n]*\n.{0,300}volume \d+, issue \d+, \d{1,2} \w+ \d{4}, pages? \S*.{0,200}$",
    ],
    # Grey literature from library repositories (OpenAlex): legal opinions, audit letters, catalogue stubs.
    "Grey literature: legal opinions, audit letters, catalogue stubs": [
        r"attorney general of texas", r"report of the texas state auditor'?s office",
        r"(letter report|correspondence|testimony) issued by the (general accounting|government accountability) "
        r"office",
        r"auditor of state .{0,40}today released an audit report", r"this audit report by the south carolina state",
        r"available from british library document supply cent(re|er)",
        r"psyndex tests? (review|info)", r"is brought to you for free and open access by",
        r"hal is a multi-disciplinary open access archive",
    ],
    # Entity stubs of Australian archival databases (Guide to Australian Business Records, Arrowsmith's maps).
    "Archival database stubs": [r"is a biographical, bibliographical and archival database"],
    # Materials Project datasets (OpenAlex, via OSTI).
    "Generated crystal-structure descriptions": [r"crystallizes in the \w+ \S+ space group"],
    # figshare: "Additional file 3 of <paper title>".
    "Supplementary files deposited as separate works": [
        r"^(additional file \d+|moesm\d+|supplementary (file|material|table|figure|data)\s*\S*) (of|for|to)\b",
    ],
    # 2026 repository uploads titled "... — E8 Intelligence Research".
    "Generated lottery and numerology posts": [r"\be8 intelligence research\b"],
}
DEFAULT_DROP_PATTERNS = [p for ps in _DROP_GROUPS.values() for p in ps]

# Removed from abstracts before tokenising (the rest of the abstract is kept).
_STRIP_GROUPS: dict[str, list[str]] = {
    "Springer's note on machine-added keywords": [
        r"these keywords were added by machine and not by the authors\.?(\s*this process is experimental and the "
        r"keywords may be updated as the learning algorithm improves\.?)?",
    ],
    "Repository cover-page notices": [
        r"the harvard community has made this article openly available\.?(\s*please share how this access "
        r"benefits you\.?)?(\s*your story matters\.?)?",
        r"\(article begins on next page\)",
    ],
    # IEEE conference records without an abstract.
    "IEEE \"summary form only given\"": [r"^\s*summary form only given\.?"],
    # Abstracts from the OSTI/INIS energy and physics databases end with the abstractor's mark: "(auth)",
    # "(tr-auth)", "(orig.)" (their formatting codes, "He/sup 4/", are removed by the tokenizer).
    "Abstracting-service marks": [r"\((?:tr-)?(?:auth|author|authors|author abstract|orig|aip)\.?\)"],
    "World Scientific's formula placeholder": [r"\[?formula:?\s*see text\]?"],
    # About 1% of abstracts: "(C) 2010 Elsevier Ltd. All rights reserved.", "Copyright: 2013 Islam et al. This is
    # an open-access article distributed under the terms of the Creative Commons Attribution License, ..."
    "Copyright and licence notices": [
        r"(?:copyright\s*:?\s*)?(?:©|\(c\)|copyright\s*:?)\s*(?:by\s+)?(?:19|20)\d{2}\b[^.]{0,150}\.?",
        r"all rights reserved\.?", r"this article is protected by copyright\.?",
        r"this is an open[- ]access article,? (?:distributed )?under the (?:terms of the )?(?:creative commons|"
        r"cc[- ]by)[^.]{0,300}\.?",
    ],
    # "Abstract: ...", "Abstract We ..."; not "Abstract algebra ...". Structured abstracts: "Background:",
    # "Methods:", "Conclusions:" (7-9 % of abstracts in biology and psychology).
    "Abstract and section headings": [
        r"^\s*(?:abstract\s*[:.]|(?-i:Abstract|ABSTRACT)\s+(?=(?-i:[A-Z])))",
        r"\b(?:background|context|importance|objectives?|aims?(?: and objectives)?|purpose|methods?|methodology|"
        r"materials and methods|design|setting|participants|interventions?|main outcome measures?|results?|findings|"
        r"conclusions?|interpretation|significance|introduction|discussion|limitations|implications)\s*:",
    ],
}
DEFAULT_STRIP_PATTERNS = [p for ps in _STRIP_GROUPS.values() for p in ps]
# Pattern -> group name, for the bundle; patterns of a config's own are shown without a name.
PATTERN_LABELS = {p: g for groups in (_DROP_GROUPS, _STRIP_GROUPS) for g, ps in groups.items() for p in ps}


class PreprocessConfig(_Section):
    fingerprint_omit_default = ("extra_drop_patterns",)

    fields: list[Literal["title", "abstract"]] = ["title", "abstract"]
    require_abstract: bool = True
    min_tokens: int = 10
    min_token_len: int = 3
    max_token_len: int = 30
    language_filter: LanguageFilterConfig = Field(default_factory=LanguageFilterConfig)
    extra_stopwords: list[str] = Field(default_factory=list)
    # Join hyphenated words with a prefix that is no word of its own: "non-linear" -> "nonlinear",
    # "co-operation" -> "cooperation", "e-learning" -> "elearning", "x-ray" -> "xray" (tst.text.HYPHEN_PREFIXES),
    # and the phrases in tst.text.JOINED_PHRASES ("well-being" -> "wellbeing", "state-of-the-art").
    join_prefixes: bool = True
    # Drop English endings after an apostrophe ("Kant's" -> "kant") and join words at other apostrophes
    # ("da'wah" -> "dawah", "Qur'an" -> "quran"), which would leave fragments ("wah", "qur") otherwise.
    join_apostrophes: bool = True
    # Remove accents: "Schrödinger" = "Schrodinger", "Poincaré" = "Poincare".
    fold_accents: bool = True
    # Keep only words in the Latin alphabet; words in other scripts (Cyrillic, Chinese, Korean, Arabic, ...) come
    # from titles and abstracts in other languages.
    latin_only: bool = True
    # Also remove frequent German/French/Spanish/Portuguese/Italian/Dutch/Indonesian function words, so
    # foreign-language titles of English abstracts do not form "language topics".
    foreign_stopwords: bool = True
    # Exact duplicate abstracts: keep the first document only.
    dedupe: bool = True
    # Abstracts occurring at least this often are boilerplate (licence texts, empty JATS tags): drop all.
    boilerplate_min_count: int = 5
    drop_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_DROP_PATTERNS))
    # Junk of this corpus only, dropped like drop_patterns: {group name: [pattern, ...]}. For sources that occur in
    # a few corpora; a new default pattern would re-run every corpus. The group names label the patterns in the
    # bundle.
    extra_drop_patterns: dict[str, list[str]] = Field(default_factory=dict)
    strip_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_STRIP_PATTERNS))

    def all_drop_patterns(self) -> list[str]:
        return self.drop_patterns + [p for ps in self.extra_drop_patterns.values() for p in ps]

    def drop_pattern_labels(self) -> dict[str, str]:
        return {**PATTERN_LABELS, **{p: g for g, ps in self.extra_drop_patterns.items() for p in ps}}


class VectorizeConfig(_Section):
    # Integers are absolute document counts, floats in (0, 1] are fractions of the corpus.
    min_df: int | float = 5
    max_df: int | float = 0.5
    max_features: int | None = 100_000
    tf: Literal["raw", "sublinear"] = "sublinear"
    ngram_max: Literal[1, 2] = 1
    # Merge variants of a word before counting, if both occur in the corpus: "plural" (models -> model,
    # studies -> study, genetics -> genetic, spectra -> spectrum) and "spelling" (British -> American,
    # behaviour -> behavior, modelling -> modeling). The most frequent form names the merged word.
    conflate: list[Literal["plural", "spelling"]] = Field(default_factory=lambda: ["plural", "spelling"])
    # Remove generic words of academic prose ("study", "results", "propose"; tst/topics/academic_stopwords.yaml)
    # with all their variants.
    academic_stopwords: bool = True
    # More words to remove from the vocabulary (with their variants), and words of the academic list to keep.
    stopwords: list[str] = Field(default_factory=list)
    keep: list[str] = Field(default_factory=list)


class TopicsConfig(_Section):
    # Reuse an existing model bundle (`<run>/model`) instead of training, e.g. to map a new corpus into the
    # topic space of an earlier run.
    model_path: Path | None = None
    k: int | Literal["auto"] = "auto"
    k_grid: list[int] = Field(default_factory=lambda: [10, 15, 20, 25, 30, 40, 50, 64])
    # Stability analysis (Greene et al. 2014): `runs` models on random `subsample`s per k.
    runs: int = 5
    subsample: float = 0.8
    train_sample: int | None = 500_000
    algorithm: Literal["nmf", "minibatch"] = "nmf"
    beta_loss: Literal["frobenius", "kullback-leibler"] = "frobenius"
    max_iter: int = 400
    tol: float = 1e-4
    alpha_W: float = 0.0
    alpha_H: float | Literal["same"] = "same"
    l1_ratio: float = 0.0
    top_n: int = 10
    coherence_sample: int = 200_000
    # combined: mean of min-max normalised NPMI, stability and diversity.
    # stable_coherence: the most coherent k among those with stability >= min_stability (falls back to
    #   the most stable k). Robust when one k is far more stable than the rest.
    selection: Literal["stable_coherence", "combined", "coherence", "stability"] = "stable_coherence"
    min_stability: float = 0.7


class PeriodConfig(_Section):
    """Years used from the embed stage on (trajectories, maps, network, bundle). Training uses all years."""

    min_year: int | None = None
    # "auto": drop trailing years with less than half the papers of the largest of the five years before.
    # Exports are incomplete for their last years, which would show up as sudden trends.
    max_year: int | Literal["auto"] | None = "auto"
    # With max_year "auto" and OpenAlex input: also drop the snapshot's own year, which is never complete.
    drop_snapshot_year: bool = True


class EntityTrajectoryConfig(_Section):
    min_papers: int = 3
    # Sliding window over calendar years (y-window, y]; 1 means no smoothing.
    window: int = 3
    # centroid_mean: average of the yearly centroids in the window (old behaviour).
    # pooled: average of all papers in the window (weights years by paper counts).
    # cumulative: average of all papers up to and including the year.
    smoothing: Literal["centroid_mean", "pooled", "cumulative"] = "centroid_mean"
    # Fractional counting divides a paper's weight among its authors.
    fractional: bool = False
    min_years: int = 1
    # Author trajectories only: papers with more authors (large collaborations) are left out, so that
    # collaboration members do not count as prolific authors. null: keep all papers.
    max_authors: int | None = 50


class PairwiseConfig(_Section):
    enabled: bool = True
    entity_type: Literal["venue", "author"] = "venue"
    max_entities: int = 200
    min_overlap: int = 3
    max_lag: int = 5


class TrajectoriesConfig(_Section):
    entities: dict[Literal["venue", "author"], EntityTrajectoryConfig] = Field(
        default_factory=lambda: {
            "venue": EntityTrajectoryConfig(min_papers=10, window=1),
            "author": EntityTrajectoryConfig(min_papers=3, window=3),
        }
    )
    pairwise: PairwiseConfig = Field(default_factory=PairwiseConfig)


class ProjectionConfig(_Section):
    method: Literal["umap", "tsne", "pca"] = "umap"
    # Topic vectors are distributions; sqrt makes Euclidean distance equal to Hellinger distance.
    hellinger: bool = True
    fit_sample: int = 200_000
    doc_sample: int | None = 1_000_000
    n_neighbors: int = 30
    min_dist: float = 0.1
    perplexity: float = 30.0
    # UMAP is only reproducible single-threaded; leave False for large corpora.
    deterministic: bool = False


class NetworkConfig(_Section):
    enabled: bool = True
    min_year: int | None = 1960
    max_year: int | None = None
    min_author_papers: int = 5
    # Hyper-authored papers create m(m+1)/2 pairs each; skip papers with more authors than this.
    max_authors_per_paper: int | None = 50
    # Keep the top_k topics of each paper (old pipeline: 8 of 64, i.e. k/8). "auto" = max(1, round(k/8)).
    top_k_topics: int | Literal["auto"] = "auto"
    window: int = 3
    # window: expertise = self-loop weight in the accumulated window (old topicflow.reorient)
    # cumulative: expertise summed over all accumulated years up to the year (old network_accumulate)
    expertise: Literal["window", "cumulative"] = "window"


class PageRankConfig(_Section):
    enabled: bool = True
    weighted: bool = True
    per_topic: bool = True
    top_n: int = 100


class SNAConfig(_Section):
    enabled: bool = True
    measures: list[
        Literal[
            "nodes", "edges", "density", "mean_degree", "max_degree", "transitivity",
            "avg_local_transitivity", "giant_fraction", "avg_path_length", "mean_betweenness",
            "avg_shortest_path",
        ]
    ] = ["nodes", "edges", "density", "mean_degree", "max_degree", "transitivity",
         "avg_local_transitivity", "giant_fraction", "avg_shortest_path"]
    per_topic: bool = True
    # avg_shortest_path: mean number of co-authorship steps between two authors of the largest connected part of
    # the undirected graph; exact up to this many authors, above estimated from breadth-first searches from this
    # many random authors (Eppstein & Wang 2004). avg_path_length is the exact directed version (slow).
    path_sources: int = 500


class CommunityConfig(_Section):
    enabled: bool = True
    method: Literal["leiden", "walktrap"] = "leiden"
    resolution: float = 1.0
    # Leiden iterations; -1 runs until the partition is stable, which takes an hour on graphs with tens of
    # millions of edges. Set e.g. 5 to bound the time.
    iterations: int = -1
    coreness: bool = True


class AnalysisConfig(_Section):
    years: list[int] | None = None
    # Years analysed at once (default: all). Each process holds one year's graph, so lower this when the
    # graph of a year has tens of millions of edges.
    max_workers: int | None = None
    flows: bool = True
    flow_top_n: int = 25
    pagerank: PageRankConfig = Field(default_factory=PageRankConfig)
    sna: SNAConfig = Field(default_factory=SNAConfig)
    communities: CommunityConfig = Field(default_factory=CommunityConfig)
    graphml: bool = False


class NoveltyConfig(_Section):
    # A paper has a topic if the topic's share in the paper is at least this.
    threshold: float = 0.1
    # Topic pairs are compared with the papers of this many preceding years.
    window: int = 3
    # Years whose preceding window has fewer papers get no novelty scores.
    min_window_papers: int = 1000
    # A paper "combines topics unusually" if one of its topic pairs occurs at most this share of the expected
    # number of times (observed / expected under independence).
    unusual_ratio: float = 0.5


class CitationsConfig(_Section):
    """Citation flow between topics (`tst citations`, tst.citations, notes/citation_flow_theory.md)."""

    enabled: bool = True
    # Store written by `tst openalex-fetch <store> --links`; default: links/ in the OpenAlex store (input.path).
    links: Path | None = None
    # Citing years per window within a corpus (all corpora together: `tst citations --global-window`).
    window: int = 5
    # Each work keeps its strongest topics (renormalised); the EM costs grow with the square of this.
    top_topics: int = 6
    # Works without an abstract (same types, subfields and years as the corpus) get a topic mix from their title.
    title_only: bool = True
    # "exclude": citations between papers with a shared author are left out of the flows.
    self_citations: Literal["exclude", "include"] = "exclude"
    # Pseudo-citations per topic pair, pulling rates with little exposure towards the mean rate (gamma prior).
    prior: float = 0.5
    max_iter: int = 500
    # Relative change of the log-likelihood at which EM stops.
    tol: float = 1e-8
    # Bootstrap replicates (citing papers resampled with Poisson(1) weights) for the main resolution.
    bootstrap: int = 20
    bootstrap_iter: int = 40
    # Topic pairs with fewer citations in both directions together are left out of the Hodge decomposition.
    hodge_min_flow: float = 20.0
    # Windows with fewer citations are not fitted.
    min_citations: int = 2000
    # Flows between different topics listed per window in the bundle.
    top_flows: int = 200
    # Papers with fewer resolved references get no knowledge-base surprise.
    min_refs_surprise: int = 3
    # Channels between fields whose two topics are each other's closest topic in the other corpus and share at
    # least this cosine similarity of their topic-word weights are flagged as twins (one subject in two maps).
    twin_similarity: float = 0.5


class IndicatorsConfig(_Section):
    # Topic life cycle: shares smoothed over this many years; "growing"/"declining" if the share of the last
    # `recent` years differs by at least `change` (relative) from the `recent` years before.
    smoothing: int = 3
    recent: int = 5
    change: float = 0.2
    # Lists on the topic pages.
    top_papers: int = 8
    top_entities: int = 10
    related: int = 5
    # Citations need time: impact averages leave out the last this many years.
    citation_lag: int = 2


class TermsConfig(_Section):
    # Papers per year using each vocabulary word, for the term explorer; two-word phrases (adjacent words after
    # stop-word removal, both in the vocabulary) if they occur in at least `min_bigram_papers` papers.
    enabled: bool = True
    bigrams: bool = True
    min_bigram_papers: int = 100


class ExportConfig(_Section):
    fingerprint_omit_default = ("publish",)

    # Display name and group of the corpus, for the index of a web app (e.g. "Medicine: surgery", "Health Sciences").
    label: str | None = None
    group: str | None = None
    # Topic names: a TSV file (id, name, top terms when named) or one name per line, in topic order. The fine
    # resolution reads the same name with "_fine" before the suffix. A missing file means unnamed topics; names
    # whose top terms no longer match the model are dropped. Only changes the bundle, so renaming is cheap.
    labels_file: Path | None = None
    # False: `tst collect` leaves the corpus out of the web app (index, term explorer, researcher mobility) and
    # removes an earlier entry of it. The bundle is still written.
    publish: bool = True
    max_docs: int | None = 200_000
    density_grid: int = 256
    density_sample: int = 20_000
    # Link per paper. Placeholders: {title} (URL-encoded) and {source_id}, e.g. "https://openalex.org/{source_id}"
    # or "https://doi.org/{source_id}". Papers without a source_id get no link if the template uses it.
    url_template: str | None = None
    # Why this corpus has its own settings, by config key, e.g. {"vectorize.keep": "subjects in mathematics"}.
    # The bundle shows each note with the parameter that uses the key (manifest.json run.parameters, `note`).
    notes: dict[str, str] = Field(default_factory=dict)


class FineConfig(_Section):
    # A second, finer topic resolution in <workdir>/fine with all stages from train on. It shares the
    # documents, tokens, TF-IDF matrix and the k sweep with the main run, so the main run must exist.
    enabled: bool = False
    # "auto": the largest k of topics.k_grid that is stable (topics.min_stability) and larger than the main
    # k; if none is stable, the next larger k of the grid.
    k: int | Literal["auto"] = "auto"


class RunConfig(_Section):
    name: str
    workdir: Path
    n_jobs: int = Field(default=-1, validate_default=True)
    seed: int = 42
    input: InputConfig
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    vectorize: VectorizeConfig = Field(default_factory=VectorizeConfig)
    topics: TopicsConfig = Field(default_factory=TopicsConfig)
    period: PeriodConfig = Field(default_factory=PeriodConfig)
    trajectories: TrajectoriesConfig = Field(default_factory=TrajectoriesConfig)
    projection: ProjectionConfig = Field(default_factory=ProjectionConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    novelty: NoveltyConfig = Field(default_factory=NoveltyConfig)
    indicators: IndicatorsConfig = Field(default_factory=IndicatorsConfig)
    terms: TermsConfig = Field(default_factory=TermsConfig)
    citations: CitationsConfig = Field(default_factory=CitationsConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    fine: FineConfig = Field(default_factory=FineConfig)

    @field_validator("n_jobs")
    @classmethod
    def _resolve_jobs(cls, v: int) -> int:
        import os

        return (os.cpu_count() or 1) if v < 1 else v


def load_config(path: str | Path) -> RunConfig:
    """Load a YAML config; relative paths inside it are resolved against the YAML's directory."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    cfg = RunConfig.model_validate(raw)
    base = path.parent

    def _abs(p: Path | None) -> Path | None:
        return None if p is None else (p if p.is_absolute() else (base / p).resolve())

    cfg.workdir = _abs(cfg.workdir)
    cfg.input.path = _abs(cfg.input.path)
    cfg.topics.model_path = _abs(cfg.topics.model_path)
    cfg.export.labels_file = _abs(cfg.export.labels_file)
    return cfg
