"""How a bundle was made, for readers: the resolved config, the parameters with the reasons behind them, and the
details of text cleaning and vocabulary curation.

* describe(): manifest.json run.parameters, a list of groups of parameters. Each parameter is one decision:
  label, value, description (what it does), rationale (why), references (ids into tst.references), note (why
  this corpus deviates, from export.notes), details (a section of methods.json) and config (the keys).
* methods(): methods.json, the long lists behind some parameters (patterns, stop words, variant rules).
* vocabulary(): vocabulary.json, the vocabulary with document frequencies and the merged word forms.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import PATTERN_LABELS, RunConfig
from .references import REFERENCES

# Settings that only concern this machine or the run's speed, not its results.
_LOCAL = {"workdir", "n_jobs"}
_PATHS = (("input", "path"), ("topics", "model_path"), ("export", "labels_file"))

_SELECTION = {
    "stable_coherence": "the most coherent k among those with stability ≥ {min_stability}",
    "combined": "best mean of normalised coherence, stability and diversity",
    "coherence": "the most coherent k",
    "stability": "the most stable k",
}
_SMOOTHING = {
    "pooled": "average of all papers in a {window}-year window",
    "centroid_mean": "average of the yearly centroids in a {window}-year window",
    "cumulative": "average of all papers up to the year",
}
_LANGUAGES = {"de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese", "it": "Italian", "nl": "Dutch",
              "id": "Indonesian"}
# Reasons in tokens.parquet, in the order the filters apply, with a label for readers.
_REASONS = {"spam": "junk text", "boilerplate": "boilerplate", "duplicate": "duplicate",
            "no_abstract": "no abstract", "too_short": "too short", "language": "not English"}


def resolved_config(cfg: RunConfig) -> dict:
    """The full config with defaults, without machine-specific settings; paths are reduced to file names."""
    out = cfg.model_dump(mode="json", exclude=_LOCAL)
    for section, key in _PATHS:
        p = getattr(getattr(cfg, section), key)
        out[section][key] = None if p is None else p.name
    return out


def _p(label: str, value: Any, description: str, *keys: str, why: str = "", refs: tuple[str, ...] = (),
       details: str | None = None) -> dict:
    return {"label": label, "value": value, "description": description, "rationale": why,
            "references": list(refs), "note": "", "details": details, "config": list(keys)}


def _n(x) -> str:
    return f"{x:,}" if isinstance(x, int) else str(x)


def _removed(pre_info: dict, *reasons: str) -> str:
    return f"; {_n(sum(pre_info.get(r, 0) for r in reasons))} papers removed" if pre_info else ""


def describe(cfg: RunConfig, stages: dict, k: int, resolution: str, named: bool = False) -> list[dict]:
    """Parameter groups for an "about this map" panel: [{group, parameters: [...]}], see the module doc.
    Values are display strings; `config` names the config keys they come from. `named`: the topics have names."""
    st = lambda name: stages.get(name) or {}  # noqa: E731
    groups = [_corpus(cfg, st, stages), _cleaning(cfg, st), _vocabulary(cfg, st), _topic_model(cfg, st, k, resolution),
              _trajectories(cfg, st), _map(cfg, st)]
    if cfg.network.enabled:
        groups.append(_network(cfg, st))
    groups.append(_derived(cfg, st, resolution, named))
    _attach_notes(groups, cfg.export.notes)
    return groups


def _corpus(cfg: RunConfig, st, stages: dict) -> dict:
    inp, emb = cfg.input, st("embed")
    corpus = []
    if inp.format == "openalex":
        sel = [f"fields {inp.openalex.fields}"] if inp.openalex.fields else []
        sel += [f"subfields {inp.openalex.subfields}"] if inp.openalex.subfields else []
        corpus.append(_p("Source", f"OpenAlex, {', '.join(sel)}",
                         "Works whose primary OpenAlex topic lies in these fields or subfields.",
                         "input.openalex.fields", "input.openalex.subfields",
                         why="OpenAlex is an open (CC0) index of scholarly works with abstracts, authors and venues "
                             "in all fields, so the data may be shared with the map and the run can be repeated.",
                         refs=("openalex",)))
        if inp.openalex.types:
            corpus.append(_p("Work types", ", ".join(inp.openalex.types), "Other work types are left out.",
                             "input.openalex.types",
                             why="Keeps research texts; books and chapters, datasets, editorials, letters, errata "
                                 "and paratext have no comparable abstracts."))
    else:
        corpus.append(_p("Source", f"{inp.format} file {inp.path.name}", "The publication list of this run.",
                         "input.path"))
    lo = emb.get("min_year") or cfg.period.min_year or (st("ingest").get("year_range") or [None])[0]
    hi = emb.get("max_year")
    corpus.append(_p("Years", f"{lo or 'first'}–{hi or 'last'}",
                     "Years shown. Trailing years with few papers (incomplete data) are cut off.",
                     "input.min_year", "period.min_year", "period.max_year", "period.drop_snapshot_year",
                     why="A database is incomplete for its last years, which would show up as a sudden decline of "
                         "every topic; years with less than half the papers of the years before are dropped. Early "
                         "years have few papers with abstracts. The model is trained on all years."))
    if "ingest" in stages and "embedded" in emb:
        corpus.append(_p("Papers", f"{_n(emb['embedded'])} of {_n(st('ingest').get('documents', 0))}",
                         "Papers in the topic space, of all papers read; see Text cleaning for the others.",
                         "preprocess"))
    return {"group": "Corpus", "parameters": corpus}


def _cleaning(cfg: RunConfig, st) -> dict:
    pre, info = cfg.preprocess, st("preprocess")
    lf = pre.language_filter
    out = [_p("Text", " and ".join(pre.fields), "Text of each paper used for the topics.", "preprocess.fields",
              why="Titles and abstracts summarise a paper and are openly available for most works; full texts "
                  "are not.")]
    if lf.method != "none":
        desc = f"English function words must make up more than {lf.threshold:.0%} of a paper's words"
        if lf.method == "stopword_vote":
            desc += " and outnumber those of " + ", ".join(_LANGUAGES.values())
        desc += "."
        if lf.method == "stopword_vote" and lf.sentence_filter:
            desc += " Bilingual abstracts are first cut down to their English sentences."
        if lf.max_other_script < 1:
            desc += (f" Abstracts with more than {lf.max_other_script:.0%} of their letters in other scripts "
                     "(Cyrillic, Chinese, Japanese, Korean, Arabic, ...) are removed as well, also when the title is "
                     "English.")
        value = "English" + _removed(info, "language")
        if lf.method == "stopword_vote" and lf.ignore_foreign_titles:
            desc += (" A title with more function words of another language than English ones is left out and the "
                     "paper is kept with its English abstract.")
            if "titles_ignored" in info:
                value += f"; titles of {_n(info['titles_ignored'])} papers left out"
        out.append(_p("Language", value, desc, "preprocess.language_filter",
                      why="A topic model puts each language into topics of its own. In OpenAlex 8–35 % of the works "
                          "in a field have no English abstract, and some abstracts are bilingual. The function-word "
                          "vote cannot see languages in other scripts, so a Japanese abstract with an English title "
                          "passed it (0.1–0.4 % of the papers kept in physics, philosophy and political science). "
                          "Titles in another language over an English abstract (about 10 % of the papers in business, "
                          "7 % in philosophy and political science, under 1 % in physics; mostly Indonesian, Spanish, "
                          "Portuguese, French and German) put their words into the topics, e.g. \"kinerja\" and "
                          "\"pelanggan\" next to \"performance\" and \"customer\"."))
    out.append(_p("Length", f"at least {pre.min_tokens} words" + _removed(info, "too_short", "no_abstract"),
                  f"Papers with fewer words after cleaning are removed"
                  + (", and papers without an abstract." if pre.require_abstract else "."),
                  "preprocess.min_tokens", "preprocess.require_abstract",
                  why="A topic mixture estimated from a few words is unreliable, and records with a title only are "
                      "often not research papers."))
    if pre.dedupe or pre.boilerplate_min_count:
        parts = []
        if pre.boilerplate_min_count:
            parts.append(f"abstracts occurring {pre.boilerplate_min_count}+ times are removed with all copies")
        if pre.dedupe:
            parts.append("of other repeated abstracts only the first paper is kept")
        out.append(_p("Repeated texts", f"{_n(info.get('boilerplate', 0))} boilerplate, "
                      f"{_n(info.get('duplicate', 0))} duplicates removed" if info else "removed",
                      parts[0][0].upper() + parts[0][1:] + ("; " + parts[1] if len(parts) > 1 else "") + ".",
                      "preprocess.boilerplate_min_count", "preprocess.dedupe",
                      why="Texts that occur many times are not abstracts (licence texts, empty tags, placeholders), "
                          "and duplicates (reprints, repository copies) would count a paper twice. Both would form "
                          "topics of their own."))
    if drop := pre.all_drop_patterns():
        extra = (f" This corpus also drops: {', '.join(pre.extra_drop_patterns)}."
                 if pre.extra_drop_patterns else "")
        out.append(_p("Junk texts", f"{len(drop)} patterns" + _removed(info, "spam"),
                      "Papers whose abstract is not an abstract are removed: spam, scraped publisher pages, a "
                      "citation instead of an abstract, grey literature, generated texts and supplementary "
                      "files." + extra,
                      "preprocess.drop_patterns", *(["preprocess.extra_drop_patterns"] if extra else []),
                      why="Each kind was found by reading the topics and their most typical papers in every field; "
                          "such texts form topics of their own.", details="cleaning.drop_patterns"))
    if pre.strip_patterns:
        out.append(_p("Removed passages", f"{len(pre.strip_patterns)} patterns",
                      "Copyright and licence notices, publisher notes, placeholders, abstracting-service marks "
                      "(\"(auth)\") and the headings of structured abstracts (\"Background:\", \"Methods:\") are cut "
                      "out; the rest of the abstract is kept.",
                      "preprocess.strip_patterns",
                      why="They recur in the abstracts of some publishers and journals (copyright notices in about "
                          "1 %, structured headings in 7–9 % of biology and psychology abstracts) and would make "
                          "topics of publishers and formats instead of subjects.", details="cleaning.strip_patterns"))
    value = ("Latin letters only" if pre.latin_only else "letters only") + \
        f", {pre.min_token_len}–{pre.max_token_len} characters, lower case" + \
        (", without accents" if pre.fold_accents else "")
    desc = ("Formulas, HTML tags and entities, the formatting codes of old database records (\"He/sup 4/\", "
            "\"/spl mu/m\", \"{minus}\"), URLs, e-mail addresses and numbers are removed; a word is a run of letters.")
    if pre.latin_only:
        desc += " Words in other scripts (Cyrillic, Chinese, Korean, Arabic, ...) are left out."
    if pre.fold_accents:
        desc += " Accents are removed: \"Schrödinger\" = \"Schrodinger\", \"Poincaré\" = \"Poincare\"."
    why = ("Shorter tokens are mostly symbols, initials and parts of formulas. The formatting codes are in 3 % of "
           "physics abstracts (OSTI, IEEE) and formed a topic \"sup, minus, yields, spl\".")
    if pre.latin_only:
        why += (" Words in other scripts come from titles and abstracts in other languages and took 3–17 % of the "
                "vocabulary.")
    if pre.fold_accents:
        why += " Names and loanwords are spelled with and without accents, which split one word in two."
    out.append(_p("Words", value, desc, "preprocess.min_token_len", "preprocess.max_token_len", "preprocess.latin_only",
                  "preprocess.fold_accents", why=why, details="cleaning.words"))
    if pre.join_apostrophes:
        out.append(_p("Apostrophes", "joined",
                      "English endings after an apostrophe are dropped (\"Kant's\" = \"Kant\", \"don't\" = "
                      "\"do\"), and words are joined at other apostrophes (\"da'wah\" = \"dawah\", \"Qur'an\" = "
                      "\"Quran\").", "preprocess.join_apostrophes",
                      why="Splitting at the apostrophe left fragments such as \"wah\" and \"qur\" that became top "
                          "words of topics.", details="cleaning.words"))
    if pre.join_prefixes:
        out.append(_p("Hyphenated words", "prefixes joined",
                      "Prefixes that are no words of their own are joined to the word: \"non-linear\" = "
                      "\"nonlinear\", \"e-learning\" = \"elearning\"; also the phrases \"well-being\" = "
                      "\"wellbeing\" and \"state-of-the-art\" = \"stateoftheart\" (an academic stop word).",
                      "preprocess.join_prefixes",
                      why="Otherwise the same word is split in two or not depending on how it is written, and its "
                          "prefix is lost; \"state of the art\" left the word \"art\" in engineering topics.",
                      details="cleaning.hyphen_prefixes"))
    langs = ["English"] + (list(_LANGUAGES.values()) if pre.foreign_stopwords else [])
    out.append(_p("Function words", "removed (" + ", ".join(langs) + ")"
                  + (f", plus {len(pre.extra_stopwords)} own words" if pre.extra_stopwords else ""),
                  "Words without topical meaning (\"the\", \"which\", \"however\") are removed: scikit-learn's "
                  "English list" + (" and the most frequent function words of seven other languages, so that "
                                    "foreign-language titles of English papers do not form topics"
                                    if pre.foreign_stopwords else "") + ".",
                  "preprocess.foreign_stopwords", "preprocess.extra_stopwords",
                  why="Very frequent function words would otherwise appear in all topics; removing them is the step "
                      "of stop-word removal that matters (Schofield et al. 2017).", refs=("schofield2017",),
                  details="cleaning.function_words"))
    return {"group": "Text cleaning", "parameters": out}


def _vocabulary(cfg: RunConfig, st) -> dict:
    v, vec = cfg.vectorize, st("vectorize")
    out = []
    if cfg.topics.model_path is not None:
        return {"group": "Vocabulary", "parameters": [
            _p("Vocabulary", f"{_n(vec.get('features', 0))} words", "Taken from the existing model "
               f"{cfg.topics.model_path.name} with its curation.", "topics.model_path")]}
    if v.conflate:
        kinds = {"plural": "singular/plural", "spelling": "British/American spelling"}
        value = ", ".join(kinds[c] for c in v.conflate)
        if vec.get("variants_merged"):
            value += f"; {_n(vec['variants_merged'])} forms merged into {_n(vec['variant_groups'])} words"
        out.append(_p("Word variants", value,
                      "Forms of a word count as one word if both occur in the corpus: singular and plural "
                      "(\"model\", \"models\"; \"genetic\", \"genetics\"; \"spectrum\", \"spectra\"), verb forms in -s "
                      "(\"argue\", \"argues\") and British and American spelling (\"behaviour\", \"behavior\"). The "
                      "most frequent form names the word.",
                      "vectorize.conflate",
                      why="Topic word lists showed both forms side by side. Full stemming (Porter) also merges "
                          "unrelated words and does not improve topic models (Schofield & Mimno 2016), so only plural "
                          "endings are removed, as in Harman's S-stemmer (1991), with English rules (\"-es\" only "
                          "after s, x, z, ch, sh and o) and exceptions (\"news\", \"bias\", \"lens\").",
                      refs=("schofield2016", "harman1991"), details="vocabulary.variants"))
    if v.academic_stopwords or v.stopwords:
        value = f"{_n(vec['stopwords_removed'])} words removed" if vec.get("stopwords_removed") else "removed"
        if v.stopwords:
            value += f", {len(v.stopwords)} of them added for this corpus"
        if v.keep:
            value += f"; kept: {', '.join(v.keep)}"
        out.append(_p("Academic stop words", value,
                      "Generic words of academic writing (\"study\", \"results\", \"propose\", \"significant\") are "
                      "removed with all their forms. A corpus can keep words of the list that are subjects in it, "
                      "or add its own.",
                      "vectorize.academic_stopwords", "vectorize.stopwords", "vectorize.keep",
                      why="They are frequent in every field and appear in many topics without describing any. Long "
                          "generic stop lists add little beyond function words; words specific to a kind of text are "
                          "better found from the texts themselves (Schofield et al. 2017; Gerlach et al. 2019). The "
                          "list was scored over the 14 OpenAlex fields and reviewed by hand (Sarica & Luo 2021); "
                          "candidates that name a subject in some field were left off.",
                      refs=("schofield2017", "gerlach2019", "sarica2021"), details="vocabulary.stopwords"))
    df_unit = "papers" if isinstance(v.min_df, int) else "of the papers"
    out.append(_p("Vocabulary", f"{_n(vec.get('features', v.max_features))} words",
                  f"Words in at least {v.min_df} {df_unit} and at most {v.max_df:.0%} of them"
                  + (f", the {_n(v.max_features)} most frequent" if v.max_features else "")
                  + (", plus word pairs" if v.ngram_max == 2 else "") + ".",
                  "vectorize.min_df", "vectorize.max_df", "vectorize.max_features", "vectorize.ngram_max",
                  why="Rare words (typos, names, words of other languages) say little about topics and cost memory; "
                      "words in a large share of all papers do not tell topics apart.", details="vocabulary"))
    out.append(_p("Weights", f"{'sublinear ' if v.tf == 'sublinear' else ''}TF-IDF",
                  ("Each word count c becomes 1 + log c" if v.tf == "sublinear" else "Word counts are")
                  + " weighted by the word's inverse document frequency; each paper's vector has length 1.",
                  "vectorize.tf",
                  why="Down-weights words that occur in many papers and repetitions within a paper (Salton & Buckley "
                      "1988), so that a topic is defined by distinctive words.", refs=("salton1988",)))
    return {"group": "Vocabulary", "parameters": out}


def _topic_model(cfg: RunConfig, st, k: int, resolution: str) -> dict:
    t = cfg.topics
    model = [_p("Method", "NMF" if t.algorithm == "nmf" else "mini-batch NMF",
                f"Non-negative matrix factorisation ({t.beta_loss} loss, NNDSVDa initialisation) of the TF-IDF "
                "matrix: each topic is a weighted list of words, each paper a mixture of topics.",
                "topics.algorithm", "topics.beta_loss",
                why="NMF gives sparse, readable topics (Lee & Seung 1999); in comparisons its top words were more "
                    "coherent and less generic than those of LDA (O'Callaghan et al. 2015). NNDSVDa initialisation "
                    "makes the result deterministic (Boutsidis & Gallopoulos 2008).",
                refs=("lee1999", "ocallaghan2015", "boutsidis2008"))]
    if t.model_path is not None:
        model.append(_p("Topics", str(k), f"Taken from the existing model {t.model_path.name}.",
                        "topics.model_path", why="Places this corpus in the topic space of an earlier run."))
    else:
        rule = _SELECTION[t.selection].format(min_stability=t.min_stability)
        if resolution == "fine":
            how = (f"Fine resolution: k = {cfg.fine.k} set in the config." if cfg.fine.k != "auto" else
                   f"Fine resolution: the largest k with stability ≥ {t.min_stability} above the main "
                   "resolution's k (otherwise the next larger k).")
            why = ("A second, more detailed view of the same papers: as many topics as can still be fitted "
                   "reproducibly.")
        else:
            how = f"Chosen automatically: {rule}." if t.k == "auto" else f"Set in the config (k = {t.k})."
            why = ("Too few topics merge subjects, too many split them at random. Topics that reappear when the "
                   "model is fitted on other samples are reliable (Greene et al. 2014); among those, the most "
                   "coherent top words are the easiest to interpret (Lau et al. 2014).")
        model.append(_p("Topics", str(k), how, "topics.k", "topics.selection", "topics.min_stability", "fine.k",
                        why=why, refs=("greene2014", "lau2014")))
        model.append(_p("Candidates", ", ".join(map(str, t.k_grid)),
                        f"Numbers of topics compared. Each k is fitted once as reference and {t.runs} times on "
                        f"random {t.subsample:.0%} samples. Stability: agreement of the top {t.top_n} words between "
                        "these fits. Coherence: NPMI of the top words in held-out papers.",
                        "topics.k_grid", "topics.runs", "topics.subsample", "topics.top_n",
                        why="Fits on random subsamples show how much the topics depend on the sample (Greene et al. "
                            "2014); NPMI of top words agrees best with human judgements of coherence (Lau et al. "
                            "2014; Bouma 2009).", refs=("greene2014", "lau2014", "bouma2009")))
        row = next((r for r in st("train").get("sweep", []) if r.get("k") == k), None)
        if row:
            model.append(_p("Scores", f"coherence {row['npmi']:.3f}, stability {row['stability']:.2f}, "
                            f"diversity {row['diversity']:.2f}",
                            "Scores of the chosen k (NPMI coherence, top-word stability, share of unique top "
                            "words).", "topics",
                            why="Diversity checks that topics do not repeat each other (Dieng et al. 2020).",
                            refs=("lau2014", "greene2014", "dieng2020")))
    labels = cfg.export.labels_file
    model.append(_p("Topic names", "named by hand" if labels is not None else "top words",
                    "Topic names from a list written for this map." if labels is not None else
                    "Each topic is named by its three top words.", "export.labels_file"))
    model.append(_p("Topic vectors", "non-negative least squares",
                    "Each paper's topic mixture: its TF-IDF vector solved against the fixed topics, normalised "
                    "to sum to 1.", "embed",
                    why="With fixed topics and an exact solver a text always gets the same topic vector, and new "
                        "texts can be placed in the same space (Lawson & Hanson 1974).", refs=("lawson1974",)))
    return {"group": "Topic model", "parameters": model}


def _trajectories(cfg: RunConfig, st) -> dict:
    traj, tinfo = [], st("trajectories")
    for e, c in cfg.trajectories.entities.items():
        rule = (f"at least {c.min_papers} paper{'s' if c.min_papers != 1 else ''} in a year, "
                f"in at least {c.min_years} year{'s' if c.min_years != 1 else ''}")
        desc = (f"{'An' if e[0] in 'aeiou' else 'A'} {e} gets a trajectory point in each year with "
                f"{c.min_papers}+ papers, and a trajectory when it "
                f"has {c.min_years}+ points. Position: {_SMOOTHING[c.smoothing].format(window=c.window)}"
                + (", papers counted fractionally among their authors" if c.fractional else "") + ".")
        why = ("A yearly position needs enough papers to be reliable; a window of years evens out years with few "
               "papers. A trajectory is the path of these positions through the topic space (Schaefermeier et al. "
               "2021).")
        refs: tuple[str, ...] = ("schaefermeier2021",)
        if e == "author":
            why += (" Publication rates differ several-fold between fields (D'Angelo & Abramo 2015; Piro et al. "
                    "2013), so the author threshold is set per field, to about the 95th percentile of papers per "
                    "author and year: the output of an established researcher.")
            refs += ("dangelo2015", "piro2013")
            if c.max_authors is not None:
                desc += f" Papers with more than {c.max_authors} authors are left out."
                why += " Members of large collaborations would otherwise count as prolific authors."
        n = tinfo.get(e)
        traj.append(_p(f"{e.capitalize()}s", rule + (f" ({_n(n)} {e}s)" if n is not None else ""), desc,
                       f"trajectories.entities.{e}", why=why, refs=refs))
    pw = cfg.trajectories.pairwise
    if pw.enabled:
        traj.append(_p("Lead/lag", f"{pw.max_entities} largest {pw.entity_type}s, up to {pw.max_lag} years",
                       "Pairs of trajectories compared for which one follows the other with a time lag.",
                       "trajectories.pairwise",
                       why="Shows whether one venue takes up topics earlier than another; limited to the largest, "
                           "whose positions are reliable."))
    return {"group": "Trajectories", "parameters": traj}


def _map(cfg: RunConfig, st) -> dict:
    pr, pinfo = cfg.projection, st("project")
    params = {"umap": f"{pr.n_neighbors} neighbours, min. distance {pr.min_dist}",
              "tsne": f"perplexity {pr.perplexity}", "pca": "first two components"}[pr.method]
    return {"group": "Map", "parameters": [
        _p("Projection", f"{pr.method.upper() if pr.method != 'tsne' else 't-SNE'} ({params})",
           "2D map of the topic vectors" + (" under Hellinger distance (square roots of the topic shares)"
                                            if pr.hellinger else "") + ".",
           "projection.method", "projection.n_neighbors", "projection.min_dist", "projection.perplexity",
           "projection.hellinger",
           why="Topic vectors are distributions; Hellinger distance between them is bounded and is not dominated "
               "by the largest topic shares. UMAP keeps neighbourhoods and scales to millions of points (McInnes "
               "et al. 2018). Distances between far-apart regions of the map are not meaningful.",
           refs=("mcinnes2018",) if pr.method == "umap" else ()),
        _p("Fitted on", f"{_n(pinfo.get('fit', pr.fit_sample))} papers",
           "Random papers the map is fitted on; all other papers and trajectory points are placed into it.",
           "projection.fit_sample",
           why="Fitting on a sample is much faster; placing everything else into the same map keeps papers and "
               "trajectories in one coordinate system."),
    ]}


def _network(cfg: RunConfig, st) -> dict:
    nw, ninfo, an = cfg.network, st("network"), cfg.analysis
    years = ninfo.get("years") or [nw.min_year, nw.max_year]
    net = [
        _p("Years", f"{years[0] or 'first'}–{years[1] or 'last'}", "Years of the topic collaboration graph.",
           "network.min_year", "network.max_year"),
        _p("Authors", f"at least {nw.min_author_papers} papers"
           + (f" ({_n(ninfo['authors'])} authors)" if "authors" in ninfo else ""),
           f"Authors in the graph. Papers with more than {nw.max_authors_per_paper} authors are left out."
           if nw.max_authors_per_paper else "Authors in the graph.",
           "network.min_author_papers", "network.max_authors_per_paper",
           why="Authors with few papers add many nodes but little structure; large collaborations would link "
               "hundreds of authors through one paper."),
        _p("Topics per paper", str(ninfo.get("top_k_topics", nw.top_k_topics)),
           "Each paper links its authors in its strongest topics.", "network.top_k_topics",
           why="A paper contributes to several topics; its weakest topic shares are noise."),
        _p("Expertise", f"{nw.window}-year window" if nw.expertise == "window" else "cumulative",
           "An author's papers in a topic within the window; co-authorship edges point from higher to lower "
           "expertise.", "network.window", "network.expertise",
           why="Oriented edges show how expertise passes between co-authors and topics; a window lets expertise "
               "fade when an author leaves a topic."),
    ]
    if an.communities.enabled:
        com = an.communities
        net.append(_p("Communities", f"{com.method.capitalize()}, resolution {com.resolution}",
                      "Community detection on each year's graph.", "analysis.communities",
                      why="Leiden guarantees connected communities, unlike Louvain (Traag et al. 2019)."
                      if com.method == "leiden" else "", refs=("traag2019",) if com.method == "leiden" else ()))
    if an.sna.enabled and "avg_shortest_path" in an.sna.measures:
        net.append(_p("Average shortest path", f"exact up to {_n(an.sna.path_sources)} authors, else estimated from "
                      f"{_n(an.sna.path_sources)} random start authors",
                      "The mean number of co-authorship steps between two authors of the largest connected part of a "
                      "year's graph (edges without direction), for the whole field and for each topic. It describes "
                      "only that part: where few authors write together (e.g. in philosophy) it holds under 1 % of "
                      "the authors (giant_fraction), and the value says little about the field as a whole.",
                      "analysis.sna.measures", "analysis.sna.path_sources",
                      why="Short paths mean that knowledge can travel between any two researchers through few "
                          "co-authors (a \"small world\"). The exact value needs a search from every author, which "
                          "takes days on graphs with 100,000s of authors; searches from a random sample of start "
                          "authors give an unbiased estimate (Eppstein & Wang 2004) whose error is far below the "
                          "year-to-year changes.", refs=("eppstein2004",)))
    if an.pagerank.enabled:
        net.append(_p("PageRank", "weighted" if an.pagerank.weighted else "unweighted",
                      "PageRank of authors on the reversed graph, so that it ranks the authors expertise flows "
                      "from.", "analysis.pagerank",
                      why="PageRank counts how much an author is linked to by authors who are themselves central "
                          "(Page et al. 1999).", refs=("page1999",)))
    return {"group": "Collaboration network", "parameters": net}


def _derived(cfg: RunConfig, st, resolution: str, named: bool) -> dict:
    nov, ind, ninfo, iinfo = cfg.novelty, cfg.indicators, st("novelty"), st("indicators")
    out = [
        _p("Topic names", "names written by Claude (Anthropic's AI model)" if named else "the top three terms",
           ("Each topic has a short name next to its top terms. The names were written by Claude, an AI model, after "
            "reading each topic's top terms and the titles of its most typical papers; the terms are what the topic "
            "model found, the name is an interpretation of them." if named else
            "Topics are labelled by their three most important terms."),
           "export.labels_file",
           why="A list of terms is hard to scan and to remember; a name makes topics easy to find. Names can "
               "simplify or miss part of a topic, so the terms are always shown with them."),
        _p("Topic life cycle", f"{ind.smoothing}-year moving average; growing or declining at ±{ind.change:.0%} "
           f"over {ind.recent} years",
           f"From a topic's yearly share of all papers, smoothed over {ind.smoothing} years: the peak year; the rise "
           f"year, when the share first reached half its peak; and the recent change, the mean share of the last "
           f"{ind.recent} years against the {ind.recent} years before. A topic is growing or declining if that change "
           f"is at least {ind.change:.0%} up or down, and emerging if it is growing and rose in the last "
           f"{2 * ind.recent} years.", "indicators.smoothing", "indicators.recent", "indicators.change",
           why="Shares, not paper counts, since the number of papers grows in every field; smoothing keeps single "
               "years with few papers from deciding the peak."),
        _p("Related topics", f"co-occurrence in papers (share ≥ {nov.threshold:g}); similar vocabulary",
           f"Related topics appear together in papers more often than chance: lift = P(both) / (P(A)·P(B)), where a "
           f"paper has a topic if its share is at least {nov.threshold:g}. Similar topics have similar term weights "
           "(cosine of the topic vectors)." + (" Each topic of this finer resolution has as parent the topic of the "
                                               "main resolution with the most similar term weights."
                                               if resolution != "main" else ""),
           "novelty.threshold",
           why="Co-occurrence shows which subjects are combined in research; vocabulary similarity shows topics that "
               "use the same words, e.g. a method and its application."),
        _p("Paper novelty", f"rarest topic pair against the {nov.window} preceding years",
           f"A paper has a topic if the topic's share in it is at least {nov.threshold:g}. For each pair of topics, "
           "the "
           f"papers of the {nov.window} years before the paper's year give the observed number of papers with both "
           "topics and the number expected if topics were combined at random; their ratio is the pair's commonness. "
           "A paper's novelty is −log10 of the commonness of its rarest pair: 0 = as common as expected, 1 = ten "
           "times rarer, −1 = ten times more common. A paper combines topics unusually if its rarest pair occurs at "
           f"most {nov.unusual_ratio:.0%} as often as expected. Papers with only one topic have no novelty score."
           + (f" {_n(ninfo['scored'])} of {_n(ninfo['papers'])} papers have a score." if "scored" in ninfo else ""),
           "novelty",
           why="Novel research recombines existing knowledge in unusual ways. Lee, Walsh & Wang (2015) measure this "
               "by how rarely the journals a paper cites are cited together, Uzzi et al. (2013) by atypical journal "
               "pairs; here the topics of the paper's own text take the place of the cited journals, so no "
               "references are needed. The preceding years are the knowledge a paper builds on. Novelty measures "
               "are known to reward rare combinations that are also of low quality (Wang, Veugelers & Stephan "
               "2017), so the score describes papers, it does not rate them.",
           refs=("lee2015", "uzzi2013", "wang2017")),
        _p("Citation impact", f"percentile within the publication year; the last {ind.citation_lag} years are left "
           "out of averages",
           "A paper's citation count (from OpenAlex, counted over all of OpenAlex) as a percentile among the papers "
           "of the same corpus and publication year: 50 = median, 90 or more = the top 10 % most cited.",
           "indicators.citation_lag",
           why="Older papers had more time to be cited; percentiles within a year compare papers of the same age "
               "and are not dominated by a few highly cited papers as means of counts are (Waltman 2016). Recent "
               "papers have had too little time, so averages leave out the latest years.", refs=("waltman2016",)),
        _p("Field indicators", "per year: effective topics, topics per paper, change, team size, newcomers, novelty",
           "Effective topics: exp of the Shannon entropy of the year's topic shares, the number of equally large "
           "topics that would be as diverse. Topics per paper: the mean number of topics with a share of at least "
           f"{nov.threshold:g} in a paper. "
           "Change: Hellinger distance between the topic shares of the year and the year before. Team size: authors "
           "per paper. Newcomers: authors without an earlier paper in this corpus (not shown for the first "
           f"{ind.recent} years). Novelty: mean novelty and share of papers with an unusual topic pair.",
           "indicators",
           why="Effective numbers (Hill numbers) are diversity measures that behave like counts: doubling them means "
               "twice as many equally common topics (Hill 1973, Jost 2006).", refs=("hill1973", "jost2006")),
    ]
    return {"group": "Derived measures", "parameters": out}


def _attach_notes(groups: list[dict], notes: dict[str, str]) -> None:
    """Each note goes to the parameter whose config key matches it most closely ("trajectories.entities.author"
    -> the Authors parameter); notes without such a parameter form a group of their own."""
    params = [p for g in groups for p in g["parameters"]]
    rest = []
    for key, note in notes.items():
        best, best_len = None, -1
        for p in params:
            for c in p["config"]:
                if (key == c or key.startswith(c + ".")) and len(c) > best_len:
                    best, best_len = p, len(c)
        if best is None:
            rest.append(_p(key, "", note, key))
        else:
            best["note"] = f"{best['note']} {note}".strip()
    if rest:
        groups.append({"group": "Notes", "parameters": rest})


def cited(groups: list[dict], methods_doc: dict | None = None) -> list[str]:
    """Reference ids cited by the parameters (and the stop-word list), in order of first citation."""
    ids = [r for g in groups for p in g["parameters"] for r in p["references"]]
    if methods_doc:
        ids += methods_doc.get("vocabulary", {}).get("stopwords", {}).get("references", [])
    unknown = set(ids) - REFERENCES.keys()
    if unknown:
        raise KeyError(f"unknown references {sorted(unknown)}; add them to tst/references.py")
    return list(dict.fromkeys(ids))


# ---------------------------------------------------------------------------------------------------------------
# methods.json and vocabulary.json

def methods(cfg: RunConfig, stages: dict, vectorizer_dir: Path | None) -> dict:
    """The lists behind the parameters whose `details` point here; see schema/methods.md."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    from .text import FOREIGN_STOPWORDS, HYPHEN_PREFIXES, JOINED_PHRASES, TITLE_AMBIGUOUS
    from .topics import curation

    pre, v = cfg.preprocess, cfg.vectorize
    info, vec = stages.get("preprocess") or {}, stages.get("vectorize") or {}
    drop_labels = pre.drop_pattern_labels()
    funnel = [{"reason": r, "label": label, "papers": int(info[r])} for r, label in _REASONS.items() if r in info]
    funnel += [{"reason": r, "label": r.replace("_", " "), "papers": int(n)} for r, n in info.items()
               if r not in _REASONS and r not in ("ok", "titles_ignored")]
    cleaning = {
        "papers": {"kept": int(info.get("ok", 0)), "removed": funnel,
                   "titles_ignored": int(info["titles_ignored"]) if "titles_ignored" in info else None},
        "drop_patterns": [{"group": drop_labels.get(p), "pattern": p} for p in pre.all_drop_patterns()],
        "strip_patterns": [{"group": PATTERN_LABELS.get(p), "pattern": p} for p in pre.strip_patterns],
        "hyphen_prefixes": list(HYPHEN_PREFIXES) if pre.join_prefixes else [],
        "joined_phrases": [{"word": w, "pattern": p} for w, p in JOINED_PHRASES.items()] if pre.join_prefixes else [],
        "words": {"latin_only": pre.latin_only, "fold_accents": pre.fold_accents,
                  "join_apostrophes": pre.join_apostrophes,
                  "max_other_script": pre.language_filter.max_other_script
                  if pre.language_filter.method != "none" else None},
        "function_words": {
            "english": sorted(ENGLISH_STOP_WORDS),
            "foreign": {_LANGUAGES.get(lang, lang): words.split() for lang, words in FOREIGN_STOPWORDS.items()}
            if pre.foreign_stopwords else {},
            "added": list(pre.extra_stopwords),
        },
        "foreign_titles": {
            "ignored": pre.language_filter.method == "stopword_vote" and pre.language_filter.ignore_foreign_titles,
            "not_counted": sorted(TITLE_AMBIGUOUS),
        },
    }
    vocab: dict[str, Any] = {"terms": vec.get("features"), "papers": vec.get("documents"),
                             "min_df": v.min_df, "max_df": v.max_df, "max_features": v.max_features}
    if cfg.topics.model_path is None:
        variants = _load_variants(vectorizer_dir)
        terms = _load_terms(vectorizer_dir)
        rank = {t: i for i, t in enumerate(terms)}
        groups: dict[str, list[str]] = {}
        for form, term in variants.items():
            groups.setdefault(term, []).append(form)
        examples = sorted(groups, key=lambda t: rank.get(t, len(rank)))[:60]
        vocab["variants"] = {
            "methods": list(v.conflate),
            "merged_forms": vec.get("variants_merged"), "groups": vec.get("variant_groups"),
            "rules": [r for r in _VARIANT_RULES if r["method"] in v.conflate],
            "not_plurals": sorted(curation.NO_PLURAL) if "plural" in v.conflate else [],
            "irregular_plurals": dict(curation.IRREGULAR_PLURALS) if "plural" in v.conflate else {},
            "examples": [{"term": t, "forms": sorted(groups[t])} for t in examples],
        }
        sw = curation.academic_stopword_list() if v.academic_stopwords else None
        vocab["stopwords"] = {
            "academic_list": bool(sw),
            "date": sw["date"] if sw else None,
            "method": sw["method"] if sw else None,
            "references": list(sw["references"]) if sw else [],
            "categories": [{"name": c["name"], "words": list(c["words"])} for c in sw["categories"]] if sw else [],
            "not_stopwords": [{"words": list(c["words"]), "reason": c["reason"]} for c in sw["not_stopwords"]]
            if sw else [],
            "added": list(v.stopwords), "kept": list(v.keep), "removed": vec.get("stopwords_removed"),
        }
    derived = {"novelty": cfg.novelty.model_dump(mode="json"), "indicators": cfg.indicators.model_dump(mode="json"),
               "avg_shortest_path_sources": cfg.analysis.sna.path_sources
               if "avg_shortest_path" in cfg.analysis.sna.measures else None,
               "novelty_info": stages.get("novelty"), "indicators_info": stages.get("indicators")}
    return {"cleaning": cleaning, "vocabulary": vocab, "derived": derived}


_VARIANT_RULES = [
    {"method": "plural", "rule": "-s", "example": "models → model, argues → argue",
     "description": "Final -s removed, except after s, u and i (\"process\", \"focus\", \"analysis\") and in "
                    "the listed words that are no plurals."},
    {"method": "plural", "rule": "-ies → -y", "example": "studies → study",
     "description": "Not after a and e (\"species\" stays)."},
    {"method": "plural", "rule": "-es", "example": "processes → process, approaches → approach",
     "description": "Only after s, x, z, ch, sh and o, as in English spelling."},
    {"method": "plural", "rule": "-ves → -f, -fe", "example": "leaves → leaf, lives → life", "description": ""},
    {"method": "plural", "rule": "Latin and Greek plurals", "example": "spectra → spectrum, analyses → analysis",
     "description": "A fixed list (irregular_plurals); pairs whose meanings drifted apart (media, data) are left "
                    "out."},
    {"method": "spelling", "rule": "-ise, -isation → -ize, -ization", "example": "organisation → organization",
     "description": ""},
    {"method": "spelling", "rule": "-yse → -yze", "example": "analyse → analyze", "description": ""},
    {"method": "spelling", "rule": "-our → -or", "example": "behaviour → behavior, colour → color",
     "description": ""},
    {"method": "spelling", "rule": "-tre → -ter", "example": "centre → center, fibre → fiber", "description": ""},
    {"method": "spelling", "rule": "-ogue → -og", "example": "catalogue → catalog", "description": ""},
    {"method": "spelling", "rule": "-ll- → -l-", "example": "modelling → modeling, travelled → traveled",
     "description": ""},
    {"method": "spelling", "rule": "ae, oe → e", "example": "haemoglobin → hemoglobin, oestrogen → estrogen",
     "description": "In medical and scientific words (paed-, aetiol-, oesoph-, palaeo-, ...)."},
    {"method": "spelling", "rule": "single words", "example": "programme → program, ageing → aging, grey → gray",
     "description": ""},
]
for _r in _VARIANT_RULES:
    _r["description"] = (_r["description"] + " " if _r["description"] else "") + \
        "Applied only when the resulting word also occurs in the corpus."


def vocabulary(stages: dict, vectorizer_dir: Path | None) -> dict | None:
    """vocabulary.json: the terms in order of document frequency, their document frequencies and the merged
    forms ({form: term}); see schema/methods.md."""
    terms = _load_terms(vectorizer_dir)
    if not terms:
        return None
    n_docs = (stages.get("vectorize") or {}).get("documents")
    out: dict[str, Any] = {"terms": terms}
    idf_path = vectorizer_dir / "idf.npy"
    if n_docs and idf_path.exists():
        idf = np.load(idf_path).astype(np.float64)  # ln((1+N)/(1+df)) + 1
        out["document_frequency"] = np.rint((1 + n_docs) / np.exp(idf - 1.0) - 1.0).astype(np.int64).tolist()
        out["papers"] = int(n_docs)
    out["variants"] = _load_variants(vectorizer_dir)
    return out


def _load_terms(folder: Path | None) -> list[str]:
    import json
    p = folder / "vocabulary.json" if folder else None
    return json.loads(p.read_text()) if p and p.exists() else []


def _load_variants(folder: Path | None) -> dict[str, str]:
    import json
    p = folder / "variants.json" if folder else None
    return json.loads(p.read_text()) if p and p.exists() else {}
