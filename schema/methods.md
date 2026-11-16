# How the bundle was made: parameters, decisions, references

Since schema 1.1 a bundle documents every setting and the reason for it, so that a web app can show an
"About this map" or "Methods" page without any code from this package. There are three places:

| where | what | size |
|---|---|---|
| `manifest.json` → `run.parameters` | every parameter as a decision: value, what it does, why, literature, notes | ~25 KB |
| `manifest.json` → `run.references` | the cited literature | ~6 KB |
| `methods.json` (`files.methods`) | the long lists behind some parameters: cleaning patterns, stop words, variant rules | ~50 KB |
| `vocabulary.json` (`files.vocabulary`) | the vocabulary of the topic model with document frequencies and merged word forms | ~1.2 MB |

`methods.json` and `vocabulary.json` are missing in bundles of schema 1.0; check `files`. Both resolutions of a
run share the text cleaning and the vocabulary, so these two files are identical in `<name>` and `<name>_fine`.

## run.parameters

A list of groups in pipeline order; show them in this order. Groups so far: `Corpus`, `Text cleaning`,
`Vocabulary`, `Topic model`, `Trajectories`, `Map`, `Collaboration network` (if the run has a network),
`Derived measures` (schema 1.4: Topic names, Topic life cycle, Related topics, Paper novelty, Citation impact,
Field indicators; [derived.md](derived.md)) and `Notes` (only for notes that match no parameter). Groups and
labels may grow over time. A web app showing a derived measure (a topic page, the field indicators, novelty)
should link to its parameter here: its `description` says how it is computed, its `rationale` why, with
`references`.

```json
{"group": "Vocabulary", "parameters": [
  {"label": "Academic stop words",
   "value": "489 words removed",
   "description": "Generic words of academic writing (\"study\", \"results\", ...) are removed with all their forms. ...",
   "rationale": "They are frequent in every field and appear in many topics without describing any. ... (Sarica & Luo 2021); ...",
   "references": ["schofield2017", "gerlach2019", "sarica2021"],
   "note": "",
   "details": "vocabulary.stopwords",
   "config": ["vectorize.academic_stopwords", "vectorize.stopwords", "vectorize.keep"]}]}
```

| key | type | meaning |
|---|---|---|
| `label` | string | name of the parameter |
| `value` | string | the chosen setting, ready to display (numbers formatted, counts of this run included) |
| `description` | string | what the setting does, in one or two sentences |
| `rationale` | string | why it was chosen; may be `""`. Cites literature in the text as "(Author Year)" |
| `references` | list of strings | ids of the literature cited in `rationale`, keys into `run.references`; may be empty |
| `note` | string | why *this corpus* deviates from the defaults (from the config's `export.notes`); usually `""` |
| `details` | string or null | a dotted path into `methods.json` with the full list (`"cleaning.drop_patterns"`, `"vocabulary.stopwords"`, `"vocabulary"`); null if none |
| `config` | list of strings | the config keys the parameter comes from, as dotted paths into `run.config` |

A display suggestion: label and value in a row; description below; rationale behind a "Why?" toggle with the
cited references linked; the note highlighted as specific to this corpus; a "Show list" link where `details` is
set.

## run.references

The literature cited by `run.parameters` (and by the stop-word list), in order of first citation:

```json
{"id": "schofield2016", "citation": "Schofield & Mimno 2016",
 "authors": "Schofield, A. & Mimno, D.", "year": 2016,
 "title": "Comparing Apples to Apple: The Effects of Stemmers on Topic Models",
 "venue": "Transactions of the Association for Computational Linguistics 4, 287–300",
 "url": "https://aclanthology.org/Q16-1021"}
```

`citation` is the short form used in the `rationale` texts. `url` is a DOI link where one exists, else another
stable link, or null.

## methods.json

```json
{"cleaning": {
   "papers": {"kept": 511566, "removed": [{"reason": "spam", "label": "junk text", "papers": 16886}, ...],
              "titles_ignored": 20114},
   "drop_patterns": [{"group": "E-book download spam", "pattern": "thank you (very much )?for (reading|downloading)"}, ...],
   "strip_patterns": [{"group": "Copyright and licence notices", "pattern": "all rights reserved\\.?"}, ...],
   "hyphen_prefixes": ["non", "anti", "multi", ...],
   "joined_phrases": [{"word": "wellbeing", "pattern": "well[- ]?being"}, ...],
   "words": {"latin_only": true, "fold_accents": true, "join_apostrophes": true, "max_other_script": 0.6},
   "function_words": {"english": ["a", "about", ...], "foreign": {"German": ["der", ...], ...}, "added": []},
   "foreign_titles": {"ignored": true, "not_counted": ["com", "della", ...]}},
 "vocabulary": {
   "terms": 50000, "papers": 511566, "min_df": 20, "max_df": 0.4, "max_features": 50000,
   "variants": {
     "methods": ["plural", "spelling"], "merged_forms": 46487, "groups": 45072,
     "rules": [{"method": "plural", "rule": "-es", "example": "processes → process, approaches → approach",
                "description": "Only after s, x, z, ch, sh and o, as in English spelling. Applied only when ..."}, ...],
     "not_plurals": ["aids", "bias", "news", ...],
     "irregular_plurals": {"analyses": "analysis", "spectra": "spectrum", ...},
     "examples": [{"term": "theory", "forms": ["theories"]}, ...]},
   "stopwords": {
     "academic_list": true, "date": "2026-11-04", "method": "From the 14 OpenAlex field runs ...",
     "references": ["sarica2021", "schofield2017", "gerlach2019"],
     "categories": [{"name": "Research acts, genres and parts of a paper", "words": ["paper", "article", ...]}, ...],
     "not_stopwords": [{"words": ["current"], "reason": "Electric current (physics, engineering)."}, ...],
     "added": [], "kept": ["analysis", "number"], "removed": 489}}}
```

**cleaning**

* `papers`: the paper counts of text cleaning. `kept` papers enter the topic model; `removed` lists the
  filters in the order they apply, each with the number of papers it removed. Reasons: `spam` (junk text, see
  `drop_patterns`), `boilerplate` (abstract occurs 5+ times), `duplicate`, `no_abstract`, `too_short`,
  `language` (not English). Only filters that removed papers in this corpus are listed. The papers on the map can
  be fewer than `kept` because of the years shown (`run.stages.embed`). `titles_ignored`: kept papers whose
  title was left out as not English (see `foreign_titles`); null in bundles before schema 1.3.
* `drop_patterns`: papers whose title and abstract match one of these (case-insensitive regular expressions,
  Python syntax) are removed. `group` says what the patterns catch; null for patterns a config added itself.
* `strip_patterns`: passages cut out of abstracts before tokenising (the rest is kept), same form.
* `hyphen_prefixes`: prefixes joined to the following word ("non-linear" → "nonlinear"); empty if off.
* `joined_phrases`: phrases written as one word (`word`), matched by `pattern` ("state-of-the-art" →
  "stateoftheart", which is an academic stop word); empty if off.
* `words`: how words are formed. `latin_only`: only words in the Latin alphabet are kept; `fold_accents`:
  accents are removed ("schrödinger" → "schrodinger"); `join_apostrophes`: English endings after an apostrophe
  are dropped ("kant's" → "kant") and words are joined at other apostrophes ("da'wah" → "dawah");
  `max_other_script`: papers whose abstract has a larger share of its letters in other scripts are removed as
  not English (null if the language filter is off).
* `function_words`: the stop lists of the tokenizer: scikit-learn's English list, the frequent function words
  of other languages (by language name; empty if off) and the config's own (`added`).
* `foreign_titles`: `ignored`: a title with more function words of one other language (`function_words.foreign`)
  than English ones is left out and the paper is kept with its abstract (an Indonesian or Spanish title over an
  English abstract); `not_counted`: name particles and words of English titles that do not count as foreign
  ("van der Waals", "von Neumann", "Yang-Mills", "Los Alamos").

**derived** (schema 1.4)

* `novelty`, `indicators`: the settings of paper novelty and of the field indicators and topic pages (as in
  `run.config`). `avg_shortest_path_sources`: start authors of the shortest-path estimate (null if not computed).
* `novelty_info`, `indicators_info`: run statistics, e.g. the number of papers with a novelty score.

**vocabulary**

* `terms`: size of the vocabulary; `papers`: papers it was counted on; `min_df`, `max_df`, `max_features`: the
  document-frequency limits (integers are paper counts, floats shares of the papers).
* `variants`: forms of a word counted as one word. `merged_forms` forms were merged into `groups` words in the
  whole corpus (most groups are rare words that are not in the vocabulary). `rules` lists the rules in the order
  of the `methods` they belong to (`plural`, `spelling`); `not_plurals` are words ending in -s that are kept as
  they are; `irregular_plurals` maps Latin and Greek plurals to their singular. `examples` are the merged forms of
  the 60 most frequent vocabulary words that have any; the complete mapping is in `vocabulary.json`.
* `stopwords`: the academic stop-word list. `categories` are the words removed (each with all its variant
  forms), `not_stopwords` are candidates deliberately left off with the reason, `method` and `references` say
  how the list was built, `date` is its version. `added` are stop words of this corpus's config, `kept` words of
  the list kept for this corpus (subjects in the field). `removed` is the number of vocabulary words removed in
  this corpus (null if unknown). `academic_list` is false if the list was switched off; then only `added` applies.
* For a run that reuses another run's model (`topics.model_path`) only `terms`, `papers` and the limits are set.

## vocabulary.json

Compact JSON (no indentation).

```json
{"terms": ["work", "philosophy", "theory", ...],
 "document_frequency": [116856, 106576, 104108, ...],
 "papers": 511566,
 "variants": {"theories": "theory", "behaviour": "behavior", "models": "model", ...}}
```

* `terms`: all words of the topic model, most frequent first. Topic word lists (`topics.json` `top_terms`) use
  these words.
* `document_frequency[i]`: number of papers containing `terms[i]` (or one of its forms), out of `papers`.
* `variants`: every merged form that maps to a vocabulary word, `{form: term}`. A word that is neither in
  `terms` nor in `variants` is not part of the model: too rare or too frequent, a stop word (see
  `methods.json` `vocabulary.stopwords`, which lists the base forms only), or removed during text cleaning.

A word lookup ("Is *genetics* in the model?") can thus answer: in the vocabulary as *genetic* (merged form), with
its document frequency, or removed as an academic stop word with the category.
