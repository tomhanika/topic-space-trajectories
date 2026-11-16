# Naming topics

Every topic of every map has a short name, shown in the web app next to its top terms and marked there as an
AI interpretation. Names are stored in `configs/labels/<map>.tsv` and `<map>_fine.tsv` together with the five top
terms they were written for; tst drops all names of a file when a retrained model no longer matches them.

## Procedure (after a map was retrained)

1. Before the recompute, keep the named topics: `scripts/naming/backup.sh <map> ...` copies the bundle's
   `topics.json` and `topic_pages.json` to `runs/_naming_backup/<map>/`.
2. Recompute the map (`tst run configs/<map>.yaml`). The export then has no names (the old file doesn't match).
3. Worksheets: `python scripts/naming/worksheet.py runs/<map> --old runs/_naming_backup/<map> > <map>.md`,
   and with `--fine` for the fine map (its topics show their parent topic).
4. A reviewer (person or agent) writes `id<TAB>name` for every topic, following the rules below.
5. `python scripts/naming/write_labels.py runs/<map> names.tsv configs/labels/<map>.tsv [--fine]` checks that
   every topic is named once and adds the top terms.
6. Re-export: `tst run configs/<map>.yaml --stages export --force`.

## Rules for names

- Name what the papers are about. Read the terms **and** the typical papers; the terms alone mislead (one word can
  carry two meanings, and junk records can dominate the terms).
- 2 to 6 words, sentence case, no final period, no quotes or tabs. Usually nouns joined by "and" or commas:
  "Constitutionalism and separation of powers", "Tokamak plasmas and turbulence", "GaN HEMTs and power diodes".
- Specific to the map: don't repeat the map's own subject ("Law ..." in the law map) and avoid empty words
  ("research", "studies", "issues", "aspects", "various"). A method or a language can be the name when it binds the
  topic: "Indonesian normative legal research", "Ab initio and DFT calculations".
- Names within a map are distinct. In the fine map, siblings under one parent need names that tell them apart.
- A topic mixing unrelated meanings gets both, honestly: "Femoral head and head and neck cancer".
- Records that are not research (court briefs, newsletters, scraped web pages, front matter, programmes) are named
  as what they are: "Utah appellate court briefs", "ACM Digital Library page text". Report each such topic with
  two or three example titles, so it can be filtered later (`preprocess.extra_drop_patterns`).
- Earlier names are candidates only. Keep one if the typical papers still fit it; a high cosine is not enough,
  because retraining splits and merges topics and the shared terms can hide the change. Check the second
  candidate too: a split topic often matches two earlier ones.

## Output

- `names.tsv`: one line `id<TAB>name` per topic, every id of the worksheet exactly once.
- A short note: topics whose earlier name was changed and why, topics that are not research, and topics that
  were hard to name.
