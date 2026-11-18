# Documents

Part of the bundle schema 1.0 (see [README.md](README.md) for conventions). One row per paper placed on
the map.

## documents.parquet

A random sample of at most 200,000 documents (`export.max_docs`) when the corpus is larger, which is the
case for every OpenAlex field. The full corpus is behind the density background
([map.md](map.md)), so the map looks right even though only the sample can be clicked. Sorted by `doc_id`.

| column | type | |
|---|---|---|
| doc_id | int64 | stable within one run, not across runs |
| year | int32 | publication year |
| venue | string \| null | journal/conference name; null for preprints and repository-only works |
| title | string | |
| citations | int64 | citation count at download time |
| authors | list<string> | author names in byline order |
| source_id | string | only in OpenAlex bundles: the work id, e.g. `"W581096"` |
| main_topic | int16 | the largest entry of `theta`; use it for the point colour |
| strength | float32 | how well the topic model covers the document (sum of raw NMF weights); small values mean a weak fit, e.g. for drawing such points fainter |
| theta | fixed_size_list<float32, k> | topic vector |
| x, y | float32 | map position |
| novelty | float32 \| null | how unusual the paper's combination of topics is (schema 1.4, [derived.md](derived.md)); null for papers with one topic or in the first years |
| novelty_pct | float32 \| null | novelty as a percentile within the publication year (0–100) |
| unusual | bool | the paper has a topic pair that occurs at most half as often as expected |
| citation_pct | float32 | citation count as a percentile within the corpus and publication year (0–100) |
| references, references_resolved, kb_surprise, kb_surprise_pct | | references and knowledge-base surprise (schema 1.5, [citations.md](citations.md)); null where `tst citations` has not run |
| url | string | link to the paper, from `export.url_template` (OpenAlex: `https://openalex.org/W…`; CSV input e.g. a DOI link). Only present if a template is set; null for papers without a `source_id` |

Venue names are as the source database spells them and can have leading or trailing spaces; trim them
before displaying.

The `venue` and `authors` columns connect documents to trajectories: a venue's `entity_id` is the venue
name. Author names are not unique, so documents cannot be matched to author ids exactly.
