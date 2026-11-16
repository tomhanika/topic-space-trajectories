# Trajectories

Part of the bundle schema 1.0 (see [README.md](README.md) for conventions). A trajectory is the path of
an entity (a venue or an author) through topic space over the years: one point per year, at the
smoothed topic mix of the entity's papers in a window of years around it.

## trajectories.parquet

One row per point (entity × year), sorted by entity, then year. Only entities with enough papers are
included (thresholds per corpus, e.g. venues with ≥ 20 papers and authors with ≥ 10 papers over ≥ 5
years).

| column | type | |
|---|---|---|
| entity_type | string | `venue` or `author` |
| entity_id | string | venue name, or author id (see README) |
| entity_name | string | display name |
| year | int32 | |
| n_papers | int64 | papers in the smoothing window |
| weight | float64 | sum of paper weights (differs from n_papers if papers are counted fractionally) |
| centroid | fixed_size_list<float32, k> | smoothed topic vector |
| main_topic | int32 | argmax of centroid |
| main_topic_share | float32 | its value |
| entropy | float64 | normalised Shannon entropy of centroid: 0 = one topic, 1 = all topics equally |
| effective_topics | float32 | `exp(entropy)` in topics, i.e. how many topics the entity effectively covers |
| step | float32 \| null | distance to the entity's previous point; null for its first point |
| speed | float64 \| null | step per year since the previous point |
| novelty | float64 \| null | distance to the mean of all the entity's earlier points |
| mainstream_distance | float32 | distance to the corpus-wide topic mix of that year: how unusual the entity is |
| x, y | float32 | map position of the point |

Years can be missing inside a trajectory (gaps); draw them as such or connect over them, but do not
assume consecutive years.

## trajectories_venue.json, trajectories_author.json

A compact copy of the points for the browser. A file only exists for entity types with at most 5000
entities, so check `manifest.files`; otherwise load the Parquet file.

```json
[{"id": "ARS journal", "name": "ARS journal",
  "points": [{"year": 1960, "x": 5.21, "y": 0.65, "n": 24, "main_topic": 12, "centroid": [0.03824, ...]}, ...]}]
```

`n` is `n_papers`; centroid values are rounded to 5 decimals.

## entities.parquet

One row per entity with summary statistics, for lists, search and sorting ("which venues moved most?").

| column | |
|---|---|
| entity_type, entity_id, entity_name | as above |
| n_points | number of trajectory points |
| first_year, last_year | |
| papers | total papers |
| path_length | sum of all steps |
| displacement | distance between the first and the last point |
| straightness | displacement / path_length: 1 = moved in one direction, near 0 = wandered; null for one point |
| max_step, max_step_year | the largest step and the year it ended in (max_step is null for one point) |
| mean_speed | null for one point |
| mean_entropy | how broad the entity is on average |
| mean_mainstream_distance | how unusual the entity is on average |
| main_topic_changes | how often the main topic changed |
| dominant_topic | most frequent main topic |
| radius | mean distance of the points from their mean: how much the entity varies overall |

## pairwise.parquet

Pairs among the largest entities of one type (the 200–300 largest venues): who leads, who follows. Only
pairs that coexist for a minimum number of years are listed.

| column | type | |
|---|---|---|
| entity_type | string | the type of both entities |
| a, a_name, b, b_name | string | ids and names |
| overlap_years | int64 | years in which both have a point |
| mean_distance | float64 | mean same-year distance: how similar the two are |
| lead_score | float64 | > 0: `b` later moves to where `a` was before, i.e. `a` leads; < 0: `b` leads |
| best_lag | int64 | the lag in years with the clearest lead; 0 if there is none |

Each unordered pair appears once; for the reverse direction, swap `a` and `b` and negate `lead_score`.
