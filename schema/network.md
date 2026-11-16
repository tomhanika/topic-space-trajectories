# Topic collaboration network

Part of the bundle schema 1.0 (see [README.md](README.md) for conventions). All files are in `network/`
and are listed in `manifest.files` under `network/<name>`.

The underlying graph is a directed co-authorship graph per year (using a sliding window of years) and per
topic: an edge in topic *t* runs from the co-author with more expertise in *t* to the one with less.
Expertise in *t* is the author's summed weight of *t* over their own papers in the window (each paper
keeps its few strongest topics). Only authors with a minimum number of papers are in
the graph (10 in most corpora). The graph itself is not exported; these files are its analyses per year.

| file | columns | content |
|---|---|---|
| authors.parquet | `author` int64, `name` string, `n_papers` uint32 | authors in the graph. `author` is the numeric author id; trajectories use the same id as a string |
| authors_per_topic.parquet | `year`, `main_topic`, `len` | number of active authors per main topic and year |
| flows.parquet | `year`, `source_topic`, `target_topic`, `weight` float64 | intertopic flow: weight of the edges in topic *s* from authors whose main topic is *s* to authors whose main topic is *t*. The diagonal (`s == t`) is flow within a topic |
| flows_top.json | `[{year, source_topic, target_topic, weight}]` | the 25 strongest flows between different topics per year, ready for Sankey or chord diagrams |
| pagerank.parquet | `year`, `topic`, `rank`, `author`, `name`, `pagerank` | most central authors per year and topic (`topic = -1`: whole graph). PageRank on reversed edges, so experts rank high; `rank` starts at 1 |
| sna.parquet | `year`, `topic`, `nodes`, `edges`, `density`, `mean_degree`, `max_degree`, `transitivity`, `avg_local_transitivity`, `giant_fraction`, `avg_shortest_path` | graph statistics per year and topic (`topic = -1`: whole graph); `giant_fraction` is the share of nodes in the largest connected component; `avg_shortest_path` (schema 1.4) the mean number of co-authorship steps between two authors of that component, edges without direction, exact up to 500 authors and estimated from breadth-first searches from 500 random authors above (`run.parameters`, Collaboration network) |
| communities.parquet | `year`, `main_topic`, `communities`, `community_size`, `largest_community`, `max_coreness_topic_subgraph`, `max_coreness_in_graph` | Leiden communities, each assigned to the most common main topic of its members: per year and topic, the number of communities, their total and largest size, and the maximal k-core |
| modularity.parquet | `year`, `modularity` | modularity of the Leiden partition per year |

Early years have few authors, so their statistics are noisy. OpenAlex author records occasionally are
institutions (e.g. "USDOE, Washington, DC"); they can show up in PageRank lists.
