"""Per-year analyses of the topic collaboration graph (old topicflow.py, computepageranks.py,
computesnameasures.py, compute_communities.py)."""

from __future__ import annotations

import logging
from pathlib import Path

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import igraph as ig
import numpy as np
import polars as pl

from .config import AnalysisConfig
from .run import limited_threads

log = logging.getLogger("tst.analysis")


def load_year(network_dir: Path, year: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    edges = pl.scan_parquet(network_dir / "edges.parquet").filter(pl.col("year") == year).collect()
    expertise = pl.scan_parquet(network_dir / "expertise.parquet").filter(pl.col("year") == year).collect()
    return edges, expertise


def main_topics(expertise: pl.DataFrame) -> pl.DataFrame:
    """Each active author's main topic: the topic with the highest expertise (ties: lowest topic id)."""
    return (expertise.sort(["author", "weight", "topic"], descending=[False, True, False])
            .group_by("author", maintain_order=True).first()
            .select("author", pl.col("topic").alias("main_topic")))


def intertopic_flow(edges: pl.DataFrame, mt: pl.DataFrame, k: int) -> np.ndarray:
    """flow[s, t]: weight of edges of topic s from authors with main topic s to authors with main topic t."""
    e = (edges.join(mt.rename({"author": "src", "main_topic": "ms"}), on="src")
         .join(mt.rename({"author": "dst", "main_topic": "md"}), on="dst")
         .filter(pl.col("topic") == pl.col("ms")))
    flow = np.zeros((k, k))
    np.add.at(flow, (e["ms"].to_numpy(), e["md"].to_numpy()), e["weight"].to_numpy())
    return flow


def to_graph(edges: pl.DataFrame, nodes: np.ndarray, directed: bool = True) -> ig.Graph:
    """Simple graph on `nodes` (author ids); multi-edges of different topics are summed."""
    index = pl.DataFrame({"author": nodes, "v": np.arange(len(nodes), dtype=np.int64)})
    e = (edges.group_by("src", "dst").agg(pl.col("weight").sum())
         .join(index.rename({"author": "src", "v": "s"}), on="src")
         .join(index.rename({"author": "dst", "v": "t"}), on="dst"))
    g = ig.Graph(n=len(nodes), edges=np.column_stack([e["s"].to_numpy(), e["t"].to_numpy()]).tolist(),
                 directed=directed)
    g.es["weight"] = e["weight"].to_list()
    g.vs["author"] = nodes.tolist()
    if not directed:
        g.simplify(combine_edges={"weight": "sum"})
    return g


def mean_shortest_path(u: ig.Graph, sources: int, seed: int) -> float:
    """Mean shortest-path length (co-authorship steps) between two nodes of the largest connected component of the
    undirected graph `u`. Exact up to `sources` nodes; above, the mean over breadth-first searches from `sources`
    random nodes, an unbiased estimate (Eppstein & Wang 2004)."""
    if u.vcount() < 2:
        return np.nan
    giant = u.connected_components().giant()
    n = giant.vcount()
    if n < 2:
        return np.nan
    src = np.arange(n) if n <= sources else np.random.default_rng(seed).choice(n, sources, replace=False)
    total, pairs = 0.0, 0
    for chunk in np.array_split(src, max(1, -(-len(src) // 32))):
        d = np.asarray(giant.distances(source=chunk.tolist(), mode="all"), dtype=np.float64)
        total += d.sum()
        pairs += d.size - len(chunk)  # without the distance of each source to itself
    return total / pairs


def sna_measures(g: ig.Graph, measures: list[str], path_sources: int = 500, seed: int = 0) -> dict:
    u = g.as_undirected(mode="collapse", combine_edges={"weight": "sum"}) if g.is_directed() else g
    out = {}
    n = g.vcount()
    for m in measures:
        if m == "nodes":
            out[m] = n
        elif m == "edges":
            out[m] = g.ecount()
        elif m == "density":
            out[m] = g.density(loops=False) if n > 1 else np.nan
        elif m == "mean_degree":
            out[m] = float(np.mean(u.degree())) if n else np.nan
        elif m == "max_degree":
            out[m] = int(u.maxdegree()) if n else 0
        elif m == "transitivity":
            out[m] = u.transitivity_undirected(mode="zero") if n else np.nan
        elif m == "avg_local_transitivity":
            out[m] = u.transitivity_avglocal_undirected(mode="zero") if n else np.nan
        elif m == "giant_fraction":
            out[m] = max(u.connected_components().sizes()) / n if n else np.nan
        elif m == "avg_path_length":
            out[m] = g.average_path_length(directed=True, unconn=True) if n > 1 else np.nan
        elif m == "mean_betweenness":
            out[m] = float(np.mean(g.betweenness())) if n else np.nan
        elif m == "avg_shortest_path":
            out[m] = mean_shortest_path(u, path_sources, seed)
    return out


def analyze_year(network_dir: Path, year: int, k: int, cfg: AnalysisConfig, names: dict[int, str]) -> dict:
    edges, expertise = load_year(network_dir, year)
    result: dict = {"year": year}
    if expertise.height == 0:
        return result
    mt = main_topics(expertise)
    nodes = mt["author"].to_numpy()
    result["authors_per_topic"] = (mt.group_by("main_topic").len()
                                   .with_columns(pl.lit(year).alias("year")))

    if cfg.flows:
        result["flow"] = intertopic_flow(edges, mt, k)

    g = to_graph(edges, nodes) if (cfg.pagerank.enabled or cfg.sna.enabled or cfg.communities.enabled) else None

    if cfg.pagerank.enabled:
        # Edges point from experts to non-experts; PageRank on the reversed graph ranks experts.
        rows = []
        def rank(sub_edges: pl.DataFrame, topic: int):
            sub_nodes = np.unique(np.r_[sub_edges["src"].to_numpy(), sub_edges["dst"].to_numpy()])
            if len(sub_nodes) == 0:
                return
            rev = to_graph(sub_edges.rename({"src": "dst", "dst": "src"}), sub_nodes)
            pr = np.array(rev.pagerank(weights="weight" if cfg.pagerank.weighted else None))
            for r, v in enumerate(np.argsort(-pr)[: cfg.pagerank.top_n]):
                a = int(sub_nodes[v])
                rows.append((year, topic, r + 1, a, names.get(a, ""), float(pr[v])))
        rank(edges, -1)
        if cfg.pagerank.per_topic:
            for t, sub in edges.partition_by("topic", as_dict=True).items():
                rank(sub, int(t[0]))
        result["pagerank"] = rows

    if cfg.sna.enabled:
        # Workers of an older parent process may hold a config without path_sources.
        sources = getattr(cfg.sna, "path_sources", 500)
        rows = [{"year": year, "topic": -1, **sna_measures(g, cfg.sna.measures, sources, year)}]
        if cfg.sna.per_topic:
            for t, sub in edges.partition_by("topic", as_dict=True).items():
                sub_nodes = np.unique(np.r_[sub["src"].to_numpy(), sub["dst"].to_numpy()])
                rows.append({"year": year, "topic": int(t[0]),
                             **sna_measures(to_graph(sub, sub_nodes), cfg.sna.measures, sources,
                                            year * 1000 + int(t[0]))})
        result["sna"] = rows

    if cfg.communities.enabled:
        u = to_graph(edges, nodes, directed=False)
        if cfg.communities.method == "leiden":
            part = u.community_leiden(objective_function="modularity", weights="weight",
                                      resolution=cfg.communities.resolution,
                                      n_iterations=cfg.communities.iterations)
        else:
            part = u.community_walktrap(weights="weight").as_clustering()
        comm = pl.DataFrame({"author": nodes, "community": np.array(part.membership)}).join(mt, on="author")
        # A community's topic is the most common main topic among its members.
        ctop = (comm.group_by("community", "main_topic").len()
                .sort(["community", "len", "main_topic"], descending=[False, True, False])
                .group_by("community", maintain_order=True).first()
                .join(comm.group_by("community").len().rename({"len": "size"}), on="community"))
        per_topic = ctop.group_by("main_topic").agg(
            pl.len().alias("communities"), pl.col("size").sum().alias("community_size"),
            pl.col("size").max().alias("largest_community"))
        if cfg.communities.coreness:
            core = np.array(u.coreness())
            topic_of = mt["main_topic"].to_numpy()
            cores = []
            for t in np.unique(topic_of):
                idx = np.nonzero(topic_of == t)[0]
                sub = u.induced_subgraph(idx.tolist())
                cores.append((int(t), int(max(sub.coreness())) if sub.vcount() else 0,
                              int(core[idx].max())))
            per_topic = per_topic.join(
                pl.DataFrame(cores, schema={"main_topic": pl.Int16, "max_coreness_topic_subgraph": pl.Int64,
                                            "max_coreness_in_graph": pl.Int64}, orient="row"),
                on="main_topic", how="full", coalesce=True)
        result["communities"] = per_topic.with_columns(pl.lit(year).alias("year"))
        result["modularity"] = u.modularity(part.membership, weights="weight") if u.ecount() else np.nan

    if cfg.graphml:
        gml = network_dir.parent / "analysis" / "graphml"
        gml.mkdir(parents=True, exist_ok=True)
        full = to_graph(edges, nodes)
        full.vs["name"] = [names.get(int(a), "") for a in nodes]
        full.vs["main_topic"] = mt["main_topic"].to_list()
        full.write_graphml(str(gml / f"{year}.graphml"))
    return result


def analyze(network_dir: Path, out: Path, k: int, cfg: AnalysisConfig, n_jobs: int) -> dict:
    authors = pl.read_parquet(network_dir / "authors.parquet")
    names = dict(zip(authors["author"].to_list(), authors["name"].to_list()))
    years = cfg.years or (pl.scan_parquet(network_dir / "expertise.parquet").select("year").unique()
                          .collect()["year"].sort().to_list())
    n_proc = max(1, min(n_jobs, len(years), cfg.max_workers or n_jobs))
    threads = max(1, n_jobs // n_proc)
    log.info("analysing %d years with %d processes x %d threads", len(years), n_proc, threads)
    # Fresh spawned workers, so the thread limits apply before polars/BLAS start their pools.
    with limited_threads(threads), ProcessPoolExecutor(n_proc, mp_context=mp.get_context("spawn")) as ex:
        # Largest (latest) years first so they do not end up as stragglers.
        futures = {y: ex.submit(analyze_year, network_dir, y, k, cfg, names) for y in sorted(years, reverse=True)}
        results = [futures[y].result() for y in years]

    out.mkdir(parents=True, exist_ok=True)
    present = [r for r in results if "authors_per_topic" in r]
    pl.concat([r["authors_per_topic"] for r in present]).sort("year", "main_topic").write_parquet(
        out / "authors_per_topic.parquet")
    if cfg.flows and present:
        flows = np.stack([r["flow"] for r in present])
        np.savez_compressed(out / "flows.npz", years=np.array([r["year"] for r in present]), flows=flows)
        s, t = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
        pl.concat([pl.DataFrame({"year": r["year"], "source_topic": s.ravel(), "target_topic": t.ravel(),
                                 "weight": r["flow"].ravel()}).filter(pl.col("weight") > 0)
                   for r in present]).write_parquet(out / "flows.parquet")
    if cfg.pagerank.enabled:
        pl.DataFrame([row for r in present for row in r["pagerank"]],
                     schema=["year", "topic", "rank", "author", "name", "pagerank"], orient="row"
                     ).write_parquet(out / "pagerank.parquet")
    if cfg.sna.enabled:
        pl.DataFrame([row for r in present for row in r["sna"]]).write_parquet(out / "sna.parquet")
    if cfg.communities.enabled:
        pl.concat([r["communities"] for r in present], how="diagonal_relaxed").sort(
            "year", "main_topic").write_parquet(out / "communities.parquet")
        pl.DataFrame({"year": [r["year"] for r in present], "modularity": [r["modularity"] for r in present]}
                     ).write_parquet(out / "modularity.parquet")
    return {"years": len(present)}
