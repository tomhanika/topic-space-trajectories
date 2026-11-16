"""Topic space trajectories (Schaefermeier, Stumme & Hanika 2021) and trajectory measures.

A trajectory is the sequence of yearly topic centroids of an entity (venue or author). All distances are
Hellinger distances between topic distributions, which are bounded in [0, 1].
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
import scipy.sparse as sp

from .config import EntityTrajectoryConfig, PairwiseConfig

log = logging.getLogger("tst.trajectories")


def hellinger(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2, axis=-1))


def entropy(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=-1)


def last_complete_year(years: np.ndarray, window: int = 5, min_share: float = 0.5) -> int | None:
    """Last year whose paper count is at least `min_share` of the largest count in the `window` years before it.

    Exports cover their last years only partially; trailing years below that level are dropped, and so are
    stray future years (a few works dated years ahead), which a median over sparse years would let through."""
    years = years[~np.isnan(years)] if years.dtype.kind == "f" else years
    if len(years) == 0:
        return None
    ys, counts = np.unique(years.astype(np.int64), return_counts=True)
    full = dict(zip(ys.tolist(), counts.tolist()))
    last = int(ys.max())
    while last > ys.min():
        before = [full.get(y, 0) for y in range(last - window, last)]
        if full.get(last, 0) >= min_share * max(before):
            break
        last -= 1
    return last


def memberships(docs: pl.DataFrame, entity_type: str, fractional: bool,
                max_authors: int | None = None) -> pl.DataFrame:
    """(row, entity_id, entity_name, year, w): which topic-matrix row belongs to which entity. Papers
    with more than `max_authors` authors count for no author."""
    if entity_type == "venue":
        m = docs.filter(pl.col("venue").is_not_null()).select(
            "row", "year", pl.col("venue").alias("entity_id"), pl.col("venue").alias("entity_name"),
            pl.lit(1.0).alias("w"),
        )
    else:
        m = (
            docs.select("row", "year", "authors", pl.col("authors").list.len().alias("n"))
            .filter(pl.col("n") <= (max_authors or 2**31))
            .explode("authors")
            .filter(pl.col("authors").is_not_null())
            .select(
                "row", "year",
                pl.when(pl.col("authors").struct.field("id").is_not_null())
                .then(pl.col("authors").struct.field("id").cast(pl.String))
                .otherwise(pl.lit("name:") + pl.col("authors").struct.field("name"))
                .alias("entity_id"),
                pl.col("authors").struct.field("name").alias("entity_name"),
                (1.0 / pl.col("n")).alias("w") if fractional else pl.lit(1.0).alias("w"),
            )
            .unique(["row", "entity_id"])
        )
    return m


def yearly_centroids(m: pl.DataFrame, theta: np.ndarray) -> tuple[pl.DataFrame, np.ndarray]:
    groups = (
        m.group_by("entity_id", "year")
        .agg(pl.col("entity_name").first(), pl.len().alias("n_papers"), pl.col("w").sum().alias("weight"))
        .sort("entity_id", "year")
        .with_row_index("g")
    )
    gi = m.join(groups.select("entity_id", "year", "g"), on=["entity_id", "year"])
    M = sp.csr_matrix((gi["w"].to_numpy(), (gi["g"].to_numpy(), gi["row"].to_numpy())),
                      shape=(groups.height, theta.shape[0]))
    C = np.asarray(M @ theta) / groups["weight"].to_numpy()[:, None]
    return groups.drop("g"), C.astype(np.float32)


def smooth(groups: pl.DataFrame, C: np.ndarray, cfg: EntityTrajectoryConfig) -> tuple[pl.DataFrame, np.ndarray]:
    """Apply the window smoothing to yearly centroids (rows sorted by entity, year)."""
    g = groups.with_row_index("i")
    if cfg.smoothing == "cumulative":
        # Running sums over all rows, minus the running sum before the entity's first row.
        w = g["weight"].to_numpy()
        cs = np.cumsum(C * w[:, None], axis=0, dtype=np.float64)
        cw = np.cumsum(w, dtype=np.float64)
        first = g.select(pl.col("i").min().over("entity_id"))["i"].to_numpy().astype(np.int64)
        before = np.maximum(first - 1, 0)
        prev_c = np.where(first[:, None] > 0, cs[before], 0.0)
        prev_w = np.where(first > 0, cw[before], 0.0)
        out = (cs - prev_c) / (cw - prev_w)[:, None]
        n = g.with_columns(pl.col("n_papers").cum_sum().over("entity_id"))["n_papers"]
        wsum = g.with_columns(pl.col("weight").cum_sum().over("entity_id"))["weight"]
        return groups.with_columns(n_papers=n, weight=wsum), out.astype(np.float32)
    if cfg.window <= 1:
        return groups, C
    # Each yearly centroid contributes to the years y .. y+window-1 of the same entity, if present.
    src = g.select("i", "entity_id", "year", "weight").with_columns(
        pl.int_ranges(0, cfg.window).alias("d")).explode("d").with_columns(
        (pl.col("year") + pl.col("d")).alias("target_year"))
    pairs = src.join(g.select(pl.col("i").alias("t"), "entity_id", pl.col("year").alias("target_year")),
                     on=["entity_id", "target_year"])
    if cfg.smoothing == "pooled":
        val = pairs["weight"].to_numpy()
    else:
        val = np.ones(pairs.height)
    S = sp.csr_matrix((val, (pairs["t"].to_numpy(), pairs["i"].to_numpy())), shape=(g.height, g.height))
    norm = np.asarray(S.sum(axis=1)).ravel()
    out = np.asarray(S @ C) / norm[:, None]
    n_papers = sp.csr_matrix((np.ones(pairs.height), (pairs["t"].to_numpy(), pairs["i"].to_numpy())),
                             shape=S.shape) @ g["n_papers"].to_numpy().astype(np.float64)
    wsum = sp.csr_matrix((np.ones(pairs.height), (pairs["t"].to_numpy(), pairs["i"].to_numpy())),
                         shape=S.shape) @ g["weight"].to_numpy()
    return groups.with_columns(n_papers=pl.Series(n_papers.astype(np.int64)), weight=pl.Series(wsum)), \
        out.astype(np.float32)


def build(docs: pl.DataFrame, theta: np.ndarray, entity_type: str, cfg: EntityTrajectoryConfig,
          field_centroids: dict[int, np.ndarray]) -> tuple[pl.DataFrame, np.ndarray]:
    """Trajectories with point metrics for one entity type. `docs` needs a `row` column into theta."""
    m = memberships(docs, entity_type, cfg.fractional, cfg.max_authors if entity_type == "author" else None)
    groups, C = yearly_centroids(m, theta)
    keep = (groups["n_papers"] >= cfg.min_papers).to_numpy()
    groups, C = groups.filter(pl.Series(keep)), C[keep]
    groups, C = smooth(groups, C, cfg)
    if cfg.min_years > 1:
        keep = (groups.select(pl.len().over("entity_id"))["len"] >= cfg.min_years).to_numpy()
        groups, C = groups.filter(pl.Series(keep)), C[keep]
    traj = groups.with_columns(pl.lit(entity_type).alias("entity_type"))
    traj = point_metrics(traj, C, field_centroids)
    log.info("%s trajectories: %d entities, %d points", entity_type, traj["entity_id"].n_unique(), traj.height)
    return traj, C


def point_metrics(traj: pl.DataFrame, C: np.ndarray, field_centroids: dict[int, np.ndarray]) -> pl.DataFrame:
    k = C.shape[1]
    if len(C) == 0:  # no entity reaches the thresholds
        return traj.with_columns(
            main_topic=pl.lit(None, pl.Int32), main_topic_share=pl.lit(None, pl.Float32),
            entropy=pl.lit(None, pl.Float64), effective_topics=pl.lit(None, pl.Float32),
            step=pl.lit(None, pl.Float32), speed=pl.lit(None, pl.Float64), novelty=pl.lit(None, pl.Float64),
            mainstream_distance=pl.lit(None, pl.Float32))
    ent = traj["entity_id"].to_numpy()
    years = traj["year"].to_numpy()
    same_prev = np.r_[False, ent[1:] == ent[:-1]]
    prev = np.r_[0, np.arange(len(C) - 1)]
    step = np.where(same_prev, hellinger(C, C[prev]), np.nan)
    gap = np.where(same_prev, years - years[prev], np.nan)

    # Novelty: distance to the mean of all earlier points of the same entity.
    cs = np.cumsum(C, axis=0, dtype=np.float64)
    start = traj.with_row_index("i").select(pl.col("i").min().over("entity_id"))["i"].to_numpy().astype(np.int64)
    count = np.arange(len(C)) - start
    before = cs[np.maximum(np.arange(len(C)) - 1, 0)] - np.where(start[:, None] > 0, cs[np.maximum(start - 1, 0)], 0)
    before[start == np.arange(len(C))] = 0
    past_mean = np.divide(before, count[:, None], out=np.zeros_like(before), where=count[:, None] > 0)
    novelty = np.where(count > 0, hellinger(C, past_mean), np.nan)

    field = np.stack([field_centroids.get(int(y), np.full(k, np.nan)) for y in years]) if len(years) else C
    H = entropy(C)
    return traj.with_columns(
        main_topic=pl.Series(C.argmax(axis=1).astype(np.int32)),
        main_topic_share=pl.Series(C.max(axis=1)),
        entropy=pl.Series(H / np.log(k)),
        effective_topics=pl.Series(np.exp(H)),
        # First points have no predecessor: null (not NaN), so that sums and means skip them.
        step=pl.Series(step).fill_nan(None),
        speed=pl.Series(step / gap).fill_nan(None),
        novelty=pl.Series(novelty).fill_nan(None),
        mainstream_distance=pl.Series(hellinger(C, field)),
    )


def entity_summary(traj: pl.DataFrame, C: np.ndarray) -> pl.DataFrame:
    """One row per entity: shape of the whole trajectory."""
    t = traj.with_row_index("i")
    first_last = t.group_by("entity_type", "entity_id", maintain_order=True).agg(
        pl.col("i").first().alias("i0"), pl.col("i").last().alias("i1"))
    displacement = hellinger(C[first_last["i0"].to_numpy()], C[first_last["i1"].to_numpy()])
    # Radius: mean distance of the points to the entity's mean position.
    inv = t.select((pl.col("entity_type") + "\x00" + pl.col("entity_id")).rank("dense") - 1)
    inv = inv.to_series().to_numpy().astype(np.int64)
    sums = np.zeros((inv.max() + 1 if len(inv) else 0, C.shape[1]))
    np.add.at(sums, inv, C)
    means = sums / np.bincount(inv)[:, None]
    t = t.with_columns(dist_to_mean=pl.Series(hellinger(C, means[inv])))
    summary = t.group_by("entity_type", "entity_id", maintain_order=True).agg(
        pl.col("entity_name").first(),
        pl.len().alias("n_points"),
        pl.col("year").min().alias("first_year"),
        pl.col("year").max().alias("last_year"),
        pl.col("n_papers").sum().alias("papers"),
        pl.col("step").sum().alias("path_length"),
        pl.col("step").max().alias("max_step"),
        pl.col("year").sort_by("step", descending=True, nulls_last=True).first().alias("max_step_year"),
        pl.col("speed").mean().alias("mean_speed"),
        pl.col("entropy").mean().alias("mean_entropy"),
        pl.col("mainstream_distance").mean().alias("mean_mainstream_distance"),
        (pl.col("main_topic").diff().fill_null(0) != 0).sum().alias("main_topic_changes"),
        pl.col("main_topic").mode().first().alias("dominant_topic"),
        pl.col("dist_to_mean").mean().alias("radius"),
    )
    return summary.with_columns(displacement=pl.Series(displacement)).with_columns(
        straightness=pl.when(pl.col("path_length") > 0)
        .then(pl.col("displacement") / pl.col("path_length")).otherwise(None))


def pairwise(traj: pl.DataFrame, C: np.ndarray, cfg: PairwiseConfig) -> pl.DataFrame:
    """Same-year distance and lead/lag between the largest entities of one type.

    lead_score(a, b) > 0 means b's later positions resemble a's earlier ones more than the other way
    round, i.e. a leads b. It is averaged over lags 1..max_lag.
    """
    t = traj.with_row_index("i").filter(pl.col("entity_type") == cfg.entity_type)
    if t.height == 0:
        return pl.DataFrame()
    top = (t.group_by("entity_id").agg(pl.col("n_papers").sum(), pl.col("entity_name").first())
           .sort("n_papers", descending=True).head(cfg.max_entities))
    ids = top["entity_id"].to_list()
    pos = {e: j for j, e in enumerate(ids)}
    t = t.filter(pl.col("entity_id").is_in(ids))
    y0, y1 = t["year"].min(), t["year"].max()
    E, Y, k = len(ids), y1 - y0 + 1, C.shape[1]
    S = np.full((E, Y, k), np.nan, np.float32)
    S[[pos[e] for e in t["entity_id"]], (t["year"] - y0).to_numpy()] = np.sqrt(C[t["i"].to_numpy()])

    def mean_dist(lag: int) -> tuple[np.ndarray, np.ndarray]:
        """D[a, b] = mean over y of dist(a(y), b(y+lag)), and the number of years used."""
        A = S[:, : Y - lag]
        B = S[:, lag:]
        D = np.zeros((E, E))
        N = np.zeros((E, E))
        for a in range(E):
            d = np.sqrt(0.5 * np.sum((A[a][None] - B) ** 2, axis=-1))  # (E, Y-lag)
            valid = ~np.isnan(d)
            N[a] = valid.sum(axis=1)
            D[a] = np.where(N[a] > 0, np.nansum(d, axis=1) / np.maximum(N[a], 1), np.nan)
        return D, N

    D0, N0 = mean_dist(0)
    leads, lead_n = np.zeros((E, E)), np.zeros((E, E))
    best_lag, best_gain = np.zeros((E, E), int), np.full((E, E), -np.inf)
    for lag in range(1, min(cfg.max_lag, Y - 1) + 1):
        D, N = mean_dist(lag)
        ok = (N >= cfg.min_overlap) & (N.T >= cfg.min_overlap)
        diff = np.where(ok, D.T - D, 0.0)  # >0: b follows a more closely than a follows b
        leads += diff
        lead_n += ok
        better = ok & (diff > best_gain)
        best_gain[better], best_lag[better] = diff[better], lag

    a_idx, b_idx = np.triu_indices(E, 1)
    keep = N0[a_idx, b_idx] >= cfg.min_overlap
    a_idx, b_idx = a_idx[keep], b_idx[keep]
    names = top["entity_name"].to_list()
    with np.errstate(invalid="ignore", divide="ignore"):
        lead = leads / lead_n
    return pl.DataFrame({
        "entity_type": cfg.entity_type,
        "a": [ids[i] for i in a_idx], "a_name": [names[i] for i in a_idx],
        "b": [ids[i] for i in b_idx], "b_name": [names[i] for i in b_idx],
        "overlap_years": N0[a_idx, b_idx].astype(int),
        "mean_distance": D0[a_idx, b_idx],
        "lead_score": lead[a_idx, b_idx],
        "best_lag": np.where(np.isfinite(best_gain[a_idx, b_idx]), best_lag[a_idx, b_idx], 0),
    })


def field_trends(docs: pl.DataFrame, theta: np.ndarray) -> tuple[pl.DataFrame, dict[int, np.ndarray]]:
    """Corpus-wide topic prevalence per year."""
    years = docs["year"].to_numpy()
    uniq = np.unique(years)
    cents = {int(y): theta[docs["row"].to_numpy()[years == y]].mean(axis=0) for y in uniq}
    counts = [int((years == y).sum()) for y in uniq]
    rows = pl.DataFrame({"year": uniq.astype(np.int32), "n_docs": counts})
    prev = np.stack([cents[int(y)] for y in uniq]) if len(uniq) else np.zeros((0, theta.shape[1]))
    rows = rows.with_columns(prevalence=pl.Series(prev.astype(np.float32)))
    return rows, cents
