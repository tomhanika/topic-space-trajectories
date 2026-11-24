import numpy as np
import pytest

from tst import citations as ci
from tst.config import CitationsConfig


def test_top_topics_renormalises_and_pads():
    theta = np.array([[0.5, 0.3, 0.2, 0.0], [0.0, 0.0, 1.0, 0.0]], np.float32)
    idx, w = ci.top_topics(theta, 3)
    assert idx[0].tolist() == [0, 1, 2] and np.allclose(w[0], [0.5, 0.3, 0.2])
    assert idx[1].tolist() == [2, -1, -1] and np.allclose(w[1], [1, 0, 0])
    idx, w = ci.top_topics(theta, 2)
    assert np.allclose(w[0], [0.625, 0.375])


def test_windows_end_at_the_last_year():
    assert ci.windows_of(2000, 2010, 3) == [(2002, 2004), (2005, 2007), (2008, 2010)]
    assert ci.windows_of(2000, 2011, 3) == [(2000, 2002), (2003, 2005), (2006, 2008), (2009, 2011)]
    assert ci.windows_of(2000, 2001, 3) == [(2000, 2001)]


def test_estep_matches_dense_formula():
    rng = np.random.default_rng(1)
    K, n, E = 4, 50, 400
    theta = rng.dirichlet(np.ones(K) * 0.7, n).astype(np.float32)
    ti, tw = ci.top_topics(theta, K)
    p, q = rng.integers(0, n, E), rng.integers(0, n, E)
    w = rng.random(E)
    B = rng.random((K, K)) + 0.1
    G, ll = ci._estep(p, q, w, ti, tw, B, np.zeros((ci.get_num_threads(), K, K)))
    d = np.einsum("ei,ij,ej->e", theta[q], B, theta[p])
    assert np.allclose(G, np.einsum("e,ei,ej->ij", w / d, theta[q], theta[p]), rtol=1e-5)
    assert ll == pytest.approx((w * np.log(d)).sum(), rel=1e-6)


def _simulate(B, rng, n_per_year=400, years=range(2000, 2012), refs=12, alpha=0.4, decay=0.7):
    """Papers with Dirichlet topic mixes; each cites `refs` earlier-or-same-year papers drawn with probability
    proportional to g(lag) * theta_q' B theta_p (the model with the reference count given)."""
    K = B.shape[0]
    year = np.repeat(np.array(list(years)), n_per_year)
    theta = rng.dirichlet(np.ones(K) * alpha, len(year))
    P, Q = [], []
    for p in range(len(year)):
        cand = np.flatnonzero(year <= year[p])
        cand = cand[cand != p]
        if len(cand) < refs:
            continue
        pr = decay ** (year[p] - year[cand]) * (theta[cand] @ B @ theta[p])
        q = rng.choice(cand, refs, replace=False, p=pr / pr.sum())
        P += [p] * refs
        Q += q.tolist()
    return year.astype(np.int16), theta.astype(np.float32), np.array(P), np.array(Q)


def test_em_recovers_affinity_and_direction():
    rng = np.random.default_rng(7)
    # topic 1 draws on topic 0 much more than the other way round; everyone cites its own topic most
    B = np.array([[4.0, 3.0, 0.5], [0.3, 4.0, 0.5], [0.5, 0.5, 4.0]])
    year, theta, P, Q = _simulate(B, rng)
    ti, tw = ci.top_topics(theta, 3)
    cfg = CitationsConfig(min_citations=100, bootstrap=0, prior=0.0, tol=1e-10, max_iter=2000, hodge_min_flow=1)
    fits = ci.fit_all(P, Q, year, ti, tw, 3, [(2006, 2011)], int(year.min()), cfg, bootstrap=0, lags=False, seed=0)
    f = fits[0]
    Bh = f["B"]
    # With the reference counts given, B is identified up to a scale per citing topic; r_p = |R_p| makes it the
    # rate per reference, i.e. each column of B divided by the citing topic's total affinity (equal volumes here).
    Bn = B / B.sum(axis=0, keepdims=True)
    assert np.log(Bh[0, 1] / Bh[1, 0]) == pytest.approx(np.log(Bn[0, 1] / Bn[1, 0]), abs=0.4)
    assert f["phi"][0] > f["phi"][1]
    # mixed topics smear the product attribution; the model keeps the citations on the diagonal
    assert f["diag_model"] > f["diag_product"] + 0.1
    # the lift of the strong channel 0 -> 1 exceeds that of the weak one 1 -> 0
    assert f["lift"][0, 1] > f["lift"][1, 0]


def test_hodge_gradient_and_cycle():
    K = 3
    F = np.ones((K, K)) * 100
    # perfectly ordered: 0 upstream of 1 upstream of 2 (log-ratios 1, 1, 2)
    phi_true = np.array([1.0, 0.0, -1.0])
    B = np.exp(np.subtract.outer(phi_true, phi_true) / 2)
    h = ci.hodge(B, F, 1.0)
    assert h["gradient"] == pytest.approx(1.0)
    assert np.allclose(h["phi"] - h["phi"].mean(), phi_true)
    # a pure cycle 0 -> 1 -> 2 -> 0 has no gradient part and is all curl (a triangle has no hole)
    Bc = np.ones((K, K))
    for a, b in ((0, 1), (1, 2), (2, 0)):
        Bc[a, b], Bc[b, a] = np.e, 1.0
    h = ci.hodge(Bc, F, 1.0)
    assert h["gradient"] == pytest.approx(0.0, abs=1e-9)
    assert h["curl"] == pytest.approx(1.0)
    assert h["harmonic"] == pytest.approx(0.0, abs=1e-9)


def test_twins_are_mutual_best_matches_across_corpora():
    toff = np.array([0, 2, 4])
    sim = np.array([[1.0, 0.1, 0.9, 0.6],
                    [0.1, 1.0, 0.7, 0.4],
                    [0.9, 0.7, 1.0, 0.2],
                    [0.6, 0.4, 0.2, 1.0]])
    tw = ci.twins(sim, toff, 0.5)
    # 0 and 2 are each other's best match; 1's best is 2, but 2's best is 0; 3's best is 0, 0's best is 2
    assert {tuple(x) for x in np.argwhere(tw)} == {(0, 2), (2, 0)}
    assert not ci.twins(sim, toff, 0.95).any()


def test_saved_fits_load_back(tmp_path):
    fits = [{"lo": 2000, "hi": 2001, "fitted": False, "citations": 3},
            {"lo": 2002, "hi": 2004, "fitted": True, "citations": 10.0, "F": np.eye(2), "phi": np.array([1.0, np.nan]),
             "hierarchy": {"gradient": None, "cycles": 0}}]
    ci.save_fits(tmp_path / "fit.npz", fits, {"k": 2})
    back = ci.load_fits(tmp_path / "fit.npz")
    assert back[0] == fits[0] and "F" not in back[0]
    assert np.allclose(back[1]["F"], np.eye(2)) and np.isnan(back[1]["phi"][1])
    assert back[1]["hierarchy"] == {"gradient": None, "cycles": 0}
