import numpy as np
import pytest

from peak3d.warp import TableWarp, coarse_shift, fit_warp, match_anchors, pav_increasing

T = 5.0
DT = 0.005
GRID = np.arange(0, T + DT / 2, DT)


def true_warp(t):
    return t + 0.2 * np.sin(2 * np.pi * t / T) + 0.01 * t


def _anchors(n=300, decoy_frac=0.2, noise=0.3 * DT, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.2, T - 0.2, n))
    y = true_warp(x) + noise * rng.standard_normal(n)
    k = int(decoy_frac * n)
    bad = rng.choice(n, k, replace=False)
    y[bad] += rng.uniform(-0.3, 0.3, k)
    return x, y


def test_pav_is_monotone_and_close():
    y = np.array([1.0, 3.0, 2.0, 2.5, 5.0, 4.0])
    f = pav_increasing(y, np.ones(6))
    assert np.all(np.diff(f) >= 0)
    assert f[0] == 1.0 and abs(f[1] - 2.5) < 1e-9 and abs(f[-1] - 4.5) < 1e-9


def test_table_warp_forward_inverse_and_extrapolation():
    g = GRID
    w = TableWarp(g, true_warp(g))
    t = np.linspace(-1, T + 1, 1000)
    assert np.allclose(w.inverse(w.forward(t)), t, atol=1e-9)
    assert np.allclose(w.forward(np.array([-0.5])), -0.5 + w.shift_lo)
    assert np.allclose(w.forward(np.array([T + 0.5])), T + 0.5 + w.shift_hi)
    assert TableWarp.identity(g).is_identity
    gs = TableWarp.global_shift(g, 0.1)
    assert np.allclose(gs.shift(t), 0.1)


def test_fit_recovers_nonlinear_warp_with_decoys():
    x, y = _anchors()
    w, diag = fit_warp(x, y, GRID, DT)
    assert diag.flag == "ok"
    inside = (GRID > 0.3) & (GRID < T - 0.3)
    err = w.forward(GRID[inside]) - true_warp(GRID[inside])
    assert np.sqrt(np.mean(err ** 2)) < DT
    assert np.all(np.diff(w.corr) > 0)
    assert np.allclose(w.inverse(w.forward(GRID)), GRID, atol=1e-9)
    assert diag.mad_after_s < diag.mad_before_s


def test_constant_shift_beyond_knots():
    x, y = _anchors()
    w, diag = fit_warp(x, y, GRID, DT)
    lo = min(diag.knot_x)
    ref = w.shift(np.array([lo - 2 * DT]))
    assert np.allclose(w.shift(np.array([0.0, lo / 2, lo - 3 * DT])), ref, atol=1e-6)
    hi = max(diag.knot_x)
    ref = w.shift(np.array([hi + 2 * DT]))
    assert np.allclose(w.shift(np.array([hi + 3 * DT, T, T + 1])), ref, atol=1e-6)


def test_linear_drift_with_20_anchors():
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0.3, T - 0.3, 20))
    y = x * 1.02 + 0.05 + 0.2 * DT * rng.standard_normal(20)
    w, diag = fit_warp(x, y, GRID, DT)
    assert diag.flag == "ok"
    inside = (GRID > 0.5) & (GRID < T - 0.5)
    assert np.max(np.abs(w.forward(GRID[inside]) - (GRID[inside] * 1.02 + 0.05))) < 2 * DT


def test_outlier_does_not_warp_neighbours():
    x, y = _anchors(decoy_frac=0.0, noise=0.0)
    w0, _ = fit_warp(x, y, GRID, DT)
    y2 = y.copy()
    y2[150] += 0.5
    w1, _ = fit_warp(x, y2, GRID, DT)
    far = np.abs(GRID - x[150]) > 0.4
    assert np.max(np.abs(w1.forward(GRID[far]) - w0.forward(GRID[far]))) < 0.2 * DT


def test_internal_gap_is_bridged_by_a_straight_line():
    x, y = _anchors(decoy_frac=0.0, noise=0.0)
    keep = (x < T / 3) | (x > 2 * T / 3)
    w, diag = fit_warp(x[keep], y[keep], GRID, DT)
    assert diag.flag == "ok" and diag.gap and diag.interp in ("mixed", "linear")
    inside = (GRID > T / 3) & (GRID < 2 * T / 3)
    sh = w.shift(GRID)
    lo, hi = min(sh[GRID <= T / 3].min(), sh[GRID >= 2 * T / 3].min()), max(sh[GRID <= T / 3].max(), sh[GRID >= 2 * T / 3].max())
    assert sh[inside].min() >= lo - 1e-9 and sh[inside].max() <= hi + 1e-9      # no bulge inside the gap
    assert np.all(np.diff(w.corr) > 0)
    d2 = np.diff(sh[inside], 2)
    assert np.abs(d2).max() < 1e-6                                              # straight across the gap


def test_clustered_sparse_anchors_do_not_overshoot():
    # the YEAST_12C geometry: 8 co-eluting anchors at one RT, a few sparse early anchors, dense late ones
    rng = np.random.default_rng(0)
    g = np.arange(0.0, 20.0, 1.09 / 60)
    x = np.concatenate([1.82 + 0.02 * rng.random(8), [2.6, 3.4, 4.1, 4.9, 5.6, 6.3, 7.0, 7.7, 8.1],
                        np.sort(rng.uniform(8.4, 19.5, 80))])
    d = 0.005 * rng.standard_normal(len(x))
    d[:8] += 0.004
    d[-80:] += 0.015
    w, diag = fit_warp(x, x + d, g)
    sh = 60 * w.shift(g)
    assert diag.flag == "ok"
    assert np.abs(sh).max() < 60 * d.max() + 1.0, np.abs(sh).max()       # within the anchors' own range (+1 s)
    assert np.all(np.diff(w.corr) > 0)


def test_frozen_yeast12c_anchors_fit_without_fallback():
    import pathlib
    d = pathlib.Path(__file__).parent / "data"
    a = np.loadtxt(d / "yeast12c_a_anchors.tsv", skiprows=1)
    g = np.loadtxt(d / "yeast12c_a_grid.txt")
    w, diag = fit_warp(a[:, 0], a[:, 1], g)
    sh = 60 * w.shift(g)
    assert diag.flag == "ok" and diag.gap                                   # 6-min anchor gap, bridged not abandoned
    assert np.abs(sh).max() < 60 * np.abs(a[:, 1] - a[:, 0]).max() + 1.0
    assert np.all(np.diff(w.corr) > 0)


def test_too_few_matches_identity():
    w, diag = fit_warp(np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.1, 3.1]), GRID, DT)
    assert diag.flag.startswith("identity") and w.is_identity
    w, diag = fit_warp(np.arange(1, 7, 1.0), np.arange(1, 7, 1.0) + 0.1, GRID, DT)
    assert diag.flag.startswith("global_shift") and np.allclose(w.shift(GRID), 0.1)


def test_coarse_shift_and_mutual_unique_matching():
    rng = np.random.default_rng(2)
    mz = rng.uniform(100, 900, 200)
    rt = rng.uniform(0.5, 4.5, 200)
    delta, support, sigma = coarse_shift(mz, rt, mz, rt + 0.12, 5.0, 0.5, 2 * DT)
    assert abs(delta - 0.12) < DT and support >= 100
    # add an ambiguous target twin for anchor 0 -> it must drop out of the matches
    mz_t = np.concatenate([mz, [mz[0]]])
    rt_t = np.concatenate([rt + 0.12, [rt[0] + 0.12 + 0.02]])
    a, b = match_anchors(mz, rt, mz_t, rt_t, 5.0, delta, 0.1)
    assert 0 not in a and len(a) >= 190
    assert np.all(b == a)


def test_coeluting_anchor_cluster_counts_once():
    # a compound with 7 isotopologue/charge-state anchors at one RT shifts by +5 s on its own
    # (compound-specific retention), while 6 single anchors nearby say the drift is ~0
    rng = np.random.default_rng(4)
    grid = np.arange(0.0, 20.0, 1.09 / 60)
    x_single = np.concatenate([np.sort(rng.uniform(0.5, 6.2, 14)), np.sort(rng.uniform(7.2, 19.5, 40))])
    x_cluster = 6.67 + 0.02 * rng.random(7)
    x = np.concatenate([x_single, x_cluster])
    d = 0.003 * rng.standard_normal(len(x))
    d[-7:] += 5.0 / 60
    w0 = np.concatenate([np.ones(len(x_single)), np.full(7, 1 / 7)])
    w_bad, _ = fit_warp(x, x + d, grid)
    w_good, diag = fit_warp(x, x + d, grid, w0=w0)
    at = np.argmin(np.abs(grid - 6.67))
    assert abs(60 * w_good.shift(grid)[at]) < 1.0, 60 * w_good.shift(grid)[at]
    assert abs(60 * w_bad.shift(grid)[at]) > abs(60 * w_good.shift(grid)[at])
    assert diag.flag == "ok"


def test_scaled_warp_interpolates_between_identity_and_the_fit():
    w = TableWarp(GRID, true_warp(GRID))
    assert w.scaled(np.zeros(len(GRID))).is_identity
    assert np.allclose(w.scaled(np.ones(len(GRID))).corr, w.corr)
    ramp = np.clip((GRID - 1.0) / 2.0, 0.0, 1.0)
    s = w.scaled(ramp)
    assert np.all(np.diff(s.corr) > 0)
    assert np.allclose(s.shift(GRID), ramp * w.shift(GRID), atol=1e-8)


def test_isolation_matches_the_pairwise_definition():
    from peak3d.anchors import _isolated
    rng = np.random.default_rng(1)
    mz = rng.uniform(100, 110, 3000)
    rt = rng.uniform(0, 10, 3000)
    cand = rng.choice(3000, 500, replace=False)
    got = _isolated(mz, rt, cand, 10.0, 0.2)
    for k, i in enumerate(cand):
        n = np.sum((np.abs(mz - mz[i]) <= mz[i] * 10e-6) & (np.abs(rt - rt[i]) <= 0.2))
        assert got[k] == (n <= 1)
