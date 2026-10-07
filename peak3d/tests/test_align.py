import numpy as np
import pandas as pd
import pytest

from peak3d.align import AlignParams, align_run, anchors_table, summary_table

T = 5.0
DT = 0.005
GRID = np.arange(0.0, T + DT / 2, DT)
COLS = ["feature_id", "mz", "rt", "rt_min", "rt_max", "height", "score", "snr", "n_scans", "mz_sd_ppm",
        "iso_offset", "iso_parent"]


def _truth(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"mz": np.sort(rng.uniform(100, 900, n)), "rt": rng.uniform(0.3, T - 0.3, n),
                         "height": 10 ** rng.uniform(4, 7, n)})


def _file(truth, warp_fn, presence=1.0, noise=0.2 * DT, n_decoy=0, seed=0):
    rng = np.random.default_rng(seed)
    keep = rng.random(len(truth)) < presence
    t = truth[keep].reset_index(drop=True)
    rt = warp_fn(t["rt"].values) + noise * rng.standard_normal(len(t))
    df = pd.DataFrame({"mz": t["mz"].values * (1 + 1e-6 * 0.3 * rng.standard_normal(len(t))), "rt": rt,
                       "height": t["height"].values * np.exp(0.2 * rng.standard_normal(len(t)))})
    if n_decoy:
        df = pd.concat([df, pd.DataFrame({"mz": rng.uniform(100, 900, n_decoy), "rt": rng.uniform(0, T, n_decoy),
                                          "height": 10 ** rng.uniform(4, 6, n_decoy)})], ignore_index=True)
    df["rt_min"] = df["rt"] - 0.03
    df["rt_max"] = df["rt"] + 0.03
    df["score"] = 0.9
    df["snr"] = 50.0
    df["n_scans"] = 12
    df["mz_sd_ppm"] = 0.4
    df["iso_offset"] = 0
    df["iso_parent"] = -1
    df["feature_id"] = np.arange(len(df))
    return df[COLS]


def _warps():
    return {"a": lambda t: t,
            "b": lambda t: t + 0.05 + 0.01 * t,
            # 4.8 s sinusoidal drift: ~3x the curvature measured on HZV029/LI2018; the warp is
            # piecewise-cubic between knots ~0.2 min apart, so far stronger curvature would show
            # interpolation error above half a scan
            "c": lambda t: t - 0.03 + 0.08 * np.sin(2 * np.pi * t / T)}


def test_three_files_align_to_medoid_axis_and_converge():
    truth = _truth()
    W = _warps()
    feats = {k: _file(truth, W[k], n_decoy=100, seed=i) for i, k in enumerate(W)}
    grids = {k: GRID for k in W}
    # the anchor fit itself (the do-no-harm scaling is checked against it at the end)
    res = align_run({k: f.copy() for k, f in feats.items()}, grids, AlignParams(gate=False))
    assert not res.skipped and res.medoid in W and res.n_iter_done <= 3
    # every file's corrected RT of a truth feature agrees with the medoid's native RT
    med = res.medoid
    inside = (truth["rt"] > 0.6) & (truth["rt"] < T - 0.6)
    for k in W:
        if k == med:
            continue
        fa = res.files[k]
        assert fa.diag.flag == "ok", fa.diag.flag
        rt_native = W[k](truth["rt"].values[inside])
        corr = fa.warp.forward(rt_native)
        target = W[med](truth["rt"].values[inside])
        # within one scan: the test drift (4.8 s amplitude, up to 6 s/min) is far steeper than any
        # measured drift (< 1 s/min), and knots ~0.4 min apart leave ~0.3 s of interpolation error on it
        assert np.median(np.abs(corr - target)) < DT
        assert np.all(np.diff(fa.warp.corr) > 0)
    s = summary_table(res)
    assert set(s["role"]) == {"medoid", "other"}
    o = s[s["role"] == "other"]
    assert (o["mad_after_s"] <= o["mad_before_s"] + 1e-9).all()
    a = anchors_table(res)
    assert len(a) > 100 and a["is_inlier"].mean() > 0.7
    assert res.rt_tol >= 2 * DT - 1e-9
    # the do-no-harm check keeps a coherent warp: applied error within 0.1 s of the fit's
    gated = align_run({k: f.copy() for k, f in feats.items()}, grids, AlignParams())
    assert gated.medoid == med
    for k in W:
        if k == med:
            continue
        t = truth["rt"].values[inside]
        e_fit = np.median(np.abs(res.files[k].warp.forward(W[k](t)) - W[med](t)))
        e_gate = np.median(np.abs(gated.files[k].warp.forward(W[k](t)) - W[med](t)))
        assert e_gate <= e_fit + 0.1 / 60, (k, 60 * e_fit, 60 * e_gate)


def test_heterogeneous_presence_still_aligns_all_files():
    truth = _truth()
    W = _warps()
    feats = {"a": _file(truth, W["a"], seed=1), "b": _file(truth, W["b"], seed=2),
             "c": _file(truth[truth["mz"] > 500], W["c"], seed=3)}   # half the anchors absent
    res = align_run(feats, {k: GRID for k in feats}, AlignParams())
    for k, fa in res.files.items():
        assert fa.role == "medoid" or fa.diag.flag == "ok"


def test_file_with_few_anchors_gets_global_shift_not_dropped():
    truth = _truth()
    W = _warps()
    feats = {"a": _file(truth, W["a"], seed=1), "b": _file(truth, W["b"], seed=2),
             "c": _file(truth.iloc[:6], W["b"], seed=3)}
    res = align_run(feats, {k: GRID for k in feats}, AlignParams())
    fa = res.files["c"]
    assert "anchor_poor" in fa.flags
    assert fa.diag.flag.startswith(("global_shift", "identity"))
    assert len(summary_table(res)) == 3


def test_single_and_two_files():
    truth = _truth()
    W = _warps()
    one = align_run({"a": _file(truth, W["a"])}, {"a": GRID}, AlignParams())
    assert one.skipped and one.files["a"].warp.is_identity and one.medoid is None
    two = align_run({"a": _file(truth, W["a"]), "b": _file(truth, W["b"], seed=2)}, {"a": GRID, "b": GRID})
    assert not two.skipped
    other = [k for k in two.files if k != two.medoid][0]
    assert two.files[other].diag.flag == "ok"


def test_holdout_reports_generalisation():
    truth = _truth()
    W = _warps()
    feats = {k: _file(truth, W[k], seed=i) for i, k in enumerate(W)}
    res = align_run(feats, {k: GRID for k in W}, AlignParams(holdout=True))
    for k, fa in res.files.items():
        if k == res.medoid:
            continue
        assert fa.holdout is not None
        assert fa.holdout["mad_out_s"] < 1.5 * max(fa.holdout["mad_in_s"], 60 * 0.1 * DT)
        assert fa.holdout["mad_out_s"] < fa.holdout["mad_none_s"]


def test_per_file_mz_offset_is_recovered():
    truth = _truth()
    W = _warps()
    feats = {k: _file(truth, W[k], seed=i) for i, k in enumerate(W)}
    feats["b"]["mz"] = feats["b"]["mz"] * (1 + 4e-6)          # file b reads 4 ppm high
    res = align_run(feats, {k: GRID for k in W}, AlignParams())
    offs = {k: fa.mz_offset_ppm for k, fa in res.files.items()}
    rel = offs["b"] - offs["a"]
    assert abs(rel - 4.0) < 0.6, offs
    assert abs(offs["c"] - offs["a"]) < 0.6, offs
    s = summary_table(res)
    assert "mz_offset_ppm" in s.columns


def test_cluster_weights_sum_to_clusters():
    from peak3d.anchors import cluster_weights
    rt = np.array([1.00, 1.01, 1.02, 3.0, 5.0, 5.005])
    width = np.full(6, 0.1)              # quarter base width = 0.025 min
    w = cluster_weights(rt, width)
    assert np.allclose(w, [1 / 3, 1 / 3, 1 / 3, 1, 0.5, 0.5])
    assert round(w.sum()) == 3
    # no chaining: evenly spaced anchors 0.02 apart form clusters of at most 2 with tol 0.025
    rt2 = np.arange(0, 1.0, 0.02)
    w2 = cluster_weights(rt2, np.full(len(rt2), 0.1))
    assert w2.min() >= 0.5 and round(w2.sum()) >= len(rt2) // 2


def _two_population_file(strong, weak, shift_strong, shift_weak, seed=0):
    """strong ions (anchor grade) and weak ions (score below every anchor tier, still clean peaks),
    each displaced by its own function of RT"""
    rng = np.random.default_rng(seed)
    parts = []
    for t, sh, score in ((strong, shift_strong, 0.9), (weak, shift_weak, 0.3)):
        rt = t["rt"].values + sh(t["rt"].values) + 0.2 * DT * rng.standard_normal(len(t))
        parts.append(pd.DataFrame({"mz": t["mz"].values * (1 + 0.3e-6 * rng.standard_normal(len(t))), "rt": rt,
                                   "height": t["height"].values, "score": score}))
    df = pd.concat(parts, ignore_index=True)
    df["rt_min"], df["rt_max"] = df["rt"] - 0.03, df["rt"] + 0.03
    df["snr"], df["n_scans"], df["mz_sd_ppm"], df["iso_offset"], df["iso_parent"] = 50.0, 12, 0.4, 0, -1
    df["feature_id"] = np.arange(len(df))
    return df[COLS]


def _pair_gap(res, feats, a, b, n_strong, rows):
    """median |corrected RT in b - corrected RT in a| (min) over the weak ions `rows` (same order in both files)"""
    ia, ib = n_strong + rows, n_strong + rows
    ua = res.files[a].warp.forward(feats[a]["rt"].values[ia])
    ub = res.files[b].warp.forward(feats[b]["rt"].values[ib])
    return float(np.median(np.abs(ub - ua)))


def test_gate_removes_unsupported_extrapolation_before_the_first_anchor():
    # anchor-grade ions only after 1.5 min; file b drifts by 2.4 s there, but not before 1.5 min,
    # where the anchor fit can only extrapolate its first knot as a constant shift
    tr = _truth(900, seed=4)
    strong, weak = tr.iloc[::3], tr.drop(tr.index[::3])
    strong = strong[strong["rt"] > 1.5].reset_index(drop=True)
    weak = weak.reset_index(drop=True)
    step = lambda t: np.where(t > 1.5, 0.04, 0.0)
    zero = lambda t: 0.0 * t
    feats = {"a": _two_population_file(strong, weak, zero, zero, seed=1),
             "b": _two_population_file(strong, weak, step, step, seed=2),
             "c": _two_population_file(strong, weak, zero, zero, seed=3)}
    grids = {k: GRID for k in feats}
    early = np.flatnonzero((weak["rt"].values > 0.4) & (weak["rt"].values < 1.3))
    late = np.flatnonzero((weak["rt"].values > 2.0) & (weak["rt"].values < T - 0.4))
    ns = len(strong)
    off = align_run({k: f.copy() for k, f in feats.items()}, grids, AlignParams(gate=False))
    on = align_run({k: f.copy() for k, f in feats.items()}, grids, AlignParams())
    assert on.medoid != "b"
    assert _pair_gap(off, feats, "a", "b", ns, early) > 0.03          # the fit extrapolates the 2.4 s shift
    assert _pair_gap(on, feats, "a", "b", ns, early) < 0.2 * 0.04     # the gate takes it back
    assert _pair_gap(on, feats, "a", "b", ns, late) < DT              # and keeps the supported correction
    g = on.files["b"].gate
    assert g["lam"][0] < 0.2 and g["lam"][-1] > 0.8 and g["dev_gated_s"] <= g["dev_native_s"]
    s = summary_table(on)
    assert {"gate_n_val", "gate_lambda_min", "gate_dev_applied_s"} <= set(s.columns)


def test_gate_shrinks_a_drift_only_the_anchors_have():
    # the anchor-grade ions of file b drift by 2.4 s, the (more numerous) other ions do not
    tr = _truth(900, seed=5)
    strong, weak = tr.iloc[::4].reset_index(drop=True), tr.drop(tr.index[::4]).reset_index(drop=True)
    shift = lambda t: 0.04 + 0.0 * t
    zero = lambda t: 0.0 * t
    feats = {"a": _two_population_file(strong, weak, zero, zero, seed=1),
             "b": _two_population_file(strong, weak, shift, zero, seed=2),
             "c": _two_population_file(strong, weak, zero, zero, seed=3)}
    grids = {k: GRID for k in feats}
    rows = np.flatnonzero((weak["rt"].values > 0.4) & (weak["rt"].values < T - 0.4))
    off = align_run({k: f.copy() for k, f in feats.items()}, grids, AlignParams(gate=False))
    on = align_run({k: f.copy() for k, f in feats.items()}, grids, AlignParams())
    assert on.medoid != "b"
    assert _pair_gap(off, feats, "a", "b", len(strong), rows) > 0.03
    assert _pair_gap(on, feats, "a", "b", len(strong), rows) < 0.2 * 0.04
    assert on.files["b"].warp_fit is not None and np.max(on.files["b"].gate["lam"]) < 0.2


def test_gate_keeps_a_coherent_warp():
    truth = _truth()
    W = _warps()
    feats = {k: _file(truth, W[k], n_decoy=100, seed=i) for i, k in enumerate(W)}
    res = align_run(feats, {k: GRID for k in W}, AlignParams())
    for k, fa in res.files.items():
        if k == res.medoid:
            assert fa.gate is None
            continue
        assert fa.gate is not None and np.median(fa.gate["lam"]) > 0.9, fa.gate["lam"]
        assert fa.gate["dev_gated_s"] < 0.5 * fa.gate["dev_native_s"]


def test_loo_median_and_scale_factor_match_brute_force():
    from peak3d.align import _loo_median, _scale_factor
    rng = np.random.default_rng(0)
    gid = rng.integers(0, 40, 400)
    gid = gid[np.isin(gid, np.flatnonzero(np.bincount(gid) >= 2))]
    u = rng.normal(size=len(gid))
    got = _loo_median(gid, u)
    for i in range(len(gid)):
        others = u[(gid == gid[i]) & (np.arange(len(gid)) != i)]
        assert np.isclose(got[i], np.median(others))
    lam_grid = np.linspace(0, 1, 2001)
    for seed in range(20):
        r = np.random.default_rng(seed)
        d, sh = r.normal(size=60), r.normal(0.5, 1.0, size=60)
        lam, gain = _scale_factor(d, sh)
        best = np.abs(d[None, :] + lam_grid[:, None] * sh[None, :]).sum(1).min()
        assert np.abs(d + lam * sh).sum() <= best + 1e-9
        assert 0.0 <= lam <= 1.0 and gain >= -1e-12
