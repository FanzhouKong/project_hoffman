"""min_scans_rule="local" (scan minimum at the candidate's own MS1 interval), bridge_scans (the bounds walk crosses
a short hole without a centroid inside the peak body) and narrow_rule="off" (no "too narrow" test)."""
from dataclasses import replace

import numpy as np
import pytest

from peak3d.estimate import Params, parse_overrides, prepare
from peak3d.features import build_features
from peak3d.io import from_scans
from peak3d.pick import pick_cloud

DENSE_S, SPARSE_S = 0.4, 2.9            # seconds between MS1 scans: an idle stretch and a busy data-dependent one
N_DENSE, N_SPARSE = 1500, 160
FWHM_S = 10.0                           # peaks in the fast stretch (they set the file's median width)


def run_times(n_sparse=N_SPARSE):
    t = np.concatenate([np.arange(N_DENSE) * DENSE_S, N_DENSE * DENSE_S + np.arange(n_sparse) * SPARSE_S])
    return t / 60.0                     # minutes


def cloud_of(peaks, holes=(), n_sparse=N_SPARSE, seed=0):
    """peaks: (m/z, apex time in s, height, FWHM in s); holes: (m/z, scan) scans where that ion has no centroid;
    a light scattered background"""
    rng = np.random.default_rng(seed)
    rt = run_times(n_sparse)
    ts = rt * 60.0
    drop = {(round(m, 4), s) for m, s in holes}
    mzs, ins = [[] for _ in rt], [[] for _ in rt]
    for m, t0, h, fw in peaks:
        sd = fw / 2.3548
        for s in np.flatnonzero(np.abs(ts - t0) < 4 * sd):
            if (round(m, 4), int(s)) in drop:
                continue
            mzs[s].append(m + rng.normal(0, m * 1e-6))
            ins[s].append(h * np.exp(-0.5 * ((ts[s] - t0) / sd) ** 2))
    for s in range(len(rt)):
        for m in rng.uniform(150, 650, 6):
            mzs[s].append(m)
            ins[s].append(rng.uniform(800, 2000))
    order = [np.argsort(m) for m in mzs]
    return from_scans("T", rt, [np.asarray(m)[o] for m, o in zip(mzs, order)], [np.asarray(i)[o] for i, o in zip(ins, order)])


def _feats(cloud, P):
    f, _ = build_features(cloud, pick_cloud(cloud, P))
    return f


def _near(f, m, t0, tol_s=6.0):
    return f[(np.abs(f["mz"].values - m) < 0.005) & (np.abs(f["rt"].values * 60 - t0) < tol_s)]


def dense_peaks():
    return [(200.0 + 7.1 * k, 30 + 18 * k, 10 ** (5 + 0.05 * k), FWHM_S) for k in range(30)]


def test_defaults_and_settable():
    P = Params()
    assert P.min_scans_rule == "local" and P.bridge_scans == 0   # local scan minimum by default since 2026-10-28 (GLRS)
    assert parse_overrides(["min_scans_rule=local", "bridge_scans=2"]) == {"min_scans_rule": "local", "bridge_scans": 2}
    with pytest.raises(ValueError):
        parse_overrides(["min_scans_rule=scan"])


def test_pick_prefilter_is_the_lowest_local_minimum():
    cl = cloud_of(dense_peaks())
    P = prepare(cl, {"min_scans_rule": "local"})
    assert P.min_scans == 5 and P.min_scans_pick == 3          # 10 s = 25 scans at 0.4 s, 3.4 scans at 2.9 s
    uni = cloud_of(dense_peaks(), n_sparse=0)
    Pu = prepare(uni, {"min_scans_rule": "local"})
    assert Pu.min_scans_pick == Pu.min_scans == 5               # uniform sampling: the pre-filter is unchanged


def test_local_minimum_keeps_slow_stretch_peaks():
    t_sparse0 = N_DENSE * DENSE_S
    sparse = [(400.0 + 9.3 * k, t_sparse0 + 40 + 35 * k, 3e5 + 4e4 * k, 4.5) for k in range(10)]   # 3-4 scans each
    cl = cloud_of(dense_peaks() + sparse)
    P = prepare(cl, {"min_scans_rule": "file"})
    fw = _feats(cl, P)
    lo = _feats(cl, replace(P, min_scans_rule="local", min_scans_pick=3))
    assert all(len(_near(fw, m, t)) == 1 and len(_near(lo, m, t)) == 1 for m, t, _, _ in dense_peaks())
    n_file = sum(len(_near(fw, m, t)) for m, t, _, _ in sparse)
    n_local = sum(len(_near(lo, m, t)) for m, t, _, _ in sparse)
    assert n_file <= 2 and n_local >= 8
    # nothing else changes: the width test and the score still use the file's numbers
    assert "dt_local" not in lo.columns and "w_local" not in lo.columns


def _scan_of(cl, t_s):
    return int(np.argmin(np.abs(cl.rt * 60 - t_s)))


def test_bridge_crosses_a_dropout_next_to_the_apex():
    m, t0 = 333.1234, 400.0
    base = cloud_of(dense_peaks() + [(m, t0, 2e6, FWHM_S)])
    s0 = _scan_of(base, t0)
    cl = cloud_of(dense_peaks() + [(m, t0, 2e6, FWHM_S)], holes=[(m, s0 + 1), (m, s0 + 2)])
    P = prepare(cl)
    off = _near(_feats(cl, P), m, t0, 20)
    on = _near(_feats(cl, replace(P, bridge_scans=2)), m, t0, 20)
    assert len(off) == 1 and len(on) == 1
    sd = FWHM_S / 2.3548
    assert off["rt_max"].iloc[0] * 60 < t0 + 1.0               # the walk stops at the hole
    assert on["rt_max"].iloc[0] * 60 > t0 + 2 * sd              # bridged: the far side of the peak counts
    assert on["n_scans"].iloc[0] > off["n_scans"].iloc[0] + 10


def test_bridge_leaves_a_hole_in_the_tail():
    m, t0 = 333.1234, 400.0
    base = cloud_of(dense_peaks() + [(m, t0, 2e6, FWHM_S)])
    s0 = _scan_of(base, t0)
    sd_scans = FWHM_S / 2.3548 / DENSE_S
    s_hole = s0 + int(round(2.5 * sd_scans))                   # ~4% of the apex: the trace is ending there
    cl = cloud_of(dense_peaks() + [(m, t0, 2e6, FWHM_S)], holes=[(m, s_hole), (m, s_hole + 1)])
    P = prepare(cl)
    off = _near(_feats(cl, P), m, t0, 20)
    on = _near(_feats(cl, replace(P, bridge_scans=2)), m, t0, 20)
    assert len(off) == 1 and len(on) == 1
    assert on["rt_max"].iloc[0] == off["rt_max"].iloc[0]
    assert on["rt_max"].iloc[0] * 60 < cl.rt[s_hole] * 60


def test_narrow_rule_off_keeps_a_sharp_peak_with_enough_scans():
    m, t0 = 333.1234, 400.0
    cl = cloud_of(dense_peaks() + [(m, t0, 5e5, 2.0)])          # 2 s FWHM: 5 scans at 0.4 s, below 0.3 x 10 s
    P = prepare(cl, {"min_scans_rule": "local"})
    assert parse_overrides(["narrow_rule=off"]) == {"narrow_rule": "off"}
    assert len(_near(_feats(cl, P), m, t0)) == 0
    off = _feats(cl, replace(P, narrow_rule="off"))
    assert len(_near(off, m, t0)) == 1
    assert all(len(_near(off, mm, t)) == 1 for mm, t, _, _ in dense_peaks())
