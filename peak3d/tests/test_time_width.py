"""width_rule="time": scan counts and widths judged in the local MS1 sampling interval."""
from dataclasses import replace

import numpy as np

from peak3d import kernels as K
from peak3d.estimate import Params, parse_overrides, prepare
from peak3d.features import build_features
from peak3d.io import from_scans
from peak3d.pick import pick_cloud

DENSE_S, SPARSE_S = 0.4, 2.9            # seconds between MS1 scans: an idle stretch and a busy data-dependent one
N_DENSE, N_SPARSE = 1500, 160
FWHM_S = 10.0                           # peaks in the fast stretch (they set the file's median width)
FWHM_SPARSE_S = 3.5                     # peaks in the busy stretch elute narrower, as in the urine QTOF run (3-6 s)


def run_times():
    t = np.concatenate([np.arange(N_DENSE) * DENSE_S, N_DENSE * DENSE_S + np.arange(N_SPARSE) * SPARSE_S])
    return t / 60.0                     # minutes


def cloud_of(peaks, spikes=(), seed=0):
    """peaks: (m/z, apex time in s, height, FWHM in s); spikes: (m/z, scan, height) single-scan blips; a light
    background"""
    rng = np.random.default_rng(seed)
    rt = run_times()
    ts = rt * 60.0
    mzs, ins = [[] for _ in rt], [[] for _ in rt]
    for m, t0, h, fw in peaks:
        sd = fw / 2.3548
        for s in np.flatnonzero(np.abs(ts - t0) < 4 * sd):
            mzs[s].append(m + rng.normal(0, m * 1e-6))
            ins[s].append(h * np.exp(-0.5 * ((ts[s] - t0) / sd) ** 2))
    for m, s, h in spikes:
        mzs[s].append(m)
        ins[s].append(h)
    for s in range(len(rt)):                # scattered background centroids (the noise surface)
        for m in rng.uniform(150, 650, 6):
            mzs[s].append(m)
            ins[s].append(rng.uniform(800, 2000))
    order = [np.argsort(m) for m in mzs]
    return from_scans("T", rt, [np.asarray(m)[o] for m, o in zip(mzs, order)], [np.asarray(i)[o] for i, o in zip(ins, order)])


def _feats(cloud, P):
    f, _ = build_features(cloud, pick_cloud(cloud, P))
    return f


def _found(f, m, t0):
    return int(((np.abs(f["mz"].values - m) < 0.005) & (np.abs(f["rt"].values * 60 - t0) < 6.0)).sum())


def layout():
    dense = [(200.0 + 7.1 * k, 30 + 18 * k, 10 ** (5 + 0.05 * k), FWHM_S) for k in range(30)]   # set the median width
    t_sparse0 = N_DENSE * DENSE_S
    sparse = [(400.0 + 9.3 * k, t_sparse0 + 40 + 35 * k, 3e5 + 4e4 * k, FWHM_SPARSE_S) for k in range(10)]
    spike_scans = [N_DENSE + 25 + 12 * k for k in range(5)]
    spikes = [(600.0 + 3.7 * k, s, 3e5) for k, s in enumerate(spike_scans)]
    return dense, sparse, spikes


def test_default_unchanged_and_settable():
    assert Params().width_rule == "fwhm"
    assert parse_overrides(["width_rule=time"]) == {"width_rule": "time"}


def test_local_scan_interval():
    rt = run_times()
    lo = np.array([100, N_DENSE + 40], dtype=np.int64)
    dt = K.local_scan_dt(rt, lo, lo + 4, 2) * 60
    assert abs(dt[0] - DENSE_S) < 1e-6 and abs(dt[1] - SPARSE_S) < 1e-6


def test_peaks_in_the_slow_stretch_kept_like_peaks_in_the_fast_one():
    dense, sparse, spikes = layout()
    cl = cloud_of(dense + sparse, spikes)
    P = prepare(cl, {"min_scans_rule": "file", "narrow_rule": "median"})   # the file-wide rules, to compare against
    assert abs(P.fwhm_med * 60 - FWHM_S) < 2.5 and P.min_scans == 5        # 25 scans wide where sampling is fast
    fw = _feats(cl, P)
    tm = _feats(cl, replace(P, width_rule="time", min_scans=3))
    # fast stretch: the same peaks either way
    assert all(_found(fw, m, t) == 1 and _found(tm, m, t) == 1 for m, t, _, _ in dense)
    # slow stretch: a 3.5 s peak spans 3-4 scans; the file-wide rule (5 scans, 0.3 x 10 s) loses them, the local one
    # keeps them
    assert sum(_found(fw, m, t) for m, t, _, _ in sparse) <= 2
    assert all(_found(tm, m, t) == 1 for m, t, _, _ in sparse)
    # single-scan blips stay rejected
    rt = cl.rt * 60
    assert all(_found(tm, m, rt[s]) == 0 for m, s, _ in spikes)
