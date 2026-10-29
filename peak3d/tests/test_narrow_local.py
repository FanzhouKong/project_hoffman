"""narrow_rule="local": "too narrow" judged against trusted peaks eluting nearby, not the file's median width."""
from dataclasses import replace

import numpy as np

from peak3d.estimate import Params, parse_overrides, prepare
from peak3d.features import build_features
from peak3d.io import from_scans
from peak3d.pick import pick_cloud

DT_S = 0.4                               # seconds per MS1 scan, uniform
N_SCANS = 3000
EARLY_FWHM, LATE_FWHM, BLIP_FWHM = 4.0, 15.0, 1.0
MEDIAN = {"narrow_rule": "median"}     # the file-wide rule the local ones are compared with


def cloud_of(peaks, seed=0):
    """peaks: (m/z, apex time in s, height, FWHM in s; real peaks carry an M+1 at 10 %), on a light scattered
    background"""
    rng = np.random.default_rng(seed)
    ts = np.arange(N_SCANS) * DT_S
    mzs, ins = [[] for _ in ts], [[] for _ in ts]
    for m, t0, h, fw in peaks:
        sd = fw / 2.3548
        for s in np.flatnonzero(np.abs(ts - t0) < 4 * sd):
            y = h * np.exp(-0.5 * ((ts[s] - t0) / sd) ** 2)
            mzs[s].append(m + rng.normal(0, m * 1e-6))
            ins[s].append(y)
            if fw > BLIP_FWHM:                  # compounds carry a 13C isotope peak; noise blips do not
                mzs[s].append(m + 1.0033548 + rng.normal(0, m * 1e-6))
                ins[s].append(0.1 * y)
    for s in range(len(ts)):
        for m in rng.uniform(150, 650, 6):
            mzs[s].append(m)
            ins[s].append(rng.uniform(800, 2000))
    order = [np.argsort(m) for m in mzs]
    return from_scans("T", ts / 60, [np.asarray(m)[o] for m, o in zip(mzs, order)], [np.asarray(i)[o] for i, o in zip(ins, order)])


def layout():
    early = [(200.0 + 3.1 * k, 40 + 9 * k, 10 ** (5 + 0.03 * k), EARLY_FWHM) for k in range(40)]            # 40-400 s
    late = [(330.0 + 3.3 * k, 500 + 11 * k, 10 ** (5 + 0.02 * k), LATE_FWHM) for k in range(60)]             # 500-1150 s
    blips = [(560.0 + 4.7 * k, t, 2e5, BLIP_FWHM) for k, t in enumerate([123, 247, 371, 611, 833, 1012])]
    return early, late, blips


def _found(f, m, t0):
    return int(((np.abs(f["mz"].values - m) < 0.005) & (np.abs(f["rt"].values * 60 - t0) < 4.0)).sum())


def _feats(cloud, P):
    f, _ = build_features(cloud, pick_cloud(cloud, P))
    return f


def test_default_and_settable():
    assert Params().narrow_rule == "local_median"                 # default since 2026-10-28 (variant GLRS)
    assert parse_overrides(["narrow_rule=local"]) == {"narrow_rule": "local"}


def test_narrow_early_peaks_judged_against_their_neighbours():
    early, late, blips = layout()
    cl = cloud_of(early + late + blips)
    P = prepare(cl, MEDIAN)
    assert 0.3 * P.fwhm_med * 60 > EARLY_FWHM               # the file-wide rule calls every early peak too narrow
    med = _feats(cl, P)
    loc = _feats(cl, replace(P, narrow_rule="local"))
    kept = lambda f, ps: sum(_found(f, m, t) for m, t, _, _ in ps)
    assert kept(med, early) <= 4 and kept(loc, early) >= 36  # 40 early peaks: lost by the median rule, kept locally
    assert kept(med, late) == kept(loc, late) == 60         # broad peaks: unchanged
    assert kept(loc, blips) == 0                            # 1 s blips: still too narrow for their neighbourhood


def test_falls_back_to_the_median_rule_with_few_trusted_peaks():
    early, late, blips = layout()
    cl = cloud_of(early[:5] + late[:10] + blips)
    P = prepare(cl, MEDIAN)
    med = _feats(cl, P)
    loc = _feats(cl, replace(P, narrow_rule="local"))
    assert len(med) == len(loc) and np.allclose(np.sort(med["mz"].values), np.sort(loc["mz"].values))


def test_local_median_keeps_narrow_early_peaks_and_rejects_late_blips():
    early, late, blips = layout()
    cl = cloud_of(early + late + blips)
    P = prepare(cl, MEDIAN)
    med = _feats(cl, P)
    loc = _feats(cl, replace(P, narrow_rule="local_median"))
    kept = lambda f, ps: sum(_found(f, m, t) for m, t, _, _ in ps)
    assert kept(med, early) <= 4 and kept(loc, early) >= 36
    assert kept(loc, late) == 60
    late_blips = [b for b in blips if b[1] > 450]            # among 15 s peaks, 0.3 x their median is 4.5 s
    assert kept(loc, late_blips) == 0
