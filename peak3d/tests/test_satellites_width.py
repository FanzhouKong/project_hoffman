"""Side-peak (satellite) merging and the scan-count width rule."""
from dataclasses import replace

import numpy as np

from peak3d.estimate import Params, estimate
from peak3d.features import build_features
from peak3d.pick import pick_cloud
from peak3d.synth import Peak, make_cloud

DT = 0.005
FW = 0.05           # 10 scans per FWHM
BG = dict(n_scans=800, dt=DT, thresh=200.0, floor=100.0, n_noise=30000, sig_a=0.5, sig_b=30.0)
SAT = 0.002         # side peaks 2 mDa from their ion, as on the MassCube Orbitrap files


def _feats(cloud, P):
    feats, _ = build_features(cloud, pick_cloud(cloud, P))
    return feats


def _near(feats, mz, rt, da=0.004, rt_tol=0.02):
    return feats[(np.abs(feats["mz"] - mz) <= da) & (np.abs(feats["rt"] - rt) <= rt_tol)]


def _with_satellites(n=30, frac=0.05, seed=0):
    """n strong ions (m/z 60-400), each with two side peaks at +/- SAT of frac x its height, same elution profile"""
    rng = np.random.default_rng(seed)
    ions = [(float(m), float(r)) for m, r in zip(np.sort(rng.uniform(60, 400, n)) + np.arange(n) * 0.37, rng.uniform(0.4, 3.6, n))]
    peaks = []
    for m, r in ions:
        h = 10 ** rng.uniform(6, 6.7)
        peaks += [Peak(m, r, h, FW), Peak(m - SAT, r, frac * h, FW), Peak(m + SAT, r, frac * h, FW)]
    return ions, peaks


def test_defaults_unchanged():
    P = Params()
    assert P.width_rule == "fwhm" and not P.satellite_merge and P.sat_da == 0.0 and P.narrow_rule == "local_median"


def test_side_peaks_are_learned_and_folded_into_their_ion():
    ions, peaks = _with_satellites()
    cloud, _ = make_cloud(peaks, seed=1, **BG)
    P = estimate(cloud, overrides=dict(satellite_merge=True))
    assert P.n_sat >= 20 and SAT <= P.sat_da <= 1.5 * SAT     # an upper percentile of the measured distances
    base = _feats(cloud, replace(P, satellite_merge=False, link_da=0.0))      # the merge alone, without the 2 mDa link
    merged = _feats(cloud, replace(P, link_da=0.0))
    n_base = [len(_near(base, m, r)) for m, r in ions]
    n_merged = [len(_near(merged, m, r)) for m, r in ions]
    # below m/z 250, 2 mDa is > 8 ppm: without the merge the ion and both side peaks are reported separately
    # (higher up the ppm grouping already folds them)
    assert all(k == 3 for k, (m, _) in zip(n_base, ions) if m < 250)
    assert all(k == 1 for k in n_merged)
    for m, r in ions:                                     # the ion keeps its own m/z (side peaks are symmetric and weak)
        f = _near(merged, m, r).iloc[0]
        assert abs(f["mz"] - m) / m * 1e6 < 1.0


def test_split_pair_without_a_centre_becomes_one_ion_at_the_centre():
    ions, peaks = _with_satellites(seed=2)
    m0, r0 = 203.0835, 2.05
    peaks += [Peak(m0 - SAT, r0, 2e6, FW), Peak(m0 + SAT, r0, 1.6e6, FW)]
    cloud, _ = make_cloud(peaks, seed=3, **BG)
    P = estimate(cloud, overrides=dict(satellite_merge=True))
    f = _near(_feats(cloud, P), m0, r0)
    assert len(f) == 1
    assert abs(f.iloc[0]["mz"] - (m0 + SAT * (1.6 - 2.0) / 3.6)) < 0.0003      # height-weighted centre


def test_different_ions_that_elute_apart_are_not_merged():
    ions, peaks = _with_satellites(seed=4)
    peaks += [Peak(150.0503, 2.0, 1e6, FW), Peak(150.0503 + SAT, 2.15, 8e5, FW)]
    cloud, _ = make_cloud(peaks, seed=5, **BG)
    P = estimate(cloud, overrides=dict(satellite_merge=True))
    feats = _feats(cloud, P)
    assert len(_near(feats, 150.0503, 2.0)) == 1 and len(_near(feats, 150.0503 + SAT, 2.15)) == 1


def test_file_without_side_peaks_is_untouched():
    rng = np.random.default_rng(6)
    peaks = [Peak(float(m), float(r), float(h), FW) for m, r, h in
             zip(rng.uniform(80, 900, 60), rng.uniform(0.4, 3.6, 60), 10 ** rng.uniform(5, 6.5, 60))]
    cloud, _ = make_cloud(peaks, seed=7, **BG)
    P = estimate(cloud, overrides=dict(satellite_merge=True))
    assert P.sat_da == 0.0
    a, b = _feats(cloud, replace(P, satellite_merge=False)), _feats(cloud, P)
    assert len(a) == len(b) and np.allclose(a["mz"].values, b["mz"].values)


def test_narrow_peak_kept_by_scan_rule_but_blip_still_rejected():
    # broad ions set a median FWHM of 20 scans; a clean narrow peak (FWHM 4 scans) fails 0.3 x median under "fwhm"
    broad = [Peak(300 + 25 * i, 0.5 + 0.15 * i, 10 ** (5 + 0.1 * i), 0.1) for i in range(20)]
    narrow = Peak(412.3, 2.0, 3e5, 0.02)
    cloud, _ = make_cloud(broad + [narrow], seed=8, **BG)
    s = int(np.argmin(np.abs(cloud.rt - 3.0)))           # a one-scan blip on an otherwise empty m/z
    a, b = cloud.off[s], cloud.off[s + 1]
    k = a + int(np.searchsorted(cloud.mz[a:b], 455.5))
    cloud.mz = np.insert(cloud.mz, k, 455.5)
    cloud.inten = np.insert(cloud.inten, k, np.float32(2e5))
    cloud.off = cloud.off.copy()
    cloud.off[s + 1:] += 1
    P = estimate(cloud)
    assert P.fwhm_scans > 15
    fw = _feats(cloud, P)
    sc = _feats(cloud, estimate(cloud, overrides=dict(width_rule="scans")))
    assert len(_near(fw, 412.3, 2.0)) == 0
    assert len(_near(sc, 412.3, 2.0)) == 1
    assert len(_near(sc, 455.5, 3.0)) == 0 and len(_near(fw, 455.5, 3.0)) == 0


# link_da: absolute (Da) floor on the same-scan link, so an ion's pieces a few mDa away at low m/z build one trace

def test_link_da_default_2mda_and_settable():
    from peak3d.estimate import parse_overrides
    assert Params().link_da == 0.002                              # default since 2026-10-22
    assert parse_overrides(["link_da=0.002"]) == {"link_da": 0.002}


def test_low_mz_side_peaks_join_their_ion_trace():
    ions, peaks = _with_satellites(seed=10)
    cloud, _ = make_cloud(peaks, seed=11, **BG)
    P = estimate(cloud)
    low = [(m, r) for m, r in ions if m < 250]
    base = _feats(cloud, replace(P, link_da=0.0))
    linked = _feats(cloud, replace(P, link_da=0.0025))
    assert all(len(_near(base, m, r)) == 3 for m, r in low)       # ion + both side peaks 2 mDa (> 8 ppm) away
    for m, r in ions:
        f = _near(linked, m, r)
        assert len(f) == 1
        assert abs(f.iloc[0]["mz"] - m) / m * 1e6 < 1.0             # the ion's own m/z, not pulled by its side peaks


def test_ion_beyond_the_floor_stays_separate():
    ions, peaks = _with_satellites(n=10, seed=12)
    m0, r0 = 78.0115, 2.0
    peaks += [Peak(m0, r0, 3e6, FW), Peak(m0 + 0.006, r0, 1e6, FW)]   # 6 mDa away, same elution: kept apart at 2.5 mDa
    cloud, _ = make_cloud(peaks, seed=13, **BG)
    feats = _feats(cloud, replace(estimate(cloud), link_da=0.0025))
    assert len(_near(feats, m0, r0, da=0.002)) == 1 and len(_near(feats, m0 + 0.006, r0, da=0.002)) == 1


def test_floor_below_the_ppm_floor_changes_nothing():
    rng = np.random.default_rng(14)
    peaks = [Peak(float(m), float(r), float(h), FW) for m, r, h in
             zip(rng.uniform(500, 900, 60), rng.uniform(0.4, 3.6, 60), 10 ** rng.uniform(5, 6.5, 60))]
    cloud, _ = make_cloud(peaks, seed=15, **BG)
    P = estimate(cloud)
    a, b = _feats(cloud, replace(P, link_da=0.0)), _feats(cloud, replace(P, link_da=0.002))   # 2 mDa < 5 ppm above m/z 400
    assert a.drop(columns=["feature_id"]).equals(b.drop(columns=["feature_id"]))
