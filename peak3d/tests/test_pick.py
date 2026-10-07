from dataclasses import replace

import numpy as np
import pytest

from peak3d import kernels as K
from peak3d.estimate import estimate
from peak3d.features import build_features
from peak3d.pick import pick_cloud
from peak3d.synth import Peak, _trapezoid, make_cloud, permute_within_scans

DT = 0.005
FW = 0.05           # 10 scans per FWHM
BG = dict(n_scans=800, dt=DT, thresh=200.0, floor=100.0, n_noise=30000, sig_a=0.5, sig_b=30.0)


def _pick(peaks, seed=0, **kw):
    args = dict(BG)
    args.update(kw)
    cloud, truth = make_cloud(peaks, seed=seed, **args)
    b = pick_cloud(cloud)
    feats, _ = build_features(cloud, b)
    return cloud, truth, b, feats


def _near(feats, mz, rt, ppm=5.0, rt_tol=0.01):
    m = (np.abs(feats["mz"] - mz) / mz * 1e6 <= ppm) & (np.abs(feats["rt"] - rt) <= rt_tol)
    return feats[m]


def test_single_gaussian_exact():
    cloud, truth, b, feats = _pick([Peak(400.2, 2.0, 1e6, FW)])
    f = _near(feats, 400.2, 2.0)
    assert len(f) == 1
    f = f.iloc[0]
    assert abs(f.mz - 400.2) / 400.2 * 1e6 <= 0.5
    assert abs(f.rt - 2.0) <= DT / 2
    assert f.height == pytest.approx(1e6, rel=1e-6)
    assert f.area == pytest.approx(truth.iloc[0].area, rel=0.02)
    assert f.fwhm == pytest.approx(FW, rel=0.1)
    assert f.n_gaps == 0 and f["flags"] == ""


def test_unique_membership_partitions_the_cloud():
    cloud, truth, b, feats = _pick([Peak(200 + 50 * i, 0.3 + 0.17 * i, 10 ** (4 + 0.1 * i), FW) for i in range(20)])
    lab = b.lab
    assert lab.min() >= 0 and len(lab) == cloud.n_points
    counts = np.bincount(lab)
    assert counts.sum() == cloud.n_points
    for bid in b.basin_id[:10]:
        pts = np.flatnonzero(lab == bid)
        assert len(pts) == int(b.col("n_points")[list(b.basin_id).index(bid)])
    assert len(_near(feats, 200.0, 0.3)) == 1 and len(_near(feats, 200 + 50 * 19, 0.3 + 0.17 * 19)) == 1


def test_two_peaks_same_mz_resolved_at_1p5_fwhm():
    cloud, truth, b, feats = _pick([Peak(300.1, 2.0, 1e6, FW), Peak(300.1, 2.0 + 1.5 * FW, 1e6, FW)])
    f = feats[np.abs(feats["mz"] - 300.1) < 0.005]
    assert len(f) == 2
    assert sorted(np.round(f["rt"].values, 3)) == pytest.approx([2.0, 2.0 + 1.5 * FW], abs=DT)


def test_two_peaks_same_mz_unresolvable_at_0p8_fwhm():
    cloud, truth, b, feats = _pick([Peak(300.1, 2.0, 1e6, FW), Peak(300.1, 2.0 + 0.8 * FW, 1e6, FW)])
    f = feats[np.abs(feats["mz"] - 300.1) < 0.005]
    assert len(f) == 1


def test_small_peak_on_tail_keeps_own_bounds():
    # the valley between the tail and the small peak must fall below half the small apex for the
    # two to count as resolved (chromatographic criterion): 30 % height, 2.5 widths out
    big = Peak(500.3, 2.0, 1e6, FW, tau=0.06)
    small = Peak(500.3, 2.0 + 2.5 * FW + 0.06, 3e5, FW)
    cloud, truth, b, feats = _pick([big, small])
    f = feats[np.abs(feats["mz"] - 500.3) < 0.005].sort_values("rt")
    assert len(f) == 2
    s = f.iloc[1]
    from peak3d.synth import profile
    tail = 1e6 * float(profile(np.array([small.rt]), big)[0])
    assert s["height"] == pytest.approx(3e5 + tail, rel=0.1)   # sits on the tail: height includes it
    assert s.rt_min >= 2.0 + FW                           # does not reach back into the big peak
    assert f.iloc[0].rt_max <= s.rt


# real peaks at other masses set the file's median peak width, as in a real file
WIDTH_SETTERS = [Peak(300 + 25 * i, 0.5 + 0.15 * i, 10 ** (5 + 0.1 * i), FW) for i in range(20)]


@pytest.mark.parametrize("h2", [1.5e5, 2.0e5])
def test_second_peak_far_beside_bigger_one_resolved(h2):
    # a second compound 3 widths after a bigger, tailing one: the valley stays above half its own apex
    # (54-65 %), so the half-valley rule alone merges it; that far apart a shallower valley separates
    big = Peak(500.3, 2.0, 1e6, FW, tau=0.04)
    second = Peak(500.3, 2.0 + 3 * FW, h2, FW)
    cloud, _ = make_cloud([big, second] + WIDTH_SETTERS, seed=3, **BG)
    P = estimate(cloud)
    for far_frac, n_expected in ((0.0, 1), (P.far_frac, 2)):
        b = pick_cloud(cloud, replace(P, far_frac=far_frac))
        feats, _ = build_features(cloud, b)
        f = feats[np.abs(feats["mz"] - 500.3) < 0.005].sort_values("rt")
        assert len(f) == n_expected, (far_frac, len(f))
    assert f.iloc[1]["rt"] == pytest.approx(second.rt, abs=2 * DT)
    assert f.iloc[0]["rt"] == pytest.approx(2.0, abs=DT)


@pytest.mark.parametrize("dip_fwhm,depth", [(1.0, 0.3), (2.0, 0.45)])
def test_suppression_dip_inside_broad_peak_stays_one(dip_fwhm, depth):
    # a co-eluting suppressor carves a dip (FWHM 1-2 median widths, 30-45 % deep) into the top of a
    # peak three median widths wide: still one compound, the far-apart rule must not split it
    cloud, _ = make_cloud([Peak(600.2, 2.0, 1e6, 3 * FW)] + WIDTH_SETTERS, seed=3, **BG)
    sel = np.abs(cloud.mz - 600.2) < 0.01
    t = cloud.rt[np.searchsorted(cloud.off, np.flatnonzero(sel), side="right") - 1]
    sd = dip_fwhm * FW / 2.3548
    cloud.inten[sel] *= (1 - depth * np.exp(-0.5 * ((t - 2.0) / sd) ** 2)).astype(np.float32)
    b = pick_cloud(cloud)
    feats, _ = build_features(cloud, b)
    assert len(feats[np.abs(feats["mz"] - 600.2) < 0.005]) == 1


def test_neighbouring_masses_separate():
    tol = 3 * np.sqrt(2) * 0.5 * 1e-6 * 400.0 * 3     # 3x the pair tolerance at the apex
    cloud, truth, b, feats = _pick([Peak(400.0, 2.0, 1e6, FW), Peak(400.0 + tol, 2.0, 8e5, FW)])
    f = feats[(feats["mz"] > 399.99) & (feats["mz"] < 400.01)]
    assert len(f) == 2


def test_split_centroid_absorbed():
    # a 10 % sliver 2 ppm away in every scan is the same ion; merge_ppm=0 keeps it separate in the data
    main = Peak(400.0, 2.0, 1e6, FW)
    sliver = Peak(400.0 * (1 + 2e-6), 2.0, 1e5, FW)
    cloud, truth, b, feats = _pick([main, sliver], merge_ppm=0.0)
    f = _near(feats, 400.0, 2.0)
    assert len(f) == 1
    assert f.iloc[0].height == pytest.approx(1.1e6, rel=1e-3)   # the sliver's signal is summed back in


def test_dropped_centroids_bridged():
    cloud, truth, b, feats = _pick([Peak(400.0, 2.0, 2e4, FW)], dropout_scale=4e3)
    f = _near(feats, 400.0, 2.0)
    assert len(f) == 1
    f = f.iloc[0]
    assert f.n_gaps >= 1 and "gap" in f["flags"]
    assert f.height == pytest.approx(2e4, rel=0.05)


def test_deep_single_scan_dip_stays_one_peak():
    # ion suppression / AGC: the centre scan keeps a centroid at 40 % of the expected intensity
    cloud, truth = make_cloud([Peak(400.0, 2.0, 1e6, FW)], **BG)
    scan = int(np.argmax(cloud.eic(400.0, 5.0)))
    a, b = cloud.off[scan], cloud.off[scan + 1]
    sel = a + np.flatnonzero(np.abs(cloud.mz[a:b] - 400.0) < 0.01)
    cloud.inten[sel] *= np.float32(0.4)
    b_ = pick_cloud(cloud)
    feats, _ = build_features(cloud, b_)
    f = feats[np.abs(feats["mz"] - 400.0) < 0.005]
    assert len(f) == 1
    f = f.iloc[0]
    assert abs(f.rt - 2.0) <= DT
    assert f.n_scans >= 20 and f.rt_min < 2.0 - FW and f.rt_max > 2.0 + FW
    assert f.area == pytest.approx(truth.iloc[0].area, rel=0.1)


def test_ragged_background_ion_is_rejected_but_peak_on_background_kept():
    # a constant ion over the whole run with 40 % multiplicative ripple: no feature
    rng = np.random.default_rng(11)
    # real peaks at other masses set the file's peak-width scale, as in a real file
    real = [Peak(300 + 25 * i, 0.5 + 0.15 * i, 10 ** (5 + 0.1 * i), FW) for i in range(20)]
    cloud, _ = make_cloud([Peak(89.0266, 2.0, 2e5, 100.0)] + real, **dict(BG, thresh=0.0))   # ridge ~flat over 4 min
    sel = np.abs(cloud.mz - 89.0266) < 0.01
    cloud.inten[sel] *= np.exp(0.4 * rng.standard_normal(sel.sum())).astype(np.float32)
    # plus a real peak on top of a 20 % background of the same m/z, elsewhere
    cloud2, _ = make_cloud([Peak(400.0, 2.0, 1e6, FW), Peak(400.0, 2.0, 2e5, 100.0)], **dict(BG, thresh=0.0))
    # the ridge: at most a couple of extreme (>3x background) spikes may survive; the peak on background: exactly one
    for c, mz0, lo, hi in ((cloud, 89.0266, 0, 2), (cloud2, 400.0, 1, 1)):
        b = pick_cloud(c)
        feats, _ = build_features(c, b)
        f = feats[np.abs(feats["mz"] - mz0) / mz0 * 1e6 <= 20]
        assert lo <= len(f) <= hi, (mz0, len(f))
    assert "ridge_ratio" in feats.columns and "background_ratio" in feats.columns


def test_noise_only_yields_nothing():
    cloud, truth, b, feats = _pick([], n_noise=200000)
    assert len(feats) <= 2


def test_flat_top_centred_and_flagged():
    cloud, truth, b, feats = _pick([Peak(400.0, 2.0, 1e6, FW, clip=0.9)])
    f = _near(feats, 400.0, 2.0, rt_tol=0.02)
    assert len(f) == 1
    f = f.iloc[0]
    assert abs(f.rt - 2.0) <= DT
    assert "flat_top" in f["flags"]


def test_wide_noisy_peak_single_apex():
    cloud, truth = make_cloud([Peak(400.0, 4.0, 1e6, 60 * DT)], n_scans=1600, dt=DT, thresh=200.0, floor=100.0,
                              n_noise=30000, seed=5)
    # 4 % multiplicative noise on the peak points
    rng = np.random.default_rng(1)
    sel = np.abs(cloud.mz - 400.0) < 0.01
    cloud.inten[sel] *= np.exp(0.04 * rng.standard_normal(sel.sum())).astype(np.float32)
    b = pick_cloud(cloud)
    feats, _ = build_features(cloud, b)
    f = feats[np.abs(feats["mz"] - 400.0) < 0.005]
    assert len(f) == 1


def test_acquisition_gap_not_interpolated():
    gap = list(range(395, 405))   # 10 scans removed inside the peak (apex scan 400)
    cloud, truth = make_cloud([Peak(400.0, 2.0, 1e6, 0.1)], n_scans=800, dt=DT, thresh=200.0, drop_scans=gap)
    b = pick_cloud(cloud)
    feats, _ = build_features(cloud, b)
    f = feats[np.abs(feats["mz"] - 400.0) < 0.005]
    assert len(f) == 1
    f = f.iloc[0]
    # area equals the trapezoid over the points actually present (one straight segment across the gap)
    e = cloud.eic(400.0, 5.0)
    keep = e > 0
    assert f.area == pytest.approx(_trapezoid(e[keep], cloud.rt[keep]), rel=0.02)
    assert f.n_gaps == 0  # the grid has no scans there, so nothing is counted as missing


def test_edge_of_run_flagged():
    cloud, truth, b, feats = _pick([Peak(400.0, 2 * DT, 1e6, FW)])
    f = feats[np.abs(feats["mz"] - 400.0) < 0.005]
    assert len(f) == 1 and "edge" in f.iloc[0]["flags"]


def test_snr_relative_to_floor():
    out = []
    for floor in (100.0, 1000.0):
        cloud, truth, b, feats = _pick([Peak(400.0, 2.0, 1e5, FW)], floor=floor, thresh=0.3 * floor)
        out.append(_near(feats, 400.0, 2.0).iloc[0])
    assert out[0].snr / out[1].snr == pytest.approx(10.0, rel=0.3)
    assert out[0].noise == pytest.approx(100.0, rel=0.2)


def test_isotope_annotation_including_z2():
    cloud, truth, b, feats = _pick([Peak(400.0, 2.0, 1e6, FW, iso=(0.2, 0.03)),
                                    Peak(600.0, 3.0, 1e6, FW, z=2, iso=(0.4,))])
    m = _near(feats, 400.0, 2.0).iloc[0]
    m1 = _near(feats, 400.0 + 1.0033548, 2.0).iloc[0]
    m2 = _near(feats, 400.0 + 2 * 1.0033548, 2.0).iloc[0]
    assert m.iso_offset == 0 and m1.iso_offset == 1 and m2.iso_offset == 2
    assert m1.iso_parent == m.feature_id and m2.iso_parent == m.feature_id
    assert m.charge == 1 and m1.charge == 1
    d = _near(feats, 600.0, 3.0).iloc[0]
    d1 = _near(feats, 600.0 + 1.0033548 / 2, 3.0).iloc[0]
    assert d.charge == 2 and d1.iso_offset == 1 and d1.iso_parent == d.feature_id


def test_determinism_and_order_invariance():
    peaks = [Peak(200 + 37 * i, 0.5 + 0.2 * i, 10 ** (3.5 + 0.1 * i), FW) for i in range(15)]
    cloud, _ = make_cloud(peaks, seed=4, **BG)
    f1, _ = build_features(cloud, pick_cloud(cloud))
    f2, _ = build_features(cloud, pick_cloud(cloud))
    f3, _ = build_features(permute_within_scans(cloud, 9), pick_cloud(permute_within_scans(cloud, 9)))
    assert f1.equals(f2) and f1.equals(f3)


def test_no_bench_imports_or_truth_reads():
    import pathlib
    import re
    pkg = pathlib.Path(__file__).resolve().parents[1]
    for f in pkg.glob("*.py"):
        src = f.read_text()
        assert not re.search(r"^\s*(from|import)\s+bench", src, re.M), f
        assert "data/truth" not in src, f


@pytest.mark.perf
def test_perf_smoke():
    import time
    rng = np.random.default_rng(0)
    peaks = [Peak(float(m), float(r), float(h), FW) for m, r, h in
             zip(rng.uniform(100, 1000, 3000), rng.uniform(0.3, 9.7, 3000), 10 ** rng.uniform(3.5, 6.5, 3000))]
    cloud, _ = make_cloud(peaks, n_scans=2000, dt=DT, thresh=200.0, floor=100.0, n_noise=1_900_000, seed=2)
    assert cloud.n_points >= 1_900_000
    pick_cloud(cloud)                 # warm-up (JIT)
    t0 = time.perf_counter()
    b = pick_cloud(cloud)
    assert time.perf_counter() - t0 < 5.0
    assert len(b.basin_id) >= 2900


def test_noise_surface_ignores_ripples_on_real_traces():
    # forty broad, ragged ions (50 % multiplicative ripple) fill one m/z band and minute: before the
    # merge every ripple maximum is its own one- or two-scan basin and would count as noise; the
    # surface must come from what is still isolated after the merge, so a clean peak at 10x the
    # floor noise beside them keeps its S/N
    rng = np.random.default_rng(5)
    ragged = [Peak(600.0 + 2.5 * i, 2.1 + 0.02 * i, 3e5, 4 * FW) for i in range(40)]
    weak = Peak(651.2345, 2.5, 1000.0, FW)
    cloud, _ = make_cloud(ragged + [weak] + WIDTH_SETTERS, seed=3, **BG)
    sel = np.zeros(len(cloud.mz), bool)
    for pk in ragged:
        sel |= np.abs(cloud.mz - pk.mz) < 0.01
    cloud.inten[sel] *= np.exp(0.5 * rng.standard_normal(sel.sum())).astype(np.float32)
    P = estimate(cloud)
    old = pick_cloud(cloud, replace(P, noise_post_merge=False))
    new = pick_cloud(cloud, P)
    assert old.noise_surface.max() > 2.0 * BG["floor"]
    assert new.noise_surface.max() == pytest.approx(BG["floor"], rel=0.2)
    feats, _ = build_features(cloud, new)
    f = _near(feats, weak.mz, weak.rt)
    assert len(f) == 1 and f.iloc[0]["snr"] > 6


def test_sideband_mask():
    # one scan: a trace point at 700, a weak centroid 60 ppm away (sideband), one 200 ppm away (not),
    # and a weak centroid 60 ppm from a point that is only 5x stronger (not)
    from peak3d.io import from_scans
    mzs = [np.array([500.0, 500.0 * (1 + 60e-6), 700.0, 700.0 * (1 + 60e-6), 700.0 * (1 + 200e-6)])]
    ints = [np.array([5e3, 1e3, 1e6, 1e4, 1e4], np.float32)]
    c = from_scans("t", [0.0], mzs, ints)
    o = np.argsort(c.mz)
    strong = np.zeros(c.n_points, bool)
    strong[o[[0, 2]]] = True          # 500.0 and 700.0 belong to real traces
    cand = ~strong
    m = K.sideband_mask(c.off, c.mz, c.inten, cand, strong, 100.0, 10.0)
    assert list(m[o]) == [False, False, False, True, False]


def test_snr_above_trace_floor():
    # a clean peak on an empty baseline: the S/N subtracts nothing (the trace drops out), so it is
    # height / noise surface, not height minus the level where the bounds walk stopped
    cloud, truth, b, feats = _pick([Peak(400.0, 2.0, 3e3, FW)] + WIDTH_SETTERS)
    f = _near(feats, 400.0, 2.0).iloc[0]
    assert f["baseline_lo"] == 0.0
    assert f["snr"] == pytest.approx(f["height"] / f["noise"], rel=1e-6)


def test_interfering_ion_linked_as_split_partner_does_not_shift_mz():
    # a weak co-eluting ion 10 ppm below a strong one lies within the same-scan split tolerance when weak
    # centroids scatter widely (sig_b 150): it is linked into the strong ion's basin as a split partner, where
    # both pieces share one effective intensity. The scan's representative must stay the dominant piece, so
    # the feature's m/z and m/z sd belong to the strong ion (seen on yeast: 208 credentialed peaks rejected
    # by the m/z-sd gate because a strand 12-22 ppm away stood for their scans).
    strong = Peak(400.0, 2.0, 1e6, FW)
    weak = Peak(400.0 * (1 - 10e-6), 2.0, 8e3, FW)
    cloud, truth, b, feats = _pick([strong, weak], merge_ppm=0.0, sig_b=150.0)
    f = feats[(np.abs(feats["mz"] - 400.0) / 400.0 * 1e6 <= 20) & (np.abs(feats["rt"] - 2.0) <= 0.01)]
    assert len(f) >= 1
    main = f.iloc[int(np.argmax(f["height"].values))]
    assert abs(main.mz - 400.0) / 400.0 * 1e6 <= 1.0
    assert main.mz_sd_ppm <= 1.0
    assert main.height == pytest.approx(1e6, rel=0.02)
    assert (main.mz_max - main.mz_min) / 400.0 * 1e6 <= 5.0     # the span of the representatives, not of every point
