"""sigma_sat: the m/z error model's saturation factor (estimate.fit_saturation), on by default"""
from dataclasses import replace

import numpy as np

from peak3d import kernels as K
from peak3d.estimate import SAT_BINS, Params, estimate, fit_saturation, sat_factor, seed_ions, sigma_ppm
from peak3d.features import build_features
from peak3d.pick import pick_cloud, point_cap_factor, point_sigma
from peak3d.synth import Peak, make_cloud

SAT_I, SAT_PPM = 2.0e6, 60.0      # the synthetic detector: above SAT_I a centroid moves by up to SAT_PPM
FWHM = 0.02


def _study(seed=0, saturate=True, sat_ppm=SAT_PPM, sig_a=3.0, sig_b=150.0):
    """300 ordinary peaks (1e3-3e5), 160 strong ones (4e5-8e6) of which those above SAT_I get a shifted top, and one
    known strong peak at m/z 205.0972; 4 scans per FWHM, so the m/z jumps between consecutive scans as it does in
    real QTOF data"""
    rng = np.random.default_rng(seed)
    peaks = [Peak(205.0972, 1.5, 7.0e6, FWHM)]
    peaks += [Peak(float(rng.uniform(100, 900)), float(rng.uniform(0.2, 2.8)), float(10 ** rng.uniform(3.0, 5.5)), FWHM)
              for _ in range(300)]
    peaks += [Peak(float(rng.uniform(100, 900)), float(rng.uniform(0.2, 2.8)), float(10 ** rng.uniform(5.6, 6.9)), FWHM)
              for _ in range(160)]
    kw = dict(sat_I=SAT_I, sat_ppm=sat_ppm) if saturate else {}
    return make_cloud(peaks, n_scans=600, dt=0.005, sig_a=sig_a, sig_b=sig_b, thresh=150.0, floor=200.0, n_noise=20000,
                      seed=seed, **kw)


OFF = {"sigma_sat": False}


def test_on_by_default_and_factor_one_without_bins():
    P = Params()
    assert P.sigma_sat is True and P.sat_f == []            # the factors are per file: none until one is estimated
    I = np.array([10.0, 1e4, 1e7])
    assert np.array_equal(P.sig_ppm(I), np.sqrt(P.sig_a ** 2 + P.sig_b ** 2 / I))
    assert np.array_equal(sat_factor(I, 6.9, []), np.ones(3))
    cloud, _ = _study()
    assert estimate(cloud).sat_f != [] and estimate(cloud, overrides=OFF).sat_f == []     # --set sigma_sat=false: not measured


def test_factor_lookup_matches_the_kernel():
    f = [4.0, 1.0, 2.5, 1.0]
    lg_top = 6.9
    I = np.array([1e8, 10 ** 6.9, 10 ** 6.66, 10 ** 6.64, 10 ** 6.41, 10 ** 6.39, 10 ** 6.1, 10 ** 5.91, 10 ** 5.89, 50.0])
    want = np.array([4.0, 4.0, 4.0, 1.0, 1.0, 2.5, 1.0, 1.0, 1.0, 1.0])     # above the top = the top bin; below the bins = 1
    assert np.array_equal(sat_factor(I, lg_top, f), want)
    P = replace(Params(), sig_a=3.0, sig_b=100.0, sat_lg_top=lg_top, sat_f=f)
    sig = point_sigma(I.astype(np.float32), P)
    assert np.allclose(sig, sigma_ppm(3.0, 100.0, I.astype(np.float32).astype(float), lg_top, f), rtol=1e-5)
    assert np.allclose(sig / K.per_point_sigma(I.astype(np.float32), 3.0, 100.0), want, rtol=1e-5)
    assert SAT_BINS == len(f)


def test_saturated_file_gets_a_factor_only_at_the_top():
    cloud, _ = _study()
    P = estimate(cloud)
    P0 = estimate(cloud, overrides=OFF)
    assert (P.sig_a, P.sig_b, P.tol_max_ppm) == (P0.sig_a, P0.sig_b, P0.tol_max_ppm)    # a, b and the cap are untouched
    assert abs(P.sat_lg_top - np.log10(cloud.inten.max())) < 1e-6
    assert len(P.sat_f) == SAT_BINS and P.sat_f[0] > 3.0             # the top quarter-decade: a shift of ~40 ppm vs sd 3
    assert min(P.sat_f) >= 1.0
    # below the synthetic ceiling the model is right: no factor there, and none anywhere below the top decade
    k_ceiling = int(np.floor((P.sat_lg_top - np.log10(SAT_I)) * SAT_BINS))
    assert all(f == 1.0 for f in P.sat_f[k_ceiling + 1:])
    assert np.array_equal(P.sig_ppm(np.array([1e3, 1e5])), P0.sig_ppm(np.array([1e3, 1e5])))
    d = P.to_dict()
    assert isinstance(d["sat_f"], list) and all(isinstance(x, float) for x in d["sat_f"])


def test_unsaturated_file_gets_no_factor_and_the_same_features():
    cloud, _ = _study(saturate=False)
    P0 = estimate(cloud, overrides=OFF)
    P = estimate(cloud)
    assert P.sat_f == []
    a, _ = build_features(cloud, pick_cloud(cloud, P0))
    b, _ = build_features(cloud, pick_cloud(cloud, P))
    assert a.equals(b)


def test_saturated_peak_is_one_whole_feature_with_the_factor():
    """the strong peak's top sits ~40 ppm from its flanks: without the factor the trace breaks there (the feature
    keeps only part of the peak), with it the peak is one feature over its whole width"""
    cloud, truth = _study()
    t = truth[(truth["peak_id"] == 0) & (truth["iso_n"] == 0)].iloc[0]

    def at_peak(P):
        fe, _ = build_features(cloud, pick_cloud(cloud, P))
        m = (np.abs(fe["mz"] - t["mz"]) / t["mz"] * 1e6 < 80) & (np.abs(fe["rt"] - t["rt"]) < 0.15)
        return fe[m]

    on = at_peak(estimate(cloud))
    off = at_peak(estimate(cloud, overrides=OFF))
    assert len(on) == 1
    assert abs(on["area"].iloc[0] / t["area"] - 1) < 0.1 and abs(on["height"].iloc[0] / t["height"] - 1) < 0.1
    # without it: the top alone (4-5 scans, well under the whole area), the flanks left as rejected pieces
    assert off["area"].max() < 0.8 * t["area"] and off["n_scans"].max() <= on["n_scans"].iloc[0] - 4


def test_no_factor_without_saturation_over_seeds():
    """the measurement compares a centroid with centroids it is not part of, in the top decade and in the reference
    decade alike: a file whose model is right gets no factor (a biased reference gave 1.3 in two bins)"""
    for seed in (11, 12, 13):
        cloud, _ = _study(seed, saturate=False)
        P = estimate(cloud)
        assert P.sat_f == [] and 0.8 < P.sat_base < 1.2


def test_search_cap_follows_the_factor():
    """a shift of 80 ppm at the ceiling: the step from the flank to the top (34 ppm) is beyond the 30 ppm search cap, so
    a larger sd alone leaves the peak in pieces; a centroid with a factor searches factor x the cap (60 ppm at most)
    and the peak is whole"""
    cloud, truth = _study(sat_ppm=80.0)
    t = truth[(truth["peak_id"] == 0) & (truth["iso_n"] == 0)].iloc[0]
    P = estimate(cloud)
    assert P.tol_max_ppm == 30.0 and P.sat_f[0] > 3.0

    def area(P, capf):
        b = pick_cloud(cloud, P) if capf else _pick_plain_cap(cloud, P)
        fe, _ = build_features(cloud, b)
        m = (np.abs(fe["mz"] - t["mz"]) / t["mz"] * 1e6 < 100) & (np.abs(fe["rt"] - t["rt"]) < 0.15)
        return fe.loc[m, "area"].max() / t["area"]

    assert area(P, False) < 0.8
    assert abs(area(P, True) - 1) < 0.1
    capf = point_cap_factor(cloud.inten, P)
    assert capf.max() == 2.0 and capf.min() == 1.0          # factor x 30 ppm, 60 ppm at most; 1 below the top decade
    assert np.all(capf[cloud.inten < 10 ** (P.sat_lg_top - 1)] == 1.0)
    # a file without factors searches as before
    c2, _ = _study(saturate=False)
    P2 = estimate(c2)
    assert P2.sat_f == [] and point_cap_factor(c2.inten, P2) is None
    a, _ = build_features(c2, pick_cloud(c2, P2))
    b, _ = build_features(c2, pick_cloud(c2, estimate(c2, overrides=OFF)))
    assert a.equals(b)


def _pick_plain_cap(cloud, P):
    """pick_cloud with the factors on the sd but the file-wide search cap for every centroid"""
    import peak3d.pick as pick
    keep = pick.point_cap_factor
    pick.point_cap_factor = lambda inten, P: None
    try:
        return pick_cloud(cloud, P)
    finally:
        pick.point_cap_factor = keep


def _seeds(cloud, P):
    seeds, scan_of = seed_ions(cloud, 2000)
    return seeds, scan_of, np.full(len(seeds), max(8.0, 3.0 * P.fwhm_scans))


def test_a_trace_counts_once():
    """seeds that fall on a trace already followed (given twice here; in real files a side peak or an isotope
    neighbour inside a stronger ion's window) add nothing"""
    cloud, _ = _study()
    P = estimate(cloud, overrides=OFF)
    seeds, scan_of, reach = _seeds(cloud, P)
    once = fit_saturation(cloud, seeds, scan_of, reach, P.sig_a, P.sig_b)
    twice = fit_saturation(cloud, np.tile(seeds, 2), np.tile(scan_of, 2), np.tile(reach, 2), P.sig_a, P.sig_b)
    assert once == twice and once[1][0] > 3.0


def test_factor_is_bounded_by_the_measurement_window():
    """a precise instrument (sd 0.5 ppm) whose saturated centroids move by up to 60 ppm would get a factor near 100:
    the factor stops where 3 sd reach the +/- 60 ppm the scatter was measured in"""
    cloud, _ = _study(sig_a=0.5, sig_b=30.0)
    P = estimate(cloud)
    a, b, lg_top, f = P.sig_a, P.sig_b, P.sat_lg_top, P.sat_f
    assert f[0] > 10.0
    for k, fk in enumerate(f):
        assert fk * np.sqrt(a * a + b * b / 10 ** (lg_top - (k + 1) / SAT_BINS)) <= 20.0 + 1e-9
    assert abs(f[0] * np.sqrt(a * a + b * b / 10 ** (lg_top - 1 / SAT_BINS)) - 20.0) < 1e-9      # the top bin is at the bound
    assert P.sig_ppm(np.array([cloud.inten.max()], float))[0] <= 20.0 + 1e-9


def _two_ion_study(seed=0):
    """no saturation; every strong peak has a weaker, three times wider ion 12 ppm below it (resolved or not, the
    strongest centroid near the peak is that ion's wherever the peak itself is weak) - a lipid and the M+2 of its
    neighbour. 300 ordinary peaks besides"""
    rng = np.random.default_rng(seed)
    peaks = []
    for _ in range(160):
        mz, rt, h = float(rng.uniform(300, 900)), float(rng.uniform(0.4, 2.6)), float(10 ** rng.uniform(5.6, 6.9))
        peaks += [Peak(mz, rt, h, 0.05), Peak(mz * (1 - 12e-6), rt, 0.08 * h, 0.15)]
    peaks += [Peak(float(rng.uniform(100, 900)), float(rng.uniform(0.2, 2.8)), float(10 ** rng.uniform(3.0, 5.5)), 0.05)
              for _ in range(300)]
    return make_cloud(peaks, n_scans=600, dt=0.005, sig_a=0.5, sig_b=30.0, thresh=150.0, floor=200.0, n_noise=20000,
                      seed=seed)


def test_a_second_ion_in_the_window_is_not_saturation():
    """the top of each strong peak sits 12 ppm (24 sd) from the median of its trace's calm centroids, which are the
    neighbour's: not a property of the detector. The calm centroids are two populations, so the top is not counted"""
    for seed in (0, 1):
        cloud, _ = _two_ion_study(seed)
        P = estimate(cloud)
        assert P.sat_f == [], (seed, P.sat_f, P.sat_n)
        assert point_cap_factor(cloud.inten, P) is None
