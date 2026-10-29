"""Noise-law tests for peak / no-peak decisions (merge_test="noise", eic_noise="law" / "law_trace") and the
--set parameter override."""
import json
import subprocess
import sys
from dataclasses import replace
from math import comb
from pathlib import Path

import numpy as np
import pytest

from peak3d import kernels as K
from peak3d.estimate import (Params, _loo_weights, apply_scan_gains, estimate, parse_overrides, refit_after_gains,
                             scan_gains)
from peak3d.features import add_local_width, basins_to_frame, build_features, eic_check, gate
from peak3d.pick import pick_cloud, smooth_gain
from peak3d.synth import Peak, make_cloud, write_mzml

ROOT = Path(__file__).resolve().parents[2]
DT = 0.005
FW = 0.05           # 10 scans per FWHM
SIG = FW / 2.3548200450309493
BG = dict(n_scans=800, dt=DT, thresh=200.0, floor=100.0, n_noise=30000, sig_a=0.5, sig_b=30.0)
# real peaks at other masses set the file's median peak width, as in a real file (m/z 300, 325, ..., 775: test
# peaks avoid those masses)
WIDTH_SETTERS = [Peak(300 + 25 * i, 0.5 + 0.15 * i, 10 ** (5 + 0.1 * i), FW) for i in range(20)]
NOISE = dict(merge_test="noise", eic_noise="law")


def _feats(cloud, P):
    feats, _ = build_features(cloud, pick_cloud(cloud, P))
    return feats


def _at(feats, mz):
    return feats[np.abs(feats["mz"] - mz) < 0.005].sort_values("rt")


def test_defaults_unchanged():
    P = Params()
    assert P.merge_test == "frac" and P.eic_noise == "max" and P.scan_gain     # scan gain on by default since 2026-10-21


def test_smooth_gain_matches_binomial_weights():
    for n in (1, 2, 7, 20):
        w = np.array([comb(2 * n, k) for k in range(2 * n + 1)], float) / 4.0 ** n
        assert smooth_gain(n) == pytest.approx(np.sqrt(np.sum(w * w)), rel=1e-9)


@pytest.mark.parametrize("ratio", [3, 4, 5])
def test_close_unequal_doublet_split_by_noise_test(ratio):
    # resolution 1 (apexes 4 sd = 1.7 FWHM apart), the second peak 1/ratio of the first: the valley sits at
    # 45-54 % of the smaller apex, so the half-valley rule merges ratios 4-5; the noise law splits all three
    big = Peak(500.3, 2.0, 1e6, FW)
    small = Peak(500.3, 2.0 + 4 * SIG, 1e6 / ratio, FW)
    cloud, _ = make_cloud([big, small] + WIDTH_SETTERS, seed=3, **BG)
    P = estimate(cloud)
    f = _at(_feats(cloud, replace(P, **NOISE)), 500.3)
    assert len(f) == 2
    assert f["rt"].values == pytest.approx([big.rt, small.rt], abs=2 * DT)
    if ratio >= 4:
        assert len(_at(_feats(cloud, P), 500.3)) == 1          # the default rule merges it


@pytest.mark.parametrize("mode", ["noise", "noise_raw"])
@pytest.mark.parametrize("seed", range(6))
def test_noisy_single_peak_stays_one(seed, mode):
    # 3 % proportional + shot noise on every point (a realistic Orbitrap-like law, fitted by estimate): the
    # noise test must not cut a single peak at a ripple
    cloud, _ = make_cloud([Peak(412.3, 2.0, 1e6, FW), Peak(463.7, 2.5, 3e4, FW)] + WIDTH_SETTERS, seed=seed,
                          inoise_c=20.0, inoise_r=0.03, **BG)
    P = estimate(cloud)
    assert 0.015 < P.noise_r < 0.05                                # the law is read off the file
    feats = _feats(cloud, replace(P, merge_test=mode, eic_noise="law"))
    assert len(_at(feats, 412.3)) == 1 and len(_at(feats, 463.7)) == 1


def test_narrow_doublet_beside_big_peak_needs_the_raw_dip():
    # two narrow peaks (FWHM 4 scans, the file's median is 10) 9 scans apart with 8 % proportional noise: the raw
    # valley is ~90 % below the smaller apex, but smoothing at half the median width fills it to within the
    # noise (as on the NIST plasma doublets of the MassCube benchmark); the raw-scale test keeps the two
    big = Peak(512.3, 2.0, 1e6, 0.02)
    small = Peak(512.3, 2.045, 4e5, 0.02)
    cloud, _ = make_cloud([big, small] + WIDTH_SETTERS, seed=0, inoise_c=20.0, inoise_r=0.08, **BG)
    P = estimate(cloud)
    assert len(_at(_feats(cloud, P), 512.3)) == 2
    assert len(_at(_feats(cloud, replace(P, merge_test="noise")), 512.3)) == 1
    f = _at(_feats(cloud, replace(P, merge_test="noise_raw")), 512.3)
    assert len(f) == 2 and f["rt"].values == pytest.approx([big.rt, small.rt], abs=2 * DT)


def _crowded(seed, jitter):
    """300 peaks over 4 min (about 7 strong traces per scan), 3 % independent noise, optional per-scan gains"""
    rng = np.random.default_rng(seed)
    peaks = [Peak(float(m), float(r), float(h), FW) for m, r, h in
             zip(rng.uniform(150, 950, 300), rng.uniform(0.3, 3.7, 300), 10 ** rng.uniform(5, 6.5, 300))]
    cloud, _ = make_cloud(peaks, seed=seed, inoise_c=20.0, inoise_r=0.03, **BG)
    lg = rng.normal(0.0, jitter, cloud.n_scans) if jitter else np.zeros(cloud.n_scans)
    cloud.inten *= np.repeat(np.exp(lg), np.diff(cloud.off)).astype(np.float32)
    return cloud, lg


def test_scan_gains_recovered_and_removed():
    cloud, lg = _crowded(0, 0.15)
    P = estimate(cloud)
    g, sd, n = scan_gains(cloud, P)
    assert n > 0.8 * cloud.n_scans and sd == pytest.approx(0.15, rel=0.25)
    used = np.zeros(cloud.n_scans, bool)
    used[2:-2] = True
    est = np.log(g)[used] - np.median(np.log(g)[used])
    true = lg[used] - np.median(lg[used])
    # gain changes slower than ~6 scans look like peak shape to any trace-local fit and are not recovered (about a
    # third of white jitter); the scan-to-scan part, which makes traces zig-zag, is
    assert np.corrcoef(est, true)[0, 1] > 0.7
    hp = lambda x: x - np.convolve(x, np.ones(7) / 7, mode="same")   # noqa: E731
    assert np.corrcoef(hp(est), hp(true))[0, 1] > 0.8
    r_before = P.noise_r
    apply_scan_gains(cloud, g)
    P2 = refit_after_gains(cloud, P)
    assert r_before > 0.1 and P2.noise_r < 0.04            # back to the ions' own 3 %
    assert P2.noise_r == pytest.approx(estimate(cloud).noise_r, abs=0.005)   # the refit agrees with a full estimate
    assert P2.fwhm_scans == P.fwhm_scans and P2.sig_a == P.sig_a


def test_loo_weights_predict_a_quadratic_exactly():
    rt = np.cumsum(np.random.default_rng(0).uniform(0.5, 1.5, 40))         # irregular scan times
    W = _loo_weights(rt)
    y = 3.0 - 2.0 * rt + 0.7 * rt ** 2
    j = np.arange(2, len(rt) - 2)
    pred = np.sum(W[j] * np.stack([y[j - 2], y[j - 1], y[j + 1], y[j + 2]], axis=1), axis=1)
    assert pred == pytest.approx(y[j], rel=1e-9)
    assert np.all(W[:2] == 0) and np.all(W[-2:] == 0)


def test_scan_gains_leave_a_clean_file_alone():
    cloud, _ = _crowded(1, 0.0)
    g, sd, n = scan_gains(cloud, estimate(cloud))
    assert n > 0 and sd < 0.01 and np.max(np.abs(np.log(g))) < 0.05


def test_wide_noisy_peak_stays_one_under_noise_test():
    cloud, _ = make_cloud([Peak(412.3, 4.0, 1e6, 60 * DT)] + WIDTH_SETTERS, n_scans=1600, dt=DT, thresh=200.0,
                          floor=100.0, n_noise=30000, seed=5, inoise_c=20.0, inoise_r=0.04)
    P = estimate(cloud)
    assert len(_at(_feats(cloud, replace(P, **NOISE)), 412.3)) == 1


def test_eic_law_noise_is_the_smoothed_difference_sd():
    cloud, _ = make_cloud([Peak(400.2, 2.0, 1e6, FW), Peak(500.3, 1.0, 5e4, FW)] + WIDTH_SETTERS, seed=1,
                          inoise_c=20.0, inoise_r=0.03, **BG)
    P = estimate(cloud)
    df = basins_to_frame(pick_cloud(cloud, P))
    add_local_width(df, cloud, P)
    df = df[gate(df, P).values].reset_index(drop=True)
    kern = K.gaussian_kernel(P.eic_smooth_fwhm * P.fwhm_scans)
    for mode in ("law", "law_trace"):
        E = eic_check(df, cloud, replace(P, eic_noise=mode))
        floor = df["noise"].values if mode == "law" else E["es_noise"].values
        top, base = E["es_top"].values, E["es_top"].values - E["es_prom"].values

        def var(x):
            return floor ** 2 + P.noise_c * x + (P.noise_r * x) ** 2
        sd = np.sqrt(np.sum(kern ** 2)) * np.sqrt(var(top) + var(np.maximum(base, 0)))
        assert E["eic_snr"].values == pytest.approx(E["es_prom"].values / np.maximum(sd, 1e-9), rel=1e-6)
    # the default reference is unchanged
    E = eic_check(df, cloud, P)
    ref = np.maximum(E["es_noise"].values, df["noise"].values)
    assert E["eic_snr"].values == pytest.approx(E["es_prom"].values / np.maximum(ref, 1e-9), rel=1e-6)


def test_parse_overrides_types_and_errors():
    o = parse_overrides(["merge_test=noise", "eic_noise=law_trace", "k_valley=2.5", "eic_check=false",
                         "max_valleys=2"])
    assert o == dict(merge_test="noise", eic_noise="law_trace", k_valley=2.5, eic_check=False, max_valleys=2)
    for bad in (["nope=1"], ["merge_test"], ["merge_test=maybe"], ["eic_noise=cell"], ["eic_check=yes"]):
        with pytest.raises(ValueError):
            parse_overrides(bad)


def test_estimate_applies_overrides_and_refuses_measured_fields():
    cloud, _ = make_cloud([Peak(400.0, 2.0, 1e6, FW)] + WIDTH_SETTERS, seed=0, **BG)
    P = estimate(cloud, overrides=dict(merge_test="noise", smooth_fwhm=1.0))
    assert P.merge_test == "noise" and P.n_smooth == int(np.clip(round(2 * (P.fwhm_scans / 2.3548) ** 2), 1, 50))
    with pytest.raises(ValueError, match="measured"):
        estimate(cloud, overrides=dict(fwhm_scans=3.0))


def test_cli_set_reaches_the_per_file_params(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    cloud, _ = make_cloud([Peak(400.0, 2.0, 1e6, FW)] + WIDTH_SETTERS, seed=0, name="S0", **BG)
    write_mzml(cloud, d / "S0.mzML")
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "peak3d", "pick", str(d / "S0.mzML"), "--output", str(out),
                        "--set", "merge_test=noise", "--set", "eic_noise=law"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    p = json.loads((out / "features" / "S0.params.json").read_text())
    assert p["merge_test"] == "noise" and p["eic_noise"] == "law" and p["scan_gain"] and "scan_gain" in p["timings_s"]
    r = subprocess.run([sys.executable, "-m", "peak3d", "pick", str(d / "S0.mzML"), "--output", str(out),
                        "--set", "scan_gain=false", "--set", "merge_test=noise_raw"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    p = json.loads((out / "features" / "S0.params.json").read_text())
    assert not p["scan_gain"] and p["merge_test"] == "noise_raw" and "scan_gain" not in p["timings_s"]
    r = subprocess.run([sys.executable, "-m", "peak3d", "pick", str(d / "S0.mzML"), "--output", str(out),
                        "--set", "merge_test=sometimes"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "merge_test" in r.stderr
