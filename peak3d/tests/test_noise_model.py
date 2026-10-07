"""Intensity-dependent noise model, unimodality / valley descriptors, ridge-window cap, score terms."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from peak3d import kernels as K
from peak3d.estimate import Params, estimate
from peak3d.features import basins_to_frame, build_features, gate, ridge_ratio, rule_score
from peak3d.io import from_scans
from peak3d.pick import pick_cloud
from peak3d.synth import Peak, make_cloud

DT = 0.005
FW = 0.05           # 10 scans per FWHM
BG = dict(n_scans=800, dt=DT, thresh=200.0, floor=100.0, n_noise=30000, sig_a=0.5, sig_b=30.0)


def test_defaults():
    P = Params()
    assert P.ridge_w_cap == 2.0 and P.score_loc_rip and P.max_valleys == -1 and P.k_valley == 3.0
    assert P.min_score == 0.5 and P.eic_check and P.eic_k_snr == 3.0 and P.far_level_max == 0.5
    assert P.same_scan_k == 1.0 and P.same_scan_floor_ppm == 5.0 and P.rep_raw
    assert P.noise_c == 0.0 and P.noise_r == 0.0
    assert K.noise_at(1e4, 100.0, 0.0, 0.0) == pytest.approx(100.0)
    assert K.noise_at(1e4, 100.0, 50.0, 0.02) == pytest.approx(np.sqrt(100.0 ** 2 + 50.0 * 1e4 + (0.02 * 1e4) ** 2))


def test_unimodal_peak_descriptors():
    cloud, truth = make_cloud([Peak(400.2, 2.0, 1e6, FW), Peak(500.3, 1.0, 5e4, FW)], **BG)
    b = pick_cloud(cloud)
    feats, _ = build_features(cloud, b)
    for mz in (400.2, 500.3):
        f = feats[np.abs(feats["mz"] - mz) < 0.005]
        assert len(f) == 1
        f = f.iloc[0]
        assert f["tv_ratio"] == pytest.approx(1.0, abs=0.05)
        assert f["n_valleys"] == 0 and f["tv_excess"] == pytest.approx(0.0, abs=1e-6)
        assert f["far_level"] == pytest.approx(0.0, abs=0.02)      # nothing out there but sparse floor noise
        assert f["noise_i"] >= f["noise"]
        assert 0.5 <= f["score"] <= 1.0 and "lumpy" not in f["flags"]


def test_noiseless_cloud_fits_no_intensity_noise():
    cloud, _ = make_cloud([Peak(200 + 10 * i, 0.3 + 0.1 * i, 10 ** (4 + 0.04 * i), FW) for i in range(50)], **BG)
    P = estimate(cloud)
    # exact intensities: whatever residual the centroid merging leaves is far below any real noise
    assert P.noise_c < 1.0 and P.noise_r < 1e-3


def test_tailing_peaks_without_noise_fit_no_intensity_noise():
    peaks = [Peak(200 + 10 * i, 0.3 + 0.07 * i, 10 ** (4 + 0.04 * i), FW, tau=0.04) for i in range(50)]
    cloud, _ = make_cloud(peaks, **BG)
    P = estimate(cloud)
    assert P.noise_r < 0.01 and P.noise_c < 5.0


def test_intensity_noise_fit_recovers_shot_noise_term():
    peaks = [Peak(200 + 10 * i, 0.3 + 0.07 * i, 10 ** (4 + 0.04 * i), FW) for i in range(50)]
    cloud, _ = make_cloud(peaks, inoise_c=50.0, **BG)
    P = estimate(cloud)
    assert P.n_noise_bins >= 3
    # within a factor 2 of the injected shot-noise term, and no spurious proportional term
    sd_fit = np.sqrt(P.noise_s0 ** 2 + P.noise_c * 1e5 + (P.noise_r * 1e5) ** 2)
    assert 0.7 * np.sqrt(50.0 * 1e5) <= sd_fit <= 1.4 * np.sqrt(50.0 * 1e5)
    assert P.noise_r <= 0.02        # (s0 is diagnostic only: at the low end it trades off against c)


def test_intensity_noise_fit_recovers_proportional_term():
    peaks = [Peak(200 + 10 * i, 0.3 + 0.07 * i, 10 ** (4 + 0.04 * i), FW) for i in range(50)]
    cloud, _ = make_cloud(peaks, inoise_r=0.05, **BG)
    P = estimate(cloud)
    assert 0.03 <= P.noise_r <= 0.08


def _zigzag_cloud():
    """one clean Gaussian plus one flickering background ion whose trace alternates 1e5 / 7e4 over
    20 scans (shallow valleys: the persistence merge joins the lumps into one basin)"""
    cloud, _ = make_cloud([Peak(400.2, 2.0, 1e6, FW)], **BG)
    rt = cloud.rt
    mzs, ins = [], []
    for s in range(cloud.n_scans):
        a, b = cloud.off[s], cloud.off[s + 1]
        m, i = list(cloud.mz[a:b]), list(cloud.inten[a:b])
        if 300 <= s < 320:
            m.append(600.5 * (1 + 1e-7 * ((s % 3) - 1)))
            i.append(1e5 if s % 2 == 0 else 7e4)
        mzs.append(np.asarray(m)); ins.append(np.asarray(i))
    return from_scans("zigzag", rt, mzs, ins, "pos")


def test_zigzag_trace_counts_valleys_and_is_gated_when_asked():
    cloud = _zigzag_cloud()
    b = pick_cloud(cloud)
    df = basins_to_frame(b)
    z = df[np.abs(df["mz"] - 600.5) < 0.01].sort_values("height", ascending=False).iloc[0]
    assert z["n_scans"] >= 15                      # the lumps were merged into one basin
    assert z["n_valleys"] >= 5 and z["tv_ratio"] > 3.0 and z["tv_excess"] > 3.0
    g = df[np.abs(df["mz"] - 400.2) < 0.005].iloc[0]
    assert g["n_valleys"] == 0
    P = b.params
    for name, vals in ridge_ratio(df, cloud, P).items():
        df[name] = vals
    ok_off = gate(df, replace(P, max_valleys=-1))
    ok_on = gate(df, replace(P, max_valleys=1))
    assert ok_off[g.name] and ok_on[g.name]
    assert not ok_on[z.name]
    # the unimodality term also pulls the zig-zag's rule score well below the clean peak's
    s = rule_score(df, P)
    assert s[g.name] > 0.8 and s[z.name] < s[g.name] - 0.2


def test_ridge_window_cap_judges_a_long_ridge_beside_itself():
    S = 600
    rt = DT * np.arange(S)
    mzs, ins = [], []
    rng = np.random.default_rng(0)
    for s in range(S):
        m, i = [], []
        if 100 <= s < 300:                          # a box-shaped ion, 200 scans = 20 widths long
            m.append(500.0 * (1 + 2e-7 * rng.standard_normal())); i.append(1e5)
        mzs.append(np.asarray(m)); ins.append(np.asarray(i))
    cloud = from_scans("box", rt, mzs, ins, "pos")
    P = Params(dt=DT, fwhm_med=FW, fwhm_scans=FW / DT, sig_a=0.5, sig_b=30.0)
    df = pd.DataFrame({"fwhm": [8 * FW], "scan_apex": [200], "mz": [500.0], "height": [1e5], "sig_apex": [0.5]})
    capped = ridge_ratio(df, cloud, replace(P, ridge_w_cap=2.0))
    uncapped = ridge_ratio(df, cloud, replace(P, ridge_w_cap=float("inf")))
    assert capped["ridge_ratio"][0] > 0.5 and capped["far_level"][0] == pytest.approx(1.0)
    assert uncapped["ridge_ratio"][0] == 0.0 and uncapped["far_level"][0] == 0.0


def test_old_score_reproducible_with_flag():
    cloud, _ = make_cloud([Peak(400.2, 2.0, 1e6, FW)], **BG)
    b = pick_cloud(cloud)
    df = basins_to_frame(b)
    for name, vals in ridge_ratio(df, cloud, b.params).items():
        df[name] = vals
    s5 = rule_score(df, replace(b.params, score_loc_rip=False))
    s7 = rule_score(df, b.params)
    assert np.all((s5 >= 0) & (s5 <= 1)) and np.all((s7 >= 0) & (s7 <= 1))
    assert not np.allclose(s5, s7)
