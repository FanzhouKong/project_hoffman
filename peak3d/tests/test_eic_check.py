"""1D chromatogram check of candidates, duplicate removal, cross-file confirmation and the presence rule."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from peak3d import kernels as K
from peak3d.features import basins_to_frame, build_features, eic_check
from peak3d.gapfill import fill_file
from peak3d.pick import pick_cloud
from peak3d.synth import Peak, make_cloud, write_mzml
from peak3d.warp import TableWarp

ROOT = Path(__file__).resolve().parents[2]
DT = 0.005
FW = 0.05
BG = dict(n_scans=800, dt=DT, thresh=200.0, floor=100.0, n_noise=30000, sig_a=0.5, sig_b=30.0)


def test_eic_shape_apex_inside_bounds_vs_tail_slice():
    cloud, _ = make_cloud([Peak(400.2, 2.0, 1e6, FW, tau=0.06)], **BG)
    fs = FW / DT
    kern = K.gaussian_kernel(0.5 * fs)
    sa = int(np.argmin(np.abs(cloud.rt - 2.0)))
    q_mz = np.array([400.2])
    q_tol = q_mz * 5e-6
    E = K.eic_shape(cloud.off, cloud.mz, cloud.inten, q_mz, q_tol, np.array([sa]), np.array([sa - 10]), np.array([sa + 10]),
                    int(10 * fs), kern)
    assert E[0, K.ES_APEX] == 1 and E[0, K.ES_PROM] > 0.5e6 and abs(E[0, K.ES_SCAN] - sa) <= 1
    assert E[0, K.ES_NOISE] < 0.01 * 1e6
    lo, hi = sa + int(3 * fs), sa + int(6 * fs)          # a slice of the tail: no maximum of its own
    E2 = K.eic_shape(cloud.off, cloud.mz, cloud.inten, q_mz, q_tol, np.array([lo]), np.array([lo]), np.array([hi]),
                     int(10 * fs), kern)
    assert E2[0, K.ES_APEX] == 0


def test_duplicate_mask_flags_the_weaker_same_ion_on_one_maximum():
    mz = np.array([400.0, 400.0005, 400.2])
    scan = np.array([100.0, 101.0, 100.0])
    h = np.array([1e5, 5e4, 5e4])
    o = np.argsort(mz)
    d = K.duplicate_mask(mz[o], scan[o], h[o], 5.0, 2.0)
    out = np.zeros(3, bool)
    out[o] = d
    assert out.tolist() == [False, True, False]


def test_eic_check_classes():
    cloud, _ = make_cloud([Peak(400.2, 2.0, 1e6, FW, tau=0.06)], **BG)
    b = pick_cloud(cloud)
    P = b.params
    df = basins_to_frame(b)
    main = df[np.abs(df["mz"] - 400.2) < 0.005].sort_values("height", ascending=False).iloc[[0]].copy()
    sa = int(main["scan_apex"].iloc[0])
    fs = P.fwhm_scans
    tail = main.copy()
    tail["scan_apex"] = sa + int(3 * fs)
    tail["scan_lo"] = sa + int(3 * fs)
    tail["scan_hi"] = sa + int(6 * fs)
    tail["height"] = 1e4
    dup = main.copy()
    dup["height"] = 0.5 * main["height"].iloc[0]
    cand = pd.concat([main, tail, dup], ignore_index=True)
    E = eic_check(cand, cloud, P)
    assert E["eic_class"].tolist() == ["ok", "noapex", "dup"]
    assert E["eic_snr"].iloc[0] > 3


def test_build_features_keeps_clean_peaks_and_reports_eic_snr():
    cloud, _ = make_cloud([Peak(400.2, 2.0, 1e6, FW), Peak(500.3, 1.0, 5e4, FW)], **BG)
    b = pick_cloud(cloud)
    feats, rej = build_features(cloud, b, keep_rejected=True)
    for mz in (400.2, 500.3):
        f = feats[np.abs(feats["mz"] - mz) < 0.005]
        assert len(f) == 1 and f["eic_snr"].iloc[0] > 3
    assert "reason" in rej.columns


def test_fill_confirms_real_peaks_only():
    # file B runs 0.1 min late: consensus 2.0 is native 2.1. A strong peak, a weak peak (10 % of its group's
    # reference height) and an empty position.
    cloud, _ = make_cloud([Peak(400.0, 2.1, 1e5, FW), Peak(450.0, 1.6, 5e3, FW)], **BG)
    warp = TableWarp.global_shift(cloud.rt, -0.1)
    g_mz = np.array([400.0, 450.0, 700.0])
    g_rt = np.array([2.0, 1.5, 2.0])
    g_half = np.array([0.03, 0.03, 0.03])
    h, a, c = fill_file(cloud, warp, g_mz, g_rt, g_half, 5.0, noise_ref=np.array([100.0, 100.0, 100.0]),
                        h_ref=np.array([1e5, 5e4, 1e5]), fwhm_scans=FW / DT)
    assert h[0] == np.float32(1e5) and c.tolist() == [1, 1, 0]
    # below 5 % of the reference height a bump does not confirm, and without the model nothing is confirmed
    _, _, c2 = fill_file(cloud, warp, g_mz, g_rt, g_half, 5.0, noise_ref=np.array([100.0, 100.0, 100.0]),
                         h_ref=np.array([1e5, 2e5, 1e5]), fwhm_scans=FW / DT)
    assert c2.tolist() == [1, 0, 0]
    _, _, c3 = fill_file(cloud, warp, g_mz, g_rt, g_half, 5.0)
    assert c3.tolist() == [0, 0, 0]


def _run(*argv):
    return subprocess.run([sys.executable, "-m", "peak3d", *argv], cwd=ROOT, capture_output=True, text=True)


def test_presence_rule_end_to_end(tmp_path):
    rng = np.random.default_rng(1)
    base = [(float(m), float(r), float(h)) for m, r, h in
            zip(rng.uniform(150, 900, 40), rng.uniform(0.6, 3.4, 40), 10 ** rng.uniform(4.5, 6.5, 40))]
    extra = Peak(555.555, 2.22, 3e5, FW)                  # only in S0, nothing at all in S1
    weak = Peak(666.666, 1.11, 2.5e3, FW)                 # in both files, but too weak to be picked in S1
    d = tmp_path / "data"
    d.mkdir()
    for k in range(2):
        peaks = [Peak(m, r, h, FW) for m, r, h in base] + ([extra, Peak(666.666, 1.11, 1.2e4, FW)] if k == 0 else [weak])
        cloud, _ = make_cloud(peaks, n_scans=800, dt=DT, thresh=200.0, floor=100.0, n_noise=20000, seed=k, name=f"S{k}")
        write_mzml(cloud, d / f"S{k}.mzML")
    out = tmp_path / "out"
    r = _run("process", "--input", str(d), "--output", str(out), "--cores", "1")
    assert r.returncode == 0, r.stderr + r.stdout
    t = pd.read_csv(out / "aligned_feature_table.tsv", sep="\t")
    pm = pd.read_csv(out / "presence_mask.tsv", sep="\t")

    def near(df, mz):
        return df[np.abs(df["mz"] - mz) / mz * 1e6 <= 5]
    e = near(pm, 555.555)
    assert len(e) == 1 and e["n_detected"].iloc[0] == 1 and e["n_confirmed"].iloc[0] == 0
    assert len(near(t, 555.555)) == 0                      # dropped by the presence rule
    w = near(pm, 666.666)
    assert len(w) == 1 and w["n_detected"].iloc[0] + w["n_confirmed"].iloc[0] >= 2 and len(near(t, 666.666)) == 1
    assert ((t["n_detected"] + t["n_confirmed"]) >= 2).all()
    f0 = pd.read_csv(out / "features/S0.tsv", sep="\t")
    assert "group_kept" in f0.columns and not near(f0, 555.555)["group_kept"].iloc[0]
    out2 = tmp_path / "out2"
    r = _run("process", "--input", str(d), "--output", str(out2), "--cores", "1", "--min-presence", "1")
    assert r.returncode == 0, r.stderr + r.stdout
    assert len(near(pd.read_csv(out2 / "aligned_feature_table.tsv", sep="\t"), 555.555)) == 1
