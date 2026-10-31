"""MS2 spectra: collected in the MS1 pass, mapped to aligned features, end to end through process + identify."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peak3d import ms2
from peak3d.io import (PRECURSOR_MAX_BINS, PRECURSOR_PPM, PRECURSOR_SEARCH_PPM, load_cloud,
                       precursor_calibration, precursor_error)
from peak3d.library import read_msp, write_msp_record
from peak3d.synth import Peak, make_cloud, write_mzml

ROOT = Path(__file__).resolve().parents[2]
DT = 0.005


def _frag(seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(50, 140, 6)).round(4), rng.uniform(5, 100, 6)


# ------------------------------------------------------------------ reading ----
def test_ms2_collected_in_the_ms1_pass(tmp_path):
    cloud, _ = make_cloud([Peak(300.0, 1.0, 1e5, 0.05)], n_scans=200, dt=DT, floor=50.0, n_noise=500, name="m")
    m, i = _frag(1)
    spectra = [(0.5, 300.0012, m, i), (0.7501, 412.2, m + 1, i), (0.9, 500.0, np.zeros(0), np.zeros(0))]
    p = tmp_path / "m.mzML"
    write_mzml(cloud, p, ms2=spectra)
    col = ms2.Collector()
    back = load_cloud(p, None, col)
    plain = load_cloud(p)
    assert back.n_scans == cloud.n_scans and np.allclose(back.mz, plain.mz) and np.allclose(back.rt, plain.rt)
    assert len(col) == 2 and col.n_skipped == 1                          # the empty spectrum is skipped
    sid, rt, prec, pol, p_int, purity, mz, it = col[0]
    assert sid.startswith("scan=") and rt == pytest.approx(0.5, abs=1e-6) and prec == pytest.approx(300.0012)
    assert p_int >= 0 and np.isnan(purity)                              # the test writer gives no window offsets
    assert pol == "pos" and np.allclose(mz, m) and np.allclose(it, i) and it.dtype == np.float32
    s = ms2.from_collected(col, back.polarity)
    assert len(s) == 2 and s.polarity == "pos" and np.diff(s.off).tolist() == [6, 6]
    assert s.tic[0] == pytest.approx(i.sum(), rel=1e-5)
    ms2.save(s, tmp_path / "m.npz")
    r = ms2.load(tmp_path / "m.npz")
    assert r.scan.tolist() == s.scan.tolist() and np.allclose(r.mz, s.mz) and r.polarity == "pos"
    assert np.array_equal(r.prec_int, s.prec_int) and np.isnan(r.purity).all()


@pytest.mark.parametrize("unit", ["second", None])
def test_ms2_time_follows_the_ms1_unit_rule(tmp_path, unit):
    cloud, _ = make_cloud([Peak(300.0, 4.0, 1e5, 0.05)], n_scans=500, dt=0.02, floor=50.0, n_noise=500, name="u")
    m, i = _frag(2)
    p = tmp_path / "u.mzML"
    write_mzml(cloud, p, ms2=[(4.0, 300.0, m, i)], rt_unit=unit)        # 10 min run: 600 s, seconds guard kicks in
    col = ms2.Collector()
    back = load_cloud(p, None, col)
    assert back.rt.max() == pytest.approx(cloud.rt.max(), abs=1e-3)
    assert col[0][1] == pytest.approx(4.0, abs=1e-3)


# ------------------------------------------------------------------ precursor calibration ----
def test_precursor_calibration_follows_the_error_curve():
    rng = np.random.default_rng(5)
    prec = rng.uniform(80, 900, 2000)
    true = 33.0 - 0.02 * (prec - 80)                      # QTOF-like: +33 ppm at m/z 80 falling to +17 ppm at 900
    d = true + rng.normal(0, 3.0, len(prec))
    d[::50] = np.nan                                      # no MS1 peak near the precursor
    cal_mz, cal_ppm, tol, n = precursor_calibration(prec, d)
    assert n == 1960 and len(cal_mz) == PRECURSOR_MAX_BINS and np.all(np.diff(cal_mz) > 0)
    inner = (prec > cal_mz[0]) & (prec < cal_mz[-1])
    assert np.abs(precursor_error(cal_mz, cal_ppm, prec[inner]) - true[inner]).max() < 1.5
    assert tol == pytest.approx(PRECURSOR_PPM)            # 3 sd of 3 ppm is under the 10 ppm floor
    # flat beyond the bins, not extrapolated
    assert precursor_error(cal_mz, cal_ppm, [10.0, 5000.0]).tolist() == [cal_ppm[0], cal_ppm[-1]]
    # a wide spread widens the tolerance, capped at the search window
    assert precursor_calibration(prec, rng.normal(8, 8, len(prec)))[2] == pytest.approx(24, abs=2)
    assert precursor_calibration(prec, rng.normal(0, 40, len(prec)))[2] == PRECURSOR_SEARCH_PPM
    # too few spectra: no curve, the fixed tolerance, no correction
    cal_mz, cal_ppm, tol, n = precursor_calibration(prec[1:20], d[1:20])
    assert n == 19 and len(cal_mz) == 0 and tol == PRECURSOR_PPM
    assert precursor_error(cal_mz, cal_ppm, prec[:3]).tolist() == [0.0, 0.0, 0.0]
    # fewer than two bins' worth: one bin, a constant error
    cal_mz, cal_ppm, _, _ = precursor_calibration(prec[:150], d[:150])
    assert len(cal_mz) == 1 and cal_ppm[0] == pytest.approx(np.nanmedian(d[:150]))


def _offset_file(tmp_path, ppm_off, n_peaks=30, seed=7):
    """an mzML whose MS2 precursors are written ppm_off ppm above their MS1 peaks, one spectrum per peak apex"""
    rng = np.random.default_rng(seed)
    mzs = np.sort(rng.uniform(120, 850, n_peaks))
    rts = rng.uniform(0.4, 2.6, n_peaks)
    peaks = [Peak(float(m), float(r), 3e5, 0.05) for m, r in zip(mzs, rts)]
    cloud, _ = make_cloud(peaks, n_scans=600, dt=DT, floor=50.0, n_noise=2000, seed=seed, name="off")
    frags = [_frag(20 + k) for k in range(n_peaks)]
    spectra = [(float(r) + 0.0001, float(m) * (1 + ppm_off * 1e-6), *f) for m, r, f in zip(mzs, rts, frags)]
    p = tmp_path / "off.mzML"
    write_mzml(cloud, p, ms2=spectra)
    return p, mzs, rts


def test_offset_precursors_found_in_ms1_and_mapped(tmp_path):
    p, mzs, rts = _offset_file(tmp_path, 26.0)
    col = ms2.Collector()
    load_cloud(p, None, col)
    assert col.cal_n == 30 and len(col.cal_mz) == 1 and col.cal_ppm[0] == pytest.approx(26.0, abs=1.5)
    assert all(r[4] > 0 for r in col)                     # every precursor found in the MS1 scan before it
    assert np.allclose(sorted(r[2] for r in col), mzs * (1 + 26e-6), rtol=1e-8)    # stored as the file gives it
    s = ms2.from_collected(col, "pos")
    ms2.save(s, tmp_path / "off.npz")
    back = ms2.load(tmp_path / "off.npz")
    assert back.cal_n == 30 and back.tol_ppm == col.tol_ppm and np.allclose(back.cal_ppm, col.cal_ppm)
    assert np.allclose(np.sort(back.precursor_on_ms1()), mzs, rtol=3e-6)
    out = tmp_path / "out"
    (out / "ms2").mkdir(parents=True)
    pd.DataFrame({"group_id": np.arange(30), "mz": mzs, "rt": rts, "rt_min": rts - 0.05, "rt_max": rts + 0.05,
                  "off": 1e5}).to_csv(out / "aligned_feature_table.tsv", sep="\t", index=False)
    ms2.save(s, out / "ms2" / "off.npz")
    t = ms2.map_to_groups(out)
    assert sorted(t["group_id"]) == list(range(30)) and (t["precursor_intensity"] > 0).all()
    q = pd.read_csv(out / "qc" / "ms2_precursor.tsv", sep="\t")
    assert q["mapped"].tolist() == [30] and q["error_ppm_median"][0] == pytest.approx(26.0, abs=1.5)
    # the same spectra without the calibration (as stored before it existed): 26 ppm is past the 10 ppm tolerance
    ms2.save(ms2.from_collected(list(col), "pos"), out / "ms2" / "off.npz")
    assert len(ms2.map_to_groups(out)) < 5


def test_map_applies_the_alignment_mz_offset(tmp_path):
    out, m, i = _fake_output(tmp_path)
    nan = float("nan")
    # file B's m/z runs 15 ppm above the consensus: its 400 Da precursor sits at 400.006 on its own scale
    b = [("scan=7", 1.9, 400.0 * (1 + 15e-6), "pos", nan, nan, m, i)]      # 1.9 native = 2.0 corrected
    ms2.save(ms2.from_collected(b, "pos"), out / "ms2" / "B.npz")
    assert ("B", "scan=7") not in {(r.file, r.scan) for r in ms2.map_to_groups(out).itertuples()}
    pd.DataFrame({"file": ["A", "B"], "mz_offset_ppm": [nan, 15.0]}).to_csv(out / "rt_correction" / "summary.tsv",
                                                                          sep="\t", index=False)
    t = ms2.map_to_groups(out)
    assert t.loc[(t["file"] == "B") & (t["scan"] == "scan=7"), "group_id"].tolist() == [13]


def test_map_applies_the_offset_at_the_spectrum_time(tmp_path):
    # an m/z offset that varies over the run (--mz-dynamic): 0 early, 15 ppm from 1.5 min on (native time)
    out, m, i = _fake_output(tmp_path)
    nan = float("nan")
    b = [("scan=7", 1.9, 400.0 * (1 + 15e-6), "pos", nan, nan, m, i),     # late: read 15 ppm high
         ("scan=8", 0.91, 300.0005, "pos", nan, nan, m, i)]                # early: on the consensus scale
    ms2.save(ms2.from_collected(b, "pos"), out / "ms2" / "B.npz")
    pd.DataFrame({"rt_native": [0.0, 1.49, 1.5, 3.0], "rt_corr": [0.1, 1.59, 1.6, 3.1],
                  "mz_offset_ppm": [0.0, 0.0, 15.0, 15.0]}).to_csv(out / "rt_correction" / "B.tsv", sep="\t", index=False)
    pd.DataFrame({"file": ["A", "B"], "mz_offset_ppm": [0.0, 7.5]}).to_csv(out / "rt_correction" / "summary.tsv",
                                                                         sep="\t", index=False)
    t = ms2.map_to_groups(out)
    got = dict(zip(t.loc[t["file"] == "B", "scan"], t.loc[t["file"] == "B", "group_id"]))
    assert got.get("scan=7") == 13 and got.get("scan=8") in (11, 12), got


# ------------------------------------------------------------------ mapping ----
def _fake_output(tmp_path):
    out = tmp_path / "out"
    (out / "ms2").mkdir(parents=True)
    pd.DataFrame({"group_id": [11, 12, 13], "mz": [300.0, 300.001, 400.0], "rt": [1.0, 1.02, 2.0],
                  "rt_min": [0.95, 0.99, 1.97], "rt_max": [1.05, 1.06, 2.03],
                  "A": [1e5, 5e5, 1e5], "B": [1e5, 0.0, 1e5]}).to_csv(out / "aligned_feature_table.tsv", sep="\t", index=False)
    m, i = _frag(3)
    big = i * 10
    nan = float("nan")
    a = [("scan=1", 1.01, 300.0005, "pos", nan, nan, m, i), ("scan=2", 2.0, 400.0, "pos", nan, nan, m, big),
         ("scan=3", 2.01, 400.0, "pos", nan, nan, m, i), ("scan=4", 1.0, 500.0, "pos", nan, nan, m, i)]
    b = [("controllerType=0 controllerNumber=1 scan=5", 0.91, 300.0005, "pos", nan, nan, m, i),   # 1.01 corrected
         ("scan=6", 1.9, 400.0, "pos", nan, nan, m, i * 100)]                     # 1.9 native = 2.0 corrected
    ms2.save(ms2.from_collected(a, "pos"), out / "ms2" / "A.npz")
    ms2.save(ms2.from_collected(b, "pos"), out / "ms2" / "B.npz")
    (out / "rt_correction").mkdir()
    pd.DataFrame({"rt_native": [0.0, 3.0], "rt_corr": [0.1, 3.1]}).to_csv(out / "rt_correction" / "B.tsv", sep="\t",
                                                                          index=False)
    return out, m, i


def test_map_to_groups(tmp_path):
    out, m, i = _fake_output(tmp_path)
    t = ms2.map_to_groups(out)
    key = {(r.file, r.scan): r.group_id for r in t.itertuples()}
    assert key[("A", "scan=1")] == 12                     # both 300 groups fit: the one strongest in A wins
    assert key[("B", "controllerType=0 controllerNumber=1 scan=5")] == 11       # in B the 300.001 group is absent
    assert key[("A", "scan=2")] == key[("A", "scan=3")] == 13 and ("A", "scan=4") not in key
    assert key[("B", "scan=6")] == 13                     # found through B's RT correction
    rep = t[t["representative"] == 1].set_index("group_id")
    assert rep.loc[13, "scan"] == "scan=6" and rep.loc[13, "file"] == "B"     # highest TIC of the three
    assert list(pd.read_csv(out / "ms2_map.tsv", sep="\t").columns) == ms2.MAP_COLUMNS
    r = ms2.read_representatives(out)
    assert r["group_id"].tolist() == [11, 12, 13] and r["n_ms2"].tolist() == [1, 1, 3]
    assert r["file"] == ["B", "A", "B"] and r["scan"][0] == "controllerType=0 controllerNumber=1 scan=5"
    assert r["prec_mz"].tolist() == pytest.approx([300.0, 300.001, 400.0])
    assert r["rt_s"][2] == pytest.approx(114.0) and r["ion_mode"] == ["pos"] * 3
    assert np.allclose(r["mz"][r["off"][0]:r["off"][1]], m)
    lib = read_msp(out / "ms2_spectra.msp")
    assert lib.names == ["feature_11", "feature_12", "feature_13"]
    # the consensus copy: all three spectra of feature 13 (same peaks, different scales) merge, from both files
    c = ms2.read_representatives(out, "consensus")
    assert c["group_id"].tolist() == [11, 12, 13] and c["n_merged"].tolist() == [1, 1, 3]
    assert c["n_files"].tolist() == [1, 1, 2]
    assert c["vs_single"][2] == pytest.approx(1.0, abs=1e-6)          # same peaks, other scales: alike
    assert np.allclose(c["mz"][c["off"][2]:c["off"][3]], m, atol=1e-6)
    assert t.loc[t["group_id"] == 13, "in_consensus"].tolist() == [1, 1, 1]
    assert (t["rt_aligned_s"] - t["rt_s"]).abs().max() == pytest.approx(6.0)        # B's correction: +0.1 min


def test_map_without_ms2_removes_stale_files(tmp_path):
    out, _, _ = _fake_output(tmp_path)
    ms2.map_to_groups(out)
    for f in (out / "ms2").iterdir():
        f.unlink()
    assert len(ms2.map_to_groups(out)) == 0
    assert not (out / "ms2_map.tsv").exists() and ms2.read_representatives(out) is None
    assert not (out / "ms2_consensus.msp").exists()


# ------------------------------------------------------------------ end to end ----
def test_process_maps_ms2_and_identifies(tmp_path):
    rng = np.random.default_rng(0)
    base = [(float(m), float(r), float(h)) for m, r, h in
            zip(rng.uniform(150, 900, 25), rng.uniform(0.6, 2.4, 25), 10 ** rng.uniform(5, 6.5, 25))]
    frags = [_frag(10 + k) for k in range(len(base))]
    d = tmp_path / "data"
    d.mkdir()
    for k, sh in enumerate((0.0, 0.03)):
        peaks = [Peak(m, r + sh, h, 0.05, iso=(0.15,)) for m, r, h in base]
        cloud, _ = make_cloud(peaks, n_scans=600, dt=DT, thresh=200.0, floor=100.0, n_noise=15000, seed=k, name=f"S{k}")
        spectra = [(r + sh, m + 2e-4, f[0], f[1] * (1 + 0.05 * k)) for (m, r, _), f in zip(base[:12], frags)]
        write_mzml(cloud, d / f"S{k}.mzML", ms2=spectra)
    lib = tmp_path / "lib.msp"
    with open(lib, "w") as fh:
        for j, ((m, _, _), f) in enumerate(zip(base, frags)):
            write_msp_record(fh, [("NAME", f"compound {j}"), ("PRECURSORMZ", m), ("IONMODE", "Positive")], *f)
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "peak3d", "process", "--input", str(d), "--output", str(out),
                        "--no-report", "--library", str(lib), "--similarity", "entropy"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "[peak3d] ms2:" in r.stdout and "[peak3d] identify:" in r.stdout
    assert (out / "ms2" / "S0.npz").exists() and pd.read_csv(out / "features" / "manifest.tsv", sep="\t")["n_ms2"].tolist() == [12, 12]
    t = pd.read_csv(out / "aligned_feature_table.tsv", sep="\t")
    hits = pd.read_csv(out / "identifications.tsv", sep="\t")
    assert set(hits["spectrum"]) == {"consensus", "single"}
    top = hits[(hits["rank"] == 1) & (hits["spectrum"] == "consensus")].merge(t[["group_id", "mz"]], on="group_id")
    names = {j: f"compound {j}" for j in range(len(base))}
    right = sum(any(abs(row.mz - base[j][0]) < 0.005 and row.name == names[j] for j in range(12)) for row in top.itertuples())
    assert len(top) >= 10 and right == len(top)
    ann = pd.read_csv(out / "feature_annotations.tsv", sep="\t").set_index("group_id")
    assert (ann.loc[top["group_id"], "annotation"] == top["name"].to_numpy()).all()
    assert set(ann.loc[top["group_id"], "source"]) == {"library"}
    info = json.loads((out / "params.json").read_text())
    assert "ms2_s" in info["stage_times_s"] and info["options"]["similarity"] == "entropy"
    # re-search with another score through the identify command
    r = subprocess.run([sys.executable, "-m", "peak3d", "identify", str(out), "--library", str(lib), "--similarity", "dot"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    assert json.loads((out / "identifications.json").read_text())["method"] == "dot"
    r = subprocess.run([sys.executable, "-m", "peak3d", "identify", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "--library" in r.stderr
