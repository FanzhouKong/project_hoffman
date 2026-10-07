import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peak3d.io import load_cloud
from peak3d.synth import Peak, make_cloud, write_mzml

ROOT = Path(__file__).resolve().parents[2]
DT = 0.005


def _study(tmp_path, shifts=(0.0, 0.05, -0.04), n=60, seed=0):
    rng = np.random.default_rng(seed)
    base = [(float(m), float(r), float(h)) for m, r, h in
            zip(rng.uniform(150, 900, n), rng.uniform(0.6, 3.4, n), 10 ** rng.uniform(4.5, 6.5, n))]
    d = tmp_path / "data"
    d.mkdir()
    for k, sh in enumerate(shifts):
        peaks = [Peak(m, r + sh + 0.01 * r * k, h, 0.05, iso=(0.15,)) for m, r, h in base]
        cloud, _ = make_cloud(peaks, n_scans=800, dt=DT, thresh=200.0, floor=100.0, n_noise=20000, seed=k,
                              name=f"S{k}")
        write_mzml(cloud, d / f"S{k}.mzML")
    return d, base


def test_mzml_roundtrip(tmp_path):
    cloud, _ = make_cloud([Peak(300.0, 1.0, 1e5, 0.05)], n_scans=100, dt=DT, floor=50.0, n_noise=300, name="rt")
    p = tmp_path / "rt.mzML"
    write_mzml(cloud, p)
    back = load_cloud(p)
    assert back.n_scans == cloud.n_scans and back.n_points == cloud.n_points
    assert np.allclose(back.mz, cloud.mz) and np.allclose(back.inten, cloud.inten)
    assert np.allclose(back.rt, cloud.rt, atol=1e-6) and back.polarity == "pos"


def _run(*argv):
    return subprocess.run([sys.executable, "-m", "peak3d", *argv], cwd=ROOT, capture_output=True, text=True)


def test_unknown_option_and_bad_value_fail():
    r = _run("process", "--input", "x", "--output", "y", "--bogus")
    assert r.returncode != 0 and "unrecognized" in r.stderr
    r = _run("process", "--input", "x", "--output", "y", "--min-score", "1.5")
    assert r.returncode != 0 and "must be in [0, 1]" in r.stderr


def test_end_to_end_process_and_align(tmp_path):
    d, base = _study(tmp_path)
    out = tmp_path / "out"
    r = _run("process", "--input", str(d), "--output", str(out), "--cores", "2", "--keep-cache", "--holdout")
    assert r.returncode == 0, r.stderr + r.stdout
    for f in ["aligned_feature_table.tsv", "aligned_feature_area.tsv", "filled_mask.tsv", "params.json",
              "features/S0.tsv", "features/S1.tsv", "features/S2.tsv", "features/manifest.tsv",
              "rt_correction/S0.tsv", "rt_correction/summary.tsv", "rt_correction/anchors.tsv",
              "qc/drift_curves.png", "qc/anchor_residuals.png", "qc/presence_hist.png", "qc/holdout.tsv"]:
        assert (out / f).exists(), f
    t = pd.read_csv(out / "aligned_feature_table.tsv", sep="\t")
    assert {"S0", "S1", "S2", "mz", "rt", "n_detected", "iso_offset", "charge"} <= set(t.columns)
    assert (t["iso_offset"] == 1).sum() >= 0.8 * 60      # the injected M+1 partners are labelled, not merged
    # the 60 injected monoisotopic peaks are present in all three files after correction, at the
    # medoid file's native RT (the output axis is the medoid's axis, not the generator's)
    s = pd.read_csv(out / "rt_correction/summary.tsv", sep="\t")
    assert (s["role"] == "medoid").sum() == 1 and len(s) == 3
    k = int(s.loc[s["role"] == "medoid", "file"].iloc[0][1:])
    shifts = (0.0, 0.05, -0.04)
    full = t[t["n_detected"] == 3]
    found = 0
    for m, r, h in base:
        exp_rt = r + shifts[k] + 0.01 * r * k
        hit = full[(np.abs(full["mz"] - m) / m * 1e6 <= 5) & (np.abs(full["rt"] - exp_rt) <= 0.02)]
        found += len(hit) > 0
    assert found >= 0.9 * len(base)
    others = s[s["role"] != "medoid"]
    assert (others["mad_after_s"] < others["mad_before_s"]).all()
    p = json.loads((out / "params.json").read_text())
    assert p["alignment"] == "ok" and "stage_times_s" in p
    # heights are untouched by alignment: features table height equals the matrix entry
    f0 = pd.read_csv(out / "features/S0.tsv", sep="\t")
    g = f0.iloc[0]
    assert t.loc[t["group_id"] == g["group_id"], "S0"].iloc[0] == pytest.approx(g["height"], rel=1e-5)
    # re-run alignment without correction from the cached output
    out2 = tmp_path / "out"
    r = _run("align", "--output", str(out2), "--no-rt-correction", "--no-gap-fill")
    assert r.returncode == 0, r.stderr + r.stdout
    p2 = json.loads((out / "params.json").read_text())
    assert p2["alignment"] == "disabled"
    m = pd.read_csv(out / "filled_mask.tsv", sep="\t")
    t2 = pd.read_csv(out / "aligned_feature_table.tsv", sep="\t")
    assert ((t2[["S0", "S1", "S2"]].values == 0) >= m[["S0", "S1", "S2"]].values.astype(bool)).all()


def test_single_file_run(tmp_path):
    d, base = _study(tmp_path, shifts=(0.0,))
    out = tmp_path / "one"
    r = _run("process", "--input", str(d), "--output", str(out))
    assert r.returncode == 0, r.stderr + r.stdout
    p = json.loads((out / "params.json").read_text())
    assert p["alignment"].startswith("skipped")
    t = pd.read_csv(out / "aligned_feature_table.tsv", sep="\t")
    assert (t["n_detected"] == 1).all() and "S0" in t.columns
    rc = pd.read_csv(out / "rt_correction/S0.tsv", sep="\t")
    assert np.allclose(rc["rt_native"], rc["rt_corr"])
