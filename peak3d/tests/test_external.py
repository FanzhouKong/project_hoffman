"""asari / MassCube kernels: availability, output conversion and the shared finishing steps (no real tool runs)."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peak3d import external
from peak3d.synth import Peak, make_cloud, write_mzml

ROOT = Path(__file__).resolve().parents[2]


def test_convert_asari(tmp_path):
    p = tmp_path / "full_Feature_table.tsv"
    pd.DataFrame({"id_number": ["F0", "F1"], "mz": [150.1, 300.2], "rtime": [60.0, 120.0],
                  "rtime_left_base": [57.0, 114.0], "rtime_right_base": [66.0, 126.0], "parent_masstrack_id": [0, 1],
                  "peak_area": [1, 2], "cSelectivity": [1, 1], "goodness_fitting": [0.9, 0.9], "snr": [10, 10],
                  "detection_counts": [2, 1], "A.mzML": [5.0, 0.0], "B": [7.0, 3.0]}).to_csv(p, sep="\t", index=False)
    t = external.convert_asari(p, ["A", "B", "C"])
    assert list(t.columns) == external.LAYOUT + ["A", "B", "C"]
    assert t["group_id"].tolist() == [0, 1] and t["rt"].tolist() == pytest.approx([1.0, 2.0])
    assert t["rt_min"].tolist() == pytest.approx([0.95, 1.9]) and t["rt_max"].tolist() == pytest.approx([1.1, 2.1])
    assert t["n_detected"].tolist() == [2, 1] and t["C"].tolist() == [0.0, 0.0]


def test_convert_masscube(tmp_path):
    p = tmp_path / "aligned_feature_table.txt"
    pd.DataFrame({"group_ID": [1, 1], "feature_ID": [4, 9], "m/z": [496.34, 497.34], "RT": [1.52, 1.52],
                  "adduct": ["[M+H]+", "[M+2H]2+"], "isotope_state": ["M+0", "M+1"], "is_isotope": [False, True],
                  "peak_shape": ["1.51;7.5e8|1.523;9.3e8|1.541;2.5e8", ""], "MS2": ["", ""],
                  "A": [9e8, 2e8], "B": [8e8, 0]}).to_csv(p, sep="\t", index=False)
    t = external.convert_masscube(p, ["A", "B"])
    assert t["group_id"].tolist() == [4, 9] and t["iso_offset"].tolist() == [0, 1] and t["charge"].tolist() == [1, 2]
    assert t["rt_min"].tolist() == pytest.approx([1.51, 1.47]) and t["rt_max"].tolist() == pytest.approx([1.541, 1.57])
    assert t["n_detected"].tolist() == [2, 1]


def test_kernel_status_reports_missing_interpreters(monkeypatch):
    monkeypatch.setenv("PEAK3D_ASARI_PYTHON", "/nonexistent/python")
    monkeypatch.setenv("PEAK3D_MASSCUBE_PYTHON", sys.executable)            # runs, but has no masscube
    external.kernel_status.cache_clear()
    try:
        st = external.kernel_status()
    finally:
        external.kernel_status.cache_clear()
    assert st["peak3d"]["available"] and st["peak3d"]["variant"] == "GLRS"
    assert not st["asari"]["available"] and "PEAK3D_ASARI_PYTHON" in st["asari"]["note"]
    has_mc = __import__("importlib").util.find_spec("masscube") is not None
    assert st["masscube"]["available"] == has_mc
    if not has_mc:
        assert "cannot import masscube" in st["masscube"]["note"]


def test_external_command_fails_without_the_kernel(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    cloud, _ = make_cloud([Peak(300.0, 0.5, 1e5, 0.05)], n_scans=100, dt=0.01, floor=50.0, n_noise=200, name="x")
    write_mzml(cloud, d / "x.mzML")
    env = dict(os.environ, PEAK3D_ASARI_PYTHON="/nonexistent/python")
    r = subprocess.run([sys.executable, "-m", "peak3d", "external", "--kernel", "asari", "--input", str(d),
                        "--output", str(tmp_path / "out")], cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "not available" in r.stderr


def test_finish_writes_the_viewer_layout(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    peaks = [Peak(300.0, 0.8, 1e6, 0.05), Peak(450.0, 1.5, 5e5, 0.05)]
    m = np.array([60.0, 85.5, 120.25])
    for k in range(2):
        cloud, _ = make_cloud(peaks, n_scans=400, dt=0.005, floor=50.0, n_noise=2000, seed=k, name=f"F{k}")
        write_mzml(cloud, d / f"F{k}.mzML", ms2=[(0.8, 300.0, m, np.array([10.0, 50.0, 5.0 + k]))])
    src = tmp_path / "full_Feature_table.tsv"
    pd.DataFrame({"id_number": ["F1", "F2"], "mz": [300.0, 450.0], "rtime": [48.0, 90.0],
                  "rtime_left_base": [45.0, 87.0], "rtime_right_base": [51.0, 93.0], "detection_counts": [2, 2],
                  "F0": [1e6, 5e5], "F1": [1.1e6, 0.0]}).to_csv(src, sep="\t", index=False)
    files = sorted(d.glob("*.mzML"))
    table = external.convert_asari(src, [f.stem for f in files])
    out = tmp_path / "out"
    out.mkdir()
    args = argparse.Namespace(cmd="external", kernel="asari", input=d, output=out, cores=1, polarity=None,
                              keep_cache=False, no_report=True, library=None, similarity="weighted_entropy",
                              min_similarity=0.7, precursor_ppm=10.0, fragment_tol=0.02)
    external.finish(files, args, table, "asari", "9.9", {"kernel_s": 1.0}, "pos")
    t = pd.read_csv(out / "aligned_feature_table.tsv", sep="\t")
    assert t["group_id"].tolist() == [1, 2] and list(t.columns[-2:]) == ["F0", "F1"]
    man = pd.read_csv(out / "features" / "manifest.tsv", sep="\t")
    assert man["n_features"].tolist() == [2, 1] and man["n_ms2"].tolist() == [1, 1] and man["n_scans"].tolist() == [400, 400]
    mp = pd.read_csv(out / "ms2_map.tsv", sep="\t")
    assert set(mp["group_id"]) == {1} and mp["representative"].sum() == 1
    p = json.loads((out / "params.json").read_text())
    assert p["kernel"] == "asari" and p["kernel_version"] == "9.9" and p["options"]["quant"] == "area"
    assert p["stage_times_s"]["kernel_s"] == 1.0 and "read_s" in p["stage_times_s"] and p["n_files"] == 2
    assert not (out / ".cache").exists() and (out / "ms2" / "F0.npz").exists()
