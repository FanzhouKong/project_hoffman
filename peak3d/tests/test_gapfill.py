import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from peak3d import pipeline
from peak3d.features import write_features
from peak3d.gapfill import fill_file
from peak3d.synth import Peak, _trapezoid, make_cloud, write_mzml
from peak3d.warp import TableWarp

ROOT = Path(__file__).resolve().parents[2]


def test_fill_window_is_inverse_warped_and_zero_when_absent():
    # the file's native RT runs 0.1 min late: a consensus group at 2.0 sits at native 2.1
    cloud, _ = make_cloud([Peak(400.0, 2.1, 1e5, 0.05)], n_scans=800, dt=0.005, thresh=0.0)
    warp = TableWarp.global_shift(cloud.rt, -0.1)   # native -> corrected
    h, a, _ = fill_file(cloud, warp, np.array([400.0, 400.0, 500.0]), np.array([2.0, 2.6, 2.0]),
                        np.array([0.03, 0.03, 0.03]), ppm_tol=5.0)
    assert h[0] == np.float32(1e5) and a[0] > 0
    assert h[1] == 0 and a[1] == 0           # right m/z, wrong time
    assert h[2] == 0 and a[2] == 0           # nothing at that m/z


def test_fill_area_is_trapezoid_over_present_scans():
    cloud, truth = make_cloud([Peak(400.0, 2.0, 1e5, 0.05)], n_scans=800, dt=0.005, thresh=0.0)
    h, a, _ = fill_file(cloud, TableWarp.identity(cloud.rt), np.array([400.0]), np.array([2.0]), np.array([0.2]), 5.0)
    e = cloud.eic(400.0, 5.0)
    keep = e > 0
    assert abs(a[0] - _trapezoid(e[keep], cloud.rt[keep])) / a[0] < 1e-3


def _three_files(tmp_path, base, mz_of):
    d = tmp_path / "data"
    d.mkdir()
    for k, dt_rt in enumerate((0.0, 0.02, -0.02)):
        peaks = [Peak(mz_of(k, m, r), r + dt_rt, h, 0.05, iso=(0.15,)) for m, r, h in base]
        cloud, _ = make_cloud(peaks, n_scans=600, dt=0.005, thresh=200.0, floor=100.0, n_noise=15000, seed=k,
                              name=f"S{k}")
        write_mzml(cloud, d / f"S{k}.mzML")
    return d


def test_file_whose_calibration_jumps_is_aligned_and_gap_filled_with_mz_dynamic(tmp_path):
    # the third file reads 12 ppm low from 1.5 min on (a calibration jump, as in 23 PR002464 ST004502 files):
    # with --mz-dynamic its offset is a curve with that step, its late compounds group with the others, and six
    # late compounds taken out of its list are found again by gap-filling at its own m/z at their time
    rng = np.random.default_rng(2)
    base = list(zip(rng.uniform(150, 900, 80), rng.uniform(0.4, 2.6, 80), 10 ** rng.uniform(5, 6.5, 80)))
    d = _three_files(tmp_path, base, lambda k, m, r: m * (1 - 12e-6) if k == 2 and r > 1.5 else m)
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "peak3d", "process", "--input", str(d), "--output", str(out),
                        "--no-report", "--keep-cache", "--mz-dynamic"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    c = pd.read_csv(out / "rt_correction" / "S2.tsv", sep="\t")
    a = pd.read_csv(out / "rt_correction" / "S0.tsv", sep="\t")
    step = np.interp([1.0, 2.2], c["rt_native"], c["mz_offset_ppm"]) - np.interp([1.0, 2.2], a["rt_native"],
                                                                                   a["mz_offset_ppm"])
    assert abs(step[0]) < 1.0 and abs(step[1] + 12.0) < 1.0, step
    t = pd.read_csv(out / "aligned_feature_table.tsv", sep="\t")
    one = [int(((t["mz"] - m).abs() <= m * 15e-6).sum()) for m, _, _ in base]
    assert one.count(1) >= 75, one
    late = [(m, rt) for m, rt, _ in base if rt > 1.7][:6]
    f = pd.read_csv(out / "features" / "S2.tsv", sep="\t")
    drop = np.zeros(len(f), bool)
    for m, _ in late:
        drop |= ((f["mz"] - m * (1 - 12e-6)).abs() <= m * 5e-6).to_numpy()
    assert drop.sum() == len(late)
    write_features(f[~drop].reset_index(drop=True), out / "features" / "S2.tsv")
    args = Namespace(ppm=5.0, rt_tol_max=0.25, holdout=False, no_rt_correction=False, no_gap_fill=False, cores=1,
                     min_presence=2, quant="height", mz_dynamic=True)
    pipeline.align_and_group(out, out / ".cache", args)
    p = pd.read_csv(out / "presence_mask.tsv", sep="\t")
    for m, _ in late:
        row = p[(p["mz"] - m).abs() <= m * 5e-6]
        assert len(row) == 1 and row["S2"].iloc[0] == 1, row


def test_shifted_file_is_aligned_and_gap_filled_at_its_own_mz(tmp_path):
    # three files, the third read 8 ppm high: beyond the 5 ppm anchor and gap-fill tolerance. The alignment must
    # find the offset (coarse search) and group the file with the others; six compounds are then taken out of its
    # feature list, so gap-filling must find them in its raw data, which it only does at the file's own m/z
    rng = np.random.default_rng(1)
    base = list(zip(rng.uniform(150, 900, 80), rng.uniform(0.4, 2.6, 80), 10 ** rng.uniform(5, 6.5, 80)))
    d = tmp_path / "data"
    d.mkdir()
    for k, (dt_rt, ppm) in enumerate([(0.0, 0.0), (0.02, 0.0), (-0.02, 8.0)]):
        peaks = [Peak(m * (1 + ppm * 1e-6), r + dt_rt, h, 0.05, iso=(0.15,)) for m, r, h in base]
        cloud, _ = make_cloud(peaks, n_scans=600, dt=0.005, thresh=200.0, floor=100.0, n_noise=15000, seed=k,
                              name=f"S{k}")
        write_mzml(cloud, d / f"S{k}.mzML")
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "peak3d", "process", "--input", str(d), "--output", str(out),
                        "--no-report", "--keep-cache"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    s = pd.read_csv(out / "rt_correction" / "summary.tsv", sep="\t").set_index("file")
    assert abs(s.loc["S2", "mz_offset_ppm"] - s.loc[["S0", "S1"], "mz_offset_ppm"].mean() - 8.0) < 0.5, s
    t = pd.read_csv(out / "aligned_feature_table.tsv", sep="\t")
    one = [int(((t["mz"] - m).abs() <= m * 15e-6).sum()) for m, _, _ in base]
    assert one.count(1) >= 75, one                       # one group per compound, not one per m/z scale
    f = pd.read_csv(out / "features" / "S2.tsv", sep="\t")
    drop = np.zeros(len(f), bool)
    for m, _, _ in base[:6]:
        drop |= ((f["mz"] - m * (1 + 8e-6)).abs() <= m * 5e-6).to_numpy()
    assert drop.sum() == 6
    write_features(f[~drop].reset_index(drop=True), out / "features" / "S2.tsv")
    args = Namespace(ppm=5.0, rt_tol_max=0.25, holdout=False, no_rt_correction=False, no_gap_fill=False, cores=1,
                     min_presence=2, quant="height")
    pipeline.align_and_group(out, out / ".cache", args)
    p = pd.read_csv(out / "presence_mask.tsv", sep="\t")
    for m, _, _ in base[:6]:
        row = p[(p["mz"] - m).abs() <= m * 5e-6]
        assert len(row) == 1 and row["S2"].iloc[0] == 1, row      # 1 = a peak confirmed by gap-filling
