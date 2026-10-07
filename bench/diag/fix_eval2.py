# diagnostic: the strand-fusion fix (rep_raw: a scan's representative point is the dominant raw centroid; the
# same-scan split factor same_scan_k) and the far-level ceiling, per file on the truth runs, before any benchmark
# rerun. "base" must reproduce the iteration-2 benchmark tables (rep_raw off, same_scan_k 2, far_level_max 0.3).
# Per run and variant: features per file, same-ion split pairs (two features within 5 ppm whose RT ranges overlap),
# any-file and per-file truth recall (10 ppm / 0.1 min on the corrected RT of the benchmark's warps), median |m/z
# error| of the matched truth, pick + features seconds; IDSL003: TP / FP / F1 on the cleaned labels (held-out halves).
# usage: fix_eval2.py RUN [RUN ...]
import os
import sys
import time
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/diag"))
from bench.runs import RUNS  # noqa: E402
from bench.score import IDSL_TRUTH  # noqa: E402
from fix_eval import eval_truth, near_any  # noqa: E402
from peak3d.estimate import estimate  # noqa: E402
from peak3d.features import build_features  # noqa: E402
from peak3d.io import load_cloud  # noqa: E402
from peak3d.pick import pick_cloud  # noqa: E402

VARIANTS = {"base": dict(rep_raw=False, same_scan_k=2.0, same_scan_floor_ppm=0.0, far_level_max=0.3),
            "rep_f5": dict(rep_raw=True, same_scan_k=2.0, same_scan_floor_ppm=0.0, far_level_max=0.5),
            "rep_ss1_f5": dict(rep_raw=True, same_scan_k=1.0, same_scan_floor_ppm=0.0, far_level_max=0.5),
            "rep_ss1fl": dict(rep_raw=True, same_scan_k=1.0, same_scan_floor_ppm=5.0, far_level_max=0.3),
            "rep_ss1fl_f5": dict(rep_raw=True, same_scan_k=1.0, same_scan_floor_ppm=5.0, far_level_max=0.5)}
COLS = ["mz", "rt", "rt_corr", "rt_min", "rt_max", "height", "snr", "n_scans", "fwhm", "score", "mz_sd_ppm", "far_level"]


def split_pairs(F, ppm=5.0):
    o = np.argsort(F["mz"].values)
    mz, lo, hi = F["mz"].values[o], F["rt_min"].values[o], F["rt_max"].values[o]
    n = 0
    for i in range(len(mz)):
        j = i + 1
        while j < len(mz) and (mz[j] - mz[i]) / mz[i] * 1e6 <= ppm:
            if lo[j] <= hi[i] and lo[i] <= hi[j]:
                n += 1
            j += 1
    return n


def one_file(job):
    run, fi = job
    files = RUNS[run]["files"]
    stem = Path(files[fi]).stem
    d = ROOT / "results/peak3d" / run / "peak3d_out"
    w = pd.read_csv(d / "rt_correction" / f"{stem}.tsv", sep="\t")
    n_bench = len(pd.read_csv(d / "features" / f"{stem}.tsv", sep="\t", usecols=["mz"]))
    cloud = load_cloud(files[fi])
    P0 = estimate(cloud)
    out, info = {}, {}
    for name, kw in VARIANTS.items():
        P = replace(P0, **kw)
        t0 = time.perf_counter()
        b = pick_cloud(cloud, P)
        t1 = time.perf_counter()
        F, _ = build_features(cloud, b)
        t2 = time.perf_counter()
        F["rt_corr"] = np.interp(F["rt"].values, w["rt_native"].values, w["rt_corr"].values)
        out[name] = F[COLS]
        info[name] = dict(pick_s=t1 - t0, feat_s=t2 - t1, splits=split_pairs(F))
    return run, stem, out, info, n_bench


def mz_err(T, F, ppm=10.0, rt_tol=0.1):
    o = np.argsort(F["mz"].values)
    smz, srt = F["mz"].values[o], F["rt_corr"].values[o]
    errs = []
    for m, r in zip(T["mz"].values, T["rt"].values):
        a, b = np.searchsorted(smz, m * (1 - ppm * 1e-6)), np.searchsorted(smz, m * (1 + ppm * 1e-6), side="right")
        k = np.flatnonzero(np.abs(srt[a:b] - r) <= rt_tol)
        if len(k):
            errs.append(np.min(np.abs(smz[a:b][k] - m)) / m * 1e6)
    return float(np.median(errs)) if errs else np.nan


def main(runs):
    jobs = [(r, k) for r in runs for k in range(len(RUNS[r]["files"]))]
    from peak3d.pipeline import _warm_jit
    _warm_jit()
    with get_context("spawn").Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))) as pool:
        res = pool.map(one_file, jobs, chunksize=1)
    lab = pd.read_csv(IDSL_TRUTH)
    is_tp = (lab["Manual Curation"] == "TP").values
    half = np.random.default_rng(0).random(len(lab)) < 0.5
    pd.set_option("display.width", 250)
    for run in runs:
        R = [(s, o, i, nb) for r, s, o, i, nb in res if r == run]
        print(f"\n================ {run}: {len(R)} files")
        for s, o, _, nb in R:
            print(f"  regression {s}: base {len(o['base'])} features vs benchmark table {nb}{'' if len(o['base']) == nb else '   <-- DIFFERS'}")
        T = eval_truth(run)
        rows = []
        for name in VARIANTS:
            n = sum(len(o[name]) for _, o, _, _ in R)
            row = dict(variant=name, features=n, per_file=round(n / len(R)), splits=sum(i[name]["splits"] for _, _, i, _ in R),
                       pick_s=round(float(np.median([i[name]["pick_s"] for _, _, i, _ in R])), 2),
                       feat_s=round(float(np.median([i[name]["feat_s"] for _, _, i, _ in R])), 2))
            if T is not None and "tp" not in T:
                rec = np.array([near_any(T["mz"].values, T["rt"].values, o[name]["mz"].values, o[name]["rt_corr"].values, 10, 0.1) for _, o, _, _ in R])
                row.update(truth_any_file=int(rec.any(axis=0).sum()), n_truth=len(T), per_file_recall=round(float(rec.mean()), 4),
                           mz_err_ppm=round(float(np.nanmedian([mz_err(T, o[name]) for _, o, _, _ in R])), 3))
            if run == "IDSL003":
                F = R[0][1][name]
                det = near_any(lab["m/z"].values, lab["RT(min)"].values, F["mz"].values, F["rt"].values, 10, 0.1)
                det20 = near_any(lab["m/z"].values, lab["RT(min)"].values, F["mz"].values, F["rt"].values, 20, 0.1)
                for hn, h in (("A", half), ("B", ~half), ("all", np.ones(len(lab), bool))):
                    tp = int((det & is_tp & h).sum()); fp = int((det & ~is_tp & h).sum()); fn = int((~det & is_tp & h).sum())
                    row[f"F1_{hn}"] = round(2 * tp / (2 * tp + fp + fn), 4)
                    if hn == "all":
                        row.update(TP=tp, FP=fp, precision=round(tp / max(tp + fp, 1), 3), recall=round(tp / (tp + fn), 3))
                tp = int((det20 & is_tp).sum()); fp = int((det20 & ~is_tp).sum()); fn = int((~det20 & is_tp).sum())
                row["F1_20ppm"] = round(2 * tp / (2 * tp + fp + fn), 4)
            rows.append(row)
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
