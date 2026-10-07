# diagnostic: the new picker (ridge-window cap, localisation + unimodality score terms, intensity-dependent
# noise model) against the Oct 5 behaviour on truth runs and at IDSL003 label level, per-file, before any
# benchmark rerun. Variants are Params overrides; "old" must reproduce the Oct 5 benchmark features.
# usage: newpicker_eval.py RUN [RUN ...]
import json, os, sys
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench/diag"))
from bench.runs import RUNS
from fix_eval import eval_truth, near_any
from peak3d.estimate import estimate
from peak3d.features import build_features
from peak3d.io import load_cloud
from peak3d.pick import pick_cloud

VARIANTS = {"iter1": (dict(), 0.5),                                   # the iteration-1 picker (benchmark now)
            "ff30": (dict(far_frac=0.3), 0.5),                         # tail bumps need a 30 % valley to stay separate
            "ff35": (dict(far_frac=0.35), 0.5),
            "fl30": (dict(far_level_max=0.3), 0.5),                    # bump on a tail / ridge: cleaner side still > 30 % of apex
            "ff30_fl30": (dict(far_frac=0.3, far_level_max=0.3), 0.5),
            "ff30_fl40": (dict(far_frac=0.3, far_level_max=0.4), 0.5)}
CASES = [55, 401, 124, 136, 385, 362, 163, 344, 102, 240, 6, 197, 273, 61, 155, 368, 379, 382]


def one_file(job):
    run, fi = job
    files = RUNS[run]["files"]; stem = Path(files[fi]).stem
    d = ROOT / "results/peak3d" / run / "peak3d_out"
    w = pd.read_csv(d / "rt_correction" / f"{stem}.tsv", sep="\t")
    cloud = load_cloud(files[fi])
    P0 = estimate(cloud)
    out = {}
    for name, (kw, ms) in VARIANTS.items():
        P = replace(P0, **kw)
        b = pick_cloud(cloud, P)
        F, _ = build_features(cloud, b, min_score=ms)
        F["rt_corr"] = np.interp(F["rt"].values, w["rt_native"].values, w["rt_corr"].values)
        out[name] = F[["mz", "rt", "rt_corr", "height", "snr", "n_scans", "fwhm", "score", "n_valleys", "tv_ratio", "far_level"]]
    noise = dict(noise_c=P0.noise_c, noise_r=P0.noise_r, noise_s0=P0.noise_s0, n_noise_pts=P0.n_noise_pts, n_noise_bins=P0.n_noise_bins,
                 floor=P0.floor, fwhm_scans=P0.fwhm_scans)
    return run, stem, out, noise


def main(runs):
    jobs = [(r, k) for r in runs for k in range(len(RUNS[r]["files"]))]
    from peak3d.pipeline import _warm_jit
    _warm_jit()
    with get_context("spawn").Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))) as pool:
        res = pool.map(one_file, jobs, chunksize=1)
    lab = pd.read_csv(ROOT / "data/raw/IDSL_IPA/idslipa_benchmarking_dataset.csv", encoding="utf-8-sig")
    is_tp = (lab["Manual Curation"] == "TP").values
    half = np.random.default_rng(0).random(len(lab)) < 0.5
    for run in runs:
        R = [(s, o, nz) for r, s, o, nz in res if r == run]
        print(f"\n================ {run}: {len(R)} files")
        print("  noise model per file: " + "; ".join(f"{s}: c {nz['noise_c']:.3g} r {nz['noise_r']:.3g} s0 {nz['noise_s0']:.3g} "
                                                        f"({nz['n_noise_pts']} pts, {nz['n_noise_bins']} bins; floor {nz['floor']:.3g})" for s, _, nz in R[:4]))
        # old must reproduce the benchmark output
        for s, o, _ in R[:1]:
            d = ROOT / "results/peak3d" / run / "peak3d_out" / "features" / f"{s}.tsv"
            n_bench = len(pd.read_csv(d, sep="\t", usecols=["mz"]))
            first = next(iter(VARIANTS))
            print(f"  regression check {s}: {first} variant {len(o[first])} features vs benchmark table {n_bench}")
        T = eval_truth(run)
        rows = []
        for name in VARIANTS:
            n = sum(len(o[name]) for _, o, _ in R)
            row = dict(variant=name, features=n, per_file=round(n / len(R)))
            if T is not None and "tp" not in T:
                rec = np.array([near_any(T["mz"].values, T["rt"].values, o[name]["mz"].values, o[name]["rt_corr"].values, 10, 0.1) for _, o, _ in R])
                row.update(truth_any_file=int(rec.any(axis=0).sum()), n_truth=len(T), per_file_recall=round(float(rec.mean()), 4))
            if run == "IDSL003":
                F = R[0][1][name]
                det = near_any(lab["m/z"].values, lab["RT(min)"].values, F["mz"].values, F["rt"].values, 10, 0.1)
                for hn, h in (("A", half), ("B", ~half), ("all", np.ones(len(lab), bool))):
                    tp = int((det & is_tp & h).sum()); fp = int((det & ~is_tp & h).sum()); fn = int((~det & is_tp & h).sum())
                    row[f"F1_{hn}"] = round(2 * tp / (2 * tp + fp + fn), 4)
                    if hn == "all":
                        row.update(TP=tp, FP=fp, precision=round(tp / max(tp + fp, 1), 3), recall=round(tp / (tp + fn), 3))
            rows.append(row)
        pd.set_option("display.width", 250)
        print(pd.DataFrame(rows).to_string(index=False))
        if run == "HZV029_cert":
            hzv = pd.read_csv(next((ROOT / "data/raw/ASARI_DATA/x").glob("*/data/hzv029_manual_certified.txt")), sep="\t")
            print("  the 18 review cases (files of 3 with a feature within 10 ppm / 0.1 min):")
            print("  case    " + " ".join(f"{i:>4d}" for i in CASES))
            for name in VARIANTS:
                line = []
                for i in CASES:
                    mz, rt = hzv["moverz"].values[i], hzv["RT_minutes"].values[i]
                    line.append(sum(bool(near_any(np.array([mz]), np.array([rt]), o[name]["mz"].values, o[name]["rt_corr"].values, 10, 0.1)[0]) for _, o, _ in R))
                print(f"  {name:9s} " + " ".join(f"{v:>4d}" for v in line))


if __name__ == "__main__":
    main(sys.argv[1:])
