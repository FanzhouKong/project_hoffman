# diagnostic: per-file effect of the two picker fixes (far-apart merge rule; local S/N with the
# band-capped noise) on replicate runs, before rerunning the benchmark. Variants:
#   old      far_frac 0, no band cap, old S/N (the benchmark's 2026-10-02 behaviour)
#   merge    + far-apart merge rule
#   snr      + band-capped noise and local S/N (no merge change)
#   both     both fixes (the new defaults)
# Per variant: kept features, truth recall per file, the 18 HZV029 review cases, EIC junk classes,
# replicate reproducibility; and for features a variant adds / removes relative to old: how many
# reproduce in the replicates, are EIC junk, or are reported by another tool.
# usage: fix_eval.py RUN [RUN ...]
import json
import os
import sys
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/diag"))
from bench.runs import LOCAL_TOOLS, RUNS  # noqa: E402
from bench.score import load  # noqa: E402
from junk_classes import classify_file, rep_check, truth_table  # noqa: E402
from peak3d.estimate import estimate  # noqa: E402
from peak3d.features import build_features  # noqa: E402
from peak3d.io import load_cloud  # noqa: E402
from peak3d.pick import pick_cloud  # noqa: E402

OLD = dict(far_frac=0.0, noise_post_merge=False, snr_floor_baseline=False, min_snr=3.0)   # Oct 2 picker
VARIANTS = {"old": OLD,
            "new_s3": dict(),                                   # the 3D noise surface defaults, no ripple test
            "rip3": dict(ripple_k=3.0), "rip4": dict(ripple_k=4.0),
            "rip5": dict(ripple_k=5.0), "rip6": dict(ripple_k=6.0)}


def eval_truth(run):
    """truth for the per-file recall column: the run's curated list, or the credentialed truth
    (data/truth) for the 12C / 13C credentialing runs (13C: the labelled partner's m/z)"""
    T = truth_table(run)
    if T is not None:
        return T
    for ds in ("YEAST", "SZ22"):
        if run.startswith(ds + "_"):
            t = pd.read_csv(ROOT / f"data/truth/{ds}/credentialed.tsv", sep="\t")
            mz = t["mz"].values + (t["n"].values * 1.0033548378 if run.endswith("13C") else 0.0)
            return pd.DataFrame({"mz": mz, "rt": t["rt"].values})
    return None


CASES = [55, 401, 124, 136, 385, 362, 163, 344, 102, 240, 6, 197, 273, 61, 155, 368, 379, 382]


def near_any(mz, rt, ref_mz, ref_rt, ppm, rt_tol):
    o = np.argsort(ref_mz)
    smz, srt = ref_mz[o], ref_rt[o]
    lo = np.searchsorted(smz, mz * (1 - ppm * 1e-6))
    hi = np.searchsorted(smz, mz * (1 + ppm * 1e-6), side="right")
    return np.array([np.any(np.abs(srt[a:b] - r) <= rt_tol) for a, b, r in zip(lo, hi, rt)])


def one_file(job):
    run, fi = job
    files = RUNS[run]["files"]
    d = ROOT / "results/peak3d" / run / "peak3d_out"
    stems = [Path(f).stem for f in files]
    stem = stems[fi]
    warps = {s: pd.read_csv(d / "rt_correction" / f"{s}.tsv", sep="\t") for s in stems}
    cloud = load_cloud(files[fi])
    others = [(load_cloud(files[k]), warps[stems[k]], json.load(open(d / "features" / f"{stems[k]}.params.json")))
              for k in range(len(files)) if k != fi]
    P0 = estimate(cloud)
    w = warps[stem]
    out = {}
    for name, kw in VARIANTS.items():
        P = replace(P0, **kw)
        b = pick_cloud(cloud, P)
        F, _ = build_features(cloud, b)
        F["rt_corr"] = np.interp(F["rt"].values, w["rt_native"].values, w["rt_corr"].values)
        C = classify_file(cloud, F, P.to_dict())
        F["cls"] = C["cls"].values
        F["rep_frac"] = rep_check(F.assign(_top=C["top_s"].values), P.to_dict(), others)
        F["pick_s"] = sum(v for k, v in b.timings.items() if k != "n_edges")
        out[name] = F[["mz", "rt", "rt_corr", "height", "snr", "n_scans", "fwhm", "score", "cls", "rep_frac", "pick_s"]]
    return run, stem, out


def main(runs):
    jobs = [(r, k) for r in runs for k in range(len(RUNS[r]["files"]))]
    from peak3d.pipeline import _warm_jit
    _warm_jit()   # compile (and cache) every kernel once before the workers fork
    with get_context("spawn").Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))) as pool:
        res = pool.map(one_file, jobs, chunksize=1)
    hzv = pd.read_csv(next((ROOT / "data/raw/ASARI_DATA/x").glob("*/data/hzv029_manual_certified.txt")), sep="\t")
    for run in runs:
        R = [(s, o) for r, s, o in res if r == run]
        T = eval_truth(run)
        sup_tabs = [x[0] for x in (load(t, run) for t in ("asari", "masscube", *LOCAL_TOOLS)) if x is not None]
        print(f"\n================ {run}: {len(R)} files")
        rows = []
        for name in VARIANTS:
            n = sum(len(o[name]) for _, o in R)
            rec = np.mean([near_any(T["mz"].values, T["rt"].values, o[name]["mz"].values, o[name]["rt_corr"].values, 10, 0.1).mean()
                           for _, o in R]) if T is not None and "tp" not in T else np.nan
            A = pd.concat([o[name] for _, o in R])
            rows.append(dict(variant=name, features=n, per_file=n / len(R), truth_recall_per_file=round(rec, 4),
                             junk_share=round((A["cls"] != "ok").mean(), 3), replicable=round((A["rep_frac"] >= 0.5).mean(), 3),
                             good=int(((A["rep_frac"] >= 0.5) & (A["cls"] == "ok")).sum()),
                             bad=int(((A["rep_frac"] == 0) | (A["cls"] != "ok")).sum()),
                             pick_s=round(float(np.mean([o[name]["pick_s"].iloc[0] for _, o in R])), 2)))
        print(pd.DataFrame(rows).to_string(index=False))
        # added / removed relative to old
        for name in [v for v in VARIANTS if v != "old"]:
            add, rem = [], []
            for _, o in R:
                a, b = o[name], o["old"]
                add.append(a[~near_any(a["mz"].values, a["rt"].values, b["mz"].values, b["rt"].values, 5, 2 / 60)])
                rem.append(b[~near_any(b["mz"].values, b["rt"].values, a["mz"].values, a["rt"].values, 5, 2 / 60)])
            for lab, X in (("added", pd.concat(add)), ("removed", pd.concat(rem))):
                if len(X) == 0:
                    print(f"  {name:6s} {lab:8s}     0")
                    continue
                sup = np.zeros(len(X), bool)
                for tb in sup_tabs:
                    sup |= near_any(X["mz"].values, X["rt_corr"].values, tb["mz"].values, tb["rt"].values, 10, 0.1)
                print(f"  {name:6s} {lab:8s} {len(X):6d}  replicable {np.mean(X['rep_frac'] >= 0.5):.2f}  "
                      f"EIC-junk {np.mean(X['cls'] != 'ok'):.2f}  other tool {sup.mean():.2f}  "
                      f"med snr {X['snr'].median():.1f}  med height {X['height'].median():.3g}  med scans {X['n_scans'].median():.0f}")
        if run == "HZV029_cert":
            print("\n  the 18 review cases: files (of 3) with a kept feature within 10 ppm / 0.1 min of the certified m/z, RT")
            line = {name: [] for name in VARIANTS}
            for i in CASES:
                mz, rt = hzv["moverz"].values[i], hzv["RT_minutes"].values[i]
                for name in VARIANTS:
                    line[name].append(sum(bool(near_any(np.array([mz]), np.array([rt]), o[name]["mz"].values,
                                                        o[name]["rt_corr"].values, 10, 0.1)[0]) for _, o in R))
            print("  case    " + " ".join(f"{i:>4d}" for i in CASES))
            for name in VARIANTS:
                print(f"  {name:7s} " + " ".join(f"{v:>4d}" for v in line[name]))
        pd.concat([o[name].assign(stem=s, variant=name) for s, o in R for name in VARIANTS]).to_csv(
            ROOT / f"results/diag_excess/fix_eval_{run}.tsv", sep="\t", index=False, float_format="%.6g")


if __name__ == "__main__":
    main(sys.argv[1:])
