# diagnostic: how wide must the junk check's replicate search be? For every feature a version reports in each
# injection of the four replicate runs, find in each other injection the qualifying smoothed peak (the junk check's own
# criteria: prominence >= K_SNR x max(trace residual noise, reference cell noise), top >= 5 % of the feature's smoothed
# top) nearest to the corrected expected RT within +-MAX_MIN, and record its offset. Reports how many features count as
# "not reproduced" at several windows, and the offsets of features that clearly are reproduced (strong, found in every
# other injection), whose upper percentiles are a data-driven window.
# usage: junk_rt_window.py [--raw] VERSION_DIR ...   (--raw: expect the feature at its own native RT in the other
#        injections, no RT correction; run on a compute node, SLURM_CPUS_PER_TASK workers)
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/diag"))
from bench.runs import RUNS  # noqa: E402
from junk_classes import K_SNR, SMOOTH, classify_file  # noqa: E402
from peak3d.io import load_cloud  # noqa: E402
from version_junk import reference_noise  # noqa: E402

RUNS_J = ["HZV029_cert", "YEAST_NEG", "SZ22_12C", "YEAST_12C"]
RAW = "--raw" in sys.argv
MAX_MIN = 0.25


def offsets(F, P, cloudB, warpB, PB):
    """per feature: offset (min) of the qualifying peak nearest the expected RT in cloudB, NaN if none within MAX_MIN"""
    S = cloudB.n_scans
    sig_k = SMOOTH * PB["fwhm_scans"] / 2.355
    rtB = F["rt"].values if RAW else np.interp(F["rt_corr"].values, warpB["rt_corr"].values, warpB["rt_native"].values)
    out = np.full(len(F), np.nan)
    for i in range(len(F)):
        s0, s1 = np.searchsorted(cloudB.rt, [rtB[i] - MAX_MIN - 0.3, rtB[i] + MAX_MIN + 0.3])
        s0, s1 = max(0, s0), min(S, s1)
        if s1 - s0 < 5:
            continue
        mz, h = float(F["mz"].values[i]), max(float(F["height"].values[i]), 1.0)
        ppm = max(5.0, 3.0 * np.sqrt(PB["sig_a"] ** 2 + PB["sig_b"] ** 2 / h))
        e = cloudB.eic(mz, ppm, s0, s1).astype(np.float64)
        if e.max() <= 0:
            continue
        es = gaussian_filter1d(e, sig_k, mode="nearest")
        r = e - es
        noise_res = 1.4826 * np.median(np.abs(r - np.median(r)))
        ref = max(noise_res, F["noise"].values[i])
        t = cloudB.rt[s0:s1]
        best = np.nan
        for k in range(1, len(es) - 1):
            if not (es[k] >= es[k - 1] and es[k] >= es[k + 1]) or abs(t[k] - rtB[i]) > MAX_MIN:
                continue
            top = es[k]
            if top < 0.05 * F["_top"].values[i]:
                continue
            j, lmin = k, top
            while j > 0 and es[j - 1] <= top:
                j -= 1
                lmin = min(lmin, es[j])
            j, rmin = k, top
            while j < len(es) - 1 and es[j + 1] <= top:
                j += 1
                rmin = min(rmin, es[j])
            if (top - max(lmin, rmin)) / ref >= K_SNR and (np.isnan(best) or abs(t[k] - rtB[i]) < abs(best)):
                best = t[k] - rtB[i]
        out[i] = best
    return out


def one(job):
    run, fi, vdir = job
    files = RUNS[run]["files"]
    stems = [Path(f).stem for f in files]
    d = Path(vdir) / run / "peak3d_out"
    cloud = load_cloud(files[fi])
    F = pd.read_csv(d / "features" / f"{stems[fi]}.tsv", sep="\t")
    if "group_kept" in F.columns:
        F = F[F["group_kept"]].reset_index(drop=True)
    P = json.load(open(d / "features" / f"{stems[fi]}.params.json"))
    F["noise"] = reference_noise(cloud)(F["mz"].values, F["rt"].values)
    C = classify_file(cloud, F, P)
    F["_top"] = C["top_s"].values
    cols = {}
    for k in range(len(files)):
        if k == fi:
            continue
        PB = json.load(open(d / "features" / f"{stems[k]}.params.json"))
        cols[k] = offsets(F, P, load_cloud(files[k]), pd.read_csv(d / "rt_correction" / f"{stems[k]}.tsv", sep="\t"), PB)
    O = np.stack(list(cols.values()), axis=1)
    width = np.maximum(F["fwhm"].values, P["fwhm_med"])          # the junk check's current window: one peak width
    return pd.DataFrame({"run": run, "inj": fi, "snr": F["snr"].values, "cls": C["cls"].values, "width": width,
                         **{f"off{j}": O[:, j] for j in range(O.shape[1])}})


def main():
    for vdir in [a for a in sys.argv[1:] if not a.startswith("--")]:
        jobs = [(r, k, str(ROOT / vdir)) for r in RUNS_J for k in range(len(RUNS[r]["files"]))]
        with get_context("spawn").Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))) as pool:
            A = pd.concat(pool.map(one, jobs, chunksize=1), ignore_index=True)
        off = A[[c for c in A.columns if c.startswith("off")]].abs().values
        rows = []
        for run, g in A.groupby("run", sort=False):
            o = off[g.index]
            w = g["width"].values[:, None]
            row = dict(run=run, features=len(g))
            for name, win in [("1 width (now)", w), ("2 widths", 2 * w), ("3 widths", 3 * w), ("0.05 min", 0.05),
                              ("0.10 min", 0.10), ("0.20 min", 0.20)]:
                found = np.nan_to_num(o, nan=np.inf) <= win
                row[f"not reproduced, {name}"] = int((~found.any(axis=1)).sum())
            strong = (g["snr"].values >= 20) & np.isfinite(o).all(axis=1)
            row["strong found in both: offset p50 / p95 / p99 (s)"] = " / ".join(
                f"{60 * v:.1f}" for v in np.nanpercentile(o[strong].max(axis=1), [50, 95, 99])) if strong.any() else "-"
            rows.append(row)
        T = pd.DataFrame(rows)
        pd.set_option("display.width", 250, "display.max_columns", 30)
        print(f"== {vdir} ({'raw RT' if RAW else 'corrected RT'})")
        print(T.T.to_string(header=False))
        T.to_csv(ROOT / "results/diag_double" / f"junk_rt_window_{Path(vdir).name}{'_raw' if RAW else ''}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
