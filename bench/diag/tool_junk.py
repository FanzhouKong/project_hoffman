# diagnostic: the replicate-junk check (junk_classes.py, as version_junk.py applies it to peak3d versions) applied to every
# tool's own per-injection detections, with one yardstick for all: the same reference noise surface and file parameters
# (version_junk.REFERENCE), the raw RT and per-run window of junk_classes.rep_check, and each feature's apex scan, height
# and FWHM re-measured on the raw EIC (5 ppm) inside the tool's own bounds, so no tool's own height or width scale enters.
# Junk = no peak in either other injection of the same sample, or the feature's own smoothed trace fails the check.
# What counts as a tool's detection in injection k:
#   peak3d_<V>  its per-file features (group_kept, as version_junk.py)
#   asari       every feature of its aligned table with intensity > 0 in k (asari detects on composite mass tracks, so
#               it has no per-file detection). Its composite RT sits a few seconds off each injection's raw RT (median
#               2.6-4.8 s on HZV029) while its bounds are ~6 s wide, so in each injection the apex is the strongest raw
#               point within the run's replicate RT window (junk_classes.JUNK_RT_WINDOW_S) of that RT, with asari's own
#               left and right extents around it
#   masscube    rows of its aligned table whose value in k equals the height of a feature in its single-file list of k
#               (|dm/z| <= 0.01, |dRT| <= 0.5 min); the rest are gap-filled. Apex and bounds from that single-file feature
#   idslipa     its per-file peak list (1-based MS1 scan bounds)
#   local tools (bench/local_tools.py, git-ignored) its detections(tool, run directory, file stem)
# usage: tool_junk.py TOOL ...   (run on a compute node; SLURM_CPUS_PER_TASK workers)
import os
import sys
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/diag"))
from bench.runs import LOCAL_TOOLS, RUNS, local_tools  # noqa: E402
from junk_classes import JUNK_RT_WINDOW_S, classify_file, rep_check  # noqa: E402
from peak3d.estimate import estimate  # noqa: E402
from peak3d.io import load_cloud  # noqa: E402
from version_junk import REFERENCE, reference_noise  # noqa: E402

RUNS_J = ["HZV029_cert", "YEAST_NEG", "SZ22_12C", "YEAST_12C"]
RES = ROOT / "results"
EIC_PPM = 5.0


def detections(tool, run, k):
    """tool's detections in injection k: DataFrame[mz, rt, rt_lo, rt_hi] (raw minutes) or (DataFrame, scan bounds)"""
    files = RUNS[run]["files"]
    stems = [Path(f).stem for f in files]
    s = stems[k]
    d = RES / tool / run
    if tool.startswith("peak3d"):
        F = pd.read_csv(d / "peak3d_out/features" / f"{s}.tsv", sep="\t")
        if "group_kept" in F.columns:
            F = F[F["group_kept"]]
        return pd.DataFrame({"mz": F["mz"].values, "rt": F["rt"].values, "rt_lo": F["rt_min"].values, "rt_hi": F["rt_max"].values})
    if tool == "asari":
        t = pd.read_csv(next(d.glob("asari_out*/export/full_Feature_table.tsv")), sep="\t")
        t = t[t[s] > 0]
        return pd.DataFrame({"mz": t["mz"].values, "rt": t["rtime"].values / 60, "rt_lo": t["rtime_left_base"].values / 60,
                             "rt_hi": t["rtime_right_base"].values / 60})
    if tool == "idslipa":
        t = pd.read_csv(d / "idslipa_out/peaklists" / f"peaklist_{s}.mzML.csv")
        D = pd.DataFrame({"mz": t["m/z 12C"].values, "rt": t["retentionTimeApex"].values, "rt_lo": np.nan, "rt_hi": np.nan})
        return D, (t["ScanNumberStart"].values - 1, t["ScanNumberEnd"].values - 1)
    if tool in LOCAL_TOOLS:
        return local_tools.detections(tool, d, s)
    if tool == "masscube":
        A = pd.read_csv(d / "aligned_feature_table.txt", sep="\t", low_memory=False)
        A = A[A[s] > 0]
        t = pd.read_csv(d / "single_files" / f"{s}.txt", sep="\t", low_memory=False)
        t = t[t["peak_height"] > 0].sort_values("m/z").reset_index(drop=True)
        smz = t["m/z"].values
        rows = []
        for mz, rt, v in zip(A["m/z"].values, A["RT"].values, A[s].values):
            a, b = np.searchsorted(smz, [mz - 0.01, mz + 0.01])
            j = np.arange(a, b)
            j = j[(np.abs(t["RT"].values[j] - rt) <= 0.5) & (np.abs(t["peak_height"].values[j] - v) <= 1.0)]
            if len(j):
                x = t.iloc[j[np.argmin(np.abs(t["RT"].values[j] - rt))]]
                rows.append((x["m/z"], x["RT"], x["RT_start"], x["RT_end"]))
        print(f"masscube {run} {s}: {len(rows)} of {len(A)} aligned values matched a single-file feature (detected)", flush=True)
        return pd.DataFrame(rows, columns=["mz", "rt", "rt_lo", "rt_hi"])
    raise ValueError(tool)


def recenter(cloud, D, win_min):
    """D with each apex moved to the strongest raw point within +-win_min of its RT, its bounds moved with it"""
    D = D.copy()
    for i in range(len(D)):
        rt = D["rt"].values[i]
        a, b = np.searchsorted(cloud.rt, [rt - win_min, rt + win_min])
        e = cloud.eic(float(D["mz"].values[i]), EIC_PPM, int(a), int(max(b, a + 1)))
        if len(e) and e.max() > 0:
            d = cloud.rt[a + int(np.argmax(e))] - rt
            for c in ("rt", "rt_lo", "rt_hi"):
                D.iloc[i, D.columns.get_loc(c)] += d
    return D


def nearest_scan(rt, t):
    """index of the scan nearest each time t"""
    i = np.clip(np.searchsorted(rt, t), 1, len(rt) - 1)
    return np.where(np.abs(rt[i - 1] - t) <= np.abs(rt[i] - t), i - 1, i)


def measure(cloud, D, scan_bounds=None):
    """apex scan, bounds, height and FWHM of each detection on the raw EIC of this injection, inside the tool's bounds"""
    S = cloud.n_scans
    sa = nearest_scan(cloud.rt, D["rt"].values)
    if scan_bounds is not None:
        lo, hi = (np.clip(np.asarray(x, int), 0, S - 1) for x in scan_bounds)
    else:      # nearest scan: tables round RTs, and a bound one scan short breaks the no-apex test of a 3-5 scan peak
        lo = nearest_scan(cloud.rt, np.nan_to_num(D["rt_lo"].values, nan=-1.0))
        hi = nearest_scan(cloud.rt, np.nan_to_num(D["rt_hi"].values, nan=1e9))
    lo, hi = np.minimum(lo, sa), np.maximum(hi, sa)
    n = len(D)
    apex, height, fwhm = np.zeros(n, int), np.zeros(n), np.zeros(n)
    for i in range(n):
        e = cloud.eic(float(D["mz"].values[i]), EIC_PPM, int(lo[i]), int(hi[i]) + 1).astype(float)
        j = int(np.argmax(e))
        apex[i], height[i] = lo[i] + j, e[j]
        if e[j] > 0:
            a = j
            while a > 0 and e[a - 1] >= 0.5 * e[j]:
                a -= 1
            b = j
            while b < len(e) - 1 and e[b + 1] >= 0.5 * e[j]:
                b += 1
            fwhm[i] = cloud.rt[lo[i] + b] - cloud.rt[lo[i] + a]
    return pd.DataFrame({"mz": D["mz"].values, "rt": cloud.rt[apex], "height": height, "fwhm": fwhm,
                         "scan_apex": apex, "scan_min": lo, "scan_max": hi})


def one(job):
    tool, run, k = job
    files = RUNS[run]["files"]
    cloud = load_cloud(files[k])
    look = reference_noise(cloud)
    P = asdict(estimate(cloud, overrides=REFERENCE))
    x = detections(tool, run, k)
    D, sb = (x if isinstance(x, tuple) else (x, None))
    if tool == "asari":
        D = recenter(cloud, D, JUNK_RT_WINDOW_S[run] / 60.0)
    F = measure(cloud, D, sb)
    F["noise"] = look(F["mz"].values, F["rt"].values)
    C = classify_file(cloud, F, P)
    others = []
    for j in range(len(files)):
        if j != k:
            cb = load_cloud(files[j])
            others.append((cb, None, asdict(estimate(cb, overrides=REFERENCE))))
    rep = rep_check(F.assign(_top=C["top_s"].values), P, others, run=run)
    return tool, run, k, pd.DataFrame({"cls": C["cls"].values, "rep": rep, "height": F["height"].values})


def main(tools):
    jobs = [(t, r, k) for r in RUNS_J for k in range(len(RUNS[r]["files"])) for t in tools]
    jobs.sort(key=lambda j: -os.path.getsize(RUNS[j[1]]["files"][j[2]]))
    from peak3d.pipeline import _warm_jit
    _warm_jit()
    with get_context("spawn").Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))) as pool:
        res = pool.map(one, jobs, chunksize=1)
    rows = []
    for run in RUNS_J:
        for tool in tools:
            A = pd.concat([o for t, r, k, o in res if t == tool and r == run])
            norep, bad = A["rep"] == 0, A["cls"] != "ok"
            junk = norep | bad
            rows.append(dict(run=run, tool=tool, features=len(A), junk=int(junk.sum()), junk_share=round(junk.mean(), 4),
                             not_reproduced=int(norep.sum()), fails_trace_check=int(bad.sum()),
                             only_not_reproduced=int((norep & ~bad).sum()), only_trace_check=int((bad & ~norep).sum()),
                             zero_signal=int((A["height"] <= 0).sum()),
                             **{f"cls_{c}": int((A["cls"] == c).sum()) for c in ("lowsnr", "noapex", "dup", "edge", "trunc")}))
    T = pd.DataFrame(rows)
    out = ROOT / "results/diag_double" / ("tool_junk_" + "_".join(tools) + ".tsv")
    T.to_csv(out, sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(T.to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
