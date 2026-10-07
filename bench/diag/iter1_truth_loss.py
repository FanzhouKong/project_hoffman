# diagnostic: credentialed / curated truth compounds found by the Oct 5 version but not by iteration 1: were they dropped
# by the presence rule (single detection, no confirmation) or lost at the per-file stage? For the presence-dropped ones,
# re-run the confirmation in the other files with wider windows / looser thresholds, and count how many of ALL dropped
# groups (truth or not) each variant would bring back.   usage: iter1_truth_loss.py RUN [RUN ...]
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench/diag"))
from bench.runs import RUNS
from fix_eval import eval_truth, near_any
from peak3d.gapfill import fill_file
from peak3d.io import load_cloud
from peak3d.warp import TableWarp
pd.set_option("display.width", 220)

VARIANTS = {"current (half = max(rt_tol, w/2), 5 %, k 3)": dict(half_fwhm=0.0, top=0.05, k=3.0),
            "half >= 1 FWHM": dict(half_fwhm=1.0, top=0.05, k=3.0),
            "half >= 1.5 FWHM": dict(half_fwhm=1.5, top=0.05, k=3.0),
            "half >= 1 FWHM, top 2 %": dict(half_fwhm=1.0, top=0.02, k=3.0),
            "half >= 1 FWHM, k 2": dict(half_fwhm=1.0, top=0.05, k=2.0),
            "half >= 1.5 FWHM, top 2 %, k 2": dict(half_fwhm=1.5, top=0.02, k=2.0)}

for run in sys.argv[1:]:
    files = RUNS[run]["files"]; stems = [Path(f).stem for f in files]; N = len(files)
    d = ROOT / "results/peak3d" / run / "peak3d_out"
    PM = pd.read_csv(d / "presence_mask.tsv", sep="\t")
    T = eval_truth(run)
    prm = json.load(open(d / "params.json"))
    rt_tol = float(prm["rt_tol_min"])
    old = pd.read_csv(ROOT / f"results/peak3d_2026-10-05/{run}/peak3d_out/aligned_feature_table.tsv", sep="\t", usecols=["mz", "rt"])
    kept = PM[(PM["n_detected"] + PM["n_confirmed"]) >= min(2, N)]
    h_old = near_any(T["mz"].values, T["rt"].values, old["mz"].values, old["rt"].values, 10, 0.1)
    h_all = near_any(T["mz"].values, T["rt"].values, PM["mz"].values, PM["rt"].values, 10, 0.1)
    h_kept = near_any(T["mz"].values, T["rt"].values, kept["mz"].values, kept["rt"].values, 10, 0.1)
    print(f"\n================ {run}: truth {len(T)}; found Oct 5 {h_old.sum()}, iteration 1 before presence rule {h_all.sum()}, after {h_kept.sum()}")
    print(f"  lost at the per-file stage (no group at all, had one in Oct 5): {int((h_old & ~h_all).sum())}; dropped by the presence rule: {int((h_all & ~h_kept).sum())}")
    # the dropped groups: which match truth
    dropped = PM[(PM["n_detected"] + PM["n_confirmed"]) < min(2, N)].copy()
    tmz = np.sort(T["mz"].values); o = np.argsort(T["mz"].values); trt = T["rt"].values[o]
    lo = np.searchsorted(tmz, dropped["mz"].values * (1 - 1e-5)); hi = np.searchsorted(tmz, dropped["mz"].values * (1 + 1e-5), side="right")
    dropped["is_truth"] = [np.any(np.abs(trt[a:b] - r) <= 0.1) for a, b, r in zip(lo, hi, dropped["rt"].values)]
    print(f"  dropped groups {len(dropped)}, of which truth {int(dropped['is_truth'].sum())}; detected-in-1 {int((dropped['n_detected'] == 1).sum())}")
    # per dropped group: the detecting file's feature (height, noise, width) for the reference values
    feats = {s: pd.read_csv(d / "features" / f"{s}.tsv", sep="\t") for s in stems}
    params = {s: json.load(open(d / "features" / f"{s}.params.json")) for s in stems}
    warps = {s: pd.read_csv(d / "rt_correction" / f"{s}.tsv", sep="\t") for s in stems}
    ref = {}
    for s in stems:
        F = feats[s]
        sub = F[F["group_id"].isin(dropped["group_id"].values)]
        for gid, h, nz, w in zip(sub["group_id"].values, sub["height"].values, sub["noise"].values, (sub["rt_max_corr"] - sub["rt_min_corr"]).values):
            if gid not in ref or h > ref[gid][0]:
                ref[gid] = (h, nz, w, s)
    clouds = {s: load_cloud(f) for s, f in zip(stems, files)}
    fwhm_med = float(np.median([params[s]["fwhm_med"] for s in stems]))
    rows = []
    for name, v in VARIANTS.items():
        conf = np.zeros(len(dropped), int)
        for s in stems:
            idx = [i for i, g in enumerate(dropped["group_id"].values) if ref.get(g, (0, 0, 0, None))[3] != s]
            if not idx: continue
            sub = dropped.iloc[idx]
            h_ref = np.array([ref[g][0] for g in sub["group_id"].values]); n_ref = np.array([ref[g][1] for g in sub["group_id"].values])
            w = np.array([ref[g][2] for g in sub["group_id"].values])
            half = np.maximum(np.maximum(rt_tol, 0.5 * w), v["half_fwhm"] * fwhm_med)
            wp = warps[s]
            warp = TableWarp(wp["rt_native"].values, wp["rt_corr"].values)
            _, _, c = fill_file(clouds[s], warp, sub["mz"].values, sub["rt"].values, half, float(prm["ppm_tol"]), n_ref, h_ref,
                                params[s]["fwhm_scans"], k_snr=v["k"], min_top_frac=v["top"])
            conf[np.array(idx)] += c.astype(int)
        back = conf >= 1
        rows.append(dict(variant=name, truth_rescued=int((back & dropped["is_truth"].values).sum()), truth_dropped=int(dropped["is_truth"].sum()),
                         all_rescued=int(back.sum()), all_dropped=len(dropped), rescued_share=round(back.mean(), 3)))
    print(pd.DataFrame(rows).to_string(index=False))
