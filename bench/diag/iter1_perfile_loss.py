# diagnostic: truth compounds that Oct 5 found and iteration 1 has no group for at all: which per-file test removed
# their Oct 5 features? (re-run the chromatogram check and the basin gates on the Oct 5 feature rows)
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench/diag"))
from bench.runs import RUNS
from fix_eval import eval_truth, near_any
from peak3d.estimate import estimate
from peak3d.features import eic_check, ridge_ratio
from peak3d.io import load_cloud
pd.set_option("display.width", 220)
for run in sys.argv[1:]:
    files = RUNS[run]["files"]; stems = [Path(f).stem for f in files]
    T = eval_truth(run)
    PM = pd.read_csv(ROOT / f"results/peak3d/{run}/peak3d_out/presence_mask.tsv", sep="\t")
    h_now = near_any(T["mz"].values, T["rt"].values, PM["mz"].values, PM["rt"].values, 10, 0.1)
    lost_T = T[~h_now]
    rows = []
    for s, f in zip(stems, files):
        F = pd.read_csv(ROOT / f"results/peak3d_2026-10-05/{run}/peak3d_out/features/{s}.tsv", sep="\t")
        hit = near_any(F["mz"].values, F["rt_corr"].values, lost_T["mz"].values, lost_T["rt"].values, 10, 0.1)
        sub = F[hit].copy()
        if len(sub) == 0: continue
        cloud = load_cloud(f); P = estimate(cloud)
        sub["scan_lo"] = sub["scan_min"]; sub["scan_hi"] = sub["scan_max"]
        sub["sig_apex"] = P.sig_ppm(sub["height"].values)
        E = eic_check(sub, cloud, P)
        R = ridge_ratio(sub, cloud, P)
        sub["eic_class"] = E["eic_class"].values; sub["eic_snr"] = E["eic_snr"].values; sub["es_noise"] = E["es_noise"].values
        sub["ridge_new"] = R["ridge_ratio"]; sub["far_level"] = R["far_level"]; sub["stem"] = s
        rows.append(sub)
    A = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    print(f"\n================ {run}: truth compounds with no iteration-1 group: {len(lost_T)}; their Oct 5 per-file features: {len(A)}")
    if len(A):
        print("  chromatogram-check class of those features: " + ", ".join(f"{k} {v}" for k, v in A["eic_class"].value_counts().items()))
        print("  ridge gate (usable sides) would now reject: %d; medians: height %.3g, snr %.1f, eic_snr %.2f, n_scans %.0f, fwhm/med %.2f, score %.2f, isotopologues %.2f" % (
            (A["ridge_new"] > 0.5).sum(), A["height"].median(), A["snr"].median(), A["eic_snr"].median(), A["n_scans"].median(),
            (A["fwhm"] / np.median([json.load(open(ROOT / f"results/peak3d_2026-10-05/{run}/peak3d_out/features/{s}.params.json"))["fwhm_med"] for s in stems])).median(),
            A["score"].median(), (A["iso_offset"] > 0).mean()))
        print("  eic_snr quantiles of the lost:", A["eic_snr"].quantile([.1, .25, .5, .75, .9]).round(2).to_dict())
