# diagnostic: candidate iteration-2 rules evaluated on the iteration-1 per-file tables (results/diag_idsl/remaining_*.tsv):
# yardstick junk removed, good removed, and truth lost (file-level detections and compounds found in >= 1 file).
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench/diag"))
from bench.runs import RUNS
from fix_eval import eval_truth, near_any
pd.set_option("display.width", 230)
runs = sys.argv[1:] or ["HZV029_cert", "YEAST_NEG", "SZ22_12C", "YEAST_12C"]
D = {}
for run in runs:
    R = pd.read_csv(ROOT / f"results/diag_idsl/remaining_{run}.tsv", sep="\t")
    R = R[R["group_kept"]].reset_index(drop=True)
    T = eval_truth(run)
    # truth hits per file-level feature: which features are within 10 ppm / 0.1 min of a truth entry
    o = np.argsort(T["mz"].values); tmz = T["mz"].values[o]; trt = T["rt"].values[o]
    lo = np.searchsorted(tmz, R["mz"].values * (1 - 1e-5)); hi = np.searchsorted(tmz, R["mz"].values * (1 + 1e-5), side="right")
    R["is_truth"] = [np.any(np.abs(trt[a:b] - r) <= 0.1) for a, b, r in zip(lo, hi, R["rt_corr"].values)]
    R["junk"] = (R["rep_frac"] == 0) | (R["cls"] != "ok")
    R["good"] = (R["rep_frac"] >= 0.5) & (R["cls"] == "ok")
    R["presence"] = R["n_detected"] + R["n_confirmed"]
    D[run] = (R, T)

def truth_found(R, T, keep):
    sub = R[keep]
    return int(near_any(T["mz"].values, T["rt"].values, sub["mz"].values, sub["rt_corr"].values, 10, 0.1).sum())

rules = {
    "iteration 1 (none)": lambda R: np.ones(len(R), bool),
    "score >= 0.6": lambda R: R["score"] >= 0.6,
    "score >= 0.6 unless presence = all files": lambda R: (R["score"] >= 0.6) | (R["presence"] >= R["n_files"]),
    "score >= 0.6 unless detected in >= 2 files": lambda R: (R["score"] >= 0.6) | (R["n_detected"] >= 2),
    "drop fwhm_rel > 3": lambda R: R["fwhm_rel"] <= 3,
    "drop fwhm_rel > 3 & (far_level > 0.1 | tv > 1.5)": lambda R: ~((R["fwhm_rel"] > 3) & ((R["far_level"] > 0.1) | (R["tv_ratio"] > 1.5))),
    "drop tv_ratio > 2.5": lambda R: R["tv_ratio"] <= 2.5,
    "drop far_level > 0.3": lambda R: R["far_level"] <= 0.3,
    "drop ridge_ratio > 0.3": lambda R: R["ridge_ratio"] <= 0.3,
    "drop eic_snr < 5": lambda R: R["eic_snr"] >= 5,
    "drop eic_snr < 5 unless detected in >= 2 files": lambda R: (R["eic_snr"] >= 5) | (R["n_detected"] >= 2),
    "drop prominence_rel < 0.5": lambda R: R["prominence_rel"] >= 0.5,
    "drop gauss_r2 < 0.3": lambda R: R["gauss_r2"] >= 0.3,
    "drop (fwhm_rel > 2.5 & prominence_rel < 0.8)": lambda R: ~((R["fwhm_rel"] > 2.5) & (R["prominence_rel"] < 0.8)),
    "drop single-detection groups (det = 1)": lambda R: R["n_detected"] >= 2,
    "drop det = 1 & eic_snr < 6": lambda R: ~((R["n_detected"] == 1) & (R["eic_snr"] < 6)),
    "drop det = 1 & score < 0.65": lambda R: ~((R["n_detected"] == 1) & (R["score"] < 0.65)),
}
for run, (R, T) in D.items():
    R["n_files"] = len(RUNS[run]["files"])
    base_truth = truth_found(R, T, np.ones(len(R), bool))
    print(f"\n================ {run}: {len(R)} kept features, junk {int(R['junk'].sum())}, good {int(R['good'].sum())}, truth compounds found {base_truth}/{len(T)}, truth-matched features {int(R['is_truth'].sum())}")
    rows = []
    for name, fn in rules.items():
        keep = np.asarray(fn(R), bool)
        rows.append(dict(rule=name, features_removed=int((~keep).sum()), junk_removed=int((R["junk"] & ~keep).sum()),
                         good_removed=int((R["good"] & ~keep).sum()), truth_feat_removed=int((R["is_truth"] & ~keep).sum()),
                         truth_compounds_lost=base_truth - truth_found(R, T, keep),
                         junk_left=int((R["junk"] & keep).sum())))
    print(pd.DataFrame(rows).to_string(index=False))
