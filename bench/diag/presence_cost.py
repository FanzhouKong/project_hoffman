# diagnostic: on the Oct 5 aligned tables, what does "detected in >= 2 files" remove and what truth does it cost?
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench/diag"))
from bench.runs import RUNS
from fix_eval import eval_truth, near_any
snap = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/peak3d_2026-10-05"
for run in ["HZV029_cert", "YEAST_NEG", "SZ22_12C", "YEAST_12C", "LI2018", "HZV029_full", "BM21_RP", "BM21_HILIC"]:
    t = pd.read_csv(snap / run / "peak3d_out/aligned_feature_table.tsv", sep="\t", usecols=["mz", "rt", "n_detected", "n_filled"])
    N = len(RUNS[run]["files"])
    line = f"{run:12s} files {N:3d} groups {len(t):6d}  n_detected==1 {np.mean(t['n_detected'] == 1):5.1%}  >=2 {np.mean(t['n_detected'] >= 2):5.1%}  >=3 {np.mean(t['n_detected'] >= 3):5.1%}  (filled>0 among singles {np.mean(t.loc[t['n_detected'] == 1, 'n_filled'] > 0):5.1%})"
    T = eval_truth(run)
    if T is not None and "tp" not in T:
        for k in (1, 2, 3):
            keep = t[t["n_detected"] >= k]
            h = near_any(T["mz"].values, T["rt"].values, keep["mz"].values, keep["rt"].values, 10, 0.1)
            line += f"  truth@>={k}: {h.sum()}/{len(T)}"
    print(line)
