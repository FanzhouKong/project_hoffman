"""Mosa's existing MRH output scored by the MRH preprint's stated rule: a detection is a true positive when it lies
within dMZ = 0.001 and dRT = 0.02 of a ground-truth peak; every other detection is a false positive. Units are not
stated: m/z taken as Da, RT tried as 0.02 min (1.2 s) and 0.02 s. Truth = all rows with data (unscreened), as theirs."""
import numpy as np, pandas as pd
from pathlib import Path
out = Path("results/gc_mrh/peak3d/features")
T = pd.read_csv("results/gc_mrh/score/truth_rows.tsv", sep="\t", dtype={"level": str})
LEVELS = ["0_1", "0_5", "1", "5", "10", "25", "50", "100", "250", "500", "1000"]
pub = dict(zip(LEVELS, [0.035, 0.04, 0.04, 0.083, 0.11, 0.17, 0.22, 0.26, 0.30, 0.29, 0.30]))   # MRH, read off Fig. 4
rows = []
for lv in LEVELS:
    F = pd.read_csv(out / f"NEWSTD_{lv}.tsv", sep="\t", usecols=["mz", "rt_apex", "rt", "height"])
    t = T[(T["level"] == lv) & T["matched"]].drop_duplicates(["mz", "rt"])
    d = dict(level=lv, n_truth=len(t), n_features=len(F))
    fm, fr = F["mz"].values, F["rt_apex"].values
    for name, dmz, drt in (("da001_min02", 0.001, 0.02), ("da001_s02", 0.001, 0.02 / 60), ("ppm10_2s", None, 2.0 / 60)):
        hit_t = np.zeros(len(t), bool); hit_f = np.zeros(len(F), bool)
        for i, (m, r) in enumerate(zip(t["mz"].values, t["rt"].values)):
            k = np.flatnonzero((np.abs(fm - m) <= (dmz if dmz else m * 1e-5)) & (np.abs(fr - r) <= drt))
            if len(k):
                hit_t[i] = True; hit_f[k] = True
        rec, prec = hit_t.mean(), hit_f.sum() / max(len(F), 1)
        d[f"recall_{name}"] = rec; d[f"precision_{name}"] = prec; d[f"f1_{name}"] = 2 * rec * prec / max(rec + prec, 1e-12)
    d["mrh_f1_published"] = pub[lv]
    rows.append(d)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
R = pd.DataFrame(rows); print(R.round(3).to_string(index=False))
R.to_csv("results/gc_mrh/score/f1_published_rule.tsv", sep="\t", index=False, float_format="%.4g")
