import numpy as np, pandas as pd
from pathlib import Path
out = Path("results/gc_mrh/peak3d/gc/components")
T = pd.read_csv("results/gc_mrh/score/truth_rows.tsv", sep="\t", dtype={"level": str})
T = T[T["clean"]].copy()
T["in_spectrum"] = False; T["rel"] = np.nan
for lv, t in T.groupby("level"):
    comps = pd.read_csv(out / f"NEWSTD_{lv}.tsv", sep="\t")
    z = np.load(out / f"NEWSTD_{lv}.npz")
    off, mz, ab = z["off"], z["mz"], z["abundance"]
    crt = comps["rt"].values
    for i, r in t.iterrows():
        for c in np.flatnonzero(np.abs(crt - r["rt_consensus"]) * 60 <= 1.5):
            m, a = mz[off[c]:off[c + 1]], ab[off[c]:off[c + 1]]
            k = np.flatnonzero(np.abs(m - r["mz"]) <= r["mz"] * 1e-5)
            if len(k):
                T.loc[i, "in_spectrum"] = True; T.loc[i, "rel"] = a[k].max() / a.max(); break
T["class"] = np.where(T["found"], "picked", T["outcome"])
order = ["0_1", "0_5", "1", "5", "10", "25", "50", "100", "250", "500", "1000"]
g = T.groupby("class")["in_spectrum"].agg(["size", "sum", "mean"]).round(3); print(g.to_string())
print(T.groupby("level")[["found", "in_spectrum"]].mean().reindex(order).round(3).assign(n=T.groupby("level").size()).to_string())
print("either:", (T["found"] | T["in_spectrum"]).groupby(T["level"]).mean().reindex(order).round(3).to_dict())
