# diagnostic: for groups detected in only some replicate files, is there signal in the other replicates?
# ratio = gap-filled height in a non-detected file / mean detected height (gap-fill reads raw centroids at
# the group's m/z and RT window, so a noise spike gives ~0 and a real but missed peak gives ~1)
# usage: replicate_signal.py RUN [RUN ...]
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for run in sys.argv[1:]:
    d = ROOT / "results/peak3d" / run / "peak3d_out"
    t = pd.read_csv(d / "aligned_feature_table.tsv", sep="\t")
    m = pd.read_csv(d / "filled_mask.tsv", sep="\t")
    G = pd.read_csv(ROOT / f"results/diag_excess/{run}_groups_cls.tsv", sep="\t", usecols=["group_id", "supported", "junk"])
    t = t.merge(G, on="group_id")
    stems = [c for c in m.columns if c != "group_id"]
    H = t[stems].values.astype(float)
    filled = m.set_index("group_id").loc[t["group_id"], stems].values.astype(bool)
    det = ~filled & (H > 0)
    nd = det.sum(1)
    mean_det = np.where(nd > 0, np.where(det, H, 0).sum(1) / np.maximum(nd, 1), np.nan)
    ratio = np.where(filled, H / mean_det[:, None], np.nan)
    best = np.nanmax(np.where(filled, ratio, np.nan), axis=1) if filled.any() else np.zeros(len(t))
    print(f"\n== {run}: {len(stems)} files, {len(t)} groups")
    for k in range(1, len(stems)):
        sel = nd == k
        if not sel.any():
            continue
        r = best[sel]
        print(f"  detected in {k}: {sel.sum():6d} groups; best filled/detected ratio in a non-detected replicate: "
              f"<0.1 {np.mean(r < 0.1):.0%}, 0.1-0.33 {np.mean((r >= 0.1) & (r < 0.33)):.0%}, "
              f"0.33-0.67 {np.mean((r >= 0.33) & (r < 0.67)):.0%}, >=0.67 {np.mean(r >= 0.67):.0%};  "
              f"supported {t['supported'].values[sel].mean():.0%}, EIC-junk {t['junk'].values[sel].mean():.0%}")
    t.assign(best_rep_ratio=best, n_det=nd).to_csv(ROOT / f"results/diag_excess/{run}_repsignal.tsv", sep="\t", index=False, float_format="%.5g")
