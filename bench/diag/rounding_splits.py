# diagnostic: aligned groups of one ion split only because per-file m/z was written with %.6g
# (6 significant digits: 0.0001 Da below m/z 100, 0.001 Da to 1000, 0.01 Da above) and re-read by
# the align stage. Counts complementary group pairs (no file in common, RT ranges overlapping) whose
# m/z differ by exactly one rounding step and more than the 5 ppm grouping tolerance.
# usage: rounding_splits.py RUN [RUN ...]
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for run in sys.argv[1:]:
    d = ROOT / "results/peak3d" / run / "peak3d_out"
    t = pd.read_csv(d / "aligned_feature_table.tsv", sep="\t")
    m = pd.read_csv(d / "filled_mask.tsv", sep="\t").set_index("group_id").loc[t["group_id"]]
    stems = list(m.columns)
    det = (m.values == 0) & (t[stems].values > 0)
    mz, lo, hi = t["mz"].values, t["rt_min"].values, t["rt_max"].values
    step = np.where(mz < 100, 1e-4, np.where(mz < 1000, 1e-3, 1e-2))
    o = np.argsort(mz)
    smz = mz[o]
    pairs = 0
    part = np.zeros(len(t), bool)
    for i in range(len(t)):
        a = np.searchsorted(smz, mz[i] + 0.5 * step[i])
        b = np.searchsorted(smz, mz[i] + 1.5 * step[i])
        for j in o[a:b]:
            if abs(mz[j] - mz[i]) / mz[i] * 1e6 <= 5.0:
                continue
            if min(hi[i], hi[j]) <= max(lo[i], lo[j]):
                continue
            if (det[i] & det[j]).any():
                continue
            pairs += 1
            part[i] = part[j] = True
    nd = det.sum(1)
    big = (mz >= 100) & (mz < 200) | (mz >= 1000)
    print(f"{run}: {len(t)} groups; complementary pairs one rounding step (> 5 ppm) apart: {pairs}; groups involved {part.sum()} "
          f"({part.mean():.1%}); among groups at m/z 100-200 or >= 1000: {part[big].mean():.1%}; "
          f"mean n_detected of involved groups {nd[part].mean():.1f} vs all {nd.mean():.1f} of {len(stems)} files")
