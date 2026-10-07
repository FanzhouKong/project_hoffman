# diagnostic: BM21 ratio consistency of peak3d features at matched intensity cutoffs (report-only)
import sys, re, glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = "/quobyte/metabolomicsgrp/fanzhou/hoffmann"
des = pd.read_csv(f"{ROOT}/results/inputs/BM21_design.tsv", sep="\t"); des = des[des["plasma_fraction"].notna()]
run, an = sys.argv[1] if len(sys.argv) > 1 else "BM21_RP", "RP" if (len(sys.argv) < 2 or sys.argv[1].endswith("RP")) else "HILIC"
d = des[des["analysis"] == an]; frac = d.set_index("sample")["plasma_fraction"]

def consistent(S, cols):
    M = S[cols].fillna(0).values; p = frac[cols].values
    rho = np.array([spearmanr(v, p).statistic if v.std() > 0 else np.nan for v in M])
    return np.abs(rho) >= 0.9

t = pd.read_csv(f"{ROOT}/results/peak3d/{run}/peak3d_out/aligned_feature_table.tsv", sep="\t")
cols = [c for c in t.columns if c in frac.index]
S = t[cols]; good = consistent(S, cols); hmax = S.max(axis=1).values
print(f"peak3d {run}: {len(t)} groups; ratio-consistent {good.sum()} ({good.mean():.3f})")
print(" max height across design samples: quantiles", np.percentile(hmax, [10, 25, 50, 75, 90]).round(0).tolist())
for thr in [0, 3e3, 1e4, 3e4, 1e5]:
    sel = hmax >= thr
    print(f"  height >= {thr:>8.0f}: {sel.sum():6d} features, consistent {good[sel].sum():5d} ({good[sel].mean():.3f})")
# the same for asari and masscube tables, with their own reported heights
a = pd.read_csv(glob.glob(f"{ROOT}/results/asari/{run}/asari_out*/export/full_Feature_table.tsv")[0], sep="\t")
ac = [c for c in a.columns if re.sub(r"\.mzML$", "", c) in frac.index]; A = a[ac].copy(); A.columns = [re.sub(r"\.mzML$", "", c) for c in ac]
ga = consistent(A.astype(float), list(A.columns)); ha = A.astype(float).max(axis=1).values
print(f"\nasari {run}: {len(a)} features; consistent {ga.sum()} ({ga.mean():.3f}); max-intensity quantiles", np.percentile(ha, [10, 50, 90]).round(0).tolist())
for thr in [1e4, 3e4, 1e5]:
    sel = ha >= thr; print(f"  >= {thr:>8.0f}: {sel.sum():6d}, consistent {ga[sel].sum():5d} ({ga[sel].mean():.3f})")
m = pd.read_csv(f"{ROOT}/results/masscube/{run}/aligned_feature_table.txt", sep="\t", low_memory=False)
mc = [c for c in m.columns if re.sub(r"\.mzML$", "", str(c)) in frac.index]; M = m[mc].copy(); M.columns = [re.sub(r"\.mzML$", "", str(c)) for c in mc]
gm = consistent(M.astype(float), list(M.columns)); hm = M.astype(float).max(axis=1).values
print(f"\nmasscube {run}: {len(m)} features; consistent {gm.sum()} ({gm.mean():.3f}); max-intensity quantiles", np.percentile(hm, [10, 50, 90]).round(0).tolist())
for thr in [1e4, 3e4, 1e5]:
    sel = hm >= thr; print(f"  >= {thr:>8.0f}: {sel.sum():6d}, consistent {gm[sel].sum():5d} ({gm[sel].mean():.3f})")
# purity at matched list size: top-k by max height
print("\npurity at matched list size (top-k features by max intensity):")
for k in [3000, 5000, 9000]:
    for name, h, g in [("peak3d", hmax, good), ("asari", ha, ga), ("masscube", hm, gm)]:
        if len(h) >= k:
            idx = np.argsort(-h)[:k]; print(f"  top {k:5d} {name:9s}: consistent {g[idx].sum():5d} ({g[idx].mean():.3f})")
