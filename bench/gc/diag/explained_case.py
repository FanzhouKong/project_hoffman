"""Why a component is, or is not, dissolved by the explained-component rule: its larger overlapping neighbours
(from a run with the rule off), and how their models fit its strongest ions.
    python explained_case.py LEVEL|STEM RT_MIN [OUTPUT under results/] [COMPONENT]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import nnls
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from peak3d.gc import deconv as DC
from peak3d.io import Cloud

R = Path("/quobyte/metabolomicsgrp/fanzhou/hoffmann/results")
lv, rt0 = sys.argv[1], float(sys.argv[2])
out = R / (sys.argv[3] if len(sys.argv) > 3 else "gc_mrh/peak3d")          # e.g. gc_deepcpr/mosa_dsk/fa
stem = f"NEWSTD_{lv}" if "gc_mrh" in str(out) else lv
P = DC.params_from_dict(json.loads((out / "features" / f"{stem}.params.json").read_text()))
cloud = Cloud.load_npz(out / ".cache" / f"{stem}.npz")
F = pd.read_csv(out / "features" / f"{stem}.tsv", sep="\t", low_memory=False)
d = DC.deconvolve(cloud, F, P, DC.DeconvParams(explain=False))
C = d.components
near = C[(C.rt - rt0).abs() * 60 < 6].copy()
print(f"dt {P.dt*60:.3f} s, fwhm {P.fwhm_med*60:.2f} s; components within 6 s of {rt0} min:")
print(near[["component_id", "rt", "scan_min", "scan_max", "n_features", "n_ions", "base_mz", "tic", "apex_tol_s"]].assign(dt_s=(near.rt - rt0) * 60).round(4).to_string(index=False))
b = int(near.iloc[(near.rt - rt0).abs().argmin()].component_id) if len(sys.argv) < 5 else int(sys.argv[4])
rb = C.iloc[b]; lo, hi = int(rb.scan_min), int(rb.scan_max)
big = C[(C.n_features > rb.n_features) & (C.scan_max >= lo) & (C.scan_min <= hi)]
big = big.iloc[(big.rt - rb.rt).abs().argsort()[:4]]
def on(c):
    r = C.iloc[c]; p = d.profile(c); v = np.zeros(hi - lo + 1)
    a, e = max(lo, int(r.scan_min)), min(hi, int(r.scan_max))
    if e >= a: v[a - lo:e - lo + 1] = p[a - int(r.scan_min):e - int(r.scan_min) + 1]
    return v
X = np.column_stack([np.ones(hi - lo + 1)] + [on(int(c)) for c in big.component_id])
np.set_printoptions(linewidth=250, suppress=True, precision=2)
print("calibration:", {k: d.info.get(k) for k in ("shape_classes", "snr_edges", "tau_classes_s", "shape_min")})
print(f"component {b}: rt {rb.rt:.4f}, {int(rb.n_features)} tight ions, bounds {hi-lo+1} scans; larger overlapping:", [(int(c), round((r - rb.rt) * 60, 2), int(n)) for c, r, n in zip(big.component_id, big.rt, big.n_features)])
print("own model     ", (100 * d.profile(b)).round(0).astype(int)[::max(1, (hi - lo) // 40)])
for q, c in enumerate(big.component_id):
    print(f"model of {int(c):5d}", (100 * X[:, q + 1]).round(0).astype(int)[::max(1, (hi - lo) // 40)])
y = d.profile(b); x, _ = nnls(X, y); fit = X @ x
print("own model fitted by the others: coefficients", x.round(3), "shares", (x[1:] * X[:, 1:].sum(0) / max((x[1:] * X[:, 1:].sum(0)).sum(), 1e-12)).round(2), "r", round(float(np.corrcoef(fit, y)[0, 1]), 4))
fc = pd.Series(d.feature_component)
mem = F[fc.values == b].sort_values("height", ascending=False).head(5)
for _, f in mem.iterrows():
    yy = cloud.eic(f.mz, 5.0, lo, hi + 1).astype(float); x, _ = nnls(X, yy); fit = X @ x
    x1, _ = nnls(np.column_stack([X, d.profile(b)]), yy); fit1 = np.column_stack([X, d.profile(b)]) @ x1
    rc = float(np.sqrt(max(0.0, 1.0 - 0.0)))
    print(f"  ion {f.mz:.4f} h {f.height:.0f} snr {f.snr:.0f}: others r {np.corrcoef(fit, yy)[0,1]:.4f} sse/tot {((yy-fit)**2).sum()/((yy-yy.mean())**2).sum():.4f} shares {(x[1:]*X[:,1:].sum(0)/max((x[1:]*X[:,1:].sum(0)).sum(),1e-12)).round(2)}; with own sse/tot {((yy-fit1)**2).sum()/((yy-yy.mean())**2).sum():.4f}")
