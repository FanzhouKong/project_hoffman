#!/usr/bin/env python3
"""Standard error of every feature's sub-scan apex time from its own fit (the weighted log-parabola of
peak3d.gc.deconv._log_parabola_apex), against the calibrated apex tolerance of its S/N class.

    python bench/gc/diag/apex_se.py OUTDIR STEM [T0 T1]
OUTDIR: a peak3d output folder; T0, T1 (seconds): list the features of that window."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from peak3d.gc import deconv as DC                       # noqa: E402
from peak3d.io import Cloud, eic_kernel                  # noqa: E402

out_dir, stem = Path(sys.argv[1]), sys.argv[2]
win = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else None
P = DC.params_from_dict(json.loads((out_dir / "features" / f"{stem}.params.json").read_text()))
cloud = Cloud.load_npz(out_dir / ".cache" / f"{stem}.npz")
F = pd.read_csv(out_dir / "features" / f"{stem}.tsv", sep="\t", low_memory=False)
D = DC.DeconvParams()
f_mz = F["mz"].values.astype(float); f_h = F["height"].values.astype(float)
f_tol = DC.ion_tol(f_mz, f_h, P)
S = cloud.n_scans
f_apex = np.clip(F["scan_apex"].values.astype(np.int64), 0, S - 1)
half = int(np.ceil(2.0 * P.fwhm_med / P.dt)) + 2
n = len(F)
f_t, _ = DC._apex_times(cloud.off, cloud.mz, cloud.inten, cloud.rt, f_mz, f_tol, f_apex, half, float(P.dt),
                     np.zeros(n, np.int64), np.full(n, S - 1, np.int64))
tau, rcut, info = DC.calibrate(F, f_t, cloud, f_tol, P, D)


def fit(y, k, x):
    """apex (x units), its standard error, points used"""
    top = y[k]
    lo = k
    while lo > 0 and 0.5 * top <= y[lo - 1] <= top:
        lo -= 1
    hi = k
    while hi < len(y) - 1 and 0.5 * top <= y[hi + 1] <= top:
        hi += 1
    if lo > 0 and 0 < y[lo - 1] <= top:
        lo -= 1
    if hi < len(y) - 1 and 0 < y[hi + 1] <= top:
        hi += 1
    m = hi - lo + 1
    if m < 3:
        return np.nan, np.nan, m
    w = y[lo:hi + 1].astype(float); u = x[lo:hi + 1] - x[k]
    ok = w > 0
    w, u = w[ok], u[ok]
    if len(w) < 3:
        return np.nan, np.nan, len(w)
    A = np.column_stack([np.ones(len(u)), u, u * u])
    N = A.T @ (A * w[:, None])
    try:
        Ni = np.linalg.inv(N)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, len(w)
    c = Ni @ (A.T @ (w * np.log(w)))
    if c[2] >= 0:
        return np.nan, np.nan, len(w)
    v = -c[1] / (2 * c[2])
    if len(w) == 3:
        return v, np.nan, 3
    r = np.log(w) - A @ c
    s2 = float((w * r * r).sum()) / (len(w) - 3)
    g = np.array([0.0, -1.0 / (2 * c[2]), c[1] / (2 * c[2] ** 2)])
    return v, float(np.sqrt(max(g @ (s2 * Ni) @ g, 0.0))), len(w)


se = np.full(n, np.nan); npt = np.zeros(n, int)
for i in range(n):
    a = f_apex[i]
    s0, s1 = max(0, a - half), min(S, a + half + 1)
    y = eic_kernel(cloud.off, cloud.mz, cloud.inten, f_mz[i], f_tol[i], s0, s1)
    k = a - s0
    if k > 0 and y[k - 1] > y[k]:
        k -= 1
    elif k < len(y) - 1 and y[k + 1] > y[k]:
        k += 1
    _, e, m = fit(y, k, (cloud.rt[s0:s1] - cloud.rt[a]) / P.dt)
    se[i] = e * P.dt * 60 if np.isfinite(e) else np.nan
    npt[i] = m
tau_s = tau * 60
snr = F["snr"].values
print(f"{stem}: {n} features, dt {P.dt * 60:.3f} s, median FWHM {P.fwhm_med * 60:.2f} s, tolerance classes {info.get('tau_classes_s')} s at S/N edges {info.get('snr_edges')}")
print(f"  fit on 3 points (no standard error): {int((npt <= 3).sum())}; with a standard error: {int(np.isfinite(se).sum())}")
ok = np.isfinite(se)
edges = info.get("snr_edges", [])
cls = np.searchsorted(edges, snr, side="right")
fw = F["fwhm"].values / P.fwhm_med
rows = []
for c in range(len(edges) + 1):
    m = ok & (cls == c)
    if m.sum():
        rows.append(dict(snr_class=c, n=int(m.sum()), tau=round(float(np.median(tau_s[m])), 3), se_med=np.median(se[m]), se_p90=np.quantile(se[m], 0.9),
                         share_3se_gt_tau=(3 * se[m] > tau_s[m]).mean(), share_se_gt_tau=(se[m] > tau_s[m]).mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
rows = []
for a, b in ((0, 0.75), (0.75, 1.25), (1.25, 1.75), (1.75, 2.5), (2.5, 99)):
    m = ok & (fw >= a) & (fw < b)
    if m.sum():
        rows.append(dict(fwhm_rel=f"{a}-{b}", n=int(m.sum()), se_med=np.median(se[m]), se_p90=np.quantile(se[m], 0.9), share_3se_gt_tau=(3 * se[m] > tau_s[m]).mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
if win:
    m = np.flatnonzero((f_t * 60 >= win[0]) & (f_t * 60 <= win[1]))
    T = pd.DataFrame(dict(m=np.round(f_mz[m]).astype(int), t=(f_t[m] * 60).round(3), h=f_h[m].round(0), snr=snr[m].round(1), fwhm_s=(F["fwhm"].values[m] * 60).round(2),
                          points=npt[m], se=se[m].round(3), tau=tau_s[m].round(3)))
    print(T.sort_values("t").to_string(index=False))
