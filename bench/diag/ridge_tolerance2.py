# diagnostic: ridge level statistic (quantile) x m/z window, on kept features of one BM21 RP file
import sys, json
import numpy as np, pandas as pd
from numba import njit, prange
sys.path.insert(0, "/quobyte/metabolomicsgrp/fanzhou/hoffmann")
from peak3d.io import load_cloud
from peak3d.features import RIDGE_NEAR, RIDGE_FAR
ROOT = "/quobyte/metabolomicsgrp/fanzhou/hoffmann"
stem = "MT_20211013_082"

@njit(cache=False, parallel=True)
def window_q(off, mz, inten, q_mz, q_tol, s_lo, s_hi, q):
    n = len(q_mz); S = len(off) - 1
    out = np.zeros(n)
    for i in prange(n):
        a0 = max(0, s_lo[i]); a1 = min(S, s_hi[i])
        if a1 <= a0: continue
        vals = np.zeros(a1 - a0)
        for s in range(a0, a1):
            a = off[s]; b = off[s + 1]; lo = a; hi = b; v = q_mz[i] - q_tol[i]
            while lo < hi:
                m = (lo + hi) >> 1
                if mz[m] < v: lo = m + 1
                else: hi = m
            v2 = q_mz[i] + q_tol[i]; best = 0.0; p = lo
            while p < b and mz[p] <= v2:
                if inten[p] > best: best = inten[p]
                p += 1
            vals[s - a0] = best
        vals.sort()
        out[i] = vals[min(len(vals) - 1, int(q * (len(vals) - 1) + 0.5))]
    return out

f = pd.read_csv(f"{ROOT}/results/peak3d/BM21_RP/peak3d_out/features/{stem}.tsv", sep="\t")
P = json.load(open(f"{ROOT}/results/peak3d/BM21_RP/peak3d_out/features/{stem}.params.json"))
cloud = load_cloud(f"{ROOT}/data/mzml/BM21/{stem}.mzML")
sig = np.sqrt(P["sig_a"] ** 2 + P["sig_b"] ** 2 / f["height"].values)
w = np.maximum(f["fwhm"].values, P["fwhm_med"]) / P["dt"]
sa = f["scan_apex"].values.astype(np.int64)
near = np.maximum(1, np.round(RIDGE_NEAR * w)).astype(np.int64); far = np.maximum(near + 2, np.round(RIDGE_FAR * w)).astype(np.int64)
q_mz = f["mz"].values.astype(np.float64)
bg = [89.0266, 65.0143, 100.9296, 100.9355, 91.0051, 99.0086]
is_bg = np.zeros(len(f), bool)
for m in bg: is_bg |= np.abs(q_mz - m) / m * 1e6 <= 10
good = (f["gauss_r2"] > 0.9) & (f["snr"] > 50) & (f["n_gaps"] == 0) & ~is_bg
print(f"{len(f)} kept; {is_bg.sum()} at known ragged background m/z; {good.sum()} clean strong peaks (r2>0.9, snr>50)")
print("statistic  window     flagged_all  flagged_bg  flagged_clean_strong")
for q in (0.5, 0.75, 0.9):
    for name, tol_ppm in [("3s+1", 3 * sig + 1), ("max5", np.maximum(3 * sig + 1, 5.0)), ("max8", np.maximum(3 * sig + 1, 8.0))]:
        q_tol = q_mz * tol_ppm * 1e-6
        left = window_q(cloud.off, cloud.mz, cloud.inten, q_mz, q_tol, sa - far, sa - near + 1, q)
        right = window_q(cloud.off, cloud.mz, cloud.inten, q_mz, q_tol, sa + near, sa + far + 1, q)
        rr = np.minimum(left, right) / f["height"].values
        print(f"  q={q:.2f}   {name:6s}   {np.mean(rr > 0.5):6.1%} ({int((rr > 0.5).sum()):4d})   {np.mean(rr[is_bg] > 0.5):6.1%}      {np.mean(rr[good] > 0.5):6.1%}")

# signal-to-background: lower-quartile side level (both sides) must stay below 1/3 of the apex
q_tol = q_mz * np.maximum(3 * sig + 1, 5.0) * 1e-6
l25 = window_q(cloud.off, cloud.mz, cloud.inten, q_mz, q_tol, sa - far, sa - near + 1, 0.25)
r25 = window_q(cloud.off, cloud.mz, cloud.inten, q_mz, q_tol, sa + near, sa + far + 1, 0.25)
l75 = window_q(cloud.off, cloud.mz, cloud.inten, q_mz, q_tol, sa - far, sa - near + 1, 0.75)
r75 = window_q(cloud.off, cloud.mz, cloud.inten, q_mz, q_tol, sa + near, sa + far + 1, 0.75)
bg = np.minimum(l25, r25) / f["height"].values
rg = np.minimum(l75, r75) / f["height"].values
print("\nsignal-to-background (q25 side level / apex), window max(3s+1, 5 ppm):")
for thr in (0.2, 0.25, 0.33, 0.5):
    fl = bg > thr
    print(f"  bg/apex > {thr:.2f}: flagged {np.mean(fl):.1%} ({fl.sum()}), background ions {np.mean(fl[is_bg]):.1%}, clean strong {np.mean(fl[good]):.1%}")
for thr in (0.25, 0.33):
    fl = (bg > thr) | (rg > 0.5)
    print(f"  union with q75 > 0.5, bg > {thr:.2f}: flagged {np.mean(fl):.1%} ({fl.sum()}), background ions {np.mean(fl[is_bg]):.1%}, clean strong {np.mean(fl[good]):.1%}")
fl = (bg > 0.33) | (rg > 0.5)
print(f"flagged set: m/z median {np.median(q_mz[fl]):.1f}, below m/z 150: {np.mean(q_mz[fl] < 150):.0%}, height median {np.median(f['height'][fl]):.0f}; kept: m/z median {np.median(q_mz[~fl]):.1f}")
print("clean-strong features that would be flagged:", [(round(m, 4), round(t, 2), int(h), round(b_, 2), round(r_, 2)) for m, t, h, b_, r_ in zip(q_mz[good & fl], f['rt'][good & fl], f['height'][good & fl], bg[good & fl], rg[good & fl])][:8])
