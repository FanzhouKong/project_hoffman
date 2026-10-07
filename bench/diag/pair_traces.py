# diagnostic: raw EIC traces around same-ion feature pairs in one file (is it one rippled peak or two peaks?)
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "/quobyte/metabolomicsgrp/fanzhou/hoffmann")
from peak3d.io import load_cloud
ROOT = "/quobyte/metabolomicsgrp/fanzhou/hoffmann"
run, stem, mzml = "BM21_RP", "MT_20211013_082", f"{ROOT}/data/mzml/BM21/MT_20211013_082.mzML"
f = pd.read_csv(f"{ROOT}/results/peak3d/{run}/peak3d_out/features/{stem}.tsv", sep="\t")
P = json.load(open(f"{ROOT}/results/peak3d/{run}/peak3d_out/features/{stem}.params.json"))
cloud = load_cloud(mzml)
print("scans", cloud.n_scans, "dt_s %.3f" % (60 * P["dt"]), "FWHM_med_s %.2f" % (60 * P["fwhm_med"]))
mz, rt, h = f["mz"].values, f["rt"].values, f["height"].values
o = np.argsort(mz); smz = mz[o]
pairs = []
for i in range(len(f)):
    lo = np.searchsorted(smz, mz[i] * (1 - 5e-6)); hi = np.searchsorted(smz, mz[i] * (1 + 5e-6), side="right")
    for j in o[lo:hi]:
        if j > i and abs(rt[j] - rt[i]) * 60 <= 6 and abs(mz[j] - mz[i]) / mz[i] * 1e6 < 2:
            pairs.append((i, j) if h[i] >= h[j] else (j, i))
rng = np.random.default_rng(0)
pick = [pairs[k] for k in rng.choice(len(pairs), 8, replace=False)]
pick.sort(key=lambda p: -h[p[0]])
for a, b in pick:
    s0 = int(np.searchsorted(cloud.rt, min(rt[a], rt[b]) - 8 / 60)); s1 = int(np.searchsorted(cloud.rt, max(rt[a], rt[b]) + 8 / 60))
    e = cloud.eic(float(mz[a]), 10.0, s0, s1)
    top = e.max()
    bar = "".join("#" if v >= 0.9 * top else ("=" if v >= 0.6 * top else ("-" if v >= 0.3 * top else ("." if v > 0 else " "))) for v in e)
    ia, ib = int(f.scan_apex[a]) - s0, int(f.scan_apex[b]) - s0
    marks = [" "] * len(e); marks[ia] = "A"; marks[ib] = "B"
    print(f"\nmz {mz[a]:.4f}  strong rt {rt[a]:.3f} h {h[a]:.0f} fwhm {60*f.fwhm[a]:.1f}s | weak rt {rt[b]:.3f} h {h[b]:.0f} fwhm {60*f.fwhm[b]:.1f}s  drt {60*(rt[b]-rt[a]):+.2f}s  score_b {f.score[b]:.2f}")
    print("  " + "".join(marks))
    print("  " + bar)
    print("  rel: " + " ".join(f"{v/top:.2f}" for v in e[max(0, min(ia, ib) - 6):max(ia, ib) + 7]))
