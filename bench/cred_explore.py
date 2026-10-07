#!/usr/bin/env python3
"""Describe the credentialing raw data before choosing any tolerance.

Per file: MS1 scans, RT range, scan interval, centroids/scan, intensity floor.
Per dataset: FWHM of the 300 most intense distinct ions (12C rep 1), and the RT shift
between each file and 12C rep 1, measured on those same ions (12C files) or on their
fully labeled partners (13C files; strongest partner within +/-1 min over n carbons).
usage: cred_explore.py DATASET   (SZ22 | YEAST)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.msdata import C13, load_run  # noqa: E402
from bench.runs import RUNS  # noqa: E402

ds = sys.argv[1]
f12 = RUNS[f"{ds}_12C"]["files"]
f13 = RUNS[f"{ds}_13C"]["files"]
runs12 = [load_run(f) for f in f12]
runs13 = [load_run(f) for f in f13]

print(f"== {ds}")
for r in runs12 + runs13:
    per_scan = np.bincount(r.scan, minlength=r.n_scans)
    print(f"{r.name:45s} scans={r.n_scans:5d} rt={r.rt[0]:.2f}-{r.rt[-1]:.2f} min "
          f"dt={np.median(np.diff(r.rt)) * 60:.2f}s centroids/scan={np.median(per_scan):.0f} "
          f"int min={r.inten.min():.0f} p1={np.percentile(r.inten, 1):.0f} "
          f"p50={np.median(r.inten):.0f}")

ref = runs12[0]
# 300 most intense distinct ions (>= 20 ppm apart)
order = np.argsort(ref.inten)[::-1]
seeds = []
for j in order:
    m = ref.mz[j]
    if all(abs(m - s[0]) / m > 20e-6 for s in seeds):
        seeds.append((m, ref.rt[ref.scan[j]]))
    if len(seeds) >= 300:
        break


def apex_fwhm(run, m, rt0, half=1.0):
    a, b = run.scan_window(rt0, half)
    e = run.eic(m)[a:b]
    if e.max() <= 0:
        return None
    k = int(np.argmax(e)); h = e[k] / 2
    l = k
    while l > 0 and e[l - 1] >= h:
        l -= 1
    rr = k
    while rr < len(e) - 1 and e[rr + 1] >= h:
        rr += 1
    t = run.rt[a:b]
    return t[k], t[rr] - t[l], e[k]


fw = [x[1] for m, t in seeds if (x := apex_fwhm(ref, m, t))]
print(f"FWHM of top ions (12C rep1): median {np.median(fw) * 60:.1f}s, "
      f"IQR {np.percentile(fw, 25) * 60:.1f}-{np.percentile(fw, 75) * 60:.1f}s, n={len(fw)}")

for r in runs12[1:]:
    d = [x[0] - t for m, t in seeds if (x := apex_fwhm(r, m, t))]
    print(f"RT shift {r.name} vs 12C rep1: median {np.median(d) * 60:+.1f}s, "
          f"IQR {np.percentile(d, 25) * 60:+.1f}..{np.percentile(d, 75) * 60:+.1f}s")
for r in runs13:
    d = []
    for m, t in seeds:
        best = None
        for n in range(1, int(m // 12) + 1):
            x = apex_fwhm(r, m + n * C13, t)
            if x and (best is None or x[2] > best[2]):
                best = x
        if best and best[2] > 1e5:
            d.append(best[0] - t)
    d = np.asarray(d)
    close = d[np.abs(d) < 0.5]
    print(f"RT shift {r.name} vs 12C rep1 (labeled partners): median {np.median(close) * 60:+.1f}s, "
          f"IQR {np.percentile(close, 25) * 60:+.1f}..{np.percentile(close, 75) * 60:+.1f}s, "
          f"n={len(close)}/{len(d)} within 30 s")
