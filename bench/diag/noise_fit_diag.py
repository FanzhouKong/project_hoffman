# diagnostic: why the per-file noise law (estimate.fit_intensity_noise) fits r = 8-15 % on three MassCube Orbitrap
# files when every other benchmark file gives 2.6-6 %. Same seed traces as estimate(); residuals three ways:
#   sg_index   the current estimator: 5-point Savitzky-Golay quadratic in scan index, linear intensity
#   q_time     leave-one-out quadratic through the 4 neighbours in actual scan time, linear intensity
#   q_logtime  the same on log intensity (a Gaussian peak is exactly a parabola in log space at any width or
#              sampling, so the residual is noise, not curvature); residual converted with I x d(log I)
# Each residual is divided by its own sd factor under white noise, then the same law fit (fit_noise_law).
# Also per file: MS1 spacing CV, median FWHM (scans) of the seed traces, share of seeds narrower than 5 scans.
# usage: noise_fit_diag.py RUN[:FILE_INDEX] ...
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.runs import RUNS  # noqa: E402
from peak3d.estimate import (SEED_PPM, estimate, fit_intensity_noise, fit_noise_law, fwhm_at,  # noqa: E402
                             strongest_distinct)
from peak3d.io import load_cloud  # noqa: E402


def loo_quadratic(t, y, j):
    """prediction at t[j] of the quadratic through the 4 points j-2, j-1, j+1, j+2, and the sd factor of
    (y[j] - prediction) for white noise of unit sd: sqrt(1 + sum of squared prediction weights)"""
    idx = np.array([j - 2, j - 1, j + 1, j + 2])
    x = t[idx] - t[j]
    X = np.column_stack([np.ones(4), x, x * x])
    w = np.linalg.pinv(X)[0]            # prediction at x = 0 is the intercept: w @ y[idx]
    return float(w @ y[idx]), float(np.sqrt(1.0 + np.sum(w * w)))


def residuals(cloud, seeds, scan_of, reach, top_frac=0.1, min_pts=7):
    S = cloud.n_scans
    out = {"q_time": ([], []), "q_logtime": ([], [])}
    widths = []
    for p, s, rch in zip(seeds, scan_of, reach):
        s0, s1 = max(0, s - int(rch)), min(S, s + int(rch) + 1)
        e = cloud.eic(cloud.mz[p], SEED_PPM, s0, s1).astype(np.float64)
        t = cloud.rt[s0:s1]
        if len(e) < min_pts:
            continue
        k = int(np.argmax(e))
        thr = max(top_frac * e[k], 1e-9)
        lo, hi = k, k
        while lo - 1 >= 0 and e[lo - 1] >= thr:
            lo -= 1
        while hi + 1 < len(e) and e[hi + 1] >= thr:
            hi += 1
        if hi - lo + 1 < min_pts:
            continue
        above = e[lo:hi + 1] >= 0.5 * e[k]
        widths.append(int(above.sum()))
        for j in range(lo + 2, hi - 1):
            if j == k:
                continue
            pl, fl = loo_quadratic(t, e, j)
            if pl > 0:
                out["q_time"][0].append(pl)
                out["q_time"][1].append((e[j] - pl) / fl)
            pg, fg = loo_quadratic(t, np.log(e), j)
            lev = np.exp(pg)
            out["q_logtime"][0].append(lev)
            out["q_logtime"][1].append(lev * (np.log(e[j]) - pg) / fg)
    return {k: (np.asarray(a), np.asarray(b)) for k, (a, b) in out.items()}, np.asarray(widths)


def main(specs):
    print(f"{'file':52s} {'dtCV':>5s} {'fwhm':>5s} {'<5sc':>5s} | {'sg_index r':>10s} {'c':>8s} | {'q_time r':>9s} {'c':>8s} | "
          f"{'q_logtime r':>11s} {'c':>8s} {'s0':>8s}")
    for spec in specs:
        run, _, fi = spec.partition(":")
        f = RUNS[run]["files"][int(fi or 0)]
        cloud = load_cloud(f)
        P = estimate(cloud)
        seeds = strongest_distinct(cloud, 2000)
        scan_of = np.searchsorted(cloud.off, seeds, side="right") - 1
        reach = np.full(len(seeds), 8.0)
        for i, (p, s) in enumerate(zip(seeds, scan_of)):
            r = fwhm_at(cloud, cloud.mz[p], int(s))
            if r is not None:
                reach[i] = max(8.0, 3.0 * r[0] / P.dt)
        reach = np.where(reach == 8.0, max(8.0, 3.0 * P.fwhm_scans), reach)
        a = fit_intensity_noise(cloud, seeds, scan_of, reach)
        R, widths = residuals(cloud, seeds, scan_of, reach)
        b = fit_noise_law(*R["q_time"])
        c = fit_noise_law(*R["q_logtime"])
        d = np.diff(cloud.rt)
        print(f"{run + ':' + Path(f).stem[:20]:52s} {d.std() / d.mean():5.2f} {np.median(widths):5.1f} {np.mean(widths < 5) * 100:4.0f}% | "
              f"{a[2]:10.3f} {a[1]:8.3g} | {b[2]:9.3f} {b[1]:8.3g} | {c[2]:11.3f} {c[1]:8.3g} {c[0]:8.3g}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
