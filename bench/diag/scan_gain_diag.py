# diagnostic: is the scan-to-scan noise of strong ion traces shared by every ion of a scan (a scan-wide gain, e.g.
# AGC / injection-time error, spray flicker) or independent per ion? Residuals of each seed trace from the
# leave-one-out quadratic of log intensity in actual time (noise_fit_diag.loo_quadratic); per scan the median
# residual over all traces covering it = that scan's log gain. Reports the robust sd of the residuals before
# and after removing the per-scan gain, the sd of the gain itself, and the noise law r fitted after removal.
# usage: scan_gain_diag.py RUN[:FILE_INDEX] ...
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/diag"))
from bench.runs import RUNS  # noqa: E402
from noise_fit_diag import loo_quadratic  # noqa: E402
from peak3d.estimate import SEED_PPM, estimate, fit_noise_law, strongest_distinct  # noqa: E402
from peak3d.io import load_cloud  # noqa: E402


def rsd(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def main(specs):
    print(f"{'file':44s} {'r_before':>8s} {'r_after':>8s} {'gain_sd':>8s} {'scans':>6s} {'traces/scan':>11s} {'lag1':>6s}")
    for spec in specs:
        run, _, fi = spec.partition(":")
        cloud = load_cloud(RUNS[run]["files"][int(fi or 0)])
        P = estimate(cloud)
        seeds = strongest_distinct(cloud, 2000)
        scan_of = np.searchsorted(cloud.off, seeds, side="right") - 1
        reach = int(max(8, 3 * P.fwhm_scans))
        S = cloud.n_scans
        sc, lev, res, tid = [], [], [], []
        for i, (p, s) in enumerate(zip(seeds, scan_of)):
            s0, s1 = max(0, s - reach), min(S, s + reach + 1)
            e = cloud.eic(cloud.mz[p], SEED_PPM, s0, s1).astype(np.float64)
            t = cloud.rt[s0:s1]
            k = int(np.argmax(e))
            lo, hi = k, k
            while lo - 1 >= 0 and e[lo - 1] >= 0.1 * e[k]:
                lo -= 1
            while hi + 1 < len(e) and e[hi + 1] >= 0.1 * e[k]:
                hi += 1
            for j in range(lo + 2, hi - 1):
                if j == k:
                    continue
                pg, fg = loo_quadratic(t, np.log(e), j)
                sc.append(s0 + j); lev.append(np.exp(pg)); res.append((np.log(e[j]) - pg) / fg); tid.append(i)
        sc, lev, res, tid = map(np.asarray, (sc, lev, res, tid))
        strong = lev > np.percentile(lev, 50)                  # the proportional-noise regime
        gain = np.zeros(S)
        cnt = np.bincount(sc[strong], minlength=S)
        for s in np.flatnonzero(cnt >= 5):
            gain[s] = np.median(res[strong & (sc == s)])
        res2 = res - gain[sc]
        # lag-1 correlation of consecutive residuals within a trace (alternation would make it strongly negative)
        o = np.lexsort((sc, tid))
        same = (tid[o][1:] == tid[o][:-1]) & (sc[o][1:] == sc[o][:-1] + 1)
        lag1 = np.corrcoef(res[o][:-1][same], res[o][1:][same])[0, 1]
        a = fit_noise_law(lev, lev * res)
        b = fit_noise_law(lev, lev * res2)
        used = cnt >= 5
        print(f"{run[:44]:44s} {a[2]:8.3f} {b[2]:8.3f} {rsd(gain[used]):8.3f} {used.sum():6d} {np.median(cnt[used]):11.0f} {lag1:6.2f}",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
