"""Isotope annotation of a per-file feature table (M+n partners, charge state).

Partners are searched at mz + n * C13 / z for z in 1..3, n in 1..3, co-eluting within
max(dt, 0.25 * FWHM) and with a plausible intensity ratio; the two per-scan-max traces must
correlate (r >= 0.7). Isotopic features are annotated, never removed."""
from __future__ import annotations

import numpy as np

from .estimate import Params
from .io import C13, Cloud

R_MIN = 0.7
RATIO_LO, RATIO_HI = 0.005, 1.5
MAX_Z, MAX_N = 3, 3


def _corr(a, b):
    m = (a > 0) | (b > 0)
    if m.sum() < 3:
        return 0.0
    a, b = a[m].astype(np.float64), b[m].astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def annotate_isotopes(mz, rt, height, scan_lo, scan_hi, cloud: Cloud, P: Params):
    """Returns (iso_offset, charge, iso_parent, iso_ratio) arrays aligned with the inputs.
    iso_parent holds the row index of the monoisotopic feature (-1 for none)."""
    n = len(mz)
    iso_offset = np.zeros(n, np.int32)
    charge = np.ones(n, np.int32)
    iso_parent = np.full(n, -1, np.int64)
    iso_ratio = np.full(n, np.nan)
    if n == 0:
        return iso_offset, charge, iso_parent, iso_ratio
    order = np.argsort(mz, kind="stable")
    smz = mz[order]
    sig = P.sig_ppm(height)
    rt_tol = max(P.dt, 0.25 * P.fwhm_med)
    taken = np.zeros(n, bool)
    cache = {}

    def trace(i, s0, s1):
        key = (i, s0, s1)
        if key not in cache:
            cache[key] = cloud.eic(mz[i], 3.0 * sig[i] + 1.0, s0, s1)
        return cache[key]

    for i in np.argsort(-height, kind="stable"):   # strongest parents claim partners first
        if taken[i]:
            continue
        best_z, found = 0, {}
        for z in range(1, MAX_Z + 1):
            got = {}
            for k in range(1, MAX_N + 1):
                target = mz[i] + k * C13 / z
                lo = np.searchsorted(smz, target * (1 - 2e-5))
                hi = np.searchsorted(smz, target * (1 + 2e-5), side="right")
                best, best_r = -1, R_MIN
                for j in order[lo:hi]:
                    if j == i or taken[j]:
                        continue
                    tol = 3.0 * np.sqrt(sig[i] ** 2 + sig[j] ** 2)
                    if abs(mz[j] - target) / target * 1e6 > tol:
                        continue
                    if abs(rt[j] - rt[i]) > rt_tol:
                        continue
                    ratio = height[j] / height[i]
                    if not (RATIO_LO <= ratio <= RATIO_HI):
                        continue
                    s0 = int(min(scan_lo[i], scan_lo[j]))
                    s1 = int(max(scan_hi[i], scan_hi[j])) + 1
                    r = _corr(trace(i, s0, s1), trace(j, s0, s1))
                    if r >= best_r:
                        best, best_r = j, r
                if best < 0:
                    break            # M+k missing: do not look for M+(k+1) at this z
                got[k] = best
            if len(got) > len(found):
                best_z, found = z, got
        if found:
            charge[i] = best_z
            for k, j in found.items():
                iso_offset[j] = k
                charge[j] = best_z
                iso_parent[j] = i
                iso_ratio[j] = height[j] / height[i]
                taken[j] = True
            taken[i] = True
    return iso_offset, charge, iso_parent, iso_ratio
