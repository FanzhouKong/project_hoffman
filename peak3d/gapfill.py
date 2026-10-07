"""Gap filling on each file's native axis: for groups a file lacks, the EIC box maximum (height)
and trapezoid (area) inside the inverse-warped consensus window, and whether that window holds a
confirmed chromatographic peak (a local maximum of the smoothed EIC with prominence >= k x the
larger of the trace's own residual noise and the group's reference noise, at least ``min_top_frac``
of the group's reference height). A confirmed cell counts as presence of the group in this file even
though the picker did not call it: weak peaks near the cut are picked in some injections only."""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from .io import Cloud, _lower_bound, _upper_bound
from .kernels import gaussian_kernel
from .warp import TableWarp


@njit(cache=True, parallel=True)
def _fill_kernel(off, mz, inten, rt, g_mz, g_tol, s0, s1, ctx, kern, k_snr, noise_ref, h_ref, min_top_frac):
    n = len(g_mz)
    S = len(off) - 1
    K2 = len(kern) // 2
    height = np.zeros(n, np.float32)
    area = np.zeros(n, np.float32)
    confirmed = np.zeros(n, np.int8)
    for g in prange(n):
        a0 = s0[g]
        a1 = s1[g]
        if a1 <= a0:
            continue
        best = 0.0
        acc = 0.0
        prev_v = -1.0
        prev_t = 0.0
        for s in range(a0, a1):
            a = off[s]
            b = off[s + 1]
            lo = _lower_bound(mz, a, b, g_mz[g] - g_tol[g])
            hi = _upper_bound(mz, lo, b, g_mz[g] + g_tol[g])
            v = 0.0
            for p in range(lo, hi):
                if inten[p] > v:
                    v = inten[p]
            if v > 0:
                if v > best:
                    best = v
                if prev_v >= 0:
                    acc += 0.5 * (v + prev_v) * (rt[s] - prev_t)
                prev_v = v
                prev_t = rt[s]
        height[g] = best
        area[g] = acc
        if best <= 0.0 or best < min_top_frac * h_ref[g] or ctx <= 0:
            continue
        # confirmation: smoothed EIC over the window plus context on both sides
        c0 = max(0, a0 - ctx)
        c1 = min(S, a1 + ctx)
        m = c1 - c0
        if m < 3:
            continue
        e = np.zeros(m, np.float64)
        for s in range(c0, c1):
            a = off[s]
            b = off[s + 1]
            lo = _lower_bound(mz, a, b, g_mz[g] - g_tol[g])
            hi = _upper_bound(mz, lo, b, g_mz[g] + g_tol[g])
            v = 0.0
            for p in range(lo, hi):
                if inten[p] > v:
                    v = inten[p]
            e[s - c0] = v
        es = np.zeros(m, np.float64)
        for k in range(m):
            v = 0.0
            for j in range(-K2, K2 + 1):
                kk = k + j
                if kk < 0:
                    kk = 0
                elif kk >= m:
                    kk = m - 1
                v += kern[j + K2] * e[kk]
            es[k] = v
        b0 = a0 - c0
        b1 = a1 - c0 - 1
        # highest local maximum of the smoothed trace inside the consensus window
        k = -1
        for j in range(max(b0, 1), min(b1, m - 2) + 1):
            if es[j] >= es[j - 1] and es[j] >= es[j + 1] and (k < 0 or es[j] > es[k]):
                k = j
        if k < 0:
            continue
        top = es[k]
        if top < min_top_frac * h_ref[g]:
            continue
        j = k
        lmin = top
        while j > 0 and es[j - 1] <= top:
            j -= 1
            if es[j] < lmin:
                lmin = es[j]
        j = k
        rmin = top
        while j < m - 1 and es[j + 1] <= top:
            j += 1
            if es[j] < rmin:
                rmin = es[j]
        prom = top - max(lmin, rmin)
        cnt = 0
        for j in range(m):
            if j < b0 or j > b1:
                cnt += 1
        if cnt >= 10:
            r = np.empty(cnt, np.float64)
            c = 0
            for j in range(m):
                if j < b0 or j > b1:
                    r[c] = e[j] - es[j]
                    c += 1
        else:
            r = e - es
        med = np.median(r)
        noise = 1.4826 * np.median(np.abs(r - med))
        if prom >= k_snr * max(noise, noise_ref[g]):
            confirmed[g] = 1
    return height, area, confirmed


def fill_file(cloud: Cloud, warp: TableWarp, g_mz, g_rt, g_half, ppm_tol: float, noise_ref=None, h_ref=None,
              fwhm_scans: float = 0.0, smooth_fwhm: float = 0.5, half_win_fwhm: float = 10.0, k_snr: float = 3.0,
              min_top_frac: float = 0.05):
    """heights, areas and confirmation flags for the given consensus groups in this file (all groups
    passed in). Confirmation needs noise_ref / h_ref per group and the file's peak width in scans;
    without them only heights and areas are computed (confirmed all 0)."""
    g_mz = np.asarray(g_mz, dtype=np.float64)
    g_rt = np.asarray(g_rt, dtype=np.float64)
    g_half = np.asarray(g_half, dtype=np.float64)
    lo_t = warp.inverse(g_rt - g_half)
    hi_t = warp.inverse(g_rt + g_half)
    s0 = np.searchsorted(cloud.rt, lo_t).astype(np.int64)
    s1 = np.searchsorted(cloud.rt, hi_t, side="right").astype(np.int64)
    tol = g_mz * ppm_tol * 1e-6
    n = len(g_mz)
    if noise_ref is None or h_ref is None or fwhm_scans <= 0:
        kern = np.ones(1)
        ctx = 0
        noise_ref = np.zeros(n)
        h_ref = np.zeros(n)
    else:
        kern = gaussian_kernel(smooth_fwhm * fwhm_scans)
        ctx = int(max(3, round(half_win_fwhm * fwhm_scans)))
        noise_ref = np.asarray(noise_ref, dtype=np.float64)
        h_ref = np.asarray(h_ref, dtype=np.float64)
    return _fill_kernel(cloud.off, cloud.mz, cloud.inten, cloud.rt, g_mz, tol, s0, s1, ctx, kern, float(k_snr),
                        noise_ref, h_ref, float(min_top_frac))
