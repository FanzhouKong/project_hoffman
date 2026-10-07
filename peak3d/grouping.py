"""Cross-run feature grouping on the corrected RT axis.

Strongest-first seeding: features sorted by m/z; seeds taken in descending height claim, within
the m/z window and rt_tol, the closest-in-RT unassigned feature of every OTHER file. Released
same-file duplicates seed later groups, so isomer doublets become two groups."""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _group_kernel(mz, sig, rt, width, height, file_idx, lo, hi, order, ppm_tol, rt_tol, n_files):
    """lo/hi bound the m/z search window; the m/z pair test is max(ppm_tol, 3*sqrt(sig_s^2+sig_j^2))
    so weak features (large sigma) are allowed the scatter their intensity implies; the RT pair
    test is max(rt_tol, 0.5 * min(width_s, width_j)): apexes of one compound in two files lie
    within half the narrower peak's base width, however much a broad apex wanders"""
    n = len(rt)
    assigned = np.full(n, -1, np.int32)
    best_j = np.full(n_files, -1, np.int64)
    best_d = np.zeros(n_files)
    touched = np.empty(n_files, np.int64)
    gid = 0
    for s in order:
        if assigned[s] >= 0:
            continue
        assigned[s] = gid
        nt = 0
        fs = file_idx[s]
        for j in range(lo[s], hi[s]):
            if assigned[j] >= 0:
                continue
            f = file_idx[j]
            if f == fs:
                continue
            tol = max(ppm_tol, 3.0 * np.sqrt(sig[s] * sig[s] + sig[j] * sig[j]))
            if abs(mz[j] - mz[s]) / mz[s] * 1e6 > tol:
                continue
            drt = abs(rt[j] - rt[s])
            if drt > max(rt_tol, 0.5 * min(width[s], width[j])):
                continue
            if best_j[f] < 0:
                touched[nt] = f
                nt += 1
                best_j[f] = j
                best_d[f] = drt
            elif drt < best_d[f] or (drt == best_d[f] and height[j] > height[best_j[f]]):
                best_j[f] = j
                best_d[f] = drt
        for k in range(nt):
            f = touched[k]
            assigned[best_j[f]] = gid
            best_j[f] = -1
        gid += 1
    return assigned, gid


@njit(cache=True)
def _consensus_kernel(order, goff, mz, rt, rt_min, rt_max, height, G):
    c_mz = np.empty(G)
    c_rt = np.empty(G)
    c_rtmin = np.empty(G)
    c_rtmax = np.empty(G)
    c_n = np.empty(G, np.int64)
    c_spread = np.empty(G)
    c_rtsd = np.empty(G)
    for g in range(G):
        idx = order[goff[g]:goff[g + 1]]
        n = len(idx)
        c_n[g] = n
        # height-weighted median of mz
        m = mz[idx]
        h = height[idx]
        o = np.argsort(m)
        cw = np.cumsum(h[o])
        k = np.searchsorted(cw, 0.5 * cw[-1])
        if k >= n:
            k = n - 1
        c_mz[g] = m[o][k]
        c_rt[g] = np.median(rt[idx])
        c_rtmin[g] = np.median(rt_min[idx])
        c_rtmax[g] = np.median(rt_max[idx])
        c_spread[g] = (m.max() - m.min()) / c_mz[g] * 1e6
        c_rtsd[g] = rt[idx].std() if n > 1 else 0.0
    return c_mz, c_rt, c_rtmin, c_rtmax, c_n, c_spread, c_rtsd


def group_features(mz, rt_corr, rt_min_corr, rt_max_corr, height, file_idx, n_files, ppm_tol, rt_tol, sig=None):
    """Returns (gid int32[n], consensus dict). gid is dense 0..G-1; every feature gets a group.
    sig: per-feature m/z sigma in ppm (sqrt(a^2 + b^2/height) of its file); None = flat ppm_tol."""
    mz = np.asarray(mz, dtype=np.float64)
    rt_corr = np.asarray(rt_corr, dtype=np.float64)
    height = np.asarray(height, dtype=np.float64)
    file_idx = np.asarray(file_idx, dtype=np.int64)
    n = len(mz)
    sig = np.zeros(n) if sig is None else np.asarray(sig, dtype=np.float64)
    width = np.maximum(np.asarray(rt_max_corr, dtype=np.float64) - np.asarray(rt_min_corr, dtype=np.float64), 0.0)
    if n == 0:
        return np.zeros(0, np.int32), {k: np.zeros(0) for k in
                                       ("mz", "rt", "rt_min", "rt_max", "n_detected", "mz_ppm_spread", "rt_sd")}
    o = np.argsort(mz, kind="stable")
    smz = mz[o]
    ssig = sig[o]
    # search window per feature: its own widest possible pair tolerance (partner at the floor sigma)
    win = np.maximum(ppm_tol, 3.0 * np.sqrt(ssig ** 2 + (sig.max() if n else 0.0) ** 2))
    lo = np.searchsorted(smz, smz * (1 - win * 1e-6))
    hi = np.searchsorted(smz, smz * (1 + win * 1e-6), side="right")
    order = np.argsort(-height[o], kind="stable")
    assigned_s, G = _group_kernel(smz, ssig, rt_corr[o], width[o], height[o], file_idx[o], lo, hi, order,
                                  float(ppm_tol), float(rt_tol), int(n_files))
    gid = np.empty(n, np.int32)
    gid[o] = assigned_s
    gorder = np.argsort(gid, kind="stable")
    goff = np.zeros(G + 1, np.int64)
    np.cumsum(np.bincount(gid, minlength=G), out=goff[1:])
    c = _consensus_kernel(gorder, goff, mz, rt_corr, np.asarray(rt_min_corr, dtype=np.float64),
                          np.asarray(rt_max_corr, dtype=np.float64), height, G)
    cons = dict(mz=c[0], rt=c[1], rt_min=c[2], rt_max=c[3], n_detected=c[4], mz_ppm_spread=c[5], rt_sd=c[6])
    return gid, cons


def consensus(gid, mz, rt_corr, rt_min_corr, rt_max_corr, height):
    """consensus table for an arbitrary (dense) group assignment"""
    gid = np.asarray(gid)
    G = int(gid.max()) + 1 if len(gid) else 0
    gorder = np.argsort(gid, kind="stable")
    goff = np.zeros(G + 1, np.int64)
    np.cumsum(np.bincount(gid, minlength=G), out=goff[1:])
    c = _consensus_kernel(gorder, goff, np.asarray(mz, np.float64), np.asarray(rt_corr, np.float64),
                          np.asarray(rt_min_corr, np.float64), np.asarray(rt_max_corr, np.float64),
                          np.asarray(height, np.float64), G)
    return dict(mz=c[0], rt=c[1], rt_min=c[2], rt_max=c[3], n_detected=c[4], mz_ppm_spread=c[5], rt_sd=c[6])


def merge_complementary(gid, cons, file_idx, n_files, ppm_tol, rt_tol, g_sig):
    """Second pass over groups: a group absorbs weaker groups of the same ion (m/z within the
    pair tolerance), whose RT ranges overlap (apex within half the summed widths, at least rt_tol)
    and which share NO file with it. One compound whose apex wandered between files yields such
    disjoint fragments; two co-eluting isomers are present in the same files and stay separate.
    Non-transitive (only the seed's own window), so chains cannot run away. Returns new dense gid."""
    gid = np.asarray(gid)
    G = len(cons["mz"])
    if G == 0:
        return gid
    g_mz = np.asarray(cons["mz"]); g_rt = np.asarray(cons["rt"])
    g_w = np.maximum(np.asarray(cons["rt_max"]) - np.asarray(cons["rt_min"]), 0.0)
    g_n = np.asarray(cons["n_detected"])
    # file membership per group as a boolean matrix (G x n_files)
    member = np.zeros((G, n_files), bool)
    member[gid, np.asarray(file_idx)] = True
    o = np.argsort(g_mz, kind="stable"); smz = g_mz[o]
    target = np.arange(G)
    taken = np.zeros(G, bool)
    for a in np.argsort(-g_n, kind="stable"):
        if taken[a]:
            continue
        tol = max(ppm_tol, 3.0 * np.sqrt(2.0) * g_sig[a])
        lo = np.searchsorted(smz, g_mz[a] * (1 - 2 * tol * 1e-6)); hi = np.searchsorted(smz, g_mz[a] * (1 + 2 * tol * 1e-6), side="right")
        cand = o[lo:hi]
        cand = cand[(cand != a) & ~taken[cand] & (g_n[cand] <= g_n[a])]
        if len(cand) == 0:
            continue
        dppm = np.abs(g_mz[cand] - g_mz[a]) / g_mz[a] * 1e6
        pair_tol = np.maximum(ppm_tol, 3.0 * np.sqrt(g_sig[a] ** 2 + g_sig[cand] ** 2))
        drt = np.abs(g_rt[cand] - g_rt[a])
        rt_win = np.maximum(rt_tol, 0.5 * (g_w[a] + g_w[cand]))
        ok = (dppm <= pair_tol) & (drt <= rt_win)
        cand = cand[ok]
        if len(cand) == 0:
            continue
        shared = (member[cand] & member[a][None, :]).any(axis=1)
        cand = cand[~shared]
        for b in cand[np.argsort(drt[ok][~shared], kind="stable")]:
            if (member[b] & member[a]).any():   # a previous absorption may have filled that file
                continue
            member[a] |= member[b]
            target[b] = a
            taken[b] = True
        taken[a] = True
    new = target[gid]
    uniq, dense = np.unique(new, return_inverse=True)
    return dense.astype(np.int32)
