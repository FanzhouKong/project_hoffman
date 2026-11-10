"""Isomer rows: two groups at the same m/z that the table proves to be different compounds, and the RT range each may
gap-fill from.

Gap filling reads the highest point in a window of the group's RT +- half its peak width. Two isomers a few seconds
apart share that window, so the group of an isomer missing from an injection is filled with its neighbour's peak. The
table itself says when a neighbouring group is another compound: both were picked as separate peaks in the same
injection, often, always in the same order. Such a pair is "proven"; each of its groups then owns the RT range up to
the midpoint between the two compounds' positions, and its gap fill reads only inside that range. Groups without a
proven partner are filled exactly as before.

Proven pair (all of):
  - same m/z within the grouping tolerance, fill windows that reach each other (RT gap <= the two half-windows)
  - picked in the same injection in >= min_codetected injections and in >= min_share of the injections where the
    rarer of the two is picked (one compound split into two peaks in a few injections is not proven)
  - the same elution order in >= ORDER_MIN of those injections, median apex gap >= the grouping RT tolerance
Position of a group: the median apex over the injections where it was picked together with a proven partner (the
split, unambiguous injections). Proven pairs chain (three isomers); the midpoints between neighbouring positions are
the boundaries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

ORDER_MIN = 0.9


@njit(cache=True)
def _codetect(off, files, rts, pi, pj):
    """per candidate pair: injections with both picked, how many with j later than i, median |apex gap|"""
    n = len(pi)
    co = np.zeros(n, np.int64)
    later = np.zeros(n, np.int64)
    gap = np.zeros(n)
    for q in range(n):
        a0, a1 = off[pi[q]], off[pi[q] + 1]
        b0, b1 = off[pj[q]], off[pj[q] + 1]
        buf = np.empty(min(a1 - a0, b1 - b0))
        a, b, m = a0, b0, 0
        while a < a1 and b < b1:
            if files[a] < files[b]:
                a += 1
            elif files[a] > files[b]:
                b += 1
            else:
                d = rts[b] - rts[a]
                if d > 0:
                    later[q] += 1
                buf[m] = abs(d)
                m += 1
                a += 1
                b += 1
        co[q] = m
        if m > 0:
            gap[q] = np.median(buf[:m])
    return co, later, gap


def _csr(gid, file_idx, rt, G):
    o = np.lexsort((file_idx, gid))
    off = np.zeros(G + 1, np.int64)
    np.cumsum(np.bincount(gid, minlength=G), out=off[1:])
    return off, np.ascontiguousarray(file_idx[o], np.int64), np.ascontiguousarray(rt[o], np.float64)


def proven_pairs(gid, file_idx, rt, g_mz, g_rt, g_half, ppm_tol: float, rt_tol: float, min_codetected: int = 3,
                 min_share: float = 0.5) -> pd.DataFrame:
    """gid / file_idx / rt: one row per picked peak (its group, injection index, corrected apex RT); g_mz / g_rt /
    g_half: per group (dense ids) consensus m/z, RT and gap-fill half-window. Returns one row per proven pair
    (a < b): n_codetected, share, gap_s (median apex gap, seconds)."""
    gid = np.asarray(gid, np.int64)
    file_idx = np.asarray(file_idx, np.int64)
    G = len(g_mz)
    cols = ["a", "b", "n_codetected", "share", "gap_s"]
    if G < 2 or len(gid) == 0:
        return pd.DataFrame(columns=cols)
    g_mz, g_rt, g_half = (np.asarray(x, np.float64) for x in (g_mz, g_rt, g_half))
    om = np.argsort(g_mz, kind="stable")
    smz = g_mz[om]
    lo = np.searchsorted(smz, g_mz * (1 - ppm_tol * 1e-6))
    hi = np.searchsorted(smz, g_mz * (1 + ppm_tol * 1e-6), side="right")
    cnt = hi - lo
    i = np.repeat(np.arange(G), cnt)
    j = om[lo[i] + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))]
    d = np.abs(g_rt[j] - g_rt[i])
    keep = (i < j) & (d > 0) & (d <= g_half[i] + g_half[j])
    i, j = i[keep], j[keep]
    n_det = np.bincount(gid, minlength=G)
    keep = np.minimum(n_det[i], n_det[j]) >= min_codetected
    i, j = i[keep], j[keep]
    if not len(i):
        return pd.DataFrame(columns=cols)
    off, files, rts = _csr(gid, file_idx, np.asarray(rt, np.float64), G)
    co, later, gap = _codetect(off, files, rts, i.astype(np.int64), j.astype(np.int64))
    share = co / np.maximum(np.minimum(n_det[i], n_det[j]), 1)
    order = np.maximum(later, co - later) / np.maximum(co, 1)
    ok = (co >= min_codetected) & (share >= min_share) & (order >= ORDER_MIN) & (gap >= rt_tol)
    return pd.DataFrame({"a": i[ok], "b": j[ok], "n_codetected": co[ok], "share": share[ok], "gap_s": 60 * gap[ok]})


def territories(pairs: pd.DataFrame, gid, file_idx, rt, G: int):
    """(lo, hi, pos): per group the RT range it may gap-fill from (+-inf when it has no proven partner) and its
    position (NaN without a partner)"""
    lo = np.full(G, -np.inf)
    hi = np.full(G, np.inf)
    pos = np.full(G, np.nan)
    if not len(pairs):
        return lo, hi, pos
    off, files, rts = _csr(np.asarray(gid, np.int64), np.asarray(file_idx, np.int64), np.asarray(rt, np.float64), G)
    a, b = pairs["a"].to_numpy(np.int64), pairs["b"].to_numpy(np.int64)
    # injections where each group was picked together with one of its proven partners
    together = {}
    for x, y in zip(a, b):
        fx, fy = files[off[x]:off[x + 1]], files[off[y]:off[y + 1]]
        both = np.intersect1d(fx, fy, assume_unique=True)
        together.setdefault(int(x), []).append(both)
        together.setdefault(int(y), []).append(both)
    for r, parts in together.items():
        f = files[off[r]:off[r + 1]]
        sel = np.isin(f, np.concatenate(parts))
        pos[r] = float(np.median(rts[off[r]:off[r + 1]][sel]))
    # clusters (chains of proven pairs), boundaries at the midpoints between neighbouring positions
    parent = {r: r for r in together}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for x, y in zip(a, b):
        rx, ry = find(int(x)), find(int(y))
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)
    clusters = {}
    for r in together:
        clusters.setdefault(find(r), []).append(r)
    for cl in clusters.values():
        cl.sort(key=lambda r: pos[r])
        for n, r in enumerate(cl):
            if n > 0:
                lo[r] = 0.5 * (pos[cl[n - 1]] + pos[r])
            if n < len(cl) - 1:
                hi[r] = 0.5 * (pos[r] + pos[cl[n + 1]])
    return lo, hi, pos


def clip_windows(g_rt, g_half, lo, hi):
    """gap-fill window centre and half-width per group, cut to its own RT range; groups whose window lies inside it
    keep their values unchanged (bit for bit); an empty window has half-width 0 (nothing is read)"""
    g_rt = np.asarray(g_rt, np.float64)
    g_half = np.asarray(g_half, np.float64)
    c, h = g_rt.copy(), g_half.copy()
    cut = (lo > g_rt - g_half) | (hi < g_rt + g_half)
    if cut.any():
        wlo = np.maximum(g_rt[cut] - g_half[cut], lo[cut])
        whi = np.minimum(g_rt[cut] + g_half[cut], hi[cut])
        c[cut] = 0.5 * (wlo + whi)
        h[cut] = np.maximum(0.0, 0.5 * (whi - wlo))
    return c, h, cut
