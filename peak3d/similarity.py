"""MS/MS spectral similarity: spectrum cleaning and five scores, compiled.

Cleaning and the entropy scores follow Li et al. 2021 (Nat Methods 18:1524, "Spectral entropy outperforms MS/MS dot
product similarity"; reference code ms_entropy): drop non-positive peaks and everything from 1.6 Da below the
precursor up, centroid peaks closer than max(0.05 Da, 2 x the fragment tolerance) (strongest first: intensity-
weighted m/z, summed intensity), drop peaks under 1 % of the strongest, normalise to sum 1. Peaks are matched by a
greedy two-pointer walk within the fragment tolerance (Da), each peak at most once.

Scores, all in [0, 1]:
  entropy           0.5 x sum over matched pairs of (a+b) log2(a+b) - a log2 a - b log2 b (= 1 - (2 S_AB - S_A - S_B) /
                    ln 4 with unmatched peaks kept apart)
  weighted_entropy  the same after each spectrum with entropy S = -sum p ln p below 3 is reweighted p^(0.25 + 0.25 S)
                    and renormalised (flattens spectra dominated by a few peaks)
  dot               cosine of the cleaned intensities: sum a_i b_j over matched pairs / sqrt(sum a^2 x sum b^2)
  weighted_dot      cosine of I^0.6 x (m/z)^3 (Stein & Scott 1994)
  reverse_entropy   query peaks without a library partner are dropped, the query renormalised, then weighted_entropy:
                    the library spectrum must be explained by the query, extra (co-isolated) query peaks cost nothing
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

METHOD_CODE = {"entropy": 0, "weighted_entropy": 1, "dot": 2, "weighted_dot": 3, "reverse_entropy": 4}
PRECURSOR_MARGIN = 1.6     # Da below the precursor where cleaning cuts (the precursor and its isotopes)
NOISE = 0.01               # of the strongest peak
MIN_CENTROID_DA = 0.05


@njit(cache=True)
def _centroid_once(mz, it, d):
    """one pass of strongest-first merging: each surviving peak absorbs every peak within d of it"""
    n = len(mz)
    out_m = np.empty(n)
    out_i = np.empty(n)
    k = 0
    work = it.copy()
    for idx in np.argsort(it)[::-1]:
        if work[idx] <= 0:
            continue
        lo = idx - 1
        while lo >= 0 and mz[idx] - mz[lo] <= d:
            lo -= 1
        hi = idx + 1
        while hi < n and mz[hi] - mz[idx] <= d:
            hi += 1
        s = 0.0
        sw = 0.0
        for j in range(lo + 1, hi):
            s += work[j]
            sw += work[j] * mz[j]
            work[j] = 0.0
        out_m[k] = sw / s
        out_i[k] = s
        k += 1
    o = np.argsort(out_m[:k])
    return out_m[:k][o], out_i[:k][o]


@njit(cache=True)
def clean(mz, it, prec, tol):
    """(m/z, intensity summing to 1) of one spectrum, cleaned as in the module docstring; m/z ascending"""
    m, i = clean_raw(mz, it, prec, tol)
    if len(m) == 0:
        return m, i
    return m, i / i.sum()


@njit(cache=True)
def clean_raw(mz, it, prec, tol):
    """clean() before the final normalisation: the kept peaks with their measured intensities"""
    n = len(mz)
    keep = np.zeros(n, np.bool_)
    cut = prec - PRECURSOR_MARGIN if prec > 0 else np.inf
    for p in range(n):
        keep[p] = it[p] > 0 and math.isfinite(it[p]) and math.isfinite(mz[p]) and mz[p] < cut
    m = mz[keep].astype(np.float64)
    i = it[keep].astype(np.float64)
    if len(m) == 0:
        return m, i
    o = np.argsort(m)
    m = m[o]
    i = i[o]
    d = max(MIN_CENTROID_DA, 2.0 * tol)
    while len(m) > 1:
        tight = False
        for p in range(len(m) - 1):
            if m[p + 1] - m[p] <= d:
                tight = True
                break
        if not tight:
            break
        m, i = _centroid_once(m, i, d)
    top = i.max()
    sel = i >= NOISE * top
    return m[sel], i[sel]


@njit(cache=True, parallel=True)
def clean_batch(off, mz, it, prec, tol):
    """clean every spectrum of a CSR batch (off, mz, intensity) -> a new CSR batch"""
    n = len(off) - 1
    counts = np.zeros(n, np.int64)
    for k in prange(n):
        m, _ = clean(mz[off[k]:off[k + 1]], it[off[k]:off[k + 1]], prec[k], tol)
        counts[k] = len(m)
    new_off = np.zeros(n + 1, np.int64)
    for k in range(n):
        new_off[k + 1] = new_off[k] + counts[k]
    out_m = np.empty(new_off[n])
    out_i = np.empty(new_off[n])
    for k in prange(n):
        m, i = clean(mz[off[k]:off[k + 1]], it[off[k]:off[k + 1]], prec[k], tol)
        out_m[new_off[k]:new_off[k + 1]] = m
        out_i[new_off[k]:new_off[k + 1]] = i
    return new_off, out_m, out_i


@njit(cache=True)
def _match(ma, mb, tol):
    """greedy two-pointer pairing of two ascending m/z lists within tol (Da): index pairs"""
    n = min(len(ma), len(mb))
    pa = np.empty(n, np.int64)
    pb = np.empty(n, np.int64)
    a = 0
    b = 0
    k = 0
    while a < len(ma) and b < len(mb):
        d = ma[a] - mb[b]
        if d < -tol:
            a += 1
        elif d > tol:
            b += 1
        else:
            pa[k] = a
            pb[k] = b
            k += 1
            a += 1
            b += 1
    return pa[:k], pb[:k]


@njit(cache=True)
def _entropy_weight(p):
    """Li et al.'s weighting: spectra with entropy below 3 get p^(0.25 + 0.25 S), renormalised"""
    s = 0.0
    for v in p:
        if v > 0:
            s -= v * math.log(v)
    if s >= 3.0:
        return p
    w = p ** (0.25 + 0.25 * s)
    return w / w.sum()


@njit(cache=True)
def _entropy_sum(a, b, pa, pb):
    s = 0.0
    for k in range(len(pa)):
        x = a[pa[k]]
        y = b[pb[k]]
        s += (x + y) * math.log2(x + y) - x * math.log2(x) - y * math.log2(y)
    return min(max(0.5 * s, 0.0), 1.0)


@njit(cache=True)
def _cosine(a, b, pa, pb):
    num = 0.0
    for k in range(len(pa)):
        num += a[pa[k]] * b[pb[k]]
    den = math.sqrt((a * a).sum() * (b * b).sum())
    return min(num / den, 1.0) if den > 0 else 0.0


@njit(cache=True)
def score(ma, ia, mb, ib, method, tol):
    """(score, matched peaks) of two CLEANED spectra (query a, library b; intensities summing to 1)"""
    if len(ma) == 0 or len(mb) == 0:
        return 0.0, 0
    pa, pb = _match(ma, mb, tol)
    n = len(pa)
    if n == 0:
        return 0.0, 0
    if method == 0:
        return _entropy_sum(ia, ib, pa, pb), n
    if method == 1:
        return _entropy_sum(_entropy_weight(ia), _entropy_weight(ib), pa, pb), n
    if method == 2:
        return _cosine(ia, ib, pa, pb), n
    if method == 3:
        return _cosine(ia ** 0.6 * ma ** 3, ib ** 0.6 * mb ** 3, pa, pb), n
    # reverse: the query keeps only its matched peaks (in pair order, so pair k is (k, pb[k]))
    ra = ia[pa]
    ra = ra / ra.sum()
    return _entropy_sum(_entropy_weight(ra), _entropy_weight(ib), np.arange(n), pb), n


@njit(cache=True, parallel=True)
def score_pairs(q_off, q_mz, q_it, l_off, l_mz, l_it, pair_q, pair_l, method, tol):
    """scores and matched-peak counts of (query, library) index pairs over two cleaned CSR batches"""
    n = len(pair_q)
    out = np.zeros(n)
    nm = np.zeros(n, np.int32)
    for k in prange(n):
        q = pair_q[k]
        r = pair_l[k]
        s, m = score(q_mz[q_off[q]:q_off[q + 1]], q_it[q_off[q]:q_off[q + 1]],
                     l_mz[l_off[r]:l_off[r + 1]], l_it[l_off[r]:l_off[r + 1]], method, tol)
        out[k] = s
        nm[k] = m
    return out, nm


def similarity(mz_a, int_a, mz_b, int_b, method: str = "weighted_entropy", fragment_tol: float = 0.02,
               prec_a: float = float("nan"), prec_b: float = float("nan")):
    """(score, matched peaks) of two raw spectra: a = query, b = library; both are cleaned first"""
    tol = float(fragment_tol)
    ma, ia = clean(np.asarray(mz_a, np.float64), np.asarray(int_a, np.float64), float(prec_a), tol)
    mb, ib = clean(np.asarray(mz_b, np.float64), np.asarray(int_b, np.float64), float(prec_b), tol)
    s, n = score(ma, ia, mb, ib, METHOD_CODE[method], tol)
    return float(s), int(n)
