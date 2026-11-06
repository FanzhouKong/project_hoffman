"""Compound groups: the aligned features (ions) of one compound -- adducts, in-source fragments, multimers,
charge states and their isotopes -- joined on evidence and named where a mass relation fits.

Correlation groups, mass arithmetic only names. Evidence for a pair of aligned rows:
  1. co-elution: consensus apexes within the grouping RT tolerance, RT ranges overlapping by at least half the
     narrower peak;
  2. peak shape: Pearson r of the two EICs over the union of their scan bounds, in up to SHAPE_FILES files where
     both are detected (the strongest ones), summarised as the median;
  3. intensity across files: Pearson r of log height over the files where both are present (detected or
     confirmed by gap filling).
The study calibrates its own cutoffs. The shape cut is the value that keeps ISO_KEEP of the M/M+1 pairs the
isotope step found (known same-compound pairs, the positive control); the shifted-null distribution (unrelated
pairs, apexes at least NULL_SEP peak widths apart, the second trace shifted onto the first apex) is reported next
to it, and the cut never drops below its 90th percentile. A pair with a known mass relation (adduct difference,
13C step) passes shape at a lower cutoff. Intensity correlation acts as a veto, and only when the study makes it
informative (isotope pairs separate from the null); replicate-only studies switch it off by themselves.
Clusters are stars around the strongest unassigned ion (non-transitive, so chains cannot run through a crowded
region); isotopic features follow their parent. Roles come from the neutral mass that explains the most members;
a member no relation explains carries its m/z difference to the base ion. Rows are never removed."""
from __future__ import annotations

import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .io import C13, Cloud, eic_kernel

PROTON = 1.00727646688
SHAPE_FILES = 0            # files per pair for the shape score: the strongest N, 0 = every shared file
MIN_INT_FILES = 6          # shared present files needed for an intensity correlation
NULL_SEP = 3.0             # unrelated pairs: apexes >= this many study FWHM apart
N_NULL_SHAPE = 400         # shifted unrelated pairs per file (shape null)
N_NULL_INT = 20000         # unrelated pairs for the intensity null
SHAPE_FLOOR, SHAPE_CEIL = 0.5, 0.95
SHAPE_REL_FLOOR = 0.5
ISO_KEEP = 0.95            # the shape cut keeps this share of the isotope (known same-compound) pairs
RELATED_RELAX = 0.1        # pairs with a known mass relation pass shape this much lower
PAIRS_MIN_R = 0.4          # compound_pairs.tsv keeps pairs with r_shape >= this, a relation or an edge
RATIO_TOL = 0.30103        # small studies: a file agrees with the pair's median log10 height ratio within a factor of 2
RATIO_MIN_FILES = 3        # ... over at least this many shared files
RATIO_KEEP = 0.5           # ... and the pair stays when at least this fraction of its files agree
CONSIST_ITER = 4           # group-consistency passes
CONSIST_MARGIN = 0.05      # a member moves only to a group it agrees with by this much more than its own
MIN_NEUTRAL = 40.0
MAX_HYP = 20               # strongest members of a cluster that propose neutral-mass hypotheses

# (name, n of M, mass added, z); neutral masses from the IUPAC monoisotopic table
ADDUCTS = {
    "pos": [("[M+H]+", 1, PROTON, 1), ("[M+NH4]+", 1, 18.0338257, 1), ("[M+Na]+", 1, 22.9892207, 1),
            ("[M+K]+", 1, 38.9631579, 1), ("[M+H-H2O]+", 1, PROTON - 18.0105647, 1),
            ("[M+H-2H2O]+", 1, PROTON - 36.0211294, 1), ("[M+H-NH3]+", 1, PROTON - 17.0265491, 1),
            ("[M+H-HCOOH]+", 1, PROTON - 46.0054793, 1), ("[M+2Na-H]+", 1, 44.9711601, 1),
            ("[M+CH3OH+H]+", 1, 33.0334912, 1), ("[M+ACN+H]+", 1, 42.0338257, 1), ("[M+ACN+Na]+", 1, 64.0157652, 1),
            ("[M+2H]2+", 1, 2 * PROTON, 2), ("[M+H+Na]2+", 1, PROTON + 22.9892207, 2),
            ("[2M+H]+", 2, PROTON, 1), ("[2M+NH4]+", 2, 18.0338257, 1), ("[2M+Na]+", 2, 22.9892207, 1)],
    "neg": [("[M-H]-", 1, -PROTON, 1), ("[M+Cl]-", 1, 34.9694021, 1), ("[M+HCOO]-", 1, 44.9982030, 1),
            ("[M+CH3COO]-", 1, 59.0138531, 1), ("[M-H-H2O]-", 1, -PROTON - 18.0105647, 1),
            ("[M-H-CO2]-", 1, -PROTON - 43.9898292, 1), ("[M+Na-2H]-", 1, 20.9746662, 1),
            ("[M+K-2H]-", 1, 36.9486034, 1), ("[M+Br]-", 1, 78.9188856, 1),
            ("[M-2H]2-", 1, -2 * PROTON, 2), ("[2M-H]-", 2, -PROTON, 1), ("[2M+HCOO]-", 2, 44.9982030, 1)],
}
PRIMARY = {"pos": "[M+H]+", "neg": "[M-H]-"}


def adduct_table(polarity: str):
    if polarity in ADDUCTS:
        return ADDUCTS[polarity]
    return ADDUCTS["pos"] + ADDUCTS["neg"]


def pair_relations(polarity: str):
    """m/z differences between two singly charged ions of one neutral (plus 13C steps): the deltas that let a pair
    pass the shape test at the relaxed cutoff"""
    singles = [d for (_, n, d, z) in adduct_table(polarity) if n == 1 and z == 1]
    rel = {abs(a - b) for a in singles for b in singles if a != b}
    rel |= {k * C13 for k in (1, 2, 3)}
    return np.array(sorted(rel))


# ------------------------------------------------------------------ kernels --
@njit(cache=True)
def _pearson(a, b):
    n = 0
    sa = sb = saa = sbb = sab = 0.0
    for k in range(len(a)):
        if a[k] > 0 or b[k] > 0:
            x = float(a[k]); y = float(b[k])
            n += 1; sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y
    if n < 3:
        return np.nan
    va = saa - sa * sa / n
    vb = sbb - sb * sb / n
    if va <= 0 or vb <= 0:
        return np.nan
    return (sab - sa * sb / n) / np.sqrt(va * vb)


@njit(cache=True)
def _candidate_kernel(rt, rt_min, rt_max, order, rt_win, packed, max_pairs):
    """pairs (i<j, by sorted RT) with |drt| <= rt_win, RT ranges overlapping by half the narrower width and at least
    one shared detected file (packed: G x W uint64 bit masks of detected files)"""
    n = len(rt)
    I = np.empty(max_pairs, np.int64)
    J = np.empty(max_pairs, np.int64)
    m = 0
    W = packed.shape[1]
    for a in range(n):
        i = order[a]
        wi = rt_max[i] - rt_min[i]
        for b in range(a + 1, n):
            j = order[b]
            if rt[j] - rt[i] > rt_win:
                break
            ov = min(rt_max[i], rt_max[j]) - max(rt_min[i], rt_min[j])
            wj = rt_max[j] - rt_min[j]
            if ov < 0.5 * min(wi, wj):
                continue
            shared = False
            for w in range(W):
                if packed[i, w] & packed[j, w]:
                    shared = True
                    break
            if not shared:
                continue
            if m >= max_pairs:
                return I[:m], J[:m], True
            I[m] = i; J[m] = j
            m += 1
    return I[:m], J[:m], False


@njit(cache=True)
def _pick_files_kernel(I, J, H, det, K):
    """for every pair the up to K files, where both are detected, with the largest min(height): returns
    (pair index, file) task arrays"""
    P = len(I)
    N = H.shape[1]
    tp = np.empty(P * K, np.int64)
    tf = np.empty(P * K, np.int64)
    m = 0
    bestf = np.empty(K, np.int64)
    bestv = np.empty(K)
    for p in range(P):
        i = I[p]; j = J[p]
        nb = 0
        for f in range(N):
            if not (det[i, f] and det[j, f]):
                continue
            v = min(H[i, f], H[j, f])
            if nb < K:
                bestf[nb] = f; bestv[nb] = v; nb += 1
            else:
                # replace the smallest if v is larger
                kmin = 0
                for k in range(1, K):
                    if bestv[k] < bestv[kmin]:
                        kmin = k
                if v > bestv[kmin]:
                    bestf[kmin] = f; bestv[kmin] = v
        for k in range(nb):
            tp[m] = p; tf[m] = bestf[k]; m += 1
    return tp[:m], tf[:m]


@njit(cache=True)
def _tasks_for_file(I, J, det, f):
    """the pairs (as indices into I, J) with both ions detected in file f"""
    m = 0
    for p in range(len(I)):
        if det[I[p], f] and det[J[p], f]:
            m += 1
    out = np.empty(m, np.int64)
    m = 0
    for p in range(len(I)):
        if det[I[p], f] and det[J[p], f]:
            out[m] = p; m += 1
    return out


@njit(cache=True)
def _pair_sums_kernel(E, eoff, ws0, lo, hi, pa, pb):
    """for the pooled shape score: the sums n, Sx, Sy, Sxx, Syy, Sxy of the two traces over the union of their
    bounds, each trace scaled by its own maximum there (all zero when either trace is flat or the overlap is short)"""
    K = len(pa)
    out = np.zeros((K, 6))
    for k in range(K):
        a = pa[k]; b = pb[k]
        s0 = max(min(lo[a], lo[b]), ws0[a], ws0[b])
        s1 = min(max(hi[a], hi[b]) + 1, ws0[a] + (eoff[a + 1] - eoff[a]), ws0[b] + (eoff[b + 1] - eoff[b]))
        if s1 - s0 < 3:
            continue
        x = E[eoff[a] + (s0 - ws0[a]):eoff[a] + (s1 - ws0[a])]
        y = E[eoff[b] + (s0 - ws0[b]):eoff[b] + (s1 - ws0[b])]
        mx = x.max(); my = y.max()
        if mx <= 0 or my <= 0:
            continue
        n = s1 - s0
        sx = sy = sxx = syy = sxy = 0.0
        for t in range(n):
            u = x[t] / mx; v = y[t] / my
            sx += u; sy += v; sxx += u * u; syy += v * v; sxy += u * v
        out[k, 0] = n; out[k, 1] = sx; out[k, 2] = sy; out[k, 3] = sxx; out[k, 4] = syy; out[k, 5] = sxy
    return out


def _pooled_r(acc):
    """pooled Pearson r per pair from accumulated sums (every scan of every shared file in one correlation)"""
    n = acc[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        vx = acc[:, 3] - acc[:, 1] ** 2 / n
        vy = acc[:, 4] - acc[:, 2] ** 2 / n
        r = (acc[:, 5] - acc[:, 1] * acc[:, 2] / n) / np.sqrt(vx * vy)
    r[(n < 3) | ~(vx > 0) | ~(vy > 0)] = np.nan
    return r


@njit(cache=True)
def _extract_eics(off, mz, inten, targets, tols, s0, s1):
    """dense EICs of many windows, flattened; eoff[k]..eoff[k+1] holds window k (scans s0[k]..s1[k]-1)"""
    n = len(targets)
    eoff = np.zeros(n + 1, np.int64)
    for k in range(n):
        eoff[k + 1] = eoff[k] + max(s1[k] - s0[k], 0)
    E = np.zeros(eoff[n], np.float32)
    for k in range(n):
        if s1[k] > s0[k]:
            E[eoff[k]:eoff[k + 1]] = eic_kernel(off, mz, inten, targets[k], tols[k], s0[k], s1[k])
    return E, eoff


@njit(cache=True)
def _pair_corr_kernel(E, eoff, ws0, lo, hi, pa, pb):
    """Pearson r of rows pa[k], pb[k] over the union of their bounds, clipped to both extracted windows"""
    out = np.empty(len(pa))
    for k in range(len(pa)):
        a = pa[k]; b = pb[k]
        s0 = max(min(lo[a], lo[b]), ws0[a], ws0[b])
        s1 = min(max(hi[a], hi[b]) + 1, ws0[a] + (eoff[a + 1] - eoff[a]), ws0[b] + (eoff[b + 1] - eoff[b]))
        if s1 - s0 < 3:
            out[k] = np.nan
            continue
        x = E[eoff[a] + (s0 - ws0[a]):eoff[a] + (s1 - ws0[a])]
        y = E[eoff[b] + (s0 - ws0[b]):eoff[b] + (s1 - ws0[b])]
        out[k] = _pearson(x, y)
    return out


@njit(cache=True)
def _null_corr_kernel(off, mz, inten, E, eoff, ws0, lo, hi, apex, targets, tols, na, nb, shifts, n_scans):
    """trace of row nb shifted so that its apex lands on row na's apex plus `shift` scans, correlated with na over
    na's bounds: what an unrelated co-eluting ion looks like"""
    out = np.empty(len(na))
    for k in range(len(na)):
        a = na[k]; b = nb[k]
        d = apex[a] - apex[b] + shifts[k]
        s0 = max(lo[a], ws0[a])
        s1 = min(hi[a] + 1, ws0[a] + (eoff[a + 1] - eoff[a]))
        if s1 - s0 < 3 or s0 - d < 0 or s1 - d > n_scans:
            out[k] = np.nan
            continue
        x = E[eoff[a] + (s0 - ws0[a]):eoff[a] + (s1 - ws0[a])]
        y = eic_kernel(off, mz, inten, targets[b], tols[b], s0 - d, s1 - d)
        out[k] = _pearson(x, y)
    return out


@njit(cache=True)
def _int_corr_kernel(L, present, I, J, min_n):
    out = np.empty(len(I))
    N = L.shape[1]
    for p in range(len(I)):
        i = I[p]; j = J[p]
        n = 0
        sa = sb = saa = sbb = sab = 0.0
        for f in range(N):
            if present[i, f] and present[j, f]:
                x = L[i, f]; y = L[j, f]
                n += 1; sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y
        if n < min_n:
            out[p] = np.nan
            continue
        va = saa - sa * sa / n
        vb = sbb - sb * sb / n
        if va <= 0 or vb <= 0:
            out[p] = np.nan
        else:
            out[p] = (sab - sa * sb / n) / np.sqrt(va * vb)
    return out


@njit(cache=True)
def _int_ratio_kernel(L, present, I, J, min_n, tol):
    """small studies: the fraction of shared present files whose log10 height ratio lies within tol of the pair's
    median ratio (NaN below min_n files)"""
    out = np.empty(len(I))
    N = L.shape[1]
    buf = np.empty(N)
    for p in range(len(I)):
        i = I[p]; j = J[p]
        n = 0
        for f in range(N):
            if present[i, f] and present[j, f]:
                buf[n] = L[i, f] - L[j, f]; n += 1
        if n < min_n:
            out[p] = np.nan
            continue
        med = np.median(buf[:n])
        good = 0
        for t in range(n):
            if abs(buf[t] - med) <= tol:
                good += 1
        out[p] = good / n
    return out


@njit(cache=True)
def _star_kernel(order, indptr, indices, eligible):
    """strongest-first stars: a seed claims every unassigned eligible neighbour; non-transitive"""
    n = len(order)
    cid = np.full(n, -1, np.int64)
    c = 0
    for s in order:
        if cid[s] >= 0 or not eligible[s]:
            continue
        cid[s] = c
        for k in range(indptr[s], indptr[s + 1]):
            j = indices[k]
            if cid[j] < 0 and eligible[j]:
                cid[j] = c
        c += 1
    return cid, c


# -------------------------------------------------------------- per file ----
def _shape_one(job):
    """one file: EICs of the rows involved, shape r of its pair tasks, and shifted-null r values"""
    (stem, cache_dir, rows, mz_native, sig, lo, hi, apex, pa, pb, task_idx, pad, n_null, rt_win_scans, seed, pooled) = job
    import numba
    numba.set_num_threads(1)
    cloud = Cloud.load_npz(Path(cache_dir) / f"{stem}.npz")
    S = cloud.n_scans
    ws0 = np.maximum(lo - pad, 0).astype(np.int64)
    ws1 = np.minimum(hi + pad + 1, S).astype(np.int64)
    tols = (mz_native * (3.0 * sig + 1.0) * 1e-6).astype(np.float64)
    E, eoff = _extract_eics(cloud.off, cloud.mz, cloud.inten, mz_native.astype(np.float64), tols, ws0, ws1)
    r = _pair_corr_kernel(E, eoff, ws0, lo.astype(np.int64), hi.astype(np.int64), pa, pb)
    sums = _pair_sums_kernel(E, eoff, ws0, lo.astype(np.int64), hi.astype(np.int64), pa, pb) if pooled else np.zeros((0, 6))
    null = np.zeros(0)
    if n_null > 0 and len(rows) >= 4:
        rng = np.random.default_rng(seed)
        m = min(n_null, 4 * len(rows))
        na = rng.integers(0, len(rows), m)
        nb = rng.integers(0, len(rows), m)
        far = np.abs(apex[na] - apex[nb]) >= NULL_SEP * max(pad, 1)
        na, nb = na[far], nb[far]
        shifts = rng.integers(-rt_win_scans, rt_win_scans + 1, len(na))
        null = _null_corr_kernel(cloud.off, cloud.mz, cloud.inten, E, eoff, ws0, lo.astype(np.int64),
                                 hi.astype(np.int64), apex.astype(np.int64), mz_native.astype(np.float64), tols,
                                 na, nb, shifts, S)
        null = null[np.isfinite(null)]
    return task_idx, r, null, sums, seed


def _polarity(cache_dir, stems):
    for s in stems:
        try:
            with np.load(Path(cache_dir) / f"{s}.npz", allow_pickle=False) as z:
                p = str(z["polarity"]) if "polarity" in z else ""
        except OSError:
            continue
        if p in ("pos", "neg"):
            return p
    return ""


def _nanmedian_by_pair(values, pair_idx, P):
    """median over tasks of each pair (NaN = no finite value)"""
    ok = np.isfinite(values)
    v, p = values[ok], pair_idx[ok]
    out = np.full(P, np.nan)
    if len(v) == 0:
        return out, np.zeros(P, np.int64)
    o = np.lexsort((v, p))
    p, v = p[o], v[o]
    starts = np.r_[0, np.flatnonzero(np.diff(p)) + 1]
    ends = np.r_[starts[1:], len(p)]
    n = ends - starts
    mid = starts + n // 2
    med = np.where(n % 2 == 1, v[mid], 0.5 * (v[np.maximum(mid - 1, starts)] + v[mid]))
    cnt = np.zeros(P, np.int64)
    out[p[starts]] = med
    cnt[p[starts]] = n
    return out, cnt


def _consistent_clusters(cid, C, I, J, r, t_shape, eligible, g_h, max_iter=CONSIST_ITER, margin=CONSIST_MARGIN,
                         allow_leave=True):
    """every member must agree with the rest of its group, the strong members counting most: a member's agreement
    with a group is the height-weighted mean of its pair scores with that group's other members, a pair that was
    not evaluated or failed the intensity veto counting as 0. A member whose agreement with its own group is below
    the cut leaves (a group of its own); a member that agrees with another group at least at the cut and by
    `margin` more than with its own moves there. r: per-pair score (NaN = unusable). Returns (cid, C, n_moved,
    n_left, iterations); unassigned members (-1) stay unassigned"""
    cid = cid.copy()
    h = np.where(eligible, np.maximum(g_h, 1e-12), 0.0)
    use = np.isfinite(r) & eligible[I] & eligible[J]
    a = np.concatenate([I[use], J[use]]); b = np.concatenate([J[use], I[use]]); v = np.concatenate([r[use], r[use]])
    wv = v * h[b]                      # weighted pair score, credited to member a for b's group
    n_moved = n_left = it = 0
    for it in range(1, max_iter + 1):
        if len(a) == 0:
            break
        pos = cid >= 0
        tot = np.zeros(C); np.add.at(tot, cid[pos], h[pos])          # total weight per group
        cb = cid[b]
        key = a * np.int64(C) + cb
        uniq, inv = np.unique(key, return_inverse=True)
        S = np.zeros(len(uniq)); np.add.at(S, inv, wv)
        km, kc = uniq // C, uniq % C
        denom = tot[kc] - np.where(cid[km] == kc, h[km], 0.0)        # the group's other members
        score = np.where(denom > 0, S / np.maximum(denom, 1e-300), np.nan)
        own = np.full(len(cid), np.nan)
        is_own = kc == cid[km]
        own[km[is_own]] = score[is_own]
        # a member with no pair into its own group agrees at 0 with it (unless it is alone there)
        alone = pos & (tot[np.maximum(cid, 0)] - h <= 1e-9)
        own = np.where(np.isnan(own) & pos & ~alone, 0.0, own)
        other = ~is_own & np.isfinite(score)
        best = np.full(len(cid), -1, np.int64); best_v = np.full(len(cid), -np.inf)
        if other.any():
            o = np.lexsort((-score[other], km[other]))
            mo, co, vo = km[other][o], kc[other][o], score[other][o]
            first = np.r_[True, mo[1:] != mo[:-1]]
            best[mo[first]] = co[first]; best_v[mo[first]] = vo[first]
        leave = eligible & np.isfinite(own) & (own < t_shape) if allow_leave else np.zeros(len(cid), bool)
        move = eligible & (best >= 0) & (best_v >= t_shape) & (np.isnan(own) | (best_v > own + margin))
        if not (leave | move).any():
            break
        mv = np.flatnonzero(move)
        cid[mv] = best[mv]
        lv = np.flatnonzero(leave & ~move)
        cid[lv] = C + np.arange(len(lv)); C += len(lv)
        n_moved += len(mv); n_left += len(lv)
    pos = cid >= 0
    uniq, inv = np.unique(cid[pos], return_inverse=True)
    out = np.full(len(cid), -1, np.int64)
    out[pos] = inv
    return out, int(len(uniq)), n_moved, n_left, it


# --------------------------------------------------------------------- main --
def cluster_compounds(cons, H, present, detected, A, stems, cache_dir, fwhm_scans: dict, fwhm_study: float,
                      dt_med: float, rt_tol: float, ppm_tol: float, g_iso, g_z, g_iso_parent, cores: int = 1,
                      log=print, int_veto: str = "auto", shape_mode: str = "weighted", shape_files: int = SHAPE_FILES,
                      cluster_mode: str = "star"):
    """Returns (table: DataFrame with one row per aligned group, pairs: DataFrame of evaluated pairs, info: dict).

    cons: consensus dict (mz, rt, rt_min, rt_max); H (G x N) heights incl. filled cells; present/detected (G x N)
    bool; A: per-file member frame with columns gid, file, row, height, mz_native, sig, scan_lo, scan_hi,
    scan_apex; g_iso_parent: group index of the monoisotopic parent (-1 none); int_veto: "auto" rejects pairs
    whose intensity correlation across files is below the null's q90 when the study makes that informative,
    "off" never rejects on intensity (every co-eluting pair goes to the shape test), "ratio" uses the small-study
    rule (log height ratio stable within a factor of 2 over the shared files) for any study size; shape_mode:
    "weighted" (default) = Fisher-z mean of the per-file EIC correlations weighted by the weaker ion's log height,
    "median" = median of the per-file correlations, "pooled" = one correlation over every scan of every shared
    file; shape_files = 0 (default) uses every shared file, else the shape_files strongest ones;
    cluster_mode: "star" (strongest-first, a member joins through its pair with the seed), "consistent" (stars,
    then every member must agree with the rest of its group and may move to the group it agrees with best) or
    "move" (the moves only: nobody is evicted)."""
    tic = time.perf_counter()
    t_last = [tic]

    def lap(name):
        now = time.perf_counter()
        info[f"t_{name}"] = round(now - t_last[0], 3)
        t_last[0] = now
    G = len(cons["mz"]); N = len(stems)
    mz = np.asarray(cons["mz"], np.float64); rt = np.asarray(cons["rt"], np.float64)
    rt_min = np.asarray(cons["rt_min"], np.float64); rt_max = np.asarray(cons["rt_max"], np.float64)
    polarity = _polarity(cache_dir, stems)
    info = dict(polarity=polarity, n_groups=int(G))
    empty_pairs = pd.DataFrame(columns=["gid_a", "gid_b", "r_shape", "n_files_shape", "r_int", "n_files_int",
                                        "relation", "edge"])
    if G == 0:
        return _table(G, np.arange(G), [""] * G, np.full(G, np.nan), np.ones(G, bool)), empty_pairs, info
    g_h = H.max(axis=1) if N else np.zeros(G)

    # 1. candidates
    rt_win = float(rt_tol)
    order = np.argsort(rt, kind="stable")
    W = (N + 63) // 64
    packed = np.zeros((G, W), np.uint64)
    for f in range(N):
        packed[:, f // 64] |= (detected[:, f].astype(np.uint64) << np.uint64(f % 64))
    max_pairs = 40_000_000
    I, J, truncated = _candidate_kernel(rt, rt_min, rt_max, order, rt_win, packed, max_pairs)
    if truncated:
        log(f"compounds: candidate pairs truncated at {max_pairs}")
    P = len(I)
    info["n_candidates"] = int(P)
    lap("candidates")

    # 2. intensity correlation across files (cheap: the aligned matrix), and its veto
    iso_pair = (g_iso_parent[J] == I) | (g_iso_parent[I] == J)     # positive controls
    with np.errstate(divide="ignore"):
        L = np.where(H > 0, np.log10(np.maximum(H, 1e-30)), 0.0).astype(np.float64)
    r_int = _int_corr_kernel(L, present, I, J, MIN_INT_FILES) if P else np.zeros(0)
    n_int = (present[I] & present[J]).sum(axis=1) if P else np.zeros(0, np.int64)
    int_active = False; t_int = np.nan; int_iso_median = np.nan; q_int = (np.nan, np.nan, np.nan)
    lap("intensity")
    int_mode = "off"
    if int_veto == "ratio" or (int_veto == "auto" and N < 2 * MIN_INT_FILES and N >= RATIO_MIN_FILES and P):
        r_int = _int_ratio_kernel(L, present, I, J, RATIO_MIN_FILES, RATIO_TOL) if P else np.zeros(0)
        int_active = True; t_int = RATIO_KEEP; int_mode = "ratio"
    elif int_veto != "off" and N >= 2 * MIN_INT_FILES and P:
        rng = np.random.default_rng(0)
        na = rng.integers(0, G, N_NULL_INT); nb = rng.integers(0, G, N_NULL_INT)
        far = np.abs(rt[na] - rt[nb]) >= NULL_SEP * max(fwhm_study, rt_tol)
        rn = _int_corr_kernel(L, present, na[far], nb[far], MIN_INT_FILES)
        rn = rn[np.isfinite(rn)]
        if len(rn) >= 200:
            q_int = tuple(np.quantile(rn, [0.9, 0.95, 0.99]))
            ri = r_int[iso_pair]; ri = ri[np.isfinite(ri)]
            if len(ri) >= 20:
                int_iso_median = float(np.median(ri))
                int_active = int_iso_median >= q_int[2]
                t_int = float(q_int[0])
                int_mode = "corr" if int_active else "off"
    info.update(int_veto=int_veto, int_mode=int_mode, shape_mode=shape_mode, shape_files=int(shape_files),
                cluster_mode=cluster_mode, int_active=bool(int_active), int_null_q90=float(q_int[0]), int_null_q95=float(q_int[1]),
                int_null_q99=float(q_int[2]), int_iso_median=float(int_iso_median), int_cut=float(t_int))

    # 3. shape correlation in the strongest shared files, for the pairs the veto has not already rejected
    sel = np.flatnonzero(~(int_active & np.isfinite(r_int) & (r_int < t_int))) if P else np.zeros(0, np.int64)
    info["n_shape_tested"] = int(len(sel))
    pooled = shape_mode == "pooled"
    weighted = shape_mode == "weighted"
    all_files = not (shape_files and shape_files > 0)
    r_shape = np.full(P, np.nan); n_shape = np.zeros(P, np.int64)
    null_shape = []
    acc = np.zeros((P, 6)) if pooled else None
    wz = np.zeros(P); wsum = np.zeros(P)
    med_idx, med_r = [], []
    if all_files:
        per_file = None
        files_todo = list(range(N))
    else:
        tp, tf = _pick_files_kernel(I[sel], J[sel], H.astype(np.float64), detected, int(shape_files))
        tp = sel[tp]
        order_t = np.argsort(tf, kind="stable")
        tp_s, tf_s = tp[order_t], tf[order_t]
        bounds = np.r_[0, np.flatnonzero(np.diff(tf_s)) + 1, len(tf_s)]
        per_file = {int(tf_s[a]): tp_s[a:b] for a, b in zip(bounds[:-1], bounds[1:])}
        files_todo = sorted(per_file)
    if len(sel) and files_todo:
        # (group, file) -> position in A (-1 where the file has no detected member)
        posmat = np.full((G, N), -1, np.int64)
        posmat[A["gid"].values, A["file"].values] = np.arange(len(A))
        scans_win = max(1, int(round(rt_win / max(dt_med, 1e-9))))
        I_sel, J_sel = I[sel], J[sel]

        def job_for(f):
            tasks = sel[_tasks_for_file(I_sel, J_sel, detected, f)] if all_files else per_file[f]
            if len(tasks) == 0:
                return None
            gi, gj = I[tasks], J[tasks]
            uniq = np.unique(np.concatenate([gi, gj]))
            rows = posmat[uniq, f]
            pa = np.searchsorted(uniq, gi)
            pb = np.searchsorted(uniq, gj)
            fs = fwhm_scans.get(stems[f], 0.0)
            pad = int(max(2, round(fs if fs > 0 else 4.0)))
            return (stems[f], str(cache_dir), rows, A["mz_native"].values[rows], A["sig"].values[rows],
                    A["scan_lo"].values[rows].astype(np.int64), A["scan_hi"].values[rows].astype(np.int64),
                    A["scan_apex"].values[rows].astype(np.int64), pa, pb, tasks, pad, N_NULL_SHAPE, scans_win, f, pooled)

        def take(o):
            idx, r, null, sums, f = o
            if len(null):
                null_shape.append(null)
            if pooled:
                np.add.at(acc, idx, sums)
                np.add.at(n_shape, idx[sums[:, 0] > 0], 1)
            elif weighted:
                ok = np.isfinite(r)
                w = np.log10(np.maximum(np.minimum(H[I[idx[ok]], f], H[J[idx[ok]], f]), 10.0))
                np.add.at(wz, idx[ok], w * np.arctanh(np.clip(r[ok], -0.999999, 0.999999)))
                np.add.at(wsum, idx[ok], w)
                np.add.at(n_shape, idx[ok], 1)
            else:
                med_idx.append(idx); med_r.append(r)

        # files in batches: the all-files mode holds only a batch's task arrays at a time
        n_workers = max(1, min(len(files_todo), cores))
        batch = max(n_workers * 2, 4)
        pool = get_context("spawn").Pool(n_workers) if n_workers > 1 else None
        try:
            for a in range(0, len(files_todo), batch):
                jobs = [j for j in (job_for(f) for f in files_todo[a:a + batch]) if j is not None]
                if not jobs:
                    continue
                if pool is None:
                    for j in jobs:
                        take(_shape_one(j))
                else:
                    for o in pool.imap_unordered(_shape_one, jobs, chunksize=1):
                        take(o)
        finally:
            if pool is not None:
                pool.close(); pool.join()
        if pooled:
            r_shape = _pooled_r(acc)
        elif weighted:
            with np.errstate(invalid="ignore", divide="ignore"):
                r_shape = np.where(wsum > 0, np.tanh(wz / np.maximum(wsum, 1e-300)), np.nan)
        elif med_idx:
            r_shape, n_shape = _nanmedian_by_pair(np.concatenate(med_r), np.concatenate(med_idx), P)
    lap("shape")
    null_shape = np.concatenate(null_shape) if null_shape else np.zeros(0)
    if len(null_shape) >= 50:
        q90, q95, q99 = np.quantile(null_shape, [0.9, 0.95, 0.99])
    else:
        q90 = q95 = q99 = np.nan
    r_iso = r_shape[iso_pair & np.isfinite(r_shape)]
    if len(r_iso) >= 20:
        t_shape = float(np.quantile(r_iso, 1.0 - ISO_KEEP))
    else:
        t_shape = SHAPE_FLOOR
    lo = q90 if np.isfinite(q90) else SHAPE_FLOOR
    t_shape = float(np.clip(t_shape, max(SHAPE_FLOOR, min(lo, SHAPE_CEIL)), SHAPE_CEIL))
    t_rel = float(np.clip(t_shape - RELATED_RELAX, SHAPE_REL_FLOOR, t_shape))
    info.update(n_null_shape=int(len(null_shape)), shape_null_q90=float(q90), shape_null_q95=float(q95),
                shape_null_q99=float(q99), n_iso_pairs=int(len(r_iso)),
                iso_shape_q05=float(np.quantile(r_iso, 0.05)) if len(r_iso) else np.nan,
                shape_cut=t_shape, shape_cut_related=t_rel)

    # 4. decision
    rel = pair_relations(polarity)
    relation = np.zeros(P, bool)
    if P and len(rel):
        d = np.abs(mz[J] - mz[I])
        k = np.searchsorted(rel, d)
        tol = ppm_tol * 1e-6 * np.maximum(mz[I], mz[J]) * np.sqrt(2.0)
        near = np.full(P, np.inf)
        for kk in (np.maximum(k - 1, 0), np.minimum(k, len(rel) - 1)):
            near = np.minimum(near, np.abs(rel[kk] - d))
        relation = near <= tol
    shape_ok = np.where(relation, r_shape >= t_rel, r_shape >= t_shape)
    shape_ok &= np.isfinite(r_shape)
    int_ok = np.ones(P, bool) if not int_active else (~np.isfinite(r_int)) | (r_int >= t_int)
    edge = shape_ok & int_ok
    if iso_pair.any():
        ev = iso_pair & np.isfinite(r_shape)
        info["iso_pairs_evaluated"] = int(ev.sum())
        info["iso_pass_shape"] = float(np.mean(r_shape[ev] >= t_shape)) if ev.any() else np.nan
        info["iso_shape_median"] = float(np.median(r_shape[ev])) if ev.any() else np.nan
    info.update(n_edges=int(edge.sum()), n_relation_pairs=int(relation.sum()))

    # 5. stars; isotope children follow their parent
    has_parent = g_iso_parent >= 0
    eligible = ~has_parent
    ei, ej = I[edge], J[edge]
    src = np.concatenate([ei, ej]); dst = np.concatenate([ej, ei])
    o = np.argsort(src, kind="stable")
    indptr = np.zeros(G + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=G), out=indptr[1:])
    cid, C = _star_kernel(np.argsort(-g_h, kind="stable"), indptr, dst[o], eligible)
    if cluster_mode in ("consistent", "move") and P:
        # pairs the intensity veto rejected give no support; related pairs are judged at their relaxed cut
        r_eff = np.where(int_ok, r_shape, np.nan)
        r_eff = np.where(relation, r_eff + (t_shape - t_rel), r_eff)
        cid, C, n_moved, n_left, n_iter = _consistent_clusters(cid, C, I, J, r_eff, t_shape, eligible, g_h,
                                                               allow_leave=cluster_mode == "consistent")
        info.update(consist_moved=int(n_moved), consist_left=int(n_left), consist_iter=int(n_iter))
    # children: parent's cluster (parents are eligible, so assigned); a child whose parent is itself a child
    # (should not happen) falls back to its own cluster
    for _ in range(2):
        ch = np.flatnonzero(has_parent & (cid < 0))
        if len(ch) == 0:
            break
        par = g_iso_parent[ch]
        ok = cid[par] >= 0
        cid[ch[ok]] = cid[par[ok]]
    loose = np.flatnonzero(cid < 0)
    cid[loose] = C + np.arange(len(loose)); C += len(loose)
    lap("cluster")

    # 6. roles
    roles, neutral, is_base = _assign_roles(cid, C, mz, g_h, g_iso, g_z, g_iso_parent, polarity, ppm_tol)
    n_members = np.bincount(cid, minlength=C)[cid]
    table = _table(G, cid, roles, neutral, is_base)
    table["n_members"] = n_members.astype(np.int64)
    lap("roles")
    keep_p = edge | relation | (np.nan_to_num(r_shape, nan=-1.0) >= PAIRS_MIN_R)
    pairs = pd.DataFrame({"gid_a": I[keep_p], "gid_b": J[keep_p], "r_shape": r_shape[keep_p], "n_files_shape": n_shape[keep_p],
                          "r_int": r_int[keep_p], "n_files_int": n_int[keep_p], "relation": relation[keep_p], "edge": edge[keep_p]})
    sizes = np.bincount(cid, minlength=C)
    info.update(n_compounds=int(C), n_multi=int((sizes > 1).sum()), size_max=int(sizes.max()),
                size_q99=float(np.quantile(sizes, 0.99)), n_pairs_written=int(keep_p.sum()),
                seconds=round(time.perf_counter() - tic, 3))
    return table, pairs, info


def _table(G, cid, roles, neutral, is_base):
    return pd.DataFrame({"compound_id": np.asarray(cid, np.int64), "role": list(roles),
                         "neutral_mass": np.asarray(neutral, np.float64), "is_base": np.asarray(is_base, bool).astype(np.int64)})


def _assign_roles(cid, C, mz, g_h, g_iso, g_z, g_iso_parent, polarity, ppm_tol):
    """per cluster: base = strongest member without an isotope parent; the neutral mass explaining the most
    monoisotopic members (ties: the one holding the primary ion, then the heavier total) names them; isotopic
    members take their parent's role + ' M+k'; the rest carry the m/z difference to the base"""
    G = len(mz)
    roles = [""] * G
    neutral = np.full(G, np.nan)
    is_base = np.zeros(G, bool)
    table = adduct_table(polarity)
    names = np.array([t[0] for t in table])
    n_k = np.array([t[1] for t in table], np.float64)
    d_k = np.array([t[2] for t in table], np.float64)
    z_k = np.array([t[3] for t in table], np.float64)
    primary = {PRIMARY["pos"], PRIMARY["neg"]} if polarity not in PRIMARY else {PRIMARY[polarity]}
    is_primary = np.array([nm in primary for nm in names])
    order = np.argsort(cid, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(cid[order])) + 1, len(order)]
    for a, b in zip(starts[:-1], starts[1:]):
        mem = order[a:b]
        mono = mem[g_iso_parent[mem] < 0]
        if len(mono) == 0:
            mono = mem
        base = mono[np.argmax(g_h[mono])]
        is_base[base] = True
        if len(mem) > 1:
            # implied neutral mass of every (member, adduct), where the member's charge allows the adduct
            zk = np.where(g_z[mono] > 1, g_z[mono], 1).astype(np.float64)
            allow = np.where(zk[:, None] > 1, z_k[None, :] == zk[:, None], z_k[None, :] == 1)
            Mmat = (mz[mono][:, None] * z_k[None, :] - d_k[None, :]) / n_k[None, :]
            Mmat = np.where(allow & (Mmat >= MIN_NEUTRAL), Mmat, np.nan)
            tol = ppm_tol * 1e-6 * mz[mono]                                   # Da, per member
            tolM = tol[:, None] * z_k[None, :] / n_k[None, :]                 # the same tolerance in neutral-mass units
            prop = np.argsort(-g_h[mono], kind="stable")[:MAX_HYP]
            hyps = Mmat[prop].ravel()
            hyps = hyps[np.isfinite(hyps)]
            best_expl = None
            if len(hyps):
                diff = np.abs(Mmat[None, :, :] - hyps[:, None, None])        # (h, m, k)
                ok = diff <= tolM[None, :, :]
                expl = ok.any(axis=2)                                         # (h, m)
                cnt = expl.sum(axis=1)
                if cnt.max() >= 2:
                    # which adduct per member: the closest allowed one
                    kbest = np.where(ok, diff, np.inf).argmin(axis=2)         # (h, m)
                    has_primary = (expl & is_primary[kbest]).any(axis=1)
                    weight = (expl * g_h[mono][None, :]).sum(axis=1)
                    score = cnt * 1e12 + has_primary * 1e11 + weight
                    hbest = int(np.argmax(score))
                    em = np.flatnonzero(expl[hbest])
                    best_expl = (em, kbest[hbest, em])
            if best_expl is not None:
                em, kk = best_expl
                Ms = Mmat[em, kk]; ws = g_h[mono[em]]
                M = float(np.average(Ms, weights=ws))
                for i, k in zip(em, kk):
                    roles[mono[i]] = str(names[k])
                    neutral[mono[i]] = M
            for m in mono:
                if not roles[m]:
                    roles[m] = "base" if m == base else f"{mz[m] - mz[base]:+.4f}"
            for m in mem:
                if g_iso_parent[m] >= 0:
                    p = g_iso_parent[m]
                    pr = roles[p] if roles[p] else "base"
                    roles[m] = f"{pr} M+{int(g_iso[m])}" if g_iso[m] > 0 else f"{pr} iso"
                    neutral[m] = neutral[p]
    return roles, neutral, is_base


# ------------------------------------------------------------- pipeline I/O --
def group_iso_parent(A, gid, G, g_iso):
    """group index of each isotopic group's monoisotopic parent: the majority over its members' per-file
    iso_parent rows (-1 = none / not an isotope)"""
    out = np.full(G, -1, np.int64)
    gid = np.asarray(gid, np.int64)     # grouping returns int32: child * (G + 1) below wraps past ~46k groups
    par = A["iso_parent"].values
    has = par >= 0
    if not has.any():
        return out
    # (file, row) -> gid of that row; a parent row that is not in A (dropped before alignment) resolves to -1, so the
    # per-file table is sized by the largest row index referenced, not only by the rows present
    files = A["file"].values; rows = A["row"].values
    N = int(files.max()) + 1
    nrow = np.zeros(N, np.int64)
    np.maximum.at(nrow, files, np.maximum(rows, par) + 1)
    off = np.r_[0, np.cumsum(nrow)]
    row_gid = np.full(off[-1], -1, np.int64)
    row_gid[off[files] + rows] = gid
    child = gid[has]
    pgid = row_gid[off[files[has]] + par[has]]
    ok = pgid >= 0
    child, pgid = child[ok], pgid[ok]
    # mode per child group
    key = child * (G + 1) + pgid
    uniq, cnt = np.unique(key, return_counts=True)
    c_child = uniq // (G + 1); c_par = uniq % (G + 1)
    o = np.lexsort((-cnt, c_child))
    c_child, c_par = c_child[o], c_par[o]
    first = np.r_[True, c_child[1:] != c_child[:-1]]
    out[c_child[first]] = c_par[first]
    out[(g_iso <= 0)] = -1          # a group the majority of whose members are monoisotopic is nobody's child
    out[np.arange(G) == out] = -1
    return out


def write_compounds(path, table, cons, keep_g):
    """one row per compound: base ion, members (group ids and roles), neutral mass"""
    cid = table["compound_id"].values
    roles = table["role"].values
    C = int(cid.max()) + 1 if len(cid) else 0
    order = np.argsort(cid, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(cid[order])) + 1, len(order)]
    rows = []
    mz = np.asarray(cons["mz"]); rt = np.asarray(cons["rt"])
    for a, b in zip(starts[:-1], starts[1:]):
        mem = order[a:b]
        base = mem[table["is_base"].values[mem] > 0]
        base = int(base[0]) if len(base) else int(mem[0])
        mem_sorted = mem[np.argsort(mz[mem])]
        rows.append(dict(compound_id=int(cid[mem[0]]), base_group_id=base, mz_base=mz[base], rt=rt[base],
                         n_members=len(mem), n_kept=int(keep_g[mem].sum()),
                         neutral_mass=table["neutral_mass"].values[base],
                         members=";".join(str(int(g)) for g in mem_sorted),
                         roles=";".join(str(roles[g]) for g in mem_sorted),
                         mz_members=";".join(f"{mz[g]:.4f}" for g in mem_sorted)))
    df = pd.DataFrame(rows, columns=["compound_id", "base_group_id", "mz_base", "rt", "n_members", "n_kept",
                                     "neutral_mass", "members", "roles", "mz_members"])
    out = df.copy()
    for c in ("mz_base", "neutral_mass"):
        out[c] = np.char.mod("%.6f", out[c].values.astype(np.float64))
    out.to_csv(path, sep="\t", index=False, float_format="%.6g", lineterminator="\n")
    return df
