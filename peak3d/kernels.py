"""numba kernels for the 3D watershed on a scan-major CSR point cloud.

Pipeline: find_neighbors -> link_steepest -> resolve_roots -> dense_labels -> basin_extent
          -> (noise cells) -> saddle_edges -> persistence_merge -> relabel -> sort_by_label
          -> basin_stats
All kernels are deterministic; parallel loops only write to their own element.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

# neighbour slots per point
NB_SAME_L, NB_SAME_R, NB_LEFT, NB_RIGHT = 0, 1, 2, 3

# columns of the stats table returned by basin_stats
COLS = ["apex_idx", "scan_apex", "scan_min", "scan_max", "scan_lo", "scan_hi", "n_scans", "n_points",
        "n_gaps", "height", "mz", "mz_sd_ppm", "mz_min", "mz_max", "rt", "rt_apex", "rt_min", "rt_max",
        "area", "area_bsub", "baseline", "fwhm", "asym", "gauss_r2", "n_flat", "kflags", "baseline_lo",
        "tv_ratio", "n_valleys", "tv_excess"]
# kflags bits
KF_CAP_L, KF_CAP_R, KF_NOHALF_L, KF_NOHALF_R, KF_RUN_START, KF_RUN_END = 1, 2, 4, 8, 16, 32
C = {name: i for i, name in enumerate(COLS)}
NCOL = len(COLS)


@njit(cache=True)
def per_point_sigma(inten, a, b):
    out = np.empty(len(inten), np.float32)
    for p in range(len(inten)):
        out[p] = np.sqrt(a * a + b * b / inten[p])
    return out


@njit(cache=True)
def _best_in_scan(off, mz, inten, sig, t, m, sp, cap, jptr):
    """most intense centroid of scan t within the pair tolerance of (m, sp), using the sweep
    pointer jptr (returned updated; it only moves forward as m increases). Returns (q, jptr),
    q = -1 when nothing is within tolerance."""
    a = off[t]
    b = off[t + 1]
    if a == b:
        return -1, jptr
    j = jptr
    if j < a:
        j = a
    while j + 1 < b and mz[j + 1] <= m:
        j += 1
    # mz[j] <= m < mz[j+1]  (or j == a with mz[a] > m, or j == b-1)
    best = -1
    bI = np.float32(0.0)
    q = j
    while q >= a and mz[q] >= m - cap:
        dm = abs(mz[q] - m)
        tol = 3.0 * np.sqrt(sp * sp + sig[q] * sig[q]) * m * 1e-6
        if dm <= tol and (inten[q] > bI or (inten[q] == bI and q < best)):
            best = q
            bI = inten[q]
        q -= 1
    q = j + 1
    while q < b and mz[q] <= m + cap:
        dm = mz[q] - m
        tol = 3.0 * np.sqrt(sp * sp + sig[q] * sig[q]) * m * 1e-6
        if dm <= tol and (inten[q] > bI or (inten[q] == bI and q < best)):
            best = q
            bI = inten[q]
        q += 1
    return best, j


@njit(cache=True, parallel=True)
def find_neighbors(off, mz, inten, sig, K, tol_max_ppm, same_scan_k, same_scan_floor_ppm):
    """int32[N, 4] neighbour indices (-1 = none): same-scan m/z neighbours (split pieces of one
    centroid) within same_scan_k x the pair tolerance 3*sqrt(sig_p^2 + sig_q^2) or within
    same_scan_floor_ppm, whichever is larger: a split piece sits at a small systematic offset that
    precision does not explain (the floor absorbs a 2 ppm sliver of a strong centroid), while a
    factor above 1 on the pair tolerance let a weak piece's large sigma reach a second ion 10-20
    ppm away; on each side the most intense centroid within the pair tolerance of the nearest scan
    (up to K away) that has one."""
    S = len(off) - 1
    N = len(mz)
    nbr = np.full((N, 4), -1, np.int32)
    for s in prange(S):
        a = off[s]
        b = off[s + 1]
        if a == b:
            continue
        jl = np.empty(K, np.int64)
        jr = np.empty(K, np.int64)
        for d in range(K):
            jl[d] = off[s - d - 1] if s - d - 1 >= 0 else 0
            jr[d] = off[s + d + 1] if s + d + 1 < S else 0
        for p in range(a, b):
            m = mz[p]
            sp = sig[p]
            cap = m * tol_max_ppm * 1e-6
            # same scan
            floor = same_scan_floor_ppm * m * 1e-6
            if p - 1 >= a:
                q = p - 1
                tol = same_scan_k * 3.0 * np.sqrt(sp * sp + sig[q] * sig[q]) * m * 1e-6
                if m - mz[q] <= max(min(tol, same_scan_k * cap), floor):
                    nbr[p, NB_SAME_L] = q
            if p + 1 < b:
                q = p + 1
                tol = same_scan_k * 3.0 * np.sqrt(sp * sp + sig[q] * sig[q]) * m * 1e-6
                if mz[q] - m <= max(min(tol, same_scan_k * cap), floor):
                    nbr[p, NB_SAME_R] = q
            # left: nearest scan with a centroid within tolerance
            for d in range(K):
                t = s - d - 1
                if t < 0:
                    break
                q, jl[d] = _best_in_scan(off, mz, inten, sig, t, m, sp, cap, jl[d])
                if q >= 0:
                    nbr[p, NB_LEFT] = q
                    break
            # right
            for d in range(K):
                t = s + d + 1
                if t >= S:
                    break
                q, jr[d] = _best_in_scan(off, mz, inten, sig, t, m, sp, cap, jr[d])
                if q >= 0:
                    nbr[p, NB_RIGHT] = q
                    break
    return nbr


@njit(cache=True, parallel=True)
def effective_intensity(nbr, inten):
    """intensity of a centroid plus its same-scan split neighbours: the surface height at that
    (scan, m/z) is the whole ion signal, not the piece the centroider happened to split off"""
    N = len(inten)
    eff = np.empty(N, np.float32)
    for p in prange(N):
        v = inten[p]
        q = nbr[p, NB_SAME_L]
        if q >= 0:
            v += inten[q]
        q = nbr[p, NB_SAME_R]
        if q >= 0:
            v += inten[q]
        eff[p] = v
    return eff


@njit(cache=True, parallel=True)
def _smooth_pass(nbr, src, dst):
    N = len(src)
    for p in prange(N):
        v = 0.5 * src[p]
        q = nbr[p, NB_LEFT]
        if q >= 0:
            v += 0.25 * src[q]
        q = nbr[p, NB_RIGHT]
        if q >= 0:
            v += 0.25 * src[q]
        dst[p] = v


def smooth_traces(nbr, inten, n_pass):
    """[1, 2, 1] / 4 smoothing along each centroid's chromatographic links, n_pass times (variance
    n_pass / 2 scans^2). A missing neighbour counts as 0, so isolated spikes shrink and trace ends
    taper. Used only to judge valleys between far-apart apexes; the picked heights stay raw."""
    a = inten.astype(np.float32).copy()
    b = np.empty_like(a)
    for _ in range(n_pass):
        _smooth_pass(nbr, a, b)
        a, b = b, a
    return a


@njit(cache=True)
def basin_max(lab, val, B):
    """per basin: maximum of val over its points"""
    out = np.zeros(B, np.float32)
    for p in range(len(lab)):
        if val[p] > out[lab[p]]:
            out[lab[p]] = val[p]
    return out


@njit(cache=True, parallel=True)
def link_steepest(nbr, inten, raw):
    """parent[p] = the highest neighbour if it is higher than p, else p. Ties in the surface height
    ``inten`` (two same-scan split pieces share one effective intensity) go to the piece with the
    higher raw centroid intensity ``raw``, then to the lower index: the dominant piece is the root."""
    N = len(inten)
    parent = np.empty(N, np.int32)
    for p in prange(N):
        best = -1                      # -1: no higher neighbour, p is a root
        bI = inten[p]
        bR = raw[p]
        for k in range(4):
            q = nbr[p, k]
            if q < 0:
                continue
            qI = inten[q]
            qR = raw[q]
            if qI > bI or (qI == bI and (qR > bR or (qR == bR and q < (p if best < 0 else best)))):
                best = q
                bI = qI
                bR = qR
        parent[p] = best if best >= 0 else p
    return parent


@njit(cache=True, parallel=True)
def resolve_roots(parent):
    """pointer jumping until every point holds its root"""
    N = len(parent)
    label = parent.copy()
    changed = 1
    while changed > 0:
        changed = 0
        for p in prange(N):
            g = label[label[p]]
            if g != label[p]:
                label[p] = g
                changed += 1
    return label


def dense_labels(label):
    """remap root indices to 0..B-1; returns (lab int32[N], root_idx int64[B])"""
    roots = np.flatnonzero(label == np.arange(len(label), dtype=label.dtype))
    root_id = np.full(len(label), -1, np.int32)
    root_id[roots] = np.arange(len(roots), dtype=np.int32)
    return root_id[label], roots


@njit(cache=True)
def basin_extent(lab, off, B):
    """per basin: first scan, last scan, number of points (serial, O(N))"""
    smin = np.full(B, np.iinfo(np.int32).max, np.int32)
    smax = np.full(B, -1, np.int32)
    npts = np.zeros(B, np.int64)
    S = len(off) - 1
    for s in range(S):
        for p in range(off[s], off[s + 1]):
            b = lab[p]
            if s < smin[b]:
                smin[b] = s
            if s > smax[b]:
                smax[b] = s
            npts[b] += 1
    return smin, smax, npts


@njit(cache=True)
def cell_medians(off, mz, inten, mask, rt, mz_lo, mz_hi, n_bands, block_min, min_points, floor):
    """median intensity of the masked centroids per (log-m/z band, RT block) cell; cells with
    < min_points fall back to ``floor``. Returns (cell_median float64[n_bands, n_blocks], counts)."""
    S = len(off) - 1
    n_blocks = max(1, int(np.ceil((rt[S - 1] - rt[0]) / block_min)))
    lmz_lo = np.log(mz_lo)
    lmz_w = (np.log(mz_hi) - lmz_lo) / n_bands + 1e-12
    counts = np.zeros((n_bands, n_blocks), np.int64)
    for s in range(S):
        blk = min(n_blocks - 1, int((rt[s] - rt[0]) / block_min))
        for p in range(off[s], off[s + 1]):
            if mask[p]:
                band = min(n_bands - 1, max(0, int((np.log(mz[p]) - lmz_lo) / lmz_w)))
                counts[band, blk] += 1
    starts = np.zeros(n_bands * n_blocks + 1, np.int64)
    for i in range(n_bands):
        for j in range(n_blocks):
            starts[i * n_blocks + j + 1] = starts[i * n_blocks + j] + counts[i, j]
    fill = starts[:-1].copy()
    buf = np.empty(starts[-1], np.float32)
    for s in range(S):
        blk = min(n_blocks - 1, int((rt[s] - rt[0]) / block_min))
        for p in range(off[s], off[s + 1]):
            if mask[p]:
                band = min(n_bands - 1, max(0, int((np.log(mz[p]) - lmz_lo) / lmz_w)))
                c = band * n_blocks + blk
                buf[fill[c]] = inten[p]
                fill[c] += 1
    med = np.full((n_bands, n_blocks), floor, np.float64)
    for i in range(n_bands):
        for j in range(n_blocks):
            c = i * n_blocks + j
            n = starts[c + 1] - starts[c]
            if n >= min_points:
                seg = np.sort(buf[starts[c]:starts[c + 1]])
                med[i, j] = 0.5 * (seg[(n - 1) // 2] + seg[n // 2])
    return med, counts


@njit(cache=True, parallel=True)
def short_basin_mask(lab, smin, smax, max_span):
    """per centroid: does its basin span at most max_span scans?"""
    N = len(lab)
    out = np.zeros(N, np.bool_)
    for p in prange(N):
        b = lab[p]
        out[p] = smax[b] - smin[b] + 1 <= max_span
    return out


def noise_cell_medians(off, mz, inten, lab, smin, smax, rt, mz_lo, mz_hi, n_bands, block_min, min_points,
                       floor):
    """median intensity of noise centroids (basins spanning <= 2 scans) per
    (log-m/z band, RT block) cell; cells with < min_points fall back to ``floor``.
    Returns (cell_median float64[n_bands, n_blocks], counts)."""
    return cell_medians(off, mz, inten, short_basin_mask(lab, smin, smax, 2), rt, mz_lo, mz_hi, n_bands,
                        block_min, min_points, floor)


@njit(cache=True, parallel=True)
def sideband_mask(off, mz, inten, cand, strong, ppm, ratio):
    """per candidate centroid: is there a centroid of a real trace (``strong``) at least ``ratio``
    times more intense within +/- ppm in the same scan? Orbitrap sidebands of an intense ion never
    link across scans, so they look like noise to the 3D linking, but they follow that ion."""
    S = len(off) - 1
    out = np.zeros(len(mz), np.bool_)
    for s in prange(S):
        a = off[s]
        b = off[s + 1]
        lo = a
        for p in range(a, b):
            if not cand[p]:
                continue
            lim = mz[p] * ppm * 1e-6
            while lo < b and mz[lo] < mz[p] - lim:
                lo += 1
            q = lo
            while q < b and mz[q] <= mz[p] + lim:
                if q != p and strong[q] and inten[q] >= ratio * inten[p]:
                    out[p] = True
                    break
                q += 1
    return out


@njit(cache=True)
def cell_of(mzv, rtv, rt0, mz_lo, mz_hi, n_bands, n_blocks, block_min):
    lmz_lo = np.log(mz_lo)
    lmz_w = (np.log(mz_hi) - lmz_lo) / n_bands + 1e-12
    band = min(n_bands - 1, max(0, int((np.log(mzv) - lmz_lo) / lmz_w)))
    blk = min(n_blocks - 1, max(0, int((rtv - rt0) / block_min)))
    return band, blk


@njit(cache=True, parallel=True)
def _edge_counts(nbr, lab):
    N = len(lab)
    cnt = np.zeros(N + 1, np.int64)
    for p in prange(N):
        c = 0
        for k in range(4):
            q = nbr[p, k]
            if q >= 0 and lab[q] != lab[p]:
                c += 1
        cnt[p + 1] = c
    return cnt


@njit(cache=True, parallel=True)
def _edge_fill(nbr, lab, inten, inten2, start, ea, eb, es, es2):
    N = len(lab)
    for p in prange(N):
        w = start[p]
        for k in range(4):
            q = nbr[p, k]
            if q >= 0 and lab[q] != lab[p]:
                ea[w] = lab[p]
                eb[w] = lab[q]
                es[w] = min(inten[p], inten[q])
                es2[w] = min(inten2[p], inten2[q])
                w += 1


def saddle_edges(nbr, lab, inten, inten2=None):
    """edges between different basins with saddle = min intensity of the pair; with inten2 also
    the saddle on that second surface (same edges, same order). Returns (ea, eb, es[, es2])."""
    cnt = _edge_counts(nbr, lab)
    start = np.cumsum(cnt)
    M = int(start[-1])
    ea = np.empty(M, np.int32)
    eb = np.empty(M, np.int32)
    es = np.empty(M, np.float32)
    es2 = np.empty(M, np.float32)
    _edge_fill(nbr, lab, inten, inten if inten2 is None else inten2, start[:-1], ea, eb, es, es2)
    if inten2 is None:
        return ea, eb, es
    return ea, eb, es, es2


@njit(cache=True)
def _find(uf, x):
    while uf[x] != x:
        uf[x] = uf[uf[x]]
        x = uf[x]
    return x


@njit(cache=True)
def persistence_merge(ea, eb, es, apex_int, noise_b, rel_frac, k_noise, apex_scan, apex_mz, apex_sig,
                      min_sep_scans, es_s, apex_s_in, far_scans, far_frac):
    """ToMATo: process edges by descending saddle; the lower apex dies into the higher one when
    apex_lower - saddle < max(rel_frac * apex_lower, k_noise * noise), or when the two apexes are
    the same ion (within the pair m/z tolerance) closer than min_sep_scans: two chromatographic
    peaks cannot be that close, so the valley between them is a dip (suppression, AGC), not a
    separation. Apexes at least far_scans apart are two peaks already when the valley on the
    smoothed surface (es_s, per-basin smoothed apex apex_s_in) falls far_frac below the lower
    smoothed apex: a dip inside one peak lies within about one width of its apex, so far apart
    a shallower valley separates; the smoothing keeps single-scan ripple on a tail from counting
    as a valley (far_frac = 0 disables the rule). Returns (root per basin, saddle to the first
    higher survivor per basin, or 0)."""
    B = len(apex_int)
    uf = np.arange(B, dtype=np.int32)
    pers_saddle = np.zeros(B, np.float32)
    seen = np.zeros(B, np.bool_)
    apex_s = apex_s_in.copy()
    order = np.argsort(-es, kind="mergesort")
    for e in order:
        ra = _find(uf, ea[e])
        rb = _find(uf, eb[e])
        if ra == rb:
            continue
        if apex_int[ra] < apex_int[rb] or (apex_int[ra] == apex_int[rb] and ra > rb):
            lower, higher = ra, rb
        else:
            lower, higher = rb, ra
        sad = es[e]
        pers = apex_int[lower] - sad
        tau = max(rel_frac * apex_int[lower], k_noise * noise_b[lower])
        sep = abs(apex_scan[lower] - apex_scan[higher])
        too_close = False
        if sep < min_sep_scans:
            m = apex_mz[higher]
            tol = 3.0 * np.sqrt(apex_sig[lower] ** 2 + apex_sig[higher] ** 2) * m * 1e-6
            too_close = abs(apex_mz[lower] - m) <= tol
        merge = pers < tau or too_close
        if merge and not too_close and far_frac > 0 and sep >= far_scans:
            if apex_s[lower] - es_s[e] >= max(far_frac * apex_s[lower], k_noise * noise_b[lower]):
                merge = False
        if merge:
            uf[lower] = higher
            if apex_s[lower] > apex_s[higher]:
                apex_s[higher] = apex_s[lower]
        elif not seen[lower]:
            seen[lower] = True
            pers_saddle[lower] = sad
    for b in range(B):
        uf[b] = _find(uf, b)
    return uf, pers_saddle


@njit(cache=True)
def sort_by_label(lab, B):
    """stable counting sort: order[boff[b]:boff[b+1]] are basin b's points in scan-major order"""
    N = len(lab)
    boff = np.zeros(B + 1, np.int64)
    for p in range(N):
        boff[lab[p] + 1] += 1
    for b in range(B):
        boff[b + 1] += boff[b]
    fill = boff[:-1].copy()
    order = np.empty(N, np.int64)
    for p in range(N):
        b = lab[p]
        order[fill[b]] = p
        fill[b] += 1
    return order, boff


@njit(cache=True)
def _parabola_r2(t, y, w, n):
    """weighted least-squares parabola through (t, y); returns r^2 (0.5 if n < 4)"""
    if n < 4:
        return 0.5
    # normal equations for [1, t, t^2]
    tm = 0.0
    sw = 0.0
    for i in range(n):
        tm += w[i] * t[i]
        sw += w[i]
    tm /= sw
    s0 = s1 = s2 = s3 = s4 = 0.0
    b0 = b1 = b2 = 0.0
    for i in range(n):
        x = t[i] - tm
        x2 = x * x
        s0 += w[i]
        s1 += w[i] * x
        s2 += w[i] * x2
        s3 += w[i] * x2 * x
        s4 += w[i] * x2 * x2
        b0 += w[i] * y[i]
        b1 += w[i] * y[i] * x
        b2 += w[i] * y[i] * x2
    A = np.empty((3, 3))
    A[0, 0] = s0; A[0, 1] = s1; A[0, 2] = s2
    A[1, 0] = s1; A[1, 1] = s2; A[1, 2] = s3
    A[2, 0] = s2; A[2, 1] = s3; A[2, 2] = s4
    rhs = np.empty(3)
    rhs[0] = b0; rhs[1] = b1; rhs[2] = b2
    det = np.linalg.det(A)
    if abs(det) < 1e-300:
        return 0.5
    c = np.linalg.solve(A, rhs)
    ym = b0 / s0
    ss_res = 0.0
    ss_tot = 0.0
    for i in range(n):
        x = t[i] - tm
        f = c[0] + c[1] * x + c[2] * x * x
        ss_res += w[i] * (y[i] - f) ** 2
        ss_tot += w[i] * (y[i] - ym) ** 2
    if ss_tot <= 0:
        return 0.5
    r2 = 1.0 - ss_res / ss_tot
    if r2 < 0.0:
        r2 = 0.0
    return r2


@njit(cache=True)
def _walk(trace, tidx, nbr, inten, rt, smin, ka, span, cap, lo_thr, step, side):
    """walk from the apex in direction step (-1 left, +1 right) along the per-scan-max trace.
    Stops at the basin end, at two consecutive scans <= lo_thr, or at the width cap.
    Returns (k_end, end_value, hit_cap, floor_value): k_end is the last scan kept inside the
    bounds, end_value the intensity level the walk ended on (the saddle to a neighbouring basin,
    the second low value, the capped value, or 0 when the trace dropped out), floor_value the
    lowest level the trace reached there (the lower of the two low values: 0 where a scan holds
    no centroid; otherwise as end_value)."""
    sa = smin + ka
    k = ka
    low = 0
    last_good = ka
    low_val = 0.0
    low_min = 0.0
    while True:
        kn = k + step
        if kn < 0 or kn >= span:
            # basin exhausted on this side with the last kept scan above lo_thr
            p = tidx[last_good]
            q = nbr[p, side]
            if q >= 0:
                v = min(trace[last_good], inten[q])
                return last_good, v, 0, v
            return last_good, 0.0, 0, 0.0
        if abs(rt[smin + kn] - rt[sa]) > cap:
            return last_good, trace[last_good], 1, trace[last_good]
        k = kn
        v = trace[k]
        if v <= lo_thr:
            low += 1
            low_val = v
            low_min = v if low == 1 else min(low_min, v)
            if low >= 2:
                return last_good, low_val, 0, low_min
        else:
            low = 0
            last_good = k


@njit(cache=True)
def noise_at(level, floor, c, r):
    """expected scan-to-scan sd of the ion current at ``level``: the cell floor plus the file's
    intensity-dependent part, sigma^2 = floor^2 + c * I + r^2 * I^2 (shot-noise-like and
    proportional terms fitted per file from the strongest ions' traces)"""
    return np.sqrt(floor * floor + c * level + r * r * level * level)


@njit(cache=True, parallel=True)
def basin_stats(sel, order, boff, scan_of, nbr, mz, inten, eff, rt, noise_b, fwhm_med, max_width_fwhm,
                noise_c, noise_r, k_valley, rep_raw):
    """descriptors for the selected basins; see COLS for the column order. The per-scan trace is
    the most intense point of the basin in that scan with its same-scan split neighbours summed
    (``eff``), never the sum of every basin point, which would count background centroids.
    With ``rep_raw`` the scan's representative point (its m/z, the apex index) is the basin point
    with the highest raw intensity in that scan: two split pieces share one effective intensity,
    and without this the weaker piece (a second ion linked in as a split partner) can stand for
    the scan and pull the m/z statistics to its strand; ``mz_min`` / ``mz_max`` are then the span
    of the representatives inside the bounds, otherwise of every basin point.
    Shape descriptors inside the bounds, on the present scans: ``tv_ratio`` = total variation of
    the trace / (2 x its single rise and fall), exactly 1 for a unimodal trace; ``n_valleys`` =
    local minima at least k_valley expected noise sd (noise_at, on the difference of the two
    levels) below the maxima on both sides; ``tv_excess`` = total variation beyond the single rise
    and fall, in units of the summed expected scan-to-scan noise."""
    out = np.zeros((len(sel), NCOL), np.float64)
    S = len(rt)
    for ib in prange(len(sel)):
        b = sel[ib]
        pts = order[boff[b]:boff[b + 1]]
        n_points = len(pts)
        smin = scan_of[pts[0]]
        smax = scan_of[pts[n_points - 1]]
        span = smax - smin + 1
        trace = np.zeros(span, np.float64)
        tidx = np.full(span, -1, np.int64)
        traw = np.zeros(span, np.float64)
        mz_min = 1e300
        mz_max = 0.0
        for i in range(n_points):
            p = pts[i]
            k = scan_of[p] - smin
            if rep_raw:
                if eff[p] > trace[k]:
                    trace[k] = eff[p]
                if inten[p] > traw[k]:
                    traw[k] = inten[p]
                    tidx[k] = p
            else:
                if eff[p] > trace[k]:
                    trace[k] = eff[p]
                    tidx[k] = p
                if mz[p] < mz_min:
                    mz_min = mz[p]
                if mz[p] > mz_max:
                    mz_max = mz[p]
        ka = 0
        apex = 0.0
        n_scans = 0
        for k in range(span):
            if trace[k] > 0:
                n_scans += 1
                if trace[k] > apex:
                    apex = trace[k]
                    ka = k
        apex_idx = tidx[ka]
        sa = smin + ka
        noise = noise_b[b]
        lo_thr = max(2.0 * noise, 0.01 * apex)
        cap = max_width_fwhm * fwhm_med
        klo, end_left, cap_l, floor_left = _walk(trace, tidx, nbr, inten, rt, smin, ka, span, cap, lo_thr, -1, NB_LEFT)
        khi, end_right, cap_r, floor_right = _walk(trace, tidx, nbr, inten, rt, smin, ka, span, cap, lo_thr, 1, NB_RIGHT)
        baseline = max(end_left, end_right)
        if rep_raw:
            mz_min = 1e300
            mz_max = 0.0
            for k in range(klo, khi + 1):
                if tidx[k] >= 0:
                    v = mz[tidx[k]]
                    if v < mz_min:
                        mz_min = v
                    if v > mz_max:
                        mz_max = v
        kflags = 0
        if cap_l:
            kflags += KF_CAP_L
        if cap_r:
            kflags += KF_CAP_R
        if smin + klo == 0:
            kflags += KF_RUN_START
        if smin + khi == S - 1:
            kflags += KF_RUN_END
        # area: trapezoid between consecutive present scans inside [klo, khi]
        area = 0.0
        prev_k = -1
        n_gaps = 0
        for k in range(klo, khi + 1):
            if trace[k] > 0:
                if prev_k >= 0:
                    area += 0.5 * (trace[k] + trace[prev_k]) * (rt[smin + k] - rt[smin + prev_k])
                prev_k = k
            else:
                n_gaps += 1
        width = rt[smin + khi] - rt[smin + klo]
        area_bsub = area - 0.5 * (end_left + end_right) * width
        if area_bsub < 0:
            area_bsub = 0.0
        # half-max crossings (gap scans are skipped, never interpolated)
        half = 0.5 * apex
        i = ka
        last = ka
        tl = rt[smin + klo]
        found_l = False
        while i - 1 >= klo:
            i -= 1
            if trace[i] <= 0:
                continue
            if trace[i] < half:
                t1 = rt[smin + i]
                t2 = rt[smin + last]
                tl = t1 + (t2 - t1) * (half - trace[i]) / (trace[last] - trace[i])
                found_l = True
                break
            last = i
        j = ka
        last = ka
        tr = rt[smin + khi]
        found_r = False
        while j + 1 <= khi:
            j += 1
            if trace[j] <= 0:
                continue
            if trace[j] < half:
                t1 = rt[smin + last]
                t2 = rt[smin + j]
                tr = t1 + (t2 - t1) * (trace[last] - half) / (trace[last] - trace[j])
                found_r = True
                break
            last = j
        if not found_l:
            kflags += KF_NOHALF_L
        if not found_r:
            kflags += KF_NOHALF_R
        fwhm = tr - tl
        ta = rt[sa]
        asym = (tr - ta) / (ta - tl) if ta - tl > 0 else 0.0
        # rt: weighted centroid of scans >= 90% apex; flat-top count within 2%
        num = 0.0
        den = 0.0
        n_flat = 0.0
        for k in range(klo, khi + 1):
            v = trace[k]
            if v >= 0.9 * apex:
                num += v * rt[smin + k]
                den += v
            if v >= 0.98 * apex:
                n_flat += 1.0
        rt_c = num / den if den > 0 else ta
        # m/z: intensity-weighted over per-scan maxima >= 50% apex
        num = 0.0
        den = 0.0
        for k in range(klo, khi + 1):
            v = trace[k]
            if v >= 0.5 * apex:
                num += v * mz[tidx[k]]
                den += v
        mzw = num / den
        var = 0.0
        for k in range(klo, khi + 1):
            v = trace[k]
            if v >= 0.5 * apex:
                d = (mz[tidx[k]] - mzw) / mzw * 1e6
                var += v * d * d
        mz_sd = np.sqrt(var / den)
        # gaussian r2 of log intensity over scans >= 10% apex inside the bounds
        n = 0
        for k in range(klo, khi + 1):
            if trace[k] >= 0.1 * apex:
                n += 1
        tt = np.empty(n)
        yy = np.empty(n)
        ww = np.empty(n)
        n = 0
        for k in range(klo, khi + 1):
            if trace[k] >= 0.1 * apex:
                tt[n] = rt[smin + k]
                yy[n] = np.log(trace[k])
                ww[n] = trace[k]
                n += 1
        r2 = _parabola_r2(tt, yy, ww, n)
        # unimodality: total variation of the present-scan trace inside the bounds vs one rise + fall
        npres = 0
        for k in range(klo, khi + 1):
            if trace[k] > 0:
                npres += 1
        tv_ratio = 1.0
        n_valleys = 0.0
        tv_excess = 0.0
        if npres >= 3:
            pres = np.empty(npres)
            i2 = 0
            for k in range(klo, khi + 1):
                if trace[k] > 0:
                    pres[i2] = trace[k]
                    i2 += 1
            tv = 0.0
            sig_sum = 0.0
            for i2 in range(npres - 1):
                tv += abs(pres[i2 + 1] - pres[i2])
                s1 = noise_at(pres[i2], noise, noise_c, noise_r)
                s2 = noise_at(pres[i2 + 1], noise, noise_c, noise_r)
                sig_sum += np.sqrt(s1 * s1 + s2 * s2)
            prom = apex - min(pres[0], pres[npres - 1])
            if prom > 0:
                tv_ratio = tv / (2.0 * prom)
            if sig_sum > 0:
                tv_excess = max(tv - 2.0 * prom, 0.0) / sig_sum
            # significant valleys: local minima far enough below the maxima on both sides
            for i2 in range(1, npres - 1):
                if pres[i2] < pres[i2 - 1] and pres[i2] <= pres[i2 + 1]:
                    ml = pres[0]
                    for j2 in range(1, i2):
                        if pres[j2] > ml:
                            ml = pres[j2]
                    mr = pres[npres - 1]
                    for j2 in range(i2 + 1, npres - 1):
                        if pres[j2] > mr:
                            mr = pres[j2]
                    mm = min(ml, mr)
                    sv = noise_at(pres[i2], noise, noise_c, noise_r)
                    sm = noise_at(mm, noise, noise_c, noise_r)
                    if mm - pres[i2] >= k_valley * np.sqrt(sv * sv + sm * sm):
                        n_valleys += 1.0
        o = out[ib]
        o[0] = apex_idx
        o[1] = sa
        o[2] = smin
        o[3] = smax
        o[4] = smin + klo
        o[5] = smin + khi
        o[6] = n_scans
        o[7] = n_points
        o[8] = n_gaps
        o[9] = apex
        o[10] = mzw
        o[11] = mz_sd
        o[12] = mz_min
        o[13] = mz_max
        o[14] = rt_c
        o[15] = ta
        o[16] = rt[smin + klo]
        o[17] = rt[smin + khi]
        o[18] = area
        o[19] = area_bsub
        o[20] = baseline
        o[21] = fwhm
        o[22] = asym
        o[23] = r2
        o[24] = n_flat
        o[25] = kflags
        o[26] = min(floor_left, floor_right)
        o[27] = tv_ratio
        o[28] = n_valleys
        o[29] = tv_excess
    return out


@njit(cache=True, parallel=True)
def max_in_scan(off, mz, inten, q_mz, q_tol, q_scan):
    """most intense centroid within +/- q_tol of q_mz in scan q_scan, per query (0 if none)"""
    n = len(q_mz)
    S = len(off) - 1
    out = np.zeros(n, np.float32)
    for i in prange(n):
        s = q_scan[i]
        if s < 0 or s >= S:
            continue
        a = off[s]
        b = off[s + 1]
        lo = a
        hi = b
        # lower bound
        v = q_mz[i] - q_tol[i]
        while lo < hi:
            m = (lo + hi) >> 1
            if mz[m] < v:
                lo = m + 1
            else:
                hi = m
        v2 = q_mz[i] + q_tol[i]
        best = np.float32(0.0)
        p = lo
        while p < b and mz[p] <= v2:
            if inten[p] > best:
                best = inten[p]
            p += 1
        out[i] = best
    return out


SL_N, SL_Q90, SL_Q25, SL_MED, SL_NOISE, SL_MEDNZ = 0, 1, 2, 3, 4, 5


@njit(cache=True, parallel=True)
def side_levels(off, mz, inten, q_mz, q_tol, s_apex, near, far):
    """The ion's chromatogram beside each query: left = scans [apex - far, apex - near], right =
    [apex + near, apex + far], clipped to the run; per scan the most intense centroid within
    +/- q_tol (empty scans count as 0). Per query and side: number of in-run scans, 90th and 25th
    percentile, median, the scan-to-scan noise (75th percentile of |first difference| / 1.63,
    the sd for Gaussian noise; 0 on an empty baseline, the ripple amplitude on a zig-zag), and the
    median of the scans in which the ion is present (0 if none: the ion's typical level out there).
    Returns float64[n, 2, 6] indexed by SL_*."""
    n = len(q_mz)
    S = len(off) - 1
    out = np.zeros((n, 2, 6), np.float64)
    for i in prange(n):
        for side in range(2):
            if side == 0:
                a0 = s_apex[i] - far[i]
                a1 = s_apex[i] - near[i] + 1
            else:
                a0 = s_apex[i] + near[i]
                a1 = s_apex[i] + far[i] + 1
            a0 = max(0, a0)
            a1 = min(S, a1)
            if a1 <= a0:
                continue
            m = a1 - a0
            vals = np.zeros(m, np.float64)
            for s in range(a0, a1):
                a = off[s]
                b = off[s + 1]
                lo = a
                hi = b
                v = q_mz[i] - q_tol[i]
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if mz[mid] < v:
                        lo = mid + 1
                    else:
                        hi = mid
                v2 = q_mz[i] + q_tol[i]
                best = 0.0
                p = lo
                while p < b and mz[p] <= v2:
                    if inten[p] > best:
                        best = inten[p]
                    p += 1
                vals[s - a0] = best
            if m >= 2:
                d = np.empty(m - 1, np.float64)
                for k in range(m - 1):
                    d[k] = abs(vals[k + 1] - vals[k])
                d.sort()
                out[i, side, SL_NOISE] = d[min(m - 2, int(0.75 * (m - 2) + 0.5))] / 1.63
            vals.sort()
            out[i, side, SL_N] = m
            out[i, side, SL_Q90] = vals[min(m - 1, int(0.90 * (m - 1) + 0.5))]
            out[i, side, SL_Q25] = vals[min(m - 1, int(0.25 * (m - 1) + 0.5))]
            out[i, side, SL_MED] = 0.5 * (vals[(m - 1) // 2] + vals[m // 2])
            z = 0
            while z < m and vals[z] <= 0.0:
                z += 1
            if z < m:
                mm = m - z
                out[i, side, SL_MEDNZ] = 0.5 * (vals[z + (mm - 1) // 2] + vals[z + mm // 2])
    return out


def gaussian_kernel(fwhm_scans):
    """normalised Gaussian smoothing kernel with the given FWHM in scans, truncated at 4 sd (odd length)"""
    sd = max(float(fwhm_scans), 1e-6) / 2.3548200450309493
    r = int(4.0 * sd + 0.5)
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sd) ** 2)
    return k / k.sum()


ES_TOP, ES_SCAN, ES_APEX, ES_PROM, ES_NOISE = 0, 1, 2, 3, 4


@njit(cache=True, parallel=True)
def eic_shape(off, mz, inten, q_mz, q_tol, s_apex, s_lo, s_hi, half_win, kern):
    """A 1D view of each query's ion, independent of the basin: the EIC (per-scan maximum within
    +/- q_tol) over +/- half_win scans around the apex, smoothed with ``kern`` (odd length,
    normalised, edge values repeated). Inside the query's bounds [s_lo, s_hi]: the smoothed
    maximum (height, scan), whether it is a local maximum of the smoothed trace (0 = the bounds
    hold a slice of a slope: a tail, ramp, step or a neighbour's flank), its topographic prominence
    within the window, and the residual scan-to-scan noise of the raw trace outside the bounds
    (1.4826 x MAD of raw - smoothed; the whole window when fewer than 10 scans lie outside).
    Returns float64[n, 5] indexed by ES_*."""
    n = len(q_mz)
    S = len(off) - 1
    K2 = len(kern) // 2
    out = np.zeros((n, 5), np.float64)
    for i in prange(n):
        a0 = max(0, s_apex[i] - half_win)
        a1 = min(S, s_apex[i] + half_win + 1)
        m = a1 - a0
        if m < 3:
            continue
        e = np.zeros(m, np.float64)
        for s in range(a0, a1):
            a = off[s]
            b = off[s + 1]
            lo = a
            hi = b
            v = q_mz[i] - q_tol[i]
            while lo < hi:
                mid = (lo + hi) >> 1
                if mz[mid] < v:
                    lo = mid + 1
                else:
                    hi = mid
            v2 = q_mz[i] + q_tol[i]
            best = 0.0
            p = lo
            while p < b and mz[p] <= v2:
                if inten[p] > best:
                    best = inten[p]
                p += 1
            e[s - a0] = best
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
        b0 = max(0, s_lo[i] - a0)
        b1 = min(m - 1, s_hi[i] - a0)
        if b1 < b0:
            continue
        k = b0
        for j in range(b0, b1 + 1):
            if es[j] > es[k]:
                k = j
        top = es[k]
        left_ok = (k == 0) or (es[k] >= es[k - 1])
        right_ok = (k == m - 1) or (es[k] >= es[k + 1])
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
        out[i, ES_TOP] = top
        out[i, ES_SCAN] = a0 + k
        out[i, ES_APEX] = 1.0 if (left_ok and right_ok) else 0.0
        out[i, ES_PROM] = prom
        out[i, ES_NOISE] = noise
    return out


@njit(cache=True)
def duplicate_mask(mz_s, scan_s, height_s, ppm, max_dscan):
    """inputs sorted by m/z: True where another feature of the same ion (within ppm) sits on the
    same smoothed maximum (scan within max_dscan) with a larger height: the weaker one duplicates
    the stronger one's peak"""
    n = len(mz_s)
    out = np.zeros(n, np.bool_)
    for i in range(n):
        lim = mz_s[i] * ppm * 1e-6
        j = i - 1
        while j >= 0 and mz_s[i] - mz_s[j] <= lim:
            if abs(scan_s[j] - scan_s[i]) <= max_dscan and height_s[j] > height_s[i]:
                out[i] = True
                break
            j -= 1
        if out[i]:
            continue
        j = i + 1
        while j < n and mz_s[j] - mz_s[i] <= lim:
            if abs(scan_s[j] - scan_s[i]) <= max_dscan and height_s[j] > height_s[i]:
                out[i] = True
                break
            j += 1
    return out


@njit(cache=True, parallel=True)
def window_level(off, mz, inten, q_mz, q_tol, s_lo, s_hi, q):
    """quantile q over scans s_lo..s_hi-1 of the most intense centroid within +/- q_tol of q_mz
    (empty scans count as 0); 0 when the window is empty. An upper quantile (0.75) sees an ion
    that is present in only part of the scans, which a median does not."""
    n = len(q_mz)
    S = len(off) - 1
    out = np.zeros(n, np.float64)
    for i in prange(n):
        a0 = max(0, s_lo[i])
        a1 = min(S, s_hi[i])
        if a1 <= a0:
            continue
        vals = np.zeros(a1 - a0, np.float64)
        for s in range(a0, a1):
            a = off[s]
            b = off[s + 1]
            lo = a
            hi = b
            v = q_mz[i] - q_tol[i]
            while lo < hi:
                m = (lo + hi) >> 1
                if mz[m] < v:
                    lo = m + 1
                else:
                    hi = m
            v2 = q_mz[i] + q_tol[i]
            best = 0.0
            p = lo
            while p < b and mz[p] <= v2:
                if inten[p] > best:
                    best = inten[p]
                p += 1
            vals[s - a0] = best
        vals.sort()
        out[i] = vals[min(len(vals) - 1, int(q * (len(vals) - 1) + 0.5))]
    return out
