"""EI deconvolution of one GC-MS run: picked ions -> components (compounds) and their spectra.

All fragment ions of one compound leave the column together: on a TOF or Orbitrap (no scan skew) they share one apex
time and one peak shape. Two compounds that co-elute differ in apex time, often by less than a peak width, and can
both produce the same ion.

1. Apex time of every picked ion below the scan spacing: a parabola through the logarithm of its chromatogram's top
   (the points above half the apex, weighted by intensity).
2. The file calibrates its own rules on the isotope pairs the picker annotated (M and M+1 of one fragment: known to
   be one compound), per S/N class of the weaker ion, because a weak ion's apex is less certain and its chromatogram
   noisier: the apex-time tolerance is 3 robust sd of the pairs' apex-time differences, the shape cut the
   chromatogram correlation that 99 % of the pairs reach. Files with too few pairs use fixed defaults.
3. Components: the strongest unassigned ion is a seed, and the unassigned ions whose apex lies within the tolerance
   of its apex are its neighbourhood. The reference ion of the component is, of the seed and the next strongest of
   the neighbourhood, the one most of the others correlate with. Ions at the reference's apex time whose
   chromatogram correlates with it are tight members; the others are loose members: the apex time, not the
   shape, separates two compounds, so two components never start at one apex time. (They did in the first
   version, and the next step then divided a compound between two near-identical models: on the MRH set 27 % of
   the truth compounds had ions outside their main component, 20 % now.) Each component gets a model profile,
   the per-scan median of its strongest tight members' normalised chromatograms (one member distorted by a
   neighbour does not bend it); every ion is then placed with the model it correlates with best among the
   components within its tolerance, or stays a loose member of the nearest one.
4. Composite components. An ion shared by two compounds that elute within a peak width is picked as one peak with
   its apex between theirs, where the two contributions put it, so each shared ion (with its isotopes) forms a small
   component of its own. A component of a few ions whose model is a non-negative mix of the models of the nearest
   larger component on each side, both contributing, is dissolved: its ions are shared. Peak shape alone cannot
   tell such a mix from a compound eluting between the two (the mix of two peaks half a width apart correlates
   0.99 with a single peak), hence the size rule: a real compound with very few picked ions that elutes between two
   larger ones within a peak width is dissolved too.
5. Spectra from the raw centroids, not from the picked ions. Every centroid of a component's apex scan is a
   candidate ion; its chromatogram over the component's bounds is fitted with a constant background plus the models
   of the component and of its overlapping neighbours, non-negative, by forward selection: the best single model
   first, another one only when it lowers the residual significantly (F test). The ion's abundance in a component
   is that model's coefficient (apex height above background). So a fragment too weak to be picked is in the
   spectrum, a shared ion is divided, and a background ion (column bleed) goes to the constant.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np
import pandas as pd
from numba import njit

from ..estimate import Params
from ..io import Cloud, _lower_bound, _upper_bound, eic_kernel

TOL_MIN_PPM = 5.0          # an ion's chromatogram is read within max(3 sd + 1, 5) ppm, as in the picker's checks
ISO_KEEP = 0.99            # the shape cut keeps this share of the isotope (known same-compound) pairs: every ion the
                           # cut turns away becomes a one-ion component beside its compound, and the apex-time
                           # tolerance, not the shape, is what separates two compounds
SHAPE_FLOOR, SHAPE_CEIL = 0.5, 0.95
MIN_PAIRS = 20             # isotope pairs needed to calibrate; per S/N class as well


@dataclass
class DeconvParams:
    apex_tol: float = float("nan")     # apex-time tolerance, minutes (nan: calibrated on isotope pairs)
    shape_min: float = float("nan")    # chromatogram correlation with the seed / model (nan: calibrated)
    tau_k: float = 3.0                 # tolerance = this many robust sd of the isotope pairs' apex-time differences
    model_ions: int = 5                # strongest members whose chromatograms make the model profile
    max_neighbours: int = 4            # overlapping neighbour models in one ion's fit
    f_min: float = 10.0                # F statistic a model must reach to enter an ion's fit
    min_rel: float = 1e-3              # spectrum ions below this fraction of the base peak are dropped
    min_points: int = 3                # an ion must be present in this many scans of the component's bounds
    min_ions: int = 3                  # a component whose spectrum holds fewer ions is not reported
    composite_share: float = 0.2       # a composite needs each neighbour to explain at least this share of its model
    composite_max_ions: int = 6        # ... and has at most this many picked ions (one shared ion and its isotopes)
    composite: bool = True
    ref_ions: int = 5                  # the reference ion of a component is, of the strongest free ion and the next
                                       # ref_ions - 1 at its apex time, the one most of them correlate with (1: the
                                       # strongest, whatever its shape)
    attach: bool = True                # an ion at a component's apex time whose chromatogram does not fit stays with
                                       # that component as a loose member (False: it starts a component of its own)
    min_features: int = 1              # a component with fewer picked ions that fit its model is not reported


@dataclass
class Deconv:
    """components: one row per component, by apex time. Spectrum of component c: mz[off[c]:off[c+1]] (ascending),
    abundance, and shared (True where the ion's chromatogram needed another component's model too).
    feature_component / feature_r: per row of the feature table, the component (-1: none) and the correlation
    with its model; feature_shared marks the members of dissolved composite components."""
    components: pd.DataFrame
    off: np.ndarray
    mz: np.ndarray
    abundance: np.ndarray
    shared: np.ndarray
    feature_component: np.ndarray
    feature_r: np.ndarray
    feature_shared: np.ndarray
    info: dict
    feature_loose: np.ndarray = None    # at its component's apex time, chromatogram not fitting the model

    def spectrum(self, c: int):
        a, b = self.off[c], self.off[c + 1]
        return self.mz[a:b], self.abundance[a:b]


def params_from_dict(d: dict) -> Params:
    """the picker's Params from a features/<stem>.params.json dictionary"""
    names = {f.name for f in fields(Params)}
    kw = {k: (np.asarray(v, np.float64) if isinstance(v, list) else v) for k, v in d.items() if k in names}
    return Params(**kw)


def ion_tol(mz, inten, P: Params) -> np.ndarray:
    """half-width (Da) of the m/z window an ion of this intensity is read in"""
    return np.asarray(mz, np.float64) * np.maximum(3.0 * P.sig_ppm(inten) + 1.0, TOL_MIN_PPM) * 1e-6


# ------------------------------------------------------------------ kernels --
@njit(cache=True)
def _pearson(a, b):
    n = len(a)
    sa = sb = saa = sbb = sab = 0.0
    for k in range(n):
        x = float(a[k]); y = float(b[k])
        sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y
    if n < 3:
        return 0.0
    va = saa - sa * sa / n
    vb = sbb - sb * sb / n
    if va <= 0 or vb <= 0:
        return 0.0
    return (sab - sa * sb / n) / np.sqrt(va * vb)


@njit(cache=True)
def _log_parabola_apex(y, k, x):
    """apex position (in x units) of the peak whose highest point is y[k]: weighted least-squares parabola through
    ln y over the contiguous points above half the top plus one point on each side; nan when it cannot be placed"""
    n = len(y)
    top = y[k]
    if top <= 0:
        return np.nan
    lo = k
    while lo > 0 and y[lo - 1] >= 0.5 * top and y[lo - 1] <= top:
        lo -= 1
    hi = k
    while hi < n - 1 and y[hi + 1] >= 0.5 * top and y[hi + 1] <= top:
        hi += 1
    if lo > 0 and 0 < y[lo - 1] <= top:
        lo -= 1
    if hi < n - 1 and 0 < y[hi + 1] <= top:
        hi += 1
    if hi - lo + 1 < 3:
        return np.nan
    s0 = s1 = s2 = s3 = s4 = t0 = t1 = t2 = 0.0
    for q in range(lo, hi + 1):
        w = float(y[q])
        u = x[q] - x[k]
        l = np.log(w)
        s0 += w; s1 += w * u; s2 += w * u * u; s3 += w * u * u * u; s4 += w * u * u * u * u
        t0 += w * l; t1 += w * u * l; t2 += w * u * u * l
    # normal equations of l = c0 + c1 u + c2 u^2
    det = s0 * (s2 * s4 - s3 * s3) - s1 * (s1 * s4 - s3 * s2) + s2 * (s1 * s3 - s2 * s2)
    if det == 0.0:
        return np.nan
    c1 = (s0 * (t1 * s4 - s3 * t2) - t0 * (s1 * s4 - s3 * s2) + s2 * (s1 * t2 - t1 * s2)) / det
    c2 = (s0 * (s2 * t2 - t1 * s3) - s1 * (s1 * t2 - t1 * s2) + t0 * (s1 * s3 - s2 * s2)) / det
    if c2 >= 0.0:
        return np.nan
    u = -c1 / (2.0 * c2)
    if u < x[lo] - x[k] or u > x[hi] - x[k]:
        return np.nan
    return x[k] + u


@njit(cache=True)
def _apex_times(off, mz, inten, rt, f_mz, f_tol, f_apex, half, dt):
    """sub-scan apex time (minutes) of every feature; the scan time where the parabola cannot be placed"""
    n = len(f_mz)
    S = len(rt)
    out = np.empty(n)
    for i in range(n):
        a = f_apex[i]
        s0 = max(0, a - half)
        s1 = min(S, a + half + 1)
        y = eic_kernel(off, mz, inten, f_mz[i], f_tol[i], s0, s1)
        k = a - s0
        # the picker's apex is on the summed surface; take the trace's own maximum next to it
        if k > 0 and y[k - 1] > y[k]:
            k -= 1
        elif k < len(y) - 1 and y[k + 1] > y[k]:
            k += 1
        t = _log_parabola_apex(y, k, (rt[s0:s1] - rt[a]) / dt)
        out[i] = rt[a] + t * dt if np.isfinite(t) else rt[a]
    return out


@njit(cache=True)
def _pair_r(off, mz, inten, f_mz, f_tol, f_lo, f_hi, a, b):
    """chromatogram correlation of feature pairs (a[k], b[k]) over the bounds of b[k]"""
    out = np.empty(len(a))
    for k in range(len(a)):
        i = a[k]; j = b[k]
        s0 = f_lo[j]; s1 = f_hi[j] + 1
        out[k] = _pearson(eic_kernel(off, mz, inten, f_mz[i], f_tol[i], s0, s1),
                          eic_kernel(off, mz, inten, f_mz[j], f_tol[j], s0, s1))
    return out


@njit(cache=True)
def _cluster(off, mz, inten, f_mz, f_tol, f_h, f_t, f_lo, f_hi, f_tau, f_rcut, order_h, order_t, t_sorted, tau_max,
             k_ref, attach):
    """Star clustering. Seeds in descending height; the free ions whose apex is within the larger of the two
    tolerances of the seed's are its neighbourhood. The component's reference ion is, of the seed and the k_ref - 1
    strongest of the neighbourhood, the one most of the others correlate with (chromatograms over the reference's
    bounds, the lower of the two cuts), the stronger on a tie: a clipped or disturbed strongest ion does not define
    the shape. Free ions at the reference's apex time that correlate with it are tight members; the others are
    loose members when `attach` (the apex time, not the shape, separates compounds: two components do not start
    at one apex time), else stay free. Returns (component, tight, correlation with the reference, reference ions)."""
    n = len(f_mz)
    comp = np.full(n, -1, np.int64)
    tight = np.zeros(n, np.bool_)
    r_seed = np.zeros(n)
    seeds = np.empty(n, np.int64)
    cand = np.empty(n, np.int64)
    refs = np.empty(max(k_ref, 1), np.int64)
    C = 0
    for q in range(n):
        s = order_h[q]
        while comp[s] < 0:
            a = _lower_bound(t_sorted, 0, n, f_t[s] - tau_max)
            b = _upper_bound(t_sorted, a, n, f_t[s] + tau_max)
            m = 0
            for k in range(a, b):
                j = order_t[k]
                if comp[j] < 0 and j != s and abs(f_t[j] - f_t[s]) <= max(f_tau[j], f_tau[s]):
                    cand[m] = j
                    m += 1
            ref = s
            if k_ref > 1 and m > 0:
                refs[0] = s
                nr = 1
                for _ in range(k_ref - 1):
                    best = -1
                    for k in range(m):
                        j = cand[k]
                        taken = False
                        for u in range(nr):
                            if refs[u] == j:
                                taken = True
                        if not taken and (best < 0 or f_h[j] > f_h[best]):
                            best = j
                    if best < 0:
                        break
                    refs[nr] = best
                    nr += 1
                best_n = -1
                for u in range(nr):
                    t = refs[u]
                    yt = eic_kernel(off, mz, inten, f_mz[t], f_tol[t], f_lo[t], f_hi[t] + 1)
                    cnt = 0
                    for k in range(m + 1):
                        j = cand[k] if k < m else s
                        if j == t or abs(f_t[j] - f_t[t]) > max(f_tau[j], f_tau[t]):
                            continue
                        r = _pearson(yt, eic_kernel(off, mz, inten, f_mz[j], f_tol[j], f_lo[t], f_hi[t] + 1))
                        if r >= min(f_rcut[j], f_rcut[t]):
                            cnt += 1
                    if cnt > best_n:
                        best_n = cnt
                        ref = t
            c = C
            seeds[C] = ref
            C += 1
            comp[ref] = c
            tight[ref] = True
            r_seed[ref] = 1.0
            s0 = f_lo[ref]; s1 = f_hi[ref] + 1
            ys = eic_kernel(off, mz, inten, f_mz[ref], f_tol[ref], s0, s1)
            a = _lower_bound(t_sorted, 0, n, f_t[ref] - tau_max)
            b = _upper_bound(t_sorted, a, n, f_t[ref] + tau_max)
            for k in range(a, b):
                j = order_t[k]
                if comp[j] >= 0 or abs(f_t[j] - f_t[ref]) > max(f_tau[j], f_tau[ref]):
                    continue
                r = _pearson(ys, eic_kernel(off, mz, inten, f_mz[j], f_tol[j], s0, s1))
                if r >= min(f_rcut[j], f_rcut[ref]):
                    comp[j] = c
                    tight[j] = True
                    r_seed[j] = r
                elif attach:
                    comp[j] = c
                    r_seed[j] = r
    return comp, tight, r_seed, seeds[:C]


@njit(cache=True)
def _attach(f_t, f_tau, f_skip, comp, c_t, c_tau, c_alive):
    """features no model fits join, as loose members, the live component nearest in apex time within the tolerance
    (components sorted by apex time); returns the loose mask and writes comp"""
    n = len(f_t)
    C = len(c_t)
    loose = np.zeros(n, np.bool_)
    if C == 0 or n == 0:
        return loose
    tau_max = max(f_tau.max(), c_tau.max())
    for j in range(n):
        if comp[j] >= 0 or f_skip[j]:
            continue
        a = _lower_bound(c_t, 0, C, f_t[j] - tau_max)
        b = _upper_bound(c_t, a, C, f_t[j] + tau_max)
        best = -1
        bd = np.inf
        for c in range(a, b):
            d = abs(c_t[c] - f_t[j])
            if c_alive[c] and d <= max(f_tau[j], c_tau[c]) and d < bd:
                bd = d
                best = c
        if best >= 0:
            comp[j] = best
            loose[j] = True
    return loose


@njit(cache=True)
def _models(off, mz, inten, f_mz, f_tol, c_lo, c_hi, c_apex, mem_off, mem_idx, k_top):
    """model profile of every component over its bounds, maximum 1: per-scan median (mean below three) of the
    chromatograms of its k_top strongest members, each divided by its sum over the three apex scans"""
    C = len(c_lo)
    m_off = np.zeros(C + 1, np.int64)
    for c in range(C):
        m_off[c + 1] = m_off[c] + (c_hi[c] - c_lo[c] + 1)
    m_val = np.zeros(m_off[C])
    for c in range(C):
        s0 = c_lo[c]; s1 = c_hi[c] + 1
        W = s1 - s0
        k = min(k_top, mem_off[c + 1] - mem_off[c])
        T = np.zeros((max(k, 1), W))
        kk = 0
        a = c_apex[c] - s0
        for q in range(k):
            j = mem_idx[mem_off[c] + q]
            y = eic_kernel(off, mz, inten, f_mz[j], f_tol[j], s0, s1)
            nrm = float(y[a])
            if a > 0:
                nrm += y[a - 1]
            if a + 1 < W:
                nrm += y[a + 1]
            if nrm <= 0:
                continue
            for w in range(W):
                T[kk, w] = y[w] / nrm
            kk += 1
        if kk == 0:
            continue
        top = 0.0
        for w in range(W):
            if kk >= 3:
                v = np.median(T[:kk, w])
            else:
                v = T[:kk, w].mean()
            m_val[m_off[c] + w] = v
            top = max(top, v)
        if top > 0:
            for w in range(W):
                m_val[m_off[c] + w] /= top
    return m_off, m_val


@njit(cache=True)
def _reassign(off, mz, inten, f_mz, f_tol, f_t, f_tau, f_rcut, f_skip, c_t, c_tau, c_lo, c_hi, c_alive, m_off, m_val):
    """every feature goes to the live component (apex within tolerance) whose model its chromatogram correlates
    with best, when that reaches the cut; components are sorted by apex time"""
    n = len(f_mz)
    C = len(c_t)
    comp = np.full(n, -1, np.int64)
    r_out = np.zeros(n)
    if C == 0:
        return comp, r_out
    tau_max = max(f_tau.max(), c_tau.max()) if n else 0.0
    for j in range(n):
        if f_skip[j]:
            continue
        a = _lower_bound(c_t, 0, C, f_t[j] - tau_max)
        b = _upper_bound(c_t, a, C, f_t[j] + tau_max)
        best = -1
        best_r = f_rcut[j]
        for c in range(a, b):
            if not c_alive[c] or abs(c_t[c] - f_t[j]) > max(f_tau[j], c_tau[c]):
                continue
            y = eic_kernel(off, mz, inten, f_mz[j], f_tol[j], c_lo[c], c_hi[c] + 1)
            r = _pearson(y, m_val[m_off[c]:m_off[c + 1]])
            if r >= best_r:
                best_r = r
                best = c
        comp[j] = best
        if best >= 0:
            r_out[j] = best_r
    return comp, r_out


@njit(cache=True)
def _solve(G, g, idx, m):
    """least squares on the columns idx[:m] from the Gram matrix: coefficients and the explained sum of squares;
    ok False when the system is singular"""
    A = np.empty((m, m))
    b = np.empty(m)
    for i in range(m):
        b[i] = g[idx[i]]
        for j in range(m):
            A[i, j] = G[idx[i], idx[j]]
    for i in range(m):                      # Gaussian elimination with partial pivoting
        p = i
        for r in range(i + 1, m):
            if abs(A[r, i]) > abs(A[p, i]):
                p = r
        if abs(A[p, i]) < 1e-12 * (abs(G[idx[i], idx[i]]) + 1e-300):
            return b, 0.0, False
        if p != i:
            for j in range(m):
                A[i, j], A[p, j] = A[p, j], A[i, j]
            b[i], b[p] = b[p], b[i]
        for r in range(i + 1, m):
            f = A[r, i] / A[i, i]
            for j in range(i, m):
                A[r, j] -= f * A[i, j]
            b[r] -= f * b[i]
    x = np.empty(m)
    for i in range(m - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, m):
            s -= A[i, j] * x[j]
        x[i] = s / A[i, i]
    ex = 0.0
    for i in range(m):
        ex += x[i] * g[idx[i]]
    return x, ex, True


@njit(cache=True)
def _fit_nonneg(G, g, yy, models, nm):
    """constant + models[:nm], all coefficients non-negative: (coefficients of the models, sse, uses constant, ok).
    A negative constant is dropped and the fit repeated; a non-positive model coefficient fails the fit."""
    idx = np.empty(nm + 1, np.int64)
    idx[0] = 0
    for q in range(nm):
        idx[q + 1] = models[q]
    x, ex, ok = _solve(G, g, idx, nm + 1)
    const = True
    if ok and x[0] < 0.0:
        x2, ex, ok = _solve(G, g, idx[1:], nm)
        const = False
        x = np.empty(nm + 1)
        x[0] = 0.0
        for q in range(nm):
            x[q + 1] = x2[q] if ok else 0.0
    if not ok:
        return x[1:], yy, const, False
    for q in range(nm):
        if x[q + 1] <= 0.0:
            return x[1:], yy, const, False
    return x[1:], max(yy - ex, 1e-12 * yy), const, True


@njit(cache=True)
def _fit_ions(off, mz, inten, cand, cand_tol, s0, s1, X, f_min, min_points, max_models):
    """Every candidate centroid (index cand[k] in the apex scan, window cand_tol[k]) fitted over scans s0..s1-1 with
    the columns of X: 0 the constant, 1 this component's model, 2.. its neighbours'. Returns per candidate the
    abundance in this component (0: not its ion), the number of models in the fit and the intensity-weighted m/z."""
    n = len(cand)
    W = s1 - s0
    K = X.shape[1] - 1
    G = X.T @ X
    ab = np.zeros(n)
    nmod = np.zeros(n, np.int64)
    mzw = np.zeros(n)
    y = np.empty(W)
    g = np.empty(K + 1)
    sel = np.empty(max_models, np.int64)
    trial = np.empty(max_models, np.int64)
    for k in range(n):
        p = cand[k]
        tol = cand_tol[k]
        target = mz[p]
        # a stronger centroid of the same scan inside this window is the same ion: skip the weaker one
        dup = False
        q = p - 1
        while q >= 0 and target - mz[q] <= tol and mz[q] <= target:
            if inten[q] > inten[p]:
                dup = True
            q -= 1
        q = p + 1
        while q < len(mz) and mz[q] - target <= tol and mz[q] >= target:
            if inten[q] >= inten[p]:
                dup = True
            q += 1
        if dup:
            continue
        npos = 0
        sw = 0.0
        swm = 0.0
        yy = 0.0
        for s in range(s0, s1):
            a = off[s]; b = off[s + 1]
            lo = _lower_bound(mz, a, b, target - tol)
            hi = _upper_bound(mz, lo, b, target + tol)
            best = 0.0
            bm = 0.0
            for r in range(lo, hi):
                if inten[r] > best:
                    best = inten[r]
                    bm = mz[r]
            y[s - s0] = best
            if best > 0:
                npos += 1
                sw += best
                swm += best * bm
                yy += best * best
        if npos < min_points:
            continue
        for c in range(K + 1):
            acc = 0.0
            for w in range(W):
                acc += X[w, c] * y[w]
            g[c] = acc
        # constant only
        b0 = g[0] / G[0, 0]
        sse = max(yy - b0 * g[0], 1e-12 * yy)
        npar = 1
        ns = 0
        coef = np.zeros(max_models)
        for step in range(min(K, max_models)):
            best_c = -1
            best_sse = sse
            best_const = True
            best_x = coef
            for c in range(1, K + 1):
                used = False
                for q in range(ns):
                    if sel[q] == c:
                        used = True
                if used:
                    continue
                for q in range(ns):
                    trial[q] = sel[q]
                trial[ns] = c
                x, e, const, ok = _fit_nonneg(G, g, yy, trial, ns + 1)
                if ok and e < best_sse:
                    best_sse = e
                    best_c = c
                    best_const = const
                    best_x = x
            if best_c < 0:
                break
            np_new = ns + 1 + (1 if best_const else 0)
            dof = max(W - np_new, 1)
            F = (sse - best_sse) / (best_sse / dof)
            if F < f_min:
                break
            sel[ns] = best_c
            ns += 1
            for q in range(ns):
                coef[q] = best_x[q]
            sse = best_sse
            npar = np_new
        for q in range(ns):
            if sel[q] == 1:
                ab[k] = coef[q]
                nmod[k] = ns
                mzw[k] = swm / sw
    return ab, nmod, mzw


# ------------------------------------------------------------------ steps ----
def calibrate(feats: pd.DataFrame, t_apex, cloud: Cloud, f_tol, P: Params, D: DeconvParams):
    """(per-feature apex tolerance in minutes, per-feature shape cut, info) from the file's isotope pairs"""
    n = len(feats)
    lo_tau, hi_tau = 0.1 * P.dt, 0.5 * P.fwhm_med
    default_tau = float(np.clip(max(0.5 * P.dt, 0.1 * P.fwhm_med), lo_tau, hi_tau))
    info = dict(n_iso_pairs=0, calibrated=False)
    par = feats["iso_parent"].values.astype(np.int64)
    child = np.flatnonzero((par >= 0) & (par < n))
    tau = np.full(n, default_tau)
    r_cut = np.full(n, 0.8)
    if len(child) >= MIN_PAIRS:
        parent = par[child]
        d = np.abs(t_apex[child] - t_apex[parent])
        r = _pair_r(cloud.off, cloud.mz, cloud.inten, feats["mz"].values.astype(np.float64), f_tol,
                    feats["scan_min"].values.astype(np.int64), feats["scan_max"].values.astype(np.int64), child, parent)
        snr = feats["snr"].values.astype(np.float64)
        weak = np.minimum(snr[child], snr[parent])
        nb = int(min(4, len(child) // MIN_PAIRS))
        edges = np.quantile(weak, np.linspace(0, 1, nb + 1))[1:-1]
        cls = np.searchsorted(edges, weak, side="right")
        taus = np.array([D.tau_k * 1.4826 * np.median(d[cls == b]) for b in range(nb)])
        taus = np.maximum.accumulate(taus[::-1])[::-1]          # a weaker class is never tighter than a stronger one
        taus = np.clip(taus, lo_tau, hi_tau)
        cuts = np.array([np.quantile(r[cls == b], 1.0 - ISO_KEEP) for b in range(nb)])
        cuts = np.clip(np.minimum.accumulate(cuts[::-1])[::-1], SHAPE_FLOOR, SHAPE_CEIL)   # ... nor held to a higher cut
        f_cls = np.searchsorted(edges, snr, side="right")
        tau, r_cut = taus[f_cls], cuts[f_cls]
        info.update(n_iso_pairs=int(len(child)), calibrated=True, tau_classes_s=[round(60 * float(x), 4) for x in taus],
                    shape_classes=[round(float(x), 4) for x in cuts], snr_edges=[round(float(e), 2) for e in edges],
                    iso_r_median=round(float(np.median(r)), 4))
    if np.isfinite(D.apex_tol):
        tau = np.full(n, float(D.apex_tol))
    if np.isfinite(D.shape_min):
        r_cut = np.full(n, float(D.shape_min))
    info.update(apex_tol_s=round(60 * float(np.median(tau)), 4), shape_min=round(float(np.median(r_cut)), 4))
    return tau, r_cut, info


def _member_lists(comp, height, C):
    """CSR lists of the members of each component, strongest first"""
    ok = np.flatnonzero(comp >= 0)
    o = ok[np.lexsort((-height[ok], comp[ok]))]
    counts = np.bincount(comp[ok], minlength=C)
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int64), o.astype(np.int64)


def _on_window(c, lo, hi, c_lo, m_off, m_val):
    """model of component c on scans lo..hi (0 outside its own bounds)"""
    out = np.zeros(hi - lo + 1)
    m = m_val[m_off[c]:m_off[c + 1]]
    a = max(lo, c_lo[c])
    b = min(hi, c_lo[c] + len(m) - 1)
    if b >= a:
        out[a - lo:b - lo + 1] = m[a - c_lo[c]:b - c_lo[c] + 1]
    return out


def _composites(c_t, c_lo, c_hi, n_mem, m_off, m_val, c_rcut, share, max_ions):
    """components of at most max_ions picked ions whose model is a non-negative mix of the models of the nearest
    larger component on each side, both overlapping it and both explaining at least `share` of it, as well as
    ions of one compound of its strongest ion's S/N class correlate"""
    from scipy.optimize import nnls
    C = len(c_t)
    alive = np.ones(C, bool)
    for c in np.flatnonzero(n_mem <= max_ions):
        lo, hi = c_lo[c], c_hi[c]
        left = next((a for a in range(c - 1, -1, -1) if c_hi[a] < lo or n_mem[a] > n_mem[c]), -1)
        right = next((b for b in range(c + 1, C) if c_lo[b] > hi or n_mem[b] > n_mem[c]), -1)
        if left < 0 or right < 0 or c_hi[left] < lo or c_lo[right] > hi:
            continue
        y = m_val[m_off[c]:m_off[c + 1]]
        A = np.column_stack([_on_window(left, lo, hi, c_lo, m_off, m_val), _on_window(right, lo, hi, c_lo, m_off, m_val),
                             np.ones(hi - lo + 1)])
        x, _ = nnls(A, y)
        parts = A[:, :2] * x[:2]
        tot = parts.sum()
        if tot <= 0 or min(parts[:, 0].sum(), parts[:, 1].sum()) < share * tot:
            continue
        if _pearson(A @ x, y) >= c_rcut[c]:
            alive[c] = False
    return alive


def deconvolve(cloud: Cloud, feats: pd.DataFrame, P: Params, D: DeconvParams | None = None) -> Deconv:
    """Components and spectra of one file. feats: the file's feature table (rows = feature_id)."""
    D = D or DeconvParams()
    n = len(feats)
    S = cloud.n_scans
    off, mz, inten, rt = cloud.off, cloud.mz, cloud.inten, cloud.rt
    f_mz = feats["mz"].values.astype(np.float64)
    f_h = feats["height"].values.astype(np.float64)
    f_lo = np.clip(feats["scan_min"].values.astype(np.int64), 0, max(S - 1, 0))
    f_hi = np.clip(feats["scan_max"].values.astype(np.int64), 0, max(S - 1, 0))
    f_apex = np.clip(feats["scan_apex"].values.astype(np.int64), 0, max(S - 1, 0))
    f_tol = ion_tol(f_mz, f_h, P)
    cols = ["rt", "rt_min", "rt_max", "scan_apex", "scan_min", "scan_max", "n_features", "n_loose", "n_ions", "n_shared_ions",
            "base_mz", "base_height", "quant_mz", "quant_height", "tic", "model_r", "apex_tol_s"]
    empty = Deconv(pd.DataFrame({c: [] for c in cols}), np.zeros(1, np.int64), np.zeros(0), np.zeros(0), np.zeros(0, bool),
                   np.full(n, -1, np.int64), np.zeros(n), np.zeros(n, bool), dict(n_components=0), np.zeros(n, bool))
    if n == 0 or S < 3:
        return empty
    half = int(np.ceil(2.0 * P.fwhm_med / P.dt)) + 2
    f_t = _apex_times(off, mz, inten, rt, f_mz, f_tol, f_apex, half, float(P.dt))
    f_tau, f_rcut, info = calibrate(feats, f_t, cloud, f_tol, P, D)

    # seeds and first membership
    order_t = np.argsort(f_t, kind="stable").astype(np.int64)
    comp0, tight0, _, seeds = _cluster(off, mz, inten, f_mz, f_tol, f_h, f_t, f_lo, f_hi, f_tau, f_rcut,
                                       np.argsort(-f_h, kind="stable").astype(np.int64), order_t, f_t[order_t],
                                       float(f_tau.max()), int(D.ref_ions), bool(D.attach))
    # components by apex time; bounds and apex scan are the seed's
    o = np.argsort(f_t[seeds], kind="stable")
    rank = np.empty(len(seeds), np.int64)
    rank[o] = np.arange(len(seeds))
    seeds = seeds[o]
    comp0 = rank[comp0]
    C = len(seeds)
    c_lo, c_hi, c_apex = f_lo[seeds], f_hi[seeds], f_apex[seeds]
    c_t, c_tau = f_t[seeds].copy(), f_tau[seeds]
    mem_off, mem_idx = _member_lists(np.where(tight0, comp0, -1), f_h, C)     # loose members shape no model
    m_off, m_val = _models(off, mz, inten, f_mz, f_tol, c_lo, c_hi, c_apex, mem_off, mem_idx, D.model_ions)

    # shared-ion components dissolve; their ions stay out of every model
    n_mem = np.diff(mem_off)
    alive = _composites(c_t, c_lo, c_hi, n_mem, m_off, m_val, f_rcut[seeds], D.composite_share, D.composite_max_ions) if D.composite and C > 2 \
        else np.ones(C, bool)
    f_shared = ~alive[comp0]
    # best model per ion, then the models once more from the final membership (a seed stays with its component)
    comp, f_r = _reassign(off, mz, inten, f_mz, f_tol, f_t, f_tau, f_rcut, f_shared, c_t, c_tau, c_lo, c_hi, alive,
                          m_off, m_val)
    comp[seeds[alive]] = np.flatnonzero(alive)
    f_r[seeds[alive]] = np.maximum(f_r[seeds[alive]], f_rcut[seeds[alive]])
    mem_off, mem_idx = _member_lists(comp, f_h, C)
    m_off, m_val = _models(off, mz, inten, f_mz, f_tol, c_lo, c_hi, c_apex, mem_off, mem_idx, D.model_ions)
    n_mem = np.diff(mem_off)
    f_loose = _attach(f_t, f_tau, f_shared, comp, c_t, c_tau, alive) if D.attach else np.zeros(n, bool)
    n_loose = np.bincount(comp[f_loose], minlength=C) if f_loose.any() else np.zeros(C, np.int64)

    # spectra
    live = np.flatnonzero(alive)
    new_id = np.full(C, -1, np.int64)
    new_id[live] = 0
    spec_mz, spec_ab, spec_sh, counts, rows = [], [], [], [], []
    for pos, c in enumerate(live):
        if n_mem[c] < D.min_features:
            new_id[c] = -3
            continue
        lo, hi = int(c_lo[c]), int(c_hi[c])
        # overlapping live neighbours, nearest in apex time first
        nb = [int(d) for d in live[max(0, pos - D.max_neighbours):pos + D.max_neighbours + 1]
              if d != c and c_hi[d] >= lo and c_lo[d] <= hi]
        nb = sorted(nb, key=lambda d: abs(c_t[d] - c_t[c]))[:D.max_neighbours]
        X = np.column_stack([np.ones(hi - lo + 1), _on_window(c, lo, hi, c_lo, m_off, m_val)] +
                            [_on_window(d, lo, hi, c_lo, m_off, m_val) for d in nb])
        X = X[:, np.concatenate([[True, True], X[:, 2:].max(axis=0) > 0])] if X.shape[1] > 2 else X
        a, b = int(off[c_apex[c]]), int(off[c_apex[c] + 1])
        cand = np.arange(a, b, dtype=np.int64)
        ab, nmod, mzw = _fit_ions(off, mz, inten, cand, ion_tol(mz[a:b], inten[a:b], P), lo, hi + 1, X, D.f_min,
                                  D.min_points, 3)
        keep = ab > 0
        if keep.any():
            keep &= ab >= D.min_rel * ab.max()
        if keep.sum() < D.min_ions:
            new_id[c] = -2
            continue
        m_, a_, s_ = mzw[keep], ab[keep], nmod[keep] > 1
        k = np.argsort(m_, kind="stable")
        m_, a_, s_ = m_[k], a_[k], s_[k]
        spec_mz.append(m_); spec_ab.append(a_); spec_sh.append(s_); counts.append(len(m_))
        base = int(np.argmax(a_)) if len(a_) else -1
        uniq = np.flatnonzero(~s_)
        quant = int(uniq[np.argmax(a_[uniq])]) if len(uniq) else base
        members = mem_idx[mem_off[c]:mem_off[c + 1]]
        rows.append(dict(rt=c_t[c], rt_min=rt[lo], rt_max=rt[hi], scan_apex=int(c_apex[c]), scan_min=lo, scan_max=hi,
                         n_features=int(n_mem[c]), n_loose=int(n_loose[c]), n_ions=len(m_), n_shared_ions=int(s_.sum()),
                         base_mz=m_[base] if base >= 0 else np.nan, base_height=a_[base] if base >= 0 else 0.0,
                         quant_mz=m_[quant] if quant >= 0 else np.nan, quant_height=a_[quant] if quant >= 0 else 0.0,
                         tic=float(a_.sum()), model_r=float(np.median(f_r[members])) if len(members) else np.nan,
                         apex_tol_s=60.0 * float(c_tau[c])))
    comps = pd.DataFrame(rows, columns=cols)
    comps.insert(0, "component_id", np.arange(len(comps), dtype=np.int64))
    n_small, n_few = int((new_id == -2).sum()), int((new_id == -3).sum())
    kept = np.flatnonzero(new_id >= 0)
    new_id[kept] = np.arange(len(kept))
    new_id[new_id < -1] = -1
    f_comp = np.where(comp >= 0, new_id[np.maximum(comp, 0)], -1)
    f_loose = f_loose & (f_comp >= 0)
    info.update(n_components=int(len(comps)), n_composite=int((~alive).sum()), n_below_min_ions=n_small,
                n_below_min_features=n_few, n_features=int(n), n_assigned=int((f_comp >= 0).sum()),
                n_loose=int(f_loose.sum()), params=asdict(D))
    return Deconv(comps, np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
                  np.concatenate(spec_mz) if spec_mz else np.zeros(0), np.concatenate(spec_ab) if spec_ab else np.zeros(0),
                  np.concatenate(spec_sh) if spec_sh else np.zeros(0, bool),
                  f_comp, f_r, f_shared, info, f_loose)
