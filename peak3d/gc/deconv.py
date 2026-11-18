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
   Delayed copies (_delayed): fragments formed after the column follow their compound through a lag; their
   component, a little after the compound's and smaller, joins it when the lag, and not a later elution, explains
   its profile.
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
    tau_own: bool = True               # a component of one picked ion whose apex time is less certain than its class's
                                       # (standard error of its apex fit) joins the component whose model it follows
                                       # within a tolerance wider in that proportion
    model_ions: int = 5                # strongest members whose chromatograms make the model profile
    model_net: bool = True             # the reported profiles and the spectra use models built from the members'
                                       # chromatograms above the level each holds beyond its bounds
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
    delay: bool = True                 # a component that is a delayed copy of the one before it joins that one
    delay_f: float = 5.0               # ... when a shift improves the fit of the lagged profile by less than this F
                                       # (about the 5 % point of F with 1 and 15-30 degrees of freedom)
    delay_reach: float = 1.0           # ... and its apex is at most this many peak widths later
    unimodal: bool = True              # a model profile has one maximum: where it rises again away from its apex
                                       # (a neighbour's peak in the member ions), it is continued as a decay
    model_rise: float = 0.05           # ... a rise of this share of the model's maximum above the minimum so far
    shoulders: bool = False            # a second maximum inside a picked ion's chromatogram, clear of that ion's
                                       # noise and shared by shoulder_ions ions, seeds a component like a picked peak
    shoulder_ions: int = 3
    shoulder_sep: float = 0.85         # ... at least this many median peak widths from its feature's apex: two peaks
                                       # of one width closer than 2 sd (0.85 FWHM) show no valley between them
    shoulder_cut: bool = True          # a feature's chromatogram beyond the valley towards such a maximum is left
                                       # out of its component's model profile
    explain: bool = True               # a component whose ions the larger overlapping components' models describe
                                       # as well as its own model does is dissolved (its ions are shared)
    explain_f: float = 5.0             # ... own model needed when it lowers the residual by this F or more
    explain_smaller: bool = True       # ... smaller components that passed their own test may take part in the
                                       # explanation, as long as the larger ones carry more than half of it
    rejoin_share: float = 0.8          # an ion of a dissolved component belongs to the live component whose model
                                       # carries at least this share of its fitted chromatogram (else: shared)


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
    feature_delayed: np.ndarray = None  # member of a delayed copy that joined this component
    model_off: np.ndarray = None        # model profile of component c (maximum 1) on its scans scan_min..scan_max:
    model: np.ndarray = None            # model[model_off[c]:model_off[c+1]]; abundance x model = an ion's chromatogram

    def profile(self, c: int):
        return self.model[self.model_off[c]:self.model_off[c + 1]]

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
    """apex position (in x units) of the peak whose highest point is y[k], and its standard error: weighted
    least-squares parabola through ln y over the contiguous points above half the top plus one point on each side;
    nan when it cannot be placed. The standard error is the fit's own (residuals about the parabola, propagated to
    the vertex; nan on three points): a wide, flat or tailing top places its apex less well than a sharp one."""
    n = len(y)
    top = y[k]
    if top <= 0:
        return np.nan, np.nan
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
        return np.nan, np.nan
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
        return np.nan, np.nan
    c1 = (s0 * (t1 * s4 - s3 * t2) - t0 * (s1 * s4 - s3 * s2) + s2 * (s1 * t2 - t1 * s2)) / det
    c2 = (s0 * (s2 * t2 - t1 * s3) - s1 * (s1 * t2 - t1 * s2) + t0 * (s1 * s3 - s2 * s2)) / det
    if c2 >= 0.0:
        return np.nan, np.nan
    u = -c1 / (2.0 * c2)
    if u < x[lo] - x[k] or u > x[hi] - x[k]:
        return np.nan, np.nan
    se = np.nan
    m = hi - lo + 1
    if m > 3:
        # inverse of the normal matrix [[s0 s1 s2] [s1 s2 s3] [s2 s3 s4]] (symmetric): the rows the vertex needs
        i00 = (s2 * s4 - s3 * s3) / det; i01 = (s2 * s3 - s1 * s4) / det; i02 = (s1 * s3 - s2 * s2) / det
        i11 = (s0 * s4 - s2 * s2) / det; i12 = (s1 * s2 - s0 * s3) / det; i22 = (s0 * s2 - s1 * s1) / det
        c0 = i00 * t0 + i01 * t1 + i02 * t2
        rss = 0.0
        for q in range(lo, hi + 1):
            v = x[q] - x[k]
            r = np.log(y[q]) - (c0 + c1 * v + c2 * v * v)
            rss += y[q] * r * r
        g1 = -1.0 / (2.0 * c2); g2 = c1 / (2.0 * c2 * c2)
        var = rss / (m - 3) * (g1 * g1 * i11 + 2.0 * g1 * g2 * i12 + g2 * g2 * i22)
        if var > 0.0:
            se = np.sqrt(var)
    return x[k] + u, se


@njit(cache=True)
def _apex_times(off, mz, inten, rt, f_mz, f_tol, f_apex, half, dt, clip_lo, clip_hi):
    """sub-scan apex time (minutes) of every feature, from its chromatogram on scans clip_lo..clip_hi around the
    apex (the scan time where the parabola cannot be placed), and the standard error of that time (minutes; nan
    where the fit gives none)"""
    n = len(f_mz)
    S = len(rt)
    out = np.empty(n)
    se = np.full(n, np.nan)
    for i in range(n):
        a = f_apex[i]
        s0 = max(max(0, a - half), clip_lo[i])
        s1 = min(min(S, a + half + 1), clip_hi[i] + 1)
        y = eic_kernel(off, mz, inten, f_mz[i], f_tol[i], s0, s1)
        k = a - s0
        # the picker's apex is on the summed surface; take the trace's own maximum next to it
        if k > 0 and y[k - 1] > y[k]:
            k -= 1
        elif k < len(y) - 1 and y[k + 1] > y[k]:
            k += 1
        t, e = _log_parabola_apex(y, k, (rt[s0:s1] - rt[a]) / dt)
        out[i] = rt[a] + t * dt if np.isfinite(t) else rt[a]
        if np.isfinite(t) and np.isfinite(e):
            se[i] = e * dt
    return out, se


@njit(cache=True)
def _sub_apexes(off, mz, inten, f_mz, f_tol, f_lo, f_hi, f_apex, f_thr):
    """Second maxima inside the features' chromatograms. Walking away from a feature's apex on each side: once the
    trace has risen f_thr above the lowest point since the apex (the valley) and then fallen f_thr below the highest
    point since (the maximum), that maximum is a peak of its own on this ion's noise; the walk goes on from there.
    A rise that reaches the feature's bound without falling again is the start of something outside the feature.
    Returns per maximum: the feature, the scan, its scans (from the valley towards the apex out to the feature's
    bound, or to the valley of the next maximum further out), its height, and the side of the hidden flank (+1: the
    valley, and the feature's apex, come after it)."""
    n = len(f_mz)
    cap = 2 * n + 16
    par = np.empty(cap, np.int64)
    apex = np.empty(cap, np.int64)
    lo_o = np.empty(cap, np.int64)
    hi_o = np.empty(cap, np.int64)
    h_o = np.empty(cap)
    side_o = np.empty(cap, np.int64)
    m = 0
    for j in range(n):
        lo = f_lo[j]
        W = f_hi[j] - lo + 1
        thr = f_thr[j]
        if W < 5 or not thr > 0.0:
            continue
        y = eic_kernel(off, mz, inten, f_mz[j], f_tol[j], lo, f_hi[j] + 1)
        a = min(max(f_apex[j] - lo, 0), W - 1)
        if a > 0 and y[a - 1] > y[a]:
            a -= 1
        elif a < W - 1 and y[a + 1] > y[a]:
            a += 1
        for side in (-1, 1):
            low = y[a]
            k_low = a
            k_top = -1
            top = 0.0
            prev = -1                                # the maximum before this one on this side
            k = a + side
            while 0 <= k < W and m < cap:
                v = y[k]
                if k_top < 0:
                    if v < low:
                        low = v
                        k_low = k
                    elif v - low >= thr:
                        top = v
                        k_top = k
                elif v > top:
                    top = v
                    k_top = k
                elif top - v >= thr:
                    par[m] = j
                    apex[m] = lo + k_top
                    if side < 0:
                        lo_o[m] = lo
                        hi_o[m] = lo + k_low
                        if prev >= 0:
                            lo_o[prev] = lo + k_low
                    else:
                        lo_o[m] = lo + k_low
                        hi_o[m] = f_hi[j]
                        if prev >= 0:
                            hi_o[prev] = lo + k_low
                    h_o[m] = top
                    side_o[m] = -side
                    prev = m
                    m += 1
                    low = v
                    k_low = k
                    k_top = -1
                k += side
    return par[:m], apex[:m], lo_o[:m], hi_o[:m], h_o[:m], side_o[:m]


@njit(cache=True)
def _coincident(f_t, f_tau, f_ion, n_real, order_t, need):
    """rows n_real.. (second maxima) that at least `need` - 1 other ions share (features or second maxima; f_ion: an
    ion and its isotope peaks are one): their apex within the larger of the two tolerances"""
    n = len(f_t)
    ok = np.zeros(n - n_real, np.bool_)
    tau_max = f_tau.max() if n else 0.0
    pos = np.empty(n, np.int64)
    for q in range(n):
        pos[order_t[q]] = q
    seen = np.empty(max(need, 1), np.int64)
    for i in range(n_real, n):
        cnt = 0
        for step in (-1, 1):
            q = pos[i] + step
            while 0 <= q < n and cnt < need - 1:
                j = order_t[q]
                d = abs(f_t[j] - f_t[i])
                if d > tau_max:
                    break
                if d <= max(f_tau[i], f_tau[j]) and f_ion[j] != f_ion[i]:
                    new = True
                    for u in range(cnt):
                        if seen[u] == f_ion[j]:
                            new = False
                    if new:
                        seen[cnt] = f_ion[j]
                        cnt += 1
                q += step
        ok[i - n_real] = cnt >= need - 1
    return ok


def _complete(c_lo, c_hi, c_apex, c_side, m_off, m_val, S):
    """The hidden flank of the components seeded at a second maximum. Their model is known from the bound to the
    valley; past the apex it also holds the neighbour's rise. A peak does not fall more slowly towards a
    neighbour than away from it: on the hidden side the model is the lower of what was observed and the visible
    flank mirrored at the apex, and it goes on as far as the visible flank does. Returns (m_off, m_val); c_lo and
    c_hi are widened in place."""
    parts = []
    for c in range(len(c_lo)):
        m = m_val[m_off[c]:m_off[c + 1]]
        if c_side[c] == 0 or len(m) < 2:
            parts.append(m)
            continue
        if c_side[c] < 0:                    # hidden flank before the apex: work on the reversed profile
            m = m[::-1]
            a = c_hi[c] - c_apex[c]
        else:
            a = c_apex[c] - c_lo[c]
        a = int(min(max(a, 0), len(m) - 1))
        out = np.zeros(2 * a + 1)
        out[:a + 1] = m[:a + 1]
        for d in range(1, a + 1):
            out[a + d] = min(m[a + d], m[a - d]) if a + d < len(m) else m[a - d]
        if c_side[c] < 0:
            new_lo = max(0, c_hi[c] - 2 * a)
            out = out[:c_hi[c] - new_lo + 1][::-1]
            c_lo[c] = new_lo
        else:
            new_hi = min(S - 1, c_lo[c] + 2 * a)
            out = out[:new_hi - c_lo[c] + 1]
            c_hi[c] = new_hi
        parts.append(out)
    return np.concatenate([[0], np.cumsum([len(q) for q in parts])]).astype(np.int64), np.concatenate(parts)


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
def _floors(off, mz, inten, f_mz, f_tol, f_h, f_lo, f_hi, S, reach, rise):
    """The level every picked ion's chromatogram stands on. Per side: the median of the three scans around its
    bound, or the median of the `reach` scans beyond it when that is lower; the floor is the lower side. A clean
    peak's bounds lie where it has fallen to the picker's threshold and the trace beyond falls further: no floor to
    speak of. A background ion (column bleed, reagent, air) keeps its level beyond the peak; beside a weak peak
    that level is a fifth of the height and more, and a model profile built from such chromatograms carries it as
    a pedestal that the compound's other ions do not have. A side where the trace beyond the bound climbs again, by
    more than `rise` of the peak's height above the bound, is a valley towards another peak and says nothing about
    the floor; between two valleys the floor is unknown and taken as none."""
    n = len(f_mz)
    out = np.zeros(n)
    for j in range(n):
        best = np.inf
        for side in range(2):
            b = f_lo[j] if side == 0 else f_hi[j]
            v = np.median(eic_kernel(off, mz, inten, f_mz[j], f_tol[j], max(b - 1, 0), min(b + 2, S)))
            if side == 0:
                o0 = max(b - reach, 0); o1 = b
            else:
                o0 = min(b + 1, S); o1 = min(b + 1 + reach, S)
            if o1 > o0:
                beyond = np.median(eic_kernel(off, mz, inten, f_mz[j], f_tol[j], o0, o1))
                if beyond - v > rise * max(f_h[j] - v, 0.0):
                    continue
                v = min(v, beyond)
            best = min(best, v)
        if best < np.inf:
            out[j] = best
    return out


@njit(cache=True)
def _models(off, mz, inten, f_mz, f_tol, c_lo, c_hi, c_apex, mem_off, mem_idx, k_top, f_cut_lo, f_cut_hi, f_base):
    """model profile of every component over its bounds, maximum 1: per-scan median (mean below three) of the
    chromatograms of its k_top strongest members, each divided by its sum over the three apex scans. A member's
    chromatogram counts between its valleys only (f_cut_lo, f_cut_hi: the scans of the valleys towards the second
    maxima found in it, or its bounds; _sub_apexes): beyond a valley it is another compound's peak, and the
    member continues there as a geometric decay at the steepest step it showed between apex and valley (at most
    0.9 per scan), where the trace is not lower still. Every chromatogram counts above its floor (f_base, _floors)."""
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
            if f_base[j] > 0.0:
                for w in range(W):
                    y[w] = max(y[w] - f_base[j], 0.0)
            nrm = float(y[a])
            if a > 0:
                nrm += y[a - 1]
            if a + 1 < W:
                nrm += y[a + 1]
            if nrm <= 0:
                continue
            for side in (-1, 1):
                cut = (f_cut_lo[j] if side < 0 else f_cut_hi[j]) - s0
                if cut < 0 or cut >= W or (cut - a) * side <= 0:
                    continue
                ratio = 0.9
                w = a + side
                while (cut - w) * side >= 0:
                    if y[w - side] > 0.0 and y[w] < ratio * y[w - side]:
                        ratio = y[w] / y[w - side]
                    w += side
                v = y[cut]
                w = cut + side
                while 0 <= w < W:
                    v *= ratio
                    if y[w] > v:
                        y[w] = v
                    w += side
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
def _unimodal(m_off, m_val, rise):
    """Every model profile made single-peaked. Walking away from its maximum on each side, the profile may not
    climb: small rises are levelled to the minimum reached so far; a rise of `rise` (share of the maximum) above
    that minimum is another compound's peak in the member ions, and from the minimum before it the profile is
    continued as a geometric decay, at the steepest step ratio seen between the maximum and that minimum (at most
    0.9 per scan). The component's own tail under the neighbour is unknown; the decay stands for it."""
    C = len(m_off) - 1
    for c in range(C):
        a = m_off[c]; b = m_off[c + 1]
        W = b - a
        if W < 3:
            continue
        k0 = 0
        for w in range(W):
            if m_val[a + w] > m_val[a + k0]:
                k0 = w
        top = m_val[a + k0]
        if top <= 0.0:
            continue
        for side in (-1, 1):
            low = top                   # minimum so far, and where
            k_low = k0
            q = 0.9
            cut = -1
            k = k0 + side
            while 0 <= k < W:
                v = m_val[a + k]
                if v <= low:
                    prev = m_val[a + k - side]
                    if prev > 0.0 and v / prev < q and v < prev:
                        q = v / prev
                    low = v
                    k_low = k
                elif v - low >= rise * top:
                    cut = k_low
                    break
                else:
                    m_val[a + k] = low
                k += side
            if cut >= 0:
                v = m_val[a + cut]
                k = cut + side
                while 0 <= k < W:
                    v *= q
                    m_val[a + k] = v
                    k += side


@njit(cache=True)
def _nnls_fit(G, g, yy, cols, n):
    """Non-negative least squares on the columns cols[:n] (Gram matrix G, moments g): columns with a non-positive
    coefficient are taken out one at a time, the most negative first; when the columns are linearly dependent
    (a model that is flat or zero on the window, beside the constant) the last one listed goes, so list the
    columns that matter first. Returns the coefficients by column of G (0: not in the fit) and the residual sum
    of squares."""
    coef = np.zeros(G.shape[0])
    idx = cols[:n].copy()
    m = n
    while m > 0:
        x, ex, ok = _solve(G, g, idx, m)
        if not ok:                       # two columns the same on this window: the last one adds nothing
            m -= 1
            continue
        worst = -1
        for i in range(m):
            if x[i] <= 0.0 and (worst < 0 or x[i] < x[worst]):
                worst = i
        if worst < 0:
            for i in range(m):
                coef[idx[i]] = x[i]
            return coef, max(yy - ex, 0.0)
        for i in range(worst, m - 1):
            idx[i] = idx[i + 1]
        m -= 1
    return coef, yy


@njit(cache=True)
def _nnls_sse(G, g, yy, cols, n):
    return _nnls_fit(G, g, yy, cols, n)[1]


@njit(cache=True)
def _nearest_live(t, lo, hi, skip, c_t, c_lo, c_hi, alive, rank, min_rank, nb):
    """into nb: the live components that overlap scans lo..hi (not `skip`, rank above min_rank), the len(nb) nearest
    to time t; returns how many"""
    max_nb = len(nb)
    n_nb = 0
    for a in range(len(c_t)):
        if a == skip or not alive[a] or rank[a] <= min_rank or c_hi[a] < lo or c_lo[a] > hi:
            continue
        if n_nb < max_nb:
            nb[n_nb] = a
            n_nb += 1
        else:
            far = 0
            for q in range(1, max_nb):
                if abs(c_t[nb[q]] - t) > abs(c_t[nb[far]] - t):
                    far = q
            if abs(c_t[a] - t) < abs(c_t[nb[far]] - t):
                nb[far] = a
    return n_nb


@njit(cache=True)
def _rejoin(off, mz, inten, f_mz, f_tol, f_lo, f_hi, f_t, f_rcut, todo, c_t, c_lo, c_hi, alive, m_off, m_val, max_nb,
            min_share):
    """The ions of dissolved components (todo: feature rows) placed again: each is fitted over its own bounds,
    non-negatively, with a constant and the models of the live components there (the max_nb nearest in time). When
    one model carries at least min_share of what the models explain and the fit follows the chromatogram (its
    shape cut), the ion is that compound's: its apex was moved, by a neighbour's contribution or by noise, further
    than the apex tolerance. Otherwise it stays shared. Returns the component (-1: shared) and the correlation."""
    out = np.full(len(todo), -1, np.int64)
    r_out = np.zeros(len(todo))
    nb = np.empty(max_nb, np.int64)
    cols = np.empty(max_nb + 1, np.int64)
    rank = np.zeros(len(c_t), np.int64)
    for i in range(len(todo)):
        j = todo[i]
        lo = f_lo[j]; hi = f_hi[j]
        W = hi - lo + 1
        n_nb = _nearest_live(f_t[j], lo, hi, -1, c_t, c_lo, c_hi, alive, rank, -1, nb)
        if n_nb == 0 or W < n_nb + 3:
            continue
        X = np.zeros((W, n_nb + 1))
        for w in range(W):
            X[w, 0] = 1.0
        for q in range(n_nb):
            a = nb[q]
            for w in range(W):
                s = lo + w
                if c_lo[a] <= s <= c_hi[a]:
                    X[w, q + 1] = m_val[m_off[a] + s - c_lo[a]]
        G = X.T @ X
        y = eic_kernel(off, mz, inten, f_mz[j], f_tol[j], lo, hi + 1)
        g = np.zeros(n_nb + 1)
        yy = 0.0
        sy = 0.0
        for w in range(W):
            yy += y[w] * y[w]
            sy += y[w]
            for c in range(n_nb + 1):
                g[c] += X[w, c] * y[w]
        for c in range(n_nb):
            cols[c] = c + 1
        cols[n_nb] = 0
        coef, sse = _nnls_fit(G, g, yy, cols, n_nb + 1)
        tot = yy - sy * sy / W
        if tot <= 0.0 or sse >= tot:
            continue
        best = -1
        s_best = 0.0
        s_all = 0.0
        for q in range(n_nb):
            sh = coef[q + 1] * G[0, q + 1]            # coefficient x the model's sum over the window
            s_all += sh
            if sh > s_best:
                s_best = sh
                best = q
        r = np.sqrt(1.0 - sse / tot)
        if best >= 0 and s_best >= min_share * s_all and r >= f_rcut[j]:
            out[i] = nb[best]
            r_out[i] = r
    return out, r_out


@njit(cache=True)
def _explained(off, mz, inten, f_mz, f_tol, mem_off, mem_idx, k_top, c_t, c_lo, c_hi, c_rcut, alive, rank, m_off, m_val,
               max_nb, f_max, smaller):
    """Components the models of the others explain. In ascending order of size (rank), component b is tested
    against the live components that overlap its bounds (the max_nb nearest in apex time): the ones that outrank
    it and, with `smaller`, the smaller ones too, which had their own test before b and stayed. Each of b's k_top
    strongest tight ions is fitted over b's bounds, non-negatively, with a constant and those models, and again
    with b's own model added. b is dissolved when
      (0) the larger components carry more than half of what the models explain of its ions (always so without
          `smaller`). The ions two compounds share make a component that can have more ions than the smaller
          compound: it is the larger compound's peak with the smaller one's as a shoulder. Without the limit a
          compound would be dissolved into a piece of itself that the test had left standing;
      (1) the others describe its ions: sum(sse_without) <= sum(noise), noise = (1 - shape cut^2) of an ion's
          variance, the part of one ion's chromatogram that is its own (the cut calibrated on isotope pairs: the
          others fit b's ions as well as an isotope partner would). Without this, a few weak ions on a handful
          of scans, where nothing is significant, would be dissolved by neighbours that explain nothing of them;
      (2) its own model is not needed on top:
          F = sum(sse_without - sse_with) / (sum(max(sse_with, noise)) / (points - columns)) < f_max.
    This is what happens to (i) the ions two overlapping compounds share, whose apex lies
    between theirs, in any number; (ii) the ions of one compound that a neighbour's contribution, or noise at low
    signal, moved off its apex time by more than the tolerance; (iii) a background ion's bump under a compound.
    A second compound stays: at another apex time its ions are not described by the larger one's model."""
    C = len(c_t)
    out = alive.copy()
    order = np.argsort(rank)
    nb = np.empty(max_nb, np.int64)
    cols = np.empty(max_nb + 2, np.int64)
    for b in order:
        if not out[b] or mem_off[b + 1] == mem_off[b]:
            continue
        lo = c_lo[b]; hi = c_hi[b]
        W = hi - lo + 1
        n_nb = _nearest_live(c_t[b], lo, hi, b, c_t, c_lo, c_hi, out, rank, -1 if smaller else rank[b], nb)
        if n_nb == 0 or W < n_nb + 4:
            continue
        X = np.zeros((W, n_nb + 2))
        for w in range(W):
            X[w, 0] = 1.0
        for q in range(n_nb):
            a = nb[q]
            for w in range(W):
                s = lo + w
                if c_lo[a] <= s <= c_hi[a]:
                    X[w, q + 1] = m_val[m_off[a] + s - c_lo[a]]
        for w in range(W):
            X[w, n_nb + 1] = m_val[m_off[b] + w]
        G = X.T @ X
        num = 0.0
        den = 0.0
        left = 0.0
        allowed = 0.0
        by_larger = 0.0
        by_smaller = 0.0
        g = np.empty(n_nb + 2)
        for q in range(min(k_top, mem_off[b + 1] - mem_off[b])):
            j = mem_idx[mem_off[b] + q]
            y = eic_kernel(off, mz, inten, f_mz[j], f_tol[j], lo, hi + 1)
            yy = 0.0
            sy = 0.0
            for w in range(W):
                yy += y[w] * y[w]
                sy += y[w]
            for c in range(n_nb + 2):
                acc = 0.0
                for w in range(W):
                    acc += X[w, c] * y[w]
                g[c] = acc
            for c in range(n_nb + 1):
                cols[c] = c
            coef, sse0 = _nnls_fit(G, g, yy, cols, n_nb + 1)
            for c in range(n_nb):
                sh = coef[c + 1] * G[0, c + 1]       # coefficient x the model's sum over the window
                if rank[nb[c]] > rank[b]:
                    by_larger += sh
                else:
                    by_smaller += sh
            cols[0] = n_nb + 1                 # own model first: never the column a dependent set loses
            for c in range(n_nb + 1):
                cols[c + 1] = c
            sse1 = min(_nnls_sse(G, g, yy, cols, n_nb + 2), sse0)
            tot = yy - sy * sy / W
            noise = tot * (1.0 - c_rcut[b] * c_rcut[b])
            num += max(sse0 - sse1, 0.0)
            den += max(sse1, noise)
            left += sse0
            allowed += noise
        if by_larger <= by_smaller:
            continue
        if den > 0.0 and left <= allowed and num / (den / (W - n_nb - 2)) < f_max:
            out[b] = False
    return out


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
def _lagged(pa, tau, y):
    """pa passed through a first-order lag of time constant tau scans, into y"""
    W = len(pa)
    if tau <= 0.0:
        for k in range(W):
            y[k] = pa[k]
        return
    al = np.exp(-1.0 / tau)
    acc = 0.0
    for k in range(W):
        acc = (1.0 - al) * pa[k] + al * acc
        y[k] = acc


@njit(cache=True)
def _rss_shifted(y, pb, syy, dl):
    """residual sum of squares of pb fitted by y moved dl scans later, amplitude free and positive"""
    W = len(y)
    sxx = 0.0
    sxy = 0.0
    for k in range(W):
        u = k - dl
        v = 0.0
        if u >= 0.0:
            j = int(u)
            fr = u - j
            if j + 1 < W:
                v = y[j] + fr * (y[j + 1] - y[j])
            elif fr == 0.0:
                v = y[j]
        sxx += v * v
        sxy += v * pb[k]
    return syy - sxy * sxy / sxx if (sxx > 0.0 and sxy > 0.0) else syy


@njit(cache=True)
def _lag_stats(pa, pb, d_shift, d_tau):
    """Profile pb fitted, amplitude free, by profile pa (i) passed through a first-order lag (time constant
    0..d_tau scans, 30 steps), (ii) moved later (0..d_shift scans, 20 steps), (iii) both. Returns the residual sums
    of squares of the best fit of each (lag, move, both), the total sum of squares of pb about its mean, and the
    number of scans where either profile is above 5 % of its maximum."""
    W = len(pa)
    y = np.empty(W)
    syy = 0.0
    mean = 0.0
    npts = 0
    for k in range(W):
        syy += pb[k] * pb[k]
        mean += pb[k]
        if max(pa[k], pb[k]) >= 0.05:
            npts += 1
    mean /= max(W, 1)
    tot = syy - W * mean * mean
    nt = 31
    nd = 21
    st = d_tau / (nt - 1)
    sd = d_shift / (nd - 1)
    best_lag = syy
    best_move = syy
    best = syy
    for it in range(nt):
        _lagged(pa, it * st, y)
        for i_d in range(nd):
            r = _rss_shifted(y, pb, syy, i_d * sd)
            if i_d == 0 and r < best_lag:
                best_lag = r
            if it == 0 and r < best_move:
                best_move = r
            if r < best:
                best = r
    return best_lag, best_move, best, tot, npts


@njit(cache=True)
def _is_delayed_copy(rss_lag, rss_move, rss_both, tot, npts, r_min, n_prof, f_max):
    """The decision on _lag_stats. F = (rss_0 - rss_1) / (max(rss_1, noise) / (points - 3)); noise = the share of a
    profile's variance that is its own uncertainty: one ion's chromatogram may differ from its compound's by
    1 - r_min^2 of its variance (r_min: the shape cut, calibrated on isotope pairs), and the profile is the median
    of n_prof ions. Without that floor two profiles that agree to their noise give large F from small differences.
    (1) the lag describes the profile: correlation >= r_min
    (2) the lag is needed: it fits better than the best move alone, F >= f_max. When neither is better the profiles
        are the same within their noise, and the apex times, which are measured more finely, said two components.
    (3) no move is needed on top of the lag: F < f_max"""
    if tot <= 0.0:
        return False
    r2 = 1.0 - rss_lag / tot
    if r2 <= 0.0 or np.sqrt(r2) < r_min:
        return False
    dof = max(npts - 3, 1)
    noise = tot * (1.0 - r_min * r_min) / max(n_prof, 1)
    if (rss_move - rss_lag) / (max(rss_lag, noise) / dof) < f_max:
        return False
    return (rss_lag - rss_both) / (max(rss_both, noise) / dof) < f_max


@njit(cache=True)
def _profile(off, mz, inten, f_mz, f_tol, mem_off, mem_idx, c, s0, s1, k_top, f_base):
    """elution profile of component c on scans s0..s1-1: per-scan median (mean below three) of the chromatograms of
    its k_top strongest tight members, each above its floor and divided by its maximum there; maximum 1 (all
    zero: no signal)"""
    W = s1 - s0
    k = min(k_top, mem_off[c + 1] - mem_off[c])
    T = np.zeros((max(k, 1), W))
    kk = 0
    for q in range(k):
        j = mem_idx[mem_off[c] + q]
        y = eic_kernel(off, mz, inten, f_mz[j], f_tol[j], s0, s1)
        if f_base[j] > 0.0:
            for w in range(W):
                y[w] = max(y[w] - f_base[j], 0.0)
        top = 0.0
        for w in range(W):
            if y[w] > top:
                top = y[w]
        if top <= 0.0:
            continue
        for w in range(W):
            T[kk, w] = y[w] / top
        kk += 1
    out = np.zeros(W)
    if kk == 0:
        return out
    top = 0.0
    for w in range(W):
        out[w] = np.median(T[:kk, w]) if kk >= 3 else T[:kk, w].mean()
        top = max(top, out[w])
    if top > 0.0:
        for w in range(W):
            out[w] /= top
    return out


@njit(cache=True)
def _delayed(off, mz, inten, f_mz, f_tol, mem_off, mem_idx, k_top, c_t, c_apex, c_lo, c_hi, c_alive, c_rcut,
             m_off, m_val, dt, reach, f_max, f_base):
    """parent[b] = the component b is a delayed copy of (-1: none). A fragment formed after the column (in the
    ion source, on a surface) follows its compound through a lag: its signal starts when the compound's does and
    peaks and tails later. A second compound that elutes later starts later. So b, whose apex is at most `reach`
    of a's peak widths after a's, is a's delayed copy when a's profile passed through a first-order lag describes
    b's profile as well as b's own ions describe one another (correlation >= b's shape cut), better than a's
    profile moved later does, and moving it later on top improves that by less than F = f_max (_lag_stats,
    _is_delayed_copy). Profiles: _profile on a window from three widths before
    a's apex to six after b's. Components are sorted by apex time. A copy is tested against the component itself,
    never against another copy: a second compound just after a compound's delayed fragments resembles those more
    than it resembles the compound (BDE 153, its two late ions and TBBPA on the MRH set). A copy has fewer tight
    ions than the component it follows: delayed fragments are a few of a compound's ions, and a lone ion picked
    between two compounds (a shared ion, broader than either) would otherwise take both as its copies.
    A lag much shorter than the peak is a shift to first order, so two compounds a small fraction of a scan apart
    pass as well: they are within the apex tolerance of strong ions and cannot be told apart by time either."""
    C = len(c_t)
    S = len(off) - 1
    parent = np.full(C, -1, np.int64)
    width = np.zeros(C)
    for c in range(C):
        for k in range(m_off[c], m_off[c + 1]):
            if m_val[k] >= 0.5:
                width[c] += 1.0
    for b in range(C):
        if not c_alive[b] or mem_off[b + 1] == mem_off[b]:
            continue
        best_a = -1
        best_r = np.inf
        for a in range(b - 1, max(-1, b - 13), -1):          # the dozen components before b, by apex time
            if not c_alive[a] or width[a] <= 0 or c_hi[a] < c_lo[b] or parent[a] >= 0:
                continue
            if mem_off[a + 1] - mem_off[a] <= mem_off[b + 1] - mem_off[b]:
                continue                  # the delayed fragments are the smaller part of their compound
            d_scans = (c_t[b] - c_t[a]) / dt
            if d_scans <= 0.0 or d_scans > reach * width[a]:
                continue
            w = int(width[a] + 0.5)
            s0 = max(0, c_apex[a] - 3 * w)
            s1 = min(S, c_apex[b] + 6 * w + 1)
            pa = _profile(off, mz, inten, f_mz, f_tol, mem_off, mem_idx, a, s0, s1, k_top, f_base)
            pb = _profile(off, mz, inten, f_mz, f_tol, mem_off, mem_idx, b, s0, s1, k_top, f_base)
            lag, move, both, tot, npts = _lag_stats(pa, pb, d_scans + 1.0, min(1.5 * width[a], 3.0 * d_scans + 1.0))
            n_prof = min(k_top, mem_off[b + 1] - mem_off[b])
            if _is_delayed_copy(lag, move, both, tot, npts, c_rcut[b], n_prof, f_max) and lag < best_r:
                best_r = lag
                best_a = a
        parent[b] = best_a
    return parent


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
def _fit_ions(off, mz, inten, cand, cand_tol, s0, s1, X, f_min, min_points, max_models, n_own, member):
    """Every candidate centroid (index cand[k] in the apex scan, window cand_tol[k]) fitted over scans s0..s1-1 with
    the columns of X: 0 the constant, 1..n_own this component's models (its own and its delayed copies'), then its
    neighbours'. Returns per candidate the abundance in this component (the sum over its models; 0: not its ion),
    one plus the number of neighbour models in the fit, and the intensity-weighted m/z.
    member[k]: the candidate is one of the component's picked ions. Its fit starts from the component's own model
    and the others are added to that: every component is fitted on its own window, so between two similar models
    each fit could prefer the other's, and the ion would be in neither spectrum."""
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
        if member[k] and K >= 1:
            trial[0] = 1
            x, e, const, ok = _fit_nonneg(G, g, yy, trial, 1)
            if ok:
                sel[0] = 1
                ns = 1
                coef[0] = x[0]
                sse = e
                npar = 2 if const else 1
        for step in range(ns, min(K, max_models)):
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
        own = 0.0
        n_for = 0
        for q in range(ns):
            if sel[q] <= n_own:
                own += coef[q]
            else:
                n_for += 1
        if own > 0.0:
            ab[k] = own
            nmod[k] = 1 + n_for
            mzw[k] = swm / sw
    return ab, nmod, mzw


# ------------------------------------------------------------------ steps ----
def _own_tolerance(tau, se, cap):
    """Per-feature apex tolerance for joining a component. The class tolerance (tau, calibrated on isotope pairs)
    is that of a pair of ions with the class's typical apex error; an ion whose own error is r times the typical
    one differs from a typical partner by sqrt((r^2 + 1) / 2) times as much, and gets that much more room: never
    less than its class's, at most `cap` (or the class's where that is larger)."""
    out = tau.copy()
    ok = np.isfinite(se) & (se > 0)
    for v in np.unique(tau[ok]):
        m = ok & (tau == v)
        med = float(np.median(se[m]))
        if med > 0:
            r = se[m] / med
            out[m] = np.minimum(max(v, cap), v * np.sqrt(np.maximum((r * r + 1.0) / 2.0, 1.0)))
    return out


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
    cols = ["rt", "rt_min", "rt_max", "scan_apex", "scan_min", "scan_max", "n_features", "n_loose", "n_delayed", "n_ions",
            "n_shared_ions",
            "base_mz", "base_height", "quant_mz", "quant_height", "tic", "model_r", "apex_tol_s"]
    if D.shoulders:
        cols.append("shoulder")         # the component was started by a second maximum, not by a picked peak
    empty = Deconv(pd.DataFrame({c: [] for c in cols}), np.zeros(1, np.int64), np.zeros(0), np.zeros(0), np.zeros(0, bool),
                   np.full(n, -1, np.int64), np.zeros(n), np.zeros(n, bool), dict(n_components=0), np.zeros(n, bool),
                   np.zeros(n, bool), np.zeros(1, np.int64), np.zeros(0))
    if n == 0 or S < 3:
        return empty
    half = int(np.ceil(2.0 * P.fwhm_med / P.dt)) + 2
    # second maxima inside the picked chromatograms are rows of their own from here on (n_real..): same ion, their
    # own apex, bounds from the valley outwards
    n_real = n
    f_side = np.zeros(n, np.int64)
    snr = np.maximum(feats["snr"].values.astype(np.float64), 1e-9)
    iso = feats["iso_parent"].values.astype(np.int64)
    if D.shoulders:
        thr = float(P.k_valley) * np.sqrt(2.0) * f_h / snr
        par, s_apex, s_lo, s_hi, s_h, s_side = _sub_apexes(off, mz, inten, f_mz, f_tol, f_lo, f_hi, f_apex, thr)
        far = np.abs(rt[s_apex] - rt[f_apex[par]]) >= D.shoulder_sep * P.fwhm_med
        par, s_apex, s_lo, s_hi, s_h, s_side = par[far], s_apex[far], s_lo[far], s_hi[far], s_h[far], s_side[far]
        if len(par):
            snr = np.concatenate([snr, s_h * snr[par] / np.maximum(f_h[par], 1e-9)])
            f_mz, f_tol, f_h = np.concatenate([f_mz, f_mz[par]]), np.concatenate([f_tol, f_tol[par]]), np.concatenate([f_h, s_h])
            f_lo, f_hi, f_apex = np.concatenate([f_lo, s_lo]), np.concatenate([f_hi, s_hi]), np.concatenate([f_apex, s_apex])
            f_side, iso = np.concatenate([f_side, s_side]), np.concatenate([iso, np.full(len(par), -1, np.int64)])
            n = len(f_mz)
    clip_lo, clip_hi = np.zeros(n, np.int64), np.full(n, S - 1, np.int64)
    clip_lo[n_real:], clip_hi[n_real:] = f_lo[n_real:], f_hi[n_real:]
    f_t, f_se = _apex_times(off, mz, inten, rt, f_mz, f_tol, f_apex, half, float(P.dt), clip_lo, clip_hi)
    table = feats if n == n_real else pd.DataFrame(dict(mz=f_mz, scan_min=f_lo, scan_max=f_hi, snr=snr, iso_parent=iso))
    f_tau, f_rcut, info = calibrate(table, f_t, cloud, f_tol, P, D)
    if n > n_real:
        f_tau[n_real:] = np.maximum(f_tau[n_real:], 0.5 * P.dt)      # a second maximum is placed to the scan
        ion = np.arange(n_real)                 # an ion = a feature and the isotope peaks annotated under it
        up = np.where((iso[:n_real] >= 0) & (iso[:n_real] < n_real), iso[:n_real], ion)
        for _ in range(8):
            ion = up[ion]
        ok = _coincident(f_t, f_tau, np.concatenate([ion, ion[par]]), n_real,
                         np.argsort(f_t, kind="stable").astype(np.int64), int(D.shoulder_ions))
        keep = np.concatenate([np.arange(n_real), n_real + np.flatnonzero(ok)])
        f_mz, f_tol, f_h, f_lo, f_hi, f_apex = f_mz[keep], f_tol[keep], f_h[keep], f_lo[keep], f_hi[keep], f_apex[keep]
        f_side, f_t, f_tau, f_rcut, f_se = f_side[keep], f_t[keep], f_tau[keep], f_rcut[keep], f_se[keep]
        f_par = par[ok]
        n = len(f_mz)
    info["n_shoulder_apexes"] = int(n - n_real)
    f_zero = np.zeros(n)
    # where an ion's chromatogram stops being its feature's: the valleys towards the kept second maxima
    f_cut_lo, f_cut_hi = np.full(n, -1, np.int64), np.full(n, S, np.int64)
    if n > n_real and D.shoulder_cut:
        f_cut_lo[n_real:], f_cut_hi[n_real:] = f_lo[n_real:], f_hi[n_real:]
        front = n_real + np.flatnonzero(f_side[n_real:] > 0)          # before their feature's apex
        back = n_real + np.flatnonzero(f_side[n_real:] < 0)
        np.maximum.at(f_cut_lo, f_par[front - n_real], f_hi[front])
        np.minimum.at(f_cut_hi, f_par[back - n_real], f_lo[back])

    # seeds and first membership; picked peaks seed before second maxima, which then start a component only where
    # no picked peak has its apex
    order_t = np.argsort(f_t, kind="stable").astype(np.int64)
    comp0, tight0, _, seeds = _cluster(off, mz, inten, f_mz, f_tol, f_h, f_t, f_lo, f_hi, f_tau, f_rcut,
                                       np.lexsort((-f_h, np.arange(n) >= n_real)).astype(np.int64), order_t, f_t[order_t],
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
    m_off, m_val = _models(off, mz, inten, f_mz, f_tol, c_lo, c_hi, c_apex, mem_off, mem_idx, D.model_ions,
                           f_cut_lo, f_cut_hi, f_zero)
    if D.unimodal:
        _unimodal(m_off, m_val, float(D.model_rise))

    # shared-ion components dissolve; their ions stay out of every model
    n_mem = np.diff(mem_off)
    alive = _composites(c_t, c_lo, c_hi, n_mem, m_off, m_val, f_rcut[seeds], D.composite_share, D.composite_max_ions) if D.composite and C > 2 \
        else np.ones(C, bool)
    f_shared = ~alive[comp0]
    # best model per ion, then the models once more from the final membership (a seed stays with its component)
    comp, f_r = _reassign(off, mz, inten, f_mz, f_tol, f_t, f_tau, f_rcut, f_shared, c_t, c_tau, c_lo, c_hi, alive,
                          m_off, m_val)
    # A lone ion with an uncertain apex. A component of one picked ion whose apex time is less certain than its
    # class's (a wide, flat or noisy top: _own_tolerance) is the weakest evidence of a compound: within that ion's
    # own tolerance of another component whose model its chromatogram follows, it is that component's ion and its
    # apex was misplaced. It joins without shaping the model (it sorts behind the members placed by their class's
    # tolerance). Components of several ions are left alone: ions that agree on an apex time are a compound, however
    # uncertain each fit says it is (a tailing peak is fitted badly by a parabola in all its ions alike).
    f_far = np.zeros(n, bool)
    n_lone = 0
    if D.tau_own and C > 1:
        f_own = _own_tolerance(f_tau, f_se, 0.5 * P.fwhm_med)
        lone = alive & (np.diff(mem_off) == 1) & (f_own[seeds] > f_tau[seeds])
        if lone.any():
            others = np.ones(n, bool)
            others[seeds[lone]] = False
            home, r_home = _reassign(off, mz, inten, f_mz, f_tol, f_t, f_own, f_rcut, others, c_t, c_tau, c_lo, c_hi,
                                     alive & ~lone, m_off, m_val)
            moved = np.flatnonzero(lone)[home[seeds[lone]] >= 0]
            n_lone = len(moved)
            if n_lone:
                alive[moved] = False
                comp, f_r = _reassign(off, mz, inten, f_mz, f_tol, f_t, f_tau, f_rcut, f_shared, c_t, c_tau, c_lo,
                                      c_hi, alive, m_off, m_val)
                j = seeds[moved]
                comp[j], f_r[j], f_far[j] = home[j], r_home[j], True
    info["n_lone_joined"] = int(n_lone)
    comp[seeds[alive]] = np.flatnonzero(alive)
    f_r[seeds[alive]] = np.maximum(f_r[seeds[alive]], f_rcut[seeds[alive]])
    mem_off, mem_idx = _member_lists(comp, np.where(f_far, 0.0, f_h), C)
    m_off, m_val = _models(off, mz, inten, f_mz, f_tol, c_lo, c_hi, c_apex, mem_off, mem_idx, D.model_ions,
                           f_cut_lo, f_cut_hi, f_zero)
    if D.unimodal:
        _unimodal(m_off, m_val, float(D.model_rise))
    n_mem = np.diff(mem_off)
    c_side = np.where(alive, f_side[seeds], 0)
    # the same models from the chromatograms above their floors: for the spectra and the reported profiles only.
    # Which ions and components belong together is decided on the models above (correlations and fits with a
    # constant, which a level under an ion does not move); a compound's profile and its ions' abundances are not
    # indifferent to it
    m_net = None
    if D.model_net and not c_side.any():
        _, m_net = _models(off, mz, inten, f_mz, f_tol, c_lo, c_hi, c_apex, mem_off, mem_idx, D.model_ions,
                           f_cut_lo, f_cut_hi, _floors(off, mz, inten, f_mz, f_tol, f_h, f_lo, f_hi, S, max(half, 3),
                                   float(D.model_rise)))
        if D.unimodal:
            _unimodal(m_off, m_net, float(D.model_rise))
    if c_side.any():
        m_off, m_val = _complete(c_lo, c_hi, c_apex, c_side, m_off, m_val, S)
    f_loose = _attach(f_t, f_tau, f_shared, comp, c_t, c_tau, alive) if D.attach else np.zeros(n, bool)
    n_loose = np.bincount(comp[f_loose], minlength=C) if f_loose.any() else np.zeros(C, np.int64)
    # delayed copies join the component they follow: its ions, its spectrum, a second model profile
    parent = _delayed(off, mz, inten, f_mz, f_tol, mem_off, mem_idx, D.model_ions, c_t, c_apex, c_lo, c_hi, alive,
                      f_rcut[seeds], m_off, m_val, float(P.dt), float(D.delay_reach), float(D.delay_f), f_zero) \
        if D.delay and C > 1 else np.full(C, -1, np.int64)
    n_explained = 0
    if D.explain and C > 1:
        # size = tight ions, then their summed height; a component is only ever explained by larger ones
        h_mem = np.bincount(comp[comp >= 0], weights=f_h[comp >= 0], minlength=C)
        rank = np.empty(C, np.int64)
        rank[np.lexsort((h_mem, n_mem))] = np.arange(C)
        own = alive & (parent < 0)             # a delayed copy is its parent's: not tested, not an explanation
        kept = _explained(off, mz, inten, f_mz, f_tol, mem_off, mem_idx, D.model_ions, c_t, c_lo, c_hi, f_rcut[seeds],
                          own, rank, m_off, m_val, int(D.max_neighbours), float(D.explain_f),
                          bool(D.explain_smaller))
        gone = own & ~kept
        orphan = (parent >= 0) & gone[np.maximum(parent, 0)]
        gone |= orphan
        parent[orphan] = -1
        n_explained = int(gone.sum())
        if n_explained:
            alive = alive & ~gone
            gone_f = (comp >= 0) & gone[np.maximum(comp, 0)]       # their ions: shared until placed again (_rejoin)
            f_shared = f_shared | gone_f
            comp = np.where(gone_f, -1, comp)
            mem_off, mem_idx = _member_lists(comp, f_h, C)
            n_mem = np.diff(mem_off)
    n_rejoined = 0
    if n_explained:
        todo = np.flatnonzero(gone_f)
        to, r_to = _rejoin(off, mz, inten, f_mz, f_tol, f_lo, f_hi, f_t, f_rcut, todo, c_t, c_lo, c_hi, alive, m_off,
                           m_val, int(D.max_neighbours), float(D.rejoin_share))
        ok = to >= 0
        n_rejoined = int(ok.sum())
        comp[todo[ok]] = to[ok]
        f_r[todo[ok]] = r_to[ok]
        f_shared[todo[ok]] = False
        mem_off, mem_idx = _member_lists(comp, f_h, C)
        n_mem = np.diff(mem_off)
    # a group = a component and its copies; it is reported at its member with the most tight ions
    group = [[int(c)] for c in range(C)]
    for b in np.flatnonzero(parent >= 0):
        group[parent[b]].append(int(b))
    n_own = n_mem.copy()
    for c in range(C):
        if len(group[c]) > 1:
            group[c].sort(key=lambda d: -n_own[d])
            n_mem[c] = int(n_own[group[c]].sum())
            n_loose[c] = int(n_loose[group[c]].sum())
    f_delayed = (comp >= 0) & (parent[np.maximum(comp, 0)] >= 0)

    # spectra
    if m_net is not None:
        m_val = m_net
    live = np.flatnonzero(alive)
    new_id = np.full(C, -1, np.int64)
    new_id[live] = 0
    spec_mz, spec_ab, spec_sh, counts, rows, row_of, profs = [], [], [], [], [], [], []
    for pos, c0 in enumerate(live):
        if parent[c0] >= 0:                # reported with the component it is a delayed copy of
            new_id[c0] = -4
            continue
        if n_mem[c0] < D.min_features:
            new_id[c0] = -3
            continue
        own = group[c0]
        c = own[0]                         # the member the group is reported at: the one with the most tight ions
        lo, hi = int(min(c_lo[d] for d in own)), int(max(c_hi[d] for d in own))
        # overlapping live neighbours, nearest in apex time first
        reach = D.max_neighbours + len(own)
        nb = [int(d) for d in live[max(0, pos - reach):pos + reach + 1]
              if d not in own and c_hi[d] >= lo and c_lo[d] <= hi]
        nb = sorted(nb, key=lambda d: abs(c_t[d] - c_t[c]))[:D.max_neighbours]
        X = np.column_stack([np.ones(hi - lo + 1)] + [_on_window(d, lo, hi, c_lo, m_off, m_val) for d in own + nb])
        k_own = 1 + len(own)
        X = X[:, np.concatenate([np.ones(k_own, bool), X[:, k_own:].max(axis=0) > 0])] if X.shape[1] > k_own else X
        a, b = int(off[c_apex[c]]), int(off[c_apex[c] + 1])
        cand = np.arange(a, b, dtype=np.int64)
        c_tol = ion_tol(mz[a:b], inten[a:b], P)
        mine = np.sort(np.concatenate([f_mz[mem_idx[mem_off[d]:mem_off[d + 1]]] for d in own]))
        member = np.zeros(b - a, bool)
        if len(mine):
            q = np.clip(np.searchsorted(mine, mz[a:b]), 1, len(mine) - 1) if len(mine) > 1 else np.zeros(b - a, np.int64)
            near = np.minimum(np.abs(mine[q] - mz[a:b]), np.abs(mine[np.maximum(q - 1, 0)] - mz[a:b]))
            member = near <= c_tol
        ab, nmod, mzw = _fit_ions(off, mz, inten, cand, c_tol, lo, hi + 1, X, D.f_min, D.min_points, 3, len(own),
                                  member)
        keep = ab > 0
        if keep.any():
            keep &= ab >= D.min_rel * ab.max()
        if keep.sum() < D.min_ions:
            new_id[c0] = -2
            continue
        m_, a_, s_ = mzw[keep], ab[keep], nmod[keep] > 1
        k = np.argsort(m_, kind="stable")
        m_, a_, s_ = m_[k], a_[k], s_[k]
        spec_mz.append(m_); spec_ab.append(a_); spec_sh.append(s_); counts.append(len(m_)); row_of.append(int(c0))
        profs.append(_on_window(c, lo, hi, c_lo, m_off, m_val))
        base = int(np.argmax(a_)) if len(a_) else -1
        uniq = np.flatnonzero(~s_)
        quant = int(uniq[np.argmax(a_[uniq])]) if len(uniq) else base
        members = mem_idx[mem_off[c]:mem_off[c + 1]]
        rows.append(dict(rt=c_t[c], rt_min=rt[lo], rt_max=rt[hi], scan_apex=int(c_apex[c]), scan_min=lo, scan_max=hi,
                         n_features=int(n_mem[c0]), n_loose=int(n_loose[c0]), n_delayed=len(own) - 1, n_ions=len(m_),
                         n_shared_ions=int(s_.sum()),
                         base_mz=m_[base] if base >= 0 else np.nan, base_height=a_[base] if base >= 0 else 0.0,
                         quant_mz=m_[quant] if quant >= 0 else np.nan, quant_height=a_[quant] if quant >= 0 else 0.0,
                         tic=float(a_.sum()), model_r=float(np.median(f_r[members])) if len(members) else np.nan,
                         apex_tol_s=60.0 * float(c_tau[c]), shoulder=bool(seeds[c] >= n_real)))
    # by apex time (a group reported at a copy can come after the next component's)
    order = np.argsort(np.array([r["rt"] for r in rows]), kind="stable") if rows else np.zeros(0, np.int64)
    rows = [rows[i] for i in order]
    spec_mz, spec_ab, spec_sh = [spec_mz[i] for i in order], [spec_ab[i] for i in order], [spec_sh[i] for i in order]
    counts = [counts[i] for i in order]
    profs = [profs[i] for i in order]
    comps = pd.DataFrame(rows, columns=cols)
    comps.insert(0, "component_id", np.arange(len(comps), dtype=np.int64))
    n_small, n_few = int((new_id == -2).sum()), int((new_id == -3).sum())
    for k, i in enumerate(order):
        new_id[row_of[i]] = k
    for b in np.flatnonzero(new_id == -4):          # a delayed copy's ions are its parent's
        new_id[b] = new_id[parent[b]] if new_id[parent[b]] >= 0 else -1
    new_id[new_id < -1] = -1
    f_comp = np.where(comp >= 0, new_id[np.maximum(comp, 0)], -1)[:n_real]
    f_loose = f_loose[:n_real] & (f_comp >= 0)
    f_delayed = f_delayed[:n_real] & (f_comp >= 0)
    f_r, f_shared = f_r[:n_real], f_shared[:n_real]
    info.update(n_components=int(len(comps)), n_composite=int((~alive).sum()) - n_explained, n_explained=n_explained,
                n_rejoined=n_rejoined, n_below_min_ions=n_small,
                n_below_min_features=n_few, n_features=int(n_real), n_assigned=int((f_comp >= 0).sum()),
                n_loose=int(f_loose.sum()), n_delayed_copies=int((parent >= 0).sum()), n_delayed=int(f_delayed.sum()),
                params=asdict(D))
    return Deconv(comps, np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
                  np.concatenate(spec_mz) if spec_mz else np.zeros(0), np.concatenate(spec_ab) if spec_ab else np.zeros(0),
                  np.concatenate(spec_sh) if spec_sh else np.zeros(0, bool),
                  f_comp, f_r, f_shared, info, f_loose, f_delayed,
                  np.concatenate([[0], np.cumsum([len(q) for q in profs])]).astype(np.int64),
                  np.concatenate(profs) if profs else np.zeros(0))
