"""Per-file retention-time warps.

A warp is a strictly increasing table rt_native(scan) -> rt_corr(scan) on the file's own MS1
grid; forward and inverse are piecewise-linear interpolation on that table (exact inverses),
with a constant shift beyond the ends. Fitting: monotone PCHIP through IRLS-weighted binned
medians of the anchor shifts, with fallbacks to a global shift or the identity."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import PchipInterpolator

MIN_SLOPE = 0.2
TUKEY_C = 4.685
ANCHORS_PER_KNOT = 12      # one knot per this many matched anchors


class TableWarp:
    def __init__(self, native: np.ndarray, corr: np.ndarray):
        native = np.asarray(native, dtype=np.float64)
        corr = np.asarray(corr, dtype=np.float64)
        if len(native) != len(corr) or len(native) < 1:
            raise ValueError("warp table needs equal, non-empty native/corr arrays")
        if len(native) > 1 and (np.any(np.diff(native) <= 0) or np.any(np.diff(corr) <= 0)):
            raise ValueError("warp table must be strictly increasing")
        self.native, self.corr = native, corr
        self.shift_lo = float(corr[0] - native[0])
        self.shift_hi = float(corr[-1] - native[-1])

    @classmethod
    def identity(cls, grid):
        g = np.asarray(grid, dtype=np.float64)
        return cls(g, g.copy())

    @classmethod
    def global_shift(cls, grid, d: float):
        g = np.asarray(grid, dtype=np.float64)
        return cls(g, g + d)

    def forward(self, t):
        t = np.asarray(t, dtype=np.float64)
        out = np.interp(t, self.native, self.corr)
        out = np.where(t < self.native[0], t + self.shift_lo, out)
        out = np.where(t > self.native[-1], t + self.shift_hi, out)
        return out

    def inverse(self, u):
        u = np.asarray(u, dtype=np.float64)
        out = np.interp(u, self.corr, self.native)
        out = np.where(u < self.corr[0], u - self.shift_lo, out)
        out = np.where(u > self.corr[-1], u - self.shift_hi, out)
        return out

    def shift(self, t):
        return self.forward(t) - np.asarray(t, dtype=np.float64)

    def scaled(self, lam):
        """the same warp with its shift multiplied by lam (one factor per grid point, 0 = identity)"""
        corr = self.native + np.asarray(lam, dtype=np.float64) * (self.corr - self.native)
        # a factor ramp only flattens a slope unless the shift changes by about a scan per scan;
        # guard anyway: corr[k] >= corr[k - 1] + 1e-9 (the running maximum of corr - k * 1e-9)
        step = 1e-9 * np.arange(len(corr))
        return TableWarp(self.native, np.maximum.accumulate(corr - step) + step)

    @property
    def is_identity(self):
        return bool(np.array_equal(self.native, self.corr))


@dataclass
class FitDiag:
    flag: str = "ok"
    n: int = 0
    n_inlier: int = 0
    K: int = 0
    mad_before_s: float = float("nan")
    mad_after_s: float = float("nan")
    inlier_frac: float = float("nan")
    knot_x: list = field(default_factory=list)
    knot_y: list = field(default_factory=list)
    gap: tuple = ()
    interp: str = "pchip"          # pchip | mixed (linear across unsupported segments) | linear
    residual: np.ndarray | None = None
    inlier: np.ndarray | None = None
    weight: np.ndarray | None = None


def weighted_median(x, w):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(w[o])
    if cw[-1] <= 0:
        return float(np.median(x))
    return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


def mad(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def weighted_quantile(x, w, q):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(w[o])
    if cw[-1] <= 0:
        return float(np.quantile(x, q))
    return float(x[o][min(np.searchsorted(cw, q * cw[-1]), len(x) - 1)])


def _local_line(xs, ds, ws, x_at, max_slope=0.1):    # 0.1 min/min = 6 s of drift per minute; measured drifts are < 1 s/min
    """weighted linear fit of the shift d on x, evaluated at x_at (median if degenerate)"""
    W = ws.sum()
    if len(xs) < 2 or W <= 0:
        return weighted_median(ds, ws) if W > 0 else float(np.median(ds))
    xm = float((ws * xs).sum() / W)
    dm = float((ws * ds).sum() / W)
    var = float((ws * (xs - xm) ** 2).sum())
    slope = float((ws * (xs - xm) * (ds - dm)).sum() / var) if var > 0 else 0.0
    slope = float(np.clip(slope, -max_slope, max_slope))
    return dm + slope * (x_at - xm)


def _merge_close_knots(kx, ky, kw, spacing):
    """merge knots closer than ``spacing`` (weighted means), so no near-tie can produce a steep
    secant for the cubic to extrapolate from"""
    kx, ky, kw = list(kx), list(ky), list(kw)
    i = 0
    while i + 1 < len(kx):
        if kx[i + 1] - kx[i] < spacing:
            wsum = kw[i] + kw[i + 1]
            kx[i] = (kx[i] * kw[i] + kx[i + 1] * kw[i + 1]) / wsum
            ky[i] = (ky[i] * kw[i] + ky[i + 1] * kw[i + 1]) / wsum
            kw[i] = wsum
            del kx[i + 1], ky[i + 1], kw[i + 1]
        else:
            i += 1
    return np.asarray(kx), np.asarray(ky), np.asarray(kw)


def _segment_support(kx, x, gap_max):
    """True for knot segments that may use the cubic: at least one anchor strictly inside and
    not longer than gap_max; other segments are bridged by a straight line"""
    inside = np.histogram(x, bins=kx)[0] if len(kx) > 1 else np.zeros(0, int)
    return (inside >= 1) & (np.diff(kx) <= gap_max)


def _evaluate(f, kx, ky, seg_ok, t, force_linear=False):
    """mixed interpolant on points t inside [kx[0], kx[-1]]; constant shift outside"""
    t = np.asarray(t, dtype=np.float64)
    lin = np.interp(t, kx, ky)
    if force_linear or f is None:
        out = lin
    else:
        seg = np.clip(np.searchsorted(kx, t, side="right") - 1, 0, len(kx) - 2)
        out = np.where(seg_ok[seg], f(t), lin)
    out = np.where(t < kx[0], t + (ky[0] - kx[0]), out)
    out = np.where(t > kx[-1], t + (ky[-1] - kx[-1]), out)
    return out


def pav_increasing(y, w):
    """weighted pool-adjacent-violators: the non-decreasing sequence closest to y (L2, weights w)"""
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi)
        wts.append(wi)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            c = cnt[-2] + cnt[-1]
            ww = wts[-2] + wts[-1]
            vals[-2:] = [v]
            wts[-2:] = [ww]
            cnt[-2:] = [c]
    return np.repeat(vals, cnt)


def _pairs(mz_f, rt_f, mz_t, rt_t, ppm):
    """all (i, j, rt_t[j] - rt_f[i]) with |dmz| <= ppm, vectorized through a sorted target"""
    o = np.argsort(mz_t, kind="stable")
    smz, srt = mz_t[o], rt_t[o]
    lo = np.searchsorted(smz, mz_f * (1 - ppm * 1e-6))
    hi = np.searchsorted(smz, mz_f * (1 + ppm * 1e-6), side="right")
    cnt = hi - lo
    tot = int(cnt.sum())
    if tot == 0:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0)
    i_idx = np.repeat(np.arange(len(mz_f)), cnt)
    starts = np.repeat(np.cumsum(cnt) - cnt, cnt)
    j_s = np.arange(tot) - starts + np.repeat(lo, cnt)
    j_idx = o[j_s]
    return i_idx, j_idx, srt[j_s] - rt_f[i_idx]


def coarse_shift(mz_f, rt_f, mz_t, rt_t, ppm: float, W: float, bin_width: float):
    """Mode of (rt_target - rt_file) over all m/z-matched anchor pairs within +/- W minutes.
    Returns (delta, support, sigma_mode)."""
    _, _, d = _pairs(np.asarray(mz_f, float), np.asarray(rt_f, float), np.asarray(mz_t, float),
                     np.asarray(rt_t, float), ppm)
    d = d[np.abs(d) <= W]
    if len(d) == 0:
        return 0.0, 0, float("nan")
    edges = np.arange(-W - bin_width, W + 2 * bin_width, bin_width)
    h, _ = np.histogram(d, edges)
    hs = np.convolve(h, np.ones(3), mode="same")
    k = int(np.argmax(hs))
    centre = 0.5 * (edges[k] + edges[k + 1])
    near = d[np.abs(d - centre) <= 1.5 * bin_width]
    support = int(len(near))
    if support == 0:
        return 0.0, 0, float("nan")
    delta = float(np.median(near))
    wide = d[np.abs(d - delta) <= 3 * bin_width]
    sigma = 1.4826 * mad(wide) if len(wide) > 2 else bin_width
    return delta, support, float(max(sigma, 0.25 * bin_width))


def match_anchors(mz_f, rt_f, mz_t, rt_t, ppm: float, delta: float, W_fine: float):
    """mutual-unique matches (file anchor i, target anchor j) after shifting file RTs by delta"""
    i_idx, j_idx, d = _pairs(np.asarray(mz_f, float), np.asarray(rt_f, float), np.asarray(mz_t, float),
                             np.asarray(rt_t, float), ppm)
    keep = np.abs(d - delta) <= W_fine
    i_idx, j_idx = i_idx[keep], j_idx[keep]
    if len(i_idx) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    ci = np.bincount(i_idx, minlength=len(mz_f))
    cj = np.bincount(j_idx, minlength=len(mz_t))
    m = (ci[i_idx] == 1) & (cj[j_idx] == 1)
    return i_idx[m], j_idx[m]


def fit_warp(x, y, scan_rt, dt: float | None = None, w0=None):
    """Fit rt_target = f(rt_native) from anchor pairs (x native, y target), tabulated on scan_rt.
    w0: prior weight per anchor (1/cluster size for co-eluting anchors of one compound); the
    robust (Tukey) weights multiply it. Returns (TableWarp, FitDiag)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w0 = np.ones(len(x)) if w0 is None else np.asarray(w0, dtype=np.float64)
    g = np.asarray(scan_rt, dtype=np.float64)
    if dt is None:
        dt = float(np.median(np.diff(g))) if len(g) > 1 else 0.01
    span = float(g[-1] - g[0]) if len(g) > 1 else 0.0
    diag = FitDiag(n=int(len(x)))
    d = y - x
    if len(x) < 4:
        diag.flag = "identity:too_few_matches"
        return TableWarp.identity(g), diag
    d0 = float(np.median(d))
    diag.mad_before_s = 60 * 1.4826 * mad(d)
    if len(x) < 8:
        diag.flag = "global_shift:too_few_matches"
        diag.mad_after_s = diag.mad_before_s
        return TableWarp.global_shift(g, d0), diag
    xs = np.sort(x)
    gaps = np.diff(xs)
    gap_max = max(1.0, 0.2 * span)
    if len(gaps) and gaps.max() > gap_max:   # recorded; the stretch is bridged by a straight line below
        k = int(np.argmax(gaps))
        diag.gap = (float(xs[k]), float(xs[k + 1]))
    w = w0.copy()
    spacing_min = float(np.clip(span / 32.0, 0.1, 1.0))
    f = None
    knot_x = knot_y = None
    seg_ok = None
    r = d - d0
    for _ in range(3):
        # knot count from the number of matched anchors; knot positions from unweighted quantiles
        # of x, so they stay put across IRLS iterations. Within a bin the knot is the weighted
        # median (prior 1/cluster weight x robust weight) of x and of the shift d.
        K = int(np.clip(len(x) // ANCHORS_PER_KNOT, 2, 1 + int(np.floor(span / spacing_min)) if span > 0 else 2))
        if w.sum() <= 0:
            break
        edges = np.quantile(x, np.linspace(0, 1, K + 1))
        edges[0], edges[-1] = x.min(), x.max()
        kx, ky, kw = [], [], []
        first_sel = last_sel = None
        for k in range(K):
            if k < K - 1:
                sel = (x >= edges[k]) & (x < edges[k + 1])
            else:
                sel = (x >= edges[k]) & (x <= edges[k + 1])
            ww = w[sel]
            if sel.sum() < 2:
                continue
            if ww.sum() <= 0:          # every anchor in the bin was down-weighted: keep the knot, unweighted
                ww = w0[sel]
            if first_sel is None:
                first_sel = sel
            last_sel = sel
            kx.append(weighted_median(x[sel], ww))
            ky.append(kx[-1] + weighted_median(d[sel], ww))
            kw.append(float(ww.sum()))
        if len(kx) < 2:
            break
        # end knots at the 2 % / 98 % quantiles of the anchor range, placed by a weighted local
        # line through the outermost bin (slope capped at a physical drift rate), so the fit
        # covers the anchors it has
        x_lo, x_hi = float(np.quantile(x, 0.02)), float(np.quantile(x, 0.98))
        if x_lo < kx[0] - 1e-9:
            kx.insert(0, x_lo)
            ky.insert(0, x_lo + _local_line(x[first_sel], d[first_sel], w[first_sel], x_lo))
            kw.insert(0, 0.5 * float(w[first_sel].sum()))
        if x_hi > kx[-1] + 1e-9:
            kx.append(x_hi)
            ky.append(x_hi + _local_line(x[last_sel], d[last_sel], w[last_sel], x_hi))
            kw.append(0.5 * float(w[last_sel].sum()))
        kx, ky, kw = np.asarray(kx), np.asarray(ky), np.asarray(kw)
        o = np.argsort(kx, kind="stable")
        kx, ky, kw = _merge_close_knots(kx[o], ky[o], kw[o], spacing_min)
        if len(kx) < 2:
            break
        ky = pav_increasing(ky, kw)
        for k in range(1, len(kx)):
            ky[k] = max(ky[k], ky[k - 1] + MIN_SLOPE * (kx[k] - kx[k - 1]))
        f = PchipInterpolator(kx, ky)
        seg_ok = _segment_support(kx, x, gap_max)
        knot_x, knot_y = kx, ky
        r = y - _evaluate(f, kx, ky, seg_ok, x)
        # robust scale, floored at one scan: apex positions are not known better than that, and a
        # tighter scale lets a region the first pass misfits lose all its anchors
        s = max(1.4826 * mad(r), dt)
        u = r / (TUKEY_C * s)
        w = w0 * np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0)
    if f is None:
        diag.flag = "global_shift:degenerate_knots"
        diag.mad_after_s = diag.mad_before_s
        return TableWarp.global_shift(g, d0), diag
    inlier = np.abs(r) <= max(3 * 1.4826 * mad(r), dt)
    diag.inlier_frac = float(inlier.mean())
    diag.residual, diag.inlier, diag.weight = r, inlier, w
    diag.n_inlier = int(inlier.sum())
    diag.K = int(len(knot_x))
    diag.knot_x, diag.knot_y = [float(v) for v in knot_x], [float(v) for v in knot_y]
    if diag.inlier_frac < 0.6:
        diag.flag = "global_shift:low_inlier_fraction"
        diag.mad_after_s = 60 * 1.4826 * mad(d - d0)
        return TableWarp.global_shift(g, d0), diag
    diag.mad_after_s = 60 * 1.4826 * mad(r[inlier])
    diag.interp = "pchip" if bool(np.all(seg_ok)) else "mixed"
    corr = _evaluate(f, knot_x, knot_y, seg_ok, g)
    # the curve must stay within the range of the knot shifts (plus the residual scatter): if the
    # cubic bulges beyond it anywhere, use straight lines between knots everywhere
    kshift = knot_y - knot_x
    margin = 3 * 1.4826 * mad(r[inlier]) + 2 * dt
    sh = corr - g
    if sh.min() < kshift.min() - margin or sh.max() > kshift.max() + margin:
        corr = _evaluate(f, knot_x, knot_y, seg_ok, g, force_linear=True)
        diag.interp = "linear"
    # strictly increasing: nudge exact ties (cannot happen with MIN_SLOPE > 0, kept as a guard)
    for k in range(1, len(corr)):
        if corr[k] <= corr[k - 1]:
            corr[k] = corr[k - 1] + 1e-9
    return TableWarp(g, corr), diag
