"""Study-level RT alignment: medoid reference whose native axis is the output axis, refined by
consensus iterations over the union of every file's anchors, then a do-no-harm check of every warp
on held-out (non-anchor) features. Each file's m/z offset is found first, coarsely, against the medoid
within MZ_SEARCH_PPM and removed before anchors are matched within the narrow tolerance; the fine offset
is measured on the matched anchors, and the file's offset (mz_offset_ppm) is the sum."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .anchors import _isolated, select_anchors
from .grouping import group_features
from .warp import (FitDiag, TableWarp, _pairs, coarse_mz_offset, coarse_shift, fit_warp, mad, match_anchors,
                   weighted_median)


CONSENSUS_MIN_FILES = 3    # a consensus anchor group must be present in at least this many files
GATE_MIN_SEG = 25          # validation features per do-no-harm segment, at least
GATE_MAX_SEG = 12          # do-no-harm segments per file, at most
MZ_SEARCH_PPM = 50.0       # how far the coarse m/z offset search looks, either side (ppm)
PROBE_BINS, PROBE_PER_BIN = 50, 40   # mz_probe_features: the run in equal RT bins, the tallest per bin
ANCHOR_MATCH_MIN = 0.8     # a file matching fewer anchors than this share of the median file's is flagged
RT_RESIDUAL_RATIO = 4.0    # a file whose held-out features stay this many times the study's median deviation from the
                           # other files after correction is flagged rt_residual_high (benchmark studies: <= 1.8 x)
RT_RESIDUAL_MIN_FILES = 8  # ... when at least this many files were checked (the median of a few means little)
WARN_FLAGS = ("few_anchors_matched", "rt_residual_high")   # shown first in FileAlignment.flag, so the report marks the file


@dataclass
class AlignParams:
    ppm: float = 5.0
    rt_tol_max: float = 0.25
    n_iter: int = 3
    n_medoid_candidates: int = 25
    holdout: bool = False
    gate: bool = True          # do-no-harm: scale each warp down where held-out features do not support it
    # m/z: the five below are on by default since 2026-10-31 (judged on the PR002464 m/z error test: 25 injections
    # whose calibration jumps mid-run; no change on the 12 clean benchmark datasets); all off = one offset per file
    mz_dynamic: bool = True    # each file's m/z offset as a curve over its run time (running median over its
                               # anchors, coarse and fine) instead of one number: calibration that drifts or jumps
    mz_relaxed_second_mode: bool = True   # a second coarse m/z mode needs >= 10 pairs, not >= 10 % of the anchors:
                               # a calibration jump early or late in the run leaves its short part too few pairs
    mz_step_curve: bool = True    # mz_dynamic: a file with two coarse modes gets a step at the time its pairs
                               # switch mode (when a step fits them as well as the running median does), not a
                               # curve interpolated across a stretch without pairs
    mz_fallback_ref: bool = True    # a file with no coarse mode against the reference is searched against the
                               # files on a known scale (one mode) and takes the one it shares most pairs with:
                               # a U-13C-only sample has no partners in a 12C reference
    mz_probe_features: bool = True    # mz_dynamic + mz_step_curve: look for a calibration jump again with the
                               # strongest features over the whole run (PROBE_BINS x PROBE_PER_BIN), not only the
                               # anchors, which leave out the run's ends and can have gaps


@dataclass
class FileAlignment:
    stem: str
    warp: TableWarp
    role: str = "other"
    tier: int = -1
    n_anchors: int = 0
    anchor_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    coarse_shift: float = 0.0
    coarse_support: int = 0
    diag: FitDiag = field(default_factory=FitDiag)
    flags: list = field(default_factory=list)
    matched: pd.DataFrame | None = None
    holdout: dict | None = None
    dt: float = float("nan")
    span: float = float("nan")
    mz_offset_ppm: float = 0.0     # file m/z minus consensus m/z: coarse + median over inlier anchors
    mz_coarse_ppm: float = 0.0     # the coarse part, removed before anchors are matched (0: no clear mode)
    mz_coarse_support: int = 0     # anchor pairs at the coarse mode
    anchor_w: np.ndarray = field(default_factory=lambda: np.zeros(0))   # 1/cluster size per anchor
    gate: dict | None = None       # do-no-harm step: validation count, segment nodes and factors, |dev| before/after
    warp_fit: TableWarp | None = None   # the anchor fit before the do-no-harm step (None = applied as fitted)
    match_ratio: float = float("nan")   # anchors matched / the median file's (not the medoid)
    mz_coarse_curve: np.ndarray | None = None   # mz_dynamic: the coarse offset per scan of the native grid
    mz_curve: np.ndarray | None = None          # mz_dynamic: the offset (coarse + fine) per scan of the native grid

    def mz_offset_at(self, rt):
        """the file's m/z offset (ppm) at native times rt: its curve (mz_dynamic), else the one number"""
        rt = np.asarray(rt, np.float64)
        if self.mz_curve is None:
            return np.full(rt.shape, self.mz_offset_ppm)
        return np.interp(rt, self.warp.native, self.mz_curve)

    @property
    def flag(self):
        base = "medoid" if self.role == "medoid" else self.diag.flag
        warn = [f for f in self.flags if f in WARN_FLAGS]
        return ";".join(warn + [base] + [f for f in self.flags if f not in WARN_FLAGS])


@dataclass
class AlignResult:
    files: dict
    medoid: str | None
    skipped: bool
    ppm_tol: float
    rt_tol: float
    dt_median: float
    pooled_mad_s: float
    n_iter_done: int
    notes: list = field(default_factory=list)


def _file_grid_info(scan_rt):
    g = np.asarray(scan_rt, dtype=np.float64)
    dt = float(np.median(np.diff(g))) if len(g) > 1 else 0.01
    return float(g[0]), float(g[-1]), dt, float(g[-1] - g[0])


def _mz(fa: FileAlignment, feats: pd.DataFrame, idx=None):
    """the file's feature m/z (all rows, or rows idx) with its coarse m/z offset removed (at each feature's time,
    with mz_dynamic)"""
    mz = feats["mz"].values if idx is None else feats["mz"].values[idx]
    if fa.mz_coarse_curve is None:
        return mz * (1.0 - fa.mz_coarse_ppm * 1e-6)
    rt = feats["rt"].values if idx is None else feats["rt"].values[idx]
    return mz * (1.0 - np.interp(rt, fa.warp.native, fa.mz_coarse_curve) * 1e-6)


def _mz_curve(x, y, grid):
    """running median of y (ppm) over x (native RT) in windows of k points (a quarter of them, 9 to 51, odd; moved
    inward at the ends rather than padded), tabulated on grid and flat beyond the ends: a calibration drift is
    followed and a jump stays a step (the median switches where the window's majority does), while on a steady
    file the curve stays within the median's own noise; None with fewer than 10 points"""
    n = len(x)
    if n < 10:
        return None
    o = np.argsort(x, kind="stable")
    x, y = np.asarray(x, np.float64)[o], np.asarray(y, np.float64)[o]
    k = min(n, int(np.clip(n // 4, 9, 51)) | 1)
    windows = np.median(np.lib.stride_tricks.sliding_window_view(y, k), axis=1)
    return np.interp(grid, x, windows[np.clip(np.arange(n) - k // 2, 0, n - k)])


def _mz_step(x, y, modes, grid):
    """a calibration jump as one step on grid: each point (native RT x, ppm y) is labelled by the nearer of the two
    modes, the step sits where the labels switch with fewest points on the wrong side (either order), and each side
    takes the median of its own label's points; None when the labels interleave (either side < 80 % its own label)
    or a side holds < 5 points of its label"""
    n = len(x)
    if n < 10:
        return None
    o = np.argsort(x, kind="stable")
    x, y = np.asarray(x, np.float64)[o], np.asarray(y, np.float64)[o]
    lab = np.argmin([np.abs(y - m) for m, _, _ in modes[:2]], axis=0)
    best = None
    for first in (0, 1):
        is_first = lab == first
        wrong = np.cumsum(~is_first)[:-1] + (is_first.sum() - np.cumsum(is_first)[:-1])   # split after i
        i = int(np.argmin(wrong))
        if best is None or wrong[i] < best[0]:
            best = (int(wrong[i]), i + 1, first)
    _, cut, first = best
    left, right = y[:cut][lab[:cut] == first], y[cut:][lab[cut:] != first]
    if len(left) < 0.8 * cut or len(right) < 0.8 * (n - cut) or len(left) < 5 or len(right) < 5:
        return None
    t_c = 0.5 * (x[cut - 1] + x[cut])
    return np.where(np.asarray(grid) < t_c, float(np.median(left)), float(np.median(right)))


def _step_or_curve(x, y, modes, grid):
    """(curve, True) when a step (_mz_step) explains the points about as well as the running median (_mz_curve):
    within 1.25x its median absolute residual, or 0.5 ppm, so a jump; a gradual drift between two levels fits the
    running median clearly better and keeps it: (curve, False); (None, False) without enough points"""
    med = _mz_curve(x, y, grid)
    st = _mz_step(x, y, modes, grid)
    if st is None or med is None:
        return med, False
    x, y, grid = np.asarray(x, np.float64), np.asarray(y, np.float64), np.asarray(grid, np.float64)
    r_st = np.median(np.abs(y - np.interp(x, grid, st)))
    r_med = np.median(np.abs(y - np.interp(x, grid, med)))
    return (st, True) if r_st <= max(1.25 * r_med, 0.5) else (med, False)


def _probe_idx(feats: pd.DataFrame, grid):
    """the strongest clean monoisotopic features spread over the whole run: the run in PROBE_BINS equal RT bins, the
    PROBE_PER_BIN tallest of each (score >= 0.5, S/N >= 5, >= 4 scans); no isolation rule and no margin at the ends,
    unlike the anchors (anchors.select_anchors)"""
    ok = (feats["score"].values >= 0.5) & (feats["snr"].values >= 5) & (feats["n_scans"].values >= 4) \
        & (feats["iso_offset"].values == 0)
    cand = np.flatnonzero(ok)
    t0, t1 = float(grid[0]), float(grid[-1])
    b = np.clip(((feats["rt"].values[cand] - t0) / max(t1 - t0, 1e-9) * PROBE_BINS).astype(int), 0, PROBE_BINS - 1)
    h = feats["height"].values[cand]
    o = np.lexsort((-h, b))
    rank = np.arange(len(o)) - np.searchsorted(b[o], b[o])          # position within its bin, tallest first
    return np.sort(cand[o[rank < PROBE_PER_BIN]])


def _labelled_pairs(mz_f, rt_f, mz_t, rt_t, modes, W):
    """per file anchor, its pair (within MZ_SEARCH_PPM and W) nearest one of the file's credible modes (offset,
    support, half-width), when within that mode's half-width: (anchor RT, ppm difference)"""
    i, j, d_rt = _pairs(mz_f, rt_f, mz_t, rt_t, MZ_SEARCH_PPM)
    keep = np.abs(d_rt) <= W
    i, j = i[keep], j[keep]
    d = (mz_f[i] - mz_t[j]) / mz_t[j] * 1e6
    dist = np.min([np.abs(d - m) - h for m, _, h in modes], axis=0) if len(d) else d
    sel = dist <= 0
    i, d, dist = i[sel], d[sel], dist[sel]
    o = np.lexsort((dist, i))
    first = o[np.r_[True, np.diff(i[o]) != 0]] if len(o) else o
    return rt_f[i[first]], d[first]


def _coarse_mz(files, features, stems, ref, ppm: float, dynamic: bool = False, relaxed_second: bool = False,
               step: bool = False, fallback: bool = False, probes: bool = False):
    """every file's coarse m/z offset against the anchors of file ref (coarse_mz_offset within MZ_SEARCH_PPM and
    the file's coarse RT window). A mode is credible when it holds >= 10 pairs and >= 10 % of the smaller anchor
    set and stands 4x above the chance pairs; a second mode also needs >= 25 % of the first's pairs and must lie
    >= ppm (the narrow tolerance) from it. The study's centre is the median offset of the files with exactly one
    credible mode (ref: 0). A file with two (its calibration jumps mid-run, as in 23 PR002464 ST004502 files:
    0 ppm before ~4 min, -22 after; one offset cannot fit both parts) takes the mode nearer the centre, flagged
    two_mz_modes; a file with none takes the centre, flagged no_mz_mode. All are then expressed against the
    centre, so the study's scale is its typical file's, whichever file was the reference. Anchors matched within
    the narrow tolerance see only offsets well inside it: a file 5 ppm off loses a third of its matches and its
    fine offset can fall far short, one 10 ppm off loses nearly all, and with them its RT alignment and its chance
    to be the medoid. dynamic: the offset is a curve over the file's RT instead (_mz_curve of each anchor's pair
    nearest a credible mode: a file whose calibration jumps keeps both parts). relaxed_second: the second mode
    needs >= 10 pairs instead of >= 10 % of the smaller anchor set (the other tests stay). step: with dynamic, a file
    with two modes gets a step (_mz_step) instead of the running median. fallback: a file with no mode against ref is
    searched against every file on a known scale (one mode, or ref) and takes the modes of the one it shares most
    pairs with, moved by that file's own offset (flag mz_mode_via_other_file). probes (with dynamic and step): every
    file is searched again with its probe features (_probe_idx) against those of the file its modes came from; a
    second mode found there (>= 10 pairs, 4x the chance pairs, >= ppm from the first) that splits the run in two
    (_mz_step) gives the file that step, placed by the denser probe pairs (flag mz_step_from_probes)."""
    ma = files[ref].anchor_idx
    mz_m, rt_m = features[ref]["mz"].values[ma], features[ref]["rt"].values[ma]

    def anchors(s):
        b = files[s].anchor_idx
        return features[s]["mz"].values[b], features[s]["rt"].values[b]

    def credible(s, mz_t, rt_t):
        """the credible coarse modes of file s against anchors (mz_t, rt_t): up to two (offset, support, half)"""
        mz_s, rt_s = anchors(s)
        modes, bg = coarse_mz_offset(mz_s, rt_s, mz_t, rt_t, MZ_SEARCH_PPM, float(np.clip(0.1 * files[s].span, 0.3, 4.0)))
        need = max(10, 0.1 * min(len(mz_s), len(mz_t)))
        ok = [m for m in modes[:1] if m[1] >= need and m[1] >= 4 * bg]
        if ok and len(modes) > 1:
            m2 = modes[1]
            need2 = 10 if relaxed_second else need
            if m2[1] >= max(need2, 0.25 * ok[0][1], 4 * bg) and abs(m2[0] - ok[0][0]) >= ppm:
                ok.append(m2)
        return ok

    def _probe_step(s, g, shift):
        """a step for file s from its probe features against file g's (g's offset shift added), or None"""
        ps = _probe_idx(features[s], files[s].warp.native)
        pg = _probe_idx(features[g], files[g].warp.native)
        mz_s, rt_s = features[s]["mz"].values[ps], features[s]["rt"].values[ps]
        mz_g, rt_g = features[g]["mz"].values[pg], features[g]["rt"].values[pg]
        W = float(np.clip(0.1 * files[s].span, 0.3, 4.0))
        modes, bg = coarse_mz_offset(mz_s, rt_s, mz_g, rt_g, MZ_SEARCH_PPM, W)
        if len(modes) < 2:
            return None
        m1, m2 = modes[0], modes[1]
        if not (m1[1] >= max(10, 4 * bg) and m2[1] >= max(10, 4 * bg) and abs(m2[0] - m1[0]) >= ppm):
            return None
        x, d = _labelled_pairs(mz_s, rt_s, mz_g, rt_g, [m1, m2], W)
        c, is_step = _step_or_curve(x, d + shift, [(m1[0] + shift,) + m1[1:], (m2[0] + shift,) + m2[1:]],
                                    files[s].warp.native)
        return c if is_step else None

    found = {}
    via = {}            # fallback: file -> (the file its modes were measured against, that file's offset)
    for s in stems:
        if s == ref:
            found[s] = [(0.0, len(ma))]
        elif len(files[s].anchor_idx) and len(ma):
            found[s] = credible(s, mz_m, rt_m)
    if fallback:
        known = [g for g in stems if len(found.get(g, [])) == 1]
        for s in [s for s in stems if s != ref and s in found and not found[s]]:
            best, best_ok = None, []
            for g in known:
                if g != s and len(files[g].anchor_idx):
                    ok = credible(s, *anchors(g))
                    if ok and (not best_ok or ok[0][1] > best_ok[0][1]):
                        best, best_ok = g, ok
            if best_ok:
                shift = found[best][0][0]
                found[s] = [(o + shift, n, h) for o, n, h in best_ok]
                via[s] = (best, shift)
                files[s].flags.append("mz_mode_via_other_file")
    for s, m in found.items():
        if s != ref and not m:
            files[s].flags.append("no_mz_mode")
        elif s != ref and len(m) == 2:
            files[s].flags.append("two_mz_modes")
    single = [m[0][0] for m in found.values() if len(m) == 1]
    centre = float(np.median(single)) if single else 0.0
    chosen = {s: min(m, key=lambda x: abs(x[0] - centre))[:2] for s, m in found.items() if m}
    if dynamic:
        # each file's offset against ref over its RT, then the study's centre at each time: the median over the
        # files (ref: 0) of their offsets there, so the scale follows the typical file even when ref itself jumps
        curves = {}
        for s, m in found.items():
            if m and s != ref:
                g, shift = via.get(s, (ref, 0.0))
                mz_t, rt_t = (mz_m, rt_m) if g == ref else anchors(g)
                x, d = _labelled_pairs(*anchors(s), mz_t, rt_t, [(o - shift, n, h) for o, n, h in m],
                                       float(np.clip(0.1 * files[s].span, 0.3, 4.0)))
                d = d + shift
                c = None
                if probes and step:
                    c = _probe_step(s, g, shift)
                    if c is not None:
                        if "two_mz_modes" not in files[s].flags:
                            files[s].flags.append("two_mz_modes")
                        files[s].flags.append("mz_step_from_probes")
                if c is None and step and len(m) == 2:
                    c, is_step = _step_or_curve(x, d, m, files[s].warp.native)
                    c = c if is_step else None
                if c is None:
                    c = _mz_curve(x, d, files[s].warp.native)
                if c is not None:
                    curves[s] = c
        g = np.linspace(min(files[s].warp.native[0] for s in stems), max(files[s].warp.native[-1] for s in stems), 256)
        at_g = [np.interp(g, files[s].warp.native, curves[s]) if s in curves else np.full(len(g), chosen[s][0])
                for s in chosen]
        centre_g = np.median(np.vstack(at_g), axis=0) if at_g else np.zeros(len(g))
    for s in stems:
        fa = files[s]
        off, sup = chosen.get(s, (centre, 0))
        fa.mz_coarse_ppm, fa.mz_coarse_support = off - centre, sup
        fa.mz_offset_ppm = fa.mz_coarse_ppm
        if dynamic:
            cg = np.interp(fa.warp.native, g, centre_g)
            own = curves[s] if s in curves else (off if s in chosen else cg)       # no mode: the study's scale
            fa.mz_coarse_curve = own - cg
            fa.mz_curve = fa.mz_coarse_curve.copy()
            fa.mz_coarse_ppm = fa.mz_offset_ppm = float(np.median(fa.mz_coarse_curve))


def _fit_file(fa: FileAlignment, feats: pd.DataFrame, grid, tgt_mz, tgt_rt, P: AlignParams):
    """coarse shift, mutual-unique matching and warp fit of one file against the target"""
    t0, t_end, dt, span = _file_grid_info(grid)
    a = fa.anchor_idx
    mz_f, rt_f = _mz(fa, feats, a), feats["rt"].values[a]
    W_coarse = float(np.clip(0.1 * span, 0.3, 4.0))
    bin_w = max(2 * dt, 0.5 / 60)
    delta, support, sigma = coarse_shift(mz_f, rt_f, tgt_mz, tgt_rt, P.ppm, W_coarse, bin_w)
    fa.flags = [f for f in fa.flags if f not in ("no_coarse_mode",)]
    if support < max(10, 0.1 * len(a)):
        delta, sigma = 0.0, bin_w
        fa.flags.append("no_coarse_mode")
    fa.coarse_shift, fa.coarse_support = delta, support
    W_fine = max(0.1, 6 * dt, 3 * (sigma if np.isfinite(sigma) else bin_w))
    i, j = match_anchors(mz_f, rt_f, tgt_mz, tgt_rt, P.ppm, delta, W_fine)
    x, y = rt_f[i], tgt_rt[j]
    w0 = fa.anchor_w[i] if len(fa.anchor_w) == len(a) else None
    warp, diag = fit_warp(x, y, grid, dt, w0=w0)
    fa.warp, fa.diag = warp, diag
    fa.anchor_w_matched = w0
    m = pd.DataFrame({"feature_id": feats["feature_id"].values[a][i], "mz": feats["mz"].values[a][i],
                      "rt_native": x, "rt_corr": warp.forward(x), "consensus_rt": y,
                      "cluster_w": w0 if w0 is not None else 1.0})
    m["residual_before_s"] = 60 * (y - x)
    m["residual_after_s"] = 60 * (y - m["rt_corr"].values)
    if diag.weight is not None and len(diag.weight) == len(m):
        m["weight"] = diag.weight
        m["is_inlier"] = diag.inlier
        inl = diag.inlier
        if inl.sum() >= 5:
            r = (mz_f[i][inl] - tgt_mz[j][inl]) / tgt_mz[j][inl] * 1e6
            fine = float(np.median(r))
            fa.mz_offset_ppm = fa.mz_coarse_ppm + fine
            if P.mz_dynamic:
                coarse = fa.mz_coarse_curve if fa.mz_coarse_curve is not None else fa.mz_coarse_ppm
                curve = _mz_curve(x[inl], r, grid)          # None (too few inliers): the fine offset, flat
                fa.mz_curve = np.zeros(len(grid)) + coarse + (curve if curve is not None else fine)
                fa.mz_offset_ppm = float(np.median(fa.mz_curve))
    else:
        m["weight"] = np.nan
        m["is_inlier"] = np.abs(m["residual_after_s"].values) <= 60 * max(3 * 1.4826 * mad(y - warp.forward(x)), dt) \
            if len(m) else np.zeros(0, bool)
    fa.matched = m
    return fa


def _holdout(fa: FileAlignment, grid):
    m = fa.matched
    if m is None or len(m) < 16:
        return None
    o = np.argsort(m["rt_native"].values, kind="stable")
    even, odd = o[::2], o[1::2]
    x, y = m["rt_native"].values, m["consensus_rt"].values
    _, _, dt, _ = _file_grid_info(grid)
    w0 = fa.anchor_w_matched[even] if getattr(fa, "anchor_w_matched", None) is not None else None
    w, d = fit_warp(x[even], y[even], grid, dt, w0=w0)
    r_out = y[odd] - w.forward(x[odd])
    r_in = y[even] - w.forward(x[even])
    r_none = y[odd] - x[odd]
    return dict(n_fit=int(len(even)), n_eval=int(len(odd)), mad_in_s=60 * 1.4826 * mad(r_in),
                mad_out_s=60 * 1.4826 * mad(r_out), mad_none_s=60 * 1.4826 * mad(r_none), flag=d.flag)


def _loo_median(gid, u):
    """per member, the median of u over the OTHER members of its group (groups of >= 2)"""
    o = np.lexsort((u, gid))
    g, v = gid[o], u[o]
    start = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    size = np.diff(np.r_[start, len(g)])
    s0, n = np.repeat(start, size), np.repeat(size, size)
    rank = np.arange(len(g)) - s0
    m = n - 1                                  # the others, sorted: the full list without rank

    def other(k):                              # k-th of the others = k-th of the full list, skipping rank
        return v[s0 + k + (k >= rank)]
    out = np.empty(len(g))
    out[o] = 0.5 * (other((m - 1) // 2) + other(m // 2))
    return out


def _validation_set(files, features, stems, P: AlignParams, dt_med):
    """Held-out correspondences for the do-no-harm step: every non-anchor monoisotopic feature
    (S/N >= 3, >= 4 scans) with no other feature of its m/z (2 x ppm) within 2 W in its own file,
    grouped across files on the corrected axis within W (unambiguous by construction). Groups in at
    least as many files as the external checks require (all of 2-3 files, else half and >= 3).
    Returns stem -> (native rt, group id) of its members."""
    N = len(stems)
    W = max(0.1, 6 * dt_med)
    parts = []
    for k, s in enumerate(stems):
        f = features[s]
        ok = (f["iso_offset"].values == 0) & (f["snr"].values >= 3.0) & (f["n_scans"].values >= 4)
        ok[files[s].anchor_idx] = False
        cand = np.flatnonzero(ok)
        if len(cand):
            cand = cand[_isolated(f["mz"].values, f["rt"].values, cand, 2 * P.ppm, 2 * W)]
        x = f["rt"].values[cand]
        parts.append(pd.DataFrame({"mz": _mz(files[s], f, cand), "x": x, "u": files[s].warp.forward(x),
                                   "h": f["height"].values[cand], "file": k}))
    V = pd.concat(parts, ignore_index=True)
    if len(V) == 0:
        return {}
    u = V["u"].values
    gid, cons = group_features(V["mz"].values, u, u, u, V["h"].values, V["file"].values, N, P.ppm, W)
    need = N if N <= 3 else max(3, int(np.ceil(0.5 * N)))
    keep = np.asarray(cons["n_detected"])[gid] >= max(need, 2)
    V, gid = V[keep], gid[keep]
    return {s: (V["x"].values[V["file"].values == k], gid[V["file"].values == k]) for k, s in enumerate(stems)}


def _scale_factor(d, sh):
    """lambda in [0, 1] minimising sum |d + lambda sh| over a segment's validation features (native
    deviation d from the reference, warp shift sh): the weighted median of -d/sh with weights |sh|,
    clipped (the loss is convex in lambda). Also returns the loss reduction over lambda = 0,
    relative to the loss at lambda = 0."""
    a = np.abs(sh)
    ok = a > 1e-12
    if not ok.any():
        return 1.0, 0.0
    lam = float(np.clip(weighted_median(-d[ok] / sh[ok], a[ok]), 0.0, 1.0))
    l0, l1 = np.abs(d).sum(), np.abs(d + lam * sh).sum()
    return lam, float((l0 - l1) / l0) if l0 > 0 else 0.0


def _gate_factors(x, d, sh, lo_a, hi_a):
    """Segments along native RT: the stretches before the first and after the last matched anchor
    (where the warp is an extrapolated constant shift) are their own segments when they hold
    GATE_MIN_SEG validation features; the rest is cut at quantiles into segments of at least
    2 x GATE_MIN_SEG. Returns (segment node = median x, factor, relative loss reduction) arrays."""
    o = np.argsort(x, kind="stable")
    x, d, sh = x[o], d[o], sh[o]
    head, tail = x < lo_a, x > hi_a
    segs = []
    if head.sum() >= GATE_MIN_SEG:
        segs.append(head)
    else:
        head[:] = False
    if tail.sum() < GATE_MIN_SEG:
        tail[:] = False
    mid = np.flatnonzero(~head & ~tail)
    K = int(np.clip(len(mid) // (2 * GATE_MIN_SEG), 1, GATE_MAX_SEG))
    for c in np.array_split(mid, K):
        if len(c):
            sel = np.zeros(len(x), bool)
            sel[c] = True
            segs.append(sel)
    if tail.any():
        segs.append(tail)
    nodes = np.array([float(np.median(x[s])) for s in segs])
    lam, gain = np.array([_scale_factor(d[s], sh[s]) for s in segs]).T.reshape(2, -1)
    return nodes, lam, gain


def _do_no_harm(files, features, stems, medoid, P: AlignParams, dt_med, n_rounds=2):
    """Each warp is a hypothesis fitted to anchors (the strongest, most isolated ions). Check it on
    the held-out validation set: per segment of native RT, scale the warp's shift by the factor in
    [0, 1] that minimises the absolute deviation of the file's validation features from the median
    of the other files (0 = native axis, 1 = full warp); with 2-3 files, from the medoid (the output
    axis, never warped), since the 'median' of two others is their mean and one bad fit shifts it. Where the anchors' drift is not that of the
    other compounds, or the extrapolation beyond the anchors is wrong, the shift shrinks toward zero,
    so on these features the correction is never worse than none. Factors are interpolated linearly
    between segment medians (constant beyond). Two rounds: the references move with the gated warps
    of the other files; the factor always scales the original fitted warp."""
    val = _validation_set(files, features, stems, P, dt_med)
    if not val:
        return
    full = {s: files[s].warp for s in stems}
    gids = np.concatenate([val[s][1] for s in stems])
    fidx = np.concatenate([np.full(len(val[s][0]), k) for k, s in enumerate(stems)])
    off = np.cumsum([0] + [len(val[s][0]) for s in stems])
    med_u = np.full(int(gids.max()) + 1 if len(gids) else 0, np.nan)
    if len(stems) <= 3:      # two others: their median is their mean, which one bad fit shifts
        med_u[val[medoid][1]] = val[medoid][0]
    for _ in range(n_rounds):
        us = np.concatenate([files[s].warp.forward(val[s][0]) for s in stems])
        ref = _loo_median(gids, us)
        mu = med_u[gids]
        ref = np.where(np.isfinite(mu) & (fidx != stems.index(medoid)), mu, ref)
        new = {}
        for k, s in enumerate(stems):
            fa = files[s]
            x = val[s][0]
            if s == medoid or len(x) < GATE_MIN_SEG or full[s].is_identity:
                continue
            r = ref[off[k]:off[k + 1]]
            d = x - r
            sh = full[s].shift(x)
            m = fa.matched
            xa = m["rt_native"].values[m["is_inlier"].values.astype(bool)] if m is not None and len(m) else x
            if len(xa) == 0:
                xa = x
            nodes, lam, gain = _gate_factors(x, d, sh, float(xa.min()), float(xa.max()))
            g = full[s].native
            new[s] = full[s].scaled(np.interp(g, nodes, lam))
            lam_x = np.interp(x, nodes, lam)
            fa.gate = dict(n_val=int(len(x)), nodes=nodes, lam=lam, gain=gain,
                           dev_full_s=60 * float(np.mean(np.abs(d + sh))), dev_native_s=60 * float(np.mean(np.abs(d))),
                           dev_gated_s=60 * float(np.mean(np.abs(d + lam_x * sh))))
        for s, w in new.items():
            files[s].warp, files[s].warp_fit = w, full[s]


def flag_rt_residual(files, stems, medoid, dt_med) -> list:
    """Flag rt_residual_high on the files whose held-out features (the do-no-harm step's validation set) still sit
    >= RT_RESIDUAL_RATIO x the study's median deviation from the other files after correction, and >= 2 scans: their
    retention changed in a way the warp could not follow (in MX831181, compound-specific shifts of 1.5-4.6 s in the
    first injections after the column sat idle, which a function of RT alone cannot undo). Needs the do-no-harm numbers
    of >= RT_RESIDUAL_MIN_FILES files. Returns the flagged stems."""
    dev = {s: files[s].gate["dev_gated_s"] for s in stems if s != medoid and files[s].gate}
    if len(dev) < RT_RESIDUAL_MIN_FILES:
        return []
    cut = max(RT_RESIDUAL_RATIO * float(np.median(list(dev.values()))), 2 * 60 * dt_med)
    out = [s for s, d in dev.items() if np.isfinite(d) and d >= cut]
    for s in out:
        if "rt_residual_high" not in files[s].flags:
            files[s].flags.append("rt_residual_high")
    return out


def align_run(features: dict, scan_rt: dict, P: AlignParams | None = None) -> AlignResult:
    """features: stem -> per-file feature table (picker columns); scan_rt: stem -> MS1 grid (min)."""
    P = P or AlignParams()
    stems = list(features)
    N = len(stems)
    files = {}
    dts = []
    for s in stems:
        g = np.asarray(scan_rt[s], dtype=np.float64)
        t0, t_end, dt, span = _file_grid_info(g)
        dts.append(dt)
        fa = FileAlignment(stem=s, warp=TableWarp.identity(g), dt=dt, span=span)
        W_coarse = float(np.clip(0.1 * span, 0.3, 4.0))
        idx, tier, info = select_anchors(features[s], t0, t_end, P.ppm, W_coarse)
        fa.anchor_idx, fa.tier, fa.n_anchors = idx, tier, len(idx)
        fa.anchor_w = info.get("weights", np.ones(len(idx)))
        if info["anchor_poor"]:
            fa.flags.append("anchor_poor")
        files[s] = fa
    dt_med = float(np.median(dts))
    if N == 1:
        fa = files[stems[0]]
        fa.role = "single"
        fa.diag = FitDiag(flag="identity:single_file", n=0)
        return AlignResult(files, None, True, P.ppm, 2 * dt_med, dt_med, float("nan"), 0,
                           notes=["alignment skipped (single file)"])

    # coarse m/z offsets first (against the anchor-richest file), so that no step below sees them
    rich = sorted(stems, key=lambda s: -files[s].n_anchors)[:P.n_medoid_candidates]
    _coarse_mz(files, features, stems, rich[0], P.ppm, P.mz_dynamic, P.mz_relaxed_second_mode, P.mz_step_curve,
               P.mz_fallback_ref, P.mz_probe_features)
    # medoid: among the anchor-richest files, the one whose coarse match support to the others is largest
    best, best_score = None, -1.0
    for c in rich:
        fc = files[c]
        if fc.n_anchors < 15:
            continue
        a = fc.anchor_idx
        mz_c, rt_c = _mz(fc, features[c], a), features[c]["rt"].values[a]
        sup = []
        for s in stems:
            if s == c:
                continue
            fs = files[s]
            if fs.n_anchors == 0:
                sup.append(0)
                continue
            b = fs.anchor_idx
            _, support, _ = coarse_shift(_mz(fs, features[s], b), features[s]["rt"].values[b], mz_c, rt_c,
                                         P.ppm, float(np.clip(0.1 * fs.span, 0.3, 4.0)), max(2 * fs.dt, 0.5 / 60))
            sup.append(support)
        score = float(np.median(sup)) if sup else 0.0
        if score > best_score:
            best, best_score = c, score
    notes = []
    if best is None:   # no file has enough anchors: everything stays on its native axis and m/z
        for fa in files.values():
            fa.diag = FitDiag(flag="identity:no_anchors", n=0)
            fa.mz_offset_ppm = fa.mz_coarse_ppm = 0.0
            fa.mz_coarse_curve = fa.mz_curve = None
        notes.append("no file had >= 15 anchors; alignment skipped")
        return AlignResult(files, None, True, P.ppm, 2 * dt_med, dt_med, float("nan"), 0, notes)
    medoid = best
    files[medoid].role = "medoid"
    files[medoid].diag = FitDiag(flag="medoid", n=files[medoid].n_anchors)
    a = files[medoid].anchor_idx
    tgt_mz = _mz(files[medoid], features[medoid], a)
    tgt_rt = features[medoid]["rt"].values[a]

    n_done = 0
    prev = {s: files[s].warp.corr.copy() for s in stems}
    for it in range(P.n_iter):
        for s in stems:
            if s == medoid:
                continue
            _fit_file(files[s], features[s], scan_rt[s], tgt_mz, tgt_rt, P)
        n_done = it + 1
        # consensus target from all anchors on the corrected axis
        parts = []
        for k, s in enumerate(stems):
            fa = files[s]
            f = features[s]
            idx = fa.anchor_idx
            rt_n = f["rt"].values[idx]
            parts.append(pd.DataFrame({"mz": _mz(fa, f, idx), "rt": fa.warp.forward(rt_n),
                                       "rt_min": fa.warp.forward(f["rt_min"].values[idx]),
                                       "rt_max": fa.warp.forward(f["rt_max"].values[idx]),
                                       "height": f["height"].values[idx], "file": k}))
        A = pd.concat(parts, ignore_index=True)
        res = np.concatenate([files[s].diag.residual[files[s].diag.inlier]
                              for s in stems if files[s].diag.residual is not None and files[s].diag.inlier is not None]
                             ) if any(files[s].diag.residual is not None for s in stems) else np.zeros(0)
        pooled = 1.4826 * mad(res) if len(res) else dt_med
        rt_tol_iter = max(3 * pooled, 2 * dt_med)
        # anchors are strong, well-defined peaks: group them on apex RT alone (zero width disables
        # the overlap rule used for ordinary features) so the consensus stays sharp
        gid, cons = group_features(A["mz"].values, A["rt"].values, A["rt"].values, A["rt"].values,
                                   A["height"].values, A["file"].values, N, P.ppm, rt_tol_iter)
        need = min(N, max(CONSENSUS_MIN_FILES, int(np.ceil(0.2 * N))))
        keep = cons["n_detected"] >= need
        if keep.sum() >= 10:
            tgt_mz, tgt_rt = cons["mz"][keep], cons["rt"][keep]
        # convergence
        moves = [float(np.median(np.abs(files[s].warp.corr - prev[s]))) for s in stems if s != medoid]
        prev = {s: files[s].warp.corr.copy() for s in stems}
        if moves and max(moves) < 0.1 * dt_med and it > 0:
            break

    res = np.concatenate([files[s].diag.residual[files[s].diag.inlier] for s in stems
                          if files[s].diag.residual is not None and files[s].diag.inlier is not None]) \
        if any(files[s].diag.residual is not None for s in stems) else np.zeros(0)
    pooled = 1.4826 * mad(res) if len(res) else dt_med
    if P.gate:
        _do_no_harm(files, features, stems, medoid, P, dt_med)
    rt_tol = float(np.clip(3 * pooled, 2 * dt_med, P.rt_tol_max))
    # medoid m/z offset against the final consensus (its anchors are the identity in RT)
    a = files[medoid].anchor_idx
    mm = _mz(files[medoid], features[medoid], a)
    i, j = match_anchors(mm, features[medoid]["rt"].values[a], tgt_mz, tgt_rt, P.ppm, 0.0,
                         max(0.1, 6 * files[medoid].dt))
    if len(i) >= 5:
        r = (mm[i] - tgt_mz[j]) / tgt_mz[j] * 1e6
        files[medoid].mz_offset_ppm = files[medoid].mz_coarse_ppm + float(np.median(r))
        if P.mz_dynamic:
            fm = files[medoid]
            coarse = fm.mz_coarse_curve if fm.mz_coarse_curve is not None else fm.mz_coarse_ppm
            curve = _mz_curve(features[medoid]["rt"].values[a][i], r, fm.warp.native)
            fm.mz_curve = np.zeros(len(fm.warp.native)) + coarse + (curve if curve is not None else float(np.median(r)))
            fm.mz_offset_ppm = float(np.median(fm.mz_curve))
    # a file that matched far fewer anchors than the typical one is aligned (in RT and m/z) on little evidence
    typical = float(np.median([files[s].diag.n for s in stems if s != medoid]))
    for s in stems:
        if s != medoid and typical > 0:
            files[s].match_ratio = files[s].diag.n / typical
            if files[s].match_ratio < ANCHOR_MATCH_MIN:
                files[s].flags.append("few_anchors_matched")
    high = flag_rt_residual(files, stems, medoid, dt_med) if P.gate else []
    if high:
        notes.append(f"rt_residual_high: {len(high)} file(s) whose features stay far from the other files after RT "
                     f"correction: {', '.join(high[:5])}{' ...' if len(high) > 5 else ''}")
    sd = np.concatenate([features[s]["mz_sd_ppm"].values[files[s].anchor_idx] for s in stems])
    ppm_tol = float(np.clip(max(P.ppm, 3 * np.median(sd)) if len(sd) else P.ppm, 5.0, 15.0))
    if P.holdout:
        for s in stems:
            if s != medoid:
                files[s].holdout = _holdout(files[s], scan_rt[s])
    return AlignResult(files, medoid, False, ppm_tol, rt_tol, dt_med, 60 * pooled, n_done, notes)


def summary_table(res: AlignResult) -> pd.DataFrame:
    rows = []
    for s, fa in res.files.items():
        sh = fa.warp.shift(fa.warp.native)
        rows.append(dict(file=s, role=fa.role, anchor_tier=fa.tier, n_anchors=fa.n_anchors, n_matched=fa.diag.n,
                         match_vs_median=fa.match_ratio,
                         n_inlier=fa.diag.n_inlier, K_knots=fa.diag.K, coarse_shift_s=60 * fa.coarse_shift,
                         max_abs_shift_s=60 * float(np.max(np.abs(sh))) if len(sh) else 0.0,
                         mad_before_s=fa.diag.mad_before_s, mad_after_s=fa.diag.mad_after_s,
                         mz_offset_ppm=fa.mz_offset_ppm, mz_coarse_ppm=fa.mz_coarse_ppm,
                         gate_n_val=fa.gate["n_val"] if fa.gate else 0,
                         gate_lambda_min=float(fa.gate["lam"].min()) if fa.gate else np.nan,
                         gate_lambda_med=float(np.median(fa.gate["lam"])) if fa.gate else np.nan,
                         gate_dev_native_s=fa.gate["dev_native_s"] if fa.gate else np.nan,
                         gate_dev_fit_s=fa.gate["dev_full_s"] if fa.gate else np.nan,
                         gate_dev_applied_s=fa.gate["dev_gated_s"] if fa.gate else np.nan, flag=fa.flag))
    return pd.DataFrame(rows)


def anchors_table(res: AlignResult) -> pd.DataFrame:
    parts = []
    for s, fa in res.files.items():
        if fa.matched is not None and len(fa.matched):
            parts.append(fa.matched.assign(file=s))
    cols = ["file", "feature_id", "mz", "rt_native", "rt_corr", "consensus_rt", "residual_before_s",
            "residual_after_s", "weight", "is_inlier", "cluster_w"]
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)[cols]
