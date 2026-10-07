"""Study-level RT alignment: medoid reference whose native axis is the output axis, refined by
consensus iterations over the union of every file's anchors, then a do-no-harm check of every warp
on held-out (non-anchor) features."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .anchors import _isolated, select_anchors
from .grouping import group_features
from .warp import FitDiag, TableWarp, coarse_shift, fit_warp, mad, match_anchors, weighted_median


CONSENSUS_MIN_FILES = 3    # a consensus anchor group must be present in at least this many files
GATE_MIN_SEG = 25          # validation features per do-no-harm segment, at least
GATE_MAX_SEG = 12          # do-no-harm segments per file, at most


@dataclass
class AlignParams:
    ppm: float = 5.0
    rt_tol_max: float = 0.25
    n_iter: int = 3
    n_medoid_candidates: int = 25
    holdout: bool = False
    gate: bool = True          # do-no-harm: scale each warp down where held-out features do not support it


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
    mz_offset_ppm: float = 0.0     # file m/z minus consensus m/z, median over inlier anchors
    anchor_w: np.ndarray = field(default_factory=lambda: np.zeros(0))   # 1/cluster size per anchor
    gate: dict | None = None       # do-no-harm step: validation count, segment nodes and factors, |dev| before/after
    warp_fit: TableWarp | None = None   # the anchor fit before the do-no-harm step (None = applied as fitted)

    @property
    def flag(self):
        base = "medoid" if self.role == "medoid" else self.diag.flag
        extra = [f for f in self.flags]
        return ";".join([base] + extra)


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


def _fit_file(fa: FileAlignment, feats: pd.DataFrame, grid, tgt_mz, tgt_rt, P: AlignParams):
    """coarse shift, mutual-unique matching and warp fit of one file against the target"""
    t0, t_end, dt, span = _file_grid_info(grid)
    a = fa.anchor_idx
    mz_f, rt_f = feats["mz"].values[a], feats["rt"].values[a]
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
    m = pd.DataFrame({"feature_id": feats["feature_id"].values[a][i], "mz": mz_f[i], "rt_native": x,
                      "rt_corr": warp.forward(x), "consensus_rt": y, "cluster_w": w0 if w0 is not None else 1.0})
    m["residual_before_s"] = 60 * (y - x)
    m["residual_after_s"] = 60 * (y - m["rt_corr"].values)
    if diag.weight is not None and len(diag.weight) == len(m):
        m["weight"] = diag.weight
        m["is_inlier"] = diag.inlier
        inl = diag.inlier
        if inl.sum() >= 5:
            fa.mz_offset_ppm = float(np.median((mz_f[i][inl] - tgt_mz[j][inl]) / tgt_mz[j][inl] * 1e6))
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
        parts.append(pd.DataFrame({"mz": f["mz"].values[cand], "x": x, "u": files[s].warp.forward(x),
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

    # medoid: among the anchor-richest files, the one whose coarse match support to the others is largest
    rich = sorted(stems, key=lambda s: -files[s].n_anchors)[:P.n_medoid_candidates]
    best, best_score = None, -1.0
    for c in rich:
        fc = files[c]
        if fc.n_anchors < 15:
            continue
        a = fc.anchor_idx
        mz_c, rt_c = features[c]["mz"].values[a], features[c]["rt"].values[a]
        sup = []
        for s in stems:
            if s == c:
                continue
            fs = files[s]
            if fs.n_anchors == 0:
                sup.append(0)
                continue
            b = fs.anchor_idx
            _, support, _ = coarse_shift(features[s]["mz"].values[b], features[s]["rt"].values[b], mz_c, rt_c,
                                         P.ppm, float(np.clip(0.1 * fs.span, 0.3, 4.0)), max(2 * fs.dt, 0.5 / 60))
            sup.append(support)
        score = float(np.median(sup)) if sup else 0.0
        if score > best_score:
            best, best_score = c, score
    notes = []
    if best is None:   # no file has enough anchors: everything stays on its native axis
        for fa in files.values():
            fa.diag = FitDiag(flag="identity:no_anchors", n=0)
        notes.append("no file had >= 15 anchors; alignment skipped")
        return AlignResult(files, None, True, P.ppm, 2 * dt_med, dt_med, float("nan"), 0, notes)
    medoid = best
    files[medoid].role = "medoid"
    files[medoid].diag = FitDiag(flag="medoid", n=files[medoid].n_anchors)
    a = files[medoid].anchor_idx
    tgt_mz = features[medoid]["mz"].values[a]
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
            parts.append(pd.DataFrame({"mz": f["mz"].values[idx], "rt": fa.warp.forward(rt_n),
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
    i, j = match_anchors(features[medoid]["mz"].values[a], features[medoid]["rt"].values[a], tgt_mz, tgt_rt, P.ppm,
                         0.0, max(0.1, 6 * files[medoid].dt))
    if len(i) >= 5:
        mm = features[medoid]["mz"].values[a][i]
        files[medoid].mz_offset_ppm = float(np.median((mm - tgt_mz[j]) / tgt_mz[j] * 1e6))
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
                         n_inlier=fa.diag.n_inlier, K_knots=fa.diag.K, coarse_shift_s=60 * fa.coarse_shift,
                         max_abs_shift_s=60 * float(np.max(np.abs(sh))) if len(sh) else 0.0,
                         mad_before_s=fa.diag.mad_before_s, mad_after_s=fa.diag.mad_after_s,
                         mz_offset_ppm=fa.mz_offset_ppm,
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
