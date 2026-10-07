"""Anchor ("credible feature") selection per file for RT alignment."""
from __future__ import annotations

import numpy as np
import pandas as pd

N_BINS = 20
N_MAX = 400
MIN_USABLE = 15

TIERS = [  # (score, snr, n_scans, isolation factor, allow isotopes)
    (0.7, 10.0, 5, 2.0, False),
    (0.5, 5.0, 5, 2.0, False),
    (0.5, 5.0, 4, 1.0, False),
    (0.5, 5.0, 4, 1.0, True),
]


def _isolated(mz, rt, cand_idx, tol_ppm, W):
    """True for candidates with no other feature within +/- tol_ppm in m/z and +/- W in RT"""
    o = np.argsort(mz, kind="stable")
    smz, srt = mz[o], rt[o]
    cand_idx = np.asarray(cand_idx, dtype=np.int64)
    cm, crt = mz[cand_idx], rt[cand_idx]
    lo = np.searchsorted(smz, cm * (1 - tol_ppm * 1e-6))
    hi = np.searchsorted(smz, cm * (1 + tol_ppm * 1e-6), side="right")
    cnt = hi - lo
    owner = np.repeat(np.arange(len(cand_idx)), cnt)
    j = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(lo, cnt)
    near = np.abs(srt[j] - crt[owner]) <= W
    return np.bincount(owner[near], minlength=len(cand_idx)) <= 1   # itself


def select_anchors(feats: pd.DataFrame, t0: float, t_end: float, ppm_tol: float, W_coarse: float,
                   n_max: int = N_MAX):
    """Returns (anchor index array into feats, tier used, info dict)."""
    span = t_end - t0
    n_min = int(max(30, 6 * span))
    n = len(feats)
    info = dict(n_min=n_min, n_candidates=0, tier=-1, anchor_poor=False)
    if n == 0:
        info["anchor_poor"] = True
        return np.zeros(0, np.int64), -1, info
    mz = feats["mz"].values
    rt = feats["rt"].values
    height = feats["height"].values
    top = height >= np.quantile(height, 0.9) if n >= 10 else np.ones(n, bool)
    sd_ref = float(np.median(feats["mz_sd_ppm"].values[top])) if top.any() else 1.0
    base_w = float(np.median((feats["rt_max"] - feats["rt_min"]).values[top])) if top.any() else 0.0
    E = max(2 * base_w, 0.02 * span)
    has_m1 = np.zeros(n, bool)
    if "iso_parent" in feats and "iso_offset" in feats:
        par = feats["iso_parent"].values[(feats["iso_offset"].values == 1) & (feats["iso_parent"].values >= 0)]
        id_to_row = pd.Series(np.arange(n), index=feats["feature_id"].values)
        rows = id_to_row.reindex(par).dropna().astype(int).values
        has_m1[rows] = True
    chosen, tier_used = np.zeros(0, np.int64), -1
    for t, (s_min, snr_min, ns_min, iso_f, allow_iso) in enumerate(TIERS):
        ok = (feats["score"].values >= s_min) & (feats["snr"].values >= snr_min) & (feats["n_scans"].values >= ns_min)
        ok &= feats["mz_sd_ppm"].values <= 1.5 * max(sd_ref, 0.1)
        ok &= (rt >= t0 + E) & (rt <= t_end - E)
        if not allow_iso:
            ok &= feats["iso_offset"].values == 0
        cand = np.flatnonzero(ok)
        if len(cand):
            cand = cand[_isolated(mz, rt, cand, iso_f * ppm_tol, W_coarse)]
        info["n_candidates"] = int(len(cand))
        if len(cand) >= n_min or t == len(TIERS) - 1:
            tier_used = t
            # stratified pick: N_BINS RT bins, rank by (has M+1 partner, height)
            per_bin = int(np.ceil(n_max / N_BINS))
            bins = np.clip(((rt[cand] - t0) / max(span, 1e-9) * N_BINS).astype(int), 0, N_BINS - 1)
            picks = []
            for b in range(N_BINS):
                c = cand[bins == b]
                if len(c) == 0:
                    continue
                o = np.lexsort((-height[c], ~has_m1[c]))   # primary: has_m1 True first; then height desc
                picks.append(c[o][:per_bin])
            chosen = np.sort(np.concatenate(picks)) if picks else np.zeros(0, np.int64)
            break
    info["tier"] = tier_used
    info["n_anchors"] = int(len(chosen))
    info["anchor_poor"] = len(chosen) < MIN_USABLE
    info["weights"] = cluster_weights(rt[chosen], (feats["rt_max"] - feats["rt_min"]).values[chosen])
    info["n_clusters"] = int(round(info["weights"].sum()))
    return chosen, tier_used, info


def cluster_weights(rt, width):
    """Co-eluting anchors of one file are isotopologues, adducts or charge states of one compound
    (or compounds that share the same drift anyway): one piece of evidence for the warp, not
    several. Anchors are grouped greedily along RT: a cluster starts at its first anchor and
    takes the following ones while they lie within a quarter of the typical base width (about
    half a peak width) of that start, so clusters never chain. Each anchor in a cluster of k
    gets weight 1/k."""
    n = len(rt)
    if n == 0:
        return np.zeros(0)
    tol = 0.25 * float(np.median(width)) if n else 0.0
    tol = max(tol, 1e-6)
    o = np.argsort(rt, kind="stable")
    r = rt[o]
    cid = np.zeros(n, np.int64)
    start = r[0]
    for i in range(1, n):
        if r[i] - start <= tol:
            cid[i] = cid[i - 1]
        else:
            cid[i] = cid[i - 1] + 1
            start = r[i]
    size = np.bincount(cid)[cid]
    out = np.empty(n)
    out[o] = 1.0 / size
    return out
