"""Basins -> per-file feature table: descriptors, hard gates, flags, isotopes, rule score."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import kernels as K
from .estimate import Params
from .io import Cloud
from .isotopes import annotate_isotopes
from .pick import Basins

COLUMNS = ["feature_id", "mz", "rt", "rt_apex", "height", "area", "area_bsub", "rt_min", "rt_max",
           "scan_apex", "scan_min", "scan_max", "n_scans", "n_points", "n_gaps", "fwhm", "asym",
           "mz_sd_ppm", "mz_min", "mz_max", "baseline", "baseline_lo", "noise", "snr", "prominence_rel", "persistence",
           "persistence_rel", "gauss_r2", "ridge_ratio", "background_ratio", "ripple_ratio", "far_level", "tv_ratio",
           "n_valleys", "tv_excess", "noise_i", "eic_snr", "iso_offset", "charge", "iso_parent",
           "iso_ratio",
           "score_rule", "score_model", "score", "flags"]
INT_COLS = ["feature_id", "scan_apex", "scan_min", "scan_max", "n_scans", "n_points", "n_gaps", "n_valleys",
            "iso_offset", "charge", "iso_parent"]


def basins_to_frame(b: Basins) -> pd.DataFrame:
    P = b.params
    s = b.stats
    df = pd.DataFrame({name: s[:, i] for i, name in enumerate(K.COLS)})
    for c in ["apex_idx", "scan_apex", "scan_min", "scan_max", "scan_lo", "scan_hi", "n_points", "n_gaps",
              "n_flat", "kflags", "n_valleys"]:
        df[c] = df[c].astype(np.int64)
    # n_scans inside the bounds (present scans only)
    df["n_scans"] = (df["scan_hi"] - df["scan_lo"] + 1 - df["n_gaps"]).astype(np.int64)
    df["basin_id"] = b.basin_id
    df["persistence"] = b.persistence
    df["noise"] = np.maximum(b.noise, np.finfo(np.float32).tiny)
    # S/N above the lowest level the trace reaches at its bounds (0 where it drops out of the
    # data); the level where the bounds walk stopped is a cut at 2x noise, not a baseline
    base = df["baseline_lo"] if P.snr_floor_baseline else df["baseline"]
    df["snr"] = (df["height"] - base) / df["noise"]
    df["prominence_rel"] = (df["height"] - df["baseline"]) / df["height"]
    df["persistence_rel"] = df["persistence"] / df["height"]
    df["sig_apex"] = P.sig_ppm(df["height"].values)
    # expected scan-to-scan noise of the ion current at the apex level (cell floor + intensity-dependent part)
    h = df["height"].values
    df["noise_i"] = np.sqrt(df["noise"].values ** 2 + P.noise_c * h + (P.noise_r * h) ** 2)
    return df


def mark_fragments(df: pd.DataFrame, cloud: Cloud, P: Params) -> pd.Series:
    """A candidate whose maximum sits on one of its own bounds is a fragment only if the scan
    just beyond that bound holds a HIGHER centroid of the same ion (the trace keeps rising into
    a neighbouring basin: a tail cut off by a tolerance break). A maximum at a bound next to a
    lower or absent neighbour is a genuine, if truncated, peak. Run edges are never fragments."""
    at_lo = (df["scan_apex"] == df["scan_lo"]).values
    at_hi = (df["scan_apex"] == df["scan_hi"]).values
    run_edge = ((df["kflags"].values & (K.KF_RUN_START | K.KF_RUN_END)) > 0)
    cand = np.flatnonzero((at_lo | at_hi) & ~run_edge)
    frag = np.zeros(len(df), bool)
    if len(cand):
        side = np.where(at_lo[cand], df["scan_lo"].values[cand] - 1, df["scan_hi"].values[cand] + 1).astype(np.int64)
        q_mz = df["mz"].values[cand].astype(np.float64)
        q_tol = q_mz * P.tol_max_ppm * 1e-6
        nb = K.max_in_scan(cloud.off, cloud.mz, cloud.inten, q_mz, q_tol, side)
        frag[cand] = nb > df["height"].values[cand]
    return pd.Series(frag, index=df.index)


RIDGE_MAX = 0.5          # side level above this fraction of the apex on BOTH sides = background ion
RIDGE_NEAR, RIDGE_FAR = 2.0, 6.0   # side windows, in peak widths from the apex
RIDGE_Q = 0.90           # 90th-percentile side level: an ion present in >= 10 % of the far scans at half the
                         # apex is background (measured: costs 1 of 420 certified HZV029 features, 0 of 812 LI2018)
RIDGE_MIN_PPM = 5.0      # background ions' centroids wander more than their intensity suggests
BACKGROUND_Q = 0.25      # lower-quartile level = the ion's continuous background
BACKGROUND_MAX = 1 / 3   # signal-to-background >= 3: apex at least 3x the ion's own background


def ridge_ratio(df: pd.DataFrame, cloud: Cloud, P: Params) -> dict:
    """A chromatographic peak returns to baseline within a few widths on both sides; a background
    ion does not. Level = 90th-percentile per-scan max of the ion's chromatogram in the windows
    [apex - 6w, apex - 2w] and [apex + 2w, apex + 6w], w = max(own FWHM, file median FWHM) so a
    genuinely broad peak is judged far enough out; a high quantile (not the median) also sees ions
    that flicker in and out of the centroider's threshold. Returns min(left, right) / apex: the
    lower side, so a small peak beside a big one (clean far side) is not a ridge. Returns both
    the 90th-percentile ratio (``ridge_ratio``: the ion keeps showing up out there) and the
    lower-quartile ratio (``background_ratio``: the ion's continuous background; a peak must
    rise at least 3x above it)."""
    n = len(df)
    if n == 0:
        return {"ridge_ratio": np.zeros(0), "background_ratio": np.zeros(0), "ripple_ratio": np.zeros(0),
                "far_level": np.zeros(0)}
    # window scale: the feature's own width, floored at the file's median width and capped at ridge_w_cap
    # median widths, so a ridge segment many widths long is judged right beside itself
    w = np.minimum(np.maximum(df["fwhm"].values, P.fwhm_med), P.ridge_w_cap * P.fwhm_med) / P.dt   # scans
    sa = df["scan_apex"].values.astype(np.int64)
    q_mz = df["mz"].values.astype(np.float64)
    q_tol = q_mz * np.maximum(3.0 * df["sig_apex"].values + 1.0, RIDGE_MIN_PPM) * 1e-6
    near = np.maximum(1, np.round(RIDGE_NEAR * w)).astype(np.int64)
    far = np.maximum(near + 2, np.round(RIDGE_FAR * w)).astype(np.int64)
    L = K.side_levels(cloud.off, cloud.mz, cloud.inten, q_mz, q_tol, sa, near, far)
    return ratios_from_levels(L, df["height"].values)


def ratios_from_levels(L: np.ndarray, height: np.ndarray) -> dict:
    """ridge / background / ripple ratios and far_level from side levels L (float64[n, 2, 6], SL_*)"""
    n = len(height)
    h = np.maximum(height, 1e-9)
    usable = L[:, :, K.SL_N] >= 3
    any_side = usable.any(axis=1)

    def lower_side(col):
        # the cleaner of the usable sides; a side cut to < 3 scans by the run edge is not a clean side
        v = np.where(usable, L[:, :, col], np.inf)
        return np.where(any_side, np.min(v, axis=1), 0.0) / h
    # ripple: on the cleaner side (lower median; a side cut by the run edge to < 3 scans does not
    # count), how far the apex rises above the ion's own level there, in units of that trace's
    # scan-to-scan ripple. A zig-zag on an ion that keeps flickering near the detection limit
    # rises only a few ripples; on an empty baseline the ripple is 0 and the ratio infinite.
    med = np.where(usable, L[:, :, K.SL_MED], np.inf)
    c = np.argmin(med, axis=1)
    rows = np.arange(n)
    lvl = np.where(usable.any(axis=1), L[rows, c, K.SL_MED], 0.0)
    rip = np.where(usable.any(axis=1), L[rows, c, K.SL_NOISE], 0.0)
    rise = h - lvl
    with np.errstate(divide="ignore", invalid="ignore"):
        ripple = np.where(rip > 0, rise / np.where(rip > 0, rip, 1.0), np.inf)
    # localisation: the ion's typical level where it is present, 2 to 6 widths out on the cleaner side
    mednz = np.where(usable, L[:, :, K.SL_MEDNZ], np.inf)
    far_level = np.where(usable.any(axis=1), np.min(mednz, axis=1), 0.0) / h
    return {"ridge_ratio": lower_side(K.SL_Q90), "background_ratio": lower_side(K.SL_Q25),
            "ripple_ratio": ripple, "far_level": far_level}


def eic_check(df: pd.DataFrame, cloud: Cloud, P: Params) -> pd.DataFrame:
    """1D chromatogram check of each candidate, independent of its basin (kernels.eic_shape): the
    smoothed EIC must have a local maximum inside the candidate's bounds (else the basin is a slice of
    a tail, ramp, step or a neighbour's flank: ``noapex``), that maximum's prominence must reach
    eic_k_snr x the larger of the trace's own residual noise and the cell noise (``lowsnr``), and a
    weaker same-ion candidate sitting on the same smoothed maximum is a ``dup``. Returns a frame with
    es_top, es_scan, es_prom, es_noise, eic_snr, eic_class ("ok" or the failure)."""
    n = len(df)
    cols = ["es_top", "es_scan", "es_prom", "es_noise", "eic_snr"]
    if n == 0:
        out = pd.DataFrame({c: np.zeros(0) for c in cols}, index=df.index)
        out["eic_class"] = np.zeros(0, dtype=object)
        return out
    q_mz = df["mz"].values.astype(np.float64)
    q_tol = q_mz * np.maximum(3.0 * df["sig_apex"].values + 1.0, RIDGE_MIN_PPM) * 1e-6
    kern = K.gaussian_kernel(P.eic_smooth_fwhm * P.fwhm_scans)
    half_win = int(max(3, round(P.eic_half_win_fwhm * P.fwhm_scans)))
    E = K.eic_shape(cloud.off, cloud.mz, cloud.inten, q_mz, q_tol, df["scan_apex"].values.astype(np.int64),
                    df["scan_lo"].values.astype(np.int64), df["scan_hi"].values.astype(np.int64), half_win, kern)
    noise = np.maximum(E[:, K.ES_NOISE], df["noise"].values)
    snr = E[:, K.ES_PROM] / np.maximum(noise, 1e-9)
    o = np.argsort(q_mz, kind="stable")
    dup_s = K.duplicate_mask(q_mz[o], E[o, K.ES_SCAN], df["height"].values[o].astype(np.float64), 5.0,
                             max(1.0, P.eic_dup_fwhm * P.fwhm_scans))
    dup = np.zeros(n, bool)
    dup[o] = dup_s
    cls = np.full(n, "ok", dtype=object)
    cls[snr < P.eic_k_snr] = "lowsnr"
    cls[dup] = "dup"
    cls[E[:, K.ES_APEX] <= 0] = "noapex"
    return pd.DataFrame({"es_top": E[:, K.ES_TOP], "es_scan": E[:, K.ES_SCAN].astype(np.int64), "es_prom": E[:, K.ES_PROM],
                         "es_noise": E[:, K.ES_NOISE], "eic_snr": snr, "eic_class": cls}, index=df.index)


def gate(df: pd.DataFrame, P: Params) -> pd.Series:
    ok = (df["n_scans"] >= P.min_scans)
    ok &= df["snr"] >= P.min_snr
    ok &= df["mz_sd_ppm"] <= 3.0 * df["sig_apex"] + 1.0
    ok &= df["fwhm"] >= 0.3 * P.fwhm_med
    ok &= df["prominence_rel"] >= P.min_prominence_rel
    if "fragment" in df:
        ok &= ~df["fragment"]
    # a trace still above half its apex six peak widths away on both sides is background, not a peak
    cap_both = (df["kflags"] & K.KF_CAP_L) > 0
    cap_both &= (df["kflags"] & K.KF_CAP_R) > 0
    ok &= ~(cap_both & (df["prominence_rel"] < 0.5))
    if "ridge_ratio" in df:
        ok &= df["ridge_ratio"] <= RIDGE_MAX
    if "background_ratio" in df:
        ok &= df["background_ratio"] <= BACKGROUND_MAX
    if "ripple_ratio" in df and P.ripple_k > 0:
        ok &= df["ripple_ratio"] >= P.ripple_k
    if P.max_valleys >= 0:
        ok &= df["n_valleys"] <= P.max_valleys
    if "far_level" in df and P.far_level_max < 1.0:
        ok &= df["far_level"] <= P.far_level_max
    return ok


def flag_strings(df: pd.DataFrame, P: Params) -> pd.Series:
    kf = df["kflags"].values
    cap = (kf & (K.KF_CAP_L | K.KF_CAP_R)) > 0
    conds = [
        ("edge", (kf & (K.KF_RUN_START | K.KF_RUN_END)) > 0),
        ("gap", df["n_gaps"].values > 0),
        ("wide", df["fwhm"].values > 5.0 * P.fwhm_med),
        ("ridge", (cap & (df["prominence_rel"].values < 0.5)) | (df["ridge_ratio"].values > RIDGE_MAX if "ridge_ratio" in df else False)
                  | (df["background_ratio"].values > BACKGROUND_MAX if "background_ratio" in df else False)),
        ("flat_top", df["n_flat"].values >= 3),
        ("lumpy", df["n_valleys"].values >= 2),
        ("cap", cap),
        ("nohalf", (kf & (K.KF_NOHALF_L | K.KF_NOHALF_R)) > 0),
    ]
    out = np.full(len(df), "", dtype=object)
    for name, m in conds:
        out = np.where(m, np.where(out == "", name, out + ";" + name), out)
    return pd.Series(out, index=df.index)


def rule_score(df: pd.DataFrame, P: Params, iso_support: np.ndarray | None = None) -> np.ndarray:
    snr = np.maximum(df["snr"].values, 1e-9)
    t_snr = np.clip((np.log2(snr) - np.log2(3.0)) / (np.log2(50.0) - np.log2(3.0)), 0, 1)
    t_shape = np.clip(df["gauss_r2"].values, 0, 1)
    t_mz = np.clip(1.0 - df["mz_sd_ppm"].values / (3.0 * df["sig_apex"].values + 1.0), 0, 1)
    t_prom = np.clip(df["prominence_rel"].values, 0, 1)
    t_scans = np.clip(df["n_scans"].values / P.fwhm_scans, 0, 1)
    if P.score_loc_rip and "far_level" in df and "tv_ratio" in df:
        # localisation: a peak returns to (near) nothing a few widths away, a background ion does not;
        # unimodality: one rise and one fall inside the bounds (total variation ratio 1), not a lumpy hump
        t_loc = np.clip(1.0 - df["far_level"].values, 0, 1)
        t_rip = np.clip(1.0 / np.maximum(df["tv_ratio"].values, 1e-9), 0, 1)
        score = (t_snr * t_shape * t_mz * t_prom * t_scans * t_loc * t_rip) ** (1.0 / 7.0)
    else:
        score = (t_snr * t_shape * t_mz * t_prom * t_scans) ** 0.2
    if iso_support is not None:
        score = np.where(iso_support, np.minimum(1.0, score + 0.15), score)
    return score


def build_features(cloud: Cloud, b: Basins, min_score: float | None = None, keep_rejected: bool = False):
    """Returns (features, rejected): rejected is None unless keep_rejected."""
    P = b.params
    thr = P.min_score if min_score is None else min_score
    df = basins_to_frame(b)
    df["fragment"] = mark_fragments(df, cloud, P)
    for name, vals in ridge_ratio(df, cloud, P).items():
        df[name] = vals
    ok = gate(df, P)
    df["gate"] = ok.values
    cand = df[ok].reset_index(drop=True)
    cand["eic_snr"] = np.inf
    eic_rej = None
    if P.eic_check:
        E = eic_check(cand, cloud, P)
        for c in E.columns:
            cand[c] = E[c].values
        ok2 = (E["eic_class"].values == "ok")
        eic_rej = cand[~ok2].assign(reason="eic:" + cand.loc[~ok2, "eic_class"].astype(str))
        cand = cand[ok2].reset_index(drop=True)
    iso_offset, charge, iso_parent, iso_ratio = annotate_isotopes(
        cand["mz"].values, cand["rt"].values, cand["height"].values, cand["scan_lo"].values,
        cand["scan_hi"].values, cloud, P)
    has_partner = np.zeros(len(cand), bool)
    has_partner[iso_parent[iso_parent >= 0]] = True
    iso_support = (iso_offset > 0) | has_partner
    cand["iso_offset"], cand["charge"], cand["iso_ratio"] = iso_offset, charge, iso_ratio
    cand["iso_parent_row"] = iso_parent
    cand["score_rule"] = rule_score(cand, P, iso_support)
    cand["score_model"] = np.nan
    cand["score"] = cand["score_rule"]
    cand["flags"] = flag_strings(cand, P)
    keep = cand["score"].values >= thr
    feats = cand[keep].reset_index(drop=True)
    # feature ids and parent ids in the kept table
    feats["feature_id"] = np.arange(len(feats), dtype=np.int64)
    row_to_id = np.full(len(cand), -1, np.int64)
    row_to_id[np.flatnonzero(keep)] = feats["feature_id"].values
    par = feats["iso_parent_row"].values
    feats["iso_parent"] = np.where(par >= 0, row_to_id[np.maximum(par, 0)], -1)
    feats["rt_min"] = feats["rt_min"]
    feats["scan_min"] = feats["scan_lo"]
    feats["scan_max"] = feats["scan_hi"]
    out = feats[COLUMNS].copy()
    for c in INT_COLS:
        out[c] = out[c].astype(np.int64)
    rejected = None
    if keep_rejected:
        parts = [df[~ok].assign(reason="gate")]
        if eic_rej is not None:
            parts.append(eic_rej)
        parts.append(cand[~keep].assign(reason="score"))
        rejected = pd.concat(parts, ignore_index=True)
    return out, rejected


MZ_COLUMNS = ("mz", "mz_min", "mz_max", "mz_corr")


def precise_mz(df: pd.DataFrame) -> pd.DataFrame:
    """m/z columns as fixed 6-decimal text (sub-ppb), so a %.6g float format elsewhere in the table does not
    round m/z to 0.001 Da (1 ppm at m/z 500, 5 ppm above 1,000)"""
    cols = [c for c in MZ_COLUMNS if c in df.columns]
    if not cols:
        return df
    out = df.copy()
    for c in cols:
        out[c] = np.char.mod("%.6f", out[c].values.astype(np.float64))
    return out


def write_features(df: pd.DataFrame, path) -> None:
    precise_mz(df).to_csv(path, sep="\t", index=False, float_format="%.6g", lineterminator="\n")
