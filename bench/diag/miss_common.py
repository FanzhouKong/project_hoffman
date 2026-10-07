# shared helpers for the truth-miss attributions (yeast_truth_misses.py, idsl_tp_misses.py): re-pick a file in
# memory with the rejected candidates kept, name the gate that rejected a candidate, raw-EIC descriptors at a truth
# position judged with the picker's own cell noise, and EIC galleries.
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from peak3d import kernels as K  # noqa: E402
from peak3d.estimate import estimate  # noqa: E402
from peak3d.features import BACKGROUND_MAX, RIDGE_MAX, build_features  # noqa: E402
from peak3d.io import load_cloud  # noqa: E402
from peak3d.pick import pick_cloud  # noqa: E402

STAGE_RANK = {"gate": 0, "eic:noapex": 1, "eic:lowsnr": 2, "eic:dup": 3, "score": 4}


def repick(path):
    """cloud, params, basins, kept features, rejected candidates (same code as the benchmark, nothing written)"""
    c = load_cloud(path)
    P = estimate(c)
    b = pick_cloud(c, P)
    fe, rj = build_features(c, b, None, True)
    rj = rj.copy()
    rj["stage"] = rj["reason"].astype(str)
    return c, P, b, fe, rj


def noise_lookup(cloud, P, b):
    """the picker's cell noise at (mz, rt): the post-merge 3D noise surface"""
    S = b.noise_surface
    mz_lo, mz_hi = float(cloud.mz.min()), float(cloud.mz.max()) * (1 + 1e-9)
    lw = (np.log(mz_hi) - np.log(mz_lo)) / P.n_mz_bands + 1e-12

    def look(mz, rt):
        band = int(np.clip((np.log(mz) - np.log(mz_lo)) / lw, 0, S.shape[0] - 1))
        blk = int(np.clip((rt - cloud.rt[0]) / P.rt_block_min, 0, S.shape[1] - 1))
        return float(S[band, blk])
    return look


def gate_reasons(r, P):
    """why a rejected candidate row failed (hard gates, chromatogram check or score)"""
    out = []
    stage = str(r.get("stage", r.get("reason", "")))
    if stage == "gate":
        if r["n_scans"] < P.min_scans:
            out.append(f"n_scans {int(r['n_scans'])} < {P.min_scans}")
        if r["snr"] < P.min_snr:
            out.append(f"snr {r['snr']:.1f} < {P.min_snr:g}")
        if r["mz_sd_ppm"] > 3.0 * r["sig_apex"] + 1.0:
            out.append(f"mz_sd {r['mz_sd_ppm']:.1f} ppm > {3 * r['sig_apex'] + 1:.1f}")
        if r["fwhm"] < 0.3 * P.fwhm_med:
            out.append(f"fwhm {60 * r['fwhm']:.2f} s < 0.3 median")
        if r["prominence_rel"] < P.min_prominence_rel:
            out.append(f"prominence {r['prominence_rel']:.2f} < {P.min_prominence_rel:g}")
        if bool(r.get("fragment", False)):
            out.append("fragment")
        kf = int(r["kflags"])
        if (kf & K.KF_CAP_L) and (kf & K.KF_CAP_R) and r["prominence_rel"] < 0.5:
            out.append("capped both sides, prominence < 0.5")
        if r.get("ridge_ratio", 0) > RIDGE_MAX:
            out.append(f"ridge {r['ridge_ratio']:.2f} > {RIDGE_MAX:g}")
        if r.get("background_ratio", 0) > BACKGROUND_MAX:
            out.append(f"background {r['background_ratio']:.2f} > {BACKGROUND_MAX:.2f}")
        if P.far_level_max < 1.0 and r.get("far_level", 0) > P.far_level_max:
            out.append(f"far_level {r['far_level']:.2f} > {P.far_level_max:g}")
        if not out:
            out.append("gate (unlisted)")
    elif stage.startswith("eic:"):
        out.append(f"{stage} (eic_snr {r.get('eic_snr', np.nan):.1f}, prom {r.get('es_prom', np.nan):.3g}, "
                   f"trace noise {r.get('es_noise', np.nan):.3g}, cell noise {r['noise']:.3g})")
    elif stage == "score":
        out.append(f"score {r.get('score_rule', np.nan):.2f} < {P.min_score:g}")
    return out


def gate_short(r, P):
    """one token for the main reason (for counting)"""
    stage = str(r.get("stage", r.get("reason", "")))
    if stage != "gate":
        return stage
    if r["snr"] < P.min_snr:
        return "gate:snr"
    if r["n_scans"] < P.min_scans:
        return "gate:n_scans"
    if r.get("far_level", 0) > P.far_level_max:
        return "gate:far_level"
    if r.get("ridge_ratio", 0) > RIDGE_MAX or r.get("background_ratio", 0) > BACKGROUND_MAX:
        return "gate:ridge"
    if r["prominence_rel"] < P.min_prominence_rel:
        return "gate:prominence"
    if r["mz_sd_ppm"] > 3.0 * r["sig_apex"] + 1.0:
        return "gate:mz_sd"
    if r["fwhm"] < 0.3 * P.fwhm_med:
        return "gate:fwhm"
    if bool(r.get("fragment", False)):
        return "gate:fragment"
    return "gate:other"


def near(df, mz, rt, ppm, rt_win, rt_col="rt", mz_col="mz"):
    x = df[(np.abs(df[mz_col].values - mz) / mz * 1e6 <= ppm) & (np.abs(df[rt_col].values - rt) <= rt_win)].copy()
    x["dppm"] = (x[mz_col].values - mz) / mz * 1e6
    x["drt_s"] = 60 * (x[rt_col].values - rt)
    return x


def eic_desc(c, mz, ppm, rt_nat, half_min, tol_min, floor, cell_noise):
    """raw EIC at +/- ppm around rt_nat (native minutes): the best apex inside +/- tol_min and its evidence"""
    s0 = int(max(0, np.searchsorted(c.rt, rt_nat - half_min)))
    s1 = int(min(c.n_scans, np.searchsorted(c.rt, rt_nat + half_min)))
    e = c.eic(mz, ppm, s0, s1).astype(np.float64)
    rt = c.rt[s0:s1]
    d = dict(s0=s0, s1=s1, rt=rt, e=e, apex_h=0.0, drt_s=np.nan, local_max=False, n_contig=0, fwhm_s=np.nan,
             noise=float(max(floor, cell_noise)), cell_noise=float(cell_noise), level_out=0.0, snr=0.0,
             snr_cell=0.0, n_nonzero_win=0)
    if len(e) < 3:
        return d
    win = np.abs(rt - rt_nat) <= tol_min
    if not win.any():
        return d
    idx = np.flatnonzero(win)
    d["n_nonzero_win"] = int((e[idx] > 0).sum())
    if d["n_nonzero_win"] == 0:
        return d
    sm = np.convolve(e, [0.25, 0.5, 0.25], mode="same")
    lm = [i for i in idx if 0 < i < len(sm) - 1 and sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1] and e[i] > 0]
    if lm:
        i = max(lm, key=lambda k: e[k])
        d["local_max"] = True
    else:
        i = int(idx[np.argmax(e[idx])])
    d["apex_h"] = float(e[i])
    d["drt_s"] = float(60 * (rt[i] - rt_nat))
    lo = i
    while lo - 1 >= 0 and e[lo - 1] > 0:
        lo -= 1
    hi = i
    while hi + 1 < len(e) and e[hi + 1] > 0:
        hi += 1
    d["n_contig"] = int(hi - lo + 1)
    half = e[i] / 2
    a = i
    while a - 1 >= lo and e[a - 1] >= half:
        a -= 1
    bb = i
    while bb + 1 <= hi and e[bb + 1] >= half:
        bb += 1
    dt = float(np.median(np.diff(c.rt))) if c.n_scans > 1 else 0.0
    d["fwhm_s"] = float(60 * (rt[bb] - rt[a] + dt))
    out = np.abs(rt - rt[i]) > 3 * tol_min
    v = e[out]
    v = v[v > 0]
    if len(v) >= 5:
        mad = 1.4826 * np.median(np.abs(v - np.median(v)))
        d["level_out"] = float(np.median(v))
        d["noise"] = float(max(floor, cell_noise, mad))
    d["snr"] = float(e[i] / d["noise"])
    d["snr_cell"] = float(e[i] / max(cell_noise, 1e-9))
    return d


def to_native(w, rt_cons):
    """consensus -> native minutes through the file's monotone warp table"""
    return float(np.interp(rt_cons, w["rt_corr"].values, w["rt_native"].values))


def plot_gallery(items, path, title, per_row=None):
    """items: list of dicts with 'title' and 'panels' (list of dicts: rt, e, e_wide, rt_truth, tol, kept, rejected,
    label). One row per item, one panel per file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not items:
        return
    ncol = per_row or max(len(it["panels"]) for it in items)
    fig, axes = plt.subplots(len(items), ncol, figsize=(4.2 * ncol, 2.3 * len(items)), squeeze=False)
    for r, it in enumerate(items):
        for cidx in range(ncol):
            ax = axes[r, cidx]
            if cidx >= len(it["panels"]):
                ax.axis("off")
                continue
            p = it["panels"][cidx]
            rt, e = p["rt"], p["e"]
            if p.get("e_wide") is not None:
                ax.plot(rt, p["e_wide"], color="0.75", lw=0.8, label="wide ppm")
            ax.plot(rt, e, color="k", lw=0.9, drawstyle="steps-mid")
            ax.axvspan(p["rt_truth"] - p["tol"], p["rt_truth"] + p["tol"], color="gold", alpha=0.25)
            ax.axvline(p["rt_truth"], color="orange", lw=0.8)
            ymax = max(float(np.max(e)) if len(e) else 1.0, 1.0)
            for k in p.get("kept", []):
                ax.axvspan(k["rt_min"], k["rt_max"], color="green", alpha=0.18)
                ax.text(k["rt_min"], ymax * 0.95, k.get("label", "kept"), color="green", fontsize=6, va="top")
            for k in p.get("rejected", []):
                ax.axvspan(k["rt_min"], k["rt_max"], color="red", alpha=0.12)
                ax.text(k["rt_min"], ymax * 0.75, k.get("label", "rej"), color="red", fontsize=6, va="top")
            if p.get("noise"):
                ax.axhline(p["noise"], color="blue", lw=0.6, ls=":")
                ax.axhline(3 * p["noise"], color="blue", lw=0.6, ls="--")
            ax.set_xlim(rt[0], rt[-1])
            ax.set_ylim(0, ymax * 1.08)
            ax.tick_params(labelsize=6)
            ax.set_title(p.get("label", ""), fontsize=7)
        axes[r, 0].set_ylabel(it["title"], fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
