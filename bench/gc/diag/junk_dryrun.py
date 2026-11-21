#!/usr/bin/env python3
"""Dry run of candidate junk rules on stored GC-mode outputs: every file is picked again from its point-cloud cache
with its own params.json (the run's settings), the gate table is captured, and per kept feature the measures the
rules would use are written beside the feature id:

  rip_scan   rise above the ion's level on the cleaner side / scan-to-scan ripple there (the current net rule)
  sig_peak   that side's spread at the peak scale: (median - lower quartile) / 0.6745 of the side window
  rip_peak   rise / max(ripple, sig_peak)
  net_w      FWHM above the ion's level (net_fwhm) / the file's median FWHM
  fill_*     unit-resolution runs: filled share of the peak's scans, measured climb toward the apex on each side
             (non-filled points only, in units of the feature's noise), apex next to a filled point

    python bench/gc/diag/junk_dryrun.py OUTPUT --files STEM ... --out DIR [--truth SCORE_DIR]
With --truth (an mrh_score.py folder): truth_rows.tsv marks which features are clean truth ions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import peak3d.features as FT  # noqa: E402
from peak3d import kernels as K  # noqa: E402
from peak3d.gc.deconv import params_from_dict  # noqa: E402
from peak3d.io import Cloud  # noqa: E402
from peak3d.pick import pick_cloud  # noqa: E402


def side_stats(df, cloud, P):
    dt = df["dt_local"].values if "dt_local" in df else P.dt
    w = np.minimum(np.maximum(df["fwhm"].values, P.fwhm_med), P.ridge_w_cap * P.fwhm_med) / dt
    q_mz = df["mz"].values.astype(np.float64)
    q_tol = q_mz * np.maximum(3.0 * df["sig_apex"].values + 1.0, FT.RIDGE_MIN_PPM) * 1e-6
    near = np.maximum(1, np.round(FT.RIDGE_NEAR * w)).astype(np.int64)
    far = np.maximum(near + 2, np.round(FT.RIDGE_FAR * w)).astype(np.int64)
    L = K.side_levels(cloud.off, cloud.mz, cloud.inten, q_mz, q_tol, df["scan_apex"].values.astype(np.int64), near, far)
    usable = L[:, :, K.SL_N] >= 3
    med = np.where(usable, L[:, :, K.SL_MED], np.inf)
    c = np.argmin(med, axis=1)
    rows = np.arange(len(df))
    anyu = usable.any(axis=1)
    lvl = np.where(anyu, L[rows, c, K.SL_MED], 0.0)
    rip = np.where(anyu, L[rows, c, K.SL_NOISE], 0.0)
    sig = np.where(anyu, (L[rows, c, K.SL_MED] - L[rows, c, K.SL_Q25]) / 0.6745, 0.0)
    rise = df["height"].values - lvl
    with np.errstate(divide="ignore", invalid="ignore"):
        rip_scan = np.where(rip > 0, rise / np.where(rip > 0, rip, 1.0), np.inf)
        den = np.maximum(rip, sig)
        rip_peak = np.where(den > 0, rise / np.where(den > 0, den, 1.0), np.inf)
    return pd.DataFrame({"rip_scan": rip_scan, "sig_peak": sig, "rip_peak": rip_peak, "rise": rise}, index=df.index)


def fill_stats(feats, cloud, scale):
    """per feature: filled share, measured climb toward the apex on each side (max measured - min measured before
    the apex, and after), apex scan next to a filled point"""
    n = len(feats)
    out = dict(fill_share=np.zeros(n), climb_l=np.zeros(n), climb_r=np.zeros(n), apex_by_fill=np.zeros(n, bool))
    if cloud.bridged is None:
        return pd.DataFrame(out, index=feats.index)
    nom = np.rint(cloud.mz / scale).astype(np.int64)
    sc = cloud.scan_index()
    for i, r in enumerate(feats.itertuples()):
        a, b = int(r.scan_min), int(r.scan_max) + 1
        p0, p1 = cloud.off[a], cloud.off[b]
        j = np.flatnonzero(nom[p0:p1] == int(round(r.mz / scale)))
        s = sc[p0:p1][j]
        y = cloud.inten[p0 + j].astype(np.float64)
        f = cloud.bridged[p0 + j]
        if not len(s):
            continue
        out["fill_share"][i] = f.mean()
        ap = int(r.scan_apex)
        meas_l, meas_r = (s <= ap) & ~f, (s >= ap) & ~f
        top = y[~f].max() if (~f).any() else 0.0
        top_l = y[meas_l].max() if meas_l.any() else 0.0
        top_r = y[meas_r].max() if meas_r.any() else 0.0
        out["climb_l"][i] = (top_l - y[meas_l].min()) if meas_l.any() else 0.0
        out["climb_r"][i] = (top_r - y[meas_r].min()) if meas_r.any() else 0.0
        k = np.flatnonzero(s == ap)
        if len(k):
            k = int(k[0])
            out["apex_by_fill"][i] = bool(f[k] or (k > 0 and f[k - 1]) or (k + 1 < len(f) and f[k + 1]))
    return pd.DataFrame(out, index=feats.index)


def run_file(output: Path, stem: str):
    pj = json.loads((output / "features" / f"{stem}.params.json").read_text())
    P = params_from_dict(pj)
    cloud = Cloud.load_npz(output / ".cache" / f"{stem}.npz")
    scale = float(pj.get("mass_scale") or 1.0)
    cloud.mass_scale = scale
    cap = {}
    orig = FT.gate

    def spy(df, P_):
        cap["df"] = df
        return orig(df, P_)
    FT.gate = spy
    try:
        b = pick_cloud(cloud, P)
        feats, _ = FT.build_features(cloud, b)
    finally:
        FT.gate = orig
    df = cap["df"]
    key = pd.Series(np.arange(len(df)), index=pd.MultiIndex.from_arrays([df["scan_apex"].astype(np.int64).values,
                                                                       np.round(df["mz"].values, 6)]))
    key = key[~key.index.duplicated()]
    row = key.reindex(pd.MultiIndex.from_arrays([feats["scan_apex"].astype(np.int64).values,
                                                 np.round(feats["mz"].values, 6)])).values
    g = df.iloc[np.nan_to_num(row, nan=0).astype(np.int64)].reset_index(drop=True)
    S = side_stats(g, cloud, P)
    out = feats[["feature_id", "mz", "rt_apex", "rt_min", "rt_max", "height", "baseline", "fwhm", "snr", "flags",
                 "scan_apex", "scan_min", "scan_max"]].copy()
    out["matched"] = np.isfinite(row)
    out["on_background"] = g["on_background"].values if "on_background" in g else False
    out["net_w"] = g["net_fwhm"].values / P.fwhm_med if "net_fwhm" in g else np.nan
    out["fwhm_w"] = g["fwhm"].values / P.fwhm_med
    out["noise"] = g["noise"].values
    for c in S.columns:
        out[c] = S[c].values
    F = fill_stats(out, cloud, scale)
    for c in F.columns:
        out[c] = F[c].values
    out["file"] = stem
    out["fwhm_med_s"] = 60.0 * P.fwhm_med
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", type=Path)
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--truth", type=Path, default=None)
    a = ap.parse_args()
    stems = a.files or sorted(p.name[:-len(".params.json")] for p in (a.output / "features").glob("*.params.json"))
    a.out.mkdir(parents=True, exist_ok=True)
    parts = []
    for s in stems:
        t = run_file(a.output, s)
        parts.append(t)
        print(f"{s}: {len(t)} features, matched {t['matched'].mean():.3f}, on_background {int(t['on_background'].sum())}",
              flush=True)
    T = pd.concat(parts, ignore_index=True)
    if a.truth is not None:
        tr = pd.read_csv(a.truth / "truth_rows.tsv", sep="\t")
        tr = tr[tr["clean"] & tr["found"]]
        fcol = "file" if "file" in tr else ("level" if "level" in tr else None)
        tr["stem"] = tr[fcol].astype(str) if fcol else ""
        if not tr["stem"].str.startswith("NEWSTD").all():
            tr["stem"] = "NEWSTD_" + tr["stem"]
        hit = set(zip(tr["stem"], tr["feature_id"].astype(int)))
        T["truth"] = [(f, int(i)) in hit for f, i in zip(T["file"], T["feature_id"])]
    T.to_csv(a.out / "features.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
