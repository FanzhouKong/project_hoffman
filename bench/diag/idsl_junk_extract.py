# diagnostic (report-only): every peak3d candidate basin of IDSL003 within 10 ppm / 0.1 min of a labelled
# m/z-RT pair (TP or TN), with the picker's own descriptors, the gate / score decision and extra raw-trace
# shape descriptors, so the features that land on TN positions ("junk" by the curators) can be compared
# with those on TP positions. Also the labels with no candidate at all. Reads the mzML -> run in srun.
# usage: idsl_junk_extract.py OUT_DIR
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from peak3d.io import load_cloud
from peak3d.estimate import estimate
from peak3d.pick import pick_cloud
from peak3d.features import basins_to_frame, mark_fragments, ridge_ratio, gate, rule_score
from peak3d.isotopes import annotate_isotopes
from peak3d import kernels as K

PPM, RTT = 10.0, 0.1
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()
cloud = load_cloud(ROOT / "data/raw/IDSL_IPA/003.mzML")
P = estimate(cloud)
b = pick_cloud(cloud, P)
df = basins_to_frame(b)
df["fragment"] = mark_fragments(df, cloud, P)
for name, vals in ridge_ratio(df, cloud, P).items():
    df[name] = vals
ok = gate(df, P).values
df["gate"] = ok
# isotopes + score exactly as build_features does, on the gated candidates
cand = np.flatnonzero(ok)
iso_offset, charge, iso_parent, iso_ratio = annotate_isotopes(
    df["mz"].values[cand], df["rt"].values[cand], df["height"].values[cand], df["scan_lo"].values[cand],
    df["scan_hi"].values[cand], cloud, P)
has_partner = np.zeros(len(cand), bool); has_partner[iso_parent[iso_parent >= 0]] = True
df["iso_offset"] = 0; df["charge"] = 1; df["iso_support"] = False; df["iso_ratio"] = np.nan
df.loc[df.index[cand], "iso_offset"] = iso_offset
df.loc[df.index[cand], "charge"] = charge
df.loc[df.index[cand], "iso_support"] = (iso_offset > 0) | has_partner
df.loc[df.index[cand], "iso_ratio"] = iso_ratio
sc = np.full(len(df), np.nan)
sc[cand] = rule_score(df.iloc[cand], P, df["iso_support"].values[cand])
df["score"] = sc
df["kept"] = ok & (np.nan_to_num(sc, nan=-1) >= P.min_score)
# which single gate(s) a rejected candidate fails
g = pd.DataFrame(index=df.index)
g["g_scans"] = df["n_scans"] < P.min_scans
g["g_snr"] = df["snr"] < P.min_snr
g["g_mzsd"] = df["mz_sd_ppm"] > 3 * df["sig_apex"] + 1
g["g_width"] = df["fwhm"] < 0.3 * P.fwhm_med
g["g_prom"] = df["prominence_rel"] < P.min_prominence_rel
g["g_frag"] = df["fragment"]
cap_both = ((df["kflags"] & K.KF_CAP_L) > 0) & ((df["kflags"] & K.KF_CAP_R) > 0)
g["g_cap"] = cap_both & (df["prominence_rel"] < 0.5)
g["g_ridge"] = df["ridge_ratio"] > 0.5
g["g_bg"] = df["background_ratio"] > 1 / 3
g["g_score"] = ok & ~df["kept"].values
df = pd.concat([df, g], axis=1)
print(f"pick+describe {time.perf_counter() - t0:.1f} s: {len(df)} candidates, {ok.sum()} pass gates, {df['kept'].sum()} kept")
print("params:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in P.to_dict().items()
                  if k in ("dt", "fwhm_med", "fwhm_scans", "sig_a", "sig_b", "tol_max_ppm", "K", "min_scans", "floor")})

lab = pd.read_csv(ROOT / "data/raw/IDSL_IPA/idslipa_benchmarking_dataset.csv", encoding="utf-8-sig")
lab = lab.rename(columns={lab.columns[0]: "label_id"})
is_tp = (lab["Manual Curation"] == "TP").values
qm, qr = lab["m/z"].values, lab["RT(min)"].values

# candidate <-> label matching
mzv, rtv = df["mz"].values, df["rt"].values
o = np.argsort(mzv); smz = mzv[o]
rows = []
for i, (m, t) in enumerate(zip(qm, qr)):
    lo = np.searchsorted(smz, m * (1 - PPM * 1e-6)); hi = np.searchsorted(smz, m * (1 + PPM * 1e-6), side="right")
    k = o[lo:hi]; k = k[np.abs(rtv[k] - t) <= RTT]
    for j in k:
        rows.append((i, j))
M = pd.DataFrame(rows, columns=["label_i", "cand_i"])
M["tp"] = is_tp[M["label_i"].values]
print(f"label-candidate pairs: {len(M)}; labels with >= 1 candidate: {M['label_i'].nunique()} / {len(lab)}; "
      f"candidates matching >= 1 label: {M['cand_i'].nunique()}")

# extra raw-trace shape descriptors for every candidate near a label
S = cloud.n_scans; fs = P.fwhm_scans; dt = P.dt
sig_k = 0.5 * fs / 2.355
uniq = np.unique(M["cand_i"].values)
ex = {k: np.full(len(df), np.nan) for k in ("n_lmax", "tv_ratio", "gap_frac", "ripple_rel", "apex_sharp", "far_presence",
                                            "far_med_rel", "near_cont", "mz_trend_ppm", "n_pts_per_scan", "smooth_prom_rel",
                                            "left_rise", "right_rise")}
tic = time.perf_counter()
for j in uniq:
    f = df.iloc[j]
    sa, slo, shi = int(f["scan_apex"]), int(f["scan_lo"]), int(f["scan_hi"])
    w = max(f["fwhm"], P.fwhm_med) / dt
    far = int(max(6 * w, 8)); near = int(max(2 * w, 2))
    s0, s1 = max(0, sa - far), min(S, sa + far + 1)
    ppm = max(5.0, 3.0 * f["sig_apex"] + 1.0)
    e = cloud.eic(float(f["mz"]), ppm, s0, s1).astype(np.float64)
    tr = e[slo - s0: shi - s0 + 1]
    pos = tr > 0
    ex["gap_frac"][j] = 1 - pos.mean()
    # local maxima of the raw trace inside the bounds (zeros are gaps, not minima)
    tp_ = tr[pos]
    if len(tp_) >= 3:
        lm = (tp_[1:-1] > tp_[:-2]) & (tp_[1:-1] >= tp_[2:])
        ex["n_lmax"][j] = 1 + lm.sum() if (tp_[0] > tp_[1] or tp_[-1] > tp_[-2]) else lm.sum() + ((tp_[0] >= tp_[1]) | (tp_[-1] >= tp_[-2]))
        ex["n_lmax"][j] = max(1, lm.sum() + int(tp_[0] > tp_[1]) + int(tp_[-1] > tp_[-2]))
        apex = tp_.max(); base = min(tp_[0], tp_[-1])
        ex["tv_ratio"][j] = np.abs(np.diff(tp_)).sum() / max(2 * (apex - base), 1e-9) if apex > base else np.nan
    else:
        ex["n_lmax"][j] = 1; ex["tv_ratio"][j] = 1.0
    es = gaussian_filter1d(e, sig_k, mode="nearest")
    ex["ripple_rel"][j] = np.median(np.abs(e - es)[slo - s0: shi - s0 + 1]) / max(f["height"], 1e-9)
    ka = sa - s0
    nb = [e[ka - 1] if ka - 1 >= 0 else 0.0, e[ka + 1] if ka + 1 < len(e) else 0.0]
    ex["apex_sharp"][j] = e[ka] / max(np.mean(nb), 1e-9)
    ex["left_rise"][j] = e[ka] / max(nb[0], 1e-9); ex["right_rise"][j] = e[ka] / max(nb[1], 1e-9)
    farmask = np.ones(len(e), bool); farmask[max(0, ka - near): ka + near + 1] = False
    fe = e[farmask]
    ex["far_presence"][j] = (fe > 0).mean() if len(fe) else np.nan
    ex["far_med_rel"][j] = (np.median(fe[fe > 0]) / f["height"]) if (fe > 0).any() else 0.0
    # contiguity: longest run of present scans around the apex / span of bounds
    run = 1; k = ka
    while k - 1 >= slo - s0 and e[k - 1] > 0: k -= 1; run += 1
    k = ka
    while k + 1 <= shi - s0 and e[k + 1] > 0: k += 1; run += 1
    ex["near_cont"][j] = run / (shi - slo + 1)
    # smoothed prominence relative to smoothed apex, inside the bounds
    seg = es[slo - s0: shi - s0 + 1]
    ex["smooth_prom_rel"][j] = (seg.max() - min(seg[0], seg[-1])) / max(seg.max(), 1e-9)
    ex["n_pts_per_scan"][j] = f["n_points"] / max(shi - slo + 1, 1)
    # m/z trend across the peak: per-scan-max m/z of left half vs right half (ppm)
    idx = cloud.eic_argmax(float(f["mz"]), ppm, slo, shi + 1); idx = idx[idx >= 0]
    if len(idx) >= 4:
        h = len(idx) // 2
        ex["mz_trend_ppm"][j] = (np.median(cloud.mz[idx[h:]]) - np.median(cloud.mz[idx[:h]])) / f["mz"] * 1e6
print(f"extra descriptors for {len(uniq)} candidates: {time.perf_counter() - tic:.1f} s")
for k, v in ex.items():
    df[k] = v

keep_cols = ["mz", "rt", "height", "n_scans", "n_points", "n_gaps", "fwhm", "asym", "mz_sd_ppm", "sig_apex", "baseline",
             "baseline_lo", "noise", "snr", "prominence_rel", "persistence", "persistence_rel", "gauss_r2", "ridge_ratio",
             "background_ratio", "ripple_ratio", "fragment", "kflags", "scan_apex", "scan_lo", "scan_hi", "n_flat",
             "iso_offset", "charge", "iso_support", "iso_ratio", "score", "gate", "kept"] + list(g.columns) + list(ex)
D = df.iloc[uniq][keep_cols].copy(); D["cand_i"] = uniq
M = M.merge(D, on="cand_i", how="left")
M["label_mz"] = qm[M["label_i"].values]; M["label_rt"] = qr[M["label_i"].values]
M["d_ppm"] = (M["mz"] - M["label_mz"]) / M["label_mz"] * 1e6; M["d_rt_s"] = 60 * (M["rt"] - M["label_rt"])
M.to_csv(out / "pairs.tsv", sep="\t", index=False, float_format="%.6g")
# all kept features (for the "unlabelled" population) with the picker descriptors only
df.loc[df["kept"], [c for c in keep_cols if c not in ex]].to_csv(out / "kept_features.tsv", sep="\t", index=False, float_format="%.6g")
# labels with no candidate: what is in the raw data there?
none = np.setdiff1d(np.arange(len(lab)), M["label_i"].unique())
lab_pts = b.lab; B = int(lab_pts.max()) + 1
smin, smax, npts = K.basin_extent(lab_pts, cloud.off, B)
rows = []
for i in none:
    m, t = qm[i], qr[i]
    s0, s1 = int(np.searchsorted(cloud.rt, t - RTT)), int(np.searchsorted(cloud.rt, t + RTT))
    e = cloud.eic(m, 20.0, s0, s1)
    idx = cloud.eic_argmax(m, 20.0, s0, s1); idx = idx[idx >= 0]
    rec = dict(label_i=i, tp=is_tp[i], eic_nz=int((e > 0).sum()), eic_n=int(s1 - s0), eic_max=float(e.max()) if len(e) else 0.0)
    if len(idx):
        labs = lab_pts[idx]; u, inv = np.unique(labs, return_inverse=True)
        wts = np.bincount(inv, weights=cloud.inten[idx]); dom = u[np.argmax(wts)]
        rec.update(dom_span=int(smax[dom] - smin[dom] + 1), dom_npts=int(npts[dom]), n_basins=len(u))
    rows.append(rec)
pd.DataFrame(rows).to_csv(out / "no_candidate.tsv", sep="\t", index=False, float_format="%.6g")
json.dump(P.to_dict(), open(out / "params.json", "w"), indent=1)
print(f"labels without any candidate: {len(none)} (TP {is_tp[none].sum()}, TN {(~is_tp[none]).sum()}); total {time.perf_counter() - t0:.1f} s")
