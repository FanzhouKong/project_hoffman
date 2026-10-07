# diagnostic used while tuning peak3d (reads results/peak3d and the labels; report-only)
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "/quobyte/metabolomicsgrp/fanzhou/hoffmann")
from peak3d.io import load_cloud
from peak3d.pick import pick_cloud
from peak3d.features import basins_to_frame, gate
from peak3d import kernels as K

cloud = load_cloud("data/raw/IDSL_IPA/003.mzML")
b = pick_cloud(cloud)
P = b.params
df = basins_to_frame(b)              # every basin with span >= min_scans
df["ok"] = gate(df, P).values
lab_pts = b.lab                      # final basin id per centroid
scan_of = cloud.scan_index()
# basin extents for ALL final basins (including tiny ones)
B = int(lab_pts.max()) + 1
smin, smax, npts = K.basin_extent(lab_pts, cloud.off, B)
row_of_basin = pd.Series(df.index.values, index=df["basin_id"].values)

lab = pd.read_csv("data/raw/IDSL_IPA/idslipa_benchmarking_dataset.csv", encoding="utf-8-sig")
is_tp = (lab["Manual Curation"] == "TP").values
qm, qr = lab["m/z"].values, lab["RT(min)"].values
print("params: fwhm_scans %.2f min_scans %d K %d sig_a %.2f tol_max %.1f dt_s %.2f" % (P.fwhm_scans, P.min_scans, P.K, P.sig_a, P.tol_max_ppm, 60 * P.dt))

def cands(mz, rt, m, t, ppm, rt_tol, order, smz):
    lo = np.searchsorted(smz, m * (1 - ppm * 1e-6)); hi = np.searchsorted(smz, m * (1 + ppm * 1e-6), side="right")
    k = order[lo:hi]
    return k[np.abs(rt[k] - t) <= rt_tol]

order = np.argsort(df["mz"].values); smz = df["mz"].values[order]
mzv, rtv = df["mz"].values, df["rt"].values
rows = []
for i in np.flatnonzero(is_tp):
    m, t = qm[i], qr[i]
    k10 = cands(mzv, rtv, m, t, 10, 0.1, order, smz)
    kept = k10[df["ok"].values[k10] & (df["score"].values[k10] >= P.min_score)] if "score" in df else k10[df["ok"].values[k10]]
    rec = dict(i=i, mz=m, rt=t, n_cand10=len(k10), detected=len(kept) > 0)
    if len(k10) == 0:
        # looser tolerance
        k30 = cands(mzv, rtv, m, t, 30, 0.2, order, smz)
        rec["n_cand30"] = len(k30)
        # raw points near the label
        s0, s1 = cloud.scan_window(t, 0.1) if hasattr(cloud, "scan_window") else (np.searchsorted(cloud.rt, t - 0.1), np.searchsorted(cloud.rt, t + 0.1))
        e = cloud.eic(m, 20.0, int(s0), int(s1))
        rec["eic_nz"] = int(np.count_nonzero(e)); rec["eic_max"] = float(e.max()) if len(e) else 0.0
        idx = cloud.eic_argmax(m, 20.0, int(s0), int(s1)); idx = idx[idx >= 0]
        if len(idx):
            labs = lab_pts[idx]
            # dominant basin by summed intensity
            u, inv = np.unique(labs, return_inverse=True)
            w = np.bincount(inv, weights=cloud.inten[idx])
            dom = u[np.argmax(w)]
            span = smax[dom] - smin[dom] + 1
            rec["dom_span"] = int(span); rec["dom_npts"] = int(npts[dom])
            if dom in row_of_basin.index:
                r = df.loc[row_of_basin[dom]]
                rec["dom_in_df"] = True; rec["dom_rt_off"] = float(r["rt"] - t); rec["dom_ppm_off"] = float((r["mz"] - m) / m * 1e6)
                rec["dom_ok"] = bool(r["ok"]); rec["dom_height"] = float(r["height"])
            else:
                rec["dom_in_df"] = False
    rows.append(rec)
R = pd.DataFrame(rows)
miss = R[~R["detected"]]
nc = miss[miss["n_cand10"] == 0]
print(f"\nTP total {len(R)}, detected {R['detected'].sum()}, missed {len(miss)}, of which no candidate within 10ppm/0.1min: {len(nc)}")
print("  of those, candidate exists within 30 ppm / 0.2 min:", int((nc['n_cand30'] > 0).sum()))
print("  raw EIC (20 ppm, +/-0.1 min) nonzero scans distribution:", nc["eic_nz"].value_counts().sort_index().head(12).to_dict())
print("  eic_max quantiles:", nc["eic_max"].quantile([0.1, 0.5, 0.9]).round(0).to_dict())
has = nc[nc["eic_nz"] > 0]
small = has[~has["dom_in_df"].astype(bool)]
merged = has[has["dom_in_df"].astype(bool)]
print(f"  no raw points at all: {(nc['eic_nz'] == 0).sum()}")
print(f"  points belong to a basin too small to be a candidate (span < min_scans): {len(small)}; their span distribution {small['dom_span'].value_counts().sort_index().to_dict()}")
print(f"  points belong to a larger basin whose apex is elsewhere (merged/climbed): {len(merged)}")
if len(merged):
    print("    |rt offset| quantiles (min):", merged["dom_rt_off"].abs().quantile([0.1, 0.5, 0.9]).round(3).to_dict())
    print("    |ppm offset| quantiles:", merged["dom_ppm_off"].abs().quantile([0.1, 0.5, 0.9]).round(1).to_dict())
    print("    dominant basin passes gates:", int(merged["dom_ok"].sum()), "of", len(merged))
    print("    cases with |rt off| <= 0.1 but |ppm| > 10:", int(((merged["dom_rt_off"].abs() <= 0.1) & (merged["dom_ppm_off"].abs() > 10)).sum()),
          " | |ppm| <= 10 but |rt off| > 0.1:", int(((merged["dom_rt_off"].abs() > 0.1) & (merged["dom_ppm_off"].abs() <= 10)).sum()))

# fragment-gated TPs: does the apex-side bound abut another basin (true fragment) or a dropout?
print("\n== fragment-gated TP candidates: bound type ==")
at_lo = (df["scan_apex"] == df["scan_lo"]).values; at_hi = (df["scan_apex"] == df["scan_hi"]).values
run_edge = ((df["kflags"].values.astype(int) & (K.KF_RUN_START | K.KF_RUN_END)) > 0)
frag = (at_lo | at_hi) & ~run_edge
n_sad = n_drop = 0; heights = []
for i in np.flatnonzero(is_tp):
    k10 = cands(mzv, rtv, qm[i], qr[i], 10, 0.1, order, smz)
    k10 = k10[frag[k10]]
    if len(k10) == 0:
        continue
    r = df.loc[k10[np.argmax(df["height"].values[k10])]]
    side_scan = int(r["scan_lo"]) - 1 if at_lo[r.name] else int(r["scan_hi"]) + 1
    if 0 <= side_scan < cloud.n_scans:
        e = cloud.eic(float(r["mz"]), P.tol_max_ppm, side_scan, side_scan + 1)
        if e[0] > 0: n_sad += 1
        else: n_drop += 1
    heights.append(r["height"])
print(f"  apex-side bound abuts a centroid (neighbouring basin, true fragment): {n_sad}; dropout (no centroid beyond): {n_drop}; median height {np.median(heights):.0f}")
