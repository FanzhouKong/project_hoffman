# diagnostic (report-only): the IDSL003 TP labels with no candidate basin within 10 ppm / 0.1 min. For each:
# where did the centroids at the label go? (a) were they an apex BEFORE the persistence merge (killed by the
# merge) or never an apex (steepest-ascent linking)? (b) where is the apex of the basin that holds them now
# (RT / ppm offset, height ratio, valley depth between)? Plus a gallery with basin membership coloured.
# usage: idsl_nocand.py OUT_PREFIX [N_GALLERY=60]
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from peak3d.io import load_cloud
from peak3d.estimate import estimate
from peak3d.pick import pick_cloud
from peak3d import kernels as K

pre = sys.argv[1]; NG = int(sys.argv[2]) if len(sys.argv) > 2 else 60
cloud = load_cloud(ROOT / "data/raw/IDSL_IPA/003.mzML")
P = estimate(cloud)
b = pick_cloud(cloud, P)
lab2 = b.lab; B2 = int(lab2.max()) + 1
# pre-merge labels (replicate the first stages of pick_cloud)
sig = K.per_point_sigma(cloud.inten, P.sig_a, P.sig_b)
nbr = K.find_neighbors(cloud.off, cloud.mz, cloud.inten, sig, P.K, P.tol_max_ppm, P.same_scan_k, P.same_scan_floor_ppm)
eff = K.effective_intensity(nbr, cloud.inten)
parent = K.link_steepest(nbr, eff, cloud.inten)
label0 = K.resolve_roots(parent)
lab0, roots0 = K.dense_labels(label0)
smin0, smax0, npts0 = K.basin_extent(lab0, cloud.off, len(roots0))
smin2, smax2, npts2 = K.basin_extent(lab2, cloud.off, B2)
order2, boff2 = K.sort_by_label(lab2, B2)
scan_of = cloud.scan_index()
cand_basins = set(b.basin_id.tolist())
NC = pd.read_csv(ROOT / "results/diag_idsl/no_candidate.tsv", sep="\t")
labdf = pd.read_csv(ROOT / "data/raw/IDSL_IPA/idslipa_benchmarking_dataset.csv", encoding="utf-8-sig")
qm, qr = labdf["m/z"].values, labdf["RT(min)"].values
fs = P.fwhm_scans
rows = []
for i in NC.loc[NC["tp"], "label_i"].values:
    m, t = qm[i], qr[i]
    s0, s1 = int(np.searchsorted(cloud.rt, t - 0.1)), int(np.searchsorted(cloud.rt, t + 0.1))
    idx = cloud.eic_argmax(m, 10.0, s0, s1)
    if (idx >= 0).sum() == 0:
        idx = cloud.eic_argmax(m, 20.0, s0, s1)
    ok = idx >= 0
    if ok.sum() == 0:
        rows.append(dict(label_i=i, cls="no points")); continue
    pts = idx[ok]
    e = cloud.inten[pts]
    pk = pts[np.argmax(e)]                      # the most intense centroid of the ion near the label
    s_pk = scan_of[pk]
    rec = dict(label_i=i, label_mz=m, label_rt=t, h_label=float(cloud.inten[pk]), rt_label_pk=float(cloud.rt[s_pk]))
    # pre-merge: was pk (or something within 1 FWHM of it) a root?
    r0 = lab0[pk]
    root_pt0 = roots0[r0]
    rec["pre_root_scan_off"] = int(scan_of[root_pt0] - s_pk)
    rec["pre_root_ppm"] = float((cloud.mz[root_pt0] - m) / m * 1e6)
    rec["pre_span"] = int(smax0[r0] - smin0[r0] + 1)
    rec["pre_was_apex"] = abs(rec["pre_root_scan_off"]) <= max(1, 0.5 * fs) and abs(rec["pre_root_ppm"]) <= 10
    # post-merge basin that holds pk
    d = lab2[pk]
    dp = order2[boff2[d]:boff2[d + 1]]
    apex_pt = dp[np.argmax(eff[dp])]
    rec["post_span"] = int(smax2[d] - smin2[d] + 1)
    rec["post_apex_rt_off_s"] = float(60 * (cloud.rt[scan_of[apex_pt]] - t))
    rec["post_apex_ppm"] = float((cloud.mz[apex_pt] - m) / m * 1e6)
    rec["post_apex_h"] = float(eff[apex_pt])
    rec["post_in_candidates"] = d in cand_basins
    # valley between the label bump and the basin apex along the ion's EIC (per-scan max at 10 ppm)
    sa = scan_of[apex_pt]
    lo, hi = min(sa, s_pk), max(sa, s_pk)
    if hi > lo:
        seg = cloud.eic(m, 10.0, lo, hi + 1).astype(float)
        rec["valley_rel"] = float(seg.min() / max(cloud.inten[pk], 1e-9))
    else:
        rec["valley_rel"] = 1.0
    # how many kept features does the file have within 0.25 min at this m/z (10 ppm)?
    rec["n_pts_10ppm"] = int(ok.sum())
    rows.append(rec)
R = pd.DataFrame(rows)
def cls(r):
    if r.get("cls") == "no points": return "no points"
    if r["pre_span"] < P.min_scans and r["post_span"] < P.min_scans: return "noise-size basin (< min_scans)"
    if abs(r["post_apex_ppm"]) > 10 and abs(r["post_apex_rt_off_s"]) <= 6: return "m/z off > 10 ppm (same RT)"
    if r["pre_was_apex"]:
        return "apex pre-merge, merged into neighbour <= 2 FWHM" if abs(r["post_apex_rt_off_s"]) <= 2 * 60 * P.fwhm_med + 6 else "apex pre-merge, merged into far apex (> 2 FWHM)"
    return "never an apex (linking climbed to neighbour)" if abs(r["post_apex_rt_off_s"]) <= 2 * 60 * P.fwhm_med + 6 else "never an apex, far basin"
R["cls"] = R.apply(cls, axis=1)
R.to_csv(f"{pre}_nocand.tsv", sep="\t", index=False, float_format="%.6g")
pd.set_option("display.width", 220)
print("params fwhm_med %.4f min (%.2f scans), min_scans %d, dt %.3f s" % (P.fwhm_med, fs, P.min_scans, 60 * P.dt))
print(R["cls"].value_counts().to_string())
Q = R[R["cls"] != "no points"]
print("\npost-merge apex offset from label (s): ", Q["post_apex_rt_off_s"].abs().quantile([.1, .25, .5, .75, .9]).round(1).to_dict())
print("post-merge apex height / label bump height:", (Q["post_apex_h"] / Q["h_label"]).quantile([.1, .25, .5, .75, .9]).round(2).to_dict())
print("valley between bump and apex, relative to the bump:", Q["valley_rel"].quantile([.1, .25, .5, .75, .9]).round(2).to_dict())
print("bump height quantiles:", Q["h_label"].quantile([.1, .25, .5, .75, .9]).round(0).to_dict())
print("post-merge basin is a candidate (>= min_scans):", Q["post_in_candidates"].mean().round(3))
for c, g in Q.groupby("cls"):
    print(f"\n  {c}: n {len(g)}  |rt off| med {g['post_apex_rt_off_s'].abs().median():.1f} s  apex/bump med {(g['post_apex_h'] / g['h_label']).median():.2f}  "
          f"valley_rel med {g['valley_rel'].median():.2f}  bump h med {g['h_label'].median():.0f}  pre_span med {g['pre_span'].median():.0f}  post_span med {g['post_span'].median():.0f}")

# gallery
rng = np.random.default_rng(2)
pick = Q.sample(min(NG, len(Q)), random_state=2)
ncol = 5; nrow = int(np.ceil(len(pick) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(16, 2.6 * nrow), dpi=110)
cmap = plt.cm.tab20
for ax, (_, r) in zip(axes.ravel(), pick.iterrows()):
    m, t = r["label_mz"], r["label_rt"]
    W = int(max(10 * fs, 30)); sc = int(np.searchsorted(cloud.rt, t))
    s0, s1 = max(0, sc - W), min(cloud.n_scans, sc + W + 1)
    x = cloud.rt[s0:s1] * 60
    e30 = cloud.eic(m, 30.0, s0, s1); ax.plot(x, e30, color="#cccccc", lw=0.8)
    idx = cloud.eic_argmax(m, 10.0, s0, s1)
    e = np.where(idx >= 0, cloud.inten[np.maximum(idx, 0)], 0.0)
    ax.plot(x, e, color="#444444", lw=0.7)
    okk = idx >= 0
    labs = lab2[idx[okk]]
    u = np.unique(labs)
    for j, lb in enumerate(u):
        mm = labs == lb
        ax.scatter(x[okk][mm], e[okk][mm], s=9, color=cmap(j % 20), zorder=3)
    ax.axvline(t * 60, color="#cf222e", lw=0.8, ls="--")
    ax.set_title(f"m/z {m:.4f} rt {t:.2f}  bump h {r['h_label']:.0f}  {r['cls']}\napex off {r['post_apex_rt_off_s']:+.1f} s, {r['post_apex_ppm']:+.1f} ppm, apex/bump {r['post_apex_h'] / r['h_label']:.1f}, valley {r['valley_rel']:.2f}, span pre {r['pre_span']} post {r['post_span']}", fontsize=6)
    ax.tick_params(labelsize=5)
for ax in axes.ravel()[len(pick):]: ax.axis("off")
fig.suptitle("IDSL003 TP labels with no candidate basin: EIC at 10 ppm (dark), 30 ppm (grey); dots coloured by FINAL basin membership; red dashed = label RT", fontsize=9)
fig.tight_layout(); fig.savefig(f"{pre}_nocand.png"); plt.close(fig)
print("gallery written")
