# figures explaining the peak3d method on one real file (HZV029 batch4 ...003C) and its two replicate injections
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.runs import RUNS
from peak3d import kernels as K
from peak3d.estimate import estimate
from peak3d.features import build_features, basins_to_frame
from peak3d.io import load_cloud
from peak3d.pick import pick_cloud
from peak3d.warp import TableWarp
OUT = ROOT / "results/figures/method_demo"; OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, GRID = "#1f2328", "#59636e", "#d0d7de"
C1, C2, C3, C4, C5 = "#0969da", "#bf8700", "#8250df", "#cf222e", "#1a7f37"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
run = "HZV029_cert"; files = RUNS[run]["files"]; stems = [Path(f).stem for f in files]
cloud = load_cloud(files[0]); P = estimate(cloud)
print("file", cloud.name, "centroids", cloud.n_points, "scans", cloud.n_scans, "dt s", 60 * P.dt, "fwhm s", 60 * P.fwhm_med, "fwhm scans", P.fwhm_scans,
      "sig_a", P.sig_a, "sig_b", P.sig_b, "floor", P.floor, "noise law c r", P.noise_c, P.noise_r)
# ---- pre-merge stages
sig = K.per_point_sigma(cloud.inten, P.sig_a, P.sig_b)
nbr = K.find_neighbors(cloud.off, cloud.mz, cloud.inten, sig, P.K, P.tol_max_ppm, P.same_scan_k, P.same_scan_floor_ppm)
eff = K.effective_intensity(nbr, cloud.inten)
parent = K.link_steepest(nbr, eff, cloud.inten)
lab0, roots0 = K.dense_labels(K.resolve_roots(parent))
b = pick_cloud(cloud, P)
feats, rej = build_features(cloud, b, keep_rejected=True)
df_all = basins_to_frame(b)
scan_of = cloud.scan_index()
print("basins initial", b.n_basins_initial, "after merge", b.n_after_merge, "candidates", len(b.basin_id), "kept", len(feats), "rejected", len(rej))
print("rejection reasons:", rej["reason"].value_counts().to_dict())

# ---- choose a demo feature: moderate, isotope-supported, clean, with a neighbour ion within 0.03 Da
F = feats.copy()
F["iso_sup"] = (F["iso_offset"] > 0) | F["feature_id"].isin(F["iso_parent"])
cand = F[(F["height"].between(2e5, 2e6)) & (F["far_level"] < 0.05) & (F["iso_offset"] == 0) & F["iso_sup"] & (F["rt"].between(1.0, 4.5)) & (F["n_scans"] >= 8)]
mz_sorted = np.sort(F["mz"].values)
def n_near(m, dm=0.03):
    return np.searchsorted(mz_sorted, m + dm) - np.searchsorted(mz_sorted, m - dm) - 1
cand = cand.assign(nn=[n_near(m) for m in cand["mz"].values])
cand = cand[cand["nn"].between(1, 4)].sort_values("height", ascending=False)
demo = cand.iloc[len(cand) // 3]
print("demo feature:", demo[["mz", "rt", "height", "snr", "eic_snr", "score", "n_scans", "fwhm", "far_level", "ridge_ratio", "tv_ratio", "gauss_r2"]].to_dict())
sa, slo, shi = int(demo["scan_apex"]), int(demo["scan_min"]), int(demo["scan_max"])
fs = P.fwhm_scans

# ---- Fig 1: the point cloud around the demo feature (RT x m/z, colour = intensity) and its 3D rendering
W = int(12 * fs); s0, s1 = max(0, sa - W), min(cloud.n_scans, sa + W + 1)
m0, m1 = demo["mz"] - 0.02, demo["mz"] + 0.045
sel = np.flatnonzero((scan_of >= s0) & (scan_of < s1) & (cloud.mz >= m0) & (cloud.mz <= m1))
fig = plt.figure(figsize=(12, 4.6))
ax = fig.add_subplot(1, 2, 1)
sc = ax.scatter(cloud.rt[scan_of[sel]] * 60, cloud.mz[sel], c=cloud.inten[sel], s=9, cmap="viridis", norm=LogNorm())
fig.colorbar(sc, ax=ax, label="centroid intensity")
ax.set_xlabel("retention time (s)"); ax.set_ylabel("m/z"); ax.set_title(f"a. all MS1 centroids in a 0.065 Da x {2 * W * P.dt * 60:.0f} s window ({len(sel)} points)")
ax3 = fig.add_subplot(1, 2, 2, projection="3d")
ax3.scatter(cloud.rt[scan_of[sel]] * 60, cloud.mz[sel], np.log10(cloud.inten[sel]), c=cloud.inten[sel], s=6, cmap="viridis", norm=LogNorm())
ax3.set_xlabel("RT (s)"); ax3.set_ylabel("m/z"); ax3.set_zlabel("log10 intensity"); ax3.set_title("b. the same points as a surface: peaks are its hills")
ax3.view_init(elev=28, azim=-55)
fig.tight_layout(); fig.savefig(OUT / "01_point_cloud.png"); plt.close(fig)

# ---- Fig 2: linking: every centroid points to its most intense neighbour; basins before and after the merge
Wl = int(5 * fs); s0l, s1l = max(0, sa - Wl), min(cloud.n_scans, sa + Wl + 1)
m0l, m1l = demo["mz"] - 0.006, demo["mz"] + 0.006
sel2 = np.flatnonzero((scan_of >= s0l) & (scan_of < s1l) & (cloud.mz >= m0l) & (cloud.mz <= m1l))
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
x = cloud.rt[scan_of[sel2]] * 60; y = cloud.mz[sel2]
for p in sel2:
    q = parent[p]
    if q != p:
        axes[0].annotate("", xy=(cloud.rt[scan_of[q]] * 60, cloud.mz[q]), xytext=(cloud.rt[scan_of[p]] * 60, cloud.mz[p]),
                         arrowprops=dict(arrowstyle="->", color=INK2, lw=0.6, alpha=0.7))
axes[0].scatter(x, y, c=cloud.inten[sel2], s=14, cmap="viridis", norm=LogNorm(), zorder=3)
roots_here = [p for p in sel2 if parent[p] == p]
axes[0].scatter(cloud.rt[scan_of[roots_here]] * 60, cloud.mz[roots_here], s=70, facecolors="none", edgecolors=C4, linewidths=1.2, zorder=4)
axes[0].set_title(f"a. steepest-ascent links; {len(roots_here)} apexes (red rings) before the merge"); axes[0].set_xlabel("RT (s)"); axes[0].set_ylabel("m/z")
u0, inv0 = np.unique(lab0[sel2], return_inverse=True)
axes[1].scatter(x, y, c=inv0 % 20, s=14, cmap="tab20", zorder=3); axes[1].set_title(f"b. basins before the persistence merge: {len(u0)}"); axes[1].set_xlabel("RT (s)")
u2, inv2 = np.unique(b.lab[sel2], return_inverse=True)
axes[2].scatter(x, y, c=inv2 % 20, s=14, cmap="tab20", zorder=3); axes[2].set_title(f"c. after the merge: {len(u2)} (ripple maxima and noise died into their neighbours)"); axes[2].set_xlabel("RT (s)")
fig.tight_layout(); fig.savefig(OUT / "02_linking_basins.png"); plt.close(fig)

# ---- Fig 3: persistence merge on one ragged trace: the ion's EIC with points coloured by pre-merge basin, half-valley level
# pick a kept feature with many pre-merge basins along its trace
cnt = []
for i, f in F.iterrows():
    pts = np.flatnonzero(b.lab == int(df_all.loc[df_all["apex_idx"] == f["scan_apex"] * 0 + df_all["apex_idx"].iloc[0], "basin_id"].iloc[0])) if False else None
    break
# simpler: per kept feature, count distinct pre-merge labels of the per-scan-max centroids of its EIC inside the bounds
def premerge_count(f):
    idx = cloud.eic_argmax(float(f["mz"]), 6.0, int(f["scan_min"]), int(f["scan_max"]) + 1); idx = idx[idx >= 0]
    return len(np.unique(lab0[idx])), len(np.unique(b.lab[idx]))
sub = F[(F["n_scans"] >= 12) & (F["height"].between(5e4, 5e5)) & (F["rt"].between(0.8, 5))].copy()
pc = np.array([premerge_count(f) for _, f in sub.iterrows()])
sub["n_pre"], sub["n_post"] = pc[:, 0], pc[:, 1]
rag = sub[(sub["n_pre"] >= 4) & (sub["n_post"] == 1)].sort_values("n_pre", ascending=False)
r = rag.iloc[min(2, len(rag) - 1)]
sa_r, lo_r, hi_r = int(r["scan_apex"]), int(r["scan_min"]), int(r["scan_max"])
Wr = int(6 * fs); s0r, s1r = max(0, sa_r - Wr), min(cloud.n_scans, sa_r + Wr + 1)
idx = cloud.eic_argmax(float(r["mz"]), 6.0, s0r, s1r)
e = np.where(idx >= 0, eff[np.maximum(idx, 0)], 0.0); t = cloud.rt[s0r:s1r] * 60
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
ax = axes[0]; ax.plot(t, e, color=INK2, lw=0.8)
ok = idx >= 0; labs = lab0[idx[ok]]; u, inv = np.unique(labs, return_inverse=True)
ax.scatter(t[ok], e[ok], c=inv % 20, cmap="tab20", s=28, zorder=3)
ax.axvspan(cloud.rt[lo_r] * 60, cloud.rt[hi_r] * 60, color=C2, alpha=0.12)
ax.set_title(f"a. one ion's trace: {len(u)} apexes before the merge (colours = pre-merge basins)"); ax.set_xlabel("RT (s)"); ax.set_ylabel("intensity")
ax = axes[1]; ax.plot(t, e, color=C1, lw=1.2, marker=".", ms=4)
apex_v = e.max(); ax.axhline(0.5 * apex_v, color=C4, ls="--", lw=0.8); ax.text(t[0], 0.5 * apex_v, " half of the apex: a valley must drop below this for two peaks", color=C4, va="bottom", fontsize=8)
sm = gaussian_filter1d(e, 0.5 * fs / 2.355, mode="nearest"); ax.plot(t, sm, color=C3, lw=1.0, ls=":", label="lightly smoothed (far-apart rule)")
ax.axvspan(cloud.rt[lo_r] * 60, cloud.rt[hi_r] * 60, color=C2, alpha=0.12, label="feature bounds")
ax.legend(frameon=False, fontsize=8); ax.set_title("b. after the merge: one basin, one feature"); ax.set_xlabel("RT (s)")
fig.tight_layout(); fig.savefig(OUT / "03_persistence_merge.png"); plt.close(fig)

# ---- Fig 4: the noise surface
S = b.noise_surface
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
im = axes[0].imshow(S, aspect="auto", origin="lower", cmap="magma", norm=LogNorm(), extent=[cloud.rt[0], cloud.rt[-1], 0, S.shape[0]])
fig.colorbar(im, ax=axes[0], label="cell noise (median isolated centroid)")
axes[0].set_xlabel("RT (min)"); axes[0].set_ylabel("log-m/z band (16 over the mass range)"); axes[0].set_title("a. the 3D noise surface: 16 m/z bands x 1-min blocks")
iso = K.short_basin_mask(b.lab, *K.basin_extent(b.lab, cloud.off, int(b.lab.max()) + 1)[:2], 2)
axes[1].hist(np.log10(cloud.inten[iso]), bins=80, color=INK2, alpha=0.6, label=f"isolated centroids ({iso.sum():,}): the noise")
axes[1].hist(np.log10(cloud.inten[~iso]), bins=80, color=C1, alpha=0.6, label=f"centroids in traces of >= 3 scans ({(~iso).sum():,})")
axes[1].axvline(np.log10(P.floor), color=C4, ls="--", lw=0.8); axes[1].text(np.log10(P.floor), axes[1].get_ylim()[1] * 0.9, " file floor (1st percentile)", color=C4, fontsize=8)
axes[1].set_xlabel("log10 intensity"); axes[1].set_ylabel("centroids"); axes[1].legend(frameon=False, fontsize=8); axes[1].set_title("b. what the surface is made of")
fig.tight_layout(); fig.savefig(OUT / "04_noise_surface.png"); plt.close(fig)

# ---- Fig 5: descriptors and gates on the demo feature
w = min(max(demo["fwhm"], P.fwhm_med), 2 * P.fwhm_med) / P.dt
near = int(max(1, round(2 * w))); far = int(max(near + 2, round(6 * w)))
s0d, s1d = max(0, sa - far - 3), min(cloud.n_scans, sa + far + 4)
ppm = max(5.0, 3 * P.sig_ppm(demo["height"]) + 1)
e = cloud.eic(float(demo["mz"]), ppm, s0d, s1d).astype(float); t = cloud.rt[s0d:s1d] * 60
fig, ax = plt.subplots(figsize=(13, 4.6))
ax.plot(t, e, color=C1, lw=1.2, marker=".", ms=4, label="EIC (per-scan maximum within the m/z tolerance)")
ax.axvspan(cloud.rt[slo] * 60, cloud.rt[shi] * 60, color=C2, alpha=0.15, label="bounds (walk from the apex until 2 scans <= 2x noise)")
for a0, a1, lab in ((sa - far, sa - near, "far side windows: 2 to 6 widths out"), (sa + near, sa + far, None)):
    ax.axvspan(cloud.rt[max(0, a0)] * 60, cloud.rt[min(cloud.n_scans - 1, a1)] * 60, color=C3, alpha=0.10, label=lab)
ax.axhline(demo["noise"], color=C4, ls="--", lw=0.8, label=f"cell noise {demo['noise']:.3g}")
ax.axhline(demo["baseline_lo"], color=INK2, ls=":", lw=0.8, label=f"baseline_lo {demo['baseline_lo']:.3g} (lowest level at the bounds)")
half = 0.5 * demo["height"]; ax.axhline(half, color=C5, ls="--", lw=0.7); ax.text(t[0], half, f" half max: FWHM {60 * demo['fwhm']:.2f} s", color=C5, fontsize=8, va="bottom")
ax.plot(cloud.rt[sa] * 60, demo["height"], marker="v", color=C4, ms=9)
txt = (f"height {demo['height']:.3g}   S/N = (height - baseline_lo) / cell noise = {demo['snr']:.1f}\n"
       f"prominence_rel {demo['prominence_rel']:.2f}   gauss_r2 {demo['gauss_r2']:.2f}   mz_sd {demo['mz_sd_ppm']:.2f} ppm (model sigma {P.sig_ppm(demo['height']):.2f})\n"
       f"ridge_ratio {demo['ridge_ratio']:.2f} (<= 0.5)   background_ratio {demo['background_ratio']:.2f} (<= 0.33)   far_level {demo['far_level']:.2f} (<= 0.3)\n"
       f"tv_ratio {demo['tv_ratio']:.2f} (1 = one rise and fall)   n_valleys {int(demo['n_valleys'])}   eic_snr {demo['eic_snr']:.1f} (>= 3)   score {demo['score']:.2f} (>= 0.5)")
ax.text(0.01, 0.98, txt, transform=ax.transAxes, va="top", fontsize=8.5, family="monospace", bbox=dict(boxstyle="round", fc="white", ec=GRID))
ax.legend(frameon=False, fontsize=8, loc="upper right"); ax.set_xlabel("RT (s)"); ax.set_ylabel("intensity")
ax.set_title(f"descriptors of one kept feature: m/z {demo['mz']:.4f}, RT {demo['rt']:.2f} min")
fig.tight_layout(); fig.savefig(OUT / "05_descriptors_gates.png"); plt.close(fig)

# ---- Fig 6: the chromatogram check: ok / noapex / lowsnr / dup examples
ex = {}
ex["ok"] = demo
for cls in ("noapex", "lowsnr", "dup"):
    rr = rej[rej["reason"] == f"eic:{cls}"]
    rr = rr[rr["height"].between(2e4, 5e5)] if len(rr) > 20 else rr
    if len(rr): ex[cls] = rr.sort_values("height", ascending=False).iloc[len(rr) // 3]
fig, axes = plt.subplots(1, len(ex), figsize=(4.3 * len(ex), 3.9))
for ax, (cls, f) in zip(np.atleast_1d(axes), ex.items()):
    sa_, lo_, hi_ = int(f["scan_apex"]), int(f.get("scan_lo", f.get("scan_min"))), int(f.get("scan_hi", f.get("scan_max")))
    Wc = int(10 * fs); c0, c1 = max(0, sa_ - Wc), min(cloud.n_scans, sa_ + Wc + 1)
    ppm_ = max(5.0, 3 * P.sig_ppm(f["height"]) + 1)
    e_ = cloud.eic(float(f["mz"]), ppm_, c0, c1).astype(float); es_ = gaussian_filter1d(e_, 0.5 * fs / 2.355, mode="nearest"); t_ = cloud.rt[c0:c1] * 60
    ax.plot(t_, e_, color=INK2, lw=0.7, label="raw EIC"); ax.plot(t_, es_, color=C1, lw=1.4, label="smoothed, kernel = half a peak width")
    ax.axvspan(cloud.rt[lo_] * 60, cloud.rt[hi_] * 60, color=C2, alpha=0.15, label="candidate bounds")
    b0, b1 = lo_ - c0, hi_ - c0
    k = b0 + int(np.argmax(es_[b0:b1 + 1])); top = es_[k]
    j = k; lmin = top
    while j > 0 and es_[j - 1] <= top: j -= 1; lmin = min(lmin, es_[j])
    j = k; rmin = top
    while j < len(es_) - 1 and es_[j + 1] <= top: j += 1; rmin = min(rmin, es_[j])
    base = max(lmin, rmin); ax.vlines(t_[k], base, top, color=C4, lw=2, label=f"prominence {top - base:.3g}")
    outside = np.ones(len(e_), bool); outside[b0:b1 + 1] = False
    rr_ = (e_ - es_)[outside]; nres = 1.4826 * np.median(np.abs(rr_ - np.median(rr_))) if len(rr_) >= 10 else 0
    ax.fill_between(t_, es_ - max(nres, f["noise"]), es_ + max(nres, f["noise"]), color=C4, alpha=0.12, label=f"noise band: max(residual {nres:.3g}, cell {f['noise']:.3g})")
    ax.set_title(f"{cls}: eic_snr {f.get('eic_snr', np.nan):.1f}" + ("  (kept)" if cls == "ok" else "  (rejected)"), color=C5 if cls == "ok" else C4)
    ax.set_xlabel("RT (s)"); ax.legend(frameon=False, fontsize=6.5)
axes[0].set_ylabel("intensity")
fig.suptitle("the 1D chromatogram check: the smoothed EIC must peak inside the bounds, 3x above the noise, and not duplicate a stronger same-ion maximum", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT / "06_chromatogram_check.png"); plt.close(fig)

# ---- Fig 7: the seven score terms for a strong clean, a weak clean and a score-rejected candidate
def terms(f):
    t_snr = np.clip((np.log2(max(f["snr"], 1e-9)) - np.log2(3)) / (np.log2(50) - np.log2(3)), 0, 1)
    return dict(snr=t_snr, shape=np.clip(f["gauss_r2"], 0, 1), mz=np.clip(1 - f["mz_sd_ppm"] / (3 * P.sig_ppm(f["height"]) + 1), 0, 1),
                prominence=np.clip(f["prominence_rel"], 0, 1), scans=np.clip(f["n_scans"] / fs, 0, 1), localisation=np.clip(1 - f["far_level"], 0, 1),
                unimodality=np.clip(1 / max(f["tv_ratio"], 1e-9), 0, 1))
weak = F[(F["score"].between(0.5, 0.6)) & (F["height"] < 1e5)].sort_values("height").iloc[len(F[(F["score"].between(0.5, 0.6)) & (F["height"] < 1e5)]) // 2]
srej = rej[rej["reason"] == "score"]; srej = srej.sort_values("score", ascending=False).iloc[len(srej) // 4]
fig, ax = plt.subplots(figsize=(11, 3.8))
names = list(terms(demo)); xpos = np.arange(len(names))
for k, (lab, f, col) in enumerate((("strong clean feature, score %.2f" % demo["score"], demo, C1), ("weak clean feature, score %.2f" % weak["score"], weak, C5), ("rejected candidate, score %.2f" % srej["score"], srej, C4))):
    v = terms(f); ax.bar(xpos + (k - 1) * 0.27, [v[n] for n in names], width=0.27, color=col, label=lab)
ax.set_xticks(xpos); ax.set_xticklabels(names); ax.set_ylabel("term (0 to 1)"); ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=8)
ax.set_title("the rule score = geometric mean of seven terms (+0.15 with an isotope partner); kept at >= 0.5")
fig.tight_layout(); fig.savefig(OUT / "07_score_terms.png"); plt.close(fig)

# ---- Fig 8: grouping, confirmation and the presence rule across the 3 injections
d = ROOT / "results/peak3d" / run / "peak3d_out"
PM = pd.read_csv(d / "presence_mask.tsv", sep="\t")
clouds = [cloud] + [load_cloud(f) for f in files[1:]]
warps = [TableWarp(pd.read_csv(d / "rt_correction" / f"{s}.tsv", sep="\t")["rt_native"].values, pd.read_csv(d / "rt_correction" / f"{s}.tsv", sep="\t")["rt_corr"].values) for s in stems]
feats_run = [pd.read_csv(d / "features" / f"{s}.tsv", sep="\t") for s in stems]
def pick_group(cond):
    g = PM[cond & PM["mz"].between(150, 900) & PM["rt"].between(1, 5)]
    return g.iloc[len(g) // 2]
cases = [("detected in all 3", pick_group((PM["n_detected"] == 3))),
         ("detected in 2, confirmed in the third", pick_group((PM["n_detected"] == 2) & (PM["n_confirmed"] == 1))),
         ("detected in 1, confirmed in 1 -> kept", pick_group((PM["n_detected"] == 1) & (PM["n_confirmed"] == 1))),
         ("detected in 1, confirmed in none -> dropped", pick_group((PM["n_detected"] == 1) & (PM["n_confirmed"] == 0)))]
fig, axes = plt.subplots(1, 4, figsize=(17, 4.0))
cols = [C1, C2, C3]
for ax, (title, g) in zip(axes, cases):
    rt_c = g["rt"]; W6 = 8 * P.fwhm_med
    for ci in range(3):
        c = clouds[ci]; wp = warps[ci]
        s0_ = int(np.searchsorted(c.rt, wp.inverse(np.array([rt_c - W6]))[0])); s1_ = int(np.searchsorted(c.rt, wp.inverse(np.array([rt_c + W6]))[0]))
        x = wp.forward(c.rt[s0_:s1_]) * 60; e_ = c.eic(float(g["mz"]), 8.0, s0_, s1_)
        status = {2: "detected", 1: "confirmed", 0: "absent"}[int(g[stems[ci]])]
        ax.plot(x, e_, color=cols[ci], lw=1.1, label=f"file {'CGK'[ci]}: {status}")
        if int(g[stems[ci]]) == 2:
            f = feats_run[ci][feats_run[ci]["group_id"] == g["group_id"]].iloc[0]
            ax.axvspan(f["rt_min_corr"] * 60, f["rt_max_corr"] * 60, color=cols[ci], alpha=0.08)
    ax.axvline(rt_c * 60, color=INK2, ls=":", lw=0.8)
    ax.set_title(f"{title}\nm/z {g['mz']:.4f}", fontsize=9); ax.set_xlabel("corrected RT (s)"); ax.legend(frameon=False, fontsize=7.5)
axes[0].set_ylabel("intensity")
fig.suptitle("one group across the three injections: detected = a feature of that file; confirmed = a peak found in the gap-filled chromatogram; presence rule: detected + confirmed >= 2", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT / "08_group_presence.png"); plt.close(fig)
# copy QC figures
import shutil
for q in ("drift_curves.png", "anchor_residuals.png", "presence_hist.png"):
    if (d / "qc" / q).exists(): shutil.copy(d / "qc" / q, OUT / f"qc_{q}")
# stage counts for the funnel
json.dump(dict(centroids=int(cloud.n_points), scans=int(cloud.n_scans), basins_initial=int(b.n_basins_initial), after_merge=int(b.n_after_merge),
               candidates=int(len(b.basin_id)), passed_gates=int((rej["reason"] != "gate").sum() + len(feats)), kept=int(len(feats)),
               rejected=rej["reason"].value_counts().to_dict(), dt_s=60 * P.dt, fwhm_s=60 * P.fwhm_med, fwhm_scans=P.fwhm_scans, sig_a=P.sig_a, sig_b=P.sig_b,
               floor=P.floor, noise_c=P.noise_c, noise_r=P.noise_r, groups_all=int(len(PM)), groups_kept=int(((PM["n_detected"] + PM["n_confirmed"]) >= 2).sum()),
               demo=dict(mz=float(demo["mz"]), rt=float(demo["rt"]))), open(OUT / "stage_counts.json", "w"), indent=1)
print("figures written to", OUT)
