# diagnostic (report-only): for every candidate near an IDSL003 label (results/diag_idsl/pairs.tsv), per-side
# chromatogram levels for three side-window definitions, zig-zag measured in units of the 3D noise surface,
# and an inspection of the TP labels whose nearest kept feature is 10-30 ppm off in m/z. Galleries for the
# "clean-looking" kept features at TN positions and for the m/z-offset cases. Reads the mzML -> srun.
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from peak3d.io import load_cloud
D = ROOT / "results/diag_idsl"
cloud = load_cloud(ROOT / "data/raw/IDSL_IPA/003.mzML")
P = json.load(open(D / "params.json"))
M = pd.read_csv(D / "pairs.tsv", sep="\t")
S = cloud.n_scans; dt = P["dt"]; fw_med = P["fwhm_med"]
C = M.drop_duplicates("cand_i").set_index("cand_i")
out = {}
def q(v, p):
    v = np.sort(v); return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]
for ci, f in C.iterrows():
    sa, slo, shi = int(f["scan_apex"]), int(f["scan_lo"]), int(f["scan_hi"])
    ppm = max(5.0, 3 * f["sig_apex"] + 1)
    h = max(f["height"], 1e-9); noise = max(f["noise"], 1e-9)
    rec = {}
    for wname, w in (("own", max(f["fwhm"], fw_med) / dt), ("med", fw_med / dt), ("cap2", min(max(f["fwhm"], fw_med), 2 * fw_med) / dt)):
        near = int(max(1, round(2 * w))); far = int(max(near + 2, round(6 * w)))
        for side, (a0, a1) in (("L", (sa - far, sa - near + 1)), ("R", (sa + near, sa + far + 1))):
            a0, a1 = max(0, a0), min(S, a1)
            if a1 - a0 < 3:
                rec[f"{wname}_{side}_n"] = a1 - a0; continue
            e = cloud.eic(float(f["mz"]), ppm, a0, a1).astype(float)
            rec[f"{wname}_{side}_n"] = a1 - a0
            rec[f"{wname}_{side}_q25"] = q(e, 0.25) / h; rec[f"{wname}_{side}_q50"] = q(e, 0.5) / h
            rec[f"{wname}_{side}_q75"] = q(e, 0.75) / h; rec[f"{wname}_{side}_q90"] = q(e, 0.9) / h
            rec[f"{wname}_{side}_pres"] = (e > 0).mean()
    # zig-zag in noise units inside the bounds
    e = cloud.eic(float(f["mz"]), ppm, slo, shi + 1).astype(float)
    tr = e[e > 0]
    if len(tr) >= 3:
        tv = np.abs(np.diff(tr)).sum(); prom = tr.max() - min(tr[0], tr[-1])
        rec["tv_excess_noise"] = max(tv - 2 * prom, 0.0) / (len(tr) * noise)
        # deep valleys: local minima at least k*noise below both neighbouring maxima
        lm = np.flatnonzero((tr[1:-1] < tr[:-2]) & (tr[1:-1] <= tr[2:])) + 1
        nd = 0
        for j in lm:
            left = tr[:j].max(); right = tr[j + 1:].max()
            if min(left, right) - tr[j] >= 3 * noise:
                nd += 1
        rec["n_deep_valleys"] = nd
        rec["n_deep_valleys_5"] = sum(1 for j in lm if min(tr[:j].max(), tr[j + 1:].max()) - tr[j] >= 5 * noise)
    out[ci] = rec
X = pd.DataFrame.from_dict(out, orient="index"); X.index.name = "cand_i"
X.to_csv(D / "sides.tsv", sep="\t", float_format="%.5g")
print("sides written", X.shape)

# ---- m/z-offset TP cases: nearest kept feature within 0.1 min is 10-30 ppm off
lab = pd.read_csv(ROOT / "data/raw/IDSL_IPA/idslipa_benchmarking_dataset.csv", encoding="utf-8-sig")
is_tp = (lab["Manual Curation"] == "TP").values; qm, qr = lab["m/z"].values, lab["RT(min)"].values
F = pd.read_csv(ROOT / "results/peak3d/IDSL003/peak3d_out/features/003.tsv", sep="\t")
NC = pd.read_csv(D / "no_candidate.tsv", sep="\t")
mz, rt, h = F["mz"].values, F["rt"].values, F["height"].values
o = np.argsort(mz); smz = mz[o]
rows = []
for i in NC.loc[NC["tp"], "label_i"].values:
    m, t = qm[i], qr[i]
    lo = np.searchsorted(smz, m * (1 - 30e-6)); hi = np.searchsorted(smz, m * (1 + 30e-6), side="right")
    k = o[lo:hi]; k = k[np.abs(rt[k] - t) <= 0.1]
    if len(k) == 0: continue
    j = k[np.argmax(h[k])]
    f = F.iloc[j]
    if abs((f["mz"] - m) / m * 1e6) <= 10: continue
    slo, shi = int(f["scan_min"]), int(f["scan_max"])
    # all centroids within +/- 40 ppm of the LABEL m/z over the feature's bounds
    pts_mz, pts_i, pts_s = [], [], []
    for s in range(slo, shi + 1):
        a, b = cloud.off[s], cloud.off[s + 1]
        l2 = np.searchsorted(cloud.mz[a:b], m * (1 - 40e-6)); h2 = np.searchsorted(cloud.mz[a:b], m * (1 + 40e-6), side="right")
        for p in range(a + l2, a + h2):
            pts_mz.append(cloud.mz[p]); pts_i.append(cloud.inten[p]); pts_s.append(s)
    pts_mz = np.array(pts_mz); pts_i = np.array(pts_i); pts_s = np.array(pts_s)
    dppm = (pts_mz - m) / m * 1e6
    near_lab = np.abs(dppm) <= 8; near_feat = np.abs((pts_mz - f["mz"]) / m * 1e6) <= 8
    rows.append(dict(label_i=i, label_mz=m, label_rt=t, feat_mz=f["mz"], feat_ppm=(f["mz"] - m) / m * 1e6, feat_h=f["height"], feat_mzsd=f["mz_sd_ppm"],
                     feat_mzmin_ppm=(f["mz_min"] - m) / m * 1e6, feat_mzmax_ppm=(f["mz_max"] - m) / m * 1e6, feat_nscans=f["n_scans"], feat_npts=f["n_points"],
                     n_cent=len(pts_mz), n_scans_lab=len(np.unique(pts_s[near_lab])), n_scans_feat=len(np.unique(pts_s[near_feat])),
                     sum_lab=pts_i[near_lab].sum(), sum_feat=pts_i[near_feat].sum(), max_lab=pts_i[near_lab].max() if near_lab.any() else 0,
                     cent_per_scan=len(pts_mz) / (shi - slo + 1), feat_snr=f["snr"], feat_score=f["score"]))
Z = pd.DataFrame(rows); Z.to_csv(D / "mzoff.tsv", sep="\t", index=False, float_format="%.6g")
print("\n== TP labels whose nearest kept feature (0.1 min) is 10-30 ppm off: %d" % len(Z))
print("feature ppm offset: sign + %d / - %d; |ppm| quantiles" % ((Z["feat_ppm"] > 0).sum(), (Z["feat_ppm"] < 0).sum()), Z["feat_ppm"].abs().quantile([.1, .5, .9]).round(1).to_dict())
print("feature mz_sd_ppm median %.2f (vs 1.5 for kept TPs); mz range of the feature covers the label m/z: %d of %d" % (
    Z["feat_mzsd"].median(), ((Z["feat_mzmin_ppm"] <= 0) & (Z["feat_mzmax_ppm"] >= 0)).sum(), len(Z)))
print("centroids per scan within +/-40 ppm of the label over the feature's bounds: median %.2f; scans with a centroid within 8 ppm of the LABEL: median %.0f; within 8 ppm of the FEATURE m/z: median %.0f" % (
    Z["cent_per_scan"].median(), Z["n_scans_lab"].median(), Z["n_scans_feat"].median()))
print("intensity near label / near feature m/z: median ratio %.2f; cases where the label-side ion is weaker than 1/3 of the feature-side ion: %d" % (
    (Z["sum_lab"] / Z["sum_feat"].clip(lower=1)).median(), (Z["sum_lab"] < Z["sum_feat"] / 3).sum()))

# ---- galleries
K = M[M["kept"]].drop_duplicates("cand_i").copy()
K["fwhm_rel"] = K["fwhm"] / fw_med
def panel(ax, f, title, extra_mz=None):
    sa, slo, shi = int(f["scan_apex"]), int(f["scan_lo"]), int(f["scan_hi"])
    W = int(max(10 * P["fwhm_scans"], 30)); s0, s1 = max(0, sa - W), min(S, sa + W + 1)
    ppm = max(5.0, 3 * f["sig_apex"] + 1)
    x = cloud.rt[s0:s1] * 60
    ax.plot(x, cloud.eic(float(f["mz"]), 30.0, s0, s1), color="#bbbbbb", lw=0.8)
    ax.plot(x, cloud.eic(float(f["mz"]), ppm, s0, s1), color="#0969da", lw=1.0, marker=".", ms=2.5)
    if extra_mz is not None:
        ax.plot(x, cloud.eic(float(extra_mz), 8.0, s0, s1), color="#cf222e", lw=0.9)
    ax.axvspan(cloud.rt[slo] * 60, cloud.rt[shi] * 60, color="#bf8700", alpha=0.15)
    ax.axhline(f["noise"], color="#cf222e", lw=0.6, ls="--")
    ax.set_title(title, fontsize=6.5, loc="left"); ax.tick_params(labelsize=5)
clean = K[~K["tp"] & (K["fwhm_rel"] < 3) & ~((K["fwhm_rel"] < 0.75) & (K["far_presence"] > 0.25)) & (K["n_lmax"] < 4) & (K["tv_ratio"] < 1.5) & (K["far_med_rel"] <= 0.4)]
pick = clean.sample(min(40, len(clean)), random_state=3)
fig, axes = plt.subplots(8, 5, figsize=(16, 19), dpi=110)
for ax, (_, f) in zip(axes.ravel(), pick.iterrows()):
    panel(ax, f, f"m/z {f['mz']:.4f} rt {f['rt']:.2f} h {f['height']:.0f} snr {f['snr']:.0f} sc {f['score']:.2f} iso {int(f['iso_support'])}\nfwhm {f['fwhm_rel']:.1f}x mzsd {f['mz_sd_ppm']:.1f} r2 {f['gauss_r2']:.2f} lmax {f['n_lmax']:.0f} tv {f['tv_ratio']:.2f} far {f['far_med_rel']:.2f} label d_rt {f['d_rt_s']:+.1f}s d_ppm {f['d_ppm']:+.1f}")
for ax in axes.ravel()[len(pick):]: ax.axis("off")
fig.suptitle("'clean-looking' kept features at TN positions (not plateau / spike / lumpy / background classes)", fontsize=9)
fig.tight_layout(); fig.savefig(D / "gallery_TN_clean.png"); plt.close(fig)
# m/z offset gallery: EIC at feature m/z (blue) and at label m/z (red, 8 ppm)
Fi = F.set_index(F.index)
pick = Z.sample(min(30, len(Z)), random_state=4)
fig, axes = plt.subplots(6, 5, figsize=(16, 14), dpi=110)
for ax, (_, z) in zip(axes.ravel(), pick.iterrows()):
    j = int(np.flatnonzero((F["mz"].values == z["feat_mz"]) & (F["height"].values == z["feat_h"]))[0])
    f = F.iloc[j].copy(); f["scan_lo"], f["scan_hi"] = f["scan_min"], f["scan_max"]; f["sig_apex"] = np.sqrt(P["sig_a"] ** 2 + P["sig_b"] ** 2 / f["height"])
    panel(ax, f, f"label m/z {z['label_mz']:.4f} rt {z['label_rt']:.2f}; feature m/z {z['feat_mz']:.4f} ({z['feat_ppm']:+.1f} ppm) h {z['feat_h']:.0f}\nmzsd {z['feat_mzsd']:.1f} range [{z['feat_mzmin_ppm']:+.0f},{z['feat_mzmax_ppm']:+.0f}] ppm, cent/scan {z['cent_per_scan']:.1f}, I(label side)/I(feat side) {z['sum_lab'] / max(z['sum_feat'], 1):.2f}", extra_mz=z["label_mz"])
    ax.axvline(z["label_rt"] * 60, color="#cf222e", lw=0.6, ls=":")
for ax in axes.ravel()[len(pick):]: ax.axis("off")
fig.suptitle("TP labels whose nearest kept feature is 10-30 ppm off: blue = EIC at the feature m/z, red = EIC at the label m/z (8 ppm), grey = 30 ppm", fontsize=9)
fig.tight_layout(); fig.savefig(D / "gallery_mzoff.png"); plt.close(fig)
print("galleries written")
