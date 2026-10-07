# diagnostic used while tuning peak3d (reads results/peak3d and the labels; report-only)
import numpy as np, pandas as pd, glob
tr = pd.read_csv(glob.glob("data/raw/ASARI_DATA/x/*/data/hzv029_manual_certified.txt")[0], sep="\t")
def nearest(mz, rt, qm, qr, ppm=5, rt_tol=0.1):
    o = np.argsort(mz); smz = mz[o]; out = np.full(len(qm), -1)
    for i, (m, t) in enumerate(zip(qm, qr)):
        lo = np.searchsorted(smz, m * (1 - ppm * 1e-6)); hi = np.searchsorted(smz, m * (1 + ppm * 1e-6), side="right")
        k = o[lo:hi]; k = k[np.abs(rt[k] - t) <= rt_tol]
        if len(k): out[i] = k[np.argmin(np.abs(rt[k] - t))]
    return out
# peak3d
t = pd.read_csv("results/peak3d/HZV029_full/peak3d_out/aligned_feature_table.tsv", sep="\t", usecols=["group_id", "mz", "rt", "n_detected", "n_filled"])
N = 268
print("peak3d HZV029_full groups", len(t))
nd = t["n_detected"].values
for q in [1, 2, 5, 27, 54, 134, 214, 268]:
    print(f"  groups with n_detected >= {q:3d}: {(nd >= q).sum():7d}   features in them: {nd[nd >= q].sum():8d} ({nd[nd >= q].sum() / nd.sum():.1%} of all features)")
k = nearest(t["mz"].values, t["rt"].values, tr["moverz"].values, tr["RT_minutes"].values)
hit = k >= 0
print(f"  certified features matched: {hit.sum()}/{len(tr)}; their n_detected: median {np.median(nd[k[hit]]):.0f}, p10 {np.percentile(nd[k[hit]], 10):.0f}, min {nd[k[hit]].min()}, >=80%: {(nd[k[hit]] >= 0.8 * N).mean():.1%}")
# how many groups per certified feature within 5 ppm / 0.2 min (split groups?)
o = np.argsort(t["mz"].values); smz = t["mz"].values[o]
multi = []
for m, r in zip(tr["moverz"].values, tr["RT_minutes"].values):
    lo = np.searchsorted(smz, m * (1 - 5e-6)); hi = np.searchsorted(smz, m * (1 + 5e-6), side="right")
    kk = o[lo:hi]; kk = kk[np.abs(t["rt"].values[kk] - r) <= 0.2]
    multi.append((len(kk), nd[kk].sum() if len(kk) else 0))
multi = np.array(multi)
print(f"  groups within 5 ppm/0.2 min of a certified feature: median {np.median(multi[:, 0]):.0f}, p90 {np.percentile(multi[:, 0], 90):.0f}; summed detections median {np.median(multi[:, 1]):.0f}")
# masscube
m = pd.read_csv("results/masscube/HZV029_full/aligned_feature_table.txt", sep="\t", low_memory=False)
print("\nmasscube groups", len(m), "detection_rate >= 0.8:", (m["detection_rate"] >= 0.8).mean().round(3))
k = nearest(m["m/z"].values, m["RT"].values, tr["moverz"].values, tr["RT_minutes"].values)
hit = k >= 0
dr = m["detection_rate"].values
print(f"  certified matched {hit.sum()}/{len(tr)}; detection_rate median {np.median(dr[k[hit]]):.2f}, p10 {np.percentile(dr[k[hit]], 10):.2f}, >=0.8: {(dr[k[hit]] >= 0.8).mean():.1%}")
# asari
a = pd.read_csv(glob.glob("results/asari/HZV029_full/asari_out*/export/full_Feature_table.tsv")[0], sep="\t")
print("\nasari groups", len(a), "detection_counts >= 0.8N:", (a["detection_counts"] >= 0.8 * N).mean().round(3))
k = nearest(a["mz"].values, a["rtime"].values / 60, tr["moverz"].values, tr["RT_minutes"].values)
hit = k >= 0
dc = a["detection_counts"].values
print(f"  certified matched {hit.sum()}/{len(tr)}; detection_counts median {np.median(dc[k[hit]]):.0f}, p10 {np.percentile(dc[k[hit]], 10):.0f}, >=80%: {(dc[k[hit]] >= 0.8 * N).mean():.1%}")

# --- example dump: all groups near three certified features (peak3d) ---
print("\n=== example certified features: groups within 5 ppm / 0.2 min (peak3d) ===")
pd.set_option("display.width", 200)
for q in [50, 150, 300]:
    m, r = tr["moverz"].values[q], tr["RT_minutes"].values[q]
    lo = np.searchsorted(smz, m * (1 - 5e-6)); hi = np.searchsorted(smz, m * (1 + 5e-6), side="right")
    kk = o[lo:hi]; kk = kk[np.abs(t["rt"].values[kk] - r) <= 0.2]
    sub = t.iloc[kk].sort_values("rt")
    print(f"truth mz {m:.4f} rt {r:.3f}  intensity {tr['Intensity'].values[q]:.0f}")
    print(sub[["group_id", "mz", "rt", "n_detected", "n_filled"]].assign(dppm=((sub["mz"] - m) / m * 1e6).round(2), drt_s=((sub["rt"] - r) * 60).round(2)).to_string(index=False))
# per-file feature rows for the first example in 3 files: how many features of that ion per file?
f0 = pd.read_csv("results/peak3d/HZV029_full/peak3d_out/features/batch12_MT_20210806_177.tsv", sep="\t")
m, r = tr["moverz"].values[50], tr["RT_minutes"].values[50]
sel = f0[(np.abs(f0["mz"] - m) / m * 1e6 <= 10) & (np.abs(f0["rt_corr"] - r) <= 0.3)]
print("\nmedoid file features near example 1:")
print(sel[["feature_id", "mz", "rt", "rt_corr", "height", "n_scans", "fwhm", "snr", "score", "group_id", "flags"]].to_string(index=False))
