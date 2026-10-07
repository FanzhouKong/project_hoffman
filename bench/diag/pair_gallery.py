# figures: same-ion neighbour pairs inside one file (EIC gallery + basin-coloured point clouds)
import sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/quobyte/metabolomicsgrp/fanzhou/hoffmann")
from peak3d.io import load_cloud
from peak3d.pick import pick_cloud
from peak3d.features import basins_to_frame, gate, mark_fragments, ridge_ratio, rule_score
ROOT = "/quobyte/metabolomicsgrp/fanzhou/hoffmann"
stem, mzml = "MT_20211013_082", f"{ROOT}/data/mzml/BM21/MT_20211013_082.mzML"
OUT = f"{ROOT}/results/figures"
INK, INK2, GRID = "#1f2328", "#59636e", "#d0d7de"
CA, CB, CO = "#0969da", "#bf8700", "#b1b8c0"      # strong feature, weak feature, other centroids

cloud = load_cloud(mzml)
b = pick_cloud(cloud); P = b.params
df = basins_to_frame(b); df["fragment"] = mark_fragments(df, cloud, P)
for _k, _v in ridge_ratio(df, cloud, P).items():
    df[_k] = _v
cand = df[gate(df, P)].copy(); cand["score"] = rule_score(cand, P)
kept = cand[cand["score"] >= P.min_score].reset_index(drop=True)
scan_of = cloud.scan_index()
print(f"{stem}: {len(kept)} kept features; dt {60*P.dt:.2f} s, FWHM_med {60*P.fwhm_med:.2f} s")

mz, rt, h = kept["mz"].values, kept["rt"].values, kept["height"].values
o = np.argsort(mz); smz = mz[o]
pairs = []
for i in range(len(kept)):
    lo = np.searchsorted(smz, mz[i] * (1 - 5e-6)); hi = np.searchsorted(smz, mz[i] * (1 + 5e-6), side="right")
    for j in o[lo:hi]:
        if j <= i or abs(rt[j] - rt[i]) * 60 > 6 or abs(mz[j] - mz[i]) / mz[i] * 1e6 > 2:
            continue
        a, bb = (i, j) if h[i] >= h[j] else (j, i)
        sa, sb = int(kept["scan_apex"][a]), int(kept["scan_apex"][bb])
        s0, s1 = min(sa, sb), max(sa, sb) + 1
        e = cloud.eic(float(mz[a]), 10.0, s0, s1)
        valley = float(e.min()) / float(h[bb]) if len(e) else np.nan
        pairs.append(dict(a=a, b=bb, drt_s=60 * (rt[bb] - rt[a]), valley=valley, h_ratio=h[bb] / h[a], zeros=int((e == 0).sum())))
R = pd.DataFrame(pairs)
print(f"pairs: {len(R)}; valley ratio quantiles {np.nanpercentile(R['valley'], [10, 50, 90]).round(2).tolist()}; with zero scans between: {(R['zeros'] > 0).mean():.1%}")
R["cat"] = np.where(R["zeros"] > 0, "flicker (empty scans between)", np.where(R["valley"] < 0.5, "resolved pair (valley < 50 %)", "shallow valley (not adjacent basins)"))
print(R["cat"].value_counts().to_dict())
rng = np.random.default_rng(3)
sel = []
for cat, n in [("resolved pair (valley < 50 %)", 7), ("flicker (empty scans between)", 3), ("shallow valley (not adjacent basins)", 2)]:
    idx = R.index[R["cat"] == cat].values
    if len(idx):
        # spread over intensity: sort by strong height and take evenly spaced
        idx = idx[np.argsort(-h[R.loc[idx, "a"].values])]
        sel += list(idx[np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)])
sel = sel[:12]

def style(ax, title):
    ax.set_title(title, loc="left", fontsize=8, color=INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=7); ax.grid(True, color=GRID, lw=0.5, alpha=0.6); ax.set_axisbelow(True)

# ---- figure 1: EIC gallery ----
fig, axes = plt.subplots(3, 4, figsize=(15, 11), dpi=150)
for ax, k in zip(axes.ravel(), sel):
    r = R.loc[k]; a, bb = int(r["a"]), int(r["b"])
    t0 = min(rt[a], rt[bb]) - 8 / 60; t1 = max(rt[a], rt[bb]) + 8 / 60
    s0, s1 = int(np.searchsorted(cloud.rt, t0)), int(np.searchsorted(cloud.rt, t1))
    e = cloud.eic(float(mz[a]), 10.0, s0, s1); t = 60 * (cloud.rt[s0:s1] - rt[a])
    ax.plot(t, e, color=INK2, lw=1.0, marker="o", ms=2.5, mfc=INK2, mec="none")
    top = e.max()
    for idx_, col, lab in [(a, CA, "A strong"), (bb, CB, "B weak")]:
        ax.axvline(60 * (rt[idx_] - rt[a]), color=col, lw=1.2, alpha=0.9)
        ax.hlines(-0.06 * top if col == CA else -0.12 * top, 60 * (kept["rt_min"][idx_] - rt[a]), 60 * (kept["rt_max"][idx_] - rt[a]), color=col, lw=3)
        ax.annotate(lab.split()[0], (60 * (rt[idx_] - rt[a]), top * 1.02), color=col, fontsize=8, ha="center", fontweight="bold")
    ax.set_ylim(-0.16 * top, 1.12 * top)
    short = {"resolved pair (valley < 50 %)": "resolved pair", "flicker (empty scans between)": "flicker",
             "shallow valley (not adjacent basins)": "shallow valley"}[r["cat"]]
    style(ax, f"m/z {mz[a]:.4f} \u00b7 RT {rt[a]:.2f} min \u00b7 {short}\n"
              f"\u0394RT {r['drt_s']:+.1f} s ({abs(r['drt_s'])/(60*P.fwhm_med):.1f} FWHM) \u00b7 B/A {r['h_ratio']:.2f} \u00b7 valley {r['valley']:.2f} of B")
    ax.set_xlabel("RT relative to A (s)", color=INK2, fontsize=7); ax.set_ylabel("intensity", color=INK2, fontsize=7)
for ax in axes.ravel()[len(sel):]:
    ax.axis("off")
fig.suptitle(f"Same-ion neighbour pairs kept as two features, {stem} (BM21 RP)\n"
             f"line: extracted ion chromatogram at \u00b110 ppm; vertical lines: apexes; bars: RT bounds of A (blue, stronger) and B (orange, weaker)",
             fontsize=9, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.subplots_adjust(hspace=0.6, wspace=0.3)
fig.savefig(f"{OUT}/same_ion_neighbours_eic.png"); plt.close(fig)

# ---- figure 2: point clouds for three pairs, coloured by basin ----
cloud_mz = cloud.mz; cloud_in = cloud.inten; lab = b.lab
basin_of_feature = kept["basin_id"].values
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=150)
for ax, k in zip(axes, [sel[0], sel[3], sel[7]] if len(sel) > 7 else sel[:3]):
    r = R.loc[k]; a, bb = int(r["a"]), int(r["b"])
    t0 = min(rt[a], rt[bb]) - 8 / 60; t1 = max(rt[a], rt[bb]) + 8 / 60
    s0, s1 = int(np.searchsorted(cloud.rt, t0)), int(np.searchsorted(cloud.rt, t1))
    p0, p1 = cloud.off[s0], cloud.off[s1]
    pts = np.arange(p0, p1)
    pts = pts[np.abs(cloud_mz[pts] - mz[a]) / mz[a] * 1e6 <= 15]
    x = 60 * (cloud.rt[scan_of[pts]] - rt[a]); y = (cloud_mz[pts] - mz[a]) / mz[a] * 1e6
    size = 6 + 40 * np.log10(np.maximum(cloud_in[pts], 1)) / np.log10(max(cloud_in[pts].max(), 10))
    isA = lab[pts] == basin_of_feature[a]; isB = lab[pts] == basin_of_feature[bb]
    ax.scatter(x[~isA & ~isB], y[~isA & ~isB], s=size[~isA & ~isB], color=CO, label="other basins", linewidths=0)
    ax.scatter(x[isA], y[isA], s=size[isA], color=CA, label="basin of A", linewidths=0)
    ax.scatter(x[isB], y[isB], s=size[isB], color=CB, label="basin of B", linewidths=0)
    ax.axhline(0, color=GRID, lw=0.8)
    style(ax, f"m/z {mz[a]:.4f} \u00b7 RT {rt[a]:.2f} min \u00b7 \u0394RT {r['drt_s']:+.1f} s \u00b7 valley {r['valley']:.2f} of B\n"
              f"centroids within \u00b115 ppm; marker size ~ log intensity")
    ax.set_xlabel("RT relative to A (s)", color=INK2, fontsize=7); ax.set_ylabel("m/z offset from A (ppm)", color=INK2, fontsize=7)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
fig.suptitle("The same pairs as a point cloud: every centroid climbs to one apex; blue and orange are the two basins the picker kept",
             fontsize=9, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{OUT}/same_ion_neighbours_pointcloud.png"); plt.close(fig)
print("written", f"{OUT}/same_ion_neighbours_eic.png", f"{OUT}/same_ion_neighbours_pointcloud.png")
