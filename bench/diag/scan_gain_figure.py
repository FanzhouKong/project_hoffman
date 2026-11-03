# figure: the scan-gain idea on real data. Three strong co-eluting ions of one file as recorded, the per-scan
# coefficient peak3d estimates (estimate.scan_gains), and the same ions after dividing every scan by it.
# Picks a window where >= 3 broad peaks of distinct compounds overlap; writes a PNG for a slide.
# usage: scan_gain_figure.py RUN OUT_PNG
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.runs import RUNS  # noqa: E402
from peak3d.estimate import SEED_PPM, estimate, scan_gains, strongest_distinct  # noqa: E402
from peak3d.io import load_cloud  # noqa: E402

run, out = sys.argv[1], Path(sys.argv[2])
cloud = load_cloud(RUNS[run]["files"][0])
P = estimate(cloud)
g, sd, n = scan_gains(cloud, P)
print(f"{run}: gain sd {sd:.3f} over {n} scans")
S = cloud.n_scans

# broad, strong peaks: apex scan, scans above half max, m/z
seeds = strongest_distinct(cloud, 600)
cand = []
for p in seeds:
    s = int(np.searchsorted(cloud.off, p, side="right") - 1)
    e = cloud.eic(cloud.mz[p], SEED_PPM, max(0, s - 40), min(S, s + 41)).astype(float)
    k = int(np.argmax(e))
    half = int((e >= 0.5 * e[k]).sum())
    if half >= 10:
        cand.append((max(0, s - 40) + k, float(cloud.mz[p]), float(e[k]), half))
cand.sort()


def rough(y):
    """median |second difference| of log intensity over the scans >= 20 % of the trace maximum"""
    y = np.asarray(y, float)
    m = y >= 0.2 * y.max()
    ly = np.log(np.maximum(y, 1e-9))
    d2 = np.abs(ly[2:] - 2 * ly[1:-1] + ly[:-2])[m[1:-1] & m[2:] & m[:-2]]
    return np.median(d2) if len(d2) >= 5 else np.nan


# an illustrative window: three distinct compounds whose apexes fall within 4 scans of each other, each a single peak
# in the window, chosen where dividing by the coefficients smooths all three the most
best = None
for c, *_rest in cand:
    if c - 20 < 2 or c + 21 > S - 2:
        continue
    grp = sorted([x for x in cand if abs(x[0] - c) <= 4], key=lambda x: -x[2])
    picked = []
    for x in grp:
        if all(abs(x[1] - y[1]) > 2.5 for y in picked):
            picked.append(x)
    if len(picked) < 3:
        continue
    picked = picked[:3]
    a, b = c - 20, c + 21
    gains = []
    for _, mz, _, _ in picked:
        r = cloud.eic(mz, 5.0, a, b).astype(float)
        single = int(np.argmax(r)) in range(14, 27) and r[0] < 0.3 * r.max() and r[-1] < 0.3 * r.max()
        before, after = rough(r), rough(r / g[a:b])
        gains.append(before / after if single and after > 0 else np.nan)
    if np.all(np.isfinite(gains)) and min(gains) > 1:
        score = float(np.mean(gains))
        if best is None or score > best[0]:
            best = (score, c, picked)
score, c, ions = best
print(f"window centre scan {c}: roughness before / after = {score:.2f}")
s0, s1 = c - 20, c + 21
t = cloud.rt[s0:s1]
raw = [cloud.eic(mz, 5.0, s0, s1).astype(float) for _, mz, _, _ in ions]
cor = [r / g[s0:s1] for r in raw]
norm = [x.max() for x in cor]
gw = g[s0:s1]
hl = np.argsort(-np.abs(np.log(gw)))[:3]
print("window", s0, s1, "ions", [round(x[1], 4) for x in ions], "coefficients", np.round(gw, 3).tolist())

INK, INK2, GRID, BAND = "#1b1b19", "#52514e", "#e4e2dc", "#efece5"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3 (validated all-pairs)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 24, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 1.2})
fig, ax = plt.subplots(3, 1, figsize=(11, 9.2), dpi=200, sharex=True, gridspec_kw={"height_ratios": [3, 1.6, 3]})
for a in ax:
    for j in hl:
        a.axvspan(t[j] - 0.5 * np.median(np.diff(t)), t[j] + 0.5 * np.median(np.diff(t)), color=BAND, zorder=0)
    a.grid(axis="y", color=GRID, lw=1)
    a.spines[["top", "right"]].set_visible(False)
for k, (r, x) in enumerate(zip(raw, cor)):
    lab = f"m/z {ions[k][1]:.2f}"
    ax[0].plot(t, r / norm[k], color=SERIES[k], lw=2.2, marker="o", ms=5, label=lab)
    ax[2].plot(t, x / norm[k], color=SERIES[k], lw=2.2, marker="o", ms=5, label=lab)
ax[1].bar(t, gw - 1, bottom=1, width=0.75 * np.median(np.diff(t)), color=INK2)
ax[1].axhline(1, color=INK, lw=1.2)
ax[0].set_ylabel("relative\nintensity", fontsize=23)
ax[2].set_ylabel("relative\nintensity", fontsize=23)
ax[1].set_ylabel("coef.", fontsize=23)
ax[0].set_title("As recorded: ions jump and dip together", loc="left", fontsize=25, color=INK)
ax[1].set_title("Estimated coefficient of each scan", loc="left", fontsize=25, color=INK)
ax[2].set_title("Corrected: each scan ÷ its coefficient", loc="left", fontsize=25, color=INK)
ax[2].set_xlabel("retention time (min)")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=23, bbox_to_anchor=(0.55, 1.0))
lo, hi = float(gw.min()), float(gw.max())
ax[1].set_ylim(min(0.8, lo - 0.03), max(1.2, hi + 0.03))
for a in (ax[0], ax[2]):
    a.set_ylim(0, 1.35)
fig.tight_layout(h_pad=1.2, rect=(0, 0, 1, 0.95))
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor="#fbfbf8")
print("wrote", out)
