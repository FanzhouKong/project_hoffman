# diagnostic: EIC gallery of peak3d features by category (what do the extra features look like?)
# Each panel: the feature's file (black) and the other files of the run (colour, native RT), the
# feature's bounds (shaded), its noise level and 3x noise (dashed), other peak3d features of the
# same ion in that file (grey ticks), and features of asari (a) / MassCube (m) / local tools (first letter) within
# 10 ppm (letters on top). Needs results/diag_excess/<RUN>_groups.tsv from feature_excess.py.
# usage: excess_gallery.py RUN CATEGORY [N=36] [SEED=0] [FILE_INDEX=0]
#   CATEGORY: single (1 file, no other tool), lowsnr_sup (snr < 10, all files, supported),
#             split (has a stronger same-ion feature within 3 FWHM), narrow (fwhm < 0.6 median),
#             unsup (any unsupported), random
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.runs import LOCAL_TOOLS, RUNS  # noqa: E402
from bench.score import load  # noqa: E402
from peak3d.io import load_cloud  # noqa: E402

run, cat = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 36
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
fi = int(sys.argv[5]) if len(sys.argv) > 5 else 0
d = ROOT / "results/peak3d" / run / "peak3d_out"
files = RUNS[run]["files"][:6]
stems = [Path(f).stem for f in files]
stem = stems[fi]
G = pd.read_csv(ROOT / f"results/diag_excess/{run}_groups.tsv", sep="\t")
nf = len(RUNS[run]["files"])
F = pd.read_csv(d / "features" / f"{stem}.tsv", sep="\t")
P = json.load(open(d / "features" / f"{stem}.params.json"))
fw = P["fwhm_med"]
F = F.merge(G[["group_id", "n_detected", "supported"]], on="group_id", how="left")
mz, rt, h = F["mz"].values, F["rt"].values, F["height"].values
o = np.argsort(mz)
smz = mz[o]
lo5 = np.searchsorted(smz, mz * (1 - 5e-6))
hi5 = np.searchsorted(smz, mz * (1 + 5e-6), side="right")
split = np.zeros(len(F), bool)
for i in range(len(F)):
    k = o[lo5[i]:hi5[i]]
    split[i] = np.any((k != i) & (np.abs(rt[k] - rt[i]) <= 3 * fw) & (h[k] > h[i]))
sel = {
    "single": (F["n_detected"] == 1) & ~F["supported"],
    "lowsnr_sup": (F["snr"] < 10) & (F["n_detected"] == nf) & F["supported"],
    "split": pd.Series(split),
    "narrow": F["fwhm"] < 0.6 * fw,
    "unsup": ~F["supported"],
    "random": pd.Series(np.ones(len(F), bool)),
} if not (cat.startswith(("cls:", "q:")) or cat == "lost") else {}
if cat.startswith(("cls:", "q:")) or cat == "lost":
    X = pd.read_csv(ROOT / f"results/diag_excess/{run}_{'lost' if cat == 'lost' else 'classes'}.tsv", sep="\t")
    X = X[X["stem"] == stem]
    if cat.startswith("cls:"):
        X = X[X["cls"] == cat[4:]]
    if cat.startswith("q:"):
        X = X.query(cat[2:])
    idx = np.unique(X["row"].values.astype(int))
else:
    idx = np.flatnonzero(sel[cat].values)
rng = np.random.default_rng(seed)
pick = np.sort(rng.choice(idx, min(N, len(idx)), replace=False))
print(f"{run} {stem} {cat}: {len(idx)} of {len(F)} features ({len(idx) / len(F):.1%}); plotting {len(pick)}")
clouds = [load_cloud(f) for f in files]
others = {}
for tool, letter in (("asari", "a"), ("masscube", "m"), *((t, t[0]) for t in LOCAL_TOOLS)):
    x = load(tool, run)
    if x is not None:
        others[letter] = x[0]
cols = 6
rows = int(np.ceil(len(pick) / cols))
fig, axs = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.5 * rows), squeeze=False)
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#9b59b6", "#d4a017"]
for ax, i in zip(axs.flat, pick):
    f = F.iloc[i]
    ppm = max(5.0, 3.0 * np.sqrt(P["sig_a"] ** 2 + P["sig_b"] ** 2 / max(f["height"], 1.0)))
    half = max(10 * fw, 2.0 * (f["rt_max"] - f["rt_min"]))
    t0, t1 = f["rt"] - half, f["rt"] + half
    for k, c in enumerate(clouds):
        s0, s1 = int(np.searchsorted(c.rt, t0)), int(np.searchsorted(c.rt, t1))
        e = c.eic(float(f["mz"]), ppm, s0, s1)
        if k == fi:
            ax.plot(c.rt[s0:s1], e, color="k", lw=1.0, marker=".", ms=2, zorder=3)
        else:
            ax.plot(c.rt[s0:s1], e, color=COL[(k - (k > fi)) % len(COL)], lw=0.8, alpha=0.7, zorder=2)
    ax.axvspan(f["rt_min"], f["rt_max"], color="#f5c518", alpha=0.25, zorder=1)
    ax.axvline(f["rt_apex"], color="#c0392b", lw=0.6)
    ax.axhline(f["noise"], color="grey", ls=":", lw=0.7)
    ax.axhline(3 * f["noise"] + f["baseline"], color="grey", ls="--", lw=0.7)
    k = o[lo5[i]:hi5[i]]
    k = k[(k != i) & (np.abs(rt[k] - f["rt"]) <= half)]
    ymax = ax.get_ylim()[1]
    for j in k:
        ax.plot([F["rt_min"].values[j], F["rt_max"].values[j]], [0.97 * ymax] * 2, color="grey", lw=2)
    for n, (letter, of) in enumerate(others.items()):
        m = (np.abs(of["mz"].values - f["mz"]) <= f["mz"] * 10e-6) & (of["rt"].values >= t0) & (of["rt"].values <= t1)
        for r in of["rt"].values[m]:
            ax.text(r, ymax * (0.88 - 0.09 * n), letter, color=["#2a78d6", "#1baf7a", "#eb6834"][n], fontsize=8,
                    ha="center", fontweight="bold")
    ax.set_xlim(t0, t1)
    ax.set_title(f"{f['mz']:.4f} @{f['rt']:.2f}  snr {f['snr']:.1f} sc {f['n_scans']} fw {60 * f['fwhm']:.1f}s\n"
                 f"score {f['score']:.2f} r2 {f['gauss_r2']:.2f} det {f['n_detected']}/{nf} "
                 f"{'S' if f['supported'] else '-'} iso{f['iso_offset']} {f['flags'] if isinstance(f['flags'], str) else ''}",
                 fontsize=7)
    ax.tick_params(labelsize=6)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(6)
for ax in list(axs.flat)[len(pick):]:
    ax.axis("off")
fig.suptitle(f"{run} {stem}  category={cat}  ({len(idx)} of {len(F)} features)  FWHM_med {60 * fw:.1f}s", fontsize=10)
fig.tight_layout()
tag = cat if not cat.startswith("q:") else "q" + str(abs(hash(cat)) % 10000)
out = ROOT / f"results/diag_excess/gallery_{run}_{tag.replace(':', '-')}_s{seed}_f{fi}.png"
fig.savefig(out, dpi=80)
print(out)
