#!/usr/bin/env python3
"""EIC gallery for one peak3d run (report-only diagnostic).

For a few curated truth features: (1) each file's EIC alone on its native RT axis, (2) all files
overlaid on native RT, (3) all files overlaid on peak3d's corrected RT (results/peak3d/<run>/
peak3d_out/rt_correction/<stem>.tsv). Writes SVG + PNG per feature, selected.tsv, summary.json.

usage: eic_overlay.py RUN TRUTH_TSV OUTDIR [--ppm 5] [--window 1.2] [--n 6]
       [--bins 2.5-4.5,2.5-4.5,6-9,11-13,13-15,16-19]   (default: --n equal bins over the run)
       [--c13-shift]   (13C run: look for the truth ion at mz + n * 1.0033548)

Truth tables accepted: peakbench credentialed (mz, rt, tier, ...), HZV029 certified (moverz,
RT_minutes, Intensity), NetID yeast curation (medMz, medRt, Confidence), MTBLS733 maf
(mass_to_charge, retention_time). Columns are normalised to truth_id, mz, rt, height, tier, n.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.runs import RUNS  # noqa: E402
from peak3d.io import _scan_rt_minutes  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]      # validated categorical slots 1-3 (light)
FG, MUTED = "#010203", "#040506"                 # sentinels -> var(--fg) / var(--muted) in SVG


C13 = 1.0033548378


def normalise(truth):
    """map the known truth formats onto truth_id, mz, rt, height, tier, n and a 'good' filter"""
    t = truth.copy()
    if "moverz" in t:                                   # HZV029 certified list
        t = t.rename(columns={"moverz": "mz", "RT_minutes": "rt", "Intensity": "height"})
        t["truth_id"] = [f"HZV029_{i:03d}" for i in range(len(t))]
        t["tier"], t["n"], t["good"] = "cert", 0, True
    elif "medMz" in t:                                  # NetID manual curation
        t = t.rename(columns={"medMz": "mz", "medRt": "rt", "id": "truth_id"})
        t["height"] = t["maxQuality"] if "maxQuality" in t else 1.0
        if "Confidence" in t:
            t = t[t["Confidence"].astype(str).str.upper() == "TRUE"]
        t["tier"] = t["class"].astype(str) if "class" in t else "curated"
        t["n"], t["good"] = t["C"].fillna(0) if "C" in t else 0, True
    elif "mass_to_charge" in t:                         # MTBLS733 maf
        t = t.rename(columns={"mass_to_charge": "mz", "retention_time": "rt"})
        t["truth_id"] = t["database_identifier"].astype(str) if "database_identifier" in t else [f"LI_{i:04d}" for i in range(len(t))]
        t["height"], t["tier"], t["n"], t["good"] = 1.0, "standard", 0, True
    else:                                               # peakbench credentialed table
        t = t.rename(columns={"height12": "height"})
        good = (t["tier"] == "A") & (~t["ambiguous"].astype(bool))
        if "feature_class" in t:
            good &= t["feature_class"] == "primary"
        if "reps12" in t:
            good &= t["reps12"] == 3
        t["good"] = good
    return t.reset_index(drop=True)


def select(truth, bins, min_sep=0.5):
    t = truth[truth["good"]]
    chosen = []
    for lo, hi in bins:
        c = t[(t["rt"] >= lo) & (t["rt"] <= hi)].sort_values("height", ascending=False)
        for _, r in c.iterrows():
            if all(abs(r["rt"] - q["rt"]) >= min_sep for q in chosen):
                chosen.append(r)
                break
    return pd.DataFrame(chosen).reset_index(drop=True)


def read_file(path):
    from pyteomics import mzml
    rts, mzs, ints = [], [], []
    with mzml.read(str(path), use_index=False) as r:
        for s in r:
            if s.get("ms level") != 1:
                continue
            rts.append(_scan_rt_minutes(s))
            mzs.append(np.asarray(s["m/z array"], float))
            ints.append(np.asarray(s["intensity array"], float))
    rt = np.asarray(rts)
    if rt.max() > 200:
        rt = rt / 60.0
    return rt, mzs, ints


def eic(mzs, ints, mz, ppm):
    tol = mz * ppm * 1e-6
    out = np.zeros(len(mzs))
    for k, (m, i) in enumerate(zip(mzs, ints)):
        sel = np.abs(m - mz) <= tol
        if sel.any():
            out[k] = i[sel].sum()
    return out


def style():
    plt.rcParams.update({
        "font.size": 9, "font.family": "sans-serif", "svg.fonttype": "none",
        "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": MUTED, "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.grid": True,
        "lines.linewidth": 1.6, "legend.frameon": False, "figure.facecolor": "none", "axes.facecolor": "none",
        "savefig.facecolor": "none",
    })


def plot_feature(feat, stems, letters, rt_nat, rt_cor, eics, window, out_stem):
    truth_rt = float(feat["rt"])
    apex = {}
    for s in stems:
        m = np.flatnonzero(np.abs(rt_nat[s] - truth_rt) <= window)
        e = eics[s][m]
        # the local maximum nearest the truth RT (>= 20 % of the window maximum), so a doublet is
        # tracked at the same member in every file instead of jumping to whichever is taller
        is_max = np.r_[True, e[1:] >= e[:-1]] & np.r_[e[:-1] >= e[1:], True] & (e >= 0.2 * e.max()) & (e > 0)
        cand = m[is_max] if is_max.any() else m[[int(np.argmax(e))]]
        k = cand[np.argmin(np.abs(rt_nat[s][cand] - truth_rt))]
        apex[s] = dict(k=int(k), nat=float(rt_nat[s][k]), cor=float(rt_cor[s][k]), height=float(eics[s][k]))
    x0 = min(truth_rt - window, min(a["cor"] for a in apex.values()) - 0.3)
    x1 = max(truth_rt + window, max(a["cor"] for a in apex.values()) + 0.3)
    ymax = max(eics[s][(rt_nat[s] >= x0) & (rt_nat[s] <= x1)].max() for s in stems) * 1.1

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), constrained_layout=True)
    for j, s in enumerate(stems):   # row 1: one file per panel
        ax = axes[0, j]
        m = (rt_nat[s] >= x0) & (rt_nat[s] <= x1)
        ax.plot(rt_nat[s][m], eics[s][m], color=SERIES[j])
        ax.axvline(truth_rt, color=MUTED, ls="--", lw=0.8)
        ax.set_title(f"file {letters[s]}: {s}, native RT", loc="left", fontsize=9)
        ax.annotate(letters[s], (apex[s]["nat"], apex[s]["height"]), xytext=(4, 2), textcoords="offset points")
    for ax, key, ttl in [(axes[1, 0], "nat", "overlay, native RT"), (axes[1, 1], "cor", "overlay, corrected RT")]:
        for j, s in enumerate(stems):
            x = rt_nat[s] if key == "nat" else rt_cor[s]
            m = (x >= x0) & (x <= x1)
            ax.plot(x[m], eics[s][m], color=SERIES[j], label=f"file {letters[s]}")
            ax.annotate(letters[s], (apex[s][key], apex[s]["height"]), xytext=(4, 2), textcoords="offset points")
        ax.axvline(truth_rt, color=MUTED, ls="--", lw=0.8)
        ax.set_title(ttl, loc="left", fontsize=9)
        ax.legend(loc="upper right")
    ax = axes[1, 2]
    ax.axis("off")
    lines = ["file   apex native   apex corrected   shift"]
    for s in stems:
        a = apex[s]
        lines.append(f"  {letters[s]}     {a['nat']:7.3f} min    {a['cor']:7.3f} min   {60 * (a['cor'] - a['nat']):+6.1f} s")
    ax.text(0.0, 0.95, "\n".join(lines), family="monospace", fontsize=8.5, va="top", transform=ax.transAxes)
    ax.text(0.0, 0.45, f"truth: m/z {feat['mz']:.4f}, RT {truth_rt:.3f} min\ntier {feat['tier']}, "
            f"n carbons {int(float(feat['n']))}, truth_id {feat['truth_id']}\nEIC window +/- {args.ppm:g} ppm",
            fontsize=8.5, va="top", transform=ax.transAxes)
    for ax in axes.ravel()[:5]:
        ax.set_xlim(x0, x1)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("RT (min)")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0, 0].set_ylabel("intensity")
    axes[1, 0].set_ylabel("intensity")
    fig.suptitle(f"m/z {feat['mz']:.4f}   truth RT {truth_rt:.2f} min", x=0.01, ha="left", fontsize=10)
    fig.savefig(f"{out_stem}.png", dpi=110, facecolor="white")
    fig.savefig(f"{out_stem}.svg", transparent=True)
    plt.close(fig)
    svg = Path(f"{out_stem}.svg").read_text()
    svg = svg.replace(FG, "var(--fg)").replace(MUTED, "var(--muted)")
    svg = re.sub(r'(<svg[^>]*?)\s(width|height)="[^"]*"', r"\1", svg, count=2)
    Path(f"{out_stem}.svg").write_text(svg)
    return apex


ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("truth")
ap.add_argument("outdir")
ap.add_argument("--ppm", type=float, default=5.0)
ap.add_argument("--window", type=float, default=1.2)
ap.add_argument("--bins", default=None)
ap.add_argument("--n", type=int, default=6, help="features to show (equal RT bins unless --bins)")
ap.add_argument("--c13-shift", action="store_true", help="13C run: ion at mz + n * C13")
args = ap.parse_args()

out = Path(args.outdir)
out.mkdir(parents=True, exist_ok=True)
truth = normalise(pd.read_csv(args.truth, sep="\t", encoding="utf-8-sig"))
if args.c13_shift:
    truth["mz"] = truth["mz"] + truth["n"].astype(float) * C13
    truth["truth_id"] = truth["truth_id"].astype(str) + "+" + truth["n"].astype(int).astype(str) + "C13"

files = RUNS[args.run]["files"]
stems = [Path(f).stem for f in files]
# short file letters: the stem's trailing letter/digit group when it is unique, else a, b, c, ...
suffix = [re.sub(r"^.*?([A-Za-z0-9]{1,4})$", r"\1", s.rsplit("-", 1)[-1]) for s in stems]
letters = dict(zip(stems, suffix)) if len(set(suffix)) == len(stems) and max(map(len, suffix)) <= 4 else \
    dict(zip(stems, "abcdefghijklmnopqrstuvwxyz"[:len(stems)]))
rt_nat, rt_cor, eics = {}, {}, {}
wdir = ROOT / "results/peak3d" / args.run / "peak3d_out/rt_correction"
raw = {}
for f, s in zip(files, stems):
    rt, mzs, ints = read_file(f)
    w = pd.read_csv(wdir / f"{s}.tsv", sep="\t")
    assert len(w) == len(rt) and np.abs(w["rt_native"].values - rt).max() < 1e-3, f"{s}: warp table does not match scans"
    rt_nat[s], rt_cor[s] = rt, w["rt_corr"].values
    raw[s] = (mzs, ints)
    print(f"{s}: {len(rt)} MS1 scans, max |corr - native| {60 * np.abs(rt_cor[s] - rt).max():.1f} s")
if args.bins:
    bins = [tuple(float(v) for v in b.split("-")) for b in args.bins.split(",")]
else:
    t0 = max(r[0] for r in rt_nat.values()); t1 = min(r[-1] for r in rt_nat.values())
    span = t1 - t0
    edges = np.linspace(t0 + 0.05 * span, t1 - 0.05 * span, args.n + 1)
    bins = list(zip(edges[:-1], edges[1:]))
sel = select(truth, bins, min_sep=0.25 * (bins[0][1] - bins[0][0]))
sel.to_csv(out / "selected.tsv", sep="\t", index=False)
print(f"selected {len(sel)} features:\n{sel[['truth_id', 'mz', 'rt', 'height', 'n', 'tier']].to_string(index=False)}")
for s in stems:
    mzs, ints = raw[s]
    eics[s] = {i: eic(mzs, ints, float(r["mz"]), args.ppm) for i, r in sel.iterrows()}

style()
summary = []
for i, r in sel.iterrows():
    apex = plot_feature(r, stems, letters, rt_nat, rt_cor, {s: eics[s][i] for s in stems}, args.window,
                        str(out / f"feature_{i + 1:02d}"))
    summary.append(dict(index=i + 1, truth_id=str(r["truth_id"]), mz=float(r["mz"]), rt=float(r["rt"]),
                        files={letters[s]: apex[s] for s in stems}))
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print("wrote", out)
