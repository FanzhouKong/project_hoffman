"""Figures for the specdenoise benchmarks, modelled on Kong et al. 2025 (Nat Methods) Fig. 2, Fig. 3 and ED Fig. 2,
with our modes added as extra benchmarks.

usage: python bench/denoise/figures.py [dilution|synthetic|nist|all]   -> results/denoise/figures/*.png"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = "results/denoise/figures"
PMOL = {2: 200, 3: 100, 4: 40, 5: 10, 6: 4, 7: 2, 8: 1, 9: 0.4, 10: 0.2, 11: 0.1, 12: 0.04, 13: 0.02}
# fixed series colors (identity never changes between panels)
C = {"raw": "#7a7975", "published": "#eb6834", "free": "#eda100", "t0_top5": "#2a78d6", "t1": "#6250d6", "clean": "#1baf7a"}
NAME = {"raw": "Raw", "published": "Spectral Denoising (published)", "free": "Formula-free loss rule",
        "t0_top5": "T0: formula inferred (top-5)", "t1": "T1: formula given", "clean": "Clean spectrum", "contaminated": "Raw + noise"}
ORDER = ["raw", "published", "free", "t0_top5", "t1"]
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e3e2de"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 10,
                     "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "text.color": TEXT})


def kde(x, grid):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5 or np.std(x) < 1e-6:
        return np.zeros_like(grid)
    return gaussian_kde(x, bw_method=0.15)(grid)


def panel_label(ax, s):
    ax._panel = s


def title(ax, text):
    ax.set_title(f"{getattr(ax, '_panel', '')}   {text}", loc="left", fontsize=9.5, pad=8)


def fig_dilution():
    R = pd.read_csv("results/denoise/dilution/mapped.csv")
    m = "weighted_entropy_"
    grid = np.linspace(0, 1, 300)
    fig = plt.figure(figsize=(13, 8.2))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.32)
    # a: raw similarity density by amount (as the paper's Fig. 2b)
    ax = fig.add_subplot(gs[0, 0]); panel_label(ax, "a")
    cmap = plt.get_cmap("Blues")
    concs = sorted(PMOL)
    for k, c in enumerate(concs):
        x = R.loc[R.conc == c, m + "raw"]
        ax.plot(grid, kde(x, grid), color=cmap(0.25 + 0.7 * k / (len(concs) - 1)), lw=1.6, label=f"{PMOL[c]:g} pmol")
    ax.set(xlabel="Entropy similarity of raw spectra to 500 pmol reference", ylabel="Density"); title(ax, "Raw spectra by amount")
    ax.legend(ncol=2, fontsize=7, title="injected amount", title_fontsize=7)
    # b: mean similarity vs amount by method
    ax = fig.add_subplot(gs[0, 1]); panel_label(ax, "b")
    xs = [PMOL[c] for c in concs]
    for mode in ORDER:
        g = R.groupby("conc")[m + mode]
        mu = g.mean().reindex(concs); se = g.sem().reindex(concs)
        ax.plot(xs, mu, "-o", color=C[mode], lw=2, ms=4, label=NAME[mode])
        ax.fill_between(xs, mu - 1.96 * se, mu + 1.96 * se, color=C[mode], alpha=0.12, lw=0)
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set(xlabel="Injected amount (pmol)", ylabel="Mean entropy similarity to reference"); title(ax, "Similarity to the reference by amount")
    ax.legend(fontsize=7, loc="lower left")
    # c: improvement over raw by amount (paper's Fig. 2c)
    ax = fig.add_subplot(gs[0, 2]); panel_label(ax, "c")
    for mode in ORDER[1:]:
        d = (R[m + mode] - R[m + "raw"]).groupby(R.conc)
        mu = d.mean().reindex(concs); se = d.sem().reindex(concs)
        ax.plot(xs, mu, "-o", color=C[mode], lw=2, ms=4, label=NAME[mode])
        ax.fill_between(xs, mu - 1.96 * se, mu + 1.96 * se, color=C[mode], alpha=0.12, lw=0)
    ax.axhline(0, color=TEXT2, lw=0.8)
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set(xlabel="Injected amount (pmol)", ylabel="Similarity gain over raw (mean, 95 % CI)"); title(ax, "Improvement after denoising")
    ax.legend(fontsize=7, loc="upper left")
    # d: cumulative distribution (paper's Fig. 2e)
    ax = fig.add_subplot(gs[1, 0]); panel_label(ax, "d")
    for mode in ORDER:
        x = np.sort(R[m + mode].values)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=C[mode], lw=2, label=NAME[mode])
    ax.axvline(0.75, color=TEXT2, lw=0.8, ls=":")
    ax.text(0.755, 0.03, "0.75 annotation\nthreshold", fontsize=7, color=TEXT2)
    ax.set(xlabel="Entropy similarity to reference", ylabel="Cumulative fraction of spectra"); title(ax, f"Cumulative distribution, {len(R)} spectra")
    ax.legend(fontsize=7, loc="upper left")
    # e: strip + box of absolute improvement (paper's Fig. 2f)
    ax = fig.add_subplot(gs[1, 1]); panel_label(ax, "e")
    rng = np.random.default_rng(0)
    for k, mode in enumerate(ORDER[1:]):
        d = (R[m + mode] - R[m + "raw"]).values
        ax.scatter(k + rng.uniform(-0.22, 0.22, len(d)), d, s=2, color=C[mode], alpha=0.18, lw=0, rasterized=True)
        ax.boxplot([d], positions=[k], widths=0.5, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="white", edgecolor=TEXT, lw=1), medianprops=dict(color=TEXT, lw=1.4),
                   whiskerprops=dict(color=TEXT, lw=1), capprops=dict(color=TEXT, lw=1))
        ax.text(k, 0.92, f"mean\n{d.mean():+.3f}", ha="center", fontsize=7, color=TEXT2)
    ax.axhline(0, color=TEXT2, lw=0.8)
    ax.set_xticks(range(len(ORDER) - 1)); ax.set_xticklabels(["Published", "Formula-free", "T0 top-5", "T1"], fontsize=8)
    ax.set(ylabel="Similarity improvement over raw", ylim=(-0.6, 1.0)); title(ax, "Per-spectrum improvement over raw")
    # f: compounds annotated at each amount (detection limit)
    ax = fig.add_subplot(gs[1, 2]); panel_label(ax, "f")
    for mode in ORDER:
        best = R[R[m + mode] > 0.75].groupby("name")["conc"].max()
        ys = [int((best >= c).sum()) for c in concs]
        ax.plot(xs, ys, "-o", color=C[mode], lw=2, ms=4, label=NAME[mode])
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set(xlabel="Injected amount (pmol)", ylabel="Compounds annotated (similarity > 0.75)"); title(ax, f"Compounds annotated at each amount ({R['name'].nunique()} total)")
    ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("Dilution series (Kong et al. 2025): 181 standards, 200 to 0.02 pmol, scored against the 500 pmol reference", fontsize=11, y=0.995)
    fig.savefig(f"{OUT}/fig2_dilution.png", bbox_inches="tight")
    plt.close(fig)


def fig_synthetic():
    path = "results/denoise/synthetic/paper.csv"
    if not os.path.exists(path):
        print("no synthetic results yet"); return
    R = pd.read_csv(path)
    grid = np.linspace(0, 1, 300)
    chem = ["0.1", "0.2", "0.5"]; elec = ["2", "10", "100"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9.5), sharex=True, sharey=False)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    modes = ["contaminated", "published", "free", "t0_top5", "t1"]
    col = dict(C, contaminated=C["raw"])
    for i, e in enumerate(elec):
        for j, c in enumerate(chem):
            ax = axes[i, j]
            g = R[R.file == f"sn_{c}_{e}"]
            if len(g) == 0:
                ax.set_visible(False); continue
            ax.plot(grid, kde(g["sim_clean"], grid), color=C["clean"], lw=1.4, ls="--", label=NAME["clean"])
            for mode in modes:
                ax.plot(grid, kde(g["sim_" + mode], grid), color=col[mode], lw=2, label=NAME[mode])
            ax.set_title(f"noise {c}× base peak, {e}× as many ions", fontsize=9, loc="left")
            ax.text(0.03, 0.96, "mean similarity\n" + "\n".join(f"{NAME[mo].split(':')[0].split(' (')[0]}: {g['sim_' + mo].mean():.2f}" for mo in modes),
                    transform=ax.transAxes, fontsize=6.5, va="top", color=TEXT2)
            if i == 2:
                ax.set_xlabel("Entropy similarity to reference")
            if j == 0:
                ax.set_ylabel("Density")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(bottom=0.1)
    fig.suptitle("Synthetic noise added to the dilution-series spectra (paper's files): similarity to the 500 pmol reference", fontsize=11, y=0.95)
    fig.savefig(f"{OUT}/fig3_synthetic.png", bbox_inches="tight")
    plt.close(fig)


def fig_nist():
    from specdenoise.similarity import clean_batch, score_pairs
    z = np.load("/quobyte/metabolomicsgrp/fanzhou/parsed_msp/nist23/nist23_MH_pos.npz", allow_pickle=True)
    r = np.load("results/denoise/nist23/t1_MH_pos.npz"); k = np.load("results/denoise/nist23/t1_MH_pos_keep.npz")
    n = int(r["n"]); off = z["peak_offsets"][: n + 1].astype(np.int64); nion = int(off[-1])
    mz = z["peak_mz"][:nion].astype(np.float64); it = z["peak_intensity"][:nion].astype(np.float64)
    keep = k["keep"]; status = r["status"]; pmz = r["pmz_theo"]
    rng = np.random.default_rng(0)
    sel = rng.choice(np.flatnonzero(status == 0), 20000, replace=False)
    # CSR of the sampled raw and denoised spectra
    def csr(mask):
        o = [0]; mm = []; ii = []
        for i in sel:
            a, b = off[i], off[i + 1]
            s = mask[a:b]
            mm.append(mz[a:b][s]); ii.append(it[a:b][s]); o.append(o[-1] + int(s.sum()))
        return np.array(o, np.int64), np.concatenate(mm), np.concatenate(ii)
    qo, qm, qi = csr(keep); lo, lm, li = csr(np.ones(nion, bool))
    prec = pmz[sel]
    qo, qm, qi = clean_batch(qo, qm, qi, prec, 0.02); lo, lm, li = clean_batch(lo, lm, li, prec, 0.02)
    idx = np.arange(len(sel), dtype=np.int64)
    sim, _ = score_pairs(qo, qm, qi, lo, lm, li, idx, idx, 1, 0.02)
    sim = np.where(np.diff(qo) == 0, 0.0, sim)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8)); fig.subplots_adjust(wspace=0.3)
    ax = axes[0]; panel_label(ax, "a")
    ax.hist(sim, bins=np.linspace(0.5, 1.0, 101), color=C["t1"], lw=0)
    ax.set(xlabel="Entropy similarity, denoised vs raw NIST23 spectrum", ylabel="Spectra"); title(ax, "Denoised vs raw, formula given (20,000 NIST23 spectra)")
    ax.text(0.03, 0.95, f"median {np.median(sim):.3f}\nmean {sim.mean():.3f}\n> 0.99: {100 * np.mean(sim > 0.99):.1f} %\n> 0.95: {100 * np.mean(sim > 0.95):.1f} %",
            transform=ax.transAxes, va="top", fontsize=8, color=TEXT2)
    ax.set_yscale("log")
    # b: retention vs tolerance from the summary table (includes the isotopologue rescue)
    ax = axes[1]; panel_label(ax, "b")
    rows = {}
    for line in open("results/denoise/nist23/t1_MH_pos.md"):
        mm = re.match(r"\| (\d+) mDa \| ([\d.]+) % \| ([\d.]+) % \|", line)
        if mm:
            rows[int(mm.group(1))] = (float(mm.group(2)), float(mm.group(3)))
    tols = sorted(rows)
    ions = [rows[t][0] for t in tols]; inten = [rows[t][1] for t in tols]
    ax.plot(tols, ions, "-o", color=C["t1"], lw=2, label="ions kept")
    ax.plot(tols, inten, "-o", color=C["t0_top5"], lw=2, label="intensity kept")
    for t, a_, b_ in zip(tols, ions, inten):
        ax.text(t, a_ - 0.45, f"{a_:.1f}", ha="center", va="top", fontsize=7, color=TEXT2); ax.text(t, b_ + 0.3, f"{b_:.1f}", ha="center", fontsize=7, color=TEXT2)
    ax.set_xscale("log"); ax.set_xticks(tols); ax.set_xticklabels([str(t) for t in tols]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set(xlabel="Fragment m/z tolerance (mDa)", ylabel="% of NIST23 fragment ions / intensity kept", ylim=(91, 100.3))
    title(ax, "NIST23 ions kept vs tolerance (567k spectra)")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(f"{OUT}/fig_nist.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("dilution", "all"):
        fig_dilution()
    if what in ("synthetic", "all"):
        fig_synthetic()
    if what in ("nist", "all"):
        fig_nist()
