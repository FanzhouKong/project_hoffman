"""QC figures for one processed study (matplotlib, Agg). Static PNGs under <output>/qc/.

Colour carries one job per figure: file order is an ordinal, so drift curves use one
sequential hue (light = early injection, dark = late); before/after residuals use two fixed
categorical hues; the medoid and flagged files are called out by direct labels, not colour alone."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .align import AlignResult  # noqa: E402

INK, INK2, GRID = "#1f2328", "#59636e", "#d0d7de"
CAT = ["#0969da", "#bf8700"]          # before, after
SEQ = plt.cm.Blues                     # one hue for file order


def _style(ax, xlabel, ylabel, title):
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)


def drift_curves(res: AlignResult, path: Path):
    stems = list(res.files)
    n = len(stems)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
    for k, s in enumerate(stems):
        fa = res.files[s]
        g = fa.warp.native
        if len(g) < 2:
            continue
        sh = 60 * fa.warp.shift(g)
        c = SEQ(0.3 + 0.65 * k / max(n - 1, 1))
        ax.plot(g, sh, color=c, linewidth=1.0 if fa.role != "medoid" else 2.0, alpha=0.9)
        if fa.role == "medoid":
            ax.annotate("medoid: " + s, (g[-1], sh[-1]), fontsize=7, color=INK, ha="right", va="bottom")
        elif not fa.diag.flag.startswith("ok"):
            ax.annotate(s + " (" + fa.diag.flag.split(":")[0] + ")", (g[-1], sh[-1]), fontsize=6, color=INK2,
                        ha="right", va="bottom")
    _style(ax, "native RT (min)", "shift to consensus axis (s)",
           f"RT correction per file (n={n}; light = first file, dark = last)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def residual_hist(res: AlignResult, path: Path):
    before, after = [], []
    for fa in res.files.values():
        m = fa.matched
        if m is None or not len(m):
            continue
        before.append(m["residual_before_s"].values)
        after.append(m.loc[m["is_inlier"], "residual_after_s"].values)
    if not before:
        return
    b, a = np.concatenate(before), np.concatenate(after)
    lim = np.percentile(np.abs(np.concatenate([b, a])), 99) if len(a) else np.abs(b).max()
    bins = np.linspace(-lim, lim, 61)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
    ax.hist(b, bins, color=CAT[0], alpha=0.75, label=f"before (MAD {1.4826 * np.median(np.abs(b - np.median(b))):.2f} s)")
    ax.hist(a, bins, color=CAT[1], alpha=0.75, label=f"after (MAD {1.4826 * np.median(np.abs(a - np.median(a))):.2f} s)")
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "anchor RT residual to consensus (s)", "anchors", "Anchor residuals before and after correction")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def anchor_fit(fa, path: Path):
    m = fa.matched
    if m is None or not len(m):
        return
    fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
    ok = m["is_inlier"].values.astype(bool)
    ax.scatter(m["rt_native"][~ok], m["residual_before_s"][~ok], s=10, facecolors="none", edgecolors=INK2,
               linewidths=0.6, label="matched anchor (outlier)")
    ax.scatter(m["rt_native"][ok], m["residual_before_s"][ok], s=10, color=CAT[0], label="matched anchor")
    g = fa.warp.native
    if fa.warp_fit is not None:
        ax.plot(g, 60 * fa.warp_fit.shift(g), color=INK2, linewidth=1.0, linestyle="--", label="anchor fit")
    ax.plot(g, 60 * fa.warp.shift(g), color=INK, linewidth=1.5,
            label="applied warp (after the do-no-harm check)" if fa.warp_fit is not None else "fitted warp")
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "native RT (min)", "consensus - native (s)", f"{fa.stem}: {fa.flag}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def presence_hist(cons, n_files, path: Path):
    nd = np.asarray(cons["n_detected"])
    if len(nd) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
    ax.hist(nd, bins=np.arange(0.5, n_files + 1.5, max(1, n_files // 50)), color=CAT[0])
    _style(ax, "files in which the group was detected", "groups",
           f"Presence of {len(nd)} groups across {n_files} files; {np.mean(nd >= 0.8 * n_files):.1%} in >= 80 %")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_qc(output: Path, res: AlignResult, cons):
    qc = Path(output) / "qc"
    qc.mkdir(exist_ok=True)
    n = len(res.files)
    if n > 1:
        drift_curves(res, qc / "drift_curves.png")
        residual_hist(res, qc / "anchor_residuals.png")
        flagged = [fa for fa in res.files.values() if fa.role != "medoid" and not fa.diag.flag.startswith("ok")]
        others = [fa for fa in res.files.values() if fa.role != "medoid" and fa.diag.flag.startswith("ok")]
        for fa in flagged[:12] + others[:6]:
            anchor_fit(fa, qc / f"anchor_fit_{fa.stem}.png")
    presence_hist(cons, n, qc / "presence_hist.png")
