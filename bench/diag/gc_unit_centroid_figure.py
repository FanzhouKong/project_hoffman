"""Why peaks were cut in the CSF .cdf runs: three ions of one file, each with the chromatogram of its nominal mass,
the mass the file writes for it, and the features Mosa reported when the masses were read as accurate masses and
when the file is read at nominal masses.
Usage: python bench/diag/gc_unit_centroid_figure.py OLD_OUT NEW_OUT CDF_FILE OUT.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from peak3d import cdf  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
EXAMPLES = [(217, 533.2, 5.5, "Mass 217: fragment over a background ion"),
            (575, 914.9, 5.5, "Mass 575: detector saturates at the top"),
            (284, 486.0, 4.5, "Mass 284: points missing from the file")]


def main():
    old_dir, new_dir, path, out = map(Path, sys.argv[1:5])
    stem = path.stem
    rt, off, mz, it = cdf.read_cdf(path)
    scan = np.repeat(np.arange(len(rt)), np.diff(off))
    nom = np.floor(mz + cdf.BIN_SHIFT).astype(int)
    old = pd.read_csv(old_dir / "features" / f"{stem}.tsv", sep="\t")
    new = pd.read_csv(new_dir / "features" / f"{stem}.tsv", sep="\t")
    old["nom"] = np.floor(old.mz + cdf.BIN_SHIFT).astype(int)
    new["nom"] = cdf.unit_mass(new.mz)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 7.6), sharex="col", gridspec_kw=dict(height_ratios=[3, 2.4, 1.1], hspace=0.12, wspace=0.26),
                             facecolor=SURFACE)
    for c, (n, t0, half, title) in enumerate(EXAMPLES):
        q = (nom == n) & (rt[scan] > t0 - half) & (rt[scan] < t0 + half)
        t, m, y = rt[scan][q], mz[q], it[q]
        a, b, d = axes[0, c], axes[1, c], axes[2, c]
        for ax in (a, b, d):
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        # the chromatogram is drawn point by point: a line would bridge the scans the file has no centroid for
        a.plot(t, y / 1e6, ls="none", marker="o", ms=3.2, color=BLUE)
        a.set_title(title, fontsize=10.5, loc="left", color=INK, pad=8)
        a.set_ylim(0, None)
        b.plot(t, m, ls="none", marker="o", ms=3.2, color=BLUE)
        b.ticklabel_format(useOffset=False, axis="y")
        for lane, (f, col) in enumerate(((old, ORANGE), (new, AQUA))):
            g = f[(f.nom == n) & (f.rt_max * 60 > t0 - half) & (f.rt_min * 60 < t0 + half)]
            for r in g.itertuples():
                d.plot([r.rt_min * 60, r.rt_max * 60], [1 - lane, 1 - lane], color=col, lw=7, solid_capstyle="butt")
            if not len(g):
                d.text(t0, 1 - lane, "no feature reported", ha="center", va="center", fontsize=8.5, color=MUTED)
        d.set_ylim(-0.6, 1.6)
        d.set_yticks([1, 0])
        d.set_yticklabels(["read as accurate mass", "read at nominal mass"] if c == 0 else ["before", "after"])
        d.tick_params(axis="y", length=0)
        d.set_xlim(t0 - half, t0 + half)
        d.set_xlabel("Retention time (s)")
        if c == 0:
            a.set_ylabel("Intensity (1e6 counts)")
            b.set_ylabel("m/z written in the file")
            d.set_ylabel("Features")
    fig.suptitle(f"{stem}: one centroid per nominal mass; its written mass slides, and some points are missing",
                 x=0.06, ha="left", fontsize=12, color=INK, y=0.985)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
