#!/usr/bin/env python3
"""QC for the PR002464 spiked-standard truth (bench/spiketruth.py): EIC galleries for review.

Writes data/truth/PR002464/<study>/gallery/
  overview.png        one panel per standard: its strongest truth ion (found) or its [M+H]+ /
                      [M-H]- over the whole run (not found)
  std_<NN>.png        every truth ion of a found standard
EIC colours: vial 18 (spike A only, 4 injections) red, vial 19 (spike B only) blue, vial 9 (A+B
1:1) black, process and solvent blanks grey. Dashed line: truth apex RT.
usage: spike_gallery.py STUDY
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import spiketruth as stt  # noqa: E402
from bench.msdata import load_run  # noqa: E402

STYLE = {18: ("tab:red", "A only"), 19: ("tab:blue", "B only"), 9: ("k", "A+B"), 0: ("0.6", "blank")}


def pick_files(files):
    out = []
    for v in (18, 19):
        out += [(v, p) for p in files.loc[files["vial"] == v, "path"]]
    out.append((9, files.loc[files["vial"] == 9, "path"].iloc[0]))
    for t in ("process.blank", "solvent.blank"):
        out.append((0, files.loc[files["sample_type"] == t, "path"].iloc[0]))
    return out


def draw(ax, runs, mz, ppm, t0, t1, rt_mark=None, title=""):
    for v, run in runs:
        s0, s1 = run.scan_window(0.5 * (t0 + t1), 0.5 * (t1 - t0))
        if s1 <= s0:
            continue
        e, _ = stt.eic_win(run, mz, ppm, s0, s1)
        c, _ = STYLE[v]
        ax.plot(run.rt[s0:s1], e, color=c, lw=0.8, alpha=0.9 if v else 0.7)
    if rt_mark is not None:
        ax.axvline(rt_mark, ls="--", lw=0.6, color="0.3")
    ax.set_title(title, fontsize=6)
    ax.tick_params(labelsize=5)
    ax.yaxis.get_offset_text().set_fontsize(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=list(stt.STUDIES))
    a = ap.parse_args()
    mode, instrument, _ = stt.STUDIES[a.study]
    ppm = stt.PPM[instrument]
    d = ROOT / "data/truth/PR002464" / a.study
    truth = pd.read_csv(d / "spiked_truth.tsv", sep="\t")
    summ = pd.read_csv(d / "standards_summary.tsv", sep="\t")
    files = stt.study_files(a.study)
    runs = [(v, load_run(p)) for v, p in pick_files(files)]
    stds = stt.standards_table().set_index("std")
    g = d / "gallery"
    g.mkdir(exist_ok=True)

    # overview: 96 panels
    fig, axes = plt.subplots(12, 8, figsize=(24, 30))
    for ax, (_, s) in zip(axes.flat, summ.iterrows()):
        tr = truth[(truth["std"] == s["std"]) & (truth["tier"] == "A") & (truth["rt_rank"] == 1)]
        if len(tr):
            b = tr.sort_values("height_dom", ascending=False).iloc[0]
            draw(ax, runs, b["mz_obs"], ppm, b["rt_apex"] - 0.4, b["rt_apex"] + 0.4, b["rt_apex"],
                 f"{s['std']} {s['name'][:28]}\n{b['annotation']} {b['mz_obs']:.4f} @ {b['rt_apex']:.2f} "
                 f"FC {s['log2FC']:+.0f} obs {b['log2_18_19']:+.1f}")
        else:
            mz = stds.loc[s["std"], "mass"] + (stt.PROTON if mode == "pos" else -stt.PROTON)
            t1 = max(r.rt[-1] for _, r in runs)
            draw(ax, runs, mz, ppm, 0, t1, None,
                 f"{s['std']} {s['name'][:28]}\nNOT FOUND {'[M+H]+' if mode == 'pos' else '[M-H]-'} "
                 f"{mz:.4f} FC {s['log2FC']:+.0f}")
            ax.set_facecolor("#fff4f0")
    for ax in axes.flat[len(summ):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=c, label=l) for c, l in STYLE.values()]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    fig.suptitle(f"{a.study} spiked standards: strongest truth ion (or [M+/-H] if not found)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(g / "overview.png", dpi=110)
    plt.close(fig)

    # per standard: every truth ion
    for s, tr in truth.groupby("std"):
        tr = tr.sort_values(["tier", "rt_rank", "height_dom"], ascending=[True, True, False]).head(24)
        n = len(tr)
        ncol = min(4, n)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.3 * nrow), squeeze=False)
        for ax, (_, r) in zip(axes.flat, tr.iterrows()):
            draw(ax, runs, r["mz_obs"], ppm, r["rt_apex"] - 0.4, r["rt_apex"] + 0.4, r["rt_apex"],
                 f"{r['tier']} {r['annotation'][:30]} rank {r['rt_rank']} {r['flags'] if isinstance(r['flags'], str) else ''}\n"
                 f"{r['mz_obs']:.4f} obs {r['log2_18_19']:+.1f} blank {r['blank_frac']:.2f}")
        for ax in axes.flat[n:]:
            ax.axis("off")
        fig.suptitle(f"{a.study} std {s} {stds.loc[s, 'name'][:50]} ({stds.loc[s, 'formula']}, "
                     f"log2FC {stds.loc[s, 'log2FC']:+.0f})", fontsize=9)
        fig.tight_layout()
        fig.savefig(g / f"std_{s:02d}.png", dpi=100)
        plt.close(fig)
    print(f"wrote {g}")


if __name__ == "__main__":
    main()
