#!/usr/bin/env python3
"""External validation of peak3d's RT correction with curated truth lists that the alignment never
sees (yeast21 credentialed / NetID curated, HZV029 certified, SZ22 credentialed, LI2018 standards).

For every truth feature matched in enough files (5 ppm, 0.15 min, strongest feature): the robust
spread (1.4826 x MAD) of its apex RT across files, native vs corrected, plus the per-file median
deviation from the feature-wise median (a file-level bias) before and after.
usage: rt_external_validation.py OUTDIR RUN [RUN ...]
"""
import glob
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench/diag"))
from apex_spread import truth_table  # noqa: E402

INK, INK2, GRID = "#1f2328", "#59636e", "#d0d7de"
CA, CB = "#0969da", "#bf8700"   # native, corrected


def collect(run):
    d = ROOT / "results/peak3d" / run / "peak3d_out"
    tr = truth_table(run)
    if tr is None or not (ROOT / "results/peak3d" / run / "DONE").exists():
        return None
    tr = tr.reset_index(drop=True)
    stems = sorted(p.stem for p in (d / "features").glob("*.tsv") if not p.name.endswith(("manifest.tsv", "rejected.tsv")))
    dt = json.loads((d / "features" / f"{stems[0]}.params.json").read_text())["dt"]
    nat = np.full((len(tr), len(stems)), np.nan); cor = np.full_like(nat, np.nan)
    for j, s in enumerate(stems):
        f = pd.read_csv(d / "features" / f"{s}.tsv", sep="\t", usecols=["mz", "rt", "rt_corr", "height"])
        o = np.argsort(f.mz.values); smz = f.mz.values[o]
        for i, (m, t) in enumerate(zip(tr.mz.values, tr.rt.values)):
            lo = np.searchsorted(smz, m * (1 - 5e-6)); hi = np.searchsorted(smz, m * (1 + 5e-6), side="right")
            k = o[lo:hi]; k = k[np.abs(f.rt.values[k] - t) <= 0.15]
            if len(k):
                kk = k[np.argmax(f.height.values[k])]
                nat[i, j], cor[i, j] = f.rt.values[kk], f.rt_corr.values[kk]
    return tr, stems, dt, nat, cor


def robust_spread(a):
    """1.4826 x MAD across files (s), per feature, ignoring missing files"""
    med = np.nanmedian(a, axis=1, keepdims=True)
    return 60 * 1.4826 * np.nanmedian(np.abs(a - med), axis=1)


def validate(run, out):
    got = collect(run)
    if got is None:
        return None
    tr, stems, dt, nat, cor = got
    N = len(stems)
    need = N if N <= 3 else max(3, int(np.ceil(0.5 * N)))
    ok = np.isfinite(nat).sum(axis=1) >= need
    sn, sc = robust_spread(nat[ok]), robust_spread(cor[ok])
    # file-level bias: per file, median deviation of truth features from the feature-wise median
    dev_n = nat[ok] - np.nanmedian(nat[ok], axis=1, keepdims=True)
    dev_c = cor[ok] - np.nanmedian(cor[ok], axis=1, keepdims=True)
    file_bias = pd.DataFrame({"file": stems, "n_truth": np.isfinite(dev_n).sum(axis=0),
                              "bias_native_s": 60 * np.nanmedian(dev_n, axis=0), "bias_corr_s": 60 * np.nanmedian(dev_c, axis=0),
                              "spread_native_s": 60 * 1.4826 * np.nanmedian(np.abs(dev_n - np.nanmedian(dev_n, axis=0)), axis=0),
                              "spread_corr_s": 60 * 1.4826 * np.nanmedian(np.abs(dev_c - np.nanmedian(dev_c, axis=0)), axis=0)})
    if N > 20:
        file_bias["batch"] = [re.match(r"(batch\d+)", s).group(1) if re.match(r"(batch\d+)", s) else "" for s in stems]
    row = dict(run=run, n_files=N, n_truth=len(tr), n_truth_used=int(ok.sum()), min_files_per_truth=need, dt_s=60 * dt,
               spread_native_med_s=float(np.median(sn)), spread_corr_med_s=float(np.median(sc)),
               spread_native_p90_s=float(np.percentile(sn, 90)), spread_corr_p90_s=float(np.percentile(sc, 90)),
               frac_tightened=float((sc < sn - 1e-9).mean()), frac_widened=float((sc > sn + 1e-9).mean()),
               file_bias_native_mad_s=float(1.4826 * np.median(np.abs(file_bias.bias_native_s - np.median(file_bias.bias_native_s)))),
               file_bias_corr_mad_s=float(1.4826 * np.median(np.abs(file_bias.bias_corr_s - np.median(file_bias.bias_corr_s)))),
               all_dev_native_mad_s=float(60 * 1.4826 * np.nanmedian(np.abs(dev_n))), all_dev_corr_mad_s=float(60 * 1.4826 * np.nanmedian(np.abs(dev_c))))
    file_bias.to_csv(out / f"{run}_file_bias.tsv", sep="\t", index=False, float_format="%.3f")
    # figure: per-feature spread native vs corrected; all per-file deviations native vs corrected
    fig, axes = plt.subplots(1, 3 if N > 20 else 2, figsize=(14 if N > 20 else 10, 4.4), dpi=140)
    ax = axes[0]
    lim = max(np.percentile(np.r_[sn, sc], 99), 60 * dt) * 1.1
    ax.scatter(sn, sc, s=9, color=CA, alpha=0.6, linewidths=0)
    ax.plot([0, lim], [0, lim], color=GRID, lw=1)
    ax.axhline(60 * dt, color=GRID, lw=0.8, ls="--"); ax.axvline(60 * dt, color=GRID, lw=0.8, ls="--")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("spread of apex RT across files, native (s)", color=INK2, fontsize=8); ax.set_ylabel("spread, corrected (s)", color=INK2, fontsize=8)
    ax.set_title(f"{run}: {ok.sum()} truth features in >= {need} of {N} files\nbelow the diagonal = tightened ({row['frac_tightened']:.0%}); dashed = one scan",
                 loc="left", fontsize=8.5, color=INK)
    ax = axes[1]
    dn, dc = 60 * dev_n[np.isfinite(dev_n)], 60 * dev_c[np.isfinite(dev_c)]
    b = np.linspace(-max(np.percentile(np.abs(dn), 99), 60 * dt) * 1.1, max(np.percentile(np.abs(dn), 99), 60 * dt) * 1.1, 61)
    ax.hist(dn, b, color=CA, alpha=0.7, label=f"native (MAD {row['all_dev_native_mad_s']:.2f} s)")
    ax.hist(dc, b, color=CB, alpha=0.7, label=f"corrected (MAD {row['all_dev_corr_mad_s']:.2f} s)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("apex RT deviation from the feature's median across files (s)", color=INK2, fontsize=8); ax.set_ylabel("truth feature x file", color=INK2, fontsize=8)
    ax.set_title("\nevery truth feature in every file", loc="left", fontsize=8.5, color=INK)
    if N > 20:
        ax = axes[2]
        x = np.arange(N)
        ax.plot(x, file_bias.bias_native_s, color=CA, lw=1, label="native")
        ax.plot(x, file_bias.bias_corr_s, color=CB, lw=1, label="corrected")
        ax.axhline(0, color=GRID, lw=0.8)
        ax.legend(frameon=False, fontsize=8)
        ax.set_xlabel("file (acquisition order)", color=INK2, fontsize=8); ax.set_ylabel("median deviation of truth features (s)", color=INK2, fontsize=8)
        ax.set_title(f"\nper-file bias of the truth features: MAD {row['file_bias_native_mad_s']:.2f} s -> {row['file_bias_corr_mad_s']:.2f} s", loc="left", fontsize=8.5, color=INK)
    for ax in axes:
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=7); ax.grid(True, color=GRID, lw=0.5, alpha=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out / f"{run}.png"); plt.close(fig)
    return row


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in (validate(run, out) for run in sys.argv[2:]) if r]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(df.round(2).to_string(index=False))
    df.to_csv(out / "summary.tsv", sep="\t", index=False, float_format="%.3f")
