#!/usr/bin/env python3
"""Does the warp apply the *right* shift?  For one run, per file and per RT bin: the shift the
truth features imply (feature-wise median RT across files minus this file's native apex) against
the shift the fitted warp applies (rt_corr - rt), plus what is left after correction.
usage: warp_vs_truth_by_rt.py RUN OUT.png [bin_min]
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rt_external_validation import collect, INK, INK2, GRID, CA, CB

run, out = sys.argv[1], Path(sys.argv[2]); bw = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
tr, stems, dt, nat, cor = collect(run)
d = Path(__file__).resolve().parents[2] / "results/peak3d" / run / "peak3d_out"
S = pd.read_csv(d / "rt_correction/summary.tsv", sep="\t")
ok = np.isfinite(nat).sum(1) >= max(3, len(stems) // 2)
nat, cor = nat[ok], cor[ok]
ref = np.nanmedian(nat, axis=1)              # consensus native RT of each truth feature (min)
refc = np.nanmedian(cor, axis=1)
rows = []
nf = len(stems); ncol = 5; nrow = int(np.ceil(nf / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), dpi=130, sharex=True, sharey=True)
for j, s in enumerate(stems):
    w = pd.read_csv(d / f"rt_correction/{s}.tsv", sep="\t")
    sh_truth = 60 * (ref - nat[:, j]); res_corr = 60 * (refc - cor[:, j]); m = np.isfinite(sh_truth)
    ax = axes.flat[j]
    ax.scatter(nat[m, j], sh_truth[m], s=5, color=CA, alpha=0.45, linewidths=0, label="truth-implied shift (native)")
    ax.scatter(nat[m, j], res_corr[m], s=5, color=CB, alpha=0.45, linewidths=0, label="residual after correction")
    ax.plot(w.rt_native, 60 * (w.rt_corr - w.rt_native), color=INK, lw=1.2, label="warp shift")
    ax.axhline(0, color=GRID, lw=0.8)
    role = S.loc[S.file == s, "role"].iloc[0]
    ax.set_title(f"{s} ({role})", loc="left", fontsize=8.5, color=INK)
    for lo in np.arange(0, np.nanmax(ref) + bw, bw):
        b = m & (nat[:, j] >= lo) & (nat[:, j] < lo + bw)
        if b.sum() >= 5:
            wi = np.interp(lo + bw / 2, w.rt_native, 60 * (w.rt_corr - w.rt_native))
            rows.append(dict(file=s, rt_bin=f"{lo:.0f}-{lo + bw:.0f}", n=int(b.sum()), truth_shift_s=np.median(sh_truth[b]),
                             warp_shift_s=wi, residual_s=np.median(res_corr[b]),
                             mad_native_s=1.4826 * np.median(np.abs(sh_truth[b] - np.median(sh_truth[b]))),
                             mad_corr_s=1.4826 * np.median(np.abs(res_corr[b] - np.median(res_corr[b])))))
for ax in axes.flat[nf:]:
    ax.axis("off")
axes.flat[0].legend(frameon=False, fontsize=7, loc="upper left")
for ax in axes.flat[:nf]:
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=7); ax.grid(True, color=GRID, lw=0.5, alpha=0.6); ax.set_axisbelow(True)
lim = np.nanpercentile(np.abs(60 * (ref[:, None] - nat)), 98) * 1.2
axes.flat[0].set_ylim(-lim, lim)
fig.supxlabel("native RT (min)", fontsize=8, color=INK2); fig.supylabel("shift (s): + = this file elutes earlier than the consensus", fontsize=8, color=INK2)
fig.suptitle(f"{run}: what the truth features say vs what the warp does, {ok.sum()} truth features", fontsize=9.5, color=INK, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(out); plt.close(fig)
T = pd.DataFrame(rows)
T.to_csv(out.with_suffix(".tsv"), sep="\t", index=False, float_format="%.3f")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(T.round(2).to_string(index=False))
print("\nabs(truth - warp) shift disagreement, median over (file, bin):", round(float(np.median(np.abs(T.truth_shift_s - T.warp_shift_s))), 2), "s;",
      "bins where correction made the median residual larger than native:", int((T.residual_s.abs() > T.truth_shift_s.abs()).sum()), "of", len(T))
