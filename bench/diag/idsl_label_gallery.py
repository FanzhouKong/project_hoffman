# diagnostic: EIC galleries of IDSL.IPA labels by audit verdict (idsl_label_verdict.py): labelled file 003 at 10 ppm
# (blue) and 30 ppm (grey), label RT (red line), scoring window (shaded), and 6 neighbouring injections at 10 ppm
# (thin orange, drift-free: the audit measured < 0.2 s median drift).
# usage: idsl_label_gallery.py AUDIT_DIR OUT_PREFIX N "verdict substring" ["verdict substring" ...]   (reads mzML -> srun)
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.msdata import load_run

D, prefix, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
groups = sys.argv[4:]
A = pd.read_csv(D / "verdict.tsv", sep="\t")
pub = ["IDSL.IPA", "XCMS", "MZMINE", "MSDIAL"]
A["n_tools"] = A[pub].isin(["TP", "FP"]).sum(axis=1)
run = load_run(ROOT / "data/raw/IDSL_IPA/003.mzML")
reps = [load_run(ROOT / f"data/mzml/MTBLS1684/{s}.mzML") for s in ("004", "007", "010", "015", "020", "025")]
HALF = 0.6


def trace(r, mz, ppm, rt):
    lo, hi = r.scan_window(rt, HALF)
    return r.rt[lo:hi] * 60, r.eic(mz, ppm)[lo:hi]


for g in groups:
    sel = A[A.verdict.str.contains(re.escape(g))]
    pick = sel.sample(min(n, len(sel)), random_state=2) if len(sel) else sel
    if not len(pick):
        continue
    ncol = 5
    nrow = int(np.ceil(len(pick) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(17, 2.5 * nrow), dpi=100, squeeze=False)
    for ax, (_, f) in zip(axes.ravel(), pick.iterrows()):
        mz, rt = float(f["m/z"]), float(f["RT(min)"])
        for r in reps:
            x, y = trace(r, mz, 10, rt)
            ax.plot(x, y, color="#e3a008", lw=0.5, alpha=0.7)
        x, y = trace(run, mz, 30, rt)
        ax.plot(x, y, color="#aaaaaa", lw=0.7)
        x, y = trace(run, mz, 10, rt)
        ax.plot(x, y, color="#0969da", lw=1.0, marker=".", ms=2)
        ax.axvspan((rt - 0.1) * 60, (rt + 0.1) * 60, color="#0969da", alpha=0.07)
        ax.axvline(rt * 60, color="#cf222e", lw=0.7)
        ax.set_title(f"#{int(f['id'])} {f['Manual Curation']} m/z {mz:.4f} rt {rt:.2f} | {f['cls003']} h {f['h']:.0f}\n"
                     f"pbr {f['pbr']:.1f} pts {f['n_pts']:.0f} r2 {f['r2']:.2f} rep {f['rep']:.2f} "
                     f"tools {f['n_tools']} p {f['p_tp']:.2f} drt {f['drt_s']:.1f}s", fontsize=6.5, loc="left")
        ax.tick_params(labelsize=5)
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    fig.suptitle(f"{g}  (n = {len(sel)}, {len(pick)} random)  blue: 003 at 10 ppm, grey: 30 ppm, orange: 6 other "
                 f"injections at 10 ppm, red: label RT, shaded: +/-0.1 min scoring window", fontsize=9)
    fig.tight_layout()
    tag = re.sub(r"[^A-Za-z0-9]+", "_", g).strip("_")
    fig.savefig(f"{prefix}_{tag}.png")
    plt.close(fig)
    print("written", f"{prefix}_{tag}.png", flush=True)
