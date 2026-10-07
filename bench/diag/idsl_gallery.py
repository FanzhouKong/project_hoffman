# diagnostic: EIC gallery of kept peak3d features at TN (curated non-peak) and TP positions of IDSL003
# usage: idsl_gallery.py OUT_PNG_PREFIX [N_PER_CLASS=40]
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from peak3d.io import load_cloud
cloud = load_cloud(ROOT / "data/raw/IDSL_IPA/003.mzML")
M = pd.read_csv(ROOT / "results/diag_idsl/pairs.tsv", sep="\t")
import json; P = json.load(open(ROOT / "results/diag_idsl/params.json"))
K = M[M["kept"]].drop_duplicates("cand_i")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
rng = np.random.default_rng(1)
fs = P["fwhm_scans"]
def panel(ax, f, title):
    sa, slo, shi = int(f["scan_apex"]), int(f["scan_lo"]), int(f["scan_hi"])
    W = int(max(10 * fs, 30))
    s0, s1 = max(0, sa - W), min(cloud.n_scans, sa + W + 1)
    ppm = max(5.0, 3 * f["sig_apex"] + 1)
    e = cloud.eic(float(f["mz"]), ppm, s0, s1)
    e30 = cloud.eic(float(f["mz"]), 30.0, s0, s1)
    x = cloud.rt[s0:s1] * 60
    ax.plot(x, e30, color="#bbbbbb", lw=0.8)
    ax.plot(x, e, color="#0969da", lw=1.0, marker=".", ms=2.5)
    ax.axvspan(cloud.rt[slo] * 60, cloud.rt[shi] * 60, color="#bf8700", alpha=0.15)
    ax.axhline(f["noise"], color="#cf222e", lw=0.6, ls="--")
    ax.set_title(title, fontsize=6.5, loc="left")
    ax.tick_params(labelsize=5)
for cls, sel in (("TN", K[~K["tp"]]), ("TP", K[K["tp"]])):
    pick = sel.sample(min(n, len(sel)), random_state=1)
    ncol = 5; nrow = int(np.ceil(len(pick) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 2.4 * nrow), dpi=110)
    for ax, (_, f) in zip(axes.ravel(), pick.iterrows()):
        panel(ax, f, f"m/z {f['mz']:.4f} rt {f['rt']:.2f} h {f['height']:.0f} snr {f['snr']:.0f} sc {f['score']:.2f}\n"
                     f"fwhm {f['fwhm'] / P['fwhm_med']:.1f}x mzsd {f['mz_sd_ppm']:.1f}({f['mz_sd_ppm'] / f['sig_apex']:.1f}s) r2 {f['gauss_r2']:.2f} iso {int(f['iso_support'])} "
                     f"ridge {f['ridge_ratio']:.2f} lmax {f['n_lmax']:.0f} tv {f['tv_ratio']:.1f} far {f['far_presence']:.2f}")
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    fig.suptitle(f"IDSL003 kept features at {cls} positions (blue: EIC at 3sig+1 ppm; grey: 30 ppm; orange: feature bounds; red dashed: noise surface)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{sys.argv[1]}_{cls}.png")
    plt.close(fig)
print("done")
