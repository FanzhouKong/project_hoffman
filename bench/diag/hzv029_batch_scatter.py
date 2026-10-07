# diagnostic used while tuning peak3d (reads results/peak3d and the labels; report-only)
import numpy as np, pandas as pd, glob, re
from pathlib import Path
d = Path("results/peak3d/HZV029_full/peak3d_out/features")
tr = pd.read_csv(glob.glob("data/raw/ASARI_DATA/x/*/data/hzv029_manual_certified.txt")[0], sep="\t")
# example 1 (mz 280.1707 rt 0.602) and example 3 (160.0974, 0.621): per-file apex rt_corr and m/z, by batch
for q in (50, 300):
    m, r = tr["moverz"].values[q], tr["RT_minutes"].values[q]
    rows = []
    for f in sorted(d.glob("*.tsv")):
        if f.name == "manifest.tsv": continue
        t = pd.read_csv(f, sep="\t", usecols=["mz", "rt", "rt_corr", "height", "fwhm", "snr"])
        sel = t[(np.abs(t["mz"] - m) / m * 1e6 <= 8) & (np.abs(t["rt_corr"] - r) <= 0.3)]
        if len(sel):
            k = sel["height"].idxmax()
            rows.append(dict(file=f.stem, batch=int(re.match(r"batch(\d+)", f.stem).group(1)), rt_corr=sel.loc[k, "rt_corr"],
                             rt=sel.loc[k, "rt"], ppm=(sel.loc[k, "mz"] - m) / m * 1e6, height=sel.loc[k, "height"], fwhm=60 * sel.loc[k, "fwhm"], n=len(sel)))
    R = pd.DataFrame(rows)
    print(f"\n=== certified mz {m:.4f} rt {r:.3f}: found in {len(R)} files; features per file median {R['n'].median():.0f}")
    g = R.groupby("batch").agg(n=("rt_corr", "size"), rt_corr_med=("rt_corr", "median"), rt_corr_sd=("rt_corr", "std"),
                               rt_native_med=("rt", "median"), ppm_med=("ppm", "median"), ppm_sd=("ppm", "std"), fwhm_s=("fwhm", "median"))
    print(g.round(3).to_string())
    print("overall rt_corr sd (s): %.2f ; within-batch sd median (s): %.2f ; between-batch sd of medians (s): %.2f" % (
        60 * R["rt_corr"].std(), 60 * g["rt_corr_sd"].median(), 60 * g["rt_corr_med"].std()))
    print("overall ppm sd: %.2f ; within-batch ppm sd median: %.2f ; between-batch sd of ppm medians: %.2f" % (R["ppm"].std(), g["ppm_sd"].median(), g["ppm_med"].std()))
