"""what the shoulder-seeded components of an output are: python shoulder_census.py OUT_UNDER_RESULTS STEM [name=value ...]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from peak3d.gc import deconv as DC
from peak3d.io import Cloud

R = Path("/quobyte/metabolomicsgrp/fanzhou/hoffmann/results")
out, stem = R / sys.argv[1], sys.argv[2]
P = DC.params_from_dict(json.loads((out / "features" / f"{stem}.params.json").read_text()))
cloud = Cloud.load_npz(out / ".cache" / f"{stem}.npz")
F = pd.read_csv(out / "features" / f"{stem}.tsv", sep="\t", low_memory=False)
d0 = DC.deconvolve(cloud, F, P, DC.DeconvParams(shoulders=False))
d1 = DC.deconvolve(cloud, F, P, DC.DeconvParams(shoulders=True, shoulder_cut=False))
C0, C1 = d0.components, d1.components
print(f"dt {P.dt * 60:.3f} s, fwhm {P.fwhm_med * 60:.2f} s; components {len(C0)} -> {len(C1)}; second maxima kept {d1.info['n_shoulder_apexes']}; "
      f"explained {d0.info['n_explained']} -> {d1.info['n_explained']}; shoulder-seeded reported {int(C1.shoulder.sum())}")
sh = C1[C1.shoulder].copy()
t0 = np.sort(C0.rt.values)
k = np.clip(np.searchsorted(t0, sh.rt.values), 1, len(t0) - 1)
sh["d_nearest_old_s"] = 60 * np.minimum(np.abs(t0[k] - sh.rt.values), np.abs(t0[k - 1] - sh.rt.values))
sh["d_fwhm"] = sh.d_nearest_old_s / (P.fwhm_med * 60)
print(sh[["n_features", "n_ions", "tic", "d_nearest_old_s", "d_fwhm", "model_r"]].describe().round(3).to_string())
print("by distance to the nearest component of the run without shoulders (peak widths):")
print(pd.cut(sh.d_fwhm, [0, 0.05, 0.1, 0.25, 0.5, 1, 2, 100], include_lowest=True).value_counts().sort_index().to_string())
print("by tight ions:"); print(pd.cut(sh.n_features, [0, 1, 2, 3, 5, 10, 1000]).value_counts().sort_index().to_string())
big = sh.sort_values("tic", ascending=False).head(12)
print(big.assign(rt_s=(big.rt * 60).round(2))[["rt_s", "n_features", "n_ions", "base_mz", "tic", "d_nearest_old_s"]].round(3).to_string(index=False))
