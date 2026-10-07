import sys
sys.path.insert(0, "/quobyte/metabolomicsgrp/fanzhou/hoffmann"); sys.path.insert(0, "/quobyte/metabolomicsgrp/fanzhou/hoffmann/peak3d/tests")
import numpy as np, pandas as pd
from peak3d.synth import Peak, make_cloud
from peak3d.pick import pick_cloud
from peak3d.features import basins_to_frame, gate, mark_fragments
from peak3d.estimate import estimate, fwhm_at, strongest_distinct
from test_estimate import _study
DT = 0.005
# --- wide noisy peak ---
cloud, truth = make_cloud([Peak(400.0, 4.0, 1e6, 60 * DT)], n_scans=1600, dt=DT, thresh=200.0, floor=100.0, n_noise=30000, seed=5)
rng = np.random.default_rng(1); sel = np.abs(cloud.mz - 400.0) < 0.01
cloud.inten[sel] *= np.exp(0.04 * rng.standard_normal(sel.sum())).astype(np.float32)
b = pick_cloud(cloud); P = b.params
print("wide: params fwhm_med %.4f fwhm_scans %.1f K %d min_scans %d n_fwhm_seeds %d tol_max %.1f" % (P.fwhm_med, P.fwhm_scans, P.K, P.min_scans, P.n_fwhm_seeds, P.tol_max_ppm))
df = basins_to_frame(b); df["fragment"] = mark_fragments(df, cloud, P); df["ok"] = gate(df, P)
pd.set_option("display.width", 250, "display.max_columns", 40)
s = df[np.abs(df["mz"] - 400.0) < 0.01]
print(s[["mz", "rt", "height", "n_scans", "scan_apex", "scan_lo", "scan_hi", "fwhm", "mz_sd_ppm", "sig_apex", "baseline", "snr", "prominence_rel", "gauss_r2", "kflags", "fragment", "ok"]].to_string(index=False))
print("basins near 400 (all):", int((np.abs(b.col("mz") - 400.0) < 0.01).sum()))
# --- fwhm estimate ---
cloud2, _ = _study(seed=1, fwhm=0.06, sig_a=0.6, sig_b=40.0)
P2 = estimate(cloud2)
print("study: fwhm_med %.4f (truth 0.06) n_fwhm_seeds %d" % (P2.fwhm_med, P2.n_fwhm_seeds))
seeds = strongest_distinct(cloud2, 50); scan_of = np.searchsorted(cloud2.off, seeds, side="right") - 1
ws = [fwhm_at(cloud2, cloud2.mz[p], int(s_)) for p, s_ in zip(seeds, scan_of)]
print("first 15 seed widths:", [round(w[0], 4) if w else None for w in ws[:15]])
