# diagnostic: ridge_ratio of kept features as a function of the m/z window used for the side levels
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "/quobyte/metabolomicsgrp/fanzhou/hoffmann")
from peak3d.io import load_cloud
from peak3d import kernels as K
from peak3d.features import RIDGE_NEAR, RIDGE_FAR
ROOT = "/quobyte/metabolomicsgrp/fanzhou/hoffmann"
stem = "MT_20211013_082"
f = pd.read_csv(f"{ROOT}/results/peak3d/BM21_RP/peak3d_out/features/{stem}.tsv", sep="\t")
P = json.load(open(f"{ROOT}/results/peak3d/BM21_RP/peak3d_out/features/{stem}.params.json"))
cloud = load_cloud(f"{ROOT}/data/mzml/BM21/{stem}.mzML")
sig = np.sqrt(P["sig_a"] ** 2 + P["sig_b"] ** 2 / f["height"].values)
w = np.maximum(f["fwhm"].values, P["fwhm_med"]) / P["dt"]
sa = f["scan_apex"].values.astype(np.int64)
near = np.maximum(1, np.round(RIDGE_NEAR * w)).astype(np.int64); far = np.maximum(near + 2, np.round(RIDGE_FAR * w)).astype(np.int64)
q_mz = f["mz"].values.astype(np.float64)
bg = [89.0266, 65.0143, 100.9296, 100.9355, 91.0051, 99.0086]
is_bg = np.zeros(len(f), bool)
for m in bg:
    is_bg |= np.abs(q_mz - m) / m * 1e6 <= 10
print(f"{stem}: {len(f)} kept features, {is_bg.sum()} at the known ragged background m/z values")
for name, tol_ppm in [("3*sig+1 (current)", 3 * sig + 1), ("max(.,5)", np.maximum(3 * sig + 1, 5.0)), ("max(.,8)", np.maximum(3 * sig + 1, 8.0)), ("10", np.full(len(f), 10.0)), ("15", np.full(len(f), 15.0))]:
    q_tol = q_mz * tol_ppm * 1e-6
    left = K.window_level(cloud.off, cloud.mz, cloud.inten, q_mz, q_tol, sa - far, sa - near + 1)
    right = K.window_level(cloud.off, cloud.mz, cloud.inten, q_mz, q_tol, sa + near, sa + far + 1)
    rr = np.minimum(left, right) / f["height"].values
    print(f"  window {name:18s}: flagged (>0.5) overall {np.mean(rr > 0.5):.1%} ({int((rr > 0.5).sum())}), among background ions {np.mean(rr[is_bg] > 0.5):.1%} of {is_bg.sum()}, "
          f"among strong (height>1e6) non-background {np.mean(rr[(f['height'] > 1e6) & ~is_bg] > 0.5):.1%}")
# m/z scatter of the 89.0266 ion across the run
idx = cloud.eic_argmax(89.0266, 10.0, 0, cloud.n_scans); idx = idx[idx >= 0]
d = (cloud.mz[idx] - 89.0266) / 89.0266 * 1e6
print("89.0266: centroid present in %d of %d scans; ppm offset quantiles %s" % (len(idx), cloud.n_scans, np.percentile(d, [5, 25, 50, 75, 95]).round(2).tolist()))
