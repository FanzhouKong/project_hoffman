# diagnostic: would a stricter long-range test flag truth-matched features? (HZV029 certified, LI2018 standards)
import sys, json, glob
import numpy as np, pandas as pd
sys.path.insert(0, "/quobyte/metabolomicsgrp/fanzhou/hoffmann")
from peak3d.io import load_cloud
from peak3d import kernels as K
from peak3d.features import RIDGE_NEAR, RIDGE_FAR
ROOT = "/quobyte/metabolomicsgrp/fanzhou/hoffmann"

def levels(f, P, cloud, q, min_ppm=5.0):
    sig = np.sqrt(P["sig_a"] ** 2 + P["sig_b"] ** 2 / f["height"].values)
    w = np.maximum(f["fwhm"].values, P["fwhm_med"]) / P["dt"]
    sa = f["scan_apex"].values.astype(np.int64)
    near = np.maximum(1, np.round(RIDGE_NEAR * w)).astype(np.int64); far = np.maximum(near + 2, np.round(RIDGE_FAR * w)).astype(np.int64)
    q_mz = f["mz"].values.astype(np.float64); q_tol = q_mz * np.maximum(3 * sig + 1, min_ppm) * 1e-6
    l = K.window_level(cloud.off, cloud.mz, cloud.inten, q_mz, q_tol, sa - far, sa - near + 1, q)
    r = K.window_level(cloud.off, cloud.mz, cloud.inten, q_mz, q_tol, sa + near, sa + far + 1, q)
    return np.minimum(l, r) / f["height"].values

def match(f, qm, qr, ppm=5.0, rt_tol=0.1):
    mz, rt = f["mz"].values, f["rt"].values
    o = np.argsort(mz); smz = mz[o]; hit = np.zeros(len(f), bool)
    for m, t in zip(qm, qr):
        lo = np.searchsorted(smz, m * (1 - ppm * 1e-6)); hi = np.searchsorted(smz, m * (1 + ppm * 1e-6), side="right")
        k = o[lo:hi]; k = k[np.abs(rt[k] - t) <= rt_tol]
        hit[k] = True
    return hit

cases = []
tr = pd.read_csv(glob.glob(f"{ROOT}/data/raw/ASARI_DATA/x/*/data/hzv029_manual_certified.txt")[0], sep="\t")
cases.append(("HZV029_cert", "batch4_MT_20210729_003G", f"{ROOT}/data/mzml/HZV029/batch4_MT_20210729_003G.mzML", tr["moverz"].values, tr["RT_minutes"].values))
li = pd.read_csv(f"{ROOT}/data/raw/LI2018_QE/m_MTBLS733_mass_spectrometry_v2_maf.tsv", sep="\t")
cases.append(("LI2018", "SB1", f"{ROOT}/data/mzml/LI2018_QE/SB1.mzML", li["mass_to_charge"].values, li["retention_time"].values))
cases.append(("BM21_RP", "MT_20211013_082", f"{ROOT}/data/mzml/BM21/MT_20211013_082.mzML", None, None))
for run, stem, mzml, qm, qr in cases:
    f = pd.read_csv(f"{ROOT}/results/peak3d/{run}/peak3d_out/features/{stem}.tsv", sep="\t")
    P = json.load(open(f"{ROOT}/results/peak3d/{run}/peak3d_out/features/{stem}.params.json"))
    cloud = load_cloud(mzml)
    r75, r90 = levels(f, P, cloud, 0.75), levels(f, P, cloud, 0.90)
    print(f"\n{run} / {stem}: {len(f)} kept features")
    for name, rr, thr in [("q75 > 0.5 (current)", r75, 0.5), ("q75 > 0.4", r75, 0.4), ("q90 > 0.5", r90, 0.5), ("q90 > 0.4", r90, 0.4)]:
        fl = rr > thr
        line = f"  {name:22s} flags {fl.mean():5.1%} ({fl.sum():5d}) of all"
        if qm is not None:
            hit = match(f, qm, qr)
            line += f"; truth-matched features flagged {fl[hit].sum()} of {hit.sum()}"
        if run == "BM21_RP":
            for target in (84.0104, 802.5604, 89.0266, 91.0051):
                sel = np.abs(f["mz"].values - target) / target * 1e6 <= 10
                line += f"; m/z {target}: {fl[sel].sum()}/{sel.sum()}"
        print(line)
