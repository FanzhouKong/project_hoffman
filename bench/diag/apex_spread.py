# diagnostic: for every truth feature, the spread of peak3d apex RTs across files, native vs corrected
import sys, glob, json
import numpy as np, pandas as pd
ROOT = "/quobyte/metabolomicsgrp/fanzhou/hoffmann"
sys.path.insert(0, ROOT)
from bench.runs import RUNS

def truth_table(run):
    if run.startswith("YEAST_1"):
        t = pd.read_csv(f"{ROOT}/data/truth/YEAST/credentialed.tsv", sep="\t"); t = t[(t.tier == "A") & (~t.ambiguous.astype(bool))]
        if run.endswith("13C"): t = t.assign(mz=t.mz + t.n * 1.0033548378)
        return t[["mz", "rt"]]
    if run.startswith("SZ22"):
        t = pd.read_csv(f"{ROOT}/data/truth/SZ22/credentialed.tsv", sep="\t"); t = t[(t.tier == "A") & (~t.ambiguous.astype(bool))]
        if run.endswith("13C"): t = t.assign(mz=t.mz + t.n * 1.0033548378)
        return t[["mz", "rt"]]
    if run in ("HZV029_cert", "HZV029_full"):
        t = pd.read_csv(glob.glob(f"{ROOT}/data/raw/ASARI_DATA/x/*/data/hzv029_manual_certified.txt")[0], sep="\t"); return t.rename(columns={"moverz": "mz", "RT_minutes": "rt"})[["mz", "rt"]]
    if run == "YEAST_NEG":
        t = pd.read_csv(f"{ROOT}/data/raw/NETID_YEAST/paper_SI/manual_curation_yeast_neg.tsv", sep="\t"); t = t[t.Confidence.astype(str).str.upper() == "TRUE"]; return t.rename(columns={"medMz": "mz", "medRt": "rt"})[["mz", "rt"]]
    if run == "LI2018":
        t = pd.read_csv(f"{ROOT}/data/raw/LI2018_QE/m_MTBLS733_mass_spectrometry_v2_maf.tsv", sep="\t"); return t.rename(columns={"mass_to_charge": "mz", "retention_time": "rt"})[["mz", "rt"]]

def spread_stats(run):
    """truth-feature apex spread across files, native vs corrected; None when no truth for the run"""
    tt = truth_table(run)
    if tt is None:
        return None
    tr = tt.reset_index(drop=True)
    stems = [p.split("/")[-1][:-4] for p in sorted(glob.glob(f"{ROOT}/results/peak3d/{run}/peak3d_out/features/*.tsv")) if not p.endswith(("manifest.tsv", "rejected.tsv"))]
    dt = json.load(open(f"{ROOT}/results/peak3d/{run}/peak3d_out/features/{stems[0]}.params.json"))["dt"]
    nat = np.full((len(tr), len(stems)), np.nan); cor = np.full_like(nat, np.nan)
    for j, s in enumerate(stems):
        f = pd.read_csv(f"{ROOT}/results/peak3d/{run}/peak3d_out/features/{s}.tsv", sep="\t", usecols=["mz", "rt", "rt_corr", "height"])
        o = np.argsort(f.mz.values); smz = f.mz.values[o]
        for i, (m, t) in enumerate(zip(tr.mz.values, tr.rt.values)):
            lo = np.searchsorted(smz, m * (1 - 5e-6)); hi = np.searchsorted(smz, m * (1 + 5e-6), side="right")
            k = o[lo:hi]; k = k[np.abs(f.rt.values[k] - t) <= 0.15]
            if len(k):
                kk = k[np.argmax(f.height.values[k])]          # strongest match: the truth ion itself
                nat[i, j], cor[i, j] = f.rt.values[kk], f.rt_corr.values[kk]
    ok = np.isfinite(nat).sum(axis=1) == len(stems)
    if ok.sum() == 0:
        return dict(run=run, n_truth=len(tr), n_in_all_files=0)
    sn = 60 * (np.nanmax(nat[ok], axis=1) - np.nanmin(nat[ok], axis=1)); sc = 60 * (np.nanmax(cor[ok], axis=1) - np.nanmin(cor[ok], axis=1))
    return dict(run=run, n_truth=len(tr), n_in_all_files=int(ok.sum()), dt_s=60 * dt,
                spread_native_med_s=float(np.median(sn)), spread_native_p90_s=float(np.percentile(sn, 90)),
                spread_corr_med_s=float(np.median(sc)), spread_corr_p90_s=float(np.percentile(sc, 90)),
                frac_tightened=float((sc < sn - 1e-9).mean()), frac_widened=float((sc > sn + 1e-9).mean()))


if __name__ == "__main__":
    for run in sys.argv[1:]:
        r = spread_stats(run)
        if r and r["n_in_all_files"]:
            print(f"{run:12s} truth {r['n_truth']:4d}, in all files: {r['n_in_all_files']:4d}; dt {r['dt_s']:.2f} s | apex spread (s): native median {r['spread_native_med_s']:.2f} p90 {r['spread_native_p90_s']:.2f} -> corrected median {r['spread_corr_med_s']:.2f} p90 {r['spread_corr_p90_s']:.2f} | tightened {r['frac_tightened']:.0%}, widened {r['frac_widened']:.0%}")
