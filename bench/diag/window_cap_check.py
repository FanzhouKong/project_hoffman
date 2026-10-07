# diagnostic (report-only): would the two proposed ridge-gate refinements (side windows capped at 2x the file's
# median FWHM; asymmetric plateau test = either side's lower quartile >= 0.5 x apex) and the extended score
# (unimodality + localisation terms, cut 0.65) cost truth features on the Orbitrap runs? Reads results/peak3d
# feature tables + mzML. usage: window_cap_check.py RUN [RUN ...]
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench/diag"))
from bench.runs import RUNS
from fix_eval import eval_truth, near_any
from peak3d.io import load_cloud
pd.set_option("display.width", 220)

def q(v, p):
    v = np.sort(v); return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]

def side_stats(cloud, F, P, wmode):
    S = cloud.n_scans; dt = P["dt"]; fw = P["fwhm_med"]
    out = np.full((len(F), 2, 3), np.nan)   # side, (q25, q90, med_nonzero)
    for i, f in enumerate(F.itertuples()):
        w = max(f.fwhm, fw) if wmode == "own" else min(max(f.fwhm, fw), 2 * fw)
        w /= dt
        near = int(max(1, round(2 * w))); far = int(max(near + 2, round(6 * w)))
        sa = int(f.scan_apex)
        sig = np.sqrt(P["sig_a"] ** 2 + P["sig_b"] ** 2 / max(f.height, 1.0))
        ppm = max(5.0, 3 * sig + 1)
        for side, (a0, a1) in enumerate(((sa - far, sa - near + 1), (sa + near, sa + far + 1))):
            a0, a1 = max(0, a0), min(S, a1)
            if a1 - a0 < 3: continue
            e = cloud.eic(float(f.mz), ppm, a0, a1).astype(float)
            out[i, side, 0] = q(e, 0.25) / f.height; out[i, side, 1] = q(e, 0.9) / f.height
            out[i, side, 2] = (np.median(e[e > 0]) / f.height) if (e > 0).any() else 0.0
    return out

def tv_ratio(cloud, F, P):
    out = np.ones(len(F))
    for i, f in enumerate(F.itertuples()):
        sig = np.sqrt(P["sig_a"] ** 2 + P["sig_b"] ** 2 / max(f.height, 1.0)); ppm = max(5.0, 3 * sig + 1)
        e = cloud.eic(float(f.mz), ppm, int(f.scan_min), int(f.scan_max) + 1).astype(float); tr = e[e > 0]
        if len(tr) >= 3:
            apex = tr.max(); base = min(tr[0], tr[-1])
            out[i] = np.abs(np.diff(tr)).sum() / max(2 * (apex - base), 1e-9) if apex > base else 1.0
    return out

for run in sys.argv[1:]:
    files = RUNS[run]["files"]; d = ROOT / "results/peak3d" / run / "peak3d_out"
    T = eval_truth(run)
    rows = []; keep_all = {}
    for path in files:
        stem = Path(path).stem
        F = pd.read_csv(d / "features" / f"{stem}.tsv", sep="\t"); P = json.load(open(d / "features" / f"{stem}.params.json"))
        cloud = load_cloud(path)
        own = side_stats(cloud, F, P, "own"); cap = side_stats(cloud, F, P, "cap2")
        mn = lambda A, k: np.fmin(np.nan_to_num(A[:, 0, k], nan=np.inf), np.nan_to_num(A[:, 1, k], nan=np.inf))
        mx = lambda A, k: np.fmax(np.nan_to_num(A[:, 0, k], nan=-1), np.nan_to_num(A[:, 1, k], nan=-1))
        drop_cap = mn(cap, 1) > 0.5                     # ridge rule with capped windows
        drop_plateau = mx(own, 0) >= 0.5                # either side stays at >= half the apex (lower quartile)
        tv = tv_ratio(cloud, F, P)
        usable = ~np.isnan(own[:, :, 2]); med = np.where(usable, own[:, :, 2], np.inf)
        far_med = np.where(usable.any(axis=1), np.min(med, axis=1), 0.0)
        sig = np.sqrt(P["sig_a"] ** 2 + P["sig_b"] ** 2 / F["height"].values)
        t_snr = np.clip((np.log2(np.maximum(F["snr"].values, 1e-9)) - np.log2(3)) / (np.log2(50) - np.log2(3)), 0, 1)
        t_shape = np.clip(F["gauss_r2"].values, 0, 1); t_mz = np.clip(1 - F["mz_sd_ppm"].values / (3 * sig + 1), 0, 1)
        t_prom = np.clip(F["prominence_rel"].values, 0, 1); t_scans = np.clip(F["n_scans"].values / P["fwhm_scans"], 0, 1)
        t_loc = np.clip(1 - far_med, 0, 1); t_rip = np.clip(1 / tv, 0, 1)
        iso = (F["iso_offset"].values > 0) | (F["iso_parent"].values >= 0) | F["feature_id"].isin(F["iso_parent"]).values
        s7 = (t_snr * t_shape * t_mz * t_prom * t_scans * t_loc * t_rip) ** (1 / 7); s7 = np.where(iso, np.minimum(1, s7 + 0.15), s7)
        keep = {"current": np.ones(len(F), bool), "cap2 ridge": ~drop_cap, "cap2 ridge + plateau": ~(drop_cap | drop_plateau),
                "score7 >= 0.65": s7 >= 0.65, "all three": ~(drop_cap | drop_plateau) & (s7 >= 0.65)}
        keep_all[stem] = (F, keep)
    print(f"\n================ {run}: {len(files)} files, {sum(len(F) for F, _ in keep_all.values())} kept features")
    for name in ["current", "cap2 ridge", "cap2 ridge + plateau", "score7 >= 0.65", "all three"]:
        n_keep = sum(k[name].sum() for _, k in keep_all.values()); n_all = sum(len(F) for F, _ in keep_all.values())
        line = f"  {name:22s} features {n_keep:7d} ({n_keep / n_all:6.1%})"
        if T is not None:
            rec = []
            for stem, (F, k) in keep_all.items():
                sub = F[k[name]]
                rec.append(near_any(T["mz"].values, T["rt"].values, sub["mz"].values, sub["rt_corr"].values, 10, 0.1))
            rec = np.array(rec)
            anyf = rec.any(axis=0)
            line += f"   truth found in >= 1 file {anyf.sum():4d} / {len(T)}   per-file recall {rec.mean():.4f}"
        print(line)
