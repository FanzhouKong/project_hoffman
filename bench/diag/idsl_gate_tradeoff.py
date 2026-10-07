# diagnostic used while tuning peak3d (reads results/peak3d and the labels; report-only)
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/quobyte/metabolomicsgrp/fanzhou/hoffmann")
from peak3d.io import load_cloud
from peak3d.pick import pick_cloud
from peak3d.features import basins_to_frame, gate, rule_score
from peak3d import kernels as K

cloud = load_cloud("data/raw/IDSL_IPA/003.mzml".replace(".mzml", ".mzML"))
b = pick_cloud(cloud); P = b.params
df = basins_to_frame(b)
sig_apex = df["sig_apex"].values
g = pd.DataFrame(index=df.index)
g["n_scans"] = df["n_scans"].values < P.min_scans
g["snr"] = df["snr"].values < P.min_snr
g["mz_sd"] = df["mz_sd_ppm"].values > 3 * sig_apex + 1
g["width"] = df["fwhm"].values < 0.3 * P.fwhm_med
g["prominence"] = df["prominence_rel"].values < P.min_prominence_rel
at_lo = (df["scan_apex"] == df["scan_lo"]).values; at_hi = (df["scan_apex"] == df["scan_hi"]).values
run_edge = ((df["kflags"].values.astype(int) & (K.KF_RUN_START | K.KF_RUN_END)) > 0)
frag_old = (at_lo | at_hi) & ~run_edge
# refined fragment: the adjacent scan beyond the apex-side bound holds a HIGHER centroid of the same ion
frag_new = np.zeros(len(df), bool)
for i in np.flatnonzero(frag_old):
    side = int(df["scan_lo"].iloc[i]) - 1 if at_lo[i] else int(df["scan_hi"].iloc[i]) + 1
    if 0 <= side < cloud.n_scans:
        e = cloud.eic(float(df["mz"].iloc[i]), P.tol_max_ppm, side, side + 1)
        frag_new[i] = e[0] > df["height"].iloc[i]
g["fragment_old"] = frag_old; g["fragment_new"] = frag_new
score = rule_score(df, P)   # without isotope bonus (close enough for the trade-off)
df["score0"] = score

lab = pd.read_csv("data/raw/IDSL_IPA/idslipa_benchmarking_dataset.csv", encoding="utf-8-sig")
is_tp = (lab["Manual Curation"] == "TP").values
qm, qr = lab["m/z"].values, lab["RT(min)"].values
order = np.argsort(df["mz"].values); smz = df["mz"].values[order]; mzv, rtv = df["mz"].values, df["rt"].values

def best_cand(ppm, rt_tol):
    out = np.full(len(lab), -1)
    for i, (m, t) in enumerate(zip(qm, qr)):
        lo = np.searchsorted(smz, m * (1 - ppm * 1e-6)); hi = np.searchsorted(smz, m * (1 + ppm * 1e-6), side="right")
        k = order[lo:hi]; k = k[np.abs(rtv[k] - t) <= rt_tol]
        if len(k):
            out[i] = k[np.argmax(df["height"].values[k])]
    return out

rng = np.random.default_rng(0); half = rng.random(len(lab)) < 0.5
for ppm, rtt in [(10, 0.1), (20, 0.2)]:
    bc = best_cand(ppm, rtt); has = bc >= 0
    print(f"\n==== {ppm} ppm / {rtt} min: labels with a candidate basin: TP {int((has & is_tp).sum())}/{is_tp.sum()}  TN {int((has & ~is_tp).sum())}/{(~is_tp).sum()}")
    if ppm == 20:
        d_ppm = (mzv[bc[has & is_tp]] - qm[has & is_tp]) / qm[has & is_tp] * 1e6
        d_rt = 60 * (rtv[bc[has & is_tp]] - qr[has & is_tp])
        print(f"  our m/z - label m/z (TPs): median {np.median(d_ppm):+.1f} ppm, MAD {1.4826*np.median(np.abs(d_ppm-np.median(d_ppm))):.1f} ppm; RT: median {np.median(d_rt):+.1f} s, MAD {1.4826*np.median(np.abs(d_rt-np.median(d_rt))):.1f} s")
    def evaluate(gates, cutoff, h):
        ok = np.ones(len(df), bool)
        for gn in gates: ok &= ~g[gn].values
        ok &= score >= cutoff
        det = has & ok[np.maximum(bc, 0)]
        sel = h
        tp = (det & is_tp & sel).sum(); fp = (det & ~is_tp & sel).sum(); fn = (~det & is_tp & sel).sum()
        return tp, fp, tp / (tp + fn), tp / max(tp + fp, 1), 2 * tp / max(2 * tp + fp + fn, 1)
    base = ["n_scans", "snr", "mz_sd", "width", "prominence", "fragment_old"]
    print("  gate set / cutoff                      TP_B   FP_B  recall_B prec_B  F1_B   (half B = held-out half)")
    for name, gates, cut in [("current (all gates, 0.5)", base, 0.5),
                             ("fragment refined", base[:-1] + ["fragment_new"], 0.5),
                             ("no fragment gate", base[:-1], 0.5),
                             ("no snr gate", [x for x in base if x != "snr"], 0.5),
                             ("no width gate", [x for x in base if x != "width"], 0.5),
                             ("no n_scans gate", [x for x in base if x != "n_scans"], 0.5),
                             ("no prominence gate", [x for x in base if x != "prominence"], 0.5),
                             ("refined, cutoff 0.45", base[:-1] + ["fragment_new"], 0.45),
                             ("refined, cutoff 0.4", base[:-1] + ["fragment_new"], 0.4),
                             ("refined, cutoff 0.55", base[:-1] + ["fragment_new"], 0.55),
                             ("refined, cutoff 0.6", base[:-1] + ["fragment_new"], 0.6)]:
        tp, fp, rec, prec, f1 = evaluate(gates, cut, ~half)
        tpa, fpa, reca, preca, f1a = evaluate(gates, cut, half)
        print(f"  {name:36s} {tp:5d}  {fp:5d}   {rec:.3f}  {prec:.3f}  {f1:.3f}   | half A: {reca:.3f} {preca:.3f} {f1a:.3f}")
print("\nfragment flags: old", int(frag_old.sum()), "new", int(frag_new.sum()), "of", len(df), "candidates")
