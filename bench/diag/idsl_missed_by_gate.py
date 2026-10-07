# diagnostic used while tuning peak3d (reads results/peak3d and the labels; report-only)
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/quobyte/metabolomicsgrp/fanzhou/hoffmann")
from peak3d import kernels as K
d = "results/peak3d_dev/idsl_keep_rejected/features/"
feats = pd.read_csv(d + "003.tsv", sep="\t")
rej = pd.read_csv(d + "003.rejected.tsv", sep="\t", low_memory=False)
P = json.load(open(d + "003.params.json"))
lab = pd.read_csv("data/raw/IDSL_IPA/idslipa_benchmarking_dataset.csv", encoding="utf-8-sig")
is_tp = (lab["Manual Curation"] == "TP").values
qm, qr = lab["m/z"].values, lab["RT(min)"].values
print("params:", {k: (round(P[k], 3) if isinstance(P[k], float) else P[k]) for k in
                  ["dt", "fwhm_med", "fwhm_scans", "sig_a", "sig_b", "tol_max_ppm", "K", "min_scans", "floor", "n_candidates", "n_features"]})
print("features", len(feats), "rejected", len(rej), "by reason", rej["reason"].value_counts().to_dict())

# gate diagnosis for every rejected candidate (recompute each hard gate)
r = rej
sig_apex = np.sqrt(P["sig_a"] ** 2 + P["sig_b"] ** 2 / r["height"].values)
g = pd.DataFrame(index=r.index)
g["n_scans"] = r["n_scans"].values < P["min_scans"]
g["snr"] = r["snr"].values < P["min_snr"]
g["mz_sd"] = r["mz_sd_ppm"].values > 3 * sig_apex + 1
g["width"] = r["fwhm"].values < 0.3 * P["fwhm_med"]
g["prominence"] = r["prominence_rel"].values < P["min_prominence_rel"]
at_bound = (r["scan_apex"].values == r["scan_lo"].values) | (r["scan_apex"].values == r["scan_hi"].values)
run_edge = (r["kflags"].values.astype(int) & (K.KF_RUN_START | K.KF_RUN_END)) > 0
g["fragment"] = at_bound & ~run_edge
g["score"] = (r["reason"].values == "score")
gates = ["n_scans", "snr", "mz_sd", "width", "prominence", "fragment", "score"]

def match(mz, rt, qmz, qrt, ppm, rt_tol):
    """indices of candidate rows within tolerance of each query (list per query)"""
    o = np.argsort(mz); smz = mz[o]
    out = []
    for m, t in zip(qmz, qrt):
        lo = np.searchsorted(smz, m * (1 - ppm * 1e-6)); hi = np.searchsorted(smz, m * (1 + ppm * 1e-6), side="right")
        k = o[lo:hi]; k = k[np.abs(rt[k] - t) <= rt_tol]
        out.append(k)
    return out

PPM, RTT = 10.0, 0.1
hit_f = match(feats["mz"].values, feats["rt"].values, qm, qr, PPM, RTT)
hit_r = match(r["mz"].values, r["rt"].values, qm, qr, PPM, RTT)
det = np.array([len(h) > 0 for h in hit_f])
print(f"\n== at {PPM} ppm / {RTT} min: TP detected {det[is_tp].sum()}/{is_tp.sum()}  FP (TN with a feature) {det[~is_tp].sum()}/{(~is_tp).sum()}")
# why were TPs missed?
why = []
for i in np.flatnonzero(is_tp & ~det):
    k = hit_r[i]
    if len(k) == 0:
        why.append("no candidate basin"); continue
    # the best candidate = highest height among rejected ones; list all gates it failed
    kb = k[np.argmax(r["height"].values[k])]
    failed = [gname for gname in gates if g.loc[kb, gname]]
    why.append("+".join(failed) if failed else "unknown")
why = pd.Series(why)
print("\nmissed TP by cause (best rejected candidate's failed gates):")
print(why.value_counts().to_string())
# single-gate counts among missed TPs
print("\nmissed TPs failing each gate (candidate exists):")
tot = sum(len(hit_r[i]) > 0 for i in np.flatnonzero(is_tp & ~det))
for gname in gates:
    n = sum(any(g.loc[hit_r[i], gname]) for i in np.flatnonzero(is_tp & ~det) if len(hit_r[i]))
    print(f"  {gname:11s} {n:5d} of {tot}")
# what do TN-located candidates look like, and what would each gate cost/buy
tn_r = np.flatnonzero(~is_tp & ~det)
print("\nTN positions with a rejected candidate:", sum(len(hit_r[i]) > 0 for i in tn_r), "of", len(tn_r))
# score-threshold curve on candidates that passed hard gates (features + reason==score)
passed = pd.concat([feats.assign(src="feat"), r[r["reason"] == "score"].assign(src="rej")], ignore_index=True)
hp = match(passed["mz"].values, passed["rt"].values, qm, qr, PPM, RTT)
best_score = np.array([passed["score"].values[k].max() if len(k) else -1 for k in hp])
best_snr = np.array([passed["snr"].values[k].max() if len(k) else -1 for k in hp])
rng = np.random.default_rng(0)
half = rng.random(len(lab)) < 0.5   # derive on half A, report on half B
print("\n== score cutoff curve (candidates passing hard gates); derive half / report half ==")
print(" cutoff   recall_A prec_A F1_A | recall_B prec_B F1_B")
for c in [0.0, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
    row = []
    for h in (half, ~half):
        dd = (best_score >= c) & h
        tp = (dd & is_tp).sum(); fp = (dd & ~is_tp).sum(); fn = (~dd & is_tp & h).sum()
        row += [tp / (tp + fn), tp / max(tp + fp, 1), 2 * tp / max(2 * tp + fp + fn, 1)]
    print(f" {c:5.2f}    {row[0]:.3f}   {row[1]:.3f}  {row[2]:.3f} |  {row[3]:.3f}   {row[4]:.3f}  {row[5]:.3f}")
print("\n== S/N of the best candidate at TP vs TN positions (passed hard gates) ==")
for lo_, hi_ in [(-1, 0), (0, 3), (3, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)]:
    sel = (best_snr > lo_) & (best_snr <= hi_)
    print(f"  snr ({lo_:>4},{hi_:>5}]: TP {int((sel & is_tp).sum()):5d}  TN {int((sel & ~is_tp).sum()):5d}")
print("\n== intensity of missed vs detected TPs (height of best candidate incl. rejected) ==")
allc = pd.concat([feats, r], ignore_index=True)
ha = match(allc["mz"].values, allc["rt"].values, qm, qr, PPM, RTT)
hbest = np.array([allc["height"].values[k].max() if len(k) else np.nan for k in ha])
for name, sel in [("detected TP", is_tp & det), ("missed TP w/ candidate", is_tp & ~det & ~np.isnan(hbest)), ("missed TP no candidate", is_tp & ~det & np.isnan(hbest))]:
    v = hbest[sel]
    print(f"  {name:24s} n={sel.sum():5d}  height median {np.nanmedian(v) if len(v) else float('nan'):.0f}  p10 {np.nanpercentile(v, 10) if len(v) and not np.all(np.isnan(v)) else float('nan'):.0f}")
