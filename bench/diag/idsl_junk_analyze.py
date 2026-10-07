# diagnostic (report-only): analyse results/diag_idsl/pairs.tsv + no_candidate.tsv
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
pd.set_option("display.width", 250, "display.max_columns", 40)
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/diag_idsl"
M = pd.read_csv(D / "pairs.tsv", sep="\t")
NC = pd.read_csv(D / "no_candidate.tsv", sep="\t")
P = json.load(open(D / "params.json"))
lab = pd.read_csv(ROOT / "data/raw/IDSL_IPA/idslipa_benchmarking_dataset.csv", encoding="utf-8-sig")
is_tp = (lab["Manual Curation"] == "TP").values
n_tp, n_tn = is_tp.sum(), (~is_tp).sum()
M["mz_sd_ratio"] = M["mz_sd_ppm"] / M["sig_apex"]
M["fwhm_rel"] = M["fwhm"] / P["fwhm_med"]
M["log_h"] = np.log10(M["height"])
gates = [c for c in M.columns if c.startswith("g_")]

# ---------- 1. per label: best candidate, outcome
M = M.sort_values(["label_i", "kept", "gate", "height"], ascending=[True, False, False, False])
best = M.drop_duplicates("label_i").set_index("label_i")
def outcome(r):
    if r["kept"]: return "kept"
    if r["gate"]: return "score<0.5"
    f = [g[2:] for g in gates if r[g] and g != "g_score"]
    return "gate:" + "+".join(f) if f else "gate:?"
best["outcome"] = best.apply(outcome, axis=1)
print("== TP labels (%d) ==" % n_tp)
oc = best[best["tp"]]["outcome"].value_counts()
print(oc.to_string()); print("no candidate basin:", int(NC["tp"].sum()))
print("\n== TN labels (%d) ==" % n_tn)
oc = best[~best["tp"]]["outcome"].value_counts()
print(oc.head(15).to_string()); print("no candidate basin:", int((~NC["tp"]).sum()))
# single-gate attribution among TP candidates that failed gates
tpf = best[best["tp"] & ~best["gate"]]
print("\nTP best candidates failing gates: %d; failing each gate:" % len(tpf))
print({g[2:]: int(tpf[g].sum()) for g in gates if g != "g_score"})
print("  failing exactly one gate:", tpf[[g for g in gates if g != "g_score"]].sum(axis=1).eq(1).sum(),
      " -> which:", {g[2:]: int((tpf[g] & tpf[[x for x in gates if x != "g_score"]].sum(axis=1).eq(1)).sum()) for g in gates if g != "g_score"})
print("  medians of TP gate-failures: ", tpf[["height", "snr", "n_scans", "fwhm_rel", "mz_sd_ppm", "mz_sd_ratio", "prominence_rel", "gauss_r2"]].median().round(3).to_dict())

# ---------- 2. matching offsets for kept TP
k = best[best["tp"] & best["kept"]]
print("\nkept-at-TP offsets: d_ppm median %.2f MAD %.2f ; d_rt median %.2f s MAD %.2f s" % (
    k["d_ppm"].median(), 1.4826 * (k["d_ppm"] - k["d_ppm"].median()).abs().median(),
    k["d_rt_s"].median(), 1.4826 * (k["d_rt_s"] - k["d_rt_s"].median()).abs().median()))

# ---------- 3. no-candidate TPs: what is there
nc_tp = NC[NC["tp"]]
print("\n== %d TP labels with no candidate basin ==" % len(nc_tp))
print("raw EIC (20 ppm, +/-0.1 min) nonzero scans:", pd.cut(nc_tp["eic_nz"], [-1, 0, 2, 4, 8, 15, 100]).value_counts().sort_index().to_dict())
print("eic_max quantiles:", nc_tp["eic_max"].quantile([.1, .25, .5, .75, .9]).round(0).to_dict())
has = nc_tp[nc_tp["eic_nz"] > 0]
print("dominant basin span (scans):", pd.cut(has["dom_span"], [0, 1, 2, 3, 5, 10, 20, 50, 2000]).value_counts().sort_index().to_dict())
print("  (span >= min_scans=%d means the points were absorbed by a basin whose apex lies elsewhere)" % P["min_scans"])
print("number of distinct basins among the EIC maxima:", has["n_basins"].describe()[["25%", "50%", "75%"]].round(1).to_dict())
nc_tn = NC[~NC["tp"]]
print("\n== %d TN labels with no candidate: eic_nz" % len(nc_tn), pd.cut(nc_tn["eic_nz"], [-1, 0, 2, 4, 8, 15, 100]).value_counts().sort_index().to_dict())

# ---------- 4. kept features at TP vs TN positions: descriptor separation
K_ = M[M["kept"]].drop_duplicates("cand_i")
both = K_.groupby("cand_i")["tp"].transform("nunique") > 1
tpk = K_[K_["tp"]]; tnk = K_[~K_["tp"]]
print("\n== kept features: at TP positions %d, at TN positions %d ==" % (len(tpk), len(tnk)))
cols = ["height", "snr", "n_scans", "n_gaps", "fwhm_rel", "mz_sd_ppm", "mz_sd_ratio", "prominence_rel", "gauss_r2", "ridge_ratio",
        "background_ratio", "score", "iso_support", "n_lmax", "tv_ratio", "gap_frac", "ripple_rel", "apex_sharp", "far_presence",
        "far_med_rel", "near_cont", "mz_trend_ppm", "n_pts_per_scan", "smooth_prom_rel", "asym", "n_flat", "persistence_rel"]
rows = []
for c in cols:
    a, b = tpk[c].astype(float).dropna().values, tnk[c].astype(float).dropna().values
    u = mannwhitneyu(a, b).statistic / (len(a) * len(b))
    rows.append(dict(desc=c, med_TP=np.median(a), med_TN=np.median(b), q90_TP=np.percentile(a, 90), q90_TN=np.percentile(b, 90), AUC_TP_gt_TN=u))
print(pd.DataFrame(rows).round(3).sort_values("AUC_TP_gt_TN").to_string(index=False))
print("\nabs(mz_trend_ppm) AUC:", round(mannwhitneyu(tnk["mz_trend_ppm"].abs().dropna(), tpk["mz_trend_ppm"].abs().dropna()).statistic / (tnk["mz_trend_ppm"].notna().sum() * tpk["mz_trend_ppm"].notna().sum()), 3))
# conditional on height: within height bins
print("\nby height bin (kept): n TP / n TN, median mz_sd_ratio TP / TN, iso_support TP / TN, median fwhm_rel TP/TN")
hb = pd.cut(K_["log_h"], [2, 3, 3.5, 4, 4.5, 7])
for b_, grp in K_.groupby(hb, observed=True):
    t, n = grp[grp["tp"]], grp[~grp["tp"]]
    print(f"  10^{b_}: {len(t):4d} / {len(n):4d}   mz_sd_ratio {t['mz_sd_ratio'].median():.2f} / {n['mz_sd_ratio'].median():.2f}   "
          f"iso {t['iso_support'].mean():.2f} / {n['iso_support'].mean():.2f}   fwhm_rel {t['fwhm_rel'].median():.2f} / {n['fwhm_rel'].median():.2f}   "
          f"r2 {t['gauss_r2'].median():.2f} / {n['gauss_r2'].median():.2f}   n_lmax {t['n_lmax'].median():.0f} / {n['n_lmax'].median():.0f}")

# ---------- 5. filter evaluation with a held-out split (per label)
rng = np.random.default_rng(0)
half = rng.random(len(lab)) < 0.5          # A = derive, B = report
def score_rule(keep_mask):
    """keep_mask over rows of M (kept features); a label is detected if any kept row survives"""
    det = np.zeros(len(lab), bool)
    sub = M[M["kept"].values & keep_mask]
    det[sub["label_i"].unique()] = True
    out = {}
    for name, h in (("A", half), ("B", ~half)):
        tp = (det & is_tp & h).sum(); fp = (det & ~is_tp & h).sum(); fn = (~det & is_tp & h).sum()
        out[name] = (tp, fp, tp / (tp + fn), tp / max(tp + fp, 1), 2 * tp / (2 * tp + fp + fn))
    return out
base = score_rule(np.ones(len(M), bool))
print("\n== current: half A TP %d FP %d rec %.3f prec %.3f F1 %.3f | half B TP %d FP %d rec %.3f prec %.3f F1 %.3f" % (base["A"] + base["B"]))
rules = {"mz_sd_ratio <=": ("mz_sd_ratio", "le", [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]),
         "mz_sd_ppm <=": ("mz_sd_ppm", "le", [2, 2.5, 3, 4, 5, 6]),
         "fwhm_rel <=": ("fwhm_rel", "le", [1.5, 2.0, 2.5, 3.0]),
         "gauss_r2 >=": ("gauss_r2", "ge", [0.3, 0.4, 0.5, 0.6, 0.7]),
         "ridge_ratio <=": ("ridge_ratio", "le", [0.05, 0.1, 0.2, 0.3, 0.4]),
         "far_presence <=": ("far_presence", "le", [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]),
         "tv_ratio <=": ("tv_ratio", "le", [1.1, 1.2, 1.3, 1.5, 2.0]),
         "n_lmax <=": ("n_lmax", "le", [1, 2, 3]),
         "ripple_rel <=": ("ripple_rel", "le", [0.03, 0.05, 0.08, 0.1, 0.15]),
         "snr >=": ("snr", "ge", [4, 5, 6, 8, 10, 15]),
         "score >=": ("score", "ge", [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]),
         "prominence_rel >=": ("prominence_rel", "ge", [0.3, 0.4, 0.5, 0.6])}
print("\nsingle rules (kept features only): threshold -> half A (TP lost, FP removed, F1) | half B (F1)")
for name, (c, op, ths) in rules.items():
    line = []
    for t in ths:
        v = M[c].values.astype(float)
        m = (v <= t) if op == "le" else (v >= t)
        m |= np.isnan(v)
        r = score_rule(m)
        line.append(f"{t:g}: -{base['A'][0] - r['A'][0]}TP -{base['A'][1] - r['A'][1]}FP F1 {r['A'][4]:.3f}|{r['B'][4]:.3f}")
    print(f"  {name:18s} " + "   ".join(line))
# isotope-aware: require iso_support unless snr >= t
print("\niso-aware: keep if iso_support or snr >= t")
for t in [5, 8, 10, 15, 20, 30, 50]:
    m = M["iso_support"].values.astype(bool) | (M["snr"].values >= t)
    r = score_rule(m)
    print(f"  t={t:3d}: half A -{base['A'][0] - r['A'][0]}TP -{base['A'][1] - r['A'][1]}FP rec {r['A'][2]:.3f} prec {r['A'][3]:.3f} F1 {r['A'][4]:.3f} | half B F1 {r['B'][4]:.3f}")
print("\niso-aware: keep if iso_support or score >= t")
for t in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
    m = M["iso_support"].values.astype(bool) | (M["score"].values >= t)
    r = score_rule(m)
    print(f"  t={t:.2f}: half A -{base['A'][0] - r['A'][0]}TP -{base['A'][1] - r['A'][1]}FP rec {r['A'][2]:.3f} prec {r['A'][3]:.3f} F1 {r['A'][4]:.3f} | half B F1 {r['B'][4]:.3f}")

# ---------- 6. ceiling: cross-validated classifier on descriptors (TP vs TN kept features) -- diagnostic only
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    feats = ["log_h", "snr", "n_scans", "fwhm_rel", "mz_sd_ppm", "mz_sd_ratio", "prominence_rel", "gauss_r2", "ridge_ratio", "background_ratio",
             "iso_support", "n_lmax", "tv_ratio", "gap_frac", "ripple_rel", "apex_sharp", "far_presence", "far_med_rel", "near_cont", "asym",
             "smooth_prom_rel", "n_pts_per_scan"]
    X = K_[feats].astype(float).fillna(0).values; y = K_["tp"].values.astype(int)
    for name, clf in (("HGB", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4)),):
        p = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
        print(f"\nceiling ({name}, 5-fold CV on kept TP/TN features): AUC {roc_auc_score(y, p):.3f}")
        for thr in [0.3, 0.4, 0.5, 0.6]:
            keep = p >= thr
            print(f"   p>={thr}: keep TP {keep[y == 1].mean():.3f}, remove TN {1 - keep[y == 0].mean():.3f}")
        clf.fit(X, y)
        from sklearn.inspection import permutation_importance
        imp = permutation_importance(clf, X, y, n_repeats=3, random_state=0, scoring="roc_auc").importances_mean
        print("   permutation importance:", dict(sorted(zip(feats, imp.round(3)), key=lambda x: -x[1])[:10]))
except Exception as e:
    print("sklearn step skipped:", e)
