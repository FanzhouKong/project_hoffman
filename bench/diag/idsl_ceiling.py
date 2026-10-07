# diagnostic: how separable are kept features at TP vs TN positions with ALL descriptors (cross-validated
# gradient boosting = a ceiling for any rule-based filter; not a deployable model, the labels are the benchmark)
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
ROOT = Path(__file__).resolve().parents[2]
M = pd.read_csv(ROOT / "results/diag_idsl/pairs.tsv", sep="\t")
P = json.load(open(ROOT / "results/diag_idsl/params.json"))
M["mz_sd_ratio"] = M["mz_sd_ppm"] / M["sig_apex"]; M["fwhm_rel"] = M["fwhm"] / P["fwhm_med"]; M["log_h"] = np.log10(M["height"])
M["log_snr"] = np.log10(M["snr"].clip(lower=1e-3))
K_ = M[M["kept"]].drop_duplicates("cand_i").copy()
# a feature matching both a TP and a TN label counts as TP (the TN is a matching artefact)
tp_any = M[M["kept"]].groupby("cand_i")["tp"].max()
K_["y"] = tp_any.loc[K_["cand_i"]].values.astype(int)
sets = {
 "picker descriptors only": ["log_h", "log_snr", "n_scans", "fwhm_rel", "mz_sd_ppm", "mz_sd_ratio", "prominence_rel", "gauss_r2", "ridge_ratio", "background_ratio", "iso_support", "asym", "n_gaps", "persistence_rel"],
 "+ trace shape (tv, lmax, ripple)": ["tv_ratio", "n_lmax", "ripple_rel", "apex_sharp", "smooth_prom_rel", "near_cont", "gap_frac"],
 "+ localisation (far_med_rel, far_presence)": ["far_med_rel", "far_presence"],
}
feats = []
y = K_["y"].values
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for name, add in sets.items():
    feats = feats + add
    X = K_[feats].astype(float).fillna(0).values
    for mname, clf in (("logistic", make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))),
                       ("HGB", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=30))):
        p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, p)
        # operating points: fraction of TN removed at given TP loss
        o = np.argsort(p)
        line = []
        for loss in (0.02, 0.05, 0.10):
            thr = np.quantile(p[y == 1], loss)
            line.append(f"TP loss {loss:.0%}: TN removed {np.mean(p[y == 0] < thr):.1%}")
        print(f"{name:45s} {mname:9s} AUC {auc:.3f} | " + " ; ".join(line))
clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=30).fit(X, y)
imp = permutation_importance(clf, X, y, n_repeats=5, random_state=0, scoring="roc_auc").importances_mean
print("\npermutation importance (AUC drop):", dict(sorted(zip(feats, imp.round(4)), key=lambda x: -x[1])))
# F1 implication: with p threshold chosen for 5 % TP loss, on the label level
