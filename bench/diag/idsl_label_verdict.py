# diagnostic: per-label verdicts for the 20,000 IDSL.IPA labels from the raw evidence written by idsl_label_audit.py
# (labelled file 003 + recurrence in 30 neighbouring injections), a cross-validated label-disagreement model, and the
# effect of the questionable labels on every tool's IDSL003 score.
# usage: idsl_label_verdict.py AUDIT_DIR   (masscube env: sklearn)  -> AUDIT_DIR/verdict.tsv, stdout = log
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.score import load, hits  # noqa: E402
from bench.runs import TOOLS  # noqa: E402

D = Path(sys.argv[1])
A = pd.read_csv(D / "labels_003.tsv", sep="\t")
R = np.load(D / "replicates.npz")
files = list(R["files"])
assert files[0] == "003"
tp = (A["Manual Curation"] == "TP").values
FLOOR = 131.5  # q05 of centroid intensities in 003 (audit log)
pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)


def present(w, strict=True, rows=slice(1, None)):
    g = lambda k: R[f"{w}_{k}"][rows]
    if strict:
        p = (g("pbr") >= 3) & (g("n_pts") >= 4) & (g("r2") >= 0.6) & (g("flank") == 0) & (g("prom") >= 0.5)
    else:
        p = (g("pbr") >= 2) & (g("n_pts") >= 3) & (g("flank") == 0)
    ok = np.isfinite(g("pbr"))
    return np.where(ok.sum(0) > 0, (p & ok).sum(0) / np.maximum(ok.sum(0), 1), np.nan)


A["rep"] = present("tgt")
A["rep_len"] = present("tgt", False)
A["decoy"] = np.nanmean(np.vstack([present("dlo"), present("dhi")]), 0)
A["self003"] = present("tgt", rows=slice(0, 1))  # our own conversion of the same file
n_rep = len(files) - 1

# ---- evidence in the labelled file
h, pbr, npt, r2, prom = A.h.values, A.pbr.values, A.n_pts.values, A.r2.values, A.prom.values
flank = A.flank.values.astype(bool)
mz_off = (np.abs(A.d_ppm_dom.values) > 10) & (A.dom_ratio.values >= 3)
peak = (pbr >= 3) & (npt >= 4) & (r2 >= 0.6) & (prom >= 0.5) & ~flank
clear = peak & (pbr >= 5) & (npt >= 5) & (r2 >= 0.8) & (prom >= 0.7)
nothing = (h < 3 * FLOOR) | (npt < 3)
cls = np.select([nothing, flank, mz_off & ~peak, clear, peak], ["nothing", "flank", "mz_off", "clear_peak", "peak"],
                "no_local_peak")
A["cls003"] = cls
at_apex = (np.abs(A.drt_s.values) <= 1.5) & (np.abs(A.d_ppm.values) <= 5)  # label sits on the apex it scores against
A["at_apex"] = at_apex
rep = A.rep.values
recurs, absent = rep >= 0.5, rep <= 0.1

print(f"labels: {len(A)} (TP {tp.sum()}, TN {(~tp).sum()}); replicate injections: {n_rep}; floor {FLOOR}")
print(f"own conversion of 003 vs Zenodo 003: strict peak test agrees for "
      f"{(peak == (A.self003.values >= 0.5)).mean():.3f} of labels\n")
print("== evidence class in 003 x label")
print(pd.crosstab(A.cls003, A["Manual Curation"], margins=True))
print("\n== recurrence (strict peak test, fraction of", n_rep, "injections) x label; decoy = same m/z, RT +/-0.6 min")
bins = [-0.01, 0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
print(pd.crosstab(pd.cut(A.rep, bins), A["Manual Curation"]))
for nm, m in (("TP", tp), ("TN", ~tp)):
    print(f"  decoy recurrence {nm}: mean {np.nanmean(A.decoy[m]):.3f}, P(decoy >= 0.5) {np.nanmean(A.decoy[m] >= 0.5):.4f}")
print("\n== class x recurrence, TP labels")
rb = pd.cut(A.rep, [-0.01, 0.1, 0.5, 1.0], labels=["absent<=0.1", "some", "recurs>=0.5"])
print(pd.crosstab(A.cls003[tp], rb[tp], margins=True))
print("\n== class x recurrence, TN labels")
print(pd.crosstab(A.cls003[~tp], rb[~tp], margins=True))

# ---- label-label neighbours: same peak or not?
o = np.argsort(A["m/z"].values)
ms, ts = A["m/z"].values[o], A["RT(min)"].values[o]
pairs = []
for j in range(len(ms)):
    hi = np.searchsorted(ms, ms[j] * (1 + 10e-6))
    for k in range(j + 1, hi):
        if abs(ts[k] - ts[j]) <= 0.1:
            pairs.append((o[j], o[k]))
P = pd.DataFrame(pairs, columns=["a", "b"])
P["la"], P["lb"] = A["Manual Curation"].values[P.a], A["Manual Curation"].values[P.b]
P["kind"] = np.where(P.la == P.lb, P.la + "-" + P.lb, "TP-TN")
P["same_apex"] = (np.abs(A.apex_rt.values[P.a] - A.apex_rt.values[P.b]) * 60 <= 1.0)
P["drt_s"] = np.abs(A["RT(min)"].values[P.a] - A["RT(min)"].values[P.b]) * 60
P["dppm"] = np.abs(A["m/z"].values[P.a] - A["m/z"].values[P.b]) / A["m/z"].values[P.a] * 1e6
print("\n== label pairs within 10 ppm and 0.1 min (the scoring tolerance)")
print(P.groupby("kind").agg(n=("a", "size"), same_apex=("same_apex", "mean"), med_drt_s=("drt_s", "median"),
                            med_dppm=("dppm", "median")))
conf_tn = np.zeros(len(A), bool)
x = P[P.kind == "TP-TN"]
conf_tn[np.where(x.la.values == "TN", x.a.values, x.b.values)] = True

# ---- verdicts
v = np.full(len(A), "", object)
# TP labels
v[tp & peak & recurs] = "TP confirmed (peak here, recurs)"
v[tp & (v == "") & (peak | clear) & ~absent] = "TP probable (peak here, recurs sometimes)"
v[tp & (v == "") & clear & absent] = "TP probable (clear peak, this sample only)"
v[tp & (v == "") & (flank | mz_off) & ~nothing] = "TP misplaced (on flank / m/z edge of a peak)"
v[tp & (v == "") & ~peak & recurs] = "TP plausible (weak here, peak in other injections)"
v[tp & (v == "") & (nothing | (cls == "no_local_peak")) & absent] = "TP unsupported (no peak here or elsewhere)"
v[tp & (v == "")] = "TP uncertain"
# TN labels
v[~tp & conf_tn] = "TN conflicting (TP label within tolerance)"
v[~tp & (v == "") & clear & recurs & at_apex] = "TN contradicted (clear peak at label, recurs)"
v[~tp & (v == "") & peak & recurs & ~at_apex] = "TN off-apex (real peak within tolerance)"
v[~tp & (v == "") & peak & recurs] = "TN doubtful (peak at label, recurs)"
v[~tp & (v == "") & ~peak & absent] = "TN confirmed (no peak here or elsewhere)"
v[~tp & (v == "") & (flank | mz_off)] = "TN defensible (flank / m/z edge)"
v[~tp & (v == "")] = "TN uncertain"
A["verdict"] = v
print("\n== verdicts")
print(A.verdict.value_counts().sort_index().to_string())

# published tools on each verdict (TP/FP = the tool reported the pair)
print("\n== fraction of each verdict reported by the published tools (author columns)")
pub = ["IDSL.IPA", "XCMS", "MZMINE", "MSDIAL"]
rep_by = pd.DataFrame({c: A[c].isin(["TP", "FP"]) for c in pub})
rep_by["n_tools"] = rep_by.sum(1)
print(pd.concat([A.verdict, rep_by], axis=1).groupby("verdict").mean().round(3).to_string())

# ---- cross-validated disagreement model (confident-learning style)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
X = pd.DataFrame(dict(log_h=np.log10(np.maximum(h, 1)), pbr=np.log10(np.maximum(pbr, 1e-3)), n_pts=npt, r2=r2, prom=prom,
                      fwhm=A.fwhm, sig=A.sig, bg_nz=A.bg_nz, flank=flank.astype(int), climb=A.climb_ratio,
                      adrt=np.abs(A.drt_s), adppm=np.abs(A.d_ppm), mzsd=A.mz_sd_ppm, dppm_dom=np.abs(A.d_ppm_dom),
                      dom_ratio=A.dom_ratio, corr_p1=A.corr_p1, r_p1=A.r_p1, corr_m1=A.corr_m1, r_m1=A.r_m1,
                      rep=A.rep, rep_len=A.rep_len, decoy=A.decoy))
p_oof = np.zeros(len(A))
for seed in range(3):
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, tp):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed).fit(X.iloc[tr], tp[tr])
        p_oof[te] += m.predict_proba(X.iloc[te])[:, 1] / 3
A["p_tp"] = p_oof
print(f"\n== out-of-fold model on raw evidence: AUC {roc_auc_score(tp, p_oof):.3f}")
for t in (0.9, 0.8):
    print(f"  TN labels with p(TP) >= {t}: {((~tp) & (p_oof >= t)).sum()}   TP labels with p(TP) <= {1 - t:.1f}: "
          f"{(tp & (p_oof <= 1 - t)).sum()}")
print("  p(TP) by verdict (median, q10, q90):")
print(A.groupby("verdict").p_tp.describe(percentiles=[.1, .5, .9])[["count", "10%", "50%", "90%"]].round(3).to_string())

# ---- effect on scores (10 ppm, 0.1 min)
def f1(det, lab_tp, keep):
    d, t = det[keep], lab_tp[keep]
    TP, FP, FN = (d & t).sum(), (d & ~t).sum(), (~d & t).sum()
    return dict(TP=int(TP), FP=int(FP), FN=int(FN), prec=TP / max(TP + FP, 1), rec=TP / max(TP + FN, 1),
                F1=2 * TP / max(2 * TP + FP + FN, 1))


det = {f"{c} (published)": rep_by[c].values for c in pub}
for tool in TOOLS:
    x = load(tool, "IDSL003")
    if x is not None:
        det[tool] = hits(x[0], A["m/z"].values, A["RT(min)"].values, 10, 0.1)
questionable = A.verdict.str.contains("unsupported|contradicted|conflicting|misplaced|off-apex").values
flip = tp.copy()
flip[A.verdict.str.contains("TP unsupported").values] = False
flip[A.verdict.str.contains("TN contradicted").values] = True
keep_all = np.ones(len(A), bool)
rows = []
for nm, d in det.items():
    for scen, lab_tp, keep in (("as published", tp, keep_all), ("questionable labels removed", tp, ~questionable),
                               ("unsupported TP->TN, contradicted TN->TP", flip, keep_all)):
        rows.append(dict(tool=nm, scenario=scen, **f1(d, lab_tp, keep)))
S = pd.DataFrame(rows)
print(f"\n== scores at 10 ppm / 0.1 min; questionable labels = {questionable.sum()} "
      f"(TP {(questionable & tp).sum()}, TN {(questionable & ~tp).sum()})")
print(S.pivot(index="tool", columns="scenario", values="F1").round(3).to_string())
print()
print(S.round(3).to_string(index=False))
A.to_csv(D / "verdict.tsv", sep="\t", index=False)
P.to_csv(D / "label_pairs.tsv", sep="\t", index=False)
S.to_csv(D / "score_by_scenario.tsv", sep="\t", index=False)
