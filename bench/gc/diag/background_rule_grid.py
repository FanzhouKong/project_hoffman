import numpy as np, pandas as pd
A = pd.read_csv("results/gc_mrh/diag_bg/candidates.tsv", sep="\t", low_memory=False, dtype={"level": str})
A["label"] = A["label"].fillna("")
B = A[A.state == "bg_rejected"].copy()
nfile = A["level"].nunique(); acc = (A.state == "accepted").sum() / nfile
den = np.maximum(1 - B["floor_rel"], 1e-9)
B["ridge_net"] = (B["ridge_ratio"] - B["floor_rel"]) / den
B["far_net"] = (B["far_level"] - B["floor_rel"]) / den
print(B.groupby(B.label.replace("", "unlabelled"))[["ridge_net", "far_net", "ripple_ratio", "fwhm_net_rel"]].quantile([.1, .5, .9]).round(2).to_string())
rows = []
for k in (3, 5, 8):
    for rmax in (0.5, 0.33, 0.25):
        for lo, hi in ((0.3, 99), (0.5, 99), (0.5, 3.0)):
            p = (B["ripple_ratio"] >= k) & (B["ridge_net"] <= rmax) & (B["fwhm_net_rel"] >= lo) & (B["fwhm_net_rel"] <= hi)
            rows.append(dict(ripple_k=k, ridge_net_max=rmax, width=f"{lo}-{hi}", compound=int((p & (B.label == "compound")).sum()),
                             mixed=int((p & (B.label == "mixed")).sum()), flat=int((p & (B.label == "flat")).sum()),
                             unlabelled_per_file=int((p & (B.label == "")).sum() / nfile), pct_of_accepted=round(100 * (p & (B.label == "")).sum() / nfile / acc, 1)))
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).to_string(index=False))
p = (B["ripple_ratio"] >= 5) & (B["ridge_net"] <= 0.5) & (B["fwhm_net_rel"] >= 0.5)
order = ["0_1", "0_5", "1", "5", "10", "25", "50", "100", "250", "500", "1000"]
print("admitted per level:", B[p].groupby("level").size().reindex(order).to_dict())
u = B[p & (B.label == "")]
print(u[["height", "floor_rel", "ripple_ratio", "ridge_net", "fwhm_net_rel", "gauss_r2", "snr"]].quantile([.05, .25, .5, .75, .95]).round(2).to_string())
# are the admitted unlabelled ones co-eluting with accepted features (i.e. fragments of real compounds)?
acc_rt = {lv: np.sort(g["rt"].values) for lv, g in A[A.state == "accepted"].groupby("level")}
def n_near(lv, rt, tol=0.3 / 60):
    a = acc_rt[lv]; return np.searchsorted(a, rt + tol) - np.searchsorted(a, rt - tol)
u = u.assign(n_coel=[n_near(lv, rt) for lv, rt in zip(u["level"], u["rt"])])
rnd = B[(B.label == "")].sample(20000, random_state=1); rnd = rnd.assign(n_coel=[n_near(lv, rt) for lv, rt in zip(rnd["level"], rnd["rt"])])
print("accepted features within 0.3 s of the apex: admitted median", u["n_coel"].median(), "share >=3:", round((u["n_coel"] >= 3).mean(), 3), "share >=10:", round((u["n_coel"] >= 10).mean(), 3),
      "| all bg-rejected: median", rnd["n_coel"].median(), "share >=3:", round((rnd["n_coel"] >= 3).mean(), 3), "share >=10:", round((rnd["n_coel"] >= 10).mean(), 3))
