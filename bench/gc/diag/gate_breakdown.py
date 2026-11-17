import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from peak3d import kernels as K
from peak3d.features import RIDGE_MAX, BACKGROUND_MAX
from peak3d.gc.deconv import params_from_dict

out = Path("results/gc_mrh/peak3d"); T = pd.read_csv("results/gc_mrh/score/truth_rows.tsv", sep="\t")
T = T[T["clean"] & ~T["found"]]
rows = []
for lv, t in T.groupby("level"):
    stem = f"NEWSTD_{lv}"
    P = params_from_dict(json.loads((out / "features" / f"{stem}.params.json").read_text()))
    R = pd.read_csv(out / "features" / f"{stem}.rejected.tsv", sep="\t", low_memory=False)
    F = pd.read_csv(out / "features" / f"{stem}.tsv", sep="\t")
    for _, r in t.iterrows():
        d = dict(level=lv, entry=r["entry"], mz=r["mz"], rt=r["rt"], intensity=r["intensity"], outcome=r["outcome"])
        if r["outcome"] == "inside another peak":
            f = F[(np.abs(F["mz"] - r["mz"]) <= r["mz"] * 1e-5) & (F["rt_min"] <= r["rt"]) & (F["rt_max"] >= r["rt"])]
            if len(f):
                f = f.iloc[np.argmin(np.abs(f["rt"].values - r["rt"]))]
                d.update(why="inside", d_rt_s=(f["rt"] - r["rt"]) * 60, height=f["height"], width_s=(f["rt_max"] - f["rt_min"]) * 60,
                         fwhm_s=f["fwhm"] * 60, n_sat=f.get("n_sat", np.nan), flags=f.get("flags", ""))
            rows.append(d); continue
        if r["outcome"] != "rejected gate":
            rows.append(d); continue
        c = R[(R["reason"] == "gate") & (np.abs(R["mz"] - r["mz"]) <= r["mz"] * 1e-5) & (np.abs(R["rt"] - r["rt"]) * 60 <= 2.0)]
        if not len(c):
            d["why"] = "?"; rows.append(d); continue
        c = c.iloc[np.argmin(np.abs(c["rt"].values - r["rt"]))]
        why = []
        if c["n_scans"] < c["min_scans_local"]: why.append("scans")
        if c["snr"] < P.min_snr: why.append("snr")
        if c["mz_sd_ppm"] > 3 * c["sig_apex"] + 1: why.append("mz_sd")
        thr = c["narrow_thr"] if np.isfinite(c.get("narrow_thr", np.nan)) else 0.3 * P.fwhm_med
        if not (c["fwhm"] + (c.get("dt_local", 0.0) if P.width_rule == "time" else 0.0) >= thr): why.append("narrow")
        if c["prominence_rel"] < P.min_prominence_rel: why.append("prominence")
        if bool(c.get("fragment", False)): why.append("fragment")
        kf = int(c["kflags"])
        if (kf & K.KF_CAP_L) and (kf & K.KF_CAP_R) and c["prominence_rel"] < 0.5: why.append("cap_both")
        if c["ridge_ratio"] > RIDGE_MAX: why.append("ridge")
        if c["background_ratio"] > BACKGROUND_MAX: why.append("background")
        if P.ripple_k > 0 and c["ripple_ratio"] < P.ripple_k: why.append("ripple")
        if P.max_valleys >= 0 and c["n_valleys"] > P.max_valleys: why.append("valleys")
        if P.far_level_max < 1 and c["far_level"] > P.far_level_max: why.append("far_level")
        d.update(why="+".join(why) or "none?", height=c["height"], snr=c["snr"], n_scans=c["n_scans"], min_scans=c["min_scans_local"],
                 fwhm_s=c["fwhm"] * 60, thr_s=thr * 60, prom=c["prominence_rel"], mz_sd=c["mz_sd_ppm"], sig=c["sig_apex"],
                 ridge=c["ridge_ratio"], ripple=c["ripple_ratio"], far=c["far_level"])
        rows.append(d)
    0 and print(lv, "min_snr", P.min_snr, "min_scans", P.min_scans, "fwhm_med_s", round(P.fwhm_med * 60, 2), "width_rule", P.width_rule,
          "narrow_rule", P.narrow_rule, "ripple_k", P.ripple_k, "min_prom", P.min_prominence_rel, "far_max", P.far_level_max, file=sys.stderr)
D = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200); pd.set_option("display.max_columns", 40)
g = D[D["outcome"] == "rejected gate"].copy()
def cls(w):
    p = set(w.split("+"))
    if p & {"scans", "snr", "narrow"} and not p & {"ridge", "background", "far_level", "cap_both"}: return "too weak or short (scans / snr / narrow)"
    if p & {"ridge", "background", "far_level", "cap_both"} and not p & {"scans", "snr", "narrow"}: return "on a background of the same ion (ridge / background / far_level)"
    return "other: " + w
g["class"] = g["why"].map(cls)
order = ["0_1", "0_5", "1", "5", "10", "25", "50", "100", "250", "500", "1000"]
print(pd.crosstab(g["class"], g["level"]).reindex(columns=order, fill_value=0).assign(all=lambda d: d.sum(axis=1)).to_string())
print(g.groupby("class")[["height", "snr", "n_scans", "fwhm_s", "prom", "far", "ridge"]].median().round(2).to_string())
b = g[g["class"].str.startswith("on a background")]
T2 = pd.read_csv("results/gc_mrh/score/truth_rows.tsv", sep="\t")
b = b.merge(T2[["level", "entry", "mz", "response_slope", "mz_expected"]].astype({"level": str}), on=["level", "entry", "mz"], how="left")
print("== background class: response slope quartiles", b["response_slope"].quantile([.1, .25, .5, .75, .9]).round(2).tolist())
print("== background class: FWHM (s) quartiles", b["fwhm_s"].quantile([.1, .25, .5, .75, .9]).round(2).tolist())
print("== background class by expected ion (top 15)"); print(b.assign(ion=b["mz_expected"].round(3)).groupby("ion").agg(n=("mz", "size"), entries=("entry", "nunique"), slope=("response_slope", "median"), fwhm_s=("fwhm_s", "median"), height=("height", "median"), prom=("prom", "median"), far=("far", "median")).sort_values("n", ascending=False).head(15).round(2).to_string())
print("== background class, examples with the highest slope"); print(b.sort_values("response_slope", ascending=False).head(12)[["level", "entry", "mz", "intensity", "height", "snr", "n_scans", "fwhm_s", "prom", "far", "ridge", "response_slope", "why"]].round(3).to_string(index=False))
i = D[D["outcome"] == "inside another peak"]; print("== inside another peak"); print(i[["level", "entry", "mz", "rt", "intensity", "height", "d_rt_s", "width_s", "fwhm_s", "flags"]].round(3).to_string(index=False))
n = D[D["outcome"] == "no candidate"]; print("== no candidate: intensity quartiles", n["intensity"].quantile([.25, .5, .75]).round(0).tolist(), "n", len(n))
