"""Dry run for a GC background rule: every candidate the LC background family alone rejects (ridge / background /
far_level / cap_both), plus the accepted ones, with net descriptors measured on the raw trace; labelled by the MRH
truth (clean = compound ion, flat = recorded at the right m/z and RT but intensity does not follow the dose)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from peak3d import kernels as K
from peak3d.features import RIDGE_MAX, BACKGROUND_MAX
from peak3d.gc.deconv import params_from_dict
from peak3d.io import Cloud

out = Path("results/gc_mrh/peak3d"); dst = Path(sys.argv[1]); dst.mkdir(parents=True, exist_ok=True)
T = pd.read_csv("results/gc_mrh/score/truth_rows.tsv", sep="\t", dtype={"level": str})
LEVELS = ["0_1", "0_5", "1", "5", "10", "25", "50", "100", "250", "500", "1000"]
allc = []
for lv in LEVELS:
    stem = f"NEWSTD_{lv}"
    P = params_from_dict(json.loads((out / "features" / f"{stem}.params.json").read_text()))
    cloud = Cloud.load_npz(out / ".cache" / f"{stem}.npz")
    R = pd.read_csv(out / "features" / f"{stem}.rejected.tsv", sep="\t", low_memory=False)
    F = pd.read_csv(out / "features" / f"{stem}.tsv", sep="\t", low_memory=False)
    R = R[R["reason"] == "gate"].copy()
    kf = R["kflags"].values.astype(np.int64)
    cap_both = ((kf & K.KF_CAP_L) > 0) & ((kf & K.KF_CAP_R) > 0) & (R["prominence_rel"].values < 0.5)
    thr = np.where(np.isfinite(R["narrow_thr"].values), R["narrow_thr"].values, 0.3 * P.fwhm_med)
    other = (R["n_scans"] >= R["min_scans_local"]) & (R["snr"] >= P.min_snr) & (R["mz_sd_ppm"] <= 3 * R["sig_apex"] + 1) \
        & (R["fwhm"] >= thr) & (R["prominence_rel"] >= P.min_prominence_rel) & ~R["fragment"].astype(bool)
    bg = (R["ridge_ratio"] > RIDGE_MAX) | (R["background_ratio"] > BACKGROUND_MAX) | (R["far_level"] > P.far_level_max) | cap_both
    R = R[other.values & bg.values].assign(state="bg_rejected", cap_both=cap_both[other.values & bg.values])
    F = F.assign(state="accepted", cap_both=False)
    cols = ["mz", "rt", "rt_min", "rt_max", "scan_apex", "height", "snr", "n_scans", "fwhm", "prominence_rel", "sig_apex", "ridge_ratio",
            "background_ratio", "ripple_ratio", "far_level", "gauss_r2", "asym", "baseline", "noise", "state", "cap_both"]
    C = pd.concat([F[[c for c in cols if c in F]], R[[c for c in cols if c in R]]], ignore_index=True)
    # net descriptors on the raw trace
    w = P.fwhm_med / P.dt
    near, far = max(1, int(round(2 * w))), max(3, int(round(6 * w)))
    S = cloud.n_scans
    n = len(C); floor = np.zeros(n); fw_net = np.full(n, np.nan); side_noise = np.zeros(n); floor_hi = np.zeros(n)
    mzv, sa, hv, sg = C["mz"].values, C["scan_apex"].values.astype(int), C["height"].values, C["sig_apex"].values
    for i in range(n):
        ppm = max(3 * sg[i] + 1, 5.0)
        lo, hi = max(0, sa[i] - far), min(S, sa[i] + far + 1)
        y = np.asarray(cloud.eic(float(mzv[i]), ppm, lo, hi), np.float64)
        c = sa[i] - lo
        L, Rr = y[:max(0, c - near + 1)], y[c + near:]
        meds = [np.median(v) for v in (L, Rr) if len(v) >= 3]
        b = min(meds) if meds else 0.0
        floor[i] = b; floor_hi[i] = max(meds) if meds else 0.0
        v = L if (len(L) >= 3 and (len(Rr) < 3 or np.median(L) <= np.median(Rr))) else Rr
        side_noise[i] = (np.median(np.abs(np.diff(v))) / 0.6745 / np.sqrt(2)) if len(v) >= 4 else 0.0
        h = max(hv[i], y[c]); half = b + 0.5 * (h - b)
        if h <= b:
            continue
        j = c
        while j > 0 and y[j - 1] >= half: j -= 1
        tl = j - (y[j] - half) / max(y[j] - y[j - 1], 1e-9) if j > 0 else np.nan
        j = c
        while j < len(y) - 1 and y[j + 1] >= half: j += 1
        tr = j + (y[j] - half) / max(y[j] - y[j + 1], 1e-9) if j < len(y) - 1 else np.nan
        fw_net[i] = (tr - tl) * P.dt * 60
    C["floor_rel"] = floor / np.maximum(hv, 1e-9); C["floor_hi_rel"] = floor_hi / np.maximum(hv, 1e-9)
    C["fwhm_net_s"] = fw_net; C["net_snr"] = (hv - floor) / np.maximum(side_noise, 1e-9)
    C["fwhm_net_rel"] = fw_net / (P.fwhm_med * 60); C["fwhm_s"] = C["fwhm"] * 60; C["level"] = lv
    # label by truth
    t = T[(T["level"] == lv) & T["matched"] & T["mass_ok"] & T["rt_ok"] & T["raw_peak"] & T["raw_apex"]]
    lab = np.full(n, "", dtype=object); slope = np.full(n, np.nan)
    for _, r in t.iterrows():
        k = np.flatnonzero((np.abs(mzv - r["mz"]) <= r["mz"] * 1e-5) & (np.abs(C["rt"].values - r["rt"]) * 60 <= 2.0))
        if len(k):
            k = k[np.argmin(np.abs(C["rt"].values[k] - r["rt"]))]
            lab[k] = "compound" if r["responds"] else ("flat" if r["response_slope"] < 0.2 else "mixed"); slope[k] = r["response_slope"]
    C["label"] = lab; C["slope"] = slope
    allc.append(C)
    print(lv, "accepted", int((C.state == "accepted").sum()), "bg_rejected", int((C.state == "bg_rejected").sum()), flush=True)
A = pd.concat(allc, ignore_index=True)
A.to_csv(dst / "candidates.tsv", sep="\t", index=False, float_format="%.5g")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(pd.crosstab(A["state"], A["label"]))
q = [.05, .25, .5, .75, .95]
for st in ("accepted", "bg_rejected"):
    for lb in ("compound", "mixed", "flat", ""):
        g = A[(A.state == st) & (A.label == lb)]
        if len(g) < 5: continue
        print(f"== {st} / {lb or 'unlabelled'} (n={len(g)})")
        print(g[["height", "floor_rel", "floor_hi_rel", "prominence_rel", "fwhm_s", "fwhm_net_rel", "net_snr", "ripple_ratio", "far_level", "ridge_ratio", "background_ratio", "gauss_r2"]].quantile(q).round(2).to_string())
