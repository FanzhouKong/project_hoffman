# LI2018: are the anchors bimodal in shift, and are the shifted truth features the abundant/differential ones?
import numpy as np, pandas as pd, sys
sys.path.insert(0, "bench/diag")
from rt_external_validation import collect
R = "results/peak3d/LI2018/peak3d_out"
A = pd.read_csv(f"{R}/rt_correction/anchors.tsv", sep="\t")
pd.set_option("display.width", 250)
NAN = float("nan")
print("=== anchors: shift implied by anchors (consensus - native), per RT bin ===")
for f in ["SB4", "SB3", "SB5", "SA1", "SA3"]:
    a = A[A.file == f].copy(); a["bin"] = (a.rt_native // 5 * 5).astype(int)
    rows = []
    for b, x in a.groupby("bin"):
        d = x.residual_before_s.values
        h = np.histogram(d, bins=np.arange(-12, 12.5, 1.0))[0]
        rows.append(dict(bin=b, n=len(x), n_inlier=int(x.is_inlier.sum()), med=np.median(d), q25=np.percentile(d, 25), q75=np.percentile(d, 75),
                         frac_gt2p5=float((np.abs(d) > 2.5).mean()), w_sum=float(x.weight.sum()), clw_med=float(np.median(x.cluster_w)), hist_1s=" ".join(str(v) for v in h)))
    print("---", f); print(pd.DataFrame(rows).round(2).to_string(index=False))
print()
print("=== truth features 4-15 min: arm (SB earlier than SA by >2.5 s) vs core (|SB-SA|<1 s) ===")
tr, stems, dt, nat, cor = collect("LI2018")
ok = np.isfinite(nat).sum(1) >= 5; tr = tr[ok].reset_index(drop=True); nat = nat[ok]
ref = np.nanmedian(nat, axis=1); sh = 60 * (ref[:, None] - nat)
sa = [i for i, s in enumerate(stems) if s.startswith("SA")]; sb = [i for i, s in enumerate(stems) if s.startswith("SB")]
win = (ref >= 4) & (ref <= 15)
sb_shift = np.nanmedian(sh[:, sb], axis=1); sa_shift = np.nanmedian(sh[:, sa], axis=1)
arm = win & (sb_shift - sa_shift > 2.5); core = win & (np.abs(sb_shift - sa_shift) < 1.0)
print("in window:", int(win.sum()), "arm:", int(arm.sum()), "core:", int(core.sum()))
maf = pd.read_csv("data/raw/LI2018_QE/m_MTBLS733_mass_spectrometry_v2_maf.tsv", sep="\t")
sc = [c for c in maf.columns if c.startswith(("SA", "SB"))]; print("MAF sample cols:", sc[:12], "n", len(sc))
feat = {s: pd.read_csv(f"{R}/features/{s}.tsv", sep="\t", usecols=["mz", "rt", "height", "fwhm"]) for s in ["SB4", "SA5"]}
def hz(stem, idx):
    f = feat[stem]; out = []
    for i in idx:
        k = np.flatnonzero((np.abs(f.mz.values - tr.mz.values[i]) / tr.mz.values[i] * 1e6 <= 5) & (np.abs(f.rt.values - nat[i, stems.index(stem)]) <= 0.05))
        out.append(f.iloc[k[np.argmax(f.height.values[k])]] if len(k) else None)
    return [r for r in out if r is not None]
for name, mask in [("arm", arm), ("core", core)]:
    idx = np.flatnonzero(mask); m = tr.iloc[idx]
    r4 = hz("SB4", idx); r5 = hz("SA5", idx)
    h4 = [r.height for r in r4]; h5 = [r.height for r in r5]; fw = [60 * r.fwhm for r in r4]
    rat = []
    for mz, rt in zip(m.mz.values, m.rt.values):
        k = np.flatnonzero((np.abs(maf.mass_to_charge.values - mz) < 1e-4) & (np.abs(maf.retention_time.values - rt) < 1e-3))
        if len(k) and sc:
            r = maf.iloc[k[0]]; A_ = np.nanmean([r[c] for c in sc if c.startswith("SA")]); B_ = np.nanmean([r[c] for c in sc if c.startswith("SB")])
            rat.append(np.log2(B_ / A_) if A_ > 0 and B_ > 0 else NAN)
    rat = np.array(rat) if rat else np.array([NAN])
    print(f"--- {name}: n={len(idx)}  height SB4 median {np.median(h4):.3g} (p25 {np.percentile(h4, 25):.3g}, p75 {np.percentile(h4, 75):.3g}); height SA5 median {np.median(h5):.3g}; "
          f"height ratio SB4/SA5 median {np.median(np.array(h4[:min(len(h4), len(h5))]) / np.array(h5[:min(len(h4), len(h5))])):.2f}; FWHM SB4 median {np.median(fw):.1f} s; "
          f"MAF log2(SB/SA) median {np.nanmedian(rat):.2f}, frac |log2|>1: {np.nanmean(np.abs(rat) > 1):.2f}")
    print("   example m/z:", np.round(m.mz.values[:8], 4), "rt:", np.round(m.rt.values[:8], 2))
a = A[(A.file == "SB4") & (A.rt_native >= 4) & (A.rt_native <= 15)].copy()
f = pd.read_csv(f"{R}/features/SB4.tsv", sep="\t", usecols=["feature_id", "height"]).set_index("feature_id")
a["height"] = f.height.reindex(a.feature_id).values
big = a[a.residual_before_s.abs() > 2.5]; small = a[a.residual_before_s.abs() < 1]
print(f"SB4 anchors 4-15 min: big-shift n={len(big)} height med {big.height.median():.3g} inlier frac {big.is_inlier.mean():.2f}; small-shift n={len(small)} height med {small.height.median():.3g} inlier frac {small.is_inlier.mean():.2f}")
