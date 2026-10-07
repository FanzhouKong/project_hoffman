# diagnostic: pairs of kept features of the same ion inside ONE file; why are there two?
import sys, json
import numpy as np, pandas as pd
ROOT = "/quobyte/metabolomicsgrp/fanzhou/hoffmann"
run, stem = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("BM21_RP", "MT_20211013_082")
f = pd.read_csv(f"{ROOT}/results/peak3d/{run}/peak3d_out/features/{stem}.tsv", sep="\t")
P = json.load(open(f"{ROOT}/results/peak3d/{run}/peak3d_out/features/{stem}.params.json"))
dt, fwhm = P["dt"], P["fwhm_med"]
print(f"{stem}: {len(f)} features; dt {60*dt:.2f} s, FWHM_med {60*fwhm:.2f} s ({fwhm/dt:.1f} scans), sig_a {P['sig_a']:.2f} sig_b {P['sig_b']:.0f}, K {P['K']}")
mz, rt, h = f["mz"].values, f["rt"].values, f["height"].values
o = np.argsort(mz); smz = mz[o]
rows = []
for i in range(len(f)):
    lo = np.searchsorted(smz, mz[i] * (1 - 5e-6)); hi = np.searchsorted(smz, mz[i] * (1 + 5e-6), side="right")
    for j in o[lo:hi]:
        if j <= i or abs(rt[j] - rt[i]) * 60 > 6:    # within 6 s
            continue
        a, b = (i, j) if h[i] >= h[j] else (j, i)     # a strong, b weak
        rows.append(dict(a=a, b=b, dppm=(mz[b] - mz[a]) / mz[a] * 1e6, drt_s=60 * (rt[b] - rt[a]),
                         drt_fwhm=abs(rt[b] - rt[a]) / fwhm, h_ratio=h[b] / h[a],
                         overlap=min(f.rt_max[a], f.rt_max[b]) > max(f.rt_min[a], f.rt_min[b]),
                         pers_rel_b=f.persistence_rel[b], prom_b=f.prominence_rel[b], n_scans_b=f.n_scans[b], n_scans_a=f.n_scans[a],
                         fwhm_a_s=60 * f.fwhm[a], fwhm_b_s=60 * f.fwhm[b], sd_a=f.mz_sd_ppm[a], sd_b=f.mz_sd_ppm[b],
                         iso_b=f.iso_offset[b], score_b=f.score[b], flags_b=str(f["flags"][b]), gap_b=f.n_gaps[b],
                         adjacent=abs(f.scan_apex[a] - f.scan_apex[b]) <= max(f.scan_max[a] - f.scan_min[a], f.scan_max[b] - f.scan_min[b])))
R = pd.DataFrame(rows)
print(f"pairs within 5 ppm / 6 s: {len(R)} ({len(R)/len(f):.1%} of features have a partner)")
sig_b_ppm = np.sqrt(P["sig_a"] ** 2 + P["sig_b"] ** 2 / f.height.values[R["b"].values])
sig_a_ppm = np.sqrt(P["sig_a"] ** 2 + P["sig_b"] ** 2 / f.height.values[R["a"].values])
R["mz_split"] = np.abs(R["dppm"]) > 3 * np.sqrt(sig_a_ppm ** 2 + sig_b_ppm ** 2)
print("\nclassification of the weaker member:")
print(f"  m/z-split (|dppm| beyond the pair tolerance, same RT):        {R['mz_split'].sum():5d}")
rt_split = ~R["mz_split"]
print(f"  same m/z, RT-separated:                                       {rt_split.sum():5d}")
print(f"     of which |drt| < 0.8 FWHM (should have merged):            {(rt_split & (R['drt_fwhm'] < 0.8)).sum():5d}")
print(f"     0.8-2 FWHM apart (shoulder / dip > 15 %):                  {(rt_split & (R['drt_fwhm'] >= 0.8) & (R['drt_fwhm'] < 2)).sum():5d}")
print(f"     > 2 FWHM apart (distinct peaks, probably isomers):         {(rt_split & (R['drt_fwhm'] >= 2)).sum():5d}")
print(f"  RT ranges overlap: {R['overlap'].mean():.1%};  weaker is an annotated isotope: {(R['iso_b'] > 0).mean():.1%}")
print("\ndistributions for RT-separated pairs:")
S = R[rt_split]
print("  |drt| in FWHM: quantiles", np.percentile(S["drt_fwhm"], [10, 25, 50, 75, 90]).round(2).tolist())
print("  height ratio weak/strong:", np.percentile(S["h_ratio"], [10, 50, 90]).round(3).tolist())
print("  persistence_rel of weaker:", np.percentile(S["pers_rel_b"], [10, 50, 90]).round(3).tolist(), " prominence_rel:", np.percentile(S["prom_b"], [10, 50, 90]).round(3).tolist())
print("  weaker n_scans:", np.percentile(S["n_scans_b"], [10, 50, 90]).round(0).tolist(), " stronger n_scans:", np.percentile(S["n_scans_a"], [10, 50, 90]).round(0).tolist())
print("  weaker fwhm (s):", np.percentile(S["fwhm_b_s"], [10, 50, 90]).round(2).tolist(), " stronger fwhm (s):", np.percentile(S["fwhm_a_s"], [10, 50, 90]).round(2).tolist())
print("  weaker flags:", S["flags_b"].replace("nan", "").str.split(";").explode().value_counts().head(6).to_dict())
print("  weaker score:", np.percentile(S["score_b"], [10, 50, 90]).round(2).tolist())
print("\nm/z-split pairs: |dppm| quantiles", np.percentile(np.abs(R[R['mz_split']]["dppm"]), [10, 50, 90]).round(2).tolist() if R['mz_split'].any() else "-")
pd.set_option("display.width", 220)
print("\nexamples (RT-separated, 0.8-2 FWHM):")
ex = S[(S["drt_fwhm"] >= 0.8) & (S["drt_fwhm"] < 2)].head(8)
print(ex[["dppm", "drt_s", "drt_fwhm", "h_ratio", "pers_rel_b", "prom_b", "n_scans_a", "n_scans_b", "fwhm_a_s", "fwhm_b_s", "score_b", "flags_b"]].round(3).to_string(index=False))
