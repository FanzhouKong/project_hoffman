"""Why did the precursor-peak recalibration change T1 results? Dilution series and NIST by-adduct, old vs new runs."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from specdenoise import formula as fm
from specdenoise.denoise import denoise, Params
from dilution import parse_peaks, PMOL

def offset_rule(mz, it, pmz):
    win = max(0.02, 20e-6 * pmz)
    near = np.flatnonzero(np.abs(mz - pmz) <= win)
    if len(near) and it[near].max() >= 0.02 * it.max():
        j = near[np.argmax(it[near])]
        return float(mz[j] - pmz), float(it[j] / it.max())
    return 0.0, 0.0

print("=== dilution series: spectra whose T1 similarity changed by > 0.02 ===")
A = pd.read_csv("results/denoise/dilution/mapped.csv"); B = pd.read_csv("results/denoise/dilution/mapped_v2.csv")
D = pd.read_csv("data/denoise/zenodo_14920689/dilution_data_mapped.csv")
d = B.weighted_entropy_t1.values - A.weighted_entropy_t1.values
print("n spectra", len(d), "worse > 0.02:", int((d < -0.02).sum()), "better > 0.02:", int((d > 0.02).sum()), "mean delta %.4f" % d.mean())
rows = []
for k in np.flatnonzero(np.abs(d) > 0.02):
    r = D.iloc[int(A.k[k])]; mz, it = parse_peaks(r["peaks"]); pmz = float(r["precursor_mz"])
    off, frac = offset_rule(mz, it, pmz)
    rows.append(dict(k=int(A.k[k]), pmol=PMOL.get(int(r["conc"]), r["conc"]), adduct=r["adduct"], pmz=pmz, offset_mDa=1000 * off, peak_frac=frac,
                     dsim=d[k], true_rm_old=A.n_true_removed_t1[k], true_rm_new=B.n_true_removed_t1[k], noise_rm_old=A.n_noise_removed_t1[k], noise_rm_new=B.n_noise_removed_t1[k]))
R = pd.DataFrame(rows)
print(R.sort_values("dsim").head(25).to_string())
print("offset |mDa| distribution among changed spectra:", np.percentile(np.abs(R.offset_mDa), [10, 50, 90]).round(2) if len(R) else None)
print("peak_frac distribution:", np.percentile(R.peak_frac, [10, 50, 90]).round(3) if len(R) else None)
# all spectra: how often does the rule fire and with what offset
offs = []; fr = []
for i in range(len(D)):
    mz, it = parse_peaks(D.peaks[i])
    if len(mz) == 0: continue
    o, f = offset_rule(mz, it, float(D.precursor_mz[i])); offs.append(o); fr.append(f)
offs = np.array(offs) * 1000; fr = np.array(fr)
print("dilution: rule fires in %.1f %% of spectra; |offset| mDa p50 %.2f p90 %.2f p99 %.2f; offsets > 5 mDa: %.1f %%" % (100 * (fr > 0).mean(), *np.percentile(np.abs(offs[fr > 0]), [50, 90, 99]), 100 * (np.abs(offs) > 5).mean()))

print("\n=== NIST by adduct: per-spectrum T1 change, old vs new ===")
A = pd.read_csv("results/denoise/nist23/by_adduct_small.csv"); B = pd.read_csv("results/denoise/nist23/by_adduct_v2.csv")
key = ["pol", "adduct", "pmz", "nion"]
M = A.merge(B, on=key, suffixes=("_old", "_new"))
M["d"] = M.sim_t1_new - M.sim_t1_old
print("aligned", len(M), "of", len(A))
g = M.groupby(["pol", "adduct"]).agg(n=("d", "size"), mean_d=("d", "mean"), worse=("d", lambda s: int((s < -0.02).sum())), better=("d", lambda s: int((s > 0.02).sum())))
print(g[(g.worse > 5) | (g.better > 5)].sort_values("mean_d").to_string())
for pol, path in (("neg", "/quobyte/metabolomicsgrp/fanzhou/parsed_msp/nist23/nist23_neg.npz"), ("pos", "/quobyte/metabolomicsgrp/fanzhou/parsed_msp/nist23/nist23_pos.npz")):
    z = np.load(path, allow_pickle=True)
    pt = z["precursor_type"].astype(str); off = z["peak_offsets"]; mzs = z["peak_mz"]; its = z["peak_intensity"]; pm = z["precursor_mz"]; fo = z["formula"]
    for ad in (["[M+Cl]-", "[M-H]-"] if pol == "neg" else ["[M+K]+", "[M+H]+"]):
        W = M[(M.pol == pol) & (M.adduct == ad)].sort_values("d")
        print(f"\n--- {pol} {ad}: {len(W)} spectra, mean d {W.d.mean():.4f}; worst 3 ---")
        for _, w in W.head(3).iterrows():
            idx = np.flatnonzero((pt == ad) & (np.abs(pm - w.pmz) < 1e-4))
            if len(idx) == 0: continue
            k = idx[0]; a, b = int(off[k]), int(off[k + 1]); mz = mzs[a:b].astype(float); it = its[a:b].astype(float); pmz = float(pm[k])
            if len(mz) != w.nion: continue
            r_old = denoise(mz, it, pmz, formula=str(fo[k]), adduct=ad, params=Params(precursor_peak=False))
            r_new = denoise(mz, it, pmz, formula=str(fo[k]), adduct=ad)
            print(f"{z['name'][k] if 'name' in z.files else ''} {fo[k]} pmz {pmz:.4f} ions {len(mz)} d {w.d:.3f} offset {1000 * r_new.offset:.2f} mDa; kept old {r_old.keep.sum()} new {r_new.keep.sum()}")
            ch = np.flatnonzero(r_old.keep != r_new.keep)
            for j in ch[np.argsort(-it[ch])][:6]:
                print(f"    mz {mz[j]:.4f} int {100 * it[j] / it.max():.1f} % loss {pmz - mz[j]:.4f}  old keep {r_old.keep[j]} ({r_old.reason[j]})  new keep {r_new.keep[j]} ({r_new.reason[j]})")
