"""Re-run T1 and T0 on named NIST spectra (adduct + precursor m/z), twice each, and print per-ion decisions.
usage: python bench/denoise/nist23_spectrum_diag.py IN.npz ADDUCT pmz1 pmz2 ..."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from specdenoise import formula as fm
from specdenoise.denoise import denoise, Params
z = np.load(sys.argv[1], allow_pickle=True); ad = sys.argv[2]; targets = [float(v) for v in sys.argv[3:]]
ptype = np.asarray(z["precursor_type"]).astype(str); pmzs = np.asarray(z["precursor_mz"], float)
off = z["peak_offsets"]; MZ = z["peak_mz"]; IT = z["peak_intensity"]; formula = z["formula"]; name = z["name"]
for t in targets:
    for i in np.flatnonzero((ptype == ad) & (np.abs(pmzs - t) < 2e-4)):
        s, e = int(off[i]), int(off[i + 1]); mz = MZ[s:e].astype(float); it = IT[s:e].astype(float)
        print(f"\n### {name[i]} {formula[i]} {ad} pmz {pmzs[i]:.4f} ({len(mz)} ions)")
        for rep in range(2):
            r1 = denoise(mz, it, float(pmzs[i]), formula=formula[i], adduct=ad)
            r0 = denoise(mz, it, float(pmzs[i]), charge=1)
            print(f"  rep {rep}: T1 kept {int(r1.keep.sum())}/{len(mz)} inten {100 * it[r1.keep].sum() / it.sum():.1f} %; "
                  f"T0 kept {int(r0.keep.sum())} inten {100 * it[r0.keep].sum() / it.sum():.1f} %, {len(r0.candidates)} candidates, top {fm.format(r0.candidates[0]) if len(r0.candidates) else '-'}, overflow {r0.overflow}")
        o = np.argsort(mz)
        for k in o:
            if it[k] >= 0.05 * it.max() or not r1.keep[k]:
                print(f"    mz {mz[k]:10.4f} int {it[k]:5.0f} loss {pmzs[i] - mz[k]:9.4f}  T1 {'keep' if r1.keep[k] else 'DROP'} r{r1.reason[k]} err {r1.err[k] * 1e3 if r1.err is not None and np.isfinite(r1.err[k]) else float('nan'):6.1f} mDa  T0 {'keep' if r0.keep[k] else 'DROP'} r{r0.reason[k]}")
