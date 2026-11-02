"""Why does T1 remove ions of a given adduct? Classify removed fragment ions by best loss error (loose 25 mDa pass) and
list the most frequent unexplained loss masses.

usage: python bench/denoise/nist23_adduct_diag.py IN.npz '[M+K]+' '[M-2H+3Na]+' ... [--n 400]"""
import argparse, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from specdenoise import formula as fm
from specdenoise.denoise import denoise, Params
from specdenoise.losses import explain_t1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("adducts", nargs="+"); ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    z = np.load(a.inp, allow_pickle=True)
    off = z["peak_offsets"]; MZ = z["peak_mz"]; IT = z["peak_intensity"]
    formula = z["formula"]; ptype = np.asarray(z["precursor_type"]).astype(str); pmzs = z["precursor_mz"]
    keys = [k for k in z.files if "instrument" in k.lower()]
    inst = np.asarray(z[keys[0]]).astype(str) if keys else None
    print("npz keys:", z.files)
    rng = np.random.default_rng(0)
    P = Params()
    for ad in a.adducts:
        w = np.flatnonzero(ptype == ad); sel = rng.choice(w, min(a.n, len(w)), replace=False)
        cat = collections.Counter(); inten = collections.Counter(); lossmass = collections.Counter(); ninst = collections.Counter()
        tot_i = 0.0; tot_n = 0; examples = []
        for i in sel:
            s, e = int(off[i]), int(off[i + 1]); mz = MZ[s:e].astype(float); it = IT[s:e].astype(float)
            o = np.argsort(mz); mz, it = mz[o], it[o]
            f = fm.parse(formula[i]); pa = fm.precursor_formula(f, ad) if f is not None else None
            if pa is None or len(mz) == 0: continue
            p, zc = pa; pmz = float(pmzs[i])
            if abs(fm.ion_mz(p, zc) - pmz) > 0.02: continue
            r = denoise(mz, it, pmz, formula=formula[i], adduct=ad, params=P)
            if r.tier != 1: continue
            if inst is not None: ninst[inst[i]] += 1
            tot_i += it.sum(); tot_n += len(mz)
            n = len(mz); work = np.zeros((P.max_losses, fm.NE), np.int64)
            err = np.zeros(n); err_x = np.zeros(n); nexp = np.zeros(n, np.int64); flag = np.zeros(n, np.int64)
            explain_t1(mz, pmz, zc, p, 0.025, 0.0, work, err, nexp, flag, err_x)   # loose pass
            for k in np.flatnonzero(~r.keep):
                if flag[k] == 2: c = "above precursor"
                elif np.isfinite(err[k]) and err[k] <= 0.010: c = "mass error 3-10 mDa"
                elif np.isfinite(err[k]): c = "mass error 10-25 mDa"
                else:
                    c = "no permitted loss within 25 mDa"
                    lossmass[round(pmz - mz[k], 2)] += 1
                    if it[k] >= 0.2 * it.max() and len(examples) < 25:
                        examples.append((str(z["name"][i])[:40] if "name" in z.files else "", formula[i], ad, round(pmz, 4), round(mz[k], 4), round(pmz - mz[k], 4), int(it[k])))
                cat[c] += 1; inten[c] += it[k]
        print(f"\n## {ad}: {len(sel)} spectra, {tot_n} ions; instrument types {ninst.most_common(4)}")
        for c in cat: print(f"  {c}: {cat[c]} ions ({100 * cat[c] / tot_n:.2f} % of ions, {100 * inten[c] / tot_i:.2f} % of intensity)")
        print("  most frequent unexplained loss masses (Da: count):", lossmass.most_common(15))
        print("  examples (name, formula, adduct, pmz, mz, loss, intensity):"); [print("   ", x) for x in examples]


if __name__ == "__main__":
    main()
