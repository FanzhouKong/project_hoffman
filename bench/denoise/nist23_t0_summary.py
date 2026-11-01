"""Summarise a tier-0 NIST23 pass against the tier-1 pass on the same spectra.
usage: python bench/denoise/nist23_t0_summary.py IN.npz T0.npz T1.npz OUT.md"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from specdenoise import formula as fm       # noqa: E402


def main():
    inp, t0p, t1p, outp = sys.argv[1:5]
    z = np.load(inp, allow_pickle=True); t0 = np.load(t0p); t1 = np.load(t1p)
    off = z["peak_offsets"]; idx = t0["idx"]
    formulas = z["formula"]; inst = z["instrument_type"]
    rank = t0["rank"]; weak = t0["weak"]; flag0 = t0["flag"]
    ncand = t0["ncand"]; truth = t0["truth_rank"]; secs = t0["secs"]
    # tier-1 arrays for the same spectra, at 10 ppm / 3 mDa + grid
    sel = np.concatenate([np.arange(off[i], off[i + 1]) for i in idx])
    spec = np.repeat(np.arange(len(idx)), [off[i + 1] - off[i] for i in idx])
    mz = z["peak_mz"][sel].astype(np.float64); it = z["peak_intensity"][sel].astype(np.float64)
    dec = t1["decimals"][idx]; slack = np.where(dec <= 3, 0.5 * 10.0 ** -dec.astype(float), 0.0)[spec]
    tol = np.maximum(0.003, 10e-6 * mz) + slack
    err1 = np.minimum(t1["err"][sel], t1["err_x"][sel]); flag1 = t1["flag"][sel]
    keep1 = (flag1 == 1) | ((flag1 == 0) & (err1 <= tol))
    touched = (rank >= 0)
    halo = np.array([any(e in str(formulas[i]) for e in ("F", "Cl", "Br", "I")) for i in idx])
    out = [f"# NIST23 tier-0 pass: {os.path.basename(t0p)}  (elements {t0['elements']}, precursor tol {float(t0['tol_p']) * 1000:.0f} mDa)\n",
           f"spectra {len(idx)}, ions {len(sel)}; untouched (too many candidates) {np.sum(~touched[np.r_[0, np.cumsum([off[i + 1] - off[i] for i in idx])[:-1]]])} spectra; "
           f"time per spectrum mean {secs.mean() * 1000:.0f} ms, median {np.median(secs) * 1000:.0f} ms, p90 {np.quantile(secs, .9) * 1000:.0f} ms, max {secs.max():.1f} s\n",
           f"candidates per spectrum: median {np.median(ncand):.0f}, p90 {np.quantile(ncand, .9):.0f}, max {ncand.max()}\n",
           "## Formula inference (true formula among the candidates, and its strict rank by explained intensity)\n",
           "| subset | spectra | truth among candidates | rank 1 | rank <= 3 | rank <= 5 | rank <= 10 |", "|---|---|---|---|---|---|---|"]
    for lab, m in (("all", np.ones(len(idx), bool)), ("no halogen", ~halo), ("halogenated", halo)):
        tr = truth[m]
        out.append(f"| {lab} | {m.sum()} | {100 * np.mean(tr > 0):.1f} % | {100 * np.mean(tr == 1):.1f} % | {100 * np.mean((tr > 0) & (tr <= 3)):.1f} % | "
                   f"{100 * np.mean((tr > 0) & (tr <= 5)):.1f} % | {100 * np.mean((tr > 0) & (tr <= 10)):.1f} % |")
    out += ["\n## Retention (touched spectra, 10 ppm / 3 mDa + grid), tier 0 at top-k vs tier 1 with the formula\n",
            "| rule | ions kept | intensity kept | kept by T1 but removed here | removed by T1 but kept here |", "|---|---|---|---|---|"]
    m = touched
    tot_i = it[m].sum(); n_i = m.sum()
    base_pre = (flag0 == 1)
    for lab, keep0 in (("T1 (formula given)", keep1),
                       ("T0 top-1", base_pre | ((rank > 0) & (rank <= 1))), ("T0 top-3", base_pre | ((rank > 0) & (rank <= 3))),
                       ("T0 top-5", base_pre | ((rank > 0) & (rank <= 5))), ("T0 top-10", base_pre | ((rank > 0) & (rank <= 10))),
                       ("T0 any candidate", base_pre | (rank > 0)), ("formula-free: any plausible loss", base_pre | weak)):
        k = keep0 & m
        out.append(f"| {lab} | {100 * k.sum() / n_i:.3f} % | {100 * it[k].sum() / tot_i:.3f} % | "
                   f"{100 * np.sum(keep1 & m & ~keep0) / n_i:.3f} % | {100 * np.sum(~keep1 & m & keep0) / n_i:.3f} % |")
    out.append("\n(T0 rows have no isotopologue rescue yet; T1 here is without it too, for a like-for-like comparison.)\n")
    open(outp, "w").write("\n".join(out) + "\n"); print("\n".join(out))


if __name__ == "__main__":
    main()
