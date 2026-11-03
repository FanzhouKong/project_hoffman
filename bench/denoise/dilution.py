"""Dilution-series benchmark (Kong et al. 2025, Zenodo 14920689, dilution_data_mapped.csv): every spectrum from
200 to 0.02 pmol scored against its 500 pmol reference spectrum, raw and after each denoising mode.

usage: python bench/denoise/dilution.py OUT_PREFIX [--workers N] [--tol-ppm 10] [--tol-abs 0.003] [--tol-p 0.003]
Writes OUT_PREFIX.md (tables) and OUT_PREFIX.npz (per-spectrum scores and ion counts)."""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from specdenoise import Params, denoise                       # noqa: E402
from specdenoise.denoise import electronic_mask               # noqa: E402
from specdenoise.similarity import clean, score               # noqa: E402

DATA = "data/denoise/zenodo_14920689"
PMOL = {2: 200, 3: 100, 4: 40, 5: 10, 6: 4, 7: 2, 8: 1, 9: 0.4, 10: 0.2, 11: 0.1, 12: 0.04, 13: 0.02}
MS2_TOL = 0.02
MODES = ["raw", "published", "electronic", "free", "t0_top1", "t0_top5", "t0_margin", "t1"]
G = {}


def parse_peaks(text):
    if not isinstance(text, str) or not text.strip():
        return np.zeros(0), np.zeros(0)
    rows = [ln.split("\t") for ln in text.strip().split("\n") if ln.strip()]
    a = np.array([[float(r[0]), float(r[1])] for r in rows if len(r) >= 2])
    if len(a) == 0:
        return np.zeros(0), np.zeros(0)
    o = np.argsort(a[:, 0])
    return a[o, 0], a[o, 1]


def truth_mask(mz, it, pmz, ref_mz, ref_it):
    """raw ions (in the scored region, >= 1 % of base) that match a reference ion within MS2_TOL"""
    region = (mz < pmz - 1.6) & (it >= 0.01 * it[mz < pmz - 1.6].max() if np.any(mz < pmz - 1.6) else False)
    rm, ri = clean(ref_mz, ref_it, pmz, MS2_TOL)
    hit = np.zeros(len(mz), bool)
    for j in np.flatnonzero(region):
        hit[j] = np.any(np.abs(rm - mz[j]) <= MS2_TOL)
    return region, hit


def run_row(k):
    r = G["rows"][k]
    mz, it = parse_peaks(r["peaks"]); rmz, rit = parse_peaks(r["library_peaks"]); dmz, dit = parse_peaks(r["peaks_denoised"])
    pmz = float(r["precursor_mz"]); charge = 1 if str(r["adduct"]).endswith("+") else -1
    P = G["params"]
    out = {"k": k}
    rm, ri = clean(rmz, rit, pmz, MS2_TOL)
    if len(mz) == 0 or len(rm) == 0:
        return None
    region, hit = truth_mask(mz, it, pmz, rm, ri)
    keeps = {"raw": np.ones(len(mz), bool)}
    em = electronic_mask(it, 4)
    keeps["electronic"] = ~em
    t = {}
    t0 = time.time()
    base = Params(tol_ppm=P["tol_ppm"], tol_abs=P["tol_abs"], tol_precursor=P["tol_p"])
    r0 = denoise(mz, it, pmz, charge=charge, params=Params(tier="t0", top_k=5, **{k: getattr(base, k) for k in ("tol_ppm", "tol_abs", "tol_precursor")}))
    t["t0"] = time.time() - t0
    keeps["t0_top5"] = r0.keep
    keeps["t0_top1"] = r0.variant(top_k=1); keeps["t0_margin"] = r0.variant(top_k=10, margin=0.05); keeps["free"] = r0.variant(free=True)
    t0 = time.time()
    r1 = denoise(mz, it, pmz, charge=charge, formula=r["formula"], adduct=r["adduct"], params=base)
    t["t1"] = time.time() - t0
    keeps["t1"] = r1.keep
    out["t1_applied"] = r1.applied and r1.tier == 1
    out["n_candidates"] = len(r0.candidates)
    for mode in MODES:
        if mode == "published":
            qm, qi = clean(dmz, dit, pmz, MS2_TOL)
            out["n_true_removed_" + mode] = np.nan; out["n_noise_removed_" + mode] = np.nan
        else:
            kp = keeps[mode]
            qm, qi = clean(mz[kp], it[kp], pmz, MS2_TOL)
            out["n_true_removed_" + mode] = int(np.sum(region & hit & ~kp))
            out["n_noise_removed_" + mode] = int(np.sum(region & ~hit & ~kp))
        for meth, code in (("entropy", 0), ("weighted_entropy", 1)):
            out[f"{meth}_{mode}"] = score(qm, qi, rm, ri, code, MS2_TOL)[0] if len(qm) else 0.0
        out["secs_" + mode] = t.get("t0" if mode.startswith("t0") or mode == "free" else mode, 0.0)
    out["n_true"] = int(np.sum(region & hit)); out["n_noise"] = int(np.sum(region & ~hit))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--tol-ppm", type=float, default=10.0); ap.add_argument("--tol-abs", type=float, default=0.003)
    ap.add_argument("--tol-p", type=float, default=0.003); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    d = pd.read_csv(f"{DATA}/dilution_data_mapped.csv")
    allc = pd.read_csv(f"{DATA}/dilution_data_all.csv", usecols=["name", "smiles", "formula"]).drop_duplicates("smiles")
    d = d.merge(allc[["smiles", "formula"]], on="smiles", how="left")
    print(f"{len(d)} spectra, {d['name'].nunique()} compounds, formula missing for {d['formula'].isna().sum()}", flush=True)
    if a.limit:
        d = d.iloc[: a.limit]
    G["rows"] = d.to_dict("records"); G["params"] = dict(tol_ppm=a.tol_ppm, tol_abs=a.tol_abs, tol_p=a.tol_p)
    t0 = time.time()
    run_row(0)
    with Pool(a.workers) as pool:
        res = [x for x in pool.imap(run_row, range(len(d)), chunksize=8) if x is not None]
    print(f"scored {len(res)} spectra in {time.time() - t0:.0f} s", flush=True)
    R = pd.DataFrame(res)
    R["conc"] = d["conc"].values[R["k"]]; R["name"] = d["name"].values[R["k"]]; R["ce"] = d["ce"].values[R["k"]]
    R["paper_raw"] = d["entropy_similarity"].values[R["k"]]; R["paper_denoised"] = d["denoised_similarity"].values[R["k"]]
    R.to_csv(a.out + ".csv", index=False)
    out = [f"# Dilution series: {len(R)} spectra, {R['name'].nunique()} compounds; fragment tol {a.tol_ppm} ppm / {a.tol_abs * 1000:.0f} mDa, "
           f"precursor tol {a.tol_p * 1000:.0f} mDa; T1 applied to {R['t1_applied'].mean() * 100:.1f} % of spectra\n"]
    # which of our scores reproduces the paper's column?
    for meth in ("entropy", "weighted_entropy"):
        out.append(f"our {meth} vs paper's entropy_similarity (raw): r = {np.corrcoef(R[meth + '_raw'], R['paper_raw'])[0, 1]:.4f}, "
                   f"mean |diff| = {np.mean(np.abs(R[meth + '_raw'] - R['paper_raw'])):.4f}; published-denoised column vs paper's: "
                   f"mean |diff| = {np.mean(np.abs(R[meth + '_published'] - R['paper_denoised'])):.4f}")
    meth = "weighted_entropy" if np.mean(np.abs(R["weighted_entropy_raw"] - R["paper_raw"])) < np.mean(np.abs(R["entropy_raw"] - R["paper_raw"])) else "entropy"
    out.append(f"\nscores below: {meth} similarity to the 500 pmol reference (0.02 Da)\n")
    for title, fn in (("Mean similarity by amount", lambda s: s.mean()), ("Annotation rate (similarity > 0.75) by amount", lambda s: (s > 0.75).mean())):
        out.append(f"## {title}\n")
        out.append("| pmol | n | " + " | ".join(MODES) + " |"); out.append("|---|---|" + "---|" * len(MODES))
        for c, g in R.groupby("conc"):
            out.append(f"| {PMOL[c]} | {len(g)} | " + " | ".join(f"{fn(g[f'{meth}_{m}']):.3f}" for m in MODES) + " |")
        out.append(f"| all | {len(R)} | " + " | ".join(f"{fn(R[f'{meth}_{m}']):.3f}" for m in MODES) + " |")
        out.append("")
    out.append("## Lowest annotated amount per compound (any CE, similarity > 0.75): compounds annotated at each amount\n")
    out.append("| pmol | " + " | ".join(MODES) + " |"); out.append("|---|" + "---|" * len(MODES))
    best = {m: R[R[f"{meth}_{m}"] > 0.75].groupby("name")["conc"].max() for m in MODES}
    for c in sorted(PMOL):
        out.append(f"| {PMOL[c]} | " + " | ".join(f"{int((best[m] >= c).sum())}" for m in MODES) + " |")
    out.append("")
    out.append("## Ion-level: raw ions >= 1 % of base below the precursor, 'true' = within 0.02 Da of a reference ion\n")
    out.append(f"true ions {int(R['n_true'].sum())}, non-matching ions {int(R['n_noise'].sum())}\n")
    out.append("| mode | true ions removed | non-matching ions removed | mean s / spectrum |"); out.append("|---|---|---|---|")
    for m in MODES:
        if m == "published":
            continue
        out.append(f"| {m} | {100 * R['n_true_removed_' + m].sum() / R['n_true'].sum():.2f} % | "
                   f"{100 * R['n_noise_removed_' + m].sum() / R['n_noise'].sum():.2f} % | {R['secs_' + m].mean() * 1000:.1f} ms |")
    out.append("\n## Ion-level by amount: true ions removed % / non-matching ions removed %\n")
    out.append("| pmol | " + " | ".join(m for m in MODES if m != "published") + " |"); out.append("|---|" + "---|" * (len(MODES) - 1))
    for c, g in R.groupby("conc"):
        out.append(f"| {PMOL[c]} | " + " | ".join(f"{100 * g['n_true_removed_' + m].sum() / max(g['n_true'].sum(), 1):.1f} / "
                                                 f"{100 * g['n_noise_removed_' + m].sum() / max(g['n_noise'].sum(), 1):.1f}" for m in MODES if m != "published") + " |")
    open(a.out + ".md", "w").write("\n".join(out) + "\n"); print("\n".join(out))


if __name__ == "__main__":
    main()
