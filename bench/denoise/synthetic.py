"""Synthetic-noise benchmark on the paper's files (synthetic_noise_sn_<intensity>_<count>.csv): the contaminated
spectrum (msms_cont) is denoised and compared with the clean spectrum (peaks) and the 500 pmol reference
(library_peaks); the published result (msms_cont_denoised) is scored alongside.

usage: python bench/denoise/synthetic.py OUT_PREFIX [--files sn_0.1_2,...] [--workers N] [--limit N]"""
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
from bench.denoise.dilution import parse_peaks, MS2_TOL       # noqa: E402

DATA = "data/denoise/zenodo_14920689/synthetic"
MODES = ["contaminated", "published", "electronic", "free", "t0_top1", "t0_top5", "t0_margin", "t1"]
G = {}


def run_row(k):
    r = G["rows"][k]
    mz, it = parse_peaks(r["msms_cont"]); cm, ci = parse_peaks(r["peaks"]); rmz, rit = parse_peaks(r["library_peaks"])
    dmz, dit = parse_peaks(r["msms_cont_denoised"])
    pmz = float(r["precursor_mz"]); charge = 1 if str(r["adduct"]).endswith("+") else -1
    P = G["params"]
    rm, ri = clean(rmz, rit, pmz, MS2_TOL)
    if len(mz) == 0 or len(rm) == 0 or len(cm) == 0:
        return None
    # truth: contaminated ions that are copies of clean-spectrum ions (identical m/z)
    true = np.zeros(len(mz), bool)
    j = np.searchsorted(cm, mz)
    for i in range(len(mz)):
        for jj in (j[i] - 1, j[i]):
            if 0 <= jj < len(cm) and abs(cm[jj] - mz[i]) <= 1e-4:
                true[i] = True
    region = mz < pmz - 1.6
    keeps = {"contaminated": np.ones(len(mz), bool), "electronic": ~electronic_mask(it, 4)}
    out = {"k": k}
    t = {}
    t0 = time.time()
    base = Params(tol_ppm=P["tol_ppm"], tol_abs=P["tol_abs"], tol_precursor=P["tol_p"], electronic=P["electronic"])
    r0 = denoise(mz, it, pmz, charge=charge, params=Params(tier="t0", top_k=5, electronic=P["electronic"], **{k: getattr(base, k) for k in ("tol_ppm", "tol_abs", "tol_precursor")}))
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
            # published ions matched back to the contaminated spectrum by identical m/z
            kp = np.zeros(len(mz), bool)
            jj = np.searchsorted(dmz, mz)
            for i in range(len(mz)):
                for q in (jj[i] - 1, jj[i]):
                    if 0 <= q < len(dmz) and abs(dmz[q] - mz[i]) <= 1e-4:
                        kp[i] = True
        else:
            kp = keeps[mode]
            qm, qi = clean(mz[kp], it[kp], pmz, MS2_TOL)
        out["true_removed_" + mode] = int(np.sum(region & true & ~kp)); out["noise_removed_" + mode] = int(np.sum(region & ~true & ~kp))
        out["sim_" + mode] = score(qm, qi, rm, ri, 1, MS2_TOL)[0] if len(qm) else 0.0
        out["secs_" + mode] = t.get("t0" if mode.startswith("t0") or mode == "free" else mode, 0.0)
    out["n_true"] = int(np.sum(region & true)); out["n_noise"] = int(np.sum(region & ~true))
    # similarity of the clean spectrum itself (ceiling)
    qm, qi = clean(cm, ci, pmz, MS2_TOL)
    out["sim_clean"] = score(qm, qi, rm, ri, 1, MS2_TOL)[0] if len(qm) else 0.0
    out["paper_cont"] = r["entropy_cont_raw"]; out["paper_denoised"] = r["entropy_cont_denoised"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--files", default="sn_0.1_2,sn_0.1_10,sn_0.1_100,sn_0.2_2,sn_0.2_10,sn_0.2_100,sn_0.5_2,sn_0.5_10,sn_0.5_100")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tol-ppm", type=float, default=10.0); ap.add_argument("--tol-abs", type=float, default=0.003)
    ap.add_argument("--tol-p", type=float, default=0.003)
    ap.add_argument("--no-electronic", action="store_true", help="skip the identical-intensity step in our modes")
    a = ap.parse_args()
    G["params"] = dict(tol_ppm=a.tol_ppm, tol_abs=a.tol_abs, tol_p=a.tol_p, electronic=not a.no_electronic)
    cols = ["name", "conc", "ce", "formula", "adduct", "precursor_mz", "peaks", "library_peaks", "msms_cont", "msms_cont_denoised",
            "entropy_cont_raw", "entropy_cont_denoised"]
    out = [f"# Synthetic noise (paper's files): weighted entropy similarity to the 500 pmol reference, 0.02 Da{' (our modes WITHOUT the electronic step)' if a.no_electronic else ''}\n",
           "| file | n | clean | contaminated | published | electronic | free | t0_top1 | t0_top5 | t0_margin | t1 | paper cont. | paper denoised |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ion = ["\n## Ion level (below the precursor): noise ions removed % / true ions removed %\n",
           "| file | noise ions | true ions | published | electronic | free | t0_top1 | t0_top5 | t0_margin | t1 |", "|---|---|---|---|---|---|---|---|---|---|"]
    allR = []
    for f in a.files.split(","):
        t0 = time.time()
        d = pd.read_csv(f"{DATA}/synthetic_noise_{f}.csv", usecols=cols)
        if a.limit:
            d = d.iloc[: a.limit]
        G["rows"] = d.to_dict("records")
        run_row(0)
        with Pool(a.workers) as pool:
            res = [x for x in pool.imap(run_row, range(len(d)), chunksize=4) if x is not None]
        R = pd.DataFrame(res); R["file"] = f; allR.append(R)
        print(f"{f}: {len(R)} spectra in {time.time() - t0:.0f} s; t0_top5 {R['secs_t0_top5'].mean() * 1000:.0f} ms/spectrum", flush=True)
        out.append(f"| {f} | {len(R)} | {R['sim_clean'].mean():.3f} | " + " | ".join(f"{R['sim_' + m].mean():.3f}" for m in MODES)
                   + f" | {R['paper_cont'].mean():.3f} | {R['paper_denoised'].mean():.3f} |")
        ion.append(f"| {f} | {int(R['n_noise'].sum())} | {int(R['n_true'].sum())} | " + " | ".join(
            f"{100 * R['noise_removed_' + m].sum() / max(R['n_noise'].sum(), 1):.1f} / {100 * R['true_removed_' + m].sum() / max(R['n_true'].sum(), 1):.2f}"
            for m in MODES if m != "contaminated") + " |")
    R = pd.concat(allR); R.to_csv(a.out + ".csv", index=False)
    out.append(f"\nannotation rate (similarity > 0.75) over all files: " + ", ".join(f"{m} {100 * (R['sim_' + m] > 0.75).mean():.1f} %" for m in MODES)
               + f", clean {100 * (R['sim_clean'] > 0.75).mean():.1f} %")
    open(a.out + ".md", "w").write("\n".join(out + ion) + "\n"); print("\n".join(out + ion))


if __name__ == "__main__":
    main()
