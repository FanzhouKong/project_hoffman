"""Our own synthetic noise on the dilution-series spectra (dilution_data_mapped.csv): continuous intensities,
uniform-m/z ("electronic-like") and formula-mass ("chemical-like") noise ions at several counts and intensities.

usage: python bench/denoise/synthetic_own.py OUT_PREFIX [--workers N] [--limit N]"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from specdenoise import Params, denoise                       # noqa: E402
from specdenoise.denoise import bounds_for, electronic_mask   # noqa: E402
from specdenoise.losses import loss_table                     # noqa: E402
from specdenoise.similarity import clean, score               # noqa: E402
from specdenoise.synth import add_noise                       # noqa: E402
from bench.denoise.dilution import parse_peaks, MS2_TOL, DATA  # noqa: E402

CONDITIONS = [("uniform", 0.5, 0.05), ("uniform", 2.0, 0.05), ("uniform", 2.0, 0.3),
              ("formula", 0.5, 0.05), ("formula", 2.0, 0.05), ("formula", 2.0, 0.3)]   # (kind, noise ions per real ion, median intensity / base)
MODES = ["contaminated", "electronic", "free", "t0_top5", "t1"]
G = {}


def run_row(k):
    r = G["rows"][k]
    cm, ci = parse_peaks(r["peaks"]); rmz, rit = parse_peaks(r["library_peaks"])
    pmz = float(r["precursor_mz"]); charge = 1 if str(r["adduct"]).endswith("+") else -1
    rm, ri = clean(rmz, rit, pmz, MS2_TOL)
    if len(cm) < 3 or len(rm) == 0 or pmz < 120:
        return None
    rng = np.random.default_rng(k)
    outs = []
    base = Params(electronic=False)
    for kind, ratio, scale in CONDITIONS:
        region = cm < pmz - 1.6
        mz, it, isn = add_noise(cm, ci, pmz, ratio * region.sum(), kind, scale, rng, G["table"])
        out = {"k": k, "kind": kind, "ratio": ratio, "scale": scale}
        keeps = {"contaminated": np.ones(len(mz), bool), "electronic": ~electronic_mask(it, 4)}
        r0 = denoise(mz, it, pmz, charge=charge, params=Params(tier="t0", top_k=5, electronic=False))
        keeps["t0_top5"] = r0.keep; keeps["free"] = r0.variant(free=True)
        keeps["t1"] = denoise(mz, it, pmz, charge=charge, formula=r["formula"], adduct=r["adduct"], params=base).keep
        reg = mz < pmz - 1.6
        for mode in MODES:
            kp = keeps[mode]
            qm, qi = clean(mz[kp], it[kp], pmz, MS2_TOL)
            out["sim_" + mode] = score(qm, qi, rm, ri, 1, MS2_TOL)[0] if len(qm) else 0.0
            out["noise_removed_" + mode] = int(np.sum(reg & isn & ~kp)); out["true_removed_" + mode] = int(np.sum(reg & ~isn & ~kp))
        out["n_noise"] = int(np.sum(reg & isn)); out["n_true"] = int(np.sum(reg & ~isn))
        qm, qi = clean(cm, ci, pmz, MS2_TOL); out["sim_clean"] = score(qm, qi, rm, ri, 1, MS2_TOL)[0] if len(qm) else 0.0
        outs.append(out)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    d = pd.read_csv(f"{DATA}/dilution_data_mapped.csv")
    allc = pd.read_csv(f"{DATA}/dilution_data_all.csv", usecols=["smiles", "formula"]).drop_duplicates("smiles")
    d = d.merge(allc, on="smiles", how="left")
    if a.limit:
        d = d.iloc[: a.limit]
    G["rows"] = d.to_dict("records")
    G["table"] = loss_table(bounds_for(Params().elements, 1e9))
    t0 = time.time()
    run_row(0)
    with Pool(a.workers) as pool:
        res = [o for x in pool.imap(run_row, range(len(d)), chunksize=4) if x is not None for o in x]
    R = pd.DataFrame(res); R["conc"] = d["conc"].values[R["k"]]; R.to_csv(a.out + ".csv", index=False)
    print(f"{len(R)} spectrum x condition rows in {time.time() - t0:.0f} s", flush=True)
    out = ["# Our synthetic noise on the dilution spectra (continuous lognormal intensities; our modes without the electronic step)\n",
           "noise kinds: uniform m/z (electronic-like) and formula-mass losses (chemical-like); ratio = noise ions per real ion; scale = median noise intensity / base peak\n",
           "| kind | ratio | scale | n | clean | " + " | ".join(MODES) + " |", "|---|---|---|---|---|" + "---|" * len(MODES)]
    ion = ["\n## Ion level (below the precursor): noise ions removed % / true ions removed %\n",
           "| kind | ratio | scale | noise ions | true ions | " + " | ".join(m for m in MODES if m != "contaminated") + " |", "|---|---|---|---|---|" + "---|" * (len(MODES) - 1)]
    for (kind, ratio, scale), g in R.groupby(["kind", "ratio", "scale"]):
        out.append(f"| {kind} | {ratio} | {scale} | {len(g)} | {g['sim_clean'].mean():.3f} | " + " | ".join(f"{g['sim_' + m].mean():.3f}" for m in MODES) + " |")
        ion.append(f"| {kind} | {ratio} | {scale} | {int(g['n_noise'].sum())} | {int(g['n_true'].sum())} | " + " | ".join(
            f"{100 * g['noise_removed_' + m].sum() / max(g['n_noise'].sum(), 1):.1f} / {100 * g['true_removed_' + m].sum() / max(g['n_true'].sum(), 1):.2f}" for m in MODES if m != "contaminated") + " |")
    open(a.out + ".md", "w").write("\n".join(out + ion) + "\n"); print("\n".join(out + ion))


if __name__ == "__main__":
    main()
