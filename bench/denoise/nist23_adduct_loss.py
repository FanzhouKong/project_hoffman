"""How often does a fragment of a cationised precursor ([M+Na]+, [M+K]+) lose the cation, and what does T0 do with it?

usage: python bench/denoise/nist23_adduct_loss.py IN.npz OUT.md [--per-adduct 2000] [--workers N] [--seed 0]

For every sampled spectrum: T1 with the true ion formula (keep_t1); T1 with the cation swapped for H in the bounding
formula but the loss still measured from the observed precursor m/z (keep_h: only cation-free losses allowed, i.e.
the fragment must retain Na/K); T0 from the precursor m/z alone (keep_t0). Among fragment ions T1 keeps by a loss
(reason 0, not the precursor, not isotope-rescued): 'lost the cation' = ~keep_h; 'T0 removes' = ~keep_t0."""
import argparse, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from specdenoise import formula as fm
from specdenoise.denoise import denoise, Params
from specdenoise.losses import explain_t1

G = {}
CATION = {"[M+Na]+": "Na", "[M+K]+": "K", "[M+NH4]+": "NH4"}


def one(i):
    off = G["off"]; a, b = int(off[i]), int(off[i + 1])
    mz = G["mz"][a:b].astype(np.float64); it = G["it"][a:b].astype(np.float64)
    o = np.argsort(mz); mz = mz[o]; it = it[o]
    f = fm.parse(G["formula"][i]); ptype = str(G["ptype"][i]); pmz = float(G["pmz"][i])
    pa = fm.precursor_formula(f, ptype) if f is not None else None
    if pa is None or len(mz) == 0:
        return None
    p, z = pa
    if abs(fm.ion_mz(p, z) - pmz) > 0.02:
        return None
    P = Params()
    r1 = denoise(mz, it, pmz, formula=G["formula"][i], adduct=ptype, params=P)
    if r1.tier != 1:
        return None
    r0 = denoise(mz, it, pmz, charge=z, params=P)
    # cation swapped for H in the bounding formula: losses may not contain the cation
    ph = p.copy()
    piece = fm.parse(fm._PIECES[CATION[ptype]])
    ph -= piece; ph[fm.IDX["H"]] += 1
    n = len(mz)
    work = np.zeros((P.max_losses, fm.NE), np.int64)
    err = np.zeros(n); err_x = np.zeros(n); nexp = np.zeros(n, np.int64); flag = np.zeros(n, np.int64)
    explain_t1(mz, pmz, z, ph, P.tol_abs, P.tol_ppm, work, err, nexp, flag, err_x)
    keep_h = np.isfinite(err) | np.isfinite(err_x)
    sel = r1.keep & (r1.reason == 0) & (flag == 0)        # fragment ions T1 keeps through a permitted loss
    lost = sel & ~keep_h
    t0rm = sel & ~r0.keep
    base = it.max()
    return dict(ptype=ptype, n=int(sel.sum()), inten=float(it[sel].sum()), lost=int(lost.sum()),
                lost_inten=float(it[lost].sum()), t0rm=int(t0rm.sum()), t0rm_inten=float(it[t0rm].sum()),
                lost_and_t0rm=int((lost & t0rm).sum()), lost_big=int((lost & (it >= 0.1 * base)).sum()),
                t0rm_big=int((t0rm & (it >= 0.1 * base)).sum()), any_lost=bool(lost.any()), any_t0rm=bool(t0rm.any()),
                lost_base=bool(lost[np.argmax(it)]), ncand=int(len(r0.candidates)) if r0.candidates is not None else 0,
                overflow=bool(r0.overflow), pmz=pmz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--per-adduct", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.inp, allow_pickle=True)
    G["off"] = z["peak_offsets"]; G["mz"] = z["peak_mz"]; G["it"] = z["peak_intensity"]
    G["formula"] = z["formula"]; G["ptype"] = z["precursor_type"]; G["pmz"] = z["precursor_mz"]
    ptype = np.asarray(G["ptype"]).astype(str)
    u, c = np.unique(ptype, return_counts=True)
    top = sorted(zip(c, u), reverse=True)[:15]
    print(f"loaded {len(ptype)} spectra in {time.time() - t0:.0f} s; precursor types: {top}", flush=True)
    rng = np.random.default_rng(a.seed)
    idx = []
    for t in CATION:
        w = np.flatnonzero(ptype == t)
        idx += list(rng.choice(w, min(a.per_adduct, len(w)), replace=False))
    one(idx[0])                                   # compile before forking
    with Pool(a.workers) as pool:
        res = [r for r in pool.imap_unordered(one, idx, chunksize=20) if r is not None]
    print(f"{len(res)} spectra done in {time.time() - t0:.0f} s", flush=True)
    lines = [f"# NIST23 {os.path.basename(a.inp)}: fragments of cationised precursors that lose the cation, and T0's verdict",
             "", f"sampled up to {a.per_adduct} spectra per adduct (seed {a.seed}); counts over fragment ions T1 keeps "
             "through a permitted loss; 'lost cation' = not explainable by any cation-free loss; big = >= 10 % of base peak", "",
             "| adduct | spectra | in NIST | T1-kept fragment ions | lost cation (ions) | lost cation (intensity) | lost, big | "
             "spectra with any | base peak lost cation | T0 removes (ions) | T0 removes (intensity) | T0 removes, big | "
             "of which lost cation | spectra T0 touches | median candidates | overflow |", "|" + "---|" * 16]
    for t in CATION:
        R = [r for r in res if r["ptype"] == t]
        if not R:
            continue
        n = sum(r["n"] for r in R); inten = sum(r["inten"] for r in R)
        lost = sum(r["lost"] for r in R); t0rm = sum(r["t0rm"] for r in R)
        lines.append(f"| {t} | {len(R)} | {int((ptype == t).sum())} | {n} | {lost} ({100 * lost / max(n, 1):.2f} %) | "
                     f"{100 * sum(r['lost_inten'] for r in R) / max(inten, 1e-9):.2f} % | {sum(r['lost_big'] for r in R)} | "
                     f"{100 * np.mean([r['any_lost'] for r in R]):.1f} % | {100 * np.mean([r['lost_base'] for r in R]):.1f} % | "
                     f"{t0rm} ({100 * t0rm / max(n, 1):.2f} %) | {100 * sum(r['t0rm_inten'] for r in R) / max(inten, 1e-9):.2f} % | "
                     f"{sum(r['t0rm_big'] for r in R)} | {sum(r['lost_and_t0rm'] for r in R)} | "
                     f"{100 * np.mean([r['any_t0rm'] for r in R]):.1f} % | {int(np.median([r['ncand'] for r in R]))} | "
                     f"{sum(r['overflow'] for r in R)} |")
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
