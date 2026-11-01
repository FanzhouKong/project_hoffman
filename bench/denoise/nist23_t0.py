"""NIST23 tier-0 pass: precursor m/z only; candidate precursor formulas inferred from the accurate mass and ranked
by explained intensity; every ion's best explaining rank.

usage: python bench/denoise/nist23_t0.py IN.npz OUT.npz [--workers N] [--limit N] [--stride N] [--tol-p 0.002]
        [--tol-abs 0.003] [--tol-ppm 10] [--elements C,H,N,O,P,S,F,Cl,Br,I]
Per ion: rank (0 = no candidate explains it), weak (some plausible loss formula exists regardless of candidates),
flag (1 precursor, 2 above). Per spectrum: ncand, truth_rank (strict rank of the record's true formula among the
candidates, 0 = not among them), seconds."""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from specdenoise import formula as fm                                  # noqa: E402
from specdenoise.denoise import bounds_for                            # noqa: E402
from specdenoise.losses import candidate_precursors, explain_t0, loss_table       # noqa: E402

G = {}


def n_decimals(mz):
    if len(mz) < 5:
        return 4
    for d, slack in ((2, 0.02), (3, 0.1)):
        v = mz * 10.0 ** d
        if np.all(np.abs(v - np.round(v)) < slack):
            return d
    return 4


def run_chunk(args):
    idx, P = args
    off = G["off"]
    out_rank, out_weak, out_flag = [], [], []
    ncand = np.zeros(len(idx), np.int32); truth = np.zeros(len(idx), np.int32); secs = np.zeros(len(idx), np.float32)
    cands = np.zeros((P["max_candidates"], fm.NE), np.int64)
    work = np.zeros((P["max_losses"], fm.NE), np.int64)
    elements = P["elements"]
    for k, i in enumerate(idx):
        t0 = time.time()
        a, b = int(off[i]), int(off[i + 1])
        mz = G["mz"][a:b].astype(np.float64); it = G["it"][a:b].astype(np.float64)
        pmz = float(G["pmz"][i])
        z = 1 if G["mode"] == "pos" else -1
        rank = np.zeros(b - a, np.int64); nexp = np.zeros(b - a, np.int64); flag = np.zeros(b - a, np.int64)
        if b > a:
            dec = n_decimals(mz)
            slack = 0.5 * 10.0 ** -dec if dec <= 3 else 0.0
            bounds = bounds_for(elements, pmz)
            nc = candidate_precursors(pmz, z, P["tol_p"], bounds, True, cands)
            if nc < 0:
                nc = P["max_candidates"]
            ncand[k] = nc
            if nc > P["max_cand"]:
                rank[:] = -1                      # too many candidates: spectrum left untouched
                out_rank.append(rank.astype(np.int16)); out_weak.append(nexp > 0); out_flag.append(flag.astype(np.int8))
                secs[k] = time.time() - t0
                continue
            score = np.zeros(max(nc, 1)); best = np.zeros(b - a)
            explain_t0(mz, it, pmz, z, cands, nc, P["tol_abs"] + slack, P["tol_ppm"], P["tol_p"],
                       loss_table(bounds_for(elements, 1e9)), work, rank, nexp, flag, score, best, 0.1)
            ncand[k] = nc
            f = fm.parse(G["formula"][i])
            pa = fm.precursor_formula(f, G["ptype"][i]) if f is not None else None
            if pa is not None and nc > 0:
                hit = np.flatnonzero((cands[:nc] == pa[0]).all(axis=1))
                if len(hit):
                    truth[k] = 1 + int(np.sum(score[:nc] > score[hit[0]]))
        out_rank.append(rank.astype(np.int16)); out_weak.append(nexp > 0); out_flag.append(flag.astype(np.int8))
        secs[k] = time.time() - t0
    return idx, out_rank, out_weak, out_flag, ncand, truth, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--tol-p", type=float, default=0.002); ap.add_argument("--tol-abs", type=float, default=0.003)
    ap.add_argument("--tol-ppm", type=float, default=10.0)
    ap.add_argument("--elements", default="C,H,N,O,P,S,F,Cl,Br,I")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--max-cand", type=int, default=20000, help="spectra with more candidates are left untouched")
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.inp, allow_pickle=True)
    G["off"] = z["peak_offsets"]; G["mz"] = z["peak_mz"]; G["it"] = z["peak_intensity"]
    G["formula"] = z["formula"]; G["ptype"] = z["precursor_type"]; G["pmz"] = z["precursor_mz"]
    G["mode"] = "neg" if "neg" in os.path.basename(a.inp) else "pos"
    n_all = len(G["off"]) - 1
    idx = np.arange(0, n_all, a.stride)
    if a.limit:
        idx = idx[: a.limit]
    P = dict(tol_p=a.tol_p, tol_abs=a.tol_abs, tol_ppm=a.tol_ppm, elements=tuple(a.elements.split(",")),
             max_candidates=200000, max_losses=50000, max_cand=a.max_cand)
    print(f"{len(idx)} spectra of {n_all}, mode {G['mode']}, elements {P['elements']}", flush=True)
    run_chunk((idx[:1], P))
    chunks = [(idx[i:i + a.chunk], P) for i in range(0, len(idx), a.chunk)]
    rank = {}; weak = {}; flag = {}
    ncand = np.zeros(len(idx), np.int32); truth = np.zeros(len(idx), np.int32); secs = np.zeros(len(idx), np.float32)
    pos = {int(i): k for k, i in enumerate(idx)}
    done = 0
    with Pool(a.workers) as pool:
        for ids, r_, w_, f_, nc_, t_, s_ in pool.imap_unordered(run_chunk, chunks):
            for j, i in enumerate(ids):
                rank[int(i)] = r_[j]; weak[int(i)] = w_[j]; flag[int(i)] = f_[j]
                ncand[pos[int(i)]] = nc_[j]; truth[pos[int(i)]] = t_[j]; secs[pos[int(i)]] = s_[j]
            done += len(ids)
            if done % (20 * a.chunk) < a.chunk:
                print(f"  {done}/{len(idx)} spectra, {time.time() - t0:.0f} s", flush=True)
    order = [int(i) for i in idx]
    np.savez(a.out, idx=idx, rank=np.concatenate([rank[i] for i in order]), weak=np.concatenate([weak[i] for i in order]),
             flag=np.concatenate([flag[i] for i in order]), ncand=ncand, truth_rank=truth, secs=secs,
             tol_p=a.tol_p, tol_abs=a.tol_abs, tol_ppm=a.tol_ppm, elements=a.elements)
    print(f"untouched (candidates > {a.max_cand}): {np.sum(ncand > a.max_cand)}")
    print(f"done {len(idx)} spectra in {time.time() - t0:.0f} s; mean {secs.mean() * 1000:.1f} ms/spectrum, "
          f"p90 {np.quantile(secs, .9) * 1000:.0f} ms, max {secs.max():.1f} s; median candidates {np.median(ncand):.0f}, "
          f"truth in candidates {100 * np.mean(truth > 0):.1f} %, truth rank 1: {100 * np.mean(truth == 1):.1f} %, "
          f"<= 5: {100 * np.mean((truth > 0) & (truth <= 5)):.1f} %", flush=True)


if __name__ == "__main__":
    main()
