"""NIST23 tier-1 pass: every ion's best neutral-loss mass error given the record's formula and adduct.

usage: python bench/denoise/nist23_t1.py IN.npz OUT.npz [--workers N] [--tol 0.025] [--limit N]
Writes per-ion err (Da, inf = no permitted loss within tol), flag (0 fragment, 1 precursor, 2 above precursor),
nexp (permitted losses within tol), elec (identical-intensity group member); per-spectrum status (0 ok, 1 bad
formula, 2 unknown adduct, 3 formula/precursor m/z disagree > 20 mDa, 4 |charge| != 1), pmz_theo and
n_candidates_overflow."""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from specdenoise import formula as fm                      # noqa: E402
from specdenoise.denoise import electronic_mask           # noqa: E402
from specdenoise.losses import explain_t1                 # noqa: E402

G = {}


def prepare(i):
    f = fm.parse(G["formula"][i])
    if f is None:
        return 1, None, 0, 0.0
    pa = fm.precursor_formula(f, G["ptype"][i])
    if pa is None:
        return 2, None, 0, 0.0
    p, z = pa
    if abs(z) != 1:
        return 4, None, z, 0.0
    pmz = fm.ion_mz(p, z)
    if abs(pmz - float(G["pmz"][i])) > 0.02:
        return 3, None, z, pmz
    return 0, p, z, pmz


def n_decimals(mz):
    """2 or 3 when every m/z of the spectrum sits on a 2- or 3-decimal grid (5 ions or more), else 4"""
    if len(mz) < 5:
        return 4
    for d, slack in ((2, 0.02), (3, 0.1)):
        v = mz * 10.0 ** d
        if np.all(np.abs(v - np.round(v)) < slack):
            return d
    return 4


def run_chunk(args):
    lo, hi, tol = args
    off = G["off"]
    n_ion = int(off[hi] - off[lo])
    err = np.full(n_ion, np.inf, np.float32)
    err_x = np.full(n_ion, np.inf, np.float32)
    flag = np.zeros(n_ion, np.int8)
    nexp = np.zeros(n_ion, np.int16)
    elec = np.zeros(n_ion, np.bool_)
    status = np.zeros(hi - lo, np.int8)
    decimals = np.zeros(hi - lo, np.int8)
    pmz_theo = np.zeros(hi - lo, np.float64)
    work = np.zeros((50000, fm.NE), np.int64)
    e64 = np.zeros(0); ex64 = np.zeros(0); x64 = np.zeros(0, np.int64); f64 = np.zeros(0, np.int64)
    for i in range(lo, hi):
        a, b = int(off[i]), int(off[i + 1])
        s = a - int(off[lo])
        mz = G["mz"][a:b].astype(np.float64)
        it = G["it"][a:b].astype(np.float64)
        if b > a:
            elec[s:s + b - a] = electronic_mask(it, 4)
            decimals[i - lo] = n_decimals(mz)
        st, p, z, pmz = prepare(i)
        status[i - lo] = st
        pmz_theo[i - lo] = pmz
        if st != 0 or b == a:
            continue
        if len(e64) != b - a:
            e64 = np.zeros(b - a); ex64 = np.zeros(b - a); x64 = np.zeros(b - a, np.int64); f64 = np.zeros(b - a, np.int64)
        explain_t1(mz, pmz, z, p, tol, 0.0, work, e64, x64, f64, ex64)
        err[s:s + b - a] = e64
        err_x[s:s + b - a] = ex64
        nexp[s:s + b - a] = np.minimum(x64, 32767)
        flag[s:s + b - a] = f64
    return lo, hi, err, err_x, flag, nexp, elec, status, pmz_theo, decimals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--tol", type=float, default=0.025)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=1000)
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.inp, allow_pickle=True)
    G["off"] = z["peak_offsets"]; G["mz"] = z["peak_mz"]; G["it"] = z["peak_intensity"]
    G["formula"] = z["formula"]; G["ptype"] = z["precursor_type"]; G["pmz"] = z["precursor_mz"]
    n = len(G["off"]) - 1 if not a.limit else min(a.limit, len(G["off"]) - 1)
    print(f"loaded {n} spectra, {int(G['off'][n])} ions in {time.time() - t0:.0f} s", flush=True)
    run_chunk((0, 2, a.tol))                     # compile in the parent: forked workers inherit the machine code
    n_ion = int(G["off"][n])
    err = np.full(n_ion, np.inf, np.float32); err_x = np.full(n_ion, np.inf, np.float32)
    flag = np.zeros(n_ion, np.int8); nexp = np.zeros(n_ion, np.int16); decimals = np.zeros(n, np.int8)
    elec = np.zeros(n_ion, np.bool_); status = np.zeros(n, np.int8); pmz_theo = np.zeros(n)
    chunks = [(lo, min(lo + a.chunk, n), a.tol) for lo in range(0, n, a.chunk)]
    done = 0
    with Pool(a.workers) as pool:
        for lo, hi, e, ex, f, x, el, st, pt, dc in pool.imap_unordered(run_chunk, chunks):
            s0, s1 = int(G["off"][lo]), int(G["off"][hi])
            err[s0:s1] = e; err_x[s0:s1] = ex; flag[s0:s1] = f; nexp[s0:s1] = x; elec[s0:s1] = el
            status[lo:hi] = st; pmz_theo[lo:hi] = pt; decimals[lo:hi] = dc
            done += hi - lo
            if done % (50 * a.chunk) < a.chunk:
                print(f"  {done}/{n} spectra, {time.time() - t0:.0f} s", flush=True)
    np.savez(a.out, err=err, err_x=err_x, flag=flag, nexp=nexp, elec=elec, status=status, pmz_theo=pmz_theo,
             decimals=decimals, tol=a.tol, n=n)
    print(f"done {n} spectra in {time.time() - t0:.0f} s; status counts {np.bincount(status, minlength=5)}", flush=True)


if __name__ == "__main__":
    main()
