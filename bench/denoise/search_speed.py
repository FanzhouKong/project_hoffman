"""Denoising search speed (2026-11-03): T1 against each library candidate's precursor ion formula vs the T0 top-5 copies.

    python search_speed.py RUNS OUT.jsonl LIBRARY [LIBRARY ...]

RUNS holds `peak3d process` outputs (the MassCube-paper DDA files). Per run: the MS2 mapping is redone with the current
code (no denoising), then the T0 top-5 copies are timed (ms2.denoise_copies, the cost the processing pays today). Per
library: read once, one warm-up search (compiles the kernels), then every run searched with denoise="off" (raw copies
only), "t0" (raw + the T0 copies) and "search" (raw + per-candidate T1), the search once more with the ion formula
table warm and once on one thread; finally the T1 step alone without the (query, ion formula) deduplication, and
specdenoise.denoise() called per pair in Python on the first 2000 pairs of each run (the naive implementation). One
JSON line per measurement."""
import json
import os
import resource
import sys
import time
from pathlib import Path

import numba
import numpy as np

from peak3d import library, ms2
from peak3d.library import _MODE_CODE, _ion_ids, candidate_pairs, identify

N_NAIVE = 2000


def rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)


def t1_inputs(run, lib, raw):
    """the inputs _search_denoised builds: the query copy, candidate pairs, each pair's ion formula id"""
    q = ms2.read_representatives(run, raw)
    if q is None:
        return None
    q_mode = np.array([_MODE_CODE.get(m, 0) for m in q["ion_mode"]], np.int8)
    pq, pl = candidate_pairs(q["prec_mz"], q_mode, lib.precursor_mz, lib.mode, 10.0, None, lib.precursor_index())
    if not len(pq):
        return None
    u = np.unique(pl)
    pair_ion = _ion_ids(lib, u)[np.searchsorted(u, pl)].astype(np.int64)
    return q, pq, pl, pair_ion


def main(runs_dir, out, *libs):
    from specdenoise import Params, denoise
    from specdenoise import formula as fm
    from specdenoise.search import denoise_keys, unique_keys
    runs = sorted(p for p in Path(runs_dir).iterdir() if (p / "aligned_feature_table.tsv").exists())
    n_thr = numba.get_num_threads()

    def emit(**k):
        with open(out, "a") as fh:
            fh.write(json.dumps(k) + "\n")
        print(json.dumps(k), flush=True)

    emit(step="start", node=os.uname().nodename, threads=n_thr, runs=[r.name for r in runs])
    t = time.perf_counter()                    # compile the T0 kernels and build the loss table, untimed below
    ms2.map_to_groups(runs[0], workers=n_thr, denoise=False)
    ms2.denoise_copies(runs[0], workers=n_thr)
    emit(step="t0_warmup", seconds=round(time.perf_counter() - t, 2))
    for run in runs:
        t = time.perf_counter()
        ms2.map_to_groups(run, workers=n_thr, denoise=False)
        t_map = time.perf_counter() - t
        t = time.perf_counter()
        counts = ms2.denoise_copies(run, workers=n_thr)
        emit(step="t0_copies", run=run.name, seconds=round(time.perf_counter() - t, 3), map_seconds=round(t_map, 3),
             spectra={k: v[0] for k, v in counts.items()}, peaks={k: v[1] for k, v in counts.items()},
             kept={k: v[2] for k, v in counts.items()})

    for lib_path in libs:
        name = Path(lib_path).name
        t = time.perf_counter()
        lib = library.load(lib_path)
        emit(step="load", library=name, seconds=round(time.perf_counter() - t, 2), n_spectra=len(lib), rss_gb=rss_gb())
        t = time.perf_counter()
        identify(runs[0], lib_path, denoise="search")
        identify(runs[0], lib_path, denoise="t0")
        emit(step="warmup", library=name, seconds=round(time.perf_counter() - t, 2))
        lib._ions = None                       # the timed searches start with an empty ion formula table

        def search(run, mode, tag):
            t = time.perf_counter()
            identify(run, lib_path, denoise=mode)
            s = time.perf_counter() - t
            info = json.loads((run / "identifications.json").read_text())
            emit(step="search", library=name, run=run.name, mode=mode, tag=tag, threads=numba.get_num_threads(),
                 seconds=round(s, 3), n_queries=info["n_queries"], n_identified=info["n_identified"],
                 n_hits=info["n_hits"], stats=info.get("denoise_stats"), rss_gb=rss_gb())

        for run in runs:
            for mode in ("off", "t0", "search"):
                search(run, mode, "cold" if mode == "search" else "")
        for run in runs:
            search(run, "search", "warm")
        numba.set_num_threads(1)
        for run in runs:
            search(run, "search", "warm")
            search(run, "off", "")
        numba.set_num_threads(n_thr)

        p = Params()
        for run in runs:
            for raw in ms2.RAW:
                x = t1_inputs(run, lib, raw)
                if x is None:
                    continue
                q, pq, pl, pair_ion = x
                tab = lib._ions
                key_q, key_ion, pair_key = unique_keys(pq, pair_ion, q["prec_mz"], tab.mz, tab.Z)
                has = np.flatnonzero(pair_key >= 0)
                off = np.asarray(q["off"], np.int64)
                mz, it = np.asarray(q["mz"], np.float64), np.asarray(q["inten"], np.float64)
                t = time.perf_counter()
                denoise_keys(off, mz, it, q["prec_mz"], key_q, tab.P[key_ion], tab.Z[key_ion], p)
                t_dedup = time.perf_counter() - t
                t = time.perf_counter()
                denoise_keys(off, mz, it, q["prec_mz"], pq[has], tab.P[pair_ion[has]], tab.Z[pair_ion[has]], p)
                t_pairs = time.perf_counter() - t
                sub = has[:N_NAIVE]
                t = time.perf_counter()
                for k in sub.tolist():          # the ion formula as the formula of an [M]+ / [M]- ion: tier T1 as above
                    i = int(pq[k])
                    j = int(pair_ion[k])
                    denoise(mz[off[i]:off[i + 1]], it[off[i]:off[i + 1]], q["prec_mz"][i], formula=fm.format(tab.P[j]),
                            adduct="[M]+" if tab.Z[j] > 0 else "[M]-", params=p)
                t_naive = time.perf_counter() - t
                emit(step="t1_only", library=name, run=run.name, copy=raw, pairs=int(len(pq)), pairs_t1=int(len(has)),
                     keys=int(len(key_q)), queries=int(len(off) - 1), seconds_dedup=round(t_dedup, 4),
                     seconds_per_pair_kernel=round(t_pairs, 4), naive_pairs=int(len(sub)),
                     naive_seconds=round(t_naive, 3),
                     naive_extrapolated=round(t_naive / max(len(sub), 1) * len(has), 2))
    emit(step="end", rss_gb=rss_gb())


if __name__ == "__main__":
    main(*sys.argv[1:])
