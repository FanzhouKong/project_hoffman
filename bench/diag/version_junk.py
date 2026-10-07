# diagnostic: junk / good counts of each peak3d version's actual benchmark output, judged with ONE noise
# reference for every version (the post-merge 3D noise surface of each file, computed once), so the
# checks are equally strict for all versions. Junk = no peak in either other injection of the same
# sample, or the feature's own smoothed trace fails the EIC check (junk_classes.py); good = a peak
# in at least half the other injections and a clean trace.
# usage: version_junk.py LABEL=PEAK3D_RESULTS_DIR ... -- RUN ...
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/diag"))
from bench.runs import RUNS  # noqa: E402
from junk_classes import classify_file, rep_check  # noqa: E402
from peak3d.estimate import estimate  # noqa: E402
from peak3d.io import load_cloud  # noqa: E402
from peak3d.pick import pick_cloud  # noqa: E402


def reference_noise(cloud):
    P = estimate(cloud)
    b = pick_cloud(cloud, P)
    S = b.noise_surface
    mz_lo, mz_hi = float(cloud.mz.min()), float(cloud.mz.max()) * (1 + 1e-9)
    lw = (np.log(mz_hi) - np.log(mz_lo)) / P.n_mz_bands + 1e-12

    def look(mz, rt):
        band = np.clip(((np.log(mz) - np.log(mz_lo)) / lw).astype(int), 0, S.shape[0] - 1)
        blk = np.clip(((rt - cloud.rt[0]) / P.rt_block_min).astype(int), 0, S.shape[1] - 1)
        return S[band, blk]
    return look


def one(job):
    run, fi, versions = job
    files = RUNS[run]["files"]
    stems = [Path(f).stem for f in files]
    stem = stems[fi]
    cloud = load_cloud(files[fi])
    look = reference_noise(cloud)
    others_c = {k: load_cloud(files[k]) for k in range(len(files)) if k != fi}
    out = {}
    for label, d in versions.items():
        d = Path(d) / run / "peak3d_out"
        F = pd.read_csv(d / "features" / f"{stem}.tsv", sep="\t")
        if "group_kept" in F.columns:                              # versions with a presence rule: only reported groups count
            F = F[F["group_kept"]].reset_index(drop=True)
        P = json.load(open(d / "features" / f"{stem}.params.json"))
        F["noise"] = look(F["mz"].values, F["rt"].values)        # the same reference for every version
        C = classify_file(cloud, F, P)
        others = [(others_c[k], pd.read_csv(d / "rt_correction" / f"{stems[k]}.tsv", sep="\t"),
                   json.load(open(d / "features" / f"{stems[k]}.params.json"))) for k in others_c]
        rep = rep_check(F.assign(_top=C["top_s"].values), P, others)
        out[label] = pd.DataFrame({"cls": C["cls"].values, "rep": rep, "height": F["height"].values})
    return run, stem, out


def main(versions, runs):
    jobs = [(r, k, versions) for r in runs for k in range(len(RUNS[r]["files"]))]
    from peak3d.pipeline import _warm_jit
    _warm_jit()
    with get_context("spawn").Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))) as pool:
        res = pool.map(one, jobs, chunksize=1)
    rows = []
    for run in runs:
        for label in versions:
            A = pd.concat([o[label] for r, s, o in res if r == run])
            norep = A["rep"] == 0
            eic = A["cls"] != "ok"
            rows.append(dict(run=run, version=label, features=len(A), good=int(((A["rep"] >= 0.5) & ~eic).sum()),
                             junk=int((norep | eic).sum()), junk_no_replicate=int(norep.sum()),
                             junk_eic=int(eic.sum()), junk_share=round(float((norep | eic).mean()), 3)))
    T = pd.DataFrame(rows)
    T.to_csv(ROOT / "results/diag_excess/version_junk.tsv", sep="\t", index=False)
    print(T.to_string(index=False))


if __name__ == "__main__":
    a = sys.argv[1:]
    k = a.index("--")
    main(dict(x.split("=", 1) for x in a[:k]), a[k + 1:])
