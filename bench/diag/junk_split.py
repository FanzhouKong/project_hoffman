# diagnostic: the replicate-junk share of peak3d variants (version_junk.py, as variant_eval.py scores it) split into
# its two criteria: not reproduced (no peak in either other injection of the same sample) and fails the trace check
# (the feature's own smoothed trace fails the EIC check against one fixed noise reference per file). Only the first
# is independent of how a variant judges noise; the second partly restates the reference's own EIC rule.
# usage: junk_split.py VARIANT ...   (results/peak3d_<VARIANT>; run on a compute node, SLURM_CPUS_PER_TASK workers)
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/diag"))
import version_junk as vj  # noqa: E402
from bench.runs import RUNS  # noqa: E402
from peak3d.pipeline import _warm_jit  # noqa: E402

RUNS_J = ["HZV029_cert", "YEAST_NEG", "SZ22_12C", "YEAST_12C"]


def main():
    versions = {v: str(ROOT / f"results/peak3d_{v}") for v in sys.argv[1:]}
    jobs = [(r, k, versions) for r in RUNS_J for k in range(len(RUNS[r]["files"]))]
    _warm_jit()
    with get_context("spawn").Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))) as pool:
        res = pool.map(vj.one, jobs, chunksize=1)
    rows = []
    for run in RUNS_J:
        for v in versions:
            A = pd.concat([o[v] for r, s, o in res if r == run])
            norep, bad = A["rep"] == 0, A["cls"] != "ok"
            rows.append(dict(run=run, variant=v, features=len(A), junk=int((norep | bad).sum()),
                             not_reproduced=int(norep.sum()), fails_trace_check=int(bad.sum()),
                             only_not_reproduced=int((norep & ~bad).sum()), only_trace_check=int((bad & ~norep).sum())))
    T = pd.DataFrame(rows)
    out = ROOT / "results/diag_double" / ("junk_split_" + "_".join(versions) + ".tsv")
    T.to_csv(out, sep="\t", index=False)
    pd.set_option("display.width", 200)
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
