#!/usr/bin/env python3
"""Write manifest/bench_jobs.tsv (row N = array task N of 31_run_tool.sbatch) and one
input list per run in results/inputs/<run>.txt. Runs with missing files are skipped
and reported, so rerun this once downloads/conversions finish.

usage: 30_make_jobs.py [RUN ...]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.runs import RUNS, TOOLS  # noqa: E402

want = sys.argv[1:] or list(RUNS)
inp = ROOT / "results/inputs"
inp.mkdir(parents=True, exist_ok=True)
jobs = []
for run in want:
    spec = RUNS[run]
    files = spec["files"]
    missing = [f for f in files if not Path(f).exists()]
    if not files or missing:
        print(f"skip {run}: {len(missing)} of {len(files)} files missing")
        continue
    (inp / f"{run}.txt").write_text("\n".join(files) + "\n")
    for tool in TOOLS:
        jobs.append((tool, run, spec["mode"], len(files)))
    print(f"{run}: {len(files)} files")

with (ROOT / "manifest/bench_jobs.tsv").open("w") as fh:
    fh.write("task\ttool\trun\tmode\tn_files\n")
    for i, j in enumerate(jobs, 1):
        fh.write("\t".join(map(str, (i, *j))) + "\n")
print(f"{len(jobs)} jobs -> manifest/bench_jobs.tsv")
