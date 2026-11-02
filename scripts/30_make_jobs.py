#!/usr/bin/env python3
"""Write manifest/bench_jobs.tsv (row N = array task N of 31_run_tool.sbatch) and one
input list per run in results/inputs/<run>.txt. Runs with missing files are skipped
and reported, so rerun this once downloads/conversions finish.

usage: 30_make_jobs.py [--tools T1,T2] [RUN ...]   (default: every tool in bench/runs.TOOLS, every run)
A run of bench/runs.MZTEST_RUNS (PR2464_<study>_mztest: every injection, m/z-calibration tests only) is
written when named; it is never part of the default list.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.runs import MZTEST_RUNS, RUNS, TOOLS  # noqa: E402

args = sys.argv[1:]
tools = TOOLS
if args[:1] == ["--tools"]:
    tools, args = args[1].split(","), args[2:]
    unknown = set(tools) - set(TOOLS)
    if unknown:
        sys.exit(f"unknown tools {sorted(unknown)}; known: {TOOLS}")
want = args or list(RUNS)
inp = ROOT / "results/inputs"
inp.mkdir(parents=True, exist_ok=True)
jobs = []
for run in want:
    spec = RUNS[run] if run in RUNS else MZTEST_RUNS[run]
    files = spec["files"]
    missing = [f for f in files if not Path(f).exists()]
    if not files or missing:
        print(f"skip {run}: {len(missing)} of {len(files)} files missing")
        continue
    (inp / f"{run}.txt").write_text("\n".join(files) + "\n")
    for tool in tools:
        jobs.append((tool, run, spec["mode"], len(files)))
    print(f"{run}: {len(files)} files")

with (ROOT / "manifest/bench_jobs.tsv").open("w") as fh:
    fh.write("task\ttool\trun\tmode\tn_files\n")
    for i, j in enumerate(jobs, 1):
        fh.write("\t".join(map(str, (i, *j))) + "\n")
print(f"{len(jobs)} jobs -> manifest/bench_jobs.tsv")
