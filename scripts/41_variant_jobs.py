#!/usr/bin/env python3
"""Write manifest/variant_jobs.tsv: one row per (peak3d variant, benchmark run), row N = array task N of
scripts/42_peak3d_variant.sbatch. Variants and their --set options: manifest/peak3d_variants.tsv. Inputs are
the benchmark's own lists (results/inputs/<run>.txt from 30_make_jobs.py); the largest runs come first.

usage: 41_variant_jobs.py [--out FILE] [--runs R1,R2] [VARIANT ...]   (default: every variant on every run,
       manifest/variant_jobs.tsv; give a separate --out, read by 42_peak3d_variant.sbatch via VARIANT_JOBS, while
       another array still reads the default)
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.runs import RUNS  # noqa: E402

variants = {r["variant"]: r["sets"] for r in csv.DictReader((ROOT / "manifest/peak3d_variants.tsv").open(), delimiter="\t")}
args = sys.argv[1:]
out = ROOT / "manifest/variant_jobs.tsv"
runs = list(RUNS)
while args[:1] in (["--out"], ["--runs"]):
    if args[0] == "--out":
        out = ROOT / args[1]
    else:
        runs = args[1].split(",")
        if set(runs) - set(RUNS):
            sys.exit(f"unknown runs {sorted(set(runs) - set(RUNS))}")
    args = args[2:]
want = args or list(variants)
runs = sorted(runs, key=lambda r: -len(RUNS[r]["files"]))
missing = [r for r in runs if not (ROOT / f"results/inputs/{r}.txt").exists()]
if missing:
    sys.exit(f"no input list for {missing}: run scripts/30_make_jobs.py first")
rows = [(run, v) for run in runs for v in want]
with out.open("w") as fh:
    fh.write("task\tvariant\trun\tn_files\n")
    for i, (run, v) in enumerate(rows, 1):
        fh.write(f"{i}\t{v}\t{run}\t{len(RUNS[run]['files'])}\n")
print(f"{len(rows)} jobs ({len(want)} variants x {len(runs)} runs) -> {out.relative_to(ROOT)}")
