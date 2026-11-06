#!/bin/bash
# Compound grouping on real data: Mosa (peak3d, current defaults) on one PR002464 study, then the spike-truth
# check of the compound groups (bench/diag/compound_truth.py).
#   sbatch --array=1-3 scripts/46_compounds_pr2464.sbatch            # 1 ST004500, 2 ST004501, 3 ST004502
# Inputs are copied to node-local scratch (as the benchmark does); the output, with the point-cloud cache so that
# `peak3d align` can regroup without re-picking, goes to results/compounds_<date>/<study>/peak3d_out/.
#SBATCH --job-name=compounds
#SBATCH --account=metabolomicsgrp
#SBATCH --partition=high
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=logs/compounds/run-%A_%a.out
#SBATCH --error=logs/compounds/run-%A_%a.err

set -euo pipefail
source /etc/profile.d/modules.sh
cd "$SLURM_SUBMIT_DIR" || { echo "NO_MOUNT $(hostname)"; exit 9; }
ROOT=$PWD
PEAK3D_ENV=${PEAK3D_ENV:-$ROOT/envs/peak3d}
DATE=${DATE:-$(date +%F)}
studies=(ST004500 ST004501 ST004502)
study=${studies[$((SLURM_ARRAY_TASK_ID - 1))]}
run=PR2464_$study
out="results/compounds_$DATE/$study"
mkdir -p "$out"

work="$TMPDIR/compounds_$study"; mkdir -p "$work/data"
trap 'rm -rf "$work" 2>/dev/null || true' EXIT
n=0
while IFS= read -r f; do cp "$f" "$work/data/"; n=$((n + 1)); done < "results/inputs/$run.txt"
echo "$study: $n files on $(hostname), $SLURM_CPUS_PER_TASK CPUs"
export NUMBA_CACHE_DIR="$work/.numba" MPLCONFIGDIR="$work/.mpl"

cd "$work"
/usr/bin/time -v env PYTHONPATH="$ROOT" "$PEAK3D_ENV/bin/python" -m peak3d process --input data --output peak3d_out \
    --cores "$SLURM_CPUS_PER_TASK" --compounds 2>&1 | tee "$ROOT/$out/peak3d.log"
cd "$ROOT"
rm -rf "$out/peak3d_out"
cp -r "$work/peak3d_out" "$out/peak3d_out"
echo "copied to $out/peak3d_out"

env PYTHONPATH="$ROOT" "$PEAK3D_ENV/bin/python" bench/diag/compound_truth.py --out "$out/peak3d_out" --study "$study" \
    --report "$out/compound_truth" 2>&1 | tee "$out/compound_truth.log"
touch "$out/DONE"
