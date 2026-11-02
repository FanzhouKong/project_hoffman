#!/bin/bash
# Re-run alignment, grouping, gap filling and compound grouping from an existing compounds run (features + cache),
# then the spike-truth check. Picks nothing, so a rule change is tested in minutes.
#   sbatch --array=1-3 scripts/47_compounds_regroup.sbatch           # 1 ST004500, 2 ST004501, 3 ST004502
#   DATE=2026-10-31 selects the run directory results/compounds_<DATE>/
#   VARIANT=noveto EXTRA="--compound-veto off" regroups into results/compounds_<DATE>_<VARIANT>/<study>/ (features
#   copied, cache shared read-only), leaving the base run untouched:
#   sbatch --export=ALL,DATE=2026-10-31,VARIANT=noveto,EXTRA="--compound-veto off" --array=1-3 scripts/47_compounds_regroup.sbatch
#SBATCH --job-name=cmp-regroup
#SBATCH --account=metabolomicsgrp
#SBATCH --partition=high
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=logs/compounds/regroup-%A_%a.out
#SBATCH --error=logs/compounds/regroup-%A_%a.err

set -euo pipefail
source /etc/profile.d/modules.sh
cd "$SLURM_SUBMIT_DIR" || { echo "NO_MOUNT $(hostname)"; exit 9; }
ROOT=$PWD
PEAK3D_ENV=${PEAK3D_ENV:-$ROOT/envs/peak3d}
DATE=${DATE:-$(date +%F)}
studies=(ST004500 ST004501 ST004502)
study=${studies[$((SLURM_ARRAY_TASK_ID - 1))]}
base="results/compounds_$DATE/$study"
[ -d "$base/peak3d_out/.cache" ] || { echo "no cache in $base/peak3d_out"; exit 2; }
out=$base
cache="$base/peak3d_out/.cache"
if [ -n "${VARIANT:-}" ]; then
    out="results/compounds_${DATE}_$VARIANT/$study"
    mkdir -p "$out/peak3d_out"
    [ -d "$out/peak3d_out/features" ] || cp -r "$base/peak3d_out/features" "$out/peak3d_out/features"
    [ -f "$out/peak3d_out/params.json" ] || cp "$base/peak3d_out/params.json" "$out/peak3d_out/params.json"
fi
export NUMBA_CACHE_DIR="$TMPDIR/numba_$study" MPLCONFIGDIR="$TMPDIR/mpl_$study"
echo "$study regroup on $(hostname), $SLURM_CPUS_PER_TASK CPUs"
env PYTHONPATH="$ROOT" "$PEAK3D_ENV/bin/python" -m peak3d align --output "$out/peak3d_out" --cache-dir "$cache" \
    --cores "$SLURM_CPUS_PER_TASK" ${EXTRA:-} 2>&1 | tee "$out/regroup.log"
env PYTHONPATH="$ROOT" "$PEAK3D_ENV/bin/python" bench/diag/compound_truth.py --out "$out/peak3d_out" --study "$study" \
    --report "$out/compound_truth" 2>&1 | tee "$out/compound_truth.log"
echo REGROUP_DONE
