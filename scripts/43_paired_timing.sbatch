#!/bin/bash
# Paired runtime of two peak3d variants on the same node: for each run, the inputs are staged once and the
# variants run alternately A B A B (outputs discarded, /usr/bin/time kept), so node load hits both alike.
#   sbatch scripts/43_paired_timing.sbatch "base2 G2" HZV029_full LI2018 BM21_HILIC
# Variants and their --set options: manifest/peak3d_variants.tsv. Times: results/diag_double/timing/<run>/<i>_<variant>.time
#SBATCH --job-name=peak3d-timing
#SBATCH --account=metabolomicsgrp
#SBATCH --partition=high
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=zen4                 # same CPU generation as the benchmark
#SBATCH --output=logs/bench/timing-%j.out
#SBATCH --error=logs/bench/timing-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
ROOT=$PWD
PEAK3D_ENV=${PEAK3D_ENV:-$ROOT/envs/peak3d}
read -r -a variants <<<"$1"; shift
export NUMBA_CACHE_DIR="$TMPDIR/.numba" MPLCONFIGDIR="$TMPDIR/.mpl"
for run in "$@"; do
  work="$TMPDIR/timing_$run"; mkdir -p "$work/data"
  while IFS= read -r f; do cp "$f" "$work/data/"; done < "results/inputs/$run.txt"
  out="results/diag_double/timing/$run"; mkdir -p "$out"
  i=0
  for rep in 1 2; do
    for v in "${variants[@]}"; do
      i=$((i + 1))
      sets=$(awk -F'\t' -v v="$v" 'NR>1 && $1==v {print $2}' manifest/peak3d_variants.tsv)
      setargs=(); for s in $sets; do setargs+=(--set "$s"); done
      rm -rf "$work/out" "$work/.cache"
      (cd "$work" && /usr/bin/time -v -o "$ROOT/$out/${i}_$v.time" \
        env PYTHONPATH="$ROOT" "$PEAK3D_ENV/bin/python" -m peak3d process --input data --output out \
        --cores "$SLURM_CPUS_PER_TASK" --cache-dir "$work/.cache" "${setargs[@]}" > "$ROOT/$out/${i}_$v.log" 2>&1)
      echo "$run $i $v $(grep -E 'Elapsed' "$out/${i}_$v.time" | awk '{print $NF}') on $(hostname)"
    done
  done
  rm -rf "$work"
done
