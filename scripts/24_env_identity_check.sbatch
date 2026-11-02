#!/bin/bash
# After scripts/23_rebuild_env.sbatch: rerun one stored benchmark run per rebuilt env, exactly as
# 31_run_tool.sbatch does, into results/env_check_<date>/ (never results/<tool>/), and compare the
# feature tables byte for byte with the stored outputs.
#   sbatch --array=1-3 scripts/24_env_identity_check.sbatch     (1 peak3d HZV029_cert, 2 asari IDSL003, 3 masscube IDSL003)
#SBATCH --job-name=peakbench-env-check
#SBATCH --account=metabolomicsgrp
#SBATCH --partition=high
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=zen4
#SBATCH --output=logs/bench/envcheck-%A_%a.out
#SBATCH --error=logs/bench/envcheck-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
ROOT=$PWD
case "$SLURM_ARRAY_TASK_ID" in
  1) tool=peak3d run=HZV029_cert ;;
  2) tool=asari run=IDSL003 ;;
  3) tool=masscube run=IDSL003 ;;
esac
out="$ROOT/results/env_check_2026-10-19/$tool/$run"; ref="$ROOT/results/$tool/$run"
rm -rf "$out"; mkdir -p "$out"
work="$TMPDIR/envcheck_${tool}_${run}"; mkdir -p "$work/data"
trap 'rm -rf "$work" 2>/dev/null || true' EXIT
while IFS= read -r f; do cp "$f" "$work/data/"; done < "results/inputs/$run.txt"
export NUMBA_CACHE_DIR="$work/.numba" MPLCONFIGDIR="$work/.mpl"
cd "$work"
case "$tool" in
  peak3d)
    env PYTHONPATH="$ROOT" "$ROOT/envs/peak3d/bin/python" -m peak3d process --input data --output peak3d_out \
      --cores "$SLURM_CPUS_PER_TASK" --cache-dir "$work/.cache" > tool.log 2>&1
    pairs=$(cd peak3d_out && find features -type f; echo aligned_feature_table.tsv) ;;
  asari)
    "$ROOT/envs/asari/bin/asari" process --mode pos --input data --output asari_out \
      --multicores "$SLURM_CPUS_PER_TASK" > tool.log 2>&1
    pairs=export/full_Feature_table.tsv ;;
  masscube)
    "$ROOT/envs/masscube/bin/python" "$ROOT/scripts/run_masscube.py" "$work" > tool.log 2>&1
    pairs=aligned_feature_table.txt ;;
esac
cp -r tool.log "$out/"
n_diff=0
for p in $pairs; do
  case "$tool" in
    peak3d) a="$work/peak3d_out/$p"; b="$ref/peak3d_out/$p" ;;
    asari) a=$(ls -d "$work"/asari_out*/)$p; b=$(ls -d "$ref"/asari_out*/)$p ;;
    masscube) a="$work/$p"; b="$ref/$p" ;;
  esac
  mkdir -p "$out/$(dirname "$p")"; cp "$a" "$out/$p"
  if cmp -s "$a" "$b"; then echo "IDENTICAL $p"; else echo "DIFFERENT $p"; n_diff=$((n_diff+1)); fi
done
echo "$tool $run: $n_diff file(s) differ from $ref"
[ "$n_diff" -eq 0 ]
