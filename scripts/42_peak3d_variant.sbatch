#!/bin/bash
# Run one peak3d variant (row N of manifest/variant_jobs.tsv) on one benchmark run, exactly as
# 31_run_tool.sbatch runs peak3d, plus the variant's --set options (manifest/peak3d_variants.tsv).
# Output: results/peak3d_<variant>/<run>/ (scored by bench/diag/variant_eval.py, never by the benchmark).
#   python3 scripts/41_variant_jobs.py && sbatch --array=1-$(($(wc -l < manifest/variant_jobs.tsv)-1))%20 scripts/42_peak3d_variant.sbatch
# Optional env: PEAK3D_ENV (default envs/peak3d), PEAK3D_REQ (pip freeze of that env: run from a node-local venv),
# VARIANT_JOBS (job list, default manifest/variant_jobs.tsv)
#SBATCH --job-name=peak3d-variant
#SBATCH --account=metabolomicsgrp
#SBATCH --partition=high
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=zen4                 # same CPU generation as the benchmark
#SBATCH --output=logs/bench/variant-%A_%a.out
#SBATCH --error=logs/bench/variant-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
ROOT=$PWD
PEAK3D_ENV=${PEAK3D_ENV:-$ROOT/envs/peak3d}

row=$(awk -F'\t' -v t="$SLURM_ARRAY_TASK_ID" '$1==t' "${VARIANT_JOBS:-manifest/variant_jobs.tsv}")
IFS=$'\t' read -r task variant run nfiles <<<"$row"
sets=$(awk -F'\t' -v v="$variant" 'NR>1 && $1==v {print $2}' manifest/peak3d_variants.tsv)
setargs=()
for s in $sets; do setargs+=(--set "$s"); done
out="results/peak3d_$variant/$run"
if [ -f "$out/DONE" ]; then echo "done already: $out"; exit 0; fi
rm -rf "$out"; mkdir -p "$out"

work="$TMPDIR/variant_${variant}_${run}"; mkdir -p "$work/data"
trap 'rm -rf "$work" 2>/dev/null || true' EXIT
while IFS= read -r f; do cp "$f" "$work/data/"; done < "results/inputs/$run.txt"
echo "[$task] peak3d $variant ($sets) $run ($nfiles files) on $(hostname), $SLURM_CPUS_PER_TASK CPUs"
export NUMBA_CACHE_DIR="$work/.numba" MPLCONFIGDIR="$work/.mpl"
PY="$PEAK3D_ENV/bin/python"
if [ -n "${PEAK3D_REQ:-}" ]; then
  # node-local venv on the env's own interpreter with its pinned wheels (PEAK3D_REQ = its pip freeze): same packages,
  # but site-packages are read from local disk, not from Quobyte, where reads of env files can stall for minutes
  "$PEAK3D_ENV/bin/python" -m venv "$work/venv"
  PIP_NO_CACHE_DIR=1 "$work/venv/bin/pip" install -q --disable-pip-version-check -r "$PEAK3D_REQ"
  PY="$work/venv/bin/python"
  echo "node-local venv ready $(date +%T)"
fi

cd "$work"
set +e
/usr/bin/time -v -o "$work/time.txt" \
  env PYTHONPATH="$ROOT" "$PY" -m peak3d process --input data --output peak3d_out \
  --cores "$SLURM_CPUS_PER_TASK" --cache-dir "$work/.cache" "${setargs[@]}" > "$work/tool.log" 2>&1
rc=$?
set -e
cd "$ROOT"
(cd "$work" && find . -mindepth 1 -maxdepth 1 ! -name data ! -name .numba ! -name .mpl ! -name .cache ! -name venv -exec cp -r {} "$ROOT/$out/" \;)
echo "exit $rc"; [ $rc -eq 0 ] && date > "$out/DONE"
exit $rc
