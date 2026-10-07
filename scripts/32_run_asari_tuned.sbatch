#!/bin/bash
# asari on IDSL003 with a lowered peak-height cutoff (TUNED rows; the default run stays in results/asari/).
# The default min_peak_height 1e5 is above 92 % of the IDSL003 true peaks (low-count TOF file).
#   task 1 = asari_autoheight : --autoheight True (asari's own estimate from 12C/13C landmark tracks)
#   task 2 = asari_h1000      : --min_peak_height 1000 (asari's single-sample default; sensitivity check)
#   sbatch --array=1-2 scripts/32_run_asari_tuned.sbatch
# Same resources and staging as scripts/31_run_tool.sbatch, results to results/<variant>/IDSL003/.
#SBATCH --job-name=asari-tuned
#SBATCH --account=metabolomicsgrp
#SBATCH --partition=high
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=zen4                 # same CPU generation as the benchmark array
#SBATCH --output=logs/bench/asari-tuned-%A_%a.out
#SBATCH --error=logs/bench/asari-tuned-%A_%a.err

set -euo pipefail
source /etc/profile.d/modules.sh
cd "$SLURM_SUBMIT_DIR"
ROOT=$PWD
run=IDSL003
mode=pos

case "$SLURM_ARRAY_TASK_ID" in
  1) variant=asari_autoheight; extra=(--autoheight True) ;;
  2) variant=asari_h1000;      extra=(--min_peak_height 1000) ;;
  *) echo "unknown task $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac
out="results/$variant/$run"
if [ -f "$out/DONE" ]; then echo "done already: $out"; exit 0; fi
rm -rf "$out"; mkdir -p "$out"

work="$TMPDIR/bench_${variant}_${run}"; mkdir -p "$work/data"
trap 'rm -rf "$work" 2>/dev/null || true' EXIT
while IFS= read -r f; do cp "$f" "$work/data/"; done < "results/inputs/$run.txt"
echo "$variant $run (${extra[*]}) on $(hostname), $SLURM_CPUS_PER_TASK CPUs"

cd "$work"
set +e
/usr/bin/time -v -o "$work/time.txt" \
  "$ROOT/envs/asari/bin/asari" process --mode "$mode" --input data --output asari_out \
  --multicores "$SLURM_CPUS_PER_TASK" "${extra[@]}" > "$work/tool.log" 2>&1
rc=$?
set -e
cd "$ROOT"
(cd "$work" && find . -mindepth 1 -maxdepth 1 ! -name data -exec cp -r {} "$ROOT/$out/" \;)
echo "exit $rc"; [ $rc -eq 0 ] && date > "$out/DONE"
exit $rc
