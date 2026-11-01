#!/bin/bash
# Build the PR002464 spiked-standard truth for one study, then its QC gallery.
#   sbatch scripts/44_spiketruth.sbatch ST004500      (ST004501 | ST004502 | ST004504)
#   sbatch scripts/44_spiketruth.sbatch ST004500 gallery   (redraw the QC gallery only)
#SBATCH --job-name=pr2464-truth
#SBATCH --account=metabolomicsgrp
#SBATCH --partition=high
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G                       # peak ~3.5 GB (ST004501, 2026-10-21)
#SBATCH --output=logs/spiketruth-%j.out
#SBATCH --error=logs/spiketruth-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export MPLCONFIGDIR="$TMPDIR/mpl"
# Quobyte reads of env files can block for minutes at Python start-up (2026-10-19/21): a hung step
# is killed after its timeout and retried (exit 137 = killed by timeout; other failures are real)
retry() {
  local t=$1; shift
  for i in 1 2 3; do
    if timeout -s KILL "$t" "$@"; then return 0; else rc=$?; fi
    [ "$rc" -eq 137 ] || return "$rc"
    echo "attempt $i of '$*' killed after $t" >&2
  done
  return 1
}
if [ "${2:-}" != gallery ]; then   # "gallery" as 2nd argument: redraw the gallery only
  retry 30m /usr/bin/time -v envs/peak3d/bin/python bench/spiketruth.py "$1" --cpus "$SLURM_CPUS_PER_TASK"
fi
retry 20m envs/peak3d/bin/python bench/spike_gallery.py "$1"
