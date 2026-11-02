#!/bin/bash
# Yeast-credentialed and plasma truth for one PR002464 Orbitrap study (after 44_spiketruth.sbatch, whose
# RT offsets and peaks this uses).
#   sbatch scripts/45_biotruth.sbatch ST004500      (ST004501 | ST004502)
#   sbatch scripts/45_biotruth.sbatch ST004500 after-cred   (skip the credentialing step)
#SBATCH --job-name=pr2464-bio
#SBATCH --account=metabolomicsgrp
#SBATCH --partition=high
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G                       # credtruth: every worker holds the 4 + 4 yeast runs
#SBATCH --output=logs/biotruth-%j.out
#SBATCH --error=logs/biotruth-%j.err
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
PY=envs/peak3d/bin/python
if [ "${2:-}" != after-cred ]; then   # "after-cred" as 2nd argument: credentialing already done
  retry 10h /usr/bin/time -v $PY bench/credtruth.py "PR2464_$1" --cpus "$SLURM_CPUS_PER_TASK"
fi
# QC gallery + known-metabolite check; not fatal
retry 30m $PY bench/cred_gallery.py "PR2464_$1" || echo "cred_gallery failed (QC only)" >&2
retry 30m $PY bench/biotruth.py yeast "$1" --cpus "$SLURM_CPUS_PER_TASK"
retry 2h /usr/bin/time -v $PY bench/biotruth.py plasma "$1" --cpus "$SLURM_CPUS_PER_TASK"
