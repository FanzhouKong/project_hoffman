#!/bin/bash
# The peak3d test suite from the gc worktree (after a change to peak3d/cdf.py):
#   cd .worktrees/gc && sbatch scripts/56_tests_gc.sbatch
#SBATCH --job-name=peak3d-tests
#SBATCH --account=metabolomicsgrp        # group pool
#SBATCH --partition=high                 # not preemptible
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/peak3d-tests-%j.out  # %j: one writer per log on Quobyte
#SBATCH --error=logs/peak3d-tests-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || { echo "NO_MOUNT $(hostname)"; exit 9; }
MAIN=${MAIN:-/quobyte/metabolomicsgrp/fanzhou/hoffmann}
PY=${PEAK3D_ENV:-$MAIN/envs/peak3d}/bin/python
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/.numba" MPLCONFIGDIR="${TMPDIR:-/tmp}/.mpl"
"$PY" -m pytest peak3d/tests -q -p no:cacheprovider ${PYTEST_ARGS:-} 2>&1 | tail -40
