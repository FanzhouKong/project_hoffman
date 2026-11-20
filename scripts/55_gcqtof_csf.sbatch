#!/bin/bash
# Mosa GC-MS mode on the CSF .cdf files in data/gcqtof (22 runs, 17 scans/s).
# Submit from the gc worktree (the code that runs is the worktree's):
#   cd .worktrees/gc && sbatch scripts/55_gcqtof_csf.sbatch
#   sbatch --export=ALL,TAG=_x,EXTRA_PROCESS="--set name=value" scripts/55_gcqtof_csf.sbatch
#SBATCH --job-name=gcqtof-csf
#SBATCH --account=metabolomicsgrp        # group pool
#SBATCH --partition=high                 # not preemptible
#SBATCH --time=02:00:00                  # 22 files of ~95 MB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16               # one program, 16 worker processes
#SBATCH --mem=64G
#SBATCH --output=logs/gcqtof-csf-%j.out  # %j: one writer per log on Quobyte
#SBATCH --error=logs/gcqtof-csf-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || { echo "NO_MOUNT $(hostname)"; exit 9; }
MAIN=${MAIN:-/quobyte/metabolomicsgrp/fanzhou/hoffmann}
PY=${PEAK3D_ENV:-$MAIN/envs/peak3d}/bin/python
IN=${IN:-$MAIN/data/gcqtof}
TAG=${TAG:-}
OUT=$MAIN/results/gcqtof_csf/mosa$TAG
EXTRA_PROCESS=${EXTRA_PROCESS:-}
CORES=${SLURM_CPUS_PER_TASK:-4}
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/.numba" MPLCONFIGDIR="${TMPDIR:-/tmp}/.mpl"

mkdir -p "$(dirname "$OUT")"
rm -rf "$OUT"
echo "== peak3d process --mode gc $EXTRA_PROCESS -> $OUT on $(hostname)"
git log --oneline -1; git status --short peak3d | head
# shellcheck disable=SC2086
/usr/bin/time -v "$PY" -m peak3d process --input "$IN" --output "$OUT" --cores "$CORES" --keep-rejected --mode gc $EXTRA_PROCESS
touch "$OUT/DONE"
