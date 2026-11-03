#!/bin/bash
#SBATCH --job-name=peak3d-noise-figure
#SBATCH --account=metabolomicsgrp
#SBATCH --partition=high
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=presentation/peak3d/noise-render-%j.out
#SBATCH --error=presentation/peak3d/noise-render-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export NUMBA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLCONFIGDIR="$TMPDIR/peak3d-noise-mpl-$SLURM_JOB_ID"
export NUMBA_CACHE_DIR="$TMPDIR/peak3d-noise-numba-$SLURM_JOB_ID"
envs/peak3d/bin/python presentation/peak3d/render_noise_cloud.py
