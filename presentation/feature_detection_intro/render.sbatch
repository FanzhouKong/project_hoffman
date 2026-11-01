#!/bin/bash
# Name this short rendering job.
#SBATCH --job-name=feature-intro
# Use the available metabolomics group CPU allocation.
#SBATCH --account=metabolomicsgrp
#SBATCH --partition=high
# Limit rendering to two minutes, one task, two CPUs, and 2 GB RAM.
#SBATCH --time=00:02:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
# Keep logs unique to each job on Quobyte.
#SBATCH --output=presentation/feature_detection_intro/render-%j.out
#SBATCH --error=presentation/feature_detection_intro/render-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/feature-intro-mpl-$SLURM_JOB_ID"
envs/peak3d/bin/python presentation/feature_detection_intro/render.py "$@"
