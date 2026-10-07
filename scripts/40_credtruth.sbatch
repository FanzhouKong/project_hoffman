#!/bin/bash
# Build the tool-independent credentialed truth set for one dataset.
#   sbatch scripts/40_credtruth.sbatch SZ22      (or YEAST)
#SBATCH --job-name=peakbench-cred
#SBATCH --account=metabolomicsgrp
#SBATCH --partition=high
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/cred-%j.out
#SBATCH --error=logs/cred-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export MPLCONFIGDIR="$TMPDIR/mpl"
envs/masscube/bin/python bench/credtruth.py "$1" --cpus "$SLURM_CPUS_PER_TASK"
envs/masscube/bin/python bench/cred_gallery.py "$1"
