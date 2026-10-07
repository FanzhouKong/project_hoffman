#!/bin/bash
# MassIVE files, one at a time (see 04_download_massive.py). Resubmit to resume.
#SBATCH --job-name=peakbench-massive
#SBATCH --account=metabolomicsgrp
#SBATCH --partition=high
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/massive-%j.out
#SBATCH --error=logs/massive-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# yeast first: it is needed for the credentialing benchmark
python3 scripts/04_download_massive.py NETID_YEAST || echo "NETID_YEAST had failures (see FAILED lines); continuing"
python3 scripts/04_download_massive.py SLAW_ECOLI
