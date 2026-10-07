#!/bin/bash
# Runs once the download job ends: resubmit conversions so late MTBLS1684 files are included.
#SBATCH --job-name=peakbench-resubmit
#SBATCH --account=metabolomicsgrp
#SBATCH --partition=high
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=logs/resubmit-%j.out
#SBATCH --error=logs/resubmit-%j.err
set -euo pipefail
source /etc/profile.d/modules.sh
cd "$SLURM_SUBMIT_DIR"
scripts/13_submit_conversions.sh
