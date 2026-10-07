#!/bin/bash
# Download every file in manifest/aria2_input.txt into data/raw/<dataset>/.
# Safe to resubmit: aria2c resumes partial files and skips completed ones.
#SBATCH --job-name=peakbench-dl
#SBATCH --account=metabolomicsgrp       # group pool (no per-job caps)
#SBATCH --partition=high                # not preemptible
#SBATCH --time=2-00:00:00               # ~150 GB; generous for slow repositories
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2               # downloads are network-bound
#SBATCH --mem=4G
#SBATCH --output=logs/download-%j.out   # %j: one writer per log on Quobyte
#SBATCH --error=logs/download-%j.err

set -euo pipefail
source /etc/profile.d/modules.sh
module load aria2/1.37.0

cd "$SLURM_SUBMIT_DIR"
INPUT=${1:-manifest/aria2_input_nognps.txt}   # MassIVE rows: 04_download_massive.sbatch

# -j 4: four files at a time, one connection each (polite to EBI/MassIVE/MWB)
# --file-allocation=none: preallocation is slow on Quobyte
# --auto-file-renaming=false: never write foo.1.raw next to an existing file
aria2c \
  --input-file="$INPUT" \
  --continue=true \
  --max-concurrent-downloads=4 \
  --max-connection-per-server=1 \
  --split=1 \
  --max-tries=10 --retry-wait=30 --timeout=120 \
  --file-allocation=none \
  --auto-file-renaming=false \
  --allow-overwrite=false \
  --check-integrity=true \
  --summary-interval=600 \
  --console-log-level=warn \
  --log="logs/aria2-${SLURM_JOB_ID}.log" --log-level=notice \
  --save-session="logs/aria2-session-${SLURM_JOB_ID}.txt"

echo "aria2c finished; verifying against manifest"
python3 scripts/03_verify_downloads.py
