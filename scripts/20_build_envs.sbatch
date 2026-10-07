#!/bin/bash
# Build one isolated env per tool (pinned versions) under envs/ on group storage.
#SBATCH --job-name=peakbench-envs
#SBATCH --account=metabolomicsgrp
#SBATCH --partition=high
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/envs-%j.out
#SBATCH --error=logs/envs-%j.err

set -euo pipefail
source /etc/profile.d/modules.sh
module load conda/latest
cd "$SLURM_SUBMIT_DIR"

# keep package caches off the 20 GB home quota
export CONDA_PKGS_DIRS="$SLURM_SUBMIT_DIR/envs/.pkgs"
export PIP_CACHE_DIR="$SLURM_SUBMIT_DIR/envs/.pip"

build() {  # name, pip spec
  [ -x "envs/$1/bin/python" ] || conda create -y -p "envs/$1" python=3.11 pip
  "envs/$1/bin/pip" install "$2"
  "envs/$1/bin/pip" freeze > "envs/$1.freeze.txt"
}
build asari    asari-metabolomics==1.18.5
build masscube masscube==1.2.22

envs/asari/bin/asari --help | head -20 || true
envs/masscube/bin/python -c "import masscube, importlib.metadata as m; print('masscube', m.version('masscube'))"
