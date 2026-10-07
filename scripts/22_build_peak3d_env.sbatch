#!/bin/bash
# Build the isolated env for the in-house peak3d tool under envs/ on group storage.
# Versions are pinned to the ones already validated in envs/masscube.
#SBATCH --job-name=peakbench-env-peak3d
#SBATCH --account=metabolomicsgrp
#SBATCH --partition=high
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/envs-peak3d-%j.out
#SBATCH --error=logs/envs-peak3d-%j.err

set -euo pipefail
source /etc/profile.d/modules.sh
module load conda/latest
cd "$SLURM_SUBMIT_DIR"

# keep package caches off the 20 GB home quota
export CONDA_PKGS_DIRS="$SLURM_SUBMIT_DIR/envs/.pkgs"
export PIP_CACHE_DIR="$SLURM_SUBMIT_DIR/envs/.pip"

env=envs/peak3d
[ -x "$env/bin/python" ] || conda create -y -p "$env" python=3.11 pip
"$env/bin/pip" install \
  numpy==1.26.4 scipy==1.17.1 numba==0.68.0 pandas==3.0.6 pyteomics==4.6.3 \
  matplotlib==3.11.2 lxml==6.1.3 pytest
"$env/bin/pip" freeze > "$env.freeze.txt"

"$env/bin/python" - <<'PY'
import numpy, scipy, numba, pandas, pyteomics, matplotlib, lxml
print("numpy", numpy.__version__, "scipy", scipy.__version__, "numba", numba.__version__,
      "pandas", pandas.__version__, "pyteomics", pyteomics.__version__)
PY
