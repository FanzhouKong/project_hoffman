#!/bin/bash
# Pull the converter containers into containers/ (apptainer pull must not run on a login node).
#SBATCH --job-name=peakbench-pull
#SBATCH --account=metabolomicsgrp
#SBATCH --partition=high
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/pull-%j.out
#SBATCH --error=logs/pull-%j.err

set -euo pipefail
source /etc/profile.d/modules.sh
cd "$SLURM_SUBMIT_DIR"

# Keep the layer cache off the 20 GB home quota; build in node-local scratch.
export APPTAINER_CACHEDIR="$SLURM_SUBMIT_DIR/containers/.cache"
export APPTAINER_TMPDIR="$TMPDIR"

# Thermo .raw -> mzML (vendor centroiding is on by default)
[ -f containers/trfp_1.4.5.sif ] || \
  apptainer pull containers/trfp_1.4.5.sif \
    docker://quay.io/biocontainers/thermorawfileparser:1.4.5--h05cac1d_1

# ProteoWizard msconvert under Wine, with vendor DLLs (Agilent .d, and a Thermo fallback).
# Using this image means accepting the vendors' reader licenses (research use).
[ -f containers/pwiz_26.1.0.266.sif ] || \
  apptainer pull containers/pwiz_26.1.0.266.sif \
    docker://proteowizard/pwiz-skyline-i-agree-to-the-vendor-licenses:skyline_26.1.0.266-578b175

ls -la containers/
