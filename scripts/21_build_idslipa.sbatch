#!/bin/bash
# Install IDSL.IPA 3.0 (GitHub release) + writexl into a project R library (envs/Rlib).
#SBATCH --job-name=peakbench-idsl-env
#SBATCH --account=metabolomicsgrp
#SBATCH --partition=high
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/idsl-env-%j.out
#SBATCH --error=logs/idsl-env-%j.err
set -euo pipefail
source /etc/profile.d/modules.sh
module load R/4.4.2
cd "$SLURM_SUBMIT_DIR"
export R_LIBS_USER="$SLURM_SUBMIT_DIR/envs/Rlib"
mkdir -p "$R_LIBS_USER"
Rscript -e '
lib <- Sys.getenv("R_LIBS_USER")
options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 4)
install.packages(c("IDSL.MXP", "readxl", "writexl"), lib = lib)
# 3.0 is on GitHub only (CRAN has 2.9); the GitHub parameter template is written for 3.0
# IDSL.IPA 3.0 needs IDSL.MXP 2.1 (peak2list(onlyScanTable=)); both are GitHub releases
gh <- "https://raw.githubusercontent.com/idslme"
for (url in c(file.path(gh, "IDSL.MXP/main/releases/IDSL.MXP_2.1.tar.gz"),
              file.path(gh, "IDSL.IPA/main/releases/IDSL.IPA_3.0.tar.gz"))) {
  tgz <- file.path(tempdir(), basename(url)); download.file(url, tgz)
  install.packages(tgz, lib = lib, repos = NULL, type = "source")
}
library(IDSL.IPA, lib.loc = lib)
cat("IDSL.IPA", as.character(packageVersion("IDSL.IPA", lib.loc = lib)),
    "IDSL.MXP", as.character(packageVersion("IDSL.MXP", lib.loc = lib)), "\n")
writeLines(capture.output(sessionInfo()), file.path(lib, "..", "Rlib.sessionInfo.txt"))
'
