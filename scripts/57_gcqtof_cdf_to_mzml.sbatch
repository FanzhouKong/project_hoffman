#!/bin/bash
# The CSF .cdf files of data/gcqtof as centroided mzML, the centroids as the files hold them
# (bench/gc/cdf_to_mzml.py --as-is) -> data/mzml/GCQTOF_CSF/<stem>.mzML. Finished files are skipped.
#   cd .worktrees/gc && sbatch scripts/57_gcqtof_cdf_to_mzml.sbatch
#SBATCH --job-name=gcqtof-mzml
#SBATCH --account=metabolomicsgrp        # group pool
#SBATCH --partition=high                 # not preemptible
#SBATCH --time=01:00:00                  # 22 files of ~95 MB, under a minute each
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                # 8 files at a time, one process per file
#SBATCH --mem=32G
#SBATCH --output=logs/gcqtof-mzml-%j.out # %j: one writer per log on Quobyte
#SBATCH --error=logs/gcqtof-mzml-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || { echo "NO_MOUNT $(hostname)"; exit 9; }
MAIN=${MAIN:-/quobyte/metabolomicsgrp/fanzhou/hoffmann}
export PY=${PEAK3D_ENV:-$MAIN/envs/peak3d}/bin/python
export IN=${IN:-$MAIN/data/gcqtof}
export OUT=${OUT:-$MAIN/data/mzml/GCQTOF_CSF}
export WORK="${TMPDIR:-/tmp}/cdf2mzml"
mkdir -p "$OUT" "$WORK"

one() {                                   # written on the node's own disk, then copied once (one writer per file)
  src=$1; stem=$(basename "${src%.*}"); dst="$OUT/$stem.mzML"
  [ -s "$dst" ] && { echo "done already: $dst"; return 0; }
  "$PY" -I bench/gc/cdf_to_mzml.py "$src" "$WORK/$stem.mzML" --as-is --name "$stem"
  cp "$WORK/$stem.mzML" "$dst.part" && mv "$dst.part" "$dst" && rm -f "$WORK/$stem.mzML"
}
export -f one
find "$IN" -maxdepth 1 -iname '*.cdf' -print0 | sort -z | xargs -0 -P "${SLURM_CPUS_PER_TASK:-4}" -I{} bash -c 'one "$1"' _ {}
echo "$(find "$OUT" -maxdepth 1 -name '*.mzML' | wc -l) mzML files in $OUT"
