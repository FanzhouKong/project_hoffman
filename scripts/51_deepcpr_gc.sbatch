#!/bin/bash
# DeepCPR benchmark runs (Yu et al., Anal. Chem. 2026; github.com/YuChuanxiu/DeepCPR release v1.2.0 data.zip):
# 5 amino-acid mixture runs (Agilent quadrupole, 2 scans/s) and 5 fatty-acid mixture runs (Shimadzu quadrupole,
# 5 scans/s), unit-resolution CDF files.
#   1. CDF -> centroid mzML at nominal masses (bench/gc/cdf_to_mzml.py) -> $MAIN/data/mzml/DEEPCPR_GC/<set><TAG>/
#   2. peak3d process --mode gc                                        -> $MAIN/results/gc_deepcpr/mosa<TAG>/<set>/
#   3. peak3d gc (deconvolution, compounds across the five levels)     -> .../<set>/gc/
# Submit from the gc worktree (the code that runs is the worktree's):
#   cd .worktrees/gc && sbatch scripts/51_deepcpr_gc.sbatch
#   sbatch --export=ALL,TAG=_int,SCALE=1 scripts/51_deepcpr_gc.sbatch      # plain integer masses
#   sbatch --export=ALL,TAG=_dsk,DESKEW_AA=auto,DESKEW_FA=auto scripts/51_deepcpr_gc.sbatch   # scan skew removed
#SBATCH --job-name=deepcpr-gc
#SBATCH --account=metabolomicsgrp        # group pool
#SBATCH --partition=high                 # not preemptible
#SBATCH --time=01:00:00                  # 10 small files; minutes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=zen4                # the benchmark's CPU generation (timings comparable)
#SBATCH --output=logs/deepcpr-gc-%j.out  # %j: one writer per log on Quobyte
#SBATCH --error=logs/deepcpr-gc-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || { echo "NO_MOUNT $(hostname)"; exit 9; }
MAIN=${MAIN:-/quobyte/metabolomicsgrp/fanzhou/hoffmann}
PY=${PEAK3D_ENV:-$MAIN/envs/peak3d}/bin/python
RAW=$MAIN/data/raw/DEEPCPR_GC/data
TAG=${TAG:-}
SCALE=${SCALE:-1.0033548378}
DESKEW_AA=${DESKEW_AA:-0}
DESKEW_FA=${DESKEW_FA:-0}
FLOOR=${FLOOR:-auto}                     # points below 3 x rms of the file's negative intensities are dropped
EXTRA_PROCESS=${EXTRA_PROCESS:-}
CORES=${SLURM_CPUS_PER_TASK:-4}
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/.numba" MPLCONFIGDIR="${TMPDIR:-/tmp}/.mpl"

for set in ${SETS:-aa fa}; do
  if [ "$set" = aa ]; then src="$RAW/amino acids"; skew=$DESKEW_AA; else src="$RAW/fatty acids"; skew=$DESKEW_FA; fi
  MZML=$MAIN/data/mzml/DEEPCPR_GC/$set$TAG
  OUT=$MAIN/results/gc_deepcpr/mosa$TAG/$set
  mkdir -p "$MZML" "$(dirname "$OUT")"
  if [ "$skew" = auto ]; then     # one value per set (one instrument method): the median of the files' own
    skew=$(for f in "$src"/*.CDF; do "$PY" -I bench/gc/cdf_to_mzml.py "$f" --measure-skew --noise-floor "$FLOOR"; done | tee /dev/stderr \
           | sort -g | awk '{v[NR]=$1} END {print v[int((NR+1)/2)]}')
  fi
  echo "== $set: CDF -> mzML (mass scale $SCALE, deskew $skew) on $(hostname)"
  for f in "$src"/*.CDF; do
    stem=$(basename "$f" .CDF)
    "$PY" -I bench/gc/cdf_to_mzml.py "$f" "$MZML/$stem.mzML" --mass-scale "$SCALE" --deskew "$skew" --noise-floor "$FLOOR"
  done
  echo "== $set: peak3d process --mode gc -> $OUT"
  rm -rf "$OUT"
  # shellcheck disable=SC2086
  /usr/bin/time -v "$PY" -m peak3d process --input "$MZML" --output "$OUT" --cores "$CORES" --keep-rejected --mode gc $EXTRA_PROCESS
  echo "== $set: peak3d gc"
  /usr/bin/time -v "$PY" -m peak3d gc "$OUT" --cores "$CORES"
  touch "$OUT/DONE"
done
