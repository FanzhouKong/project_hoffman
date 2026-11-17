#!/bin/bash
# MRH GC-Orbitrap set (Zenodo 18596747): the picker as it is, on GC-EI data with an ion-level ground truth.
#   1. profile mzML -> centroid mzML (bench/gc/centroid_profile.py), data/mzml/MRH_GC_ORBITRAP/
#   2. peak3d process in mode MODE (lc = unchanged defaults; gc), rejected candidates kept -> results/gc_mrh/peak3d[_gc]/
#   3. peak3d gc (deconvolution only: the mixes hold no retention-index ladder) -> results/gc_mrh/peak3d/gc/
#   4. bench/gc/mrh_score.py against groundtruth_corrected_<level>.csv -> results/gc_mrh/score/
# Files still downloading (a .aria2 file beside them) are left out; rerun when they are complete.
# Download first: sbatch scripts/02_download.sbatch manifest/aria2_input_mrh.txt
# Safe to resubmit: finished centroid files are kept.
#SBATCH --job-name=mrh-gc
#SBATCH --account=metabolomicsgrp        # group pool
#SBATCH --partition=high                 # not preemptible
#SBATCH --time=06:00:00                  # 11 files of ~0.8 GB profile data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G                        # 4 profile files parsed at a time
#SBATCH --constraint=zen4                # the benchmark's CPU generation (timings comparable)
#SBATCH --output=logs/bench/mrh-gc-%j.out   # %j: one writer per log on Quobyte
#SBATCH --error=logs/bench/mrh-gc-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR" || { echo "NO_MOUNT $(hostname)"; exit 9; }
ROOT=$PWD
PY=${PEAK3D_ENV:-$ROOT/envs/peak3d}/bin/python
RAW=data/raw/MRH_GC_ORBITRAP
MZML=data/mzml/MRH_GC_ORBITRAP
MODE=${MODE:-lc}                       # picker mode; sbatch --export=ALL,MODE=gc scripts/50_mrh_gc.sbatch
SUFFIX=$([ "$MODE" = lc ] || echo "_$MODE")
OUT=${OUT:-results/gc_mrh/peak3d$SUFFIX}
SCORE=${SCORE:-results/gc_mrh/score$SUFFIX}
export NUMBA_CACHE_DIR="$TMPDIR/.numba" MPLCONFIGDIR="$TMPDIR/.mpl"
mkdir -p "$MZML" "$(dirname "$OUT")" logs/bench

echo "== centroiding on $(hostname), $SLURM_CPUS_PER_TASK CPUs"
# -I: the interpreter takes no code from the data folder or the environment; the script adds the repo itself
for f in "$RAW"/NEWSTD_*.mzML; do
  stem=$(basename "$f" .mzML)
  if [ -e "$f.aria2" ]; then echo "still downloading, left out: $f" >&2; rm -f "$MZML/$stem.mzML"; continue; fi
  [ -s "$MZML/$stem.mzML" ] || echo "$f"
done | xargs -r -P 4 -I{} sh -c '"$0" -I bench/gc/centroid_profile.py "$1" "$2/$(basename "$1")"' "$PY" {} "$MZML" \
  | tee -a "$MZML/centroid_summary_${SLURM_JOB_ID}.tsv"
n=$(ls "$MZML"/NEWSTD_*.mzML | wc -l)
[ "$n" -ge 1 ] || { echo "no centroid files"; exit 1; }
echo "$n of 11 injections"

echo "== peak3d process --mode $MODE -> $OUT"
rm -rf "$OUT"
/usr/bin/time -v "$PY" -m peak3d process --input "$MZML" --output "$OUT" --cores "$SLURM_CPUS_PER_TASK" --keep-rejected --mode "$MODE"

echo "== peak3d gc (deconvolution)"
/usr/bin/time -v "$PY" -m peak3d gc "$OUT" --cores "$SLURM_CPUS_PER_TASK"

echo "== score against the ground truth"
"$PY" -I bench/gc/mrh_score.py --output "$OUT" --truth "$RAW" --out "$SCORE"
touch "$OUT/DONE"
