#!/bin/bash
# GC-MS mode re-benchmark of one code version, so that two versions are compared on the same inputs and scorers:
#   mrh     MRH GC-Orbitrap set (11 levels): process --mode gc --keep-rejected, then mrh_score.py (ion recall on the
#           screened truth), deconv_eval.py (whole compounds, close pairs separated) and mrh_negatives.py score
#           (true-negative precision)                                        -> $R/mrh, mrh_score, mrh_deconv, mrh_neg
#   deepcpr DeepCPR amino-acid and FAME runs (.CDF read directly): process --mode gc, then the paper-style table
#                                                                            -> $R/deepcpr/{aa,fa}, deepcpr_table_*.tsv
#   gctof2  data/GCTOF Data 2 (22 CSF .cdf runs, no truth: junk and duplicate counts only) -> $R/gctof2
# CODE = the folder whose peak3d runs (a `git archive` snapshot for the baseline, the worktree for the new code);
# the scorers always come from the worktree, except deconv_eval.py (it re-runs the deconvolution of CODE).
#   sbatch --export=ALL,CODE=/path/to/snapshot,TAG=base,STEP=mrh scripts/58_gc_rebench.sbatch
#SBATCH --job-name=gc-rebench
#SBATCH --account=metabolomicsgrp        # group pool
#SBATCH --partition=high                 # not preemptible
#SBATCH --time=04:00:00                  # MRH: 11 Orbitrap files; the other two steps take minutes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12               # one program, worker processes
#SBATCH --mem=96G
#SBATCH --constraint=zen4                # the benchmark's CPU generation (timings comparable)
#SBATCH --output=logs/bench/gc-rebench-%j.out   # %j: one writer per log on Quobyte
#SBATCH --error=logs/bench/gc-rebench-%j.err

set -euo pipefail
M=/quobyte/metabolomicsgrp/fanzhou/hoffmann
WT=$M/.worktrees/gc
cd "$WT" || { echo "NO_MOUNT $(hostname)"; exit 9; }
PY=${PEAK3D_ENV:-$M/envs/peak3d}/bin/python
CODE=${CODE:-$WT}
TAG=${TAG:?TAG}
STEP=${STEP:?STEP}
R=$M/results/gc_rebench/$TAG
CORES=${SLURM_CPUS_PER_TASK:-4}
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/.numba" MPLCONFIGDIR="${TMPDIR:-/tmp}/.mpl"
mkdir -p "$R"
mosa() { (cd "$CODE" && /usr/bin/time -v "$PY" -m peak3d "$@"); }
echo "== $STEP with $CODE on $(hostname)"; (cd "$CODE" && git log --oneline -1 2>/dev/null || cat VERSION 2>/dev/null || true)

case $STEP in
mrh)
  rm -rf "$R/mrh"
  mosa process --input "$M/data/mzml/MRH_GC_ORBITRAP" --output "$R/mrh" --cores "$CORES" --keep-rejected --mode gc
  "$PY" -I bench/gc/mrh_score.py --output "$R/mrh" --truth "$M/data/raw/MRH_GC_ORBITRAP" --out "$R/mrh_score"
  (cd "$CODE" && "$PY" bench/gc/deconv_eval.py --output "$R/mrh" --score "$R/mrh_score" --cores "$CORES" --out "$R/mrh_deconv") \
    | grep -E "^all|^   \(|^ +\(|rror" || true
  "$PY" bench/gc/mrh_compound_rows.py --output "$R/mrh" --score "$R/mrh_score" --out "$R/mrh_rows"
  if [ -e bench/gc/mrh_negatives.py ]; then         # the true-negative scorer (results/gc_mrh/negatives)
    "$PY" bench/gc/mrh_negatives.py score --negatives "$M/results/gc_mrh/negatives" --features "$R/mrh/features" \
      --truth "$M/data/raw/MRH_GC_ORBITRAP" --out "$R/mrh_neg"
  fi
  ;;
deepcpr)
  for s in aa fa; do
    if [ $s = aa ]; then raw="$M/data/raw/DEEPCPR_GC/data/amino acids"; lv=LMJ-05=5,LMJ-10=10,LMJ-20=20,LMJ-30=30,LMJ-50=50
    else raw="$M/data/raw/DEEPCPR_GC/data/fatty acids"; lv=zhi50-5vs1=0.2,zhi40-5vs1=0.4,zhi30-5vs1=0.6,zhi20-5vs1=0.8,zhi10-5vs1=1.0; fi
    rm -rf "$R/deepcpr/$s"; mkdir -p "$R/deepcpr"
    mosa process --input "$raw" --output "$R/deepcpr/$s" --cores 8 --mode gc \
      --ei-library "$M/data/libraries/mona_gcms/MoNA-export-GC-MS_Spectra.msp"
    "$PY" -I bench/gc/deepcpr_paper_table.py --raw "$raw" --levels "$lv" --deepcpr "$M/results/gc_deepcpr/deepcpr/$s" \
      --mosa "$R/deepcpr/$s" --out "$R/deepcpr_table_$s.tsv" | cut -c1-430
  done
  ;;
gctof2)
  rm -rf "$R/gctof2"
  mosa process --input "$M/data/GCTOF Data 2" --output "$R/gctof2" --cores "$CORES" --mode gc --ppm 5
  ;;
*) echo "unknown STEP $STEP"; exit 2 ;;
esac
touch "$R/DONE_$STEP"
