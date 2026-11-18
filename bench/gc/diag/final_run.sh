#!/bin/bash
# With the worktree's defaults: peak3d gc on the stored DeepCPR-set picker outputs (mosa_dsk), the paper-style
# tables and the found check, then the MRH truth evaluation.   final_run.sh TAG
set -eu
cd /quobyte/metabolomicsgrp/fanzhou/hoffmann/.worktrees/gc
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/.numba" OMP_NUM_THREADS=1
M=/quobyte/metabolomicsgrp/fanzhou/hoffmann; PY=$M/envs/peak3d/bin/python; R=$M/results/gc_deepcpr; TAG=${1:-final}
for s in aa fa; do
  echo "== peak3d gc $s"; $PY -m peak3d gc $R/mosa_dsk/$s --cores 8 2>&1 | tail -n 3
  if [ $s = aa ]; then raw="$M/data/raw/DEEPCPR_GC/data/amino acids"; lv=LMJ-05=5,LMJ-10=10,LMJ-20=20,LMJ-30=30,LMJ-50=50
  else raw="$M/data/raw/DEEPCPR_GC/data/fatty acids"; lv=zhi50-5vs1=0.2,zhi40-5vs1=0.4,zhi30-5vs1=0.6,zhi20-5vs1=0.8,zhi10-5vs1=1.0; fi
  $PY -I bench/gc/deepcpr_paper_table.py --raw "$raw" --levels $lv --deepcpr $R/deepcpr/$s --mosa $R/mosa_dsk/$s --out $R/score/$s/paper_table_mosa_dsk.tsv | cut -c1-430
  ls $R/mosa_dsk/$s/gc/components/*.tsv | while read f; do echo "$(basename $f) $(($(wc -l < $f) - 1))"; done | tr '\n' ' '; echo
done
$PY bench/gc/diag/deepcpr_found_check.py mosa_dsk 2>&1 | tail -n 40
for src in peak3d:score peak3d_gc:score_gc; do
  o=${src%%:*}; sc=${src##*:}
  echo "== MRH $o $TAG"
  $PY bench/gc/deconv_eval.py --output $M/results/gc_mrh/$o --score $M/results/gc_mrh/$sc --cores 11 --out $M/results/gc_mrh/deconv_eval/${TAG}_$o 2>&1 | grep -E "^all|^   \(|^ +\(|rror"
done
