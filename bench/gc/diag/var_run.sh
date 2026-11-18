#!/bin/bash
# DeepCPR-set variant + MRH truth evaluation for deconvolution settings: var_run.sh [--mrh-only] tag:name=value,name=value ...
set -u
cd /quobyte/metabolomicsgrp/fanzhou/hoffmann/.worktrees/gc
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/.numba OMP_NUM_THREADS=1
M=/quobyte/metabolomicsgrp/fanzhou/hoffmann; PY=$M/envs/peak3d/bin/python; R=$M/results/gc_mrh
MRH_ONLY=0; [ "${1:-}" = "--mrh-only" ] && { MRH_ONLY=1; shift; }
for spec in "$@"; do
  tag=${spec%%:*}; kv=${spec#*:}; [ "$kv" = "$spec" ] && kv=""
  args=$(echo "$kv" | tr ',' ' ')
  sets=""; for a in $args; do sets="$sets --set $a"; done
  echo "################ $tag  [$args]"
  [ $MRH_ONLY = 0 ] && $PY bench/gc/diag/deepcpr_variant.py $tag $args 2>&1 | cut -c1-420 | tail -n 30
  for src in peak3d:score peak3d_gc:score_gc; do
    o=${src%%:*}; sc=${src##*:}
    echo "== MRH $o $tag"
    $PY bench/gc/deconv_eval.py --output $R/$o --score $R/$sc --cores 11 $sets --out $R/deconv_eval/${tag}_$o 2>&1 | grep -E "^all|^   \(1|^ +\(0.5|Error|error" 
  done
done
