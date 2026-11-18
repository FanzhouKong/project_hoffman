#!/bin/bash
# Score the DeepCPR benchmark runs (scripts/51_deepcpr_gc.sbatch, scripts/52_deepcpr_tool.sbatch):
#   srun -A metabolomicsgrp -p high -t 30 -c 2 --mem=8G scripts/54_deepcpr_score.sh [TOOL=KIND:DIR ...]
# Default tools: every result folder that exists (mosa = as converted, mosa_dsk = scan skew removed, deepcpr).
# Levels: amino acids 5-50 mg/L; fatty acids 0.2x-1.0x of a master mix (zhi50 = 0.2x ... zhi10 = 1.0x; paper, Datasets).
set -euo pipefail
cd "$(dirname "$0")/.."
MAIN=${MAIN:-/quobyte/metabolomicsgrp/fanzhou/hoffmann}
PY=$MAIN/envs/peak3d/bin/python
R=$MAIN/results/gc_deepcpr
LIB=${LIB:-$MAIN/data/libraries/mona_gcms/MoNA-export-GC-MS_Spectra.msp}
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/.numba"
for set in aa fa; do
  if [ "$set" = aa ]; then raw="$MAIN/data/raw/DEEPCPR_GC/data/amino acids"; lv=LMJ-05=5,LMJ-10=10,LMJ-20=20,LMJ-30=30,LMJ-50=50
  else raw="$MAIN/data/raw/DEEPCPR_GC/data/fatty acids"; lv=zhi50-5vs1=0.2,zhi40-5vs1=0.4,zhi30-5vs1=0.6,zhi20-5vs1=0.8,zhi10-5vs1=1.0; fi
  tools=()
  if [ $# -gt 0 ]; then for t in "$@"; do tools+=(--tool "${t//SET/$set}"); done
  else
    for t in mosa mosa_dsk; do [ -e "$R/$t/$set/DONE" ] && tools+=(--tool "$t=mosa:$R/$t/$set"); done
    for t in deepcpr deepcpr_adaptive; do [ -e "$R/$t/$set/DONE" ] && tools+=(--tool "$t=deepcpr:$R/$t/$set"); done
  fi
  seg=()
  [ -e "$R/deepcpr/$set/DONE" ] && seg=(--segments "$R/deepcpr/$set")
  echo "== $set"
  "$PY" -I bench/gc/deepcpr_score.py --raw "$raw" --levels "$lv" --out "$R/score/$set" --library "$LIB" "${seg[@]}" "${tools[@]}"
done
