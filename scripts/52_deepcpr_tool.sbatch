#!/bin/bash
# DeepCPR (Yu et al., Anal. Chem. 2026; github.com/YuChuanxiu/DeepCPR, release v1.2.0, BSD-3) on its own example runs,
# as the comparison for peak3d's GC deconvolution (scripts/51_deepcpr_gc.sbatch).
#   1. env  $MAIN/envs/deepcpr        Python 3.10 + the release's requirements-onnx.txt (CPU, no TensorFlow; the
#                                     gradio GUI package is left out)
#   2. tool $MAIN/envs/deepcpr_tool   source v1.2.0 + DeepCS.onnx / DeepCPR.onnx (downloaded on the node, copied once)
#   3. run  data_resolution() with the release defaults on the two sets -> $MAIN/results/gc_deepcpr/deepcpr<TAG>/<set>/
#           (single/ = components per file, seg/ = segments with R2, ms/ = spectra); ADAPTIVE=1 adds adaptive=True
# Steps 1-2 are skipped when already there.   cd .worktrees/gc && sbatch scripts/52_deepcpr_tool.sbatch
#SBATCH --job-name=deepcpr-tool
#SBATCH --account=metabolomicsgrp        # group pool
#SBATCH --partition=high                 # not preemptible
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=zen4                # same CPU generation as the peak3d runs (timings comparable)
#SBATCH --output=logs/deepcpr-tool-%j.out
#SBATCH --error=logs/deepcpr-tool-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || { echo "NO_MOUNT $(hostname)"; exit 9; }
MAIN=${MAIN:-/quobyte/metabolomicsgrp/fanzhou/hoffmann}
ENV=$MAIN/envs/deepcpr
TOOL=$MAIN/envs/deepcpr_tool
RAW=$MAIN/data/raw/DEEPCPR_GC/data
TAG=${TAG:-}
ADAPTIVE=${ADAPTIVE:-0}
REL=https://github.com/YuChuanxiu/DeepCPR
export CONDA_PKGS_DIRS="$MAIN/envs/.pkgs" PIP_CACHE_DIR="$MAIN/envs/.pip"     # off the 20 GB home quota
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/.numba" MPLCONFIGDIR="${TMPDIR:-/tmp}/.mpl" MPLBACKEND=Agg

if [ ! -s "$TOOL/DeepCPR.onnx" ] || [ ! -s "$TOOL/DeepCS.onnx" ] || [ ! -d "$TOOL/DeepCPR-1.2.0" ]; then
  echo "== download source and models"
  W=$(mktemp -d "${TMPDIR:-/tmp}/deepcpr.XXXXXX")
  curl -sSL --fail -o "$W/src.tar.gz" "$REL/archive/refs/tags/v1.2.0.tar.gz"
  curl -sSL --fail -o "$W/DeepCPR.onnx" "$REL/releases/download/v1.2.0/DeepCPR.onnx"
  curl -sSL --fail -o "$W/DeepCS.onnx" "$REL/releases/download/v1.2.0/DeepCS.onnx"
  tar -xzf "$W/src.tar.gz" -C "$W"
  mkdir -p "$TOOL"
  cp -r "$W/DeepCPR-1.2.0" "$W/DeepCPR.onnx" "$W/DeepCS.onnx" "$TOOL/"
  md5sum "$W"/*.onnx "$W/src.tar.gz" | tee "$TOOL/md5.txt"
fi

if [ ! -x "$ENV/bin/python" ]; then
  echo "== build env"
  source /etc/profile.d/modules.sh
  module load conda/latest
  conda create -y -p "$ENV" python=3.10 pip
  grep -v -i '^gradio' "$TOOL/DeepCPR-1.2.0/requirements-onnx.txt" > "$TOOL/requirements-run.txt"
  "$ENV/bin/pip" install -r "$TOOL/requirements-run.txt"
  "$ENV/bin/pip" freeze > "$ENV.freeze.txt"
fi

for set in ${SETS:-aa fa}; do
  if [ "$set" = aa ]; then src="$RAW/amino acids"; else src="$RAW/fatty acids"; fi
  OUT=$MAIN/results/gc_deepcpr/deepcpr$TAG/$set
  rm -rf "$OUT"; mkdir -p "$OUT"
  echo "== $set: DeepCPR (adaptive=$ADAPTIVE) on $(hostname) -> $OUT"
  # the tool's own package directory first on the path; nothing is imported from the data folder
  ( cd "$TOOL/DeepCPR-1.2.0" && /usr/bin/time -v "$ENV/bin/python" - "$src" "$TOOL" "$OUT" "$ADAPTIVE" <<'PY'
import sys, time, json, os
src, tool, out, adaptive = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
from DeepCPR.DeepCPR import data_resolution
tic = time.perf_counter()
data_resolution(src, os.path.join(tool, "DeepCS.onnx"), os.path.join(tool, "DeepCPR.onnx"), out, False, adaptive=adaptive)
json.dump(dict(seconds=round(time.perf_counter() - tic, 2), adaptive=adaptive, files=sorted(os.listdir(src))),
          open(os.path.join(out, "run.json"), "w"))
print("seconds", round(time.perf_counter() - tic, 2))
PY
  )
  touch "$OUT/DONE"
done
