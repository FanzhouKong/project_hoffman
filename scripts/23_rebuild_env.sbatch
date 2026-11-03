#!/bin/bash
# Rebuild one Python tool env (asari | masscube | peak3d) at its original path with the exact versions
# of the first build: conda layer from envs/conda.explicit.txt (URLs + md5, identical for the three
# envs), pip layer from envs/<name>.freeze.txt. Used 2026-10-19 after files in the old envs became
# unreadable on Quobyte; move the old env aside first (mv envs/X envs/X.broken-DATE).
#   sbatch scripts/23_rebuild_env.sbatch peak3d
# Fresh package caches: the old envs/.pkgs shares hard-linked (and unreadable) files with the old envs.
#SBATCH --job-name=peakbench-rebuild-env
#SBATCH --account=metabolomicsgrp
#SBATCH --partition=high
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/envs-rebuild-%x-%j.out
#SBATCH --error=logs/envs-rebuild-%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
name=$1
env=envs/$name
CONDA=/cvmfs/hpc.ucdavis.edu/sw/conda/root/bin/conda
[ -e "$env" ] && { echo "$env exists; move it aside first" >&2; exit 1; }

export CONDA_PKGS_DIRS="$SLURM_SUBMIT_DIR/envs/.pkgs_rebuild_$name"   # one cache per job: one writer per file
export PIP_NO_CACHE_DIR=1
"$CONDA" create -y -p "$env" --file envs/conda.explicit.txt
# exact pip layer; the conda-installed 'packaging' (a file:// line) is already there
grep -v ' @ file://' "envs/$name.freeze.txt" > "$TMPDIR/req.txt"
"$env/bin/pip" install -r "$TMPDIR/req.txt"

if [ "$name" = masscube ]; then
  # same local fix as the first build: no crash when the nearest MS1 scan is empty
  f="$env/lib/python3.11/site-packages/masscube/feature_grouping.py"
  cmp "$f" envs/masscube_feature_grouping.py.orig
  patch "$f" < envs/masscube-1.2.22-empty-scan.patch
fi

# versions must match the first build exactly
"$env/bin/pip" freeze > "$TMPDIR/freeze.txt"
diff "envs/$name.freeze.txt" "$TMPDIR/freeze.txt" && echo "pip freeze identical to envs/$name.freeze.txt"
"$CONDA" list -p "$env" --explicit --md5 | grep '^https' > "$TMPDIR/conda.txt"
diff <(grep '^https' envs/conda.explicit.txt) "$TMPDIR/conda.txt" && echo "conda layer identical"

# every file must read back (the reason for the rebuild)
n_bad=$(find "$env" -type f -print0 | xargs -0 -n 64 sh -c 'for f; do timeout 10 cat "$f" > /dev/null || echo "UNREADABLE $f"; done' _ | tee /dev/stderr | wc -l)
echo "unreadable files: $n_bad"; [ "$n_bad" -eq 0 ]

case "$name" in
  asari) "$env/bin/asari" --help | head -3 ;;
  masscube) "$env/bin/python" -c "import masscube.workflows, numba; print('masscube ok')" ;;
  peak3d) env PYTHONPATH="$SLURM_SUBMIT_DIR" "$env/bin/python" -c "import peak3d.pipeline, numba; print('peak3d ok')" ;;
esac
echo "rebuilt $env"
