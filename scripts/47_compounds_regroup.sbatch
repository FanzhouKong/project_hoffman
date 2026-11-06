#!/bin/bash
# Re-run alignment, grouping, gap filling and compound grouping from an existing compounds run (features + cache),
# then the spike-truth check. Picks nothing, so a rule change is tested in minutes.
#   sbatch --array=1-3 scripts/47_compounds_regroup.sbatch           # 1 ST004500, 2 ST004501, 3 ST004502
#   DATE=2026-10-31 selects the run directory results/compounds_<DATE>/
#   VARIANT=noveto EXTRA="--compound-veto off" regroups into results/compounds_<DATE>_<VARIANT>/<study>/ (features
#   copied, cache shared read-only), leaving the base run untouched:
#   sbatch --export=ALL,DATE=2026-10-31,VARIANT=noveto,EXTRA="--compound-veto off" --array=1-3 scripts/47_compounds_regroup.sbatch
#   STEMS_FILE=path (stems, one per line) regroups only those files (a small-study test); EVAL_EXTRA passes
#   options to the truth check; CODE=path runs the peak3d source tree there (e.g. a worktree of an earlier release),
#   EVAL_CODE=path takes the truth check from there
#SBATCH --job-name=cmp-regroup
#SBATCH --account=metabolomicsgrp
#SBATCH --partition=high
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=logs/compounds/regroup-%A_%a.out
#SBATCH --error=logs/compounds/regroup-%A_%a.err

set -euo pipefail
source /etc/profile.d/modules.sh
cd "$SLURM_SUBMIT_DIR" || { echo "NO_MOUNT $(hostname)"; exit 9; }
ROOT=$PWD
PEAK3D_ENV=${PEAK3D_ENV:-$ROOT/envs/peak3d}
CODE=${CODE:-$ROOT}
# compound grouping is opt-in since 1.1.0 (--compounds); earlier trees group by default and lack the flag
# python -m puts the working directory first on sys.path: run peak3d from inside $CODE, outputs on absolute paths
CFLAG=$(cd "$CODE" && "$PEAK3D_ENV/bin/python" -m peak3d align --help | grep -q -- "--compounds" && echo --compounds || true)
DATE=${DATE:-$(date +%F)}
studies=(ST004500 ST004501 ST004502)
study=${studies[$((SLURM_ARRAY_TASK_ID - 1))]}
base="results/compounds_$DATE/$study"
[ -d "$base/peak3d_out/.cache" ] || { echo "no cache in $base/peak3d_out"; exit 2; }
out=$base
cache="$base/peak3d_out/.cache"
if [ -n "${VARIANT:-}" ]; then
    out="results/compounds_${DATE}_$VARIANT/$study"
    mkdir -p "$out/peak3d_out"
    if [ ! -d "$out/peak3d_out/features" ]; then
        if [ -n "${STEMS_FILE:-}" ]; then   # a file subset: STEMS_FILE lists stems one per line (small-study tests)
            mkdir -p "$out/peak3d_out/features"
            while read -r st; do [ -n "$st" ] && cp "$base/peak3d_out/features/$st.tsv" "$base/peak3d_out/features/$st.params.json" "$out/peak3d_out/features/"; done < "$STEMS_FILE"
            echo "subset: $(ls "$out/peak3d_out/features"/*.tsv | wc -l) feature tables"
        else
            cp -r "$base/peak3d_out/features" "$out/peak3d_out/features"
        fi
    fi
    [ -f "$out/peak3d_out/params.json" ] || cp "$base/peak3d_out/params.json" "$out/peak3d_out/params.json"
fi
export NUMBA_CACHE_DIR="$TMPDIR/numba_$study" MPLCONFIGDIR="$TMPDIR/mpl_$study"
echo "$study regroup on $(hostname), $SLURM_CPUS_PER_TASK CPUs"
echo "code $CODE ($(git -C "$CODE" rev-parse --short HEAD)$(git -C "$CODE" diff --quiet || echo ", uncommitted changes")) $CFLAG"
(cd "$CODE" && "$PEAK3D_ENV/bin/python" -c "import peak3d; print('peak3d', peak3d.__version__, 'from', peak3d.__file__)" &&
 "$PEAK3D_ENV/bin/python" -m peak3d align --output "$ROOT/$out/peak3d_out" --cache-dir "$ROOT/$cache" \
    --cores "$SLURM_CPUS_PER_TASK" $CFLAG ${EXTRA:-}) 2>&1 | tee "$out/regroup.log"
env PYTHONPATH="$ROOT" "$PEAK3D_ENV/bin/python" "${EVAL_CODE:-$ROOT}/bench/diag/compound_truth.py" --out "$out/peak3d_out" --study "$study" \
    --report "$out/compound_truth" ${EVAL_EXTRA:-} 2>&1 | tee "$out/compound_truth.log"
echo REGROUP_DONE
