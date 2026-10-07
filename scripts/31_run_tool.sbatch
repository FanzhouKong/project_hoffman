#!/bin/bash
# Run one tool on one benchmark run (row N of manifest/bench_jobs.tsv), default settings.
#   python3 scripts/30_make_jobs.py && sbatch --array=1-$(($(wc -l < manifest/bench_jobs.tsv)-1)) scripts/31_run_tool.sbatch
# Inputs are staged to node-local scratch so both tools see the same I/O; results are
# copied to results/<tool>/<run>/ (input copies excluded). Same resources for every tool.
#SBATCH --job-name=peakbench-run
#SBATCH --account=metabolomicsgrp
#SBATCH --partition=high
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=zen4                 # one CPU generation, so runtimes are comparable
#SBATCH --output=logs/bench/run-%A_%a.out
#SBATCH --error=logs/bench/run-%A_%a.err

set -euo pipefail
source /etc/profile.d/modules.sh
cd "$SLURM_SUBMIT_DIR"
ROOT=$PWD

row=$(awk -F'\t' -v t="$SLURM_ARRAY_TASK_ID" '$1==t' manifest/bench_jobs.tsv)
IFS=$'\t' read -r task tool run mode nfiles <<<"$row"
out="results/$tool/$run"
if [ -f "$out/DONE" ]; then echo "done already: $out"; exit 0; fi
rm -rf "$out"; mkdir -p "$out"

work="$TMPDIR/bench_${tool}_${run}"; mkdir -p "$work/data"
trap 'rm -rf "$work" 2>/dev/null || true' EXIT
while IFS= read -r f; do cp "$f" "$work/data/"; done < "results/inputs/$run.txt"
echo "[$task] $tool $run ($nfiles files, $mode) on $(hostname), $SLURM_CPUS_PER_TASK CPUs"

# per-job caches: concurrent jobs writing numba/matplotlib caches into the shared env
# on Quobyte corrupt each other (one writer per file)
export NUMBA_CACHE_DIR="$work/.numba" MPLCONFIGDIR="$work/.mpl"

cd "$work"
set +e   # keep going on tool failure so logs are still copied back
case "$tool" in
  asari)
    # defaults; only mode and core count are given
    /usr/bin/time -v -o "$work/time.txt" \
      "$ROOT/envs/asari/bin/asari" process --mode "$mode" --input data --output asari_out \
      --multicores "$SLURM_CPUS_PER_TASK" > "$work/tool.log" 2>&1 ;;
  masscube)
    # defaults; instrument type and ion mode are auto-detected from the files
    /usr/bin/time -v -o "$work/time.txt" \
      "$ROOT/envs/masscube/bin/python" "$ROOT/scripts/run_masscube.py" "$work" \
      > "$work/tool.log" 2>&1 ;;
  idslipa)
    # template spreadsheet; only paths, threads and RT reference samples are filled in
    module load R/4.4.2
    export R_LIBS_USER="$ROOT/envs/Rlib"
    /usr/bin/time -v -o "$work/time.txt" \
      Rscript "$ROOT/scripts/run_idslipa.R" "$ROOT/envs/idslipa/IPA_parameters_template.xlsx" \
      data idslipa_out "$SLURM_CPUS_PER_TASK" > "$work/tool.log" 2>&1 ;;
  peak3d)
    # in-house 3D picker + RT correction; defaults only, cores from the allocation
    /usr/bin/time -v -o "$work/time.txt" \
      env PYTHONPATH="$ROOT" "$ROOT/envs/peak3d/bin/python" -m peak3d process --input data --output peak3d_out \
      --cores "$SLURM_CPUS_PER_TASK" --cache-dir "$work/.cache" > "$work/tool.log" 2>&1 ;;
  *)
    # tools that are not published: scripts/local/run_<tool>.sh (git-ignored)
    if [ -f "$ROOT/scripts/local/run_$tool.sh" ]; then
      source "$ROOT/scripts/local/run_$tool.sh"
    else
      echo "unknown tool: $tool" >&2; false
    fi ;;
esac
rc=$?
# IDSL.IPA exits 0 even when it rejects its parameter sheet,
# and prints "successfully" even when every file failed, so also look for its error lines
if [ "$tool" = idslipa ]; then
  grep -q "Completed IDSL.IPA computations successfully" "$work/tool.log" || rc=1
  grep -q -E "Problem with|Stopped IDSL.IPA workflow|ERROR" "$work/tool.log" && rc=1
fi
set -e
cd "$ROOT"

# copy everything except the staged inputs
(cd "$work" && find . -mindepth 1 -maxdepth 1 ! -name data ! -name .numba ! -name .mpl ! -name .cache -exec cp -r {} "$ROOT/$out/" \;)
echo "exit $rc"; [ $rc -eq 0 ] && date > "$out/DONE"
exit $rc
