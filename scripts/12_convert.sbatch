#!/bin/bash
# Convert one vendor file per array task (row N of manifest/convert_list.tsv) to a
# centroided, indexed mzML in data/mzml/<dataset>/<stem>.mzML.
#   sbatch --array=1-958%40 scripts/12_convert.sbatch
# Finished outputs are skipped, so resubmitting only redoes missing/failed rows.
#SBATCH --job-name=peakbench-conv
#SBATCH --account=metabolomicsgrp
#SBATCH --partition=high
#SBATCH --time=02:00:00                  # one file; most take a few minutes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G                        # Wine prefix copy lives in /tmp (counted)
#SBATCH --output=logs/convert/conv-%A_%a.out   # one log per array task
#SBATCH --error=logs/convert/conv-%A_%a.err

set -euo pipefail
source /etc/profile.d/modules.sh
cd "$SLURM_SUBMIT_DIR"

TRFP=containers/trfp_1.4.5.sif
PWIZ=containers/pwiz_26.1.0.266.sif
LIST=manifest/convert_list.tsv

row=$(awk -F'\t' -v t="$SLURM_ARRAY_TASK_ID" '$1==t' "$LIST")
[ -n "$row" ] || { echo "no row $SLURM_ARRAY_TASK_ID"; exit 1; }
IFS=$'\t' read -r task ds kind src member stem <<<"$row"

outdir="data/mzml/$ds"
final="$outdir/$stem.mzML"
mkdir -p "$outdir" manifest/convert_status
status="manifest/convert_status/${ds}__${stem}.tsv"   # one status file per output (one writer)
if [ -s "$final" ] && awk -F"\t" '$4=="ok"{f=1} END{exit !f}' "$status" 2>/dev/null; then
  echo "done already: $final"; exit 0
fi

work="$TMPDIR/conv_$task"; mkdir -p "$work/in" "$work/out"
# Slurm wipes $TMPDIR anyway; cleanup must never turn a finished conversion into a failure
trap 'chmod -R u+w "$work" 2>/dev/null; rm -rf "$work" 2>/dev/null || true' EXIT
echo "[$task] $ds $kind $src $member -> $final on $(hostname)"

case "$kind" in
  thermo_zip)
    unzip -q -j "data/raw/$src" "$member" -d "$work/in"
    in="$work/in/$(basename "$member")" ;;
  thermo_file)
    cp "data/raw/$src" "$work/in/"
    in="$work/in/$(basename "$src")" ;;
  agilent_dzip)
    unzip -q "data/raw/$src" -d "$work/in"
    in=$(find "$work/in" -maxdepth 3 -type d -name '*.d' | head -1)
    [ -n "$in" ] || { echo "no .d folder inside $src"; exit 1; } ;;
esac

case "$kind" in
  thermo_*)
    # -f 2 indexed mzML; vendor peak picking (centroiding) is the default for all MS levels
    apptainer exec --cleanenv -B "$work:/work" "$TRFP" \
      ThermoRawFileParser -i "/work/in/$(basename "$in")" -o /work/out -f 2 -l 1
    produced="$work/out/$(basename "${in%.*}").mzML" ;;
  agilent_dzip)
    # Wine refuses a read-only prefix it doesn't own: copy the image's prefix to local scratch.
    rel=${in#"$work/"}
    apptainer exec --cleanenv -B "$work:/work" "$PWIZ" bash -c '
      set -e
      cp -a /wineprefix64 /work/wineprefix
      export WINEPREFIX=/work/wineprefix WINEDEBUG=-all
      # Z: maps to / under Wine; a Unix -o path gets joined with a backslash
      wine msconvert "Z:/work/'"$rel"'" --mzML --zlib -o "Z:\\work\\out" \
        --filter "peakPicking vendor msLevel=1-"
      wineserver -w   # let Wine finish writing its prefix before we clean up'
    produced=$(find "$work/out" -maxdepth 1 -name '*.mzML' | head -1)
    [ -n "$produced" ] && mv "$produced" "$work/out/$stem.tmp" && produced="$work/out/$stem.tmp" ;;
esac

[ -s "$produced" ] || { echo "converter produced no output"; exit 1; }
check=$(python3 scripts/check_mzml.py "$produced" | tail -1)
echo "$check"
mode=$(cut -f3 <<<"$check")

cp "$produced" "$final.part" && mv "$final.part" "$final"
if [ "$mode" = centroid ]; then st=ok; else st="warn_$mode"; fi
printf '%s\t%s\t%s\t%s\n' "$task" "$ds" "$final" "$st	$(cut -f2,4 <<<"$check")" > "$status"
echo "status: $st"
