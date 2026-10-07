#!/bin/bash
# Rebuild the conversion list from what is downloaded and submit the array.
# Rows already converted (status ok) exit immediately, so this is safe to rerun.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/11_make_convert_list.py
N=$(($(wc -l < manifest/convert_list.tsv) - 1))
sbatch --parsable --array=1-"$N"%40 scripts/12_convert.sbatch
