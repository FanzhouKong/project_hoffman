#!/bin/bash
# Download and unpack Metabolomics Workbench PR002464 (Broeckling/Du/Patti benchmark) studies:
# documentation, sample metadata, resources and the centroided mzML archive of each study.
#   sbatch scripts/05_download_pr002464.sbatch                     # the four Orbitrap studies
#   sbatch scripts/05_download_pr002464.sbatch ST004503 ST004626   # others (add their checksums below)
# Result: data/raw/PR002464/<study>/{1-doc,2-sample_metadata,3-data_mzML_centroid,4-resources}
# (ST004502 unpacks its mzML into 3-mzML_centroid)
# Safe to resubmit: aria2c resumes, finished archives are not unpacked twice.
#SBATCH --job-name=pr2464-dl
#SBATCH --account=metabolomicsgrp       # group pool
#SBATCH --partition=high                # not preemptible
#SBATCH --time=12:00:00                 # ~20 GB for the Orbitrap studies
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2               # network-bound download, single-threaded unpack
#SBATCH --mem=8G
#SBATCH --output=logs/pr2464-dl-%j.out  # %j: one writer per log on Quobyte
#SBATCH --error=logs/pr2464-dl-%j.err
set -euo pipefail
source /etc/profile.d/modules.sh
module load aria2/1.37.0
cd "$SLURM_SUBMIT_DIR"

STUDIES=("$@")
[ ${#STUDIES[@]} -gt 0 ] || STUDIES=(ST004500 ST004501 ST004502 ST004504)
DEST=data/raw/PR002464
URL=https://www.metabolomicsworkbench.org/studydownload

# MD5 as listed on each study's download page (2026-10-21)
declare -A MD5=(
  [ST004500_1-doc.7z]=c6aed94ece7178765176b374f002e4e9
  [ST004500_2-sample_metadata.7z]=e52f6d1695250b3dcb00e35603911533
  [ST004500_3-data_mzML_centroid.7z]=bb1a1a7cd49899447b8b59dba1168619
  [ST004500_4-resources.7z]=7174f291944aa69729dbe34f2c417bb1
  [ST004501_1-doc.7z]=dbdf4f7af5bef2196a7113d19de2fb24
  [ST004501_2-sample_metadata.7z]=c7ffc19fe33172d010a07b6babcaf52f
  [ST004501_3-data_mzML_centroid.7z]=2979b85579a27514578e0c982bd46656
  [ST004501_4-resources.7z]=b25d72286323ec75e132f0c8b2e95499
  [ST004502_1-doc.7z]=e18df61241e999800066a4eeaa7689af
  [ST004502_2-sample_metadata.7z]=79b0452b18c765983cbf389d96c0ed6f
  [ST004502_3-data_mzML_centroid.7z]=df80a2d96fdcf27cd37463e01d581fb5
  [ST004502_4-resources.7z]=93adb68866723fbe24ba27131463af24
  [ST004504_1-doc.7z]=8fef34e3310644bd9b59e49f4695f5a0
  [ST004504_2-sample_metadata.7z]=015007827374d3543d189bb3c51f5864
  [ST004504_3-data_mzML_centroid.7z]=2e7c3c0625b092b757fbdbfae17783e0
  [ST004504_4-resources.7z]=7c04fc1b80f4ad1833b60831607834f3
)

INPUT=logs/pr2464-aria2-input-${SLURM_JOB_ID}.txt
: > "$INPUT"
for st in "${STUDIES[@]}"; do
  for part in 1-doc 2-sample_metadata 3-data_mzML_centroid 4-resources; do
    f=${st}_${part}.7z
    [ -n "${MD5[$f]:-}" ] || { echo "no checksum for $f" >&2; exit 1; }
    printf '%s/%s\n  dir=%s/%s/archives\n  out=%s\n  checksum=md5=%s\n' \
      "$URL" "$f" "$DEST" "$st" "$f" "${MD5[$f]}" >> "$INPUT"
  done
done

# same politeness settings as 02_download.sbatch
aria2c --input-file="$INPUT" --continue=true \
  --max-concurrent-downloads=2 --max-connection-per-server=1 --split=1 \
  --max-tries=10 --retry-wait=30 --timeout=120 --file-allocation=none \
  --auto-file-renaming=false --allow-overwrite=false --check-integrity=true \
  --summary-interval=600 --console-log-level=warn \
  --log="logs/pr2464-aria2-${SLURM_JOB_ID}.log" --log-level=notice

# no 7z on Hive: py7zr in node-local scratch (also openpyxl for the design export)
PYLIB="${TMPDIR:-/tmp}/pr2464-pylib"
python3 -m pip install -q --target "$PYLIB" py7zr openpyxl
for st in "${STUDIES[@]}"; do
  for a in "$DEST/$st"/archives/*.7z; do
    done_flag="$a.unpacked"
    [ -e "$done_flag" ] && continue
    echo "unpacking $a"
    PYTHONPATH="$PYLIB" python3 -c "import py7zr, sys; py7zr.SevenZipFile(sys.argv[1]).extractall(sys.argv[2])" \
      "$a" "$DEST/$st"
    touch "$done_flag"
  done
done
PYTHONPATH="$PYLIB" envs/peak3d/bin/python bench/pr2464_design.py "${STUDIES[@]}"
du -sh "$DEST"/*
