# Peak-picking benchmark (peakbench)

Benchmark of LC-MS feature detection for **coverage** (are real peaks found?) and
**false discoveries** (are noise/non-peaks rejected?).

Tools: **asari**, **MassCube**, **IDSL.IPA** and the in-house **peak3d** (3D apex picker + reference-free
RT correction, see `peak3d/README.md`). All run at
**default / recommended settings** (no per-dataset tuning). XCMS and MZmine 3 are planned but not
wired in yet.

## Datasets (`bench/datasets.py`)

| id | accession | instrument | ground truth |
|---|---|---|---|
| BM21 | MWB ST002454 | Orbitrap, HILIC+/RP | plasma x juice mixing ratios |
| HZV029 | MWB ST002233 | Orbitrap ID-X, HILIC+ | repeat injections + 402 certified features |
| ASARI_DATA | github shuzhao-li-lab/data | various | MT02, SZ22, NIST SRM1950 verified features |
| NETID_YEAST | MassIVE MSV000087434 | Q Exactive Plus | 12C/13C/15N yeast credentialing, solvent blanks |
| SLAW_ECOLI | MassIVE MSV000086486 | Q Exactive HF | ~2000 E. coli runs (scale) |
| IDSL_IPA | Zenodo 6302236 | LC/HRMS | 20,000 labeled TP/TN m/z-RT pairs; scored on the 17,004 confirmed ones (see below) |
| MTBLS1684 | MetaboLights | Agilent Q-TOF | full study behind IDSL_IPA |
| LI2018_QE | MetaboLights MTBLS733 | Q Exactive HF | ~1100 standards at defined ratios |
| LI2018_TTOF | MetaboLights MTBLS736 | TripleTOF 6600 | same design; **.wiff.scan files appear missing** |
| MASSCUBE_SI | Zenodo 14159704 | Orbitrap + QTOF | secondary check only (low priority) |

## Benchmark tiers

| tier | data | measures |
|---|---|---|
| T1 injected | real backgrounds (HZV029, BM21, yeast, LI2018_QE) + synthetic EMG peaks with isotopes, S/N ladder, noise axes | recall by S/N, width, crowding; boundary/area error |
| T2 decoy | scan-order-shuffled copies of the same files | false positives per file (no real peaks survive) |
| T3 labeled | IDSL_IPA TP/TN labels, HZV029/NIST certified features | precision/recall on human labels |
| T4 real truth | yeast 12C/13C credentialing, BM21 ratios, LI2018 spike ratios | real vs artifact; ratio fidelity |

## Pipeline

1. `python3 scripts/01_build_manifest.py` - list files via repository APIs -> `manifest/`
2. `sbatch scripts/02_download.sbatch` - aria2c into `data/raw/` (resumable; resubmit to continue)
3. `python3 scripts/03_verify_downloads.py` - size check -> `manifest/download_status.tsv`
4. `sbatch scripts/10_pull_containers.sbatch` once, then `scripts/13_submit_conversions.sh` -
   centroided indexed mzML in `data/mzml/<dataset>/` (Thermo: ThermoRawFileParser 1.4.5, vendor
   centroiding; Agilent: msconvert 26.1 under Wine, `peakPicking vendor msLevel=1-`). Per-file status
   in `manifest/convert_status/`; rerunning skips finished files. `scripts/check_mzml.py` reports
   centroid/profile per MS level.
5. `sbatch scripts/20_build_envs.sbatch`, `21_build_idslipa.sbatch`, `22_build_peak3d_env.sbatch` - tool envs
6. `python3 scripts/30_make_jobs.py && sbatch --array=1-N scripts/31_run_tool.sbatch` - one array task per
   (tool, run) from `bench/runs.py`; results in `results/<tool>/<run>/` (`DONE`, `time.txt`, `tool.log`)
7. `envs/peak3d/bin/python bench/score.py` - match features to truth (ppm x RT grid) -> `results/scores/*.tsv`
8. (next) injection / noise / decoy generator -> `data/bench/` + truth tables

## Scoring decisions

- **Isotope peaks count as features** (decided 2026-10-02). Natural isotope peaks (e.g. 13C M+1) in
  a truth list are real chromatographic peaks that a picker is expected to report, as asari,
  MassCube and peak3d do. Tools that fold isotopes into their parent (IDSL.IPA reports only the
  12C peak of each 12C/13C pair) are scored on the same lists and lose those matches.
  Affects HZV029_cert, YEAST_NEG (64/314 truth peaks are isotopes) and the credentialed sets.
- Default settings only; matching at 10 ppm and +/-0.1 min unless stated (full grid in results/scores/).
- **IDSL003 is scored on a cleaned label list** (decided 2026-10-05): `data/truth/IDSL003/labels_clean.csv`,
  1,676 TP + 15,328 TN of the original 20,000. A blind two-pass visual EIC check of every label
  (`results/diag_idsl/visual_check/SUMMARY.md`) found the curated calls conditioned on the four published tools and
  often contradicted by the raw data; every label the check contradicted, questioned or could not settle was removed,
  plus the TN member of 5 TP-TN pairs within the matching tolerance (`bench/diag/idsl_clean_truth.py`). The published
  tool columns are scored on the same subset. Scores on the original 20,000: `results/scores/idsl_original_labels.tsv`.
