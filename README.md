# peak3d: LC-MS feature detection, and a peak-picking benchmark

This repository holds two things:

- **`peak3d/`**, a feature detector for untargeted LC-MS. It finds peaks directly on the 3D
  (RT, m/z, intensity) cloud of MS1 centroids, corrects retention time across runs without internal
  standards, groups the runs into one feature table and gap-fills it. The method is described in
  [`peak3d/README.md`](peak3d/README.md).
- **`bench/` + `scripts/`**, the benchmark ("peakbench") that compares peak3d with asari, MassCube and
  IDSL.IPA on public datasets with ground truth. See [Benchmark](#benchmark-peakbench) below.

## Quick start: peak3d on your own mzML files

### 1. Install

```bash
git clone https://github.com/FanzhouKong/project_hoffman.git
cd project_hoffman

# conda (recommended)
conda env create -f environment.yml
conda activate peak3d
pip install -e .

# or plain pip, Python >= 3.10
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

This installs the `peak3d` command. Check the installation with the built-in demo, which writes three
small synthetic runs (60 compounds with RT drift between runs) and processes them:

```bash
peak3d --version
peak3d demo --output peak3d_demo        # -> peak3d_demo/peak3d_out/aligned_feature_table.tsv
```

### 2. Prepare your files

peak3d reads **centroided MS1 mzML** (`.mzML` or `.mzml`). Put all runs of one study (samples, QCs,
blanks) in one folder: they are processed together, aligned to each other and reported in one table.

Convert vendor files first, with centroiding on:

```bash
# ProteoWizard msconvert (all vendors)
msconvert data/raw/*.raw --mzML --filter "peakPicking vendor msLevel=1" -o data/mzml
# ThermoRawFileParser (Thermo .raw; centroids by default)
ThermoRawFileParser -d data/raw -o data/mzml -f 2
```

- Profile-mode files are refused with a message saying how to convert them.
- MS2 scans are ignored; DDA files work as they are.
- Polarity-switching files: run once per polarity with `--polarity pos` and `--polarity neg`.

### 3. Run

```bash
peak3d process --input data/mzml --output peak3d_out --cores 8
```

`--cores` sets how many files are processed at once. Each worker holds one file in memory, so lower it
if you run out of memory. The first run pauses briefly while numba compiles the kernels; later runs
reuse the compiled code.

### 4. Results (`peak3d_out/`)

| file | content |
|---|---|
| `aligned_feature_table.tsv` | one row per feature: `mz`, `rt` (min), RT range, detection counts, isotope label (`iso_offset`, 0 = monoisotopic), `charge`, and one **height** column per file |
| `aligned_feature_area.tsv` | the same table with peak areas |
| `filled_mask.tsv` | 1 where a value was gap-filled rather than detected |
| `features/<file>.tsv` | every feature of each file, with shape, S/N and quality descriptors |
| `rt_correction/` | per-file RT warps and a summary of the correction |
| `qc/` | RT drift curves, residual histograms, presence histogram |
| `params.json` | every estimated parameter and the software versions |

The full list of columns is in [`peak3d/README.md`](peak3d/README.md#outputs-out).

### Options

All parameters (peak width, noise, m/z error) are estimated from each file, so the defaults are meant
to be used as they are. The options you are most likely to need:

| option | default | meaning |
|---|---|---|
| `--cores N` | 1 | files processed in parallel |
| `--polarity pos\|neg` | none | keep one polarity (required for polarity-switching files) |
| `--min-presence N` | 2 | keep a feature only if it is found in at least N files (1 = keep all) |
| `--quant height\|area` | height | value in `aligned_feature_table.tsv` (the other is written too) |
| `--ppm X` | 5 | m/z tolerance for aligning runs |
| `--no-rt-correction` | off | group runs on their native RT |
| `--no-gap-fill` | off | leave undetected cells empty (0) |

Other commands: `peak3d pick FILE.mzML ... --output OUT` (per-file tables only) and
`peak3d align --output OUT` (re-run alignment on an existing output with other options).
`peak3d <command> --help` lists everything.

### On a Slurm cluster

```bash
#!/bin/bash
#SBATCH --job-name=peak3d
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=peak3d-%j.out

eval "$(conda shell.bash hook)" && conda activate peak3d   # or: source /path/to/.venv/bin/activate
peak3d process --input data/mzml --output peak3d_out --cores "$SLURM_CPUS_PER_TASK"
```

### Tests

```bash
pip install -e ".[test]"
pytest peak3d/tests -m "not perf"
```

### Troubleshooting

- `no .mzML files in ...`: `--input` must be the folder that directly contains the files.
- `both polarities present`: add `--polarity pos` or `--polarity neg`.
- `MS1 spectra are profile mode`: reconvert with centroiding (step 2).
- numba `cannot cache function` errors when peak3d is installed in a read-only location: set
  `NUMBA_CACHE_DIR` to a writable folder.

<a id="benchmark-peakbench"></a>
## Benchmark (peakbench)

Benchmark of LC-MS feature detection for **coverage** (are real peaks found?) and
**false discoveries** (are noise/non-peaks rejected?).

Tools: **asari**, **MassCube**, **IDSL.IPA** and the in-house **peak3d** (3D apex picker + reference-free
RT correction, see `peak3d/README.md`). All run at
**default / recommended settings** (no per-dataset tuning). XCMS and MZmine 3 are planned but not
wired in yet.

The benchmark scripts were written for the UC Davis HPC cluster (Slurm account, partitions and paths in `scripts/*.sbatch`); the raw data, tool environments and results are not in the repository and are rebuilt with the pipeline below.

### Datasets (`bench/datasets.py`)

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

### Benchmark tiers

| tier | data | measures |
|---|---|---|
| T1 injected | real backgrounds (HZV029, BM21, yeast, LI2018_QE) + synthetic EMG peaks with isotopes, S/N ladder, noise axes | recall by S/N, width, crowding; boundary/area error |
| T2 decoy | scan-order-shuffled copies of the same files | false positives per file (no real peaks survive) |
| T3 labeled | IDSL_IPA TP/TN labels, HZV029/NIST certified features | precision/recall on human labels |
| T4 real truth | yeast 12C/13C credentialing, BM21 ratios, LI2018 spike ratios | real vs artifact; ratio fidelity |

### Pipeline

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

### Scoring decisions

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
