# peak3d: LC-MS feature detection, and a peak-picking benchmark

This repository holds two things:

- **`peak3d/`**, a feature detector for untargeted LC-MS. It finds peaks directly on the 3D
  (RT, m/z, intensity) cloud of MS1 centroids, corrects retention time across runs without internal
  standards, groups the runs into one feature table and gap-fills it. The method is described in
  [`peak3d/README.md`](peak3d/README.md).
- **`bench/` + `scripts/`**, the benchmark ("peakbench") that compares peak3d with asari, MassCube and
  IDSL.IPA on public datasets with ground truth. See [Benchmark](#benchmark-peakbench) below.

## Run the Mosa app (download, double-click)

Mosa is the app for `peak3d`: the same program, with a window and a browser interface.

Download the file for your computer from the current release,
[Mosa v1.0.0](https://github.com/FanzhouKong/project_hoffman/releases/tag/v1.0.0) (all releases:
[releases page](https://github.com/FanzhouKong/project_hoffman/releases); the
[app-preview](https://github.com/FanzhouKong/project_hoffman/releases/tag/app-preview) pre-release is rebuilt from
every push and may be ahead of, or less tested than, the versioned one). Nothing else to install.

| Computer | File | Start |
|---|---|---|
| Windows 10/11, 64-bit | `peak3d-windows-x64.zip` | right-click > **Extract All**, open the `peak3d` folder, double-click `peak3d.exe` |
| Mac with Apple silicon (M1 or later), macOS 14+ | `peak3d-macos-arm64.zip` | unzip, move `peak3d.app` to Applications, double-click it |
| Mac with an Intel processor, macOS 14+ | `peak3d-macos-x64.zip` | the same |
| Linux, x86-64 (glibc 2.35+, e.g. Ubuntu 22.04) | `peak3d-linux-x64.tar.gz` | `tar xzf peak3d-linux-x64.tar.gz` then `./peak3d/peak3d` |

Mosa opens in your web browser, with a small window to reopen or quit it. Three feature finders are on offer, and
whichever you pick, the rest (chromatograms, MS2, library search, annotation, report) is the same:

- **Mosa** (the default).
- **asari** 1.18.5, included in the app on macOS and Linux (not yet on Windows, where it does not run inside the app).
- **MassCube** 1.2.22, not included: its licence (CC BY-NC 4.0) allows non-commercial use only. The MassCube tab has
  a **Download and install MassCube** button: after you accept the licence, Mosa installs it with a private Python
  (about 950 MB, a few minutes) and can remove it again. From the command line:
  `peak3d addon install masscube --accept-licence`. Choose a folder of mzML files, or
press **Create example data** to try it with three small synthetic runs, then **Run Mosa** and **Open
results**. The first run compiles Mosa (about a minute); later runs start at once. Your data stay on your
computer.

The app is not yet signed with an Apple or Microsoft certificate, so the first start asks you to confirm:

- **macOS**: double-click peak3d; when macOS says it cannot verify it, click **Done**, open **System Settings >
  Privacy & Security**, scroll down and click **Open Anyway** next to peak3d. Or, in Terminal:
  `xattr -dr com.apple.quarantine /Applications/peak3d.app`
- **Windows**: on "Windows protected your PC", click **More info**, then **Run anyway**. Extract the ZIP first:
  started from inside the ZIP, `peak3d.exe` cannot find its `_internal` folder.

If the app does not start, its log says why: `%LOCALAPPDATA%\peak3d\cache\peak3d.log` (Windows),
`~/Library/Caches/peak3d/peak3d.log` (macOS).

The program is also the full command line, e.g. `peak3d.exe process --input mzml --output out --cores 8` on
Windows, `/Applications/peak3d.app/Contents/MacOS/peak3d process ...` on macOS, `./peak3d/peak3d process ...` on
Linux.

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
pip install -e .            # or pip install -e ".[asari]" to run asari as well (peak3d external --kernel asari)
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
- DDA files work as they are: MS1 is used for the features, the MS2 spectra are mapped to them (step 5).
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
| `aligned_feature_table.tsv` | one row per feature: `mz`, `rt` (min), RT range, detection counts, isotope label (`iso_offset`, 0 = monoisotopic), `charge`, its compound group (`compound_id`, `role`, `neutral_mass`, `is_base`, `n_members`; see below),, the halogen check (`possible_halogen` yes / maybe / no, `halogen_pattern` the best hypothesis, e.g. Cl2; see the GUI section) and one **height** column per file |
| `compounds.tsv` | one row per compound group: base ion, members (group ids, m/z, roles), neutral mass |
| `compound_pairs.tsv` | every co-eluting pair that was tested: shape correlation, intensity correlation, mass relation, joined or not |
| `aligned_feature_area.tsv` | the same table with peak areas |
| `filled_mask.tsv` | 1 where a value was gap-filled rather than detected |
| `features/<file>.tsv` | every feature of each file, with shape, S/N and quality descriptors |
| `rt_correction/` | per-file RT warps and a summary of the correction |
| `qc/` | RT drift curves, residual histograms, presence histogram |
| `params.json` | every estimated parameter and the software versions |
| `report.html` | interactive report (feature map, feature table with chromatograms, RT alignment, run details); one self-contained file that opens in any browser, offline, and can be emailed |
| `ms2_map.tsv` | every MS2 spectrum mapped to a feature (file, scan, time, precursor intensity and purity, whether it went into the consensus) |
| `ms2_consensus.msp`, `ms2_spectra.msp` | two MS2 spectra per feature: the consensus of all its spectra, and the single highest-TIC spectrum |
| `identifications.tsv`, `feature_annotations.tsv` | with `--library`: every library hit above 0.1 on the two spectra, raw and denoised against that hit's formula, and each feature's annotation (step 5) |
| `identifications_denoise.npz` | which peaks the denoising kept for each (spectrum, candidate formula) a listed hit used |

The full list of columns is in [`peak3d/README.md`](peak3d/README.md#outputs-out).

**Compound groups.** Features of one compound (adducts, in-source fragments, multimers, charge states and
their isotopes) share a `compound_id`. Co-eluting features are joined when their peak shapes correlate
(EIC Pearson r in every file where both are detected, averaged with the weaker ion's height as weight) and,
where the study makes it informative,
their intensities correlate across files; the cutoffs are calibrated on the study itself (unrelated pairs
shifted onto one apex as the null, M/M+1 pairs as positive controls). `role` names the ion when a mass
relation to the group's neutral mass fits (`[M+H]+`, `[M+Na]+`, `[M-H-H2O]-`, `[2M+H]+`, ...), `[M+H]+ M+1`
for isotopes, and otherwise the m/z difference to the base ion (`is_base` = 1, the strongest monoisotopic
member), e.g. `-44.0262`. Nothing is removed: every feature keeps its own row.

### 5. MS2 and library search (DDA files)

Each MS2 spectrum is assigned to the feature whose m/z and RT window hold its precursor and time. Each
feature then gets two spectra:

- **consensus** (used by default): all its spectra, from every file, merged. A peak is kept only when at least
  two spectra hold it, so noise ions do not add up. Intensities are weighted by how abundant the precursor was
  in the MS1 scan before each spectrum.
- **highest TIC**: the single spectrum with the highest total ion current.

With `--library` (an `.msp` file, or a MassCube `.pkl`), both are searched (`--similarity`, default weighted
entropy) as a **denoising search** (`specdenoise`, tier T1, after Kong et al., Nat Methods 2025): every library
spectrum within the precursor tolerance is scored on the spectrum as measured and on the spectrum denoised with
that candidate's own formula and adduct, keeping only the ions whose neutral loss is a plausible sub-formula of
the candidate's precursor ion. Candidates share few formulas (one compound has many library spectra; isomers),
so each spectrum is denoised once per formula, not per candidate. A record without a formula is read from its
SMILES (a small SMILES reader; the result must fit the record's precursor m/z), and a record with neither keeps
its raw score. `--denoise t0` instead denoises each spectrum once from the feature m/z alone (tier T0 top-5:
the precursor's possible formulas ranked by the fragment intensity they explain; written as
`ms2_*_denoised.msp`), `--denoise off` searches the raw spectra only. A library spectrum scoring above 0.1 on
either version is listed in `identifications.tsv` for both (the `spectrum` column: `consensus`, `single`,
`consensus_denoised`, `single_denoised`; `denoise_ion` the precursor ion formula it was denoised with), so
every hit has a raw and a denoised score. A feature's best hit on the denoised consensus
at or above `--min-similarity` (default 0.7) is its annotation in `feature_annotations.tsv`; the app shows both
scores and switches the head-to-tail plot between the selected candidate's denoised spectrum and the raw one. In the app you can confirm
it, pick another candidate or mark the feature unknown (the table's `reviewed` column says whether you did), write
a comment, and give features red, blue or yellow stars to filter by. All of it is saved in `user_annotations.json`,
kept apart from the search results, and used in the report and the exported tables (`reviewed`, `stars`,
`comment` columns). `peak3d identify OUT --library LIB` searches again without re-processing; your notes are kept.

### In the browser: `peak3d gui`

```bash
peak3d gui                      # opens http://127.0.0.1:8765/?token=... in your browser
```

The GUI picks the mzML folder and output folder, sets the options (the same as the command line, with the
equivalent command shown), runs peak3d with live progress, and opens the results in the same viewer as
`report.html`. It runs on your computer: data never leave it, and the browser is only the window. Runs are
ordinary `peak3d process` subprocesses, so results are identical to the command line. `peak3d report OUT`
rebuilds `report.html` for an existing output.

The Features tab shows what each feature is within its compound group: `M+1` a confirmed isotope, `M+1?` a
possible one (m/z and RT fit a 13C step from a stronger feature, but the peak shapes did not confirm it), a named
ion such as `[M+Na]+`, or `Δ -57.0215`, the m/z difference to the group's base ion when no relation names it. The
Feature group column's button lists only that group's features (the adducts, fragments, multimers and isotopes of
one compound, `compound_id` in the tables); the same button sits in the feature's header.

The standalone app above is the same GUI without a Python installation.

**Halogen check.** Every run tests the MS1 isotope pattern of each monoisotopic feature, in the file where it is
strongest, for chlorine and bromine (`--no-halogen-test` skips it): the ratios M+1..M+4 / M, summed over the top
half of the peak, are fitted under elemental hypotheses (no halogen, S, S2, Si4-12 for siloxanes, Cl1-4, Br1-3,
BrCl, Br2Cl, BrCl2; the carbon count from M+1, N, O, P and H then filling the ion's mass), and where the mass
accuracy allows, the spacing of M+2 adds evidence (37Cl and 81Br sit 0.010 and 0.009 Da below 13C2). A halogen is
called when its best hypothesis beats the best halogen-free one by 9 chi-square units and fits the pattern itself,
with its M+2 where its isotope puts it; "possible" above 4 or when one of those fails. A window that does not
co-elute with M, or whose position is outside any isotopologue (the +2.016 Da homologue of a co-eluting lipid is
the common case), is left out; a saturated M and a weak signal are flagged. The table gets `possible_halogen` (yes / maybe /
no, empty for isotopes and features without signal) and `halogen_pattern` (the best hypothesis, e.g. Cl2); the
feature list shows them as a **Halogen** column with a **Possible halogen** filter, and a called feature carries a
chip in its header. The compositions shown are pattern models, not identifications.

**Show MS1** (beside a feature's chromatograms) reads the MS1 spectrum at the feature's apex from the point-cloud
cache (the strongest file by default, any file on request) and draws it head-to-tail against the best hypothesis's
pattern, with the ratios, spacings, flags and the ranking of the hypotheses; a co-eluting peak one or two units
below warns that the feature may not be the monoisotopic peak. The report has no button: it needs the server and
the cache (`--keep-cache`).

On a cluster, start the GUI inside a job with `--host 0.0.0.0` and reach it through an SSH tunnel to that
node; the access token in the link keeps other users out.

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
| `--no-compounds` | off | skip compound grouping |
| `--compound-veto` | auto | intensity veto in compound grouping: `auto` rejects co-eluting pairs whose heights do not correlate across files (12+ files) or whose height ratio is unstable across files (3 to 11 files), `off` tests every co-eluting pair on peak shape, `ratio` uses the ratio rule at any study size; on the PR002464 spike truth `off` gains recall in one study but triples the unlabelled members per compound in all three |
| `--compound-shape weighted\|median\|pooled` | weighted | peak-shape score of a co-eluting pair: per-file EIC correlations averaged with the weaker ion's log height as weight, their median, or one correlation pooled over every scan of every shared file |
| `--compound-shape-files N` | 0 | shared files per pair for the shape score (0 = all; N = the strongest N); the median over the 5 strongest was the earlier default and scores 0 to 2 points lower on the spike truth |

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
| PR002464 | MWB PR002464 (ST004500/01/02/04 used) | Orbitrap ID-X, RP/HILIC, pos/neg | 96 spiked standards at designed A/B ratios -> spiked-standard truth (below) |
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
6. `python3 scripts/30_make_jobs.py [--tools T1,T2] [RUN ...] && sbatch --array=1-N scripts/31_run_tool.sbatch` - one
   array task per (tool, run) from `bench/runs.py`; results in `results/<tool>/<run>/` (`DONE`, `time.txt`, `tool.log`)
7. `envs/peak3d/bin/python bench/score.py` - match features to truth (ppm x RT grid) -> `results/scores/*.tsv`
8. (next) injection / noise / decoy generator -> `data/bench/` + truth tables

### Spiked-standard truth (PR002464)

The Broeckling/Du/Patti benchmark (Metabolomics Workbench PR002464) spikes
96 standards into two mixes: standard k sits in spike A and B at amounts a_k, b_k with log2(a/b) = +-1..4. The vials
include matrix-free A:B series (1:0, 0:1, 1:3, 3:1, 1:1; vials 18-22) and a 1:1 dilution series (vials 9-12), each injected
once per batch (4x). `bench/spiketruth.py` turns this into a tool-independent list of every peak the standards produce:

- discovery: EICs of 9-12 adducts per standard in vials 9/18/19 -> local maxima (height >= 10x floor, >= 3x running
  20th percentile, >= 30 % prominence)
- spike-derived (on matrix-free vials only): apex (local max of +-3 scans, >= 4 contiguous scans) in >= 3/4 injections,
  blank median <= 10 % of it, E series fits alpha A + beta B (residual <= 0.3), A and B differ (|log2 h18/h19| >= 0.4),
  C series falls with dilution
- tier A: also matches the labelled standard's own log2(A/B) within 0.75 (one-sided when the weak vial is at noise),
  and the standard's peak group holds a primary ion; tier B: spike-derived, label uncertain (co-eluting isomers,
  compressed ratios, ions of another mixture at this mass)
- co-elution sweep: every other m/z at a standard's main apex with the same profile (r >= 0.8) that passes the same
  tests -> isotopes, adducts, in-source fragments, impurities
- flags (kept, for the scorer to decide): `ft_artifact?` (within 0.05 Da of a >= 20x stronger co-eluting ion: Orbitrap
  sidebands), `broad` (FWHM > 3x the study median: overload smears, bands)

Run: `sbatch scripts/05_download_pr002464.sbatch` (centroid mzML + design, ~20 GB for the four Orbitrap studies;
`bench/pr2464_design.py` exports the design spreadsheet), then `sbatch scripts/44_spiketruth.sbatch ST0045xx` (2-4 min,
~3.5 GB; writes `data/truth/PR002464/<study>/`: `spiked_truth.tsv`, `per_file.tsv.gz` (height / area / apex per truth
peak and file), `standards_summary.tsv`, `all_tested_candidates.tsv`, `decoy_qc.tsv`, `files.tsv` (incl. RT offsets),
`params.json`, `gallery/`). Synthetic end-to-end test: `bench/tests/test_spiketruth.py`.

| study | mode | truth peaks | tier A (on standards) | tier B | flagged |
|---|---|---|---|---|---|
| ST004500 | RP neg | 2,650 | 1,933 (48) | 717 | 148 ft, 54 broad |
| ST004501 | HILIC pos | 473 | 357 (37) | 116 | 40 ft, 2 broad |
| ST004502 | HILIC neg | 976 | 692 (35) | 284 | 125 ft, 16 broad |
| ST004504 | RP pos | not usable | | | |

67 of the 96 standards are found in at least one study (19 in all three); the rest are volatile or do not ionise.
Tier-A formula ions: median |m/z error| 0.4-0.8 ppm, median |observed - designed log2(A/B)| 0.25-0.30.

Known gaps:
- **ST004504 (RP pos) is not usable as deposited**: the file contents do not match the sample metadata (spike
  standards at full height in files labelled solvent blank / plasma-only / yeast-only, absent from files labelled A+B
  spike and from 2 of the test.QC injections, while the fixed-position QCs agree). Every candidate fails the blank test.
- Peaks whose RT moves with load (acesulfame in RP neg: 2.70 min at 1/8 load, 3.04 min at full load) fail the
  fixed-RT design test and are missing.
- Ratio reads take the maximum within +-3 s of the apex, so isomers closer than that (2'- and 5-methoxyflavone in
  HILIC pos, 2.4 s apart) contaminate each other's ratio and land in tier B.
- `decoy_qc.tsv` (formula masses +-0.02 Da) is not a false-truth rate here: about half the decoy passes are peaks
  already in the truth (spike ions of other standards at that mass) and the rest pass the same spike-derived tests.

### Yeast and plasma truth (PR002464)

The same studies also hold CIL yeast extract (12C and U-13C) and two NIST plasmas, which give truth for biological
peaks. `sbatch scripts/45_biotruth.sbatch ST0045xx` (after the spike truth, whose RT offsets it uses) runs:

- **Yeast credentialing**: `bench/credtruth.py PR2464_<study>`, the SZ22 / NetID pipeline on vial 23 (12C) vs
  vial 24 (U-13C), 4 injections each (3 of 4 required), seeds >= 10x floor -> `yeast_cred/` (credentialed.tsv,
  decoys, gallery). `bench/biotruth.py yeast` then adds what this design offers on top -> `yeast_truth.tsv`:
  both partners in vial 25 (12C : 13C 1:1, same injection), blank read, dilution slope over vials 5-8;
  `confirmed` = present on both sides in vial 25 and absent from the blanks.
- **Plasma truth**: `bench/biotruth.py plasma` -> `plasma_truth.tsv`. Seeds from the 1000 uL plasma vials
  (13, 14, 17); a peak is truth when present in >= 3/4 injections of vial 17, absent from the blanks, proportional
  to the plasma dilution (vials 1-4, 1000 / 500 / 200 / 100 uL: tier A slope 1 +- 0.25, R^2 >= 0.95; tier B
  1 +- 0.5, R^2 >= 0.8) and consistent with the SRM 1950 : SRM 8231 mixing series (vials 13-17). Null: the same
  tests with the dilution amounts permuted (`plasma_null_qc.tsv`). Columns flag peaks also seen in yeast-only or
  spike-only vials and peaks that are spike-truth peaks.

| study | yeast credentialed (tier A) | confirmed in vial 25 | decoy rate A / B | plasma truth (tier A) | permuted-dilution passes |
|---|---|---|---|---|---|
| ST004500 RP neg | 8,160 (2,742) | 4,452 | 4.1 % / 9.2 % | 13,549 (7,054) | 1-45 |
| ST004501 HILIC pos | 1,241 (204) | 804 | 2.0 % / 1.4 % | 1,410 (820) | 0-25 |
| ST004502 HILIC neg | 2,185 (249) | 736 | 2.8 % / 2.2 % | 3,456 (1,782) | 1-25 |

Checks: known metabolites in the plasma truth (amino acids, carnitines, hypoxanthine, xanthine, uridine,
pantothenate: 20-24 of the list per study); credentialed carbon count = formula carbon count for every tier-A
known-metabolite match. RP-negative yeast is dominated by lipids (median 28 carbons, 10-16 min), where co-eluting
labelled species make wrong spacings pass more often (higher decoy rate): prefer tier A there.
`bench/diag/pr2464_label_audit.py` estimates each injection's spike A / B, plasma and yeast 12C / 13C content from
marker peaks and flags injections that disagree with their metadata (`results/pr2464_label_audit/`).

### PR002464 runs and scores

Each Orbitrap study is one run, `PR2464_ST004500` / `01` / `02` in `bench/runs.py`: all 224 MS1 injections (samples,
QCs, blanks) processed together, because the design mixes 12C and 13C yeast in the same vials. peak3d runs here as the
GLR variant (`narrow_rule=local_median`, `scripts/42_peak3d_variant.sbatch` -> `results/peak3d_GLR/`):

```bash
envs/peak3d/bin/python scripts/30_make_jobs.py --tools asari,masscube,idslipa PR2464_ST004500 PR2464_ST004501 PR2464_ST004502
sbatch --array=1-9 scripts/31_run_tool.sbatch
envs/peak3d/bin/python scripts/41_variant_jobs.py --out manifest/variant_jobs_pr2464.tsv \
    --runs PR2464_ST004500,PR2464_ST004501,PR2464_ST004502 GLR
sbatch --array=1-3 --export=ALL,VARIANT_JOBS=manifest/variant_jobs_pr2464.tsv scripts/42_peak3d_variant.sbatch
envs/peak3d/bin/python bench/score.py pr2464_spike pr2464_yeast pr2464_plasma runtime
```

Scorers, each over
the ppm x RT grid, a truth peak matched to the feature closest in RT; quantities are each vial's median over its 4
injections, missing = 0:

- `pr2464_spike`: recall (all, tier A / B, tier A without flags, formula ions); A/B ratio of tier-A peaks whose weak
  side is above noise, tool log2(vial 18 / vial 19) vs the designed log2(a/b) (median |error|, within 2-fold, right
  direction); dilution slope over vials 9-12 vs the raw-data slope of the truth builder.
- `pr2464_yeast`: recall of the 12C peaks (all, tier A / B, primary, confirmed in vial 25), of their 13C partners and
  of both; 12C dilution slope over vials 5-8 vs the raw-data slope.
- `pr2464_plasma`: recall (all, tier A / B, tier A without flags); dilution slope over vials 1-4 vs the raw-data
  slope; SRM 1950 / 8231 ratio (vial 13 / 14) vs the raw-data ratio.

A tool's full-study table also holds features of every other sample type, so no precision is computed here.
Synthetic check of the scorers: `bench/tests/test_score_pr2464.py`.

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
