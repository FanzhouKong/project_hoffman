# peak3d.gc: GC-MS (electron ionisation)

Status: first version. Checked on synthetic runs (`peak3d/tests/test_gc_*.py`) and, with the picker in lc mode, on
the MRH GC-Orbitrap set (below). `--mode gc` has not been through real data as a run yet: its rule was sized by a
dry run on that set's candidates.

GC-EI differs from LC-MS in what a result is. An EI source breaks every compound into tens of fragment ions that
elute together, so the unit is the compound and its spectrum; a compound is identified by matching that spectrum
and its retention index against a library. Picking, RT correction and ion alignment are the LC ones, in the
picker's gc mode; the GC stages run afterwards, on an existing output folder, and write only to `OUT/gc/`.

```
peak3d process --input mzml/ --output OUT --cores 8 --mode gc     # centroided MS1 mzML
peak3d gc OUT --cores 8 --ri-ladder alkane --ri-carbons 10-40 --library nist_ei.msp
```

## Modes: `--mode lc` (default) and `--mode gc`

A mode is a set of picker settings (`estimate.PROFILES`) applied before the file's own estimation; a `--set
NAME=VALUE` goes on top, so either mode can be changed setting by setting. `params.json` records the mode and the
variant label shows what it changed (`GLRSZWM + background_rule=net`). lc is the picker as it was: an output
picked without `--mode` is the same as before.

What gc changes: `background_rule=net`. Four gates ask whether the candidate's ion is still there 2-6 peak widths
from the apex (ridge, background, far level, bounds capped on both sides) and reject it as a piece of a background
ion when it stays at a third to a half of the apex. In LC that is a solvent or contaminant ion. In GC-EI the small
fragment ions (m/z 103.054, 115.054, 152.062 ...) come from many compounds and from the column background, so a
compound's fragment is a peak on a floor of the same ion. Under `net` a candidate those gates reject is kept when
it is a peak above the ion's own level out there (the median of the cleaner side):

| setting | default | the candidate is kept when |
|---|---|---|
| `net_ripple_k` | 5 | its rise above that level is at least this many times the level's scan-to-scan noise |
| `net_ridge_max` | 0.5 | rises of this share of its own are not common out there (90th percentile above the level) |
| `net_fwhm_min` | 0.5 | its width at half the rise is at least this many median peak widths, reached on both sides |

Such a feature carries the flag `on_background`; its height includes the floor. It is scored above the floor
(prominence, far level), is not a reference for the file's peak-width thresholds and is exempt from the wide-view
background gate.

Measured on the MRH set (11 injections, 0.1-1000 ng/mL, lc-mode run, `results/gc_mrh`): 115 truth ions were
rejected by those gates, median height 75,000 counts and S/N 45. Dry run of the rule on the picker's candidates:
91 of 106 are kept; 2,680 more candidates per file pass (24 % of the features), 73 % of them with three or more
picked ions at the same apex against 27 % of all candidates those gates reject. The deconvolution reads spectra
from the raw scans with a constant term, so two thirds of these ions were in their compound's spectrum already;
what the rule adds is that they are features (they can define a component and be a quantification ion).

## Stages

**Deconvolution (`deconv.py`, per file).** On a TOF or Orbitrap all ions of one compound share an apex time and
a peak shape; two co-eluting compounds differ in apex time, often by a fraction of a peak width, and may produce
the same ion.
1. Every picked ion gets an apex time below the scan spacing (parabola through the logarithm of its top).
2. The file calibrates itself on the isotope pairs the picker annotated (M and M+1 of one fragment, known to be
   one compound), per S/N class of the weaker ion: apex tolerance = 3 robust sd of the pairs' apex-time
   differences, shape cut = the chromatogram correlation 99 % of the pairs reach.
3. Components: the strongest free ion is a seed; ions within the apex tolerance whose chromatogram correlates with
   it join. Each component's model profile is the per-scan median of its five strongest members.
4. An ion shared by two compounds less than a peak width apart is picked as one peak between them. A component of
   at most six ions whose model is a mix of the models of the nearest larger component on each side is dissolved,
   and its ions are marked shared. (Shape alone cannot tell this from a small compound eluting between two large
   ones; that case is lost.)
5. Spectra come from the raw centroids: every centroid of the component's apex scan is fitted, over the
   component's bounds, with a constant background plus the models of the component and its overlapping
   neighbours (non-negative, forward selection with an F test). The coefficient is the ion's abundance. Fragments
   too weak to be picked are therefore in the spectrum, shared ions are divided, column bleed goes to the
   constant.

**Compounds across files (`pipeline.py`).** The strongest unmatched component is a reference; in each other file
the component within the alignment's RT tolerance (corrected time) with the most similar spectrum joins at a
similarity of 0.6 or more. The quantifier is the strongest ion of the reference spectrum that needed no other
component's model; each file's value is that ion's abundance in its own deconvolved spectrum. No gap filling yet:
a file without the component has an empty cell.

**Retention index (`ri.py`).** van den Dool & Kratz: linear between the bracketing ladder compounds, the nearest
segment extended outside the ladder (flagged). `--ri-ladder alkane` (RI = 100 x carbons) or `fame` (FiehnLib
values, C8-C30) finds the ladder in the deconvolved runs: components dominated by m/z 57.0699 / 71.0855 / 85.1012
(alkanes) or 74.0362 / 87.0441 (FAMEs); the accurate-mass molecular ion names the carbon number, and the rest are
placed in elution order within `--ri-carbons`. `--ri-table` takes a ladder table instead (rt or rt_s, and ri or
carbon). The study uses one map on the corrected time axis (median time of each ladder compound over the files
where it was found), so a ladder injected once serves every file.

**Library search (`library.py`).** EI `.msp` libraries, unit-mass (NIST, Fiehn) or accurate-mass. Against a
unit-mass library the measured spectrum is binned to nominal masses (n - 0.3 <= m < n + 0.7); an accurate-mass
library is matched within max(3 mDa, 10 ppm). Scores on weights intensity^0.6 x m/z^3: `match` (both spectra in
full) and `reverse` (measured peaks without a library partner left out). Records outside `--ri-tol` of the
compound's index are not candidates (default 20 for alkane, 3000 for FAME indices); records without an index
stay candidates. Ranked by `match`; a best hit at `--min-similarity` (0.7) or more becomes the annotation.

## Outputs (`OUT/gc/`)

| file | content |
|---|---|
| `compounds.tsv` | one row per compound: compound_id, rt (min, corrected), n_files, quant_mz, base_mz, n_ions, n_shared_ions, height, reference file and component, ri, ri_extrapolated, and the best hit (name, formula, inchikey, match, reverse, library_ri, delta_ri) |
| `compound_table.tsv` | compound x file: abundance of the quantifier ion |
| `spectra.msp` | the reference spectrum of every compound (accurate m/z, base peak 999, RI, RT) |
| `identifications.tsv` | every listed hit (match >= 0.3, up to 10 per compound and library) |
| `ladder.tsv` | the ladder compounds found per file: carbon, ri, rt, rt_corr, component, molecular_ion |
| `components/<file>.tsv` | the file's components: rt, bounds, n_features, n_ions, n_shared_ions, base and quantifier ion, tic, model_r, apex_tol_s, rt_corr, ri, compound_id |
| `components/<file>.npz` | their spectra: off, mz, abundance, shared |
| `components/<file>.features.tsv` | per picked ion: component_id (-1 none), model_r, shared |
| `params.json` | settings, each file's calibration (tolerances, isotope pairs), ladder notes, timings |

## Open

- `--mode gc` not yet run on real data (`scripts/50_mrh_gc.sbatch` takes `MODE=gc`); the MRH set is standards in
  solvent, so it does not test heavy co-elution.
- Search ranking (match vs reverse vs a mix), the index window and the 0.6 cross-file similarity are untested
  choices, to be judged on a set with known compounds.
- No gap filling of compounds, no GUI view of compounds; `peak3d gc` is still a second command after
  `process --mode gc`.
- Saturated ions are not excluded from the model profile.
- Reading a large library (NIST EI, ~350k spectra) the first time is a plain text parse; the parsed library is
  cached in `OUT/gc/.library_cache/`.
