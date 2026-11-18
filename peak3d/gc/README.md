# peak3d.gc: GC-MS (electron ionisation)

Status: preview (Mosa 1.4.0b1, branch `gc-support`). Checked on synthetic runs (`peak3d/tests/test_gc_*.py`), on
the MRH GC-Orbitrap set with the picker in lc and in gc mode, and on the ten unit-resolution runs of the DeepCPR
paper (both below).

GC-EI differs from LC-MS in what a result is. An EI source breaks every compound into tens of fragment ions that
elute together, so the unit is the compound and its spectrum; a compound is identified by matching that spectrum
and its retention index against a library. Picking, RT correction and ion alignment are the LC ones, in the
picker's gc mode; the GC stages run after them and write only to `OUT/gc/`.

```
# one command: pick in gc mode, then the GC stages (centroided MS1 mzML)
peak3d process --input mzml/ --output OUT --cores 8 --mode gc --ri-ladder alkane --ri-carbons 10-40 --ei-library nist_ei.msp
# or the GC stages on an existing output (again with other settings, or on an output picked in lc mode)
peak3d gc OUT --cores 8 --ri-ladder alkane --ri-carbons 10-40 --library nist_ei.msp
```

In `process`, the EI library goes by `--ei-library` (`--library` is the MS2 library of an LC-MS run); the other
GC options (`--ri-ladder`, `--ri-carbons`, `--ri-table`, `--ri-tol`, `--ri-type`) have the same names in both, and
`--min-similarity` and `--min-presence` are shared with the LC stages. A failure of the GC stages is logged
(`gc stages failed`) and leaves the rest of the run as it is.

## In the app

On the run form, **Acquisition: GC-MS (EI)** is `--mode gc`; a GC-MS card then takes the ladder and the EI
libraries. The results (and `report.html`) open on a **Compounds** tab:

- the list: one row per compound, at its **quantifier m/z** and **retention time** (s), with the retention index,
  base peak, number of ions, files it was found in, quantifier height, and the annotation with its match. Search by
  quantifier m/z, `#id` or name; list the compounds whose spectrum holds a given m/z; an RT window (typed, or a
  box dragged on the compound map); Download table (the list with every file's abundance) and Download spectra
  (.msp).
- the selected compound: its **spectrum** (the quantifier marked; against the library spectrum of a hit, for the
  three best hits of each compound), the library hits, the list of its ions (shared ones marked), the abundance of
  the quantifier in every file, and, in the app, the **chromatograms of its strongest ions** in any file that
  holds it, with the elution profile they were fitted with (read from the point-cloud cache, so a report file has
  no such plot).

The table of single ions aligned across files, the LC view, is the **Ions** tab of a GC-MS output. A search in a
library from the results page, and the user's own annotations and marks, exist for LC-MS features only so far:
to search another EI library, run `peak3d gc OUT --library ...` again (it rewrites `OUT/gc/` and the report's
compounds).

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
1. Every picked ion gets an apex time below the scan spacing (parabola through the logarithm of its top), with
   the fit's own standard error: a wide, flat or noisy top is placed less well than a sharp one.
2. The file calibrates itself on the isotope pairs the picker annotated (M and M+1 of one fragment, known to be
   one compound), per S/N class of the weaker ion: apex tolerance = 3 robust sd of the pairs' apex-time
   differences, shape cut = the chromatogram correlation 99 % of the pairs reach.
3. Components: the strongest free ion is a seed and the free ions within the apex tolerance are its
   neighbourhood. The reference ion is the one of the five strongest there that most of the others correlate
   with. Ions that correlate with it are tight members, the others loose members of the same component: two
   components never start at one apex time. The model profile is the per-scan median of the five strongest
   tight members; every ion then goes to the model it fits best among the components within its tolerance.
   `n_features` counts tight members, `n_loose` the others; `min_features` (default 1) can require more picked
   ions per component.
   A component of one picked ion whose apex time is less certain than its class's (`tau_own`) joins the
   component whose model its chromatogram follows, within a tolerance wider in proportion to its own error
   (an ion r times as uncertain as its class differs from a typical partner by sqrt((r^2 + 1) / 2) as much). It
   does not shape that model. Components of several ions are left alone: ions that agree on an apex time are
   a compound, however badly a parabola fits each (a tailing peak is fitted badly in all its ions alike).
   The profiles that are reported and that the spectra are fitted with (`model_net`) are built from the
   members' chromatograms above the level each holds beyond its bounds: a background ion (column bleed,
   reagent, air) among the five strongest members of a weak peak stands on a fifth of its height and more,
   and would put a pedestal under the profile that the compound's other ions do not have. Grouping is decided
   on the chromatograms as they are (correlations and fits with a constant are indifferent to such a level).
3b. Delayed copies. Some fragments are formed after the column (in the ion source, on a surface): their signal
   starts when their compound's does and peaks and tails later, by 0.15-0.45 s on the MRH set, past the apex
   tolerance, so they made a component of their own. A second compound that elutes later starts later. A
   component b whose apex is within one peak width after a larger component a is a's delayed copy when
   a's profile passed through a first-order lag describes b's profile (correlation >= b's shape cut), does so
   better than a's profile moved later (F >= 5), and moving it later on top is not needed (F < 5). F uses a noise
   floor from the shape cut, so two profiles that agree within their noise are left as the apex times had them.
   The copy's ions join a; the spectrum is fitted with both profiles; `n_delayed` counts the copies.
3c. Explained components (`_explained`, `_rejoin`). In a cluster of overlapping compounds most picked ions are
   shared, and each one's apex lies somewhere between the compounds' apexes, by more than the apex tolerance: the
   steps above made a component at every such apex time (11 components for two fatty acid methyl esters 1.7 s
   apart on a quadrupole run), and the same happened to a single compound at low signal, whose noisy apexes
   scatter. So, from the smallest component up, a component is dissolved when the larger components that
   overlap it describe its strongest ions without it: (1) their models, fitted non-negatively with a constant,
   leave no more than the share of variance an isotope partner is allowed to leave (1 - shape cut^2), and
   (2) adding its own model does not lower the residual significantly (F < 5). A component that neighbours do
   not describe is never dissolved, however weak. Smaller components that passed their own test take part in
   the explanation as long as the larger ones carry more than half of it (`explain_smaller`): the ions a large
   compound shares with a small one before it are the large one's peak with a shoulder, and there can be more
   of them than the small compound has ions of its own. Each ion of a dissolved component is fitted again with the
   live models: when one model carries at least 80 % of the fit and the fit reaches the ion's shape cut, the ion
   is a member of that component (`n_rejoined`); otherwise it is shared. Model profiles are single-peaked
   (`_unimodal`): where a profile rises again away from its apex (a neighbour's peak in the member ions) it is
   continued as a decay. In the spectrum fit a component's own picked ions start from its own model.
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
| `hit_spectra.npz` | the library spectra of each compound's three best hits (compound_id, rank, off, mz, inten), for the viewer |
| `ladder.tsv` | the ladder compounds found per file: carbon, ri, rt, rt_corr, component, molecular_ion |
| `components/<file>.tsv` | the file's components: rt, bounds, n_features, n_ions, n_shared_ions, base and quantifier ion, tic, model_r, apex_tol_s, rt_corr, ri, compound_id |
| `components/<file>.npz` | their spectra: off, mz, abundance, shared; and model profiles (model_off, model: maximum 1, on the component's scans scan_min..scan_max) |
| `components/<file>.features.tsv` | per picked ion: component_id (-1 none), model_r, shared |
| `params.json` | settings, each file's calibration (tolerances, isotope pairs), ladder notes, timings |

Numbers are written with six significant digits, m/z columns (`quant_mz`, `base_mz`) to 1e-5.

## Deconvolution on the MRH set

`bench/gc/deconv_eval.py` scores the clustering on picked ions (a truth compound is whole when all its found ions
sit in one component; a pair of truth compounds less than two peak widths apart must stay in two).

| lc-mode run, 11 files | first version | one component per apex time | + delayed copies |
|---|---|---|---|
| truth compounds whole | 72.8 % | 80.2 % | 84.3 % |
| ... with at least 90 % of their ions in one component | 81.7 % | 87.2 % | 91.8 % |
| components | 17,120 | 9,051 | 8,548 |
| close pairs separated (108; BDE 153 / TBBPA at 0.2-0.5 s) | all | all | all |

gc-mode run: 69.1 %, 74.9 %, 78.7 % whole; 23,532, 11,545, 10,756 components; all pairs separated.

With the explained-component step (2026-11-18): 90.1 % whole (94.9 % with at least 90 % of their ions), 7,207
components, all 108 pairs separated; gc-mode run 86.8 %. In the spectra (`bench/gc/mrh_score.py`, compounds with at
least 3 clean ions): all clean ions in one spectrum 80.8 -> 86.3 % (gc-mode run 77.6 -> 87.4 %), compounds whose
ions sit in two or more components 27.4 -> 24.4 % (34.2 -> 24.4 %). With smaller components taking part in the
explanation (`explain_smaller`, 2026-11-18): 91.1 % whole (95.7 %), 7,069 components, all pairs separated; gc-mode
run 87.3 %. Tried and not adopted: moving a component's bound past a valley that cut its reference ion (89.2 %
whole, slower); any surviving component explaining any other, without the half limit (83.9 %); the model profile
as the lower quartile of its ions instead of the median (90.8 %, and worse fits in co-eluting clusters).

Open: a compound that is only a shoulder in every picked ion gets no component. `shoulders` (off) seeds
components at second maxima that at least three ions share (`_sub_apexes`, `_coincident`, `_complete`): it finds
the linoleate shoulder of the DeepCPR fatty-acid runs at 0.6x, but on the MRH set it adds about 2,000 components
and lowers whole compounds to 90.4 % (gc-mode run 84.9 %), so it is not a default.

With `model_net` and `tau_own` (2026-11-18): 91.3 % whole (95.7 %), 6,948 components, all 108 pairs separated;
gc-mode run 88.0 %. `model_net` alone changes no truth compound. On the DeepCPR sets: amino acids R2 0.958 ->
0.971 (5 mg/L 0.869 -> 0.929), 60 of 60 cases found (methionine at 5 mg/L was split in three); FAMEs 0.983 ->
0.984, co-eluting clusters unchanged at 0.974 except the 565-568 s segment at 0.2x (0.907 -> 0.888). Tried and
not adopted: the floor already in the models that decide the grouping (MRH 90.4 %, single-ion components no
longer dissolved); the wider tolerance for every ion joining a seed (MRH 92.3 %, but a large neighbour takes
the ions of a close tailing compound: TBBPA lost in three files, C18 FAME cluster worse).

## Unit-resolution runs: the DeepCPR benchmark

`bench/gc/cdf_to_mzml.py` turns a quadrupole .CDF into the centroid mzML the picker reads (nominal masses written
13C-spaced so that M / M+1 pairs calibrate the deconvolution, a noise floor from the file's negative points, the
scan skew measured and removed). `scripts/51_deepcpr_gc.sbatch`, `52_deepcpr_tool.sbatch`, `54_deepcpr_score.sh` run
Mosa and DeepCPR (Yu et al., Anal. Chem. 2026, 98, 32686) on that paper's amino-acid and fatty-acid mixture runs;
`bench/gc/deepcpr_paper_table.py` scores Mosa the paper's way. Results and what is still open:
`results/gc_deepcpr/summary_2026-11-17.md`.

The delayed-copy test on labelled pairs of six files (`bench/gc/diag/delay_test.py`): of 45 split-off groups of
truth compounds 19 are called copies (both Dechlorane Plus isomers at every level, BEHTBP, OBIND, BDE 153), of
25 pairs of different truth compounds 0.24-1.9 s apart none. Groups that are a pure small move of their compound
(T23BPIC, EHTBB, BTBPE: 0.1-0.15 s, the same shape) and HBCD, which decomposes on the column, stay split: a
lag that short cannot be told from a second compound. `min_features=3` halves the components (4,817 before
the delayed copies) but takes the component away from 2.5 % of the truth compound rows, so it is not the default.

## Open

- The MRH set is standards in solvent, so it does not test heavy co-elution.
- Input is centroided mzML. ANDI-MS netCDF (.cdf), the usual export of unit-resolution GC-MS, is not read: the
  DeepCPR runs went through `bench/gc/cdf_to_mzml.py`, which writes nominal masses multiplied by 1.0033548 (so
  that isotope peaks pair up); the app shows those m/z as they are, and the EI search bins them wrongly above
  m/z 209. Reading such files directly, with nominal masses reported, is the next step.
- Search ranking (match vs reverse vs a mix), the index window and the 0.6 cross-file similarity are untested
  choices, to be judged on a set with known compounds.
- No gap filling of compounds; no library search, annotation or marks for compounds from the app's results page.
- Saturated ions are not excluded from the model profile.
- Reading a large library (NIST EI, ~350k spectra) the first time is a plain text parse; the parsed library is
  cached in `OUT/gc/.library_cache/`.
