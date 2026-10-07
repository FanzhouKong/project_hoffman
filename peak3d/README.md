# peak3d: 3D apex peak picking and reference-free RT correction

`peak3d` is the in-house feature detector of this benchmark. It differs from the other tools in two
ways.

**Purely 3D peak picking.** There is no mass-trace / ROI step followed by 1D chromatogram peak
detection. All MS1 centroids of a file form one point cloud in (scan, m/z, intensity). Every
centroid is linked to its most intense neighbour within the m/z tolerance in the adjacent scans
(and to same-scan split centroids: adjacent centroids within the pair tolerance or within 5 ppm,
`Params.same_scan_k` / `same_scan_floor_ppm`, whose intensities are summed into the surface height); following the links assigns each centroid
to exactly one apex, so a feature is a basin of the intensity surface. Where a link tie has to be broken
(two split pieces share one summed height) the piece with the higher raw intensity is the root, and each
scan's representative point for the feature's m/z, m/z sd and m/z range is the basin point with the
highest raw intensity in that scan (`Params.rep_raw`): a weaker ion a few ppm away that was linked in as
a split piece cannot stand for the scan and pull the m/z statistics to its own strand (on the yeast
credentialing files this fusion used to reject hundreds of strong peaks at the m/z-sd gate). Apexes that are not persistent (the valley between two apexes stays above half the lower
apex, or within 3x the local noise of it) are merged into their neighbour (ToMATo persistence
merging with the chromatographic half-valley criterion). Apexes at least two median peak widths
apart are two peaks already when the valley on a lightly smoothed surface (kernel FWHM half the
median peak width, along each centroid's chromatographic links) falls 20 % below the lower apex: a
dip inside one peak (ion suppression, AGC) lies within about one width of its apex and keeps the
half-valley rule, while a second compound eluting a few widths after a bigger one often sits on
its tail with a valley at 55-65 % of its own height. Each surviving basin is described (bounds, area, width, m/z consistency,
S/N, prominence, shape), gated (a maximum on a bound is a fragment only if the adjacent scan
holds a higher centroid of the same ion; a candidate whose chromatogram, 2 to 6 widths away on both sides, keeps a
90th-percentile level above half its apex or a lower-quartile level above a third of it is a
background ion, not a peak), annotated with isotopes and scored. **S/N** is measured against the
feature's own chromatogram: (smoothed apex - median level of the cleaner side window, 2 to 6 widths
out) / max(that side's scan-to-scan noise, cell noise), kept at >= 5. The cell noise is the median
of isolated centroids per m/z band and 1-min block, capped at the band's run-wide median, because
where many compounds elute the short basins of a cell hold fragments of real ions. On an empty
Orbitrap baseline the cell noise sets the scale; on a zig-zagging background ion its own ripple
does; and a narrow spike loses more height to the smoothing than a peak of the expected width.

**RT correction from the data itself.** No internal standards. Each file's most credible features
(monoisotopic, high score and S/N, m/z-isolated, spread across the gradient) are anchors. A medoid
file defines the output RT axis; every other file is matched to it by mutual-unique m/z matches,
and a monotone warp (PCHIP through robustly weighted binned medians, straight lines across
anchor-free stretches, constant shift beyond the anchors) maps it onto that axis. Two to three consensus iterations refine the target with the
union of all anchors. **Do-no-harm check:** the anchors are the strongest, most isolated ions, and
their drift is not always that of the other compounds (compound-specific drift; a constant shift
extrapolated beyond the first or last anchor). Every warp is therefore checked on held-out features:
all non-anchor monoisotopic peaks (S/N >= 3, >= 4 scans) that are unambiguous in m/z and RT and are
found in as many files as the external checks require. Per segment of native RT (the anchor-free head and
tail on their own when they hold >= 25 features, the rest in quantile segments of >= 50), the warp's shift is scaled by the factor
in [0, 1] that minimises the mean absolute deviation of these features from the median of the other
files (the medoid for 2-3 files): 1 keeps the fit, 0 leaves the file on its native axis. Factors are
interpolated between segments; the scaled warp stays monotone. On the held-out features the applied
correction is therefore never worse, segment by segment, than none. Each file's m/z scale is recalibrated by the median offset of its anchors against the consensus
(batch-wise calibration drifts of a few ppm are common). Features are then grouped across files
on the corrected axes: m/z within each pair's own intensity-dependent tolerance, RT within the
larger of 3x the anchor residual MAD and half the narrower peak's width (apexes of one compound
overlap), one member per file; a second pass merges same-ion groups with overlapping RT ranges
and no file in common (fragments of one wandering apex; isomers share files and stay separate).
Missing cells are gap-filled from the raw centroids on each file's native axis.

**Intensity-dependent noise model.** The cell noise surface is the detection floor. On top of it every
file gets a variance law for the scan-to-scan noise of an ion current at level I,
sigma(I)^2 = floor^2 + c I + r^2 I^2 (a shot-noise-like and a proportional term; on a Q-TOF the
ripple on a 20,000-count peak is several times the floor). c and r are fitted per file from the residuals
of the strongest ions' traces (contiguous scans >= 10 % of the apex), binned by level, robust sd per bin,
non-negative least squares weighted by 1 / variance. The model scales the shape descriptors of every
candidate: `n_valleys` (local minima inside the bounds at least 3 sd of the level difference below the
maxima on both sides: 0 for a peak, several for a lumpy hump), `tv_excess` (total variation beyond one
rise and fall, in summed noise sd) and `noise_i` (sigma at the apex). `n_valleys` can gate
(`Params.max_valleys`, off by default); it is reported as the `lumpy` flag at >= 2. c and r come from
the residuals of each strong trace against the 5-point local quadratic through its neighbours (apex scan
excluded), which follows tailing and shoulders, so the residual is noise and not shape.

**Chromatographic localisation and unimodality.** The ridge / background side windows (2 to 6 widths
out) use the feature's own width floored at the file's median width and capped at twice it
(`Params.ridge_w_cap`): a ridge segment ten widths long is judged beside itself, not 20-60 widths away,
where the gate used to find nothing. The same windows give `far_level`, the ion's typical level where it
is present on the cleaner side, relative to the apex. The rule score is the geometric mean of seven
terms: S/N, Gaussian r2, m/z consistency, relative prominence, scans per width, localisation
(1 - far_level) and unimodality (1 / total-variation ratio, which is 1 for one rise and one fall), plus
the isotope bonus, cut at `--min-score 0.5` (coverage first; the chromatogram check and the presence
rule below remove what the score lets through). On the IDSL003 labels the window cap removes 84 curated
negatives for 8 positives.

**Chromatogram check (1D, per file).** Every candidate that passes the basin gates is also judged on the
plain extracted ion chromatogram of its m/z, 10 widths to each side, smoothed with a kernel of half the
median peak width: the smoothed trace must have a local maximum inside the candidate's bounds (otherwise
the basin is a slice of a tail, ramp, step or a neighbour's flank), that maximum's prominence must be at
least 3x the larger of the trace's own residual noise outside the bounds and the cell noise, and a
weaker same-ion candidate on the same smoothed maximum (within a quarter width) is a duplicate of the
stronger one. The 3D picking finds the candidates with full sensitivity; the 1D view removes what does
not look like a peak in the chromatogram. `eic_snr` is reported; `Params.eic_check` turns the test off.
A candidate whose ion still stands at more than 50 % of the apex 2 to 6 widths out on its cleaner side
(`far_level > Params.far_level_max`) is a bump on a tail or ridge and is rejected as well (a 30 %
ceiling also removed resolved peaks on a same-ion pedestal and small peaks beside a bigger peak of the
same ion: on the validated IDSL003 peaks it cost 33 true peaks for 27 false ones).
The ridge and background gates judge only sides that the run edge leaves at least 3 scans long, so a
feature at the run start or end is judged on the side it has.

**Presence across files.** In a study of two or more files a group is reported only if it is present in
at least `--min-presence` files (default 2, capped at the file count). Presence is being detected by
the picker *or* being confirmed in gap filling: a missing cell counts when the smoothed chromatogram at
the group's position holds a local maximum with prominence >= 3x the larger of its residual noise and
the group's reference cell noise, at least 5 % of the group's reference height. Weak peaks near the cut
are picked in some injections only, so counting picker calls alone would lose 10-20 % of credentialed
truth; counting confirmed peaks loses about 3 %. `presence_mask.tsv` holds, for every group before the
rule, 2 (detected), 1 (confirmed) or 0 per file, so any other rule can be applied afterwards;
`n_confirmed` is in the aligned tables and `group_kept` in the per-file tables.

All parameters are estimated per file (scan spacing, peak width, intensity-dependent m/z error
model, noise floor, intensity-dependent noise law); nothing is tuned per dataset.

## Usage

```
PYTHONPATH=$ROOT envs/peak3d/bin/python -m peak3d process --input DIR --output OUT --cores 8
PYTHONPATH=$ROOT envs/peak3d/bin/python -m peak3d pick FILE.mzML ... --output OUT
PYTHONPATH=$ROOT envs/peak3d/bin/python -m peak3d align --output OUT [--no-rt-correction] [--no-gap-fill]
```

Options: `--ppm 5`, `--rt-tol-max 0.25`, `--min-score 0.5`, `--min-presence 2`, `--polarity pos|neg`,
`--no-rt-correction`, `--no-gap-fill`, `--holdout`, `--quant height|area`, `--keep-cache`,
`--keep-rejected`, `--model PKL` (optional re-scorer with `feature_columns` and `predict_proba`).

## Outputs (`OUT/`)

| file | content |
|---|---|
| `aligned_feature_table.tsv` | groups passing the presence rule: group_id, mz, rt (min, consensus axis), rt_min, rt_max, n_detected, n_confirmed, n_filled, mz_ppm_spread, rt_sd, iso_offset (majority isotopologue label of the members, 0 = monoisotopic), charge, one height column per file stem |
| `presence_mask.tsv` | every group before the presence rule: 2 detected, 1 confirmed peak in the gap-filled chromatogram, 0 absent, per file |
| `aligned_feature_area.tsv` | same layout with areas |
| `filled_mask.tsv` | 1 where a value was gap-filled rather than detected |
| `features/<stem>.tsv` | per-file features: mz, rt, rt_apex, height, area, area_bsub, rt_min, rt_max, scans, n_scans, n_gaps, fwhm, asym, mz_sd_ppm, baseline (where the bounds walk stopped), baseline_lo, noise (cell floor, used in the S/N), snr, prominence_rel, persistence, gauss_r2, ridge_ratio, background_ratio, far_level, tv_ratio, n_valleys, tv_excess, noise_i (sigma at the apex under the file's noise law), iso_offset, charge, iso_parent, iso_ratio, score_rule, score_model, score, flags, rt_corr, rt_min_corr, rt_max_corr, is_anchor, group_id |
| `features/<stem>.params.json` | every per-file estimated parameter, counts and stage timings |
| `rt_correction/<stem>.tsv` | the warp: scan, rt_native, rt_corr |
| `rt_correction/summary.tsv` | per file: role, anchor tier, counts, knots, coarse shift, max shift, MAD before/after (s), do-no-harm check (validation features, smallest / median factor, their mean deviation native / anchor fit / applied, s), flag |
| `rt_correction/anchors.tsv` | every matched anchor with residuals before/after the anchor fit (before the do-no-harm scaling) |
| `qc/` | drift curves, residual histograms, anchor fits, presence histogram, `holdout.tsv` |
| `params.json` | resolved parameters, versions, medoid, tolerances, stage times |

RT is in minutes everywhere; residuals in the QC files are in seconds.

## Layout

`io.py` (CSR point cloud), `estimate.py` (per-file parameters), `kernels.py` (numba watershed),
`pick.py` (orchestration), `features.py` / `isotopes.py` (descriptors, gates, score),
`warp.py` / `anchors.py` / `align.py` (RT correction), `grouping.py`, `gapfill.py`, `qc.py`,
`pipeline.py`, `__main__.py`, `synth.py` (synthetic data = test oracle), `tests/`.

Tests: `PYTHONPATH=$ROOT envs/peak3d/bin/python -m pytest peak3d/tests -m "not perf"` (run on a
compute node; `-m perf` for the speed budget test).
