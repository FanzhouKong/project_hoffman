# Noise-floor estimation in the centroid cloud

`07_cloud_noise_estimation` shows the **local cell noise floor** in three stages:

1. A centroid cloud contains persistent peak structures and scattered points.
2. After the initial persistence merge, select centroids in basins spanning at most two
   scans. Exclude a selected point if a centroid in a trace spanning at least five scans
   is at least 10 times stronger and within 100 ppm in the same scan (the sideband rule).
3. Within each logarithmic m/z band and one-minute RT block, take the **median raw
   centroid intensity** of those eligible points. A cell with fewer than 50 eligible
   points falls back to the file's first intensity percentile.

The third panel enlarges the low-intensity range: its vertical axis differs from the
first two. Teal tiles are computed cell medians; grey tiles use the sparse-cell fallback.
Orange points contribute to the median. Red crosses in panel 2 are excluded sidebands.
The floor is an intensity scale, not an arithmetic mean, baseline fit, standard deviation
of all points, or a final feature acceptance threshold.

## Suggested narration

“We estimate background noise from points that do not form persistent chromatographic
structures. After an initial merge, we retain points belonging to very short basins and
remove short points near much stronger traces. We divide the run into time–mass cells
and use the median intensity of those points as the local noise floor. Sparse cells use
a file-derived fallback. That floor then informs the refined merge and
subsequent peak assessment.”

## What the current code does

`pick_cloud` first estimates cell medians from initial basins spanning at most two scans,
and runs an initial persistence merge. With the default `noise_post_merge=True`, it
recomputes the eligible samples using those merged labels, excludes sidebands, calculates
new medians, and repeats the merge from the initial basins with the refined floor. The
figure illustrates the **refined floor calculation**. Noise samples are captured from
the first merge, rather than inferred from the final merge's labels.

The renderer reproduces these stages using the actual Peak3D kernels and `_merge`, then
checks its complete floor array against `pick_cloud(...).noise_surface`. It independently
verifies the sample counts, medians, and sparse-cell fallback for every cell. Results,
parameters and source hashes are stored in `noise_verification.json`; point memberships
and cells are in `noise_selection.npz`. Data are synthetic. The broad mass distributions
around peaks make their structure visible and are illustrative, not typical centroid
multiplicity. Four mass bands are used for readability; the default is **16**. One-minute
blocks, 50-point support and sideband criteria match the defaults.

Some older repository prose mentions a run-wide cap on cell medians. The current
`kernels.cell_medians` implementation has no such cap; this figure follows the code.
The first-percentile fallback is not a lower bound applied to every well-supported cell.

## Distinguish the other noise calculations

This figure explains the local floor; Peak3D also estimates signal-dependent fluctuations.
`estimate.fit_intensity_noise` extracts strong-ion traces, keeps their contiguous region
above 10% of the apex, calculates residuals against a five-point local quadratic
(Savitzky–Golay coefficients `[-3, 12, 17, 12, -3] / 35`), and excludes the apex residual.
The centre point contributes to that local fit; this is not a leave-one-out smoother.
Residuals are scaled by `sqrt(35/18)` to undo the white-noise variance reduction.

`fit_noise_law` bins residuals by half-decade of fitted intensity and uses
`1.4826 × median(|residual − median(residual)|)` as the robust SD per bin. It fits
`s0² + c I + r² I²` with non-negative coefficients, weighting by inverse bin variance.
At use time, `kernels.noise_at` replaces the fitted intercept scale with the local cell
floor: `sigma(I)² = cell_floor² + c I + r² I²`. This scales shape/noise descriptors and
optional noise-based merging modes; default fractional merging uses the cell floor.

The later EIC validation also estimates each trace's residual noise; the default
`eic_noise="max"` compares prominence with the larger of that residual noise and the
cell floor. The plotted cell floor should not be described as the only noise estimate
used anywhere in the pipeline.

## Files and reproduction

The full figure and its three **title-free standalone panels** are exported in PNG, SVG
and PDF. `Peak3D_noise_figures.zip` contains all versions, notes and reproducibility files.
Run `sbatch presentation/peak3d/render_noise.sbatch` from the repository root on Hive.
It uses the metabolomics group account on `high`, one task, two CPUs, 4 GB RAM and a
10-minute limit, with unique job logs and local temporary caches. Run one renderer at a time.
