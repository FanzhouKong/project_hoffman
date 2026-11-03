# Trace noise: figures 08 and 09

Both figures use the same synthetic centroid cloud and the repository's actual
EIC extraction/noise kernels. These illustrate calculations, not benchmark results.
The third coordinate is intensity. Noise is measured along time for one ion;
it is not the width of the cloud along m/z. Panel 3 shows one residual per scan
at the query ion's m/z, without inventing additional residual points.

## 08: default trace-residual noise

Extract the maximum centroid intensity per scan within the query mass tolerance.
Smooth with the Gaussian kernel used by the EIC check (10-scan FWHM here),
repeating edge values. Subtract smooth from raw. Select residuals outside the
candidate's inclusive scan bounds [53, 107], then calculate
`1.4826 * median(abs(residual - median(residual)))`.
If fewer than 10 scans lie outside the bounds, the implementation uses the whole
window instead. The example has 106 outside scans and does not use that fallback.

Default `eic_noise="max"` uses the larger of this trace noise and the local cell
floor from figure 07. The cell floor of 4.0 in this figure is illustrative and
assigned; it is not estimated from this synthetic cloud. EIC prominence is
divided by the resulting noise reference for the SNR check.

## 09: optional trace-own noise

`own_noise` fits a five-point local quadratic with weights
`[-3, 12, 17, 12, -3] / 35`, including the centre sample. Raw-minus-fit residuals
are multiplied by `sqrt(35/18)` to correct for white-noise attenuation, then the
same robust MAD scale is computed. Only five-scan stretches with all positive
intensities count. Peak-interior scans are included. Fewer than 10 valid residuals
returns -1, and `eic_noise="own"` then falls back to the local cell floor.
Otherwise that mode uses own noise directly, without maximizing it with the cell
floor. The optional `merge_test="own"` also uses this estimator for split tests.
Neither optional setting is the repository default.

This is distinct from the file-wide intensity-dependent variance law estimated
in `estimate.py`; that calculation uses pooled residuals and intensity bins.

## Files and checks

Each slide has three title-free individual panels in PNG (300 dpi), SVG and PDF.
Axes, explanatory callouts and numerical annotations are retained. Global titles,
panel headings, slide captions and footer are removed.

`trace_noise_verification.json` records exact-kernel comparisons, values and source
hashes. Independent calculations are checked against `eic_shape`'s `ES_NOISE`
and `ES_OWN`, plus `window_noise`. Saved NPZ files contain the cloud and all plotted
traces, residuals and masks. Render with
`sbatch presentation/peak3d/render_trace_noise.sbatch` from the repository root.

The job requests account metabolomicsgrp, partition high, one task, two CPUs,
4 GB RAM and ten minutes. Each job has unique logs and temporary plotting/JIT
caches. Only one renderer should write these outputs at a time.
