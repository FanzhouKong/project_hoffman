# Explaining Peak3D peak picking

Six 16:9 figures for a talk, with a consistent reading order. All data are synthetic.
The linking, basin labels, merge decisions and EIC-check outcomes are computed by the
current repository functions, rather than assigned by hand. These are teaching examples,
not benchmark results or a full processing run.

An additional pair of point-cloud figures demonstrates a shallow split merging and a deep
valley remaining separate. See [MERGING_NOTES.md](MERGING_NOTES.md),
`peak3d_cloud_merging_demo.pdf`, and `Peak3D_cloud_merging.zip`.

The local noise-floor calculation is illustrated in `07_cloud_noise_estimation`, with
three title-free panels and notes distinguishing the other noise calculations. See
[NOISE_NOTES.md](NOISE_NOTES.md) and `Peak3D_noise_figures.zip`.

Trace noise is illustrated in `08_cloud_trace_noise` (default residual estimate) and
`09_cloud_trace_own_noise` (optional five-point local-quadratic estimate). Each has
three title-free panels. See [TRACE_NOISE_NOTES.md](TRACE_NOISE_NOTES.md) and
`Peak3D_trace_noise_figures.zip`. Figure 07's three title-free panels are also
available separately in `Peak3D_07_subfigures.zip`.

## Files and suggested narration

Start with these two figures to explain the 3D representation and the methodological distinction.
`peak3d_3d_and_comparison.pdf` contains just these two slides.

- **`00a_what_3d_means` — What does the 3D in Peak3D mean?**
  “Each centroid has a retention time, m/z and intensity. Its position in time and mass
  determines which neighbours are eligible. Intensity directs its path toward an apex.
  The colours show surviving basins. A and C share a mass but elute at different times;
  B overlaps in time but lies at a different mass. Nearby split pieces of one centroid
  are linked into the same basin.” The right panel is the same graph from above. Height
  on the left is **effective** intensity, including same-scan split pieces; the centroid
  coordinates are retained and no continuous surface is fitted. Arrows on the left are
  a subset of actual links for legibility; the top-down view shows all parent links.

- **`00b_trace_based_vs_peak3d` — The difference is how peak candidates are formed.**
  “A trace-based workflow first constructs ion traces or regions of interest, then
  detects chromatographic peaks within them. Peak3D starts with local centroid links:
  points flow to apexes, defining basins, and persistence merges weak splits. The
  surviving basins become candidates for validation. The intermediate object changes
  from an ion trace containing peaks to a basin defined by an apex.” The upper lane is
  an architecture schematic; it is **not** output from a competitor. Its schematic
  chromatograms use the synthetic main-centroid profiles, vertically offset for clarity.
  The lower lane uses Peak3D's actual graph and merged labels. This is a difference in
  representation and operation order, not an accuracy comparison on this toy data.

Continue with the original four explanatory figures:

1. **`01_centroids_to_basins` — A basin is a cloud of points that leads to one apex.**
   “Each dot is a synthetic centroid. We connect neighbouring centroids and follow higher
   effective intensity until we reach an apex, marked by a star. All points that reach
   the same apex form a basin, shown in one colour. A shallow split is then merged,
   leaving three candidate basins.” The third panel shows surviving **candidates**,
   before feature gates and chromatogram validation. Direct arrows label one apex and
   the body of its basin. The clouds contain deliberately broad synthetic split-centroid
   distributions (15 pieces per mass per scan) so the groups appear as point clouds rather
   than narrow chromatograms. This illustrates graph membership, not typical centroid
   multiplicity or an instrument mass-peak profile. All assignments are recomputed from
   these input points by Peak3D, without adding display jitter to pre-labelled traces.
   Height is effective intensity; points below 2 arbitrary units are hidden for clarity.
   Faint floor patches are convex hulls of the visible points projected onto time–m/z,
   not algorithmic peak boundaries or integration regions.

2. **`02_uphill_links` — Each centroid follows a path to one apex.**
   “This is the same idea viewed from above: retention-time order versus m/z. Circle size
   and numbers show intensity. The arrows are real parent links. An apex has no higher
   eligible neighbour, and every centroid has one destination. Same-scan split pieces
   contribute to the effective surface height: here 88 plus 22 gives 110.” The separate
   ion example is deliberately well outside the linking tolerance.

3. **`03_persistence_merging` — Should two neighbouring basins be one peak?**
   “Two local maxima do not automatically mean two peaks. Compare the lower apex with
   the saddle connecting the basins. The shallow dip has persistence 10, below the
   threshold of 35, so the basins merge. The deep valley has persistence 55, so both
   basins survive.” The shaded areas indicate the **initial** basins. Both decisions are
   computed with the actual merging function, with the same local noise of 1 and the
   same apex separation of 14 scans; the expected FWHM is 10 scans.

4. **`04_chromatogram_validation` — A 3D candidate must also look like a chromatographic peak.**
   “For a candidate passing the basin gates, inspect its extracted ion chromatogram.
   A localised peak passes this check. A slice of a rising signal has no maximum inside
   its bounds. A weak ripple has a maximum but insufficient prominence relative to noise.”
   These are deliberately constructed candidate windows passed directly to `eic_check`,
   illustrating that check in isolation. A candidate passing this check still needs the
   final score; the weak background example may already fail an earlier background gate.

Each figure is exported as a PNG (3040 × 1710), an editable SVG with live text, and a
vector PDF. `peak3d_explained.pdf` contains all six pages; `preview.png` is a contact sheet.
`comparison_preview.png` previews the two opening figures at a larger size.
`Peak3D_figures.zip` bundles the figures, these notes, verification and regeneration code.
SVG text uses DejaVu Sans; install that font or use PDF/PNG if your slide software substitutes it.

## Accuracy notes for presenting

- Both approaches use retention time, m/z and intensity. Do not imply competing methods
  lack a mass dimension or ignore intensity while building traces. Trace-building rules
  may also be local, adaptive, intensity-aware and capable of handling gaps.
- A precise summary is: **“Peak3D detects candidates as apex-centred basins on a centroid
  graph, without first constructing complete ion traces for chromatographic segmentation.”**
  Local chromatographic links still exist, and extracted traces are used elsewhere in the
  pipeline (estimation, validation and quantification). “No traces anywhere” is inaccurate.
- The 3D terminology is an intensity landscape over the two acquisition coordinates,
  time and m/z. It is not a third separation dimension such as ion mobility, a full 3D
  Euclidean clustering scheme, or a fitted volumetric model.
- Local graph construction allows split pieces and m/z variation within the link rules to
  contribute to the same basin. That is a mechanism, not evidence of superiority in every
  overlapping-peak, drifting-mass or missing-scan case. Peaks closer than the tolerance
  may still become connected; benchmark results must establish practical advantages.

- “3D” here means scan/retention time, m/z, and intensity. Peak3D works on centroid
  neighbours, not a rasterised continuous 3D surface, and intensity determines ascent.
- The actual code chooses the **highest-intensity eligible neighbour**, not the neighbour
  with the largest geometric slope. Parent paths define the initial basins.
- Eligibility uses the intensity-dependent pair m/z uncertainty and the configured
  tolerance cap in nearby scans. Same-scan links also allow a split-centroid tolerance
  floor. The figures use one adjacent scan on each side for clarity (`K=1`).
- Same-scan split neighbours contribute to effective intensity. Ties favour the higher
  raw intensity and then the lower index. The linking example includes a real split pair.
- Figure 3 demonstrates the default `merge_test="frac"` branch:
  `persistence = lower_apex - saddle`, with threshold
  `max(0.5 * lower_apex, 3 * local_noise)`. It is **not the entire merge rule**:
  same-ion apexes closer than 0.8 median FWHM merge; sufficiently separated apexes
  (at least 2 median FWHM) can survive a 20% dip on a smoothed surface, subject to noise.
  Alternative noise-based merging variants exist. The examples avoid those spacing
  branches so the basic persistence comparison is unambiguous.
- Candidate generation is 3D, but the current implementation also applies a **1D EIC
  validation** after basin-level gates. Avoid describing the full pipeline as having
  no 1D chromatogram check.
- Other steps include per-file parameter estimation, scan-gain correction when enabled,
  noise estimation, feature bounds and descriptors, background/fragment/width/mass
  consistency gates, duplicate handling, isotope annotation and scoring. These figures
  focus on the picking mechanism and omit preprocessing and cross-file alignment.
- Parameters in these small demonstrations are explicitly supplied. They are not
  estimated from the toy data, and the examples are not intended to suggest dataset tuning.

## Code provenance and reproduction

### Primary references for the trace-based comparison

- [Tautenhahn et al., centWave, BMC Bioinformatics (2008)](https://doi.org/10.1186/1471-2105-9-504):
  detects mass-trace ROIs and then applies wavelet-based chromatographic peak detection.
- [Li et al., asari, Nature Communications (2023)](https://doi.org/10.1038/s41467-023-39889-1):
  builds mass tracks, aligns and aggregates them across samples, and detects peaks on
  composite mass tracks. The schematic omits cross-sample alignment and aggregation;
  it must not be read as asari doing independent per-file peak picking.
- [Yu et al., MassCube, Nature Communications (2025)](https://doi.org/10.1038/s41467-025-60640-5):
  constructs mass traces by signal clustering and segments them using Gaussian-filter
  assisted edge detection. Its ROI construction and segmentation are distinct stages.

These methods illustrate a family of architectures; the figure does not claim all existing
peak pickers use this architecture or that Peak3D is the first graph-based method.

### Repository sources

- `peak3d/kernels.py`: `find_neighbors`, `effective_intensity`, `link_steepest`,
  `resolve_roots`, `dense_labels`, `saddle_edges`, `persistence_merge`, `eic_shape`.
- `peak3d/pick.py`: `_merge` (the same merge wrapper used by `pick_cloud`).
- `peak3d/features.py`: `eic_check`; `gate` and `build_features` establish pipeline order.
- `peak3d/estimate.py`: `Params` defaults and parameter meanings.
- `peak3d/pipeline.py`: per-file preprocessing and processing order.

From the repository root, submit `sbatch presentation/peak3d/render.sbatch` on Hive.
Run only one renderer at a time because the output filenames are shared.
The script checks the demonstrated basin counts, split-centroid parent and height,
and all three EIC outcomes. It also verifies that split pieces share basin membership and
the separate co-eluting mass stays separate in the new 3D example.
`verification.json` records those results and source hashes.
`synthetic_centroids.npz` contains the centroid data used in Figure 1.
`synthetic_3d_graph.npz` contains the data for the two opening figures.

The batch directives request the metabolomics group account and its `high` partition,
10 minutes, one task with two CPUs, and 4 GB RAM. `%j` gives each job separate logs.
Numba and plotting caches live in per-job temporary storage. Outputs stay in this folder.
