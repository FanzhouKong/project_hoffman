# Point-cloud demonstration of split merging

`05_cloud_split_merging` is the main figure: two initial basins, a shallow connecting
saddle, and the merged basin. `06_cloud_separation_preserved` is a matched control with
a deeper saddle: both basins survive. Each is available as PNG, editable SVG and PDF.
`peak3d_cloud_merging_demo.pdf` combines the two; `Peak3D_cloud_merging.zip` bundles the
figures, synthetic inputs, verification, notes and rendering code. These supplement the
existing six-slide package; they do not replace it.

## Individual panels without titles

Both figures are also exported as three standalone panels each:

- `05_cloud_split_merging_panel_1`, `_panel_2`, `_panel_3`.
- `06_cloud_separation_preserved_panel_1`, `_panel_2`, `_panel_3`.

Each panel is available as a 300-dpi PNG, editable SVG, and PDF. The main figure title,
numbered panel heading, captions, branding, and footer are omitted; axes, point clouds,
apex markers, and in-plot explanatory labels are retained. The original complete figures
remain available. `peak3d_05_06_subfigures.pdf` contains all six standalone panels in that
order; `Peak3D_05_06_subfigures.zip` bundles them. `subfigures_preview.png` shows figure 05
across the top row and figure 06 across the bottom row.

## Suggested narration

**Merge example:** “Initially, each local maximum owns a basin. The lower apex is 70,
and the connecting saddle is 60, so its persistence is only 10. The threshold here is
35, so Peak3D merges basin B into A. The former B maximum remains in the data, but is no
longer a separate basin root. Every centroid stays in place: the algorithm merges
memberships rather than smoothing away the dip.”

**Preserved separation:** “The apex heights and spacing are the same, but the saddle is
now 15. The lower apex rises 55 above that saddle, exceeding the same threshold of 35.
Both basins therefore survive, with their own apexes.”

The star is an active basin apex. In the merged view, the hollow circle marks the former
root B; the lower maximum is still visible. Faint floor polygons show convex hulls of
visible point projections, not algorithmic integration bounds.

## What is computed

All neighbour links, initial memberships, connecting saddle heights and merge decisions
come from the current Peak3D functions (`find_neighbors`, `effective_intensity`,
`link_steepest`, `saddle_edges`, and the `_merge` wrapper used by `pick_cloud`). The saddle
is the highest connecting edge level between these two initial basins; its diamond marks
an actual edge endpoint at that intensity. Labels are projected from the actual 3D points.

These are deliberately broad synthetic split-centroid clouds (15 points per scan), as in
the revised basin overview. Height is effective intensity, including same-scan neighbours.
The input is normalised so the effective apex heights are 100 and 70. Smooth monotone
interpolation between specified extrema defines the teaching data; it is not a smoothing
operation performed by Peak3D. No points are shifted between the panels. Points below 2
arbitrary intensity units are omitted consistently from all views.

The default fractional rule is illustrated:

`persistence = lower_apex - saddle`

`threshold = max(0.5 * lower_apex, 3 * local_noise)`

Local noise is explicitly set to 1. The two apexes are 14 scans apart, with a specified
median FWHM of 10 scans. Thus neither the too-close branch (<0.8 median FWHM) nor the
far-apart branch (>=2 median FWHM) changes the decision. These values explain this pair
of examples, not all Peak3D merges. Other merge modes and settings exist. Surviving basins
are candidates and still require downstream feature validation.

The renderer asserts the two initial basins, expected one/two surviving basins, exact
apex and saddle heights (within floating-point tolerance), full centroid preservation,
and correct final memberships. `cloud_merging_verification.json` records the outcomes
and source hashes. “Correct” here means faithful execution of the stated merge criterion
on synthetic examples, not experimental evidence about the number of chemical compounds.

## Reproduce

From the repository root on Hive:

```bash
sbatch presentation/peak3d/render_merge.sbatch
```

The job uses the metabolomics group account on `high`, one task with two CPUs, 4 GB RAM,
and a 10-minute limit. Each job gets separate logs; caches use job-local temporary storage.
Run only one copy at a time because the output artifact names are shared.
