# Dark-sky analogy for local cell noise estimation

Generated with the built-in image generation tool. Conceptual illustration; no experimental data are plotted. The enlarged cells are schematic examples rather than literal crops.

## Suggested caption

Localized signal occupies a small fraction of the measurement space. In telescope images, background samples between stars support local sky-background estimation. Analogously, peak3d estimates a local noise floor from isolated centroids within retention-time × m/z cells, reducing the influence of coherent ion features.

## Method details

Here, “cell” means an RT × m/z region, not a biological cell. In the current code, candidate noise centroids belong to basins spanning at most two scans. Their median intensity is computed per log-m/z band and RT block; the README describes a run-wide band-median cap. The figure omits caps and sparse-cell fallbacks for clarity.

Sky background level and background noise dispersion are distinct quantities. The figure illustrates the shared strategy of estimating background from samples outside localized signal; it does not equate the peak3d median-intensity floor with astronomical background RMS or detector dark current, nor establish a historical attribution.

Implementation: `peak3d/kernels.py`, `noise_cell_medians`; method summary: `peak3d/README.md`.

Astronomy reference: https://photutils.readthedocs.io/en/stable/user_guide/background.html

## Generation prompt

+Use case: scientific-educational
Asset type: polished conceptual scientific figure for a peak3d LC-MS methods presentation. Landscape 16:9, white background, crisp readable typography, restrained elegant editorial scientific design.
Primary request: illustrate the analogy between estimating telescope sky background and estimating a local LC-MS cell noise floor. Stars occupy only small regions of an otherwise dark sky; similarly real LC-MS features are localized in a mostly background measurement space. This is a conceptual schematic, not real experimental data.
Title text: "From dark sky to local noise"
Two equal columns with clear parallel structure, separated by generous whitespace and a small horizontal analogy arrow.
LEFT heading: "Telescope image". Large rectangular realistic dark navy star-field crop with subtle sensor grain and approximately 12 sparse small bright stars, plenty of empty dark sky, no nebula, no landscape, no telescope hardware. Overlay a thin muted grid. Highlight one grid cell with a teal border. Bright stars should occupy a very small fraction of the field. Small orange rings around a few stars and a callout "Localized stars". Teal callout to empty sky: "Background between sources". Below, an enlarged local-cell inset with bright sources excluded by subtle orange outline masks and faint background pixels remaining. Downward arrow to clean label "Estimate local sky background".
RIGHT heading: "LC–MS intensity map". Matching rectangular dark navy heatmap, horizontal axis "Retention time", vertical axis "m/z". Sparse bright warm yellow localized peaks stretched horizontally in RT and narrow vertically in m/z, many faint isolated teal speckles, subtle background. Same thin grid. Highlight one rectangular cell with teal border. Orange callout "Localized ion features". Teal callout "Isolated noise centroids". Below, enlarged local-cell inset showing the distinction between organized horizontal bright traces and scattered faint isolated points; orange outline around signal traces, teal faint isolated dots selected. Downward arrow to clean label "Estimate local cell noise floor".
A simple short line beneath the right result, in smaller but readable type: "Median intensity of isolated centroids"
Bottom unifying sentence exactly: "Sparse signal reveals the background around it."
Small footer exactly: "Conceptual analogy • Sky background level and noise dispersion are distinct quantities"
Visual constraints: Do NOT show biological cells. A cell means an RT × m/z grid region. Do not show molecular structures. Do not label the noise floor as a standard deviation, do not invent equations or data values, no Gaussian distribution charts. Avoid claiming dark-current calibration or a dark exposure. Keep all text concise as specified. Consistent orange for true signal, teal for background samples, charcoal labels. Prioritize image panels and visual explanation; all typography accurate and readable.
