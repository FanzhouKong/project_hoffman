# builds the two HTML reports (yeast credentialed-truth misses, IDSL003 clean-TP misses) from the diagnostics in
# results/diag_yeast and results/diag_idsl_tp, with the gallery PNGs embedded. usage: miss_reports.py OUT_DIR
import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

STYLE = """
<style>
/* layout: one reading column, tables and figures full width inside it */
:root { --bg:#f7f6f2; --fg:#1d1c1a; --muted:#5e5a52; --line:#d9d4c8; --accent:#8a3b12; --soft:#efe9dc; --good:#2f6b3a; --bad:#9b2c2c;
  --display:"Source Serif 4", Georgia, "Times New Roman", serif; --body:"Source Sans 3", "Segoe UI", Helvetica, Arial, sans-serif;
  --mono:"JetBrains Mono", "SFMono-Regular", Consolas, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#171614; --fg:#ece8df; --muted:#aaa397; --line:#3a3731; --accent:#e08a5a; --soft:#242220; --good:#7fcf8c; --bad:#f08c8c; color-scheme: dark } }
:root[data-theme="dark"] { --bg:#171614; --fg:#ece8df; --muted:#aaa397; --line:#3a3731; --accent:#e08a5a; --soft:#242220; --good:#7fcf8c; --bad:#f08c8c; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 16px; line-height: 1.5; margin: 0; }
.wrap { max-width: 980px; margin: 0 auto; padding-block: 32px 64px; padding-inline: 16px; }
h1 { font-family: var(--display); font-size: 2rem; line-height: 1.15; margin: 0 0 6px; text-wrap: balance; }
h2 { font-family: var(--display); font-size: 1.4rem; margin: 40px 0 10px; text-wrap: balance; border-bottom: 1px solid var(--line); padding-bottom: 4px; }
h3 { font-size: 1.05rem; margin: 22px 0 6px; }
p, li { max-width: 72ch; }
.eyebrow { color: var(--muted); text-transform: uppercase; letter-spacing: .08em; font-size: .78rem; }
.lead { font-size: 1.08rem; max-width: 72ch; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 18px 0; }
.kpi { background: var(--soft); padding: 10px 12px; border-left: 3px solid var(--accent); }
.kpi b { display: block; font-size: 1.5rem; font-variant-numeric: tabular-nums; font-family: var(--display); }
.kpi span { color: var(--muted); font-size: .85rem; }
table { border-collapse: collapse; font-size: .9rem; font-variant-numeric: tabular-nums; width: 100%; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 600; font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; }
td.n { text-align: right; } th.n { text-align: right; }
.tbl { overflow-x: auto; margin: 10px 0 18px; }
.verdict { display: inline-block; padding: 1px 7px; border-radius: 3px; font-size: .78rem; font-weight: 600; }
.fix { background: color-mix(in srgb, var(--good) 18%, transparent); color: var(--good); }
.truth { background: color-mix(in srgb, var(--accent) 18%, transparent); color: var(--accent); }
.floor { background: color-mix(in srgb, var(--muted) 18%, transparent); color: var(--muted); }
code { font-family: var(--mono); font-size: .85em; background: var(--soft); padding: 1px 4px; }
details { margin: 14px 0; border: 1px solid var(--line); }
summary { cursor: pointer; padding: 8px 12px; font-weight: 600; background: var(--soft); }
details img { display: block; max-width: 100%; height: auto; }
.cap { color: var(--muted); font-size: .85rem; padding: 6px 12px; }
.rec li { margin-bottom: 8px; }
.small { color: var(--muted); font-size: .85rem; }
</style>
"""


def img(path):
    b = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" alt="">'


def gallery(path, title, caption, open_=False):
    return (f'<details{" open" if open_ else ""}><summary>{title}</summary><div class="cap">{caption}</div>'
            f'{img(path)}</details>')


YEAST = ROOT / "results/diag_yeast"
IDSL = ROOT / "results/diag_idsl_tp"

yeast_html = f"""<title>Yeast truth misses</title>{STYLE}
<div class="wrap">
<div class="eyebrow">peakbench · YEAST_12C credentialed truth · peak3d iteration 2 · 2026-10-05</div>
<h1>Why peak3d misses 31 % of the yeast credentialed truth</h1>
<p class="lead">All 3,992 truth features were traced through the pipeline in the three 12C injections: benchmark hit at 10 ppm / 0.1 min,
near miss, dropped by the presence rule, rejected per file (which gate), or no candidate at all. For every miss the centroids at the truth
position were followed to the basin they ended in. Two picker defects and one truth-set artefact explain most of the 1,241 misses.</p>
<div class="kpis">
<div class="kpi"><b>2,751 / 3,992</b><span>found, recall 0.689 (MassCube 0.724, Oct 5 picker 0.731)</span></div>
<div class="kpi"><b>568</b><span>misses whose trace was fused with a second ion 12–22 ppm away (46 %)</span></div>
<div class="kpi"><b>273</b><span>misses sitting on the flank of a peak we did find (22 %)</span></div>
<div class="kpi"><b>0.33</b><span>recall below 10<sup>4</sup> counts (1,181 truth features); 0.95 above 10<sup>5</sup></span></div>
</div>

<h2>1. Where the misses are</h2>
<p>Tier A (carbon count confirmed) recall is 0.953, tier B 0.645. Recall falls with intensity: 0.33 below 10<sup>4</sup> counts, 0.79 at
10<sup>4</sup>–3·10<sup>4</sup>, 0.88 at 3·10<sup>4</sup>–10<sup>5</sup>, 0.95–0.99 above. Isotopologue entries (M+1, M+2) are found less often (0.58–0.64) than
primaries (0.68) and adducts (0.79–0.89). The matched hits show no frame problem: median RT offset +0.2 s, median m/z offset 0.0 ppm.</p>
<div class="tbl"><table>
<tr><th>stage</th><th>class</th><th class="n">misses</th><th>what it is</th><th>verdict</th></tr>
<tr><td>rejected per file</td><td>m/z sd gate</td><td class="n">208</td><td>strong peaks (median 1.8·10<sup>4</sup>, S/N 15, 16 scans); the truth strand itself has 0.5 ppm sd, the basin 6.9 ppm because a second ion was linked in (97 %)</td><td><span class="verdict fix">picker: strand fusion</span></td></tr>
<tr><td></td><td>chromatogram check lowsnr</td><td class="n">142</td><td>narrow peaks (FWHM 3.2 s = 0.54 of the file median), 3D S/N 4.7 but smoothed prominence / noise 1.7; 27 % fused, 11 % low prominence</td><td><span class="verdict fix">picker: kernel width</span> <span class="verdict floor">weak</span></td></tr>
<tr><td></td><td>S/N, n_scans, FWHM gates</td><td class="n">177</td><td>apex 3–7·10<sup>3</sup> against a 2·10<sup>3</sup> cell noise: the bounds walk collapses to 1–2 scans (S/N 1.8, FWHM 0)</td><td><span class="verdict floor">at the 3D noise floor</span></td></tr>
<tr><td></td><td>score &lt; 0.5</td><td class="n">69</td><td>S/N 4.5, score 0.44</td><td><span class="verdict floor">weak</span></td></tr>
<tr><td></td><td>far_level, ridge, noapex</td><td class="n">60</td><td>pedestal peaks and bumps on broad ridges; 45 % low prominence</td><td><span class="verdict truth">mixed</span></td></tr>
<tr><td>presence rule</td><td>single detection, no confirmation</td><td class="n">121</td><td>detected in one injection (height 1.1·10<sup>4</sup>, S/N 5.2); the other injections show a raw local maximum with S/N ≥ 3 and ≥ 4 scans in 92 of 121, at 2–4 sd of the confirmation test</td><td><span class="verdict floor">confirmation threshold</span></td></tr>
<tr><td>near miss</td><td>RT off by 6–11 s, same m/z</td><td class="n">167</td><td>the truth RT sits on the flank or tail of a kept peak (dominant basin is a kept feature with its apex &gt; 6 s away in 52–64 %); 51–55 % are a second truth entry of a truth feature already found; raw prominence &lt; 0.2 in 35–45 %</td><td><span class="verdict truth">truth artefact</span></td></tr>
<tr><td></td><td>m/z off 13–15 ppm</td><td class="n">40</td><td>the basin holding the truth ion also holds a second strand (79–83 %); its m/z centre is pulled 9–11 ppm</td><td><span class="verdict fix">picker: strand fusion</span></td></tr>
<tr><td>no candidate</td><td>raw peak in ≥ 2 files</td><td class="n">151</td><td>68 % fused with a second strand (basin m/z 11.5 ppm off, rejected or kept at the wrong m/z); 33 % on the flank of a kept peak 12 s away</td><td><span class="verdict fix">strand fusion</span> <span class="verdict truth">flank</span></td></tr>
<tr><td></td><td>raw peak in 1 file / weak</td><td class="n">105</td><td>bumps on tails of large peaks (prominence &lt; 0.2 in 35–44 %, flank of a kept peak in 49–56 %)</td><td><span class="verdict truth">truth artefact</span></td></tr>
</table></div>

<h2>2. Root cause A: a neighbouring ion is linked into the trace (568 misses)</h2>
<p>The 3D linker joins centroids within <code>4 × sigma(I)</code> of each other, clipped to 5–30 ppm; on this Q Exactive file the clip evaluates to
<b>24 ppm</b> for points at the intensity floor. A weak strand 12–22 ppm away (a different ion, or for 19 % of the m/z-sd class an FT sideband at
4 % of the apex) is therefore pulled into the truth ion's basin. The fused basin then fails downstream: its m/z sd is 5.5–8.5 ppm against a gate of
about 3.7 ppm (208 strong peaks rejected), its m/z centre moves by 9–11 ppm so it no longer matches at 10 ppm (40 near misses, part of the 151
no-candidate peaks), and its S/N and width are measured on a mixture (2–9 points per scan in the S/N-gate class). MassCube finds 79 % of the m/z-sd
class. The truth strand is clean in every one of these cases (median 0.5 ppm sd, 90th percentile 1.4 ppm).</p>
<p><b>Fix.</b> Keep different ions apart at linking time or at measurement time: (1) a point may only join a basin whose intensity-weighted m/z centre is
within <code>z × sqrt(se² + sigma(I_point)²)</code>, and when two centroids in one scan compete for a basin the nearer one wins and the farther one starts its
own strand; (2) measure <code>mz_sd_ppm</code>, <code>mz_min</code>, <code>mz_max</code> and the m/z centre on the apex's strand only (intensity-weighted sd is a one-line
variant).</p>
<p><b>Expected gain:</b> most of the 208 m/z-sd rejections, the 40 m/z-off near misses and roughly 100 of the 151 fused no-candidate peaks, i.e. 300–350
truth features (+8 to +9 points of recall) and a matching reduction of same-ion duplicates.</p>
{gallery(YEAST / "gallery_rejected_gate_mz_sd.png", "Gallery: rejected by the m/z-sd gate (8 of 208)", "Black: EIC at 5 ppm; grey: 15 ppm. Orange band: ± 0.1 min around the truth RT. Red spans: rejected candidates with the stage and their m/z offset. These are clean, strong peaks.", True)}

<h2>3. Root cause B: truth entries on flanks, and doublets the half-valley rule merges (about 270 misses)</h2>
<p>The credentialing builder seeds candidates with <code>find_peaks(prominence=0)</code> on the smoothed EIC, so a bump on the tail or shoulder of a
bigger peak of the same ion becomes a truth feature whenever its U-13C partner shows the same bump. A prominence screen (apex above the deeper of the
two side minima within 3 FWHM, median over the three injections; <code>bench/diag/credtruth_prominence.py</code>) flags <b>107 of the 3,992 entries
(2.7 %)</b>, 40 of them weaker siblings of a stronger entry of the same ion; peak3d finds none of the 107, MassCube 11 %. Removing them lifts the
current recall from 0.689 to 0.708 and the SZ22 list loses 3 entries. The screen is written to <code>data/truth/YEAST/credentialed_prominence.tsv</code>
(column <code>keep</code>); the benchmark still scores the full list.</p>
<p>The larger group of flank misses is different: 167 near misses whose truth RT is 6–11 s from a kept apex of the same ion, with a raw prominence of
0.24–0.38, and 273 misses in total whose centroids end in a kept feature with its apex more than 6 s away. Half of them are a second truth entry beside
a found one. These are second components of doublets and shoulders 1–2 widths apart that the half-valley rule (a valley below 50 % of the lower apex, 20 %
on the smoothed trace when the apexes are 2 widths apart) merges into one feature. Relaxing that rule re-admits tail bumps everywhere; it stays.</p>
{gallery(YEAST / "gallery_near_rt_off.png", "Gallery: near misses, RT off by 6–11 s (8 of 98)", "Green spans: kept features. The truth RT (orange) sits on the rise or the tail of a peak we report.")}
{gallery(YEAST / "gallery_nocand_raw_peak.png", "Gallery: no candidate although the raw EIC shows signal (8 of 151)", "Several are bumps on the far tail of a saturating peak at the same m/z (rows 1, 4, 6, 8); the others are fused with a strand 20–30 ppm away.")}

<h2>4. Root cause C: weak peaks at the 3D noise floor (about 450 misses)</h2>
<p>In busy regions of these yeast files the post-merge noise surface is 1.5–2.5·10<sup>3</sup>; a credentialed peak of 3–7·10<sup>3</sup> counts is then S/N 1.5–3.
Its bounds walk stops in the first scan (S/N gate 91, n_scans 48, FWHM 38), or it survives the gates and fails the chromatogram check (142: the
0.5-FWHM smoothing kernel removes 30–40 % of the height of a peak half the median width) or the score (69). Another 121 are detected in one injection
only and the other two injections show the peak at 2–4 sd instead of the required 3 sd. MassCube finds 27–59 % of these classes.</p>
<p><b>Options, with the measured costs:</b></p>
<ul class="rec">
<li><b>Chromatogram check kernel scaled to the candidate</b> (kernel FWHM = min(0.5 × median, 0.5 × own FWHM)): aims at the 142 narrow lowsnr rejections; cost not yet measured, likely small because the dup and noapex tests are unchanged.</li>
<li><b>Confirmation rule</b> "prominence ≥ 3 × max(residual, file floor) and ≥ 4 contiguous scans" (confirm_rules.py): rescues 93 of 118 presence-dropped truth groups but re-admits 1,543 of the 4,163 dropped groups on the three yeast files (37 %); truth share of the re-admitted 6 %.</li>
<li><b>Noise reference in busy cells</b>: the only lever for the 177 bounds-collapse cases. A per-trace residual floor would be selective but is the local S/N you set aside on Oct 5. Not recommended without a new idea.</li>
</ul>
{gallery(YEAST / "gallery_rejected_eic_lowsnr.png", "Gallery: rejected by the chromatogram check, lowsnr (8 of 142)", "Narrow 3–6 scan peaks at 4–13·10³ counts; two rows are tail bumps (truth artefacts).")}
{gallery(YEAST / "gallery_presence_det1_conf0.png", "Gallery: dropped by the presence rule (8 of 121)", "Green: the single detection. The other injections show the peak weakly or show a ridge / tail (rows 1, 3, 8).")}

<h2>5. Fix applied (2026-10-05)</h2>
<p>Three changes in the picker, all defaults now (<code>Params.same_scan_k</code> 1.0 with <code>same_scan_floor_ppm</code> 5, <code>rep_raw</code>,
<code>far_level_max</code> 0.5): same-scan split partners are judged at the pair tolerance or 5 ppm, whichever is larger, instead of twice the pair
tolerance, so a weak centroid's sigma no longer reaches a second ion 10–20 ppm away; link ties and each scan's representative point go to the dominant raw
centroid, and the feature's m/z, m/z sd and m/z range are taken from those representatives inside the bounds; the far-level ceiling is 0.5. A regression
test reproduces the fusion on synthetic data (<code>test_interfering_ion_linked_as_split_partner_does_not_shift_mz</code>); 134 tests pass.</p>
<div class="tbl"><table>
<tr><th>per file, before any rerun (bench/diag/fix_eval2.py)</th><th class="n">iteration 2</th><th class="n">fixed</th></tr>
<tr><td>YEAST_12C truth found in any injection / per-injection recall</td><td class="n">2,906 (0.728) / 0.580</td><td class="n">3,246 (0.813) / 0.719</td></tr>
<tr><td>YEAST_12C features per injection / same-ion split pairs</td><td class="n">22,170 / 53</td><td class="n">28,627 / 84</td></tr>
<tr><td>IDSL003 clean F1 (TP, FP) at 10 ppm; F1 at 20 ppm</td><td class="n">0.895 (1,390, 40); 0.900</td><td class="n">0.903 (1,428, 59); 0.902</td></tr>
<tr><td>HZV029_cert found / per-injection recall / features</td><td class="n">402 / 0.980 / 9,926</td><td class="n">402 / 0.982 / 10,237</td></tr>
<tr><td>YEAST_NEG, SZ22_12C, LI2018 truth</td><td class="n">309, 560, 826</td><td class="n">309, 562, 826</td></tr>
<tr><th>full pipeline, scratch run of three runs (alignment, presence rule)</th><th class="n">iteration 2</th><th class="n">fixed</th></tr>
<tr><td>YEAST credentialed recall (tier A / B)</td><td class="n">0.689 (0.953 / 0.645)</td><td class="n">0.769 (0.975 / 0.734)</td></tr>
<tr><td>YEAST 12C features / truth share of features</td><td class="n">27,710 / 0.085</td><td class="n">33,564 / 0.079</td></tr>
<tr><td>HZV029_cert found / features</td><td class="n">401 / 11,629</td><td class="n">401 / 11,861</td></tr>
<tr><td>replicate-junk share YEAST_12C / HZV029_cert (good features)</td><td class="n">1.2 % / 1.7 % (61.5k / 26.7k)</td><td class="n">1.7 % / 1.7 % (78.5k / 27.4k)</td></tr>
</table></div>
<p>Not changed: the chromatogram check (its lowsnr rejections are mostly bumps with low 1D prominence, not a kernel-width problem), the presence
confirmation (every relaxation measured re-admits 25–60 % of the dropped groups) and <code>min_snr</code>. Next step: the benchmark rerun of peak3d on
all eleven runs, then the credentialing fractions and the replicate junk on all four replicate runs.</p>

<h2>6. Fix applied (2026-10-05)</h2>
<p>Same picker changes as on the yeast page (split partners at the pair tolerance or 5 ppm, dominant-centroid representatives, far-level ceiling 0.5).
On the cleaned labels, per file: F1 0.895 → 0.903, TP 1,390 → 1,428, FP 40 → 59, precision 0.972 → 0.960, recall 0.829 → 0.852; at 20 ppm F1 0.900 → 0.902.
The far-level change alone accounts for +26 TP and +19 FP (variant rep_f5 in <code>results/diag_fix/fix_eval4_*.log</code>), the split rule for the rest
at no FP cost. HZV029_cert stays at 402 of 402 per file, YEAST_NEG at 309, LI2018 at 826. The label-precision misses (root cause A) and the floor are
untouched by design.</p>

<h2>7. Recommendation</h2>
<ol class="rec">
<li><span class="verdict fix">done</span> Strand-aware split rule and dominant-centroid representatives in the picker (root cause A), validated per file on six truth runs and through the full pipeline on three; the benchmark rerun is pending.</li>
<li><span class="verdict truth">decide</span> Whether to score on the prominence-screened truth (107 YEAST and 3 SZ22 entries removed; the screen file exists, the benchmark still uses the full list).</li>
<li><span class="verdict floor">measure</span> Width-scaled chromatogram kernel; decide on the confirmation rule with the 37 % re-admission number in view (root cause C).</li>
</ol>
<p class="small">Files: results/diag_yeast/yeast_misses.tsv (one row per missed truth feature with stage, gate values, raw EIC evidence, other tools),
points_trace.tsv (strands and basins per miss), summary.log, points_trace.log, gallery_*.png. Scripts: bench/diag/yeast_truth_misses.py,
miss_points_trace.py, miss_common.py. Truth: data/truth/YEAST/credentialed.tsv (3,992; tier A 571 / B 3,421; decoy false rate 1.4 % / 3.0 %).</p>
</div>
"""

idsl_html = f"""<title>IDSL003 TP misses</title>{STYLE}
<div class="wrap">
<div class="eyebrow">peakbench · IDSL003 cleaned labels · peak3d iteration 2 · 2026-10-05</div>
<h1>Why peak3d misses 289 of the 1,676 validated IDSL003 peaks</h1>
<p class="lead">Every cleaned TP label passed a blind visual check (a peak apex inside ± 0.1 min at 10 ppm). The 289 that peak3d does not report at
10 ppm / 0.1 min were traced to a kept feature nearby, a rejected candidate and its gate, or the basin their centroids ended in. Every one of them
is reported by at least one published tool (IPA 251). They are weaker than the hits (median apex 2.0·10<sup>3</sup> vs 8.7·10<sup>3</sup>, S/N 8 vs 35), but
two thirds are lost for reasons that have nothing to do with intensity.</p>
<div class="kpis">
<div class="kpi"><b>1,387 / 1,676</b><span>found, recall 0.828, precision 0.971 (IPA published 0.827 / 0.792)</span></div>
<div class="kpi"><b>69</b><span>found but 10–15 ppm from the label's single centroid (24 %)</span></div>
<div class="kpi"><b>68</b><span>real peaks (S/N 18) rejected by the far-level / ridge gates (24 %)</span></div>
<div class="kpi"><b>190</b><span>misses whose basin fused a second ion 12–20 ppm away (66 %)</span></div>
</div>

<h2>1. Where the misses are</h2>
<p>Recall by raw apex height at the label: 0.29 below 10<sup>3</sup> counts (91 labels), 0.71 at 1–3·10<sup>3</sup>, 0.86 at 3–10·10<sup>3</sup>, 0.94 at 10<sup>4</sup>–10<sup>5</sup>,
0.98 above 10<sup>5</sup>. Matched hits are centred (median RT offset −0.2 s, m/z offset +0.2 ppm).</p>
<div class="tbl"><table>
<tr><th>stage</th><th>class</th><th class="n">misses</th><th>what it is</th><th>IPA finds</th><th>verdict</th></tr>
<tr><td>near miss</td><td>m/z off 10–15 ppm, same RT</td><td class="n">69</td><td>our kept feature covers the peak; the label m/z lies inside the feature's own m/z range in 59 of 69; within-trace m/z sd on this TOF file is 5–7 ppm</td><td>0.88</td><td><span class="verdict truth">label precision</span></td></tr>
<tr><td></td><td>RT off 4–10 s</td><td class="n">23</td><td>shoulders and adjacent peaks whose basin apex is the bigger neighbour</td><td>0.7</td><td><span class="verdict fix">merge</span></td></tr>
<tr><td>rejected</td><td>far_level (33) and far_level + ridge (35)</td><td class="n">68</td><td>peaks of S/N 18 on a same-ion pedestal (30–45 % of the apex), beside a bigger same-ion peak 10–30 s away, or between two plateaus in the void region; raw prominence 0.72</td><td>0.85</td><td><span class="verdict fix">gate too strict</span></td></tr>
<tr><td></td><td>S/N gate</td><td class="n">33</td><td>apex 730 counts, cell noise 250: S/N 2.2–2.8; 91 % fused with a second strand</td><td>1.00</td><td><span class="verdict floor">floor</span></td></tr>
<tr><td></td><td>score &lt; 0.5</td><td class="n">19</td><td>1–1.5·10<sup>3</sup> counts, S/N 3–4, clean shape</td><td>0.89</td><td><span class="verdict floor">floor</span></td></tr>
<tr><td></td><td>lowsnr, FWHM, m/z sd, fragment, n_scans, ridge</td><td class="n">26</td><td>spikes and fused basins</td><td>0.8</td><td><span class="verdict floor">mixed</span></td></tr>
<tr><td>no candidate / neighbour only</td><td></td><td class="n">51</td><td>the centroids sit in a basin whose m/z centre is 15–25 ppm off because it fused 2–4 strands (84–100 %); rejected by the m/z-sd or S/N gate, or kept at the wrong m/z</td><td>0.85</td><td><span class="verdict fix">picker: strand fusion</span></td></tr>
</table></div>

<h2>2. Root cause A: the label's m/z is one centroid, ours is the trace (69 misses)</h2>
<p>IDSL003 labels are exact centroids of file 003. On this Q-TOF the centroids of one ion scatter with a within-trace sd of 5–7 ppm (the error model
gives 6 ppm at 10<sup>3</sup> counts), so a single apex centroid is routinely 10–15 ppm from the intensity-weighted m/z of the trace. In 59 of 69 cases the
label m/z lies inside our feature's own m/z range; at 15 ppm 59 of them match, at 20 ppm all 69 do (IDSL003 F1 0.901 at 20 ppm / 0.1 min). IPA
"finds" 88 % because it reports a centroid of the same kind as the label. Nothing to fix in the picker; the 10 ppm matching rule is simply tight for
TOF labels, and that is worth a sentence in the benchmark write-up.</p>
{gallery(IDSL / "gallery_near_mz_off.png", "Gallery: found, but 10–15 ppm from the label (12 of 58)", "Black: EIC at 10 ppm; grey: 30 ppm. Green spans: kept features with their m/z offset from the label. Row 1 also shows a same-ion split (+11 and +19 ppm).", True)}

<h2>3. Root cause B: the far-level and ridge gates reject resolved peaks (68 misses)</h2>
<p>These are not weak: median height 5.7·10<sup>3</sup>, S/N 17.6, 15 scans. Three morphologies recur in the gallery: a peak on a continuous same-ion
pedestal at 30–45 % of the apex (the far-level test, added in iteration 2 with a 0.3 ceiling, rejects it although its relative prominence is 0.7);
a small peak 10–30 s beside a bigger peak of the same ion, which the 2–6 width side windows see as "the ion is still there"; and a peak in the void region
(0.6–0.9 min, 14 cases) between two plateaus, where the trace returns to zero right beside the peak and rises again further out. 33 of the 68 fail only
the far-level test (far_level 0.32–0.45); the other 35 also fail the ridge or background test. On IDSL003 the iteration-2 far-level gate removed 33 TP for
27 FP; on the Orbitrap truth sets it cost 0 / 0 / 0 / 4 compounds for 3–5 % fewer features, which is why it was adopted.</p>
<p><b>Fix.</b> Two changes, both cheap to measure: raise <code>far_level_max</code> to 0.5 (or switch the gate off and let the localisation term of the score carry it),
and let the ridge / background / far-level tests short-circuit when the trace falls below 10 % of the apex within 2 widths on both sides, since a peak
that returns to baseline next to itself is resolved whatever happens 6 widths out. Expected gain on IDSL003: 30–60 TP; expected cost on the other
runs: the 3–5 % of features the gate removed, which were mostly unreplicated.</p>
{gallery(IDSL / "gallery_rejected_gate_far_level.png", "Gallery: rejected by the far-level / ridge gates (12 of 68)", "Red spans: the rejected candidates. Blue dotted and dashed: cell noise and 3 × cell noise.", True)}

<h2>4. Root cause C: a neighbouring ion is linked into the basin (190 of 289 misses, decisive for about 60)</h2>
<p>The linker's tolerance for weak points is <code>4 × sigma(I)</code> clipped to 30 ppm, and on this file it is 30 ppm. Two thirds of the missed labels have a
second m/z strand 12–20 ppm away inside the basin that holds their centroids (median other-ion apex 40 % of the truth apex). For the 51 no-candidate
and neighbour-only misses the fused basin's m/z centre is 15–25 ppm off and its m/z sd 14 ppm, so it is rejected by the m/z-sd gate or kept at an m/z
that no longer matches the label; the 33 S/N-gate rejections carry 2.5 points per scan for the same reason. This is the same defect found on yeast
(there it costs 208 strong peaks); the fix is the same: strand-aware linking, and m/z statistics taken on the apex's strand only.</p>
{gallery(IDSL / "gallery_nocand_raw_peak.png", "Gallery: no candidate within 10 ppm although the EIC shows a peak (12 of 20)", "The candidates that exist (red) are 11–26 ppm off: basins fused across strands.")}

<h2>5. The floor (about 60 misses)</h2>
<p>33 labels have an apex of about 730 counts against a cell noise of 250 (S/N 2.2–2.8 on our definition), 19 have 1–1.5·10<sup>3</sup> counts and a score
of 0.3–0.5, and a few are 1–2 scan spikes. They look like small peaks and the published IPA reports all of them, with a precision of 0.79 against our
0.97 (363 vs 41 false positives on the cleaned labels). Rescuing them means lowering <code>min_snr</code> or <code>min_score</code>, and at this operating point
a filter must remove more than 2 false positives per true positive to help F1; that trade was measured on Oct 5 and does not pay. Leave them.</p>
{gallery(IDSL / "gallery_rejected_gate_snr.png", "Gallery: rejected by the S/N gate (12 of 33)", "Apex 400–1,100 counts; the 3 × cell-noise line (dashed) is above most apexes.")}

<h2>6. Fix applied (2026-10-05)</h2>
<p>Same picker changes as on the yeast page (split partners at the pair tolerance or 5 ppm, dominant-centroid representatives, far-level ceiling 0.5).
On the cleaned labels, per file: F1 0.895 → 0.903, TP 1,390 → 1,428, FP 40 → 59, precision 0.972 → 0.960, recall 0.829 → 0.852; at 20 ppm F1 0.900 → 0.902.
The far-level change alone accounts for +26 TP and +19 FP (variant rep_f5 in <code>results/diag_fix/fix_eval4_*.log</code>), the split rule for the rest
at no FP cost. HZV029_cert stays at 402 of 402 per file, YEAST_NEG at 309, LI2018 at 826. The label-precision misses (root cause A) and the floor are
untouched by design.</p>

<h2>7. Recommendation</h2>
<ol class="rec">
<li><span class="verdict fix">done</span> Strand-aware split rule and dominant-centroid representatives (root cause C, shared with yeast root cause A).</li>
<li><span class="verdict fix">done</span> far_level_max 0.5 (root cause B); the return-to-baseline short-circuit for the ridge test was not added (it would also bypass the test for flickering ions).</li>
<li><span class="verdict truth">report</span> State in the benchmark that IDSL003 is matched at 10 ppm against single-centroid TOF labels; show the 20 ppm column next to it (F1 0.901).</li>
<li><span class="verdict floor">leave</span> The sub-1,500-count labels.</li>
</ol>
<p class="small">Files: results/diag_idsl_tp/idsl_tp_misses.tsv (one row per missed label with stage, gate values, raw EIC evidence, published and in-house tools),
idsl_tp_all.tsv (raw EIC descriptors of all 1,676 TP labels), points_trace.tsv, summary.log, points_trace.log, gallery_*.png. Scripts: bench/diag/idsl_tp_misses.py,
miss_points_trace.py, miss_common.py. Labels: data/truth/IDSL003/labels_clean.csv.</p>
</div>
"""

(OUT / "yeast_truth_misses.html").write_text(yeast_html)
(OUT / "idsl_tp_misses.html").write_text(idsl_html)
(YEAST / "REPORT.html").write_text(yeast_html)
(IDSL / "REPORT.html").write_text(idsl_html)
print(OUT / "yeast_truth_misses.html", len(yeast_html) // 1024, "KB;", OUT / "idsl_tp_misses.html", len(idsl_html) // 1024, "KB")
