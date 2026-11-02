"""Assemble the benchmark report page (figures embedded as data URIs + tables from the markdown summaries).
usage: python bench/denoise/report_page.py OUT.html"""
import base64
import os
import re
import sys

FIG = "results/denoise/figures"


def img(path, alt):
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{alt}"><figcaption>{alt}</figcaption></figure>'


def md_table(path, heading_contains=None, first_only=True, drop_cols=()):
    """first markdown table in the file (or the one following a heading containing heading_contains) -> HTML"""
    lines = open(path).read().split("\n")
    start = 0
    if heading_contains:
        for i, ln in enumerate(lines):
            if ln.startswith("#") and heading_contains in ln:
                start = i; break
    rows = []
    in_tab = False
    for ln in lines[start:]:
        if ln.startswith("|"):
            in_tab = True
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            rows.append(cells)
        elif in_tab:
            break
    if not rows:
        return ""
    keep = [j for j, h in enumerate(rows[0]) if h not in drop_cols]
    html = ["<div class='tw'><table><thead><tr>" + "".join(f"<th>{rows[0][j]}</th>" for j in keep) + "</tr></thead><tbody>"]
    for r in rows[1:]:
        html.append("<tr>" + "".join(f"<td>{r[j] if j < len(r) else ''}</td>" for j in keep) + "</tr>")
    html.append("</tbody></table></div>")
    return "\n".join(html)


def main():
    out = sys.argv[1]
    own = "results/denoise/synthetic/own.md"
    parts = []
    parts.append("""<title>Neutral-Loss Denoising Benchmarks</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
/* layout: one reading column, figures full width, tables scroll in their own box */
:root { --bg:#fbfbf9; --fg:#1a1a18; --fg2:#5a5955; --rule:#e2e1dc; --accent:#2a78d6; --accent2:#6250d6; --card:#f2f2ee;
        --display:"IBM Plex Sans", system-ui, sans-serif; --body:"IBM Plex Sans", system-ui, sans-serif; --mono:"IBM Plex Mono", ui-monospace, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#191918; --fg:#eeede8; --fg2:#b8b7b0; --rule:#383835; --accent:#5598e7; --accent2:#9085e9; --card:#232322; color-scheme: dark } }
:root[data-theme="dark"] { --bg:#191918; --fg:#eeede8; --fg2:#b8b7b0; --rule:#383835; --accent:#5598e7; --accent2:#9085e9; --card:#232322; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--body); line-height: 1.5; padding-block: 32px 64px;
       padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
main { max-width: 1100px; margin: 0 auto; display: grid; gap: 40px; }
h1 { font-family: var(--display); font-size: 2rem; line-height: 1.15; margin: 0 0 6px; text-wrap: balance; }
h2 { font-size: 1.3rem; margin: 0 0 10px; padding-top: 10px; border-top: 1px solid var(--rule); text-wrap: balance; }
h3 { font-size: 1rem; margin: 18px 0 6px; }
p, li { max-width: 72ch; }
.lede { color: var(--fg2); font-size: 1.05rem; max-width: 80ch; }
.eyebrow { font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; color: var(--fg2); }
figure { margin: 14px 0; } figure img { width: 100%; height: auto; border: 1px solid var(--rule); border-radius: 4px; background: #fff; }
figcaption { font-size: .85rem; color: var(--fg2); margin-top: 6px; max-width: 90ch; }
.tw { overflow-x: auto; margin: 10px 0; } table { border-collapse: collapse; font-size: .85rem; font-variant-numeric: tabular-nums; min-width: 100%; }
th, td { padding: 5px 9px; border-bottom: 1px solid var(--rule); text-align: right; white-space: nowrap; } th:first-child, td:first-child { text-align: left; }
th { color: var(--fg2); font-weight: 600; }
.keys { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.key { background: var(--card); border-radius: 6px; padding: 12px 14px; } .key b { display: block; font-size: 1.5rem; font-family: var(--mono); }
.key span { font-size: .85rem; color: var(--fg2); }
code { font-family: var(--mono); font-size: .85em; }
.mode { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono&display=swap">
<main>
<header>
<div class="eyebrow">specdenoise · 2026-10-30</div>
<h1>Neutral-Loss Denoising Benchmarks</h1>
<p class="lede">Every fragment ion is judged by its neutral loss from the precursor: the loss must be a chemically plausible formula (Seven Golden Rules) that is a sub-formula of a plausible precursor ion and leaves a plausible fragment ion. Three ways to run it: <b>T1</b> with the compound's formula given, <b>T0</b> with only the precursor m/z (candidate formulas inferred from the accurate mass and ranked by the fragment intensity they explain, top-5 kept), and the <b>formula-free</b> loss rule (the loss only has to be some plausible formula). Benchmarked against Spectral Denoising (Kong et al., Nat Methods 2025) on that paper's own data.</p>
</header>
<section>
<h2>Summary</h2>
<div class="keys">
<div class="key"><b>97.6 %</b><span>of NIST23 [M+H]+ library intensity kept at 10 ppm / 3 mDa with the formula given (567k spectra); 98.4 % in negative mode</span></div>
<div class="key"><b>0.800</b><span>mean similarity to the 500 pmol reference on the dilution series with T0 (precursor m/z only), vs 0.805 published and 0.734 raw</span></div>
<div class="key"><b>0.821</b><span>the same with the formula given (T1): better than the published method on 35 % of spectra, worse on 6 %</span></div>
<div class="key"><b>30 vs 20</b><span>compounds still annotated at 0.02 pmol, T1 vs published (T0: 22, raw: 8)</span></div>
<div class="key"><b>46 ms</b><span>per spectrum for T0 on the dilution data (0.3 ms for T1); NIST23 T1 runs in 10 s on 32 cores</span></div>
</div>
</section>
<section>
<h2>1. NIST23: a library should lose nothing</h2>
<p>NIST23 consensus spectra are curated, so removal here is false removal. With the record's formula, the method keeps 94.9 % of positive-mode ions and 97.6 % of intensity at 10 ppm / 3 mDa. Two thirds of what it removes are ions whose permitted loss exists but lies 3 to 25 mDa off: HCD fragment m/z in NIST has a p99 error of 12 mDa. The remaining 1.6 % of ions (0.4 % of intensity) match no permitted loss; the strongest examples are background ions that cannot come from the precursor (a 105.0447 base peak in Rottlerin, which has no nitrogen).</p>
""")
    parts.append(img(f"{FIG}/fig_nist.png", "NIST23 [M+H]+: (a) entropy similarity between the denoised and the raw spectrum, formula given, 20,000 random spectra (median 1.000, mean 0.985); (b) fraction of library ions and intensity kept as a function of the fragment m/z tolerance, 567k spectra."))
    parts.append("<h3>Retention vs tolerance, positive mode (isotopologues rescued; +grid adds half a step for spectra on a 2- or 3-decimal m/z grid)</h3>")
    parts.append(md_table("results/denoise/nist23/t1_MH_pos.md", "Retention vs tolerance"))
    parts.append("<h3>Why ions are removed at 10 ppm / 3 mDa</h3>")
    parts.append(md_table("results/denoise/nist23/t1_MH_pos.md", "Why ions are removed"))
    parts.append("<h3>Formula inference from the precursor m/z alone (2,028 spectra, CHNOPS + F/Cl/Br/I)</h3>")
    parts.append(md_table("results/denoise/nist23/t0_MH_pos_s280_halo.md", "Formula inference"))
    parts.append("<h3>All adducts: how much does each mode change a NIST23 spectrum?</h3>")
    parts.append("<p>Every singly charged adduct with at least 1,000 NIST23 spectra, 400 random spectra each, scored by weighted entropy similarity of the denoised spectrum against the raw one. T1 reads the formula and adduct from the record; T0 infers the formula from the precursor m/z with C, H, N, O, P, S, F, Cl, Br, I (no Na or K, so for cationised precursors its candidates are stand-ins of the same mass); the formula-free rule only asks whether the loss has some plausible formula. Medians are 1.000 almost everywhere; the means are set by the tail of spectra with ions beyond the 3 mDa tolerance. In-source fragment 'adducts' such as [M+H-H2O]+ score lower in every mode, and radical-loss precursors ([M+H-Br]+, [M+H-NO2]+, [M+H-OH]+ ...) are odd-electron ions that T0's even-electron candidates do not fit.</p>")
    parts.append(img(f"{FIG}/fig_nist_adduct.png", "NIST23 by adduct: mean similarity of the denoised spectrum to the raw one (T1 formula given, T0 top-5, formula-free), one row per adduct sorted by NIST23 frequency, 400 spectra per adduct; (a) positive mode, 68 adducts; (b) negative mode, 20 adducts."))
    parts.append(md_table("results/denoise/nist23/by_adduct.md", drop_cols=("T1 median", "T0 median", "T0 median candidates")))
    parts.append("""</section>
<section>
<h2>2. Dilution series: do low-amount spectra move toward their reference?</h2>
<p>181 standards, 13 amounts from 500 to 0.02 pmol on a Q Exactive HF, three collision energies (Zenodo 14920689). Every spectrum from 200 pmol down is scored by weighted entropy similarity against the same compound's 500 pmol spectrum. The scoring reproduces the paper's own similarity column (r = 0.986, mean difference 0.02), and the published denoising result is the column stored in the dataset, scored the same way.</p>
""")
    parts.append(img(f"{FIG}/fig2_dilution.png", "Modelled on Fig. 2 of Kong et al. 2025 with our modes added. (a) raw similarity by amount; (b) mean similarity by amount; (c) gain over raw by amount, mean and 95 % CI; (d) cumulative distributions over all 6,091 spectra; (e) per-spectrum improvement, strip and box plots; (f) compounds annotated (similarity > 0.75) at each amount."))
    parts.append("<h3>Mean similarity by amount</h3>")
    parts.append(md_table("results/denoise/dilution/mapped.md", "Mean similarity", drop_cols=("electronic", "t0_top1", "t0_margin")))
    parts.append("<h3>Annotation rate (similarity > 0.75)</h3>")
    parts.append(md_table("results/denoise/dilution/mapped.md", "Annotation rate", drop_cols=("electronic", "t0_top1", "t0_margin")))
    parts.append("<h3>Ion level: raw ions ≥ 1 % of base below the precursor; true = within 0.02 Da of a reference ion</h3>")
    parts.append(md_table("results/denoise/dilution/mapped.md", "Ion-level: raw ions"))
    parts.append("""<p>Worse than raw happens rarely (T0: 0.3 % of spectra by more than 0.01; T1: 1.6 %). The largest T1 losses are chimeric references: rosmarinic acid [M+Na]+ carries nitrogen-containing base peaks from a co-isolated compound in both the sample and the reference spectrum, so removing them is chemically right but scores as worse.</p>
</section>
<section>
<h2>3. Synthetic noise</h2>
<p>The paper's nine files add noise ions to the dilution spectra (noise intensity 0.1, 0.2 or 0.5 of the base peak; 2, 10 or 100 noise ions per real ion) and score against the 500 pmol reference. The clean low-amount spectra already contain real noise, which is why every denoiser scores above the clean spectrum itself.</p>
""")
    parts.append(img(f"{FIG}/fig3_synthetic.png", "Modelled on Fig. 3 of Kong et al. 2025: density of similarity to the reference for the contaminated spectrum, the published result and our modes, nine noise conditions, 6,091 spectra each."))
    parts.append("<h3>Mean similarity, paper's files (our modes with the identical-intensity electronic step first)</h3>")
    parts.append(md_table("results/denoise/synthetic/paper.md", drop_cols=("t0_top1", "t0_margin")))
    parts.append("""<p>The paper's noise generator draws intensities from Poisson(λ)/100, so many noise ions share identical intensities and the electronic-noise rule alone removes 75 to 99.8 % of them. To see what the loss rules do by themselves, the 2× and 10× files were rerun with that step off:</p>""")
    parts.append("<h3>Loss rules alone (electronic step off): similarity, then noise ions removed % / true ions removed %</h3>")
    parts.append(md_table("results/denoise/synthetic/paper_noelec.md", drop_cols=("t0_top1", "t0_margin")))
    parts.append(md_table("results/denoise/synthetic/paper_noelec.md", "Ion level", drop_cols=("t0_top1", "t0_margin")))
    if os.path.exists(own):
        parts.append("""<h3>Our own noise: continuous lognormal intensities, uniform-m/z (electronic-like) and formula-mass (chemical-like) ions; loss rules alone</h3>
<p>Formula-mass noise sits where some plausible formula could be, so the formula-free rule cannot remove it by construction; only the precursor constraint (T0, T1) can.</p>""")
        parts.append(md_table(own))
        parts.append(md_table(own, "Ion level"))
    parts.append("""</section>
<section>
<h2>Method notes</h2>
<ul>
<li>Rules from Kind &amp; Fiehn 2007: element caps, heteroatom ratios, element-combination checks. LEWIS/SENIOR are not applied to losses or fragment ions (radical losses and odd-electron fragments are real); precursor candidates must be even-electron (half-integer DBE) and within the common ratio ranges.</li>
<li>Losses may be sums of several neutrals (NH3 + CH4), so hydrogens are bounded by the saturated hydrides of the loss's atoms rather than by H/C. Carbon-free losses and fragment ions are small inorganics (H2O, NH3, HCl, H3PO4, PO3−, NH4+ ...).</li>
<li>Halide → OH exchange products of aryl halides, frequent in NIST flow-injection spectra, are permitted (losses of P + H2O carrying a halogen). Isotopologues of kept ions are kept, including the precursor's own above the precursor m/z.</li>
<li>Tolerance: max(ppm, floor) on the fragment, plus half a grid step for library spectra on a 2- or 3-decimal m/z grid; T1 measures losses from the observed precursor m/z so a shared calibration offset cancels.</li>
<li>T0: candidate formulas decomposed from the precursor mass (3 mDa) over CHNOPS + F/Cl/Br/I with caps F6 Cl4 Br2 I2, scored by explained intensity with intensity-ordered branch-and-bound pruning; spectra with more than 20,000 candidates get the formula-free rule.</li>
<li>Code: <code>specdenoise/</code> (numpy + numba, 18 unit tests), benchmarks in <code>bench/denoise/</code>, tables in <code>results/denoise/</code>.</li>
</ul>
</section>
</main>""")
    open(out, "w").write("\n".join(parts))
    print(out, os.path.getsize(out) / 1e6, "MB")


if __name__ == "__main__":
    main()
