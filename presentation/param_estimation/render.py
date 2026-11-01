"""Slide figures: how Mosa (peak3d) reads its parameters off each file. Data from extract.py (real HZV029 injection,
one SZ22 injection for the scan gains, the params sidecars of 113 files of the current default run).
Run on a compute node: envs/peak3d/bin/python presentation/param_estimation/render.py"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
OUT = HERE
PAN = HERE / "panels"; PAN.mkdir(exist_ok=True)
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_T, ORANGE_T, GRAY_T = "#eaf2fc", "#fdeee7", "#f4f3f0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 13, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "axes.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK2,
    "ytick.color": INK2, "xtick.labelsize": 12, "ytick.labelsize": 12, "text.color": INK, "axes.titlecolor": INK,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False, "legend.fontsize": 12,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "pdf.fonttype": 42})
PT = 1 / 72
PANEL = (460 * PT, 366 * PT)   # slide panel size used by the other talk figures


def kfmt_f(x, _):
    a = abs(x); sgn = "−" if x < 0 else ""
    if a >= 1e6: return f"{sgn}{a / 1e6:g}M"
    if a >= 1e3: return f"{sgn}{a / 1e3:g}k"
    return f"{sgn}{a:g}"


def counts(x):
    """counts for prose: 2-3 significant figures with k / M"""
    if x >= 1e6: return f"{x / 1e6:.2f}M"
    if x >= 1e3: return f"{x / 1e3:.1f}k"
    return f"{x:.0f}"


kfmt = FuncFormatter(kfmt_f)
D = np.load(HERE / "estimation_demo.npz")
META = json.loads((HERE / "demo_params.json").read_text())
P = META["main_params"]
CAT = []


def save(fig, name, title, caption):
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    CAT.append((name, title, caption))
    print("saved", name, flush=True)


def panel(name, draw, size=PANEL):
    """untitled single panel at slide size (PNG 300 dpi + PDF) in panels/"""
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0.17, 0.17, 0.78, 0.78])
    draw(ax)
    ax.set_title("")
    fig.savefig(PAN / f"{name}.png", dpi=300)
    fig.savefig(PAN / f"{name}.pdf")
    plt.close(fig)


def ygrid(ax, axis="y"):
    ax.grid(True, axis=axis, zorder=0); ax.set_axisbelow(True)


sig = lambda I: np.sqrt(P["sig_a"] ** 2 + P["sig_b"] ** 2 / np.asarray(I, float))
fl, c_, r_ = P["floor"], P["noise_c"], P["noise_r"]
noise = lambda I: np.sqrt(fl ** 2 + c_ * I + (r_ * I) ** 2)
fwhm_s, dt_s = P["fwhm_med"] * 60, P["dt"] * 60
FILE = Path(META["main_file"]).name

# =====================================================================================================
# P1 overview: file -> seeds -> measured -> derived
fig = plt.figure(figsize=(19, 7.2))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 19); ax.set_ylim(0, 7.2); ax.axis("off")


def box(x, y, w, h, fc, ec, title, lines, tsize=13.5, lsize=12, mono=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", fc=fc, ec=ec, lw=1.6))
    ax.text(x + 0.25, y + h - 0.32, title, fontsize=tsize, fontweight="bold", va="top", ha="left", color=INK)
    ax.text(x + 0.25, y + h - 0.85, "\n".join(lines), fontsize=lsize, va="top", ha="left", color=INK2, linespacing=1.55,
            family="DejaVu Sans Mono" if mono else "DejaVu Sans")


def arrow(x0, y0, x1, y1, color=INK2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18, lw=1.8, color=color))


ax.text(0.4, 6.75, "Every setting is read off the file itself: no per-dataset tuning", fontsize=19, fontweight="bold", va="center")
ax.text(0.4, 6.3, f"Worked on one real injection ({FILE}): {META['n_centroids']:,} centroids in {META['n_scans']} MS1 scans. "
                  "The same code runs on every file; the sidecar (.params.json) records what the file said.", fontsize=12.5, color=INK2, va="center")
box(0.4, 1.3, 3.0, 4.4, GRAY_T, AXIS, "1  The file",
    ["MS1 centroids only", "(RT, m/z, intensity)", "", "scan times", f"→ interval {dt_s:.2f} s", "", "all intensities",
     f"→ floor = 1st percentile", f"   {fl:,.0f} counts"])
box(3.9, 1.3, 3.4, 4.4, BLUE_T, BLUE, "2  Seeds: the 2,000\n    strongest ions",
    ["", "strongest first, each ≥ 20 ppm", "from every earlier seed", "", "each seed: its EIC around the",
     "apex (± 20 ppm, 3-point mean)", "", "that is the only input to", "steps 3 and 4"])
box(7.8, 1.3, 4.9, 4.4, BLUE_T, BLUE, "3  Measured from the seeds",
    [f"peak width: median FWHM {fwhm_s:.2f} s", f"   = {P['fwhm_scans']:.1f} scans  ({P['n_fwhm_seeds']:,} seeds measured)",
     "", "m/z error vs intensity:",
     f"   σ(I) = √(a² + b²/I),  a {P['sig_a']:.2f} ppm, b {P['sig_b']:.0f}", f"   ({P['n_sigma_points']:,} centroids of the seed traces)",
     "", "scan-to-scan noise vs intensity:", f"   σ(I)² = floor² + c·I + r²·I²,  c {c_:,.0f}, r {r_:.3f}",
     f"   ({P['n_noise_pts']:,} residuals, {P['n_noise_bins']} bins)", "", "per-scan gain shared by all ions",
     f"   (here: none beyond the noise, sd {P['gain_sd']:.3f})"], lsize=11.5)
box(13.2, 1.3, 5.4, 4.4, ORANGE_T, ORANGE, "4  Settings the picker runs with",
    [f"link centroids across K = {P['K']} scans", "   (K = half the width in scans, 1–3)", "",
     f"m/z tolerance cap {P['tol_max_ppm']:.1f} ppm", "   (4 σ at the floor intensity, 5–30 ppm)",
     "   per pair: the two centroids' own σ(I)", "", f"a peak needs ≥ {P['min_scans']} scans",
     "   (half the width in scans, 3–5)", f"smoothing kernel {P['n_smooth']} pass(es) of [1 2 1]",
     "   (half the median width)", "", "every S/N and valley test in units of the",
     "   measured noise, not a fixed count"], lsize=11.5)
for x0, x1 in [(3.4, 3.9), (7.3, 7.8), (12.7, 13.2)]:
    arrow(x0, 3.5, x1, 3.5)
ax.text(0.4, 0.7, "Fixed in the code and the same for every file: the ratios (half the width, 4 σ, 3 σ), the seed count (2,000) and "
                  "the seed EIC width (20 ppm).  Everything with a unit (seconds, ppm, counts) comes from the file.",
        fontsize=12, color=INK2, va="center", style="italic")
save(fig, "P1_overview", "Every setting is read off the file",
     "Flow of the per-file estimation: the file's scan times and intensity floor, the 2,000 strongest distinct ions as "
     "seeds, the four quantities measured on the seed traces, and the settings derived from them. Numbers from one real "
     f"HZV029 injection ({FILE}).")

# =====================================================================================================
# P2 seeds and the peak width
ex_rt, ex_raw, ex_sm = D["ex_rt"], D["ex_raw"], D["ex_smooth"]
k = int(np.argmax(ex_sm)); half = 0.5 * ex_sm[k]
i = k
while i - 1 >= 0 and ex_sm[i - 1] >= half: i -= 1
tl = ex_rt[i - 1] + (ex_rt[i] - ex_rt[i - 1]) * (half - ex_sm[i - 1]) / (ex_sm[i] - ex_sm[i - 1])
j = k
while j + 1 < len(ex_sm) and ex_sm[j + 1] >= half: j += 1
tr = ex_rt[j] + (ex_rt[j + 1] - ex_rt[j]) * (ex_sm[j] - half) / (ex_sm[j] - ex_sm[j + 1])
assert np.isclose(tr - tl, float(D["ex_fwhm"]), rtol=1e-6)
fw = D["seed_fwhm"]; fw = fw[np.isfinite(fw)] * 60


def p2a(ax):
    s = D["seed_inten"]
    ax.scatter(D["seed_rt"], D["seed_mz"], s=6 + 40 * (np.log10(s) - np.log10(s.min())) / (np.log10(s.max()) - np.log10(s.min())),
               color=BLUE, alpha=0.55, lw=0)
    ax.scatter([float(ex_rt[k])], [float(D["ex_mz"])], s=140, marker="o", fc="none", ec=ORANGE, lw=2.2)
    ax.set(xlabel="RT (min)", ylabel="m/z", title="(a) The 2,000 strongest ions, ≥ 20 ppm apart")
    ax.text(1.0, -0.2, f"dot size: apex intensity ({s.min() / 1e3:,.0f}k to {s.max() / 1e6:,.0f}M)", transform=ax.transAxes,
            fontsize=11, color=INK2, va="top", ha="right")


def p2b(ax):
    sel = slice(max(0, i - 14), min(len(ex_rt), j + 15))
    ax.plot(ex_rt[sel], ex_raw[sel], "o-", color=AXIS, ms=4.5, lw=1.2, mfc=MUTED, mec="none", label="raw EIC (± 20 ppm)")
    ax.plot(ex_rt[sel], ex_sm[sel], color=BLUE, lw=2.2, label="3-point mean")
    ax.hlines(half, tl, tr, color=ORANGE, lw=2.6)
    ax.plot([tl, tr], [half, half], "|", color=ORANGE, ms=14, mew=2.6)
    ax.axhline(ex_sm[k], color=GRID, lw=1, ls=(0, (3, 3)))
    ax.text(tr + 0.003, half, f"FWHM {float(D['ex_fwhm']) * 60:.2f} s\n= {float(D['ex_fwhm']) / P['dt']:.1f} scans", color=ORANGE, va="center", fontsize=12, fontweight="bold")
    ax.text(ex_rt[sel][0], ex_sm[k] * 1.02, "apex", color=INK2, fontsize=11, va="bottom")
    ax.text(ex_rt[sel][0], half * 1.03, "half max", color=INK2, fontsize=11, va="bottom")
    ax.set(xlabel="RT (min)", ylabel="intensity", title=f"(b) One seed (m/z {float(D['ex_mz']):.4f}): its FWHM")
    ax.yaxis.set_major_formatter(kfmt); ax.legend(loc="upper right")


def p2c(ax):
    ax.hist(fw, bins=np.arange(0, 8.05, 0.2), color=BLUE, alpha=0.85, zorder=3)
    ax.axvline(fwhm_s, color=ORANGE, lw=2.6, zorder=4)
    ax.text(fwhm_s + 0.15, ax.get_ylim()[1] * 0.95, f"median {fwhm_s:.2f} s = {P['fwhm_scans']:.1f} scans\n"
            f"→ link across K = {P['K']} scans\n→ ≥ {P['min_scans']} scans per peak\n→ smoothing: {P['n_smooth']} × [1 2 1]",
            color=INK, fontsize=12, va="top")
    ax.set(xlabel="FWHM of a seed (s)", ylabel="seeds", title=f"(c) Widths of the {len(fw):,} measurable seeds", xlim=(0, 8))
    ygrid(ax)


fig, axs = plt.subplots(1, 3, figsize=(19.5, 5.2), gridspec_kw=dict(wspace=0.32))
p2a(axs[0]); p2b(axs[1]); p2c(axs[2])
save(fig, "P2_seeds_and_peak_width", "Seeds and the peak width",
     "(a) The seeds: the 2,000 most intense centroids that are mutually ≥ 20 ppm apart, one per strong ion, at its apex "
     "(orange ring: the seed of b). (b) For each seed the EIC (max per scan within ± 20 ppm) around the apex is averaged "
     "over 3 scans and its width at half maximum read off by interpolation; seeds without a crossing on both sides or "
     "fewer than 3 raw scans above half max are skipped. (c) The median width of all measurable seeds sets how many "
     "adjacent scans are linked (K = half the width in scans, 1–3), the minimum number of scans of a peak (3–5) and the "
     "smoothing kernel (half the median width).")
panel("P2a_seeds", p2a); panel("P2b_one_seed_fwhm", p2b); panel("P2c_fwhm_histogram", p2c)

# =====================================================================================================
# P3 m/z error model
I_all, dev = D["mz_I"], D["mz_dev"]
rng = np.random.default_rng(0)
sub = rng.choice(len(I_all), min(20000, len(I_all)), replace=False)


def p3a(ax):
    """one seed trace: m/z deviation per scan widens where the ion is weak"""
    rt, ppm, I = D["tr_rt"], D["tr_mz_ppm"], D["tr_I"]
    s = sig(I)
    ax.fill_between(rt, -2 * s, 2 * s, color=BLUE_T, lw=0, label="± 2 σ(I) of the fitted model")
    ax.plot(rt, ppm, "o-", color=BLUE, ms=5, lw=1.2, label="m/z of the ion per scan (ppm from its mean)")
    ax.axhline(0, color=AXIS, lw=1)
    ax2 = ax.twinx()
    ax2.plot(rt, I, color=MUTED, lw=1.4, ls=(0, (4, 2)))
    ax2.set_yscale("log"); ax2.set_ylim(I.min() / 3, I.max() * 300); ax2.yaxis.set_visible(False)
    for sp in ax2.spines.values(): sp.set_visible(False)
    ax2.text(rt[np.argmax(I)], I.max() * 1.6, "intensity (dashed, log)", color=MUTED, fontsize=11, ha="center")
    ax.set(xlabel="RT (min)", ylabel="m/z deviation (ppm)", title=f"(a) One seed trace, m/z {float(D['tr_mz0']):.4f}",
           ylim=(-3.2, 3.2))
    ax.legend(loc="lower left", fontsize=10.5)


def p3b(ax):
    ax.scatter(I_all[sub], dev[sub], s=3, color=MUTED, alpha=0.25, lw=0, rasterized=True)
    bI, bsd = D["mz_bin_I"], D["mz_bin_sd"]
    ax.vlines(bI, -bsd, bsd, color=ORANGE, lw=3.2, zorder=4)
    ax.plot(bI, bsd, "_", color=ORANGE, ms=12, mew=2.5, zorder=4); ax.plot(bI, -bsd, "_", color=ORANGE, ms=12, mew=2.5, zorder=4)
    II = np.logspace(np.log10(I_all.min()), np.log10(I_all.max()), 200)
    ax.plot(II, sig(II), color=BLUE, lw=2.4, zorder=5); ax.plot(II, -sig(II), color=BLUE, lw=2.4, zorder=5)
    ax.axvline(fl, color=INK2, lw=1.2, ls=(0, (3, 3)))
    ax.plot([fl], [sig(fl)], "o", color=INK, ms=7, zorder=6)
    ax.text(fl * 1.3, 3.9, f"file floor {fl:,.0f}:\nσ = {sig(fl):.2f} ppm\n→ tolerance cap 4 σ = {P['tol_max_ppm']:.1f} ppm", fontsize=11.5, va="top")
    ax.text(2e6, 3.0, f"σ(I) = √(a² + b²/I)\na = {P['sig_a']:.2f} ppm, b = {P['sig_b']:.0f}", color=BLUE, fontsize=12, fontweight="bold")
    ax.text(2e6, -4.2, "orange: robust sd per\nhalf-decade bin (≥ 20 pts)", color=ORANGE, fontsize=11)
    ax.set_xscale("log"); ax.set(xlabel="centroid intensity", ylabel="m/z deviation from the ion's mean (ppm)", ylim=(-4.5, 4.5),
                                 title=f"(b) All {len(I_all):,} centroids of the seed traces")
    ax.xaxis.set_major_formatter(kfmt)


def p3c(ax):
    """what the tolerance does at linking time: pair tolerance vs the two intensities"""
    II = np.logspace(3, 9, 300)
    ax.plot(II, 4 * sig(II), color=BLUE, lw=2.4, label="4 σ(I) for a pair of equal intensity")
    ax.axhline(P["tol_max_ppm"], color=ORANGE, lw=2.2, label=f"cap {P['tol_max_ppm']:.1f} ppm (4 σ at the floor)")
    ax.axhline(P["same_scan_floor_ppm"] if "same_scan_floor_ppm" in P else 5.0, color=MUTED, lw=1.4, ls=(0, (3, 3)), label="floor 5 ppm")
    ax.fill_between(II, 0, np.minimum(4 * sig(II), P["tol_max_ppm"]), color=BLUE_T, lw=0)
    ax.text(3e5, 2.3, "two centroids in adjacent scans\nmay be the same ion when their m/z\ndistance is within the shaded band", fontsize=11, color=INK2)
    ax.set_xscale("log"); ax.set(xlabel="intensity", ylabel="m/z tolerance (ppm)", ylim=(0, 8), xlim=(1e3, 1e9),
                                 title="(c) The linking tolerance that follows")
    ax.xaxis.set_major_formatter(kfmt); ax.legend(loc="upper right", fontsize=10.5); ygrid(ax)


fig, axs = plt.subplots(1, 3, figsize=(19.5, 5.2), gridspec_kw=dict(wspace=0.32))
p3a(axs[0]); p3b(axs[1]); p3c(axs[2])
save(fig, "P3_mz_error_model", "The m/z error model, fitted per file",
     "(a) Along one seed's trace the centroid m/z scatters little at the apex and more in the tails: the error depends on "
     "intensity. (b) All centroids of the 2,000 seed traces, as deviation from each ion's intensity-weighted mean m/z. "
     "Per half-decade of intensity a robust sd (1.4826 × MAD) is taken and sd² = a² + b²/I fitted by least squares "
     "(a ≥ 0.3 ppm). (c) At linking time two centroids may be the same ion when their m/z distance is within the "
     "tolerance their own intensities allow, capped at 4 σ of the file's floor intensity (5–30 ppm).")
panel("P3a_one_trace_mz", p3a); panel("P3b_mz_error_fit", p3b); panel("P3c_linking_tolerance", p3c)

# =====================================================================================================
# P4 the noise law
nz_rt, nz_y, nz_idx, nz_fit, nz_apex = D["nz_rt"], D["nz_y"], D["nz_fit_idx"], D["nz_fit"], int(D["nz_apex"])
corr = 1.0 / np.sqrt(1.0 - 17.0 / 35.0)


def p4a(ax, axr=None):
    if axr is None:
        axr = ax.inset_axes([0, -0.5, 1, 0.4])
    ax.plot(nz_rt, nz_y, "o", color=INK, ms=6, zorder=5, label="raw EIC points (≥ 10 % of apex)")
    ax.plot(nz_rt[nz_idx], nz_fit, "-", color=BLUE, lw=2.2, zorder=4, label="5-point local quadratic through its neighbours")
    for jj, f in zip(nz_idx, nz_fit):
        col = MUTED if jj == nz_apex else ORANGE
        ax.plot([nz_rt[jj], nz_rt[jj]], [f, nz_y[jj]], color=col, lw=2.6, zorder=3)
    ax.plot([], [], color=ORANGE, lw=2.6, label="residual = noise sample (apex excluded)")
    ax.set(ylabel="intensity", title=f"(a) One seed trace (m/z {float(D['nz_mz']):.4f}): residuals",
           ylim=(0, nz_y.max() * 1.6)); ax.set_xticklabels([])
    ax.yaxis.set_major_formatter(kfmt); ax.legend(loc="upper left", fontsize=10.5)
    # the residuals on their own scale, against the noise the fitted law predicts at each level
    axr.sharex(ax)
    res = (nz_y[nz_idx] - nz_fit) * corr
    keep = nz_idx != nz_apex
    axr.fill_between(nz_rt[nz_idx], -noise(nz_fit), noise(nz_fit), color=BLUE_T, lw=0)
    axr.vlines(nz_rt[nz_idx][keep], 0, res[keep], color=ORANGE, lw=2.6)
    axr.plot(nz_rt[nz_idx][keep], res[keep], "o", color=ORANGE, ms=4)
    axr.axhline(0, color=AXIS, lw=1)
    axr.set(xlabel="RT (min)", ylabel="residual")
    axr.yaxis.set_major_formatter(kfmt)
    axr.text(0.02, 0.95, "× √(35/18); band: ± σ(level) from the fitted law", transform=axr.transAxes, fontsize=10, color=INK2, va="top")


def p4b(ax):
    bI, bsd, bn = D["nl_bin_I"], D["nl_bin_sd"], D["nl_bin_n"]
    II = np.logspace(3.5, 9, 300)
    ax.plot(II, np.full_like(II, fl), color=MUTED, lw=1.6, ls=(0, (4, 2)))
    ax.plot(II, np.sqrt(c_ * II), color=AQUA, lw=1.6, ls=(0, (4, 2)))
    ax.plot(II, r_ * II, color=ORANGE, lw=1.6, ls=(0, (4, 2)))
    ax.plot(II, noise(II), color=BLUE, lw=3, zorder=4)
    ax.scatter(bI, bsd, s=30 + 60 * np.sqrt(bn / bn.max()), color=INK, zorder=6, lw=0)
    for x, y, n in zip(bI, bsd, bn):
        ax.annotate(f"{n:,}", (x, y), (0, 9), textcoords="offset points", ha="center", fontsize=9.5, color=INK2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(3e3, 1e9); ax.set_ylim(1e3, 1e8)
    ax.text(1.5e8, fl * 1.3, f"floor {fl:,.0f}", color=INK2, fontsize=11)
    ax.text(5e4, np.sqrt(c_ * 5e4) * 0.4, f"shot-like √(c·I), c = {c_:,.0f}", color=AQUA, fontsize=11, rotation=0)
    ax.text(3e7, r_ * 3e7 * 0.35, f"proportional r·I, r = {r_:.3f}", color=ORANGE, fontsize=11)
    ax.text(1.5e4, 2.5e6, "σ(I)² = floor² + c·I + r²·I²", color=BLUE, fontsize=13, fontweight="bold")
    ax.text(1.5e4, 1.2e6, "dots: robust sd per half-decade bin (label: residuals in the bin)", color=INK2, fontsize=10.5)
    ax.set(xlabel="ion current I (fitted level)", ylabel="scan-to-scan noise σ (counts)", title="(b) The noise law fitted to the bins")
    ax.xaxis.set_major_formatter(kfmt); ax.yaxis.set_major_formatter(kfmt)


def p4c(ax):
    """what it is for: the same dip is a significant valley on a strong ion and ripple on a weak one (n_valleys rule:
    a local minimum counts when it lies >= k_valley x sqrt(sigma(min)^2 + sigma(lower max)^2) below the lower maximum)"""
    rt = np.linspace(0, 1, 400)
    shape = np.exp(-0.5 * ((rt - 0.38) / 0.08) ** 2) + 0.8 * np.exp(-0.5 * ((rt - 0.66) / 0.08) ** 2)
    i_a = np.argmax(shape); i_b = 200 + np.argmax(shape[200:]); i_v = i_a + np.argmin(shape[i_a:i_b])
    lower, saddle = shape[i_b], shape[i_v]
    ax.plot(rt, shape, color=INK, lw=2.4)
    ax.plot([rt[i_v], rt[i_v]], [saddle, lower], color=ORANGE, lw=3)
    ax.plot([rt[i_v], rt[i_b]], [lower, lower], color=AXIS, lw=1, ls=(0, (3, 3)))
    ax.text(rt[i_v] - 0.02, (saddle + lower) / 2, "dip", color=ORANGE, ha="right", va="center", fontsize=12, fontweight="bold")
    lines = []
    for lev, lab in [(4e4, "weak ion, apex 40k"), (4e6, "strong ion, apex 4M")]:
        dip = lev * (lower - saddle)
        thr = P["k_valley"] * np.sqrt(noise(lev * saddle) ** 2 + noise(lev * lower) ** 2)
        lines.append(f"{lab}: dip {counts(dip)} vs {P['k_valley']:.0f} σ = {counts(thr)} → "
                     + ("a real valley" if dip >= thr else "ripple, no valley"))
    ax.text(0.02, 0.97, "a dip counts as a valley when it exceeds 3 σ of the\ntwo levels, σ from the law:\n\n" + "\n".join(lines),
            transform=ax.transAxes, fontsize=11, color=INK2, va="top")
    ax.set(xlabel="RT (arbitrary)", ylabel="intensity (relative)", title="(c) The same shape at two levels", ylim=(0, 1.9), yticks=[])


_ = p4c  # drawn below


fig = plt.figure(figsize=(19.5, 5.6))
gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1], wspace=0.32, hspace=0.12)
axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[:, 1]), fig.add_subplot(gs[:, 2])]
p4a(axs[0], fig.add_subplot(gs[1, 0])); p4b(axs[1]); p4c(axs[2])
save(fig, "P4_noise_law", "The scan-to-scan noise law, fitted per file",
     "(a) On every seed trace (the contiguous scans ≥ 10 % of the apex) each point is compared with the local quadratic "
     "through its neighbours; the residuals are noise, not peak shape (apex excluded, scaled by √(35/18) to undo the "
     "fit's variance reduction). (b) The robust sd of the residuals per half-decade of level and the law "
     "σ(I)² = floor² + c·I + r²·I² fitted to the bins by non-negative least squares. (c) The law gives the expected "
     "noise at any level: a candidate's shape descriptors (its significant valleys, its total variation beyond one rise "
     "and fall) are counted in units of it, so the same dip is ripple on a weak ion and a real valley on a strong one.")
def _p4a_panel(fig):
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1], left=0.17, right=0.95, bottom=0.17, top=0.95, hspace=0.12)
    a0 = fig.add_subplot(gs[0]); p4a(a0, fig.add_subplot(gs[1])); a0.set_title("")


fig = plt.figure(figsize=PANEL); _p4a_panel(fig)
fig.savefig(PAN / "P4a_trace_residuals.png", dpi=300); fig.savefig(PAN / "P4a_trace_residuals.pdf"); plt.close(fig)
panel("P4b_noise_law_fit", p4b); panel("P4c_why_noise_law", p4c)

# =====================================================================================================
# P5 scan gains (SZ22 injection; HZV029 has no jitter beyond its noise)
GF = Path(META.get("gain_file", "")).name
g_rt, gains, g_sd, g_n = D["alt_g_rt"], D["alt_gains"], float(D["alt_gain_sd"]), int(D["alt_n_gain_scans"])


def p5a(ax):
    ax.plot(g_rt, gains, color=BLUE, lw=0.8)
    ax.axhline(1, color=AXIS, lw=1)
    ax.set(xlabel="RT (min)", ylabel="scan gain (median 1)", title=f"(a) Gain of every MS1 scan ({META.get('gain_run', '')} injection)",
           ylim=(0.8, 1.26))
    ax.text(0.98, 0.04, f"robust sd of the log gains {g_sd:.3f}\n({g_n} scans rest on ≥ 3 seed traces;\n"
                        f"1 where no seed trace covers the scan)\nHZV029 worked example: gains all 1,\nno jitter beyond its noise",
            transform=ax.transAxes, fontsize=10.5, color=INK2, va="bottom", ha="right")
    ygrid(ax)


def p5b(ax):
    rt, raw, cor = D["alt_g_ex_rt"], D["alt_g_ex_raw"], D["alt_g_ex_corr"]
    ax.plot(rt, raw, "o-", color=MUTED, ms=4, lw=1.3, label="raw trace")
    ax.plot(rt, cor, "o-", color=BLUE, ms=4, lw=1.6, label="divided by the scan gains")
    ax.set(xlabel="RT (min)", ylabel="intensity", title=f"(b) One strong ion (m/z {float(D['alt_g_ex_mz']):.4f})")
    ax.yaxis.set_major_formatter(kfmt); ax.legend(loc="upper left", fontsize=11)


def p5c(ax):
    rt, raw, cor = D["alt_g_ex_rt"], D["alt_g_ex_raw"], D["alt_g_ex_corr"]
    i0 = np.searchsorted(g_rt, rt[0]); gg = gains[i0:i0 + len(rt)]
    ax.plot(rt, raw / cor, "o-", color=ORANGE, ms=4, lw=1.3)
    ax.axhline(1, color=AXIS, lw=1)
    ax.set(xlabel="RT (min)", ylabel="gain applied to each scan", title="(c) The factor removed from that trace")
    ax.text(0.02, 0.96, "the same factor is removed from every ion\nof the scan: it is the scan's, not the ion's", transform=ax.transAxes,
            fontsize=11, color=INK2, va="top"); ygrid(ax)


fig, axs = plt.subplots(1, 3, figsize=(19.5, 5.2), gridspec_kw=dict(wspace=0.32))
p5a(axs[0]); p5b(axs[1]); p5c(axs[2])
save(fig, "P5_scan_gains", "Per-scan gain, read off the strongest ions",
     "A gain all ions of a scan share (AGC / injection-time error, spray flicker) is estimated from the seed traces: each "
     "point's log intensity minus the quadratic through its four neighbours in time is that scan's log gain minus the "
     "neighbours'; the median over the traces covering a scan gives one equation per scan and all gains are solved at "
     "once with a ridge set by the spread actually seen, so a file without jitter beyond its noise gets gains of 1 "
     "(the HZV029 demo file does). Shown on an SZ22 injection, the replicate-run file with the largest gain sd. "
     "Every scan is divided by its gain before picking; the floor and the noise law are then measured again.")
panel("P5a_scan_gains", p5a); panel("P5b_trace_before_after", p5b); panel("P5c_gain_factor", p5c)

# =====================================================================================================
# P6 what the files said: the measured settings of 113 files of the benchmark
rows = list(csv.DictReader(open(HERE / "all_file_params.tsv"), delimiter="\t"))
order = ["HZV029_cert", "BM21_HILIC", "BM21_RP", "LI2018", "SZ22_12C", "YEAST_12C", "YEAST_NEG", "IDSL003",
         "MC_NIST_Plasma_Orbitrap_RP_positive", "MC_NIST_Plasma_Orbitrap_RP_negative", "MC_Mouse_Plasma_Orbitrap_HILIC_positive",
         "MC_NIST_Feces_Orbitrap_HILIC_negative", "MC_Human_Serum_QTOF_RP_positive", "MC_Mouse_Feces_QTOF_HILIC_positive",
         "MC_Fruit_Fly_QTOF_HILIC_positive", "MC_Human_Urine_QTOF_HILIC_negative"]
label = {"HZV029_cert": "HZV029 (human, RP+)", "BM21_HILIC": "BM21 HILIC", "BM21_RP": "BM21 RP", "LI2018": "LI2018 (spike-in)",
         "SZ22_12C": "SZ22 E. coli ¹²C", "YEAST_12C": "Yeast ¹²C", "YEAST_NEG": "Yeast neg", "IDSL003": "IDSL003",
         "MC_NIST_Plasma_Orbitrap_RP_positive": "MC plasma Orbitrap RP+", "MC_NIST_Plasma_Orbitrap_RP_negative": "MC plasma Orbitrap RP−",
         "MC_Mouse_Plasma_Orbitrap_HILIC_positive": "MC mouse plasma Orbitrap HILIC+", "MC_NIST_Feces_Orbitrap_HILIC_negative": "MC feces Orbitrap HILIC−",
         "MC_Human_Serum_QTOF_RP_positive": "MC serum QTOF RP+", "MC_Mouse_Feces_QTOF_HILIC_positive": "MC mouse feces QTOF HILIC+",
         "MC_Fruit_Fly_QTOF_HILIC_positive": "MC fruit fly QTOF HILIC+", "MC_Human_Urine_QTOF_HILIC_negative": "MC urine QTOF HILIC−"}
order = [r for r in order if any(x["run"] == r for x in rows)]
demo_stem = Path(META["main_file"]).stem
cols = [("dt", "MS1 interval (s)", lambda r: float(r["dt"]) * 60, False), ("fwhm_med", "median peak width (s)", lambda r: float(r["fwhm_med"]) * 60, False),
        ("fwhm_scans", "width in scans", lambda r: float(r["fwhm_scans"]), False),
        ("tol", "m/z tolerance cap (ppm)", lambda r: float(r["tol_max_ppm"]), False),
        ("floor", "intensity floor (counts)", lambda r: float(r["floor"]), True),
        ("gain", "scan-gain sd", lambda r: float(r["gain_sd"]), False)]
fig, axs = plt.subplots(1, len(cols), figsize=(21, 7.0), sharey=True, gridspec_kw=dict(wspace=0.12))
y = np.arange(len(order))[::-1]
for ax, (key, lab, f, log) in zip(axs, cols):
    for yi, run in zip(y, order):
        v = np.array([f(r) for r in rows if r["run"] == run])
        jit = (np.random.default_rng(1).random(len(v)) - 0.5) * 0.5 if len(v) > 3 else np.zeros(len(v))
        ax.scatter(v, yi + jit, s=26, color=BLUE, alpha=0.55 if len(v) > 3 else 0.9, lw=0, zorder=3)
        if run == "HZV029_cert":
            vd = [f(r) for r in rows if r["run"] == run and r["stem"] == demo_stem]
            ax.scatter(vd, [yi], s=110, marker="*", color=ORANGE, zorder=4)
    if log: ax.set_xscale("log")
    ax.set_xlabel(lab); ax.set_yticks(y); ax.set_yticklabels([label[r] for r in order]); ax.tick_params(axis="y", length=0)
    ax.grid(True, axis="x", zorder=0); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    if key == "tol":
        ax.axvline(5, color=GRID, lw=1); ax.axvline(30, color=GRID, lw=1); ax.set_xlim(0, 33)
        ax.text(5.5, len(order) - 0.6, "clipped to 5–30 ppm", fontsize=10.5, color=INK2)
axs[0].text(0, 1.03, f"{len(rows)} files, {len(order)} datasets; one dot per file, star = the worked example",
            transform=axs[0].transAxes, fontsize=12, color=INK2)
save(fig, "P6_what_the_files_said", "What the files said: measured settings across the benchmark",
     f"The per-file sidecars of the current default run ({len(rows)} files in {len(order)} datasets): scan interval, "
     "median peak width in seconds and in scans, the m/z tolerance cap, the intensity floor and the scan-gain sd. They "
     "span 0.2–1.4 s per scan, 1.7–11 s peak widths, 5–30 ppm and two orders of magnitude in floor between Orbitrap and "
     "Q-TOF files; no single setting would fit them all, and none was chosen by hand. Star: the HZV029 injection of the "
     "worked example.")

# catalog
with open(OUT / "INDEX.md", "w") as fh:
    fh.write("# Parameter-estimation demo figures\n\nPNG 220 dpi + PDF; `panels/` holds untitled 460 × 366 pt panels at 300 dpi.\n"
             f"Worked example: {META['main_file']} (HZV029_cert). Scan gains: {META.get('gain_file', '')}.\n"
             "Data: estimation_demo.npz, demo_params.json, all_file_params.tsv (extract.py).\n\n| file | title | caption |\n|---|---|---|\n")
    for n, t, c in CAT:
        fh.write(f"| {n} | {t} | {c} |\n")
print("done")
