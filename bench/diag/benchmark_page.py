# builds the complete benchmark overview (every tool, every run) from results/scores/*.tsv: truth metrics at
# 10 ppm / 0.1 min, aligned-table feature counts per run, wall / CPU time and peak memory per run.
# usage: benchmark_page.py OUT_HTML OUT_MD
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.runs import LOCAL_NAMES  # noqa: E402

S = ROOT / "results/scores"
OUT_HTML, OUT_MD = Path(sys.argv[1]), Path(sys.argv[2])
TOOLS = ["peak3d", "asari", "masscube", "idslipa", *LOCAL_NAMES]
NAMES = {"peak3d": "peak3d", "asari": "asari", "masscube": "MassCube", "idslipa": "IDSL.IPA", **LOCAL_NAMES}
RUNS = ["HZV029_cert", "HZV029_full", "YEAST_NEG", "YEAST_12C", "YEAST_13C", "SZ22_12C", "SZ22_13C", "LI2018", "IDSL003", "BM21_HILIC", "BM21_RP"]


def at(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= df[k].values == v
    return df[m]


hzv = at(pd.read_csv(S / "hzv029.tsv", sep="\t"), ppm=10, rt_tol_min=0.1).set_index("tool")
yn = at(pd.read_csv(S / "yeast_neg.tsv", sep="\t"), ppm=10, rt_tol_min=0.1).set_index("tool")
sp = at(pd.read_csv(S / "spike.tsv", sep="\t"), ppm=10, rt_tol_min=0.1).set_index("tool")
idsl = at(pd.read_csv(S / "idsl.tsv", sep="\t"), ppm=10, rt_tol_min=0.1).set_index("tool")
idsl20 = at(pd.read_csv(S / "idsl.tsv", sep="\t"), ppm=20, rt_tol_min=0.1).set_index("tool")
ct = at(pd.read_csv(S / "credtruth.tsv", sep="\t"), ppm=10, rt_tol_min=0.1)
cts = at(pd.read_csv(S / "credtruth_screened.tsv", sep="\t"), ppm=10, rt_tol_min=0.1)
cred = pd.read_csv(S / "credentialing.tsv", sep="\t")
ratio = pd.read_csv(S / "ratio.tsv", sep="\t")
ratio = ratio[ratio["min_presence"].astype(str) == "all"]
rt = pd.read_csv(S / "runtime.tsv", sep="\t")


def g(df, tool, col, fmt="{:.3f}", ds=None, run=None):
    try:
        x = df
        if ds is not None:
            x = x[x["dataset"] == ds]
        if run is not None:
            x = x[x["run"] == run]
        if "tool" in x.columns:
            x = x[x["tool"] == tool]
            v = x[col].values[0]
        else:
            v = x.loc[tool, col]
        return fmt.format(v)
    except (KeyError, IndexError):
        return "–"


# TUNED asari on IDSL003 (scripts/32_run_asari_tuned.sbatch): the default min_peak_height 1e5 sits above 92 % of the
# IDSL003 true peaks, so asari is also shown with a lowered cutoff. Not part of the default-settings comparison.
TUNED = {"asari": "asari default (min_peak_height 1e5)",
         "asari_autoheight": "asari --autoheight True (estimated cutoff 1,024)",
         "asari_h1000": "asari --min_peak_height 1000"}
tuned = pd.DataFrame([{"setting": lab,
                       "F1 (precision / recall)": f"{g(idsl, t, 'F1')} ({g(idsl, t, 'precision')} / {g(idsl, t, 'recall')})",
                       "F1 at 20 ppm": g(idsl20, t, "F1"),
                       "features": g(rt[rt["run"] == "IDSL003"], t, "n_features", "{:,.0f}") if t == "asari" else
                       "{:,}".format(len(pd.read_csv(next((S.parent / t / "IDSL003").glob("asari_out*/export/full_Feature_table.tsv")), sep="\t")))}
                      for t, lab in TUNED.items()]).set_index("setting")

rows = []
for t in TOOLS:
    rows.append({
        "tool": NAMES[t],
        "HZV029 certified found / 402": g(hzv, t, "found", "{:.0f}"),
        "YEAST_NEG curated found / 314": g(yn, t, "found", "{:.0f}"),
        "LI2018 spike recall": g(sp, t, "recall"),
        "LI2018 log2 error / within 2-fold": f"{g(sp, t, 'diff_median_abs_log2_err')} / {g(sp, t, 'diff_within_2fold')}",
        "IDSL003 clean F1 (precision / recall)": f"{g(idsl, t, 'F1')} ({g(idsl, t, 'precision')} / {g(idsl, t, 'recall')})",
        "IDSL003 F1 at 20 ppm": g(idsl20, t, "F1"),
        "SZ22 credentialed recall, full / screened": f"{g(ct, t, 'recall_12C', ds='SZ22')} / {g(cts, t, 'recall_12C', ds='SZ22')}",
        "YEAST credentialed recall, full / screened": f"{g(ct, t, 'recall_12C', ds='YEAST')} / {g(cts, t, 'recall_12C', ds='YEAST')}",
        "YEAST tier A / tier B (full)": f"{g(ct, t, 'recall_A', ds='YEAST')} / {g(ct, t, 'recall_B', ds='YEAST')}",
        "credentialed fraction of own features, SZ22 / YEAST": f"{g(cred, t, 'frac_credentialed_net', ds='SZ22')} / {g(cred, t, 'frac_credentialed_net', ds='YEAST')}",
        "BM21 ratio-consistent fraction HILIC / RP": f"{g(ratio, t, 'frac_ratio_consistent', run='BM21_HILIC')} / {g(ratio, t, 'frac_ratio_consistent', run='BM21_RP')}",
    })
truth = pd.DataFrame(rows).set_index("tool")

feat = rt.pivot(index="run", columns="tool", values="n_features").reindex(RUNS)[TOOLS].rename(columns=NAMES)
wall = rt.pivot(index="run", columns="tool", values="wall_min").reindex(RUNS)[TOOLS].rename(columns=NAMES)
cpu = rt.pivot(index="run", columns="tool", values="cpu_min").reindex(RUNS)[TOOLS].rename(columns=NAMES)
mem = rt.pivot(index="run", columns="tool", values="max_rss_gb").reindex(RUNS)[TOOLS].rename(columns=NAMES)
nfiles = rt.drop_duplicates("run").set_index("run")["n_files"].reindex(RUNS)
feat.insert(0, "files", nfiles)
wall.loc["total"] = wall.sum()
cpu.loc["total"] = cpu.sum()


def md_table(df, fmt=None):
    """markdown table without tabulate"""
    d = df.copy()
    cols = [str(c) for c in d.columns]
    lines = ["| " + (d.index.name or "") + " | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for idx, row in d.iterrows():
        vals = []
        for v in row.values:
            if isinstance(v, (float, np.floating)) and fmt:
                vals.append("–" if pd.isna(v) else fmt.format(v))
            else:
                vals.append("–" if (isinstance(v, float) and pd.isna(v)) else str(v))
        lines.append("| " + str(idx) + " | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def html_table(df, fmt=None, index_name=""):
    d = df.copy()
    if fmt:
        d = d.map(lambda v: "–" if pd.isna(v) else fmt.format(v))
    d.index.name = index_name
    return d.to_html(border=0, classes="t", escape=False, na_rep="–")


STYLE = """<style>
:root { --bg:#f6f5f1; --fg:#1b1a17; --muted:#625d53; --line:#d7d2c6; --accent:#245a8c; --soft:#ecebe4;
  --display:"Source Serif 4", Georgia, serif; --body:"Source Sans 3", "Segoe UI", Helvetica, Arial, sans-serif; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#161513; --fg:#ebe7de; --muted:#a8a194; --line:#39362f; --accent:#7fb3e0; --soft:#22211d; color-scheme: dark } }
:root[data-theme="dark"] { --bg:#161513; --fg:#ebe7de; --muted:#a8a194; --line:#39362f; --accent:#7fb3e0; --soft:#22211d; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 15px; line-height: 1.45; margin: 0; }
.wrap { max-width: 1100px; margin: 0 auto; padding-block: 28px 56px; padding-inline: 16px; }
h1 { font-family: var(--display); font-size: 1.9rem; margin: 0 0 4px; text-wrap: balance; }
h2 { font-family: var(--display); font-size: 1.3rem; margin: 34px 0 8px; border-bottom: 1px solid var(--line); padding-bottom: 3px; }
p { max-width: 78ch; } .eyebrow { color: var(--muted); text-transform: uppercase; letter-spacing: .08em; font-size: .76rem; }
.tbl { overflow-x: auto; margin: 8px 0 14px; }
table.t { border-collapse: collapse; font-size: .86rem; font-variant-numeric: tabular-nums; width: 100%; }
table.t th, table.t td { padding: 4px 8px; border-bottom: 1px solid var(--line); text-align: right; vertical-align: top; white-space: nowrap; }
table.t th { color: var(--muted); font-weight: 600; font-size: .76rem; text-transform: uppercase; letter-spacing: .03em; }
table.t td:first-child, table.t th:first-child { text-align: left; }
table.t tbody tr:first-child td { font-weight: 600; }
.note { color: var(--muted); font-size: .86rem; max-width: 80ch; }
</style>"""

truth_t = truth.T
truth_t.index.name = "metric (10 ppm / 0.1 min)"
html = f"""<title>Peakbench overview</title>{STYLE}
<div class="wrap">
<div class="eyebrow">peakbench · five tools · eleven runs · default settings · scored 2026-10-06</div>
<h1>Benchmark overview</h1>
<p>Every tool at its default settings on the same converted mzML files; peak3d is the 2026-10-06 recipe (iteration-2 gates,
strand-fusion fix, far-level ceiling 0.5). Matching at 10 ppm and ± 0.1 min unless stated. The credentialed truth is scored on
the full list and on the prominence-screened list; IDSL003 on the 1,676 + 15,328 cleaned labels.</p>
<h2>Truth metrics</h2>
<div class="tbl">{truth_t.to_html(border=0, classes="t", escape=False)}</div>
<p class="note">HZV029: certified features in 3 repeat injections. YEAST_NEG: curated list (metabolites and artefacts). LI2018: standards at
defined ratios (recall, median |log2 error| of the differential ratios, share within 2-fold). IDSL003: TP/TN labels. SZ22 / YEAST:
tool-independent 12C/13C credentialed truth (recall in the 12C run; tier A = carbon count confirmed). Credentialed fraction:
share of a tool's own 12C features with a 13C partner, decoy-corrected. BM21: share of features whose plasma / juice mixing
ratios are consistent.</p>
<h2>IDSL003: asari with a lowered height cutoff (tuned)</h2>
<div class="tbl">{tuned.to_html(border=0, classes="t", escape=False)}</div>
<p class="note">asari's default min_peak_height of 1e5 suits Orbitrap intensities but sits above 92 % of the IDSL003 true peaks
(low-count TOF, median true-peak height about 8e3), so the default run reports 420 features. These rows are tuned and are not part
of the default-settings comparison above. Autoheight is asari's own estimate (half the weakest 12C/13C landmark track; it also lowers
the per-point intensity threshold to 102). The fixed 1000 cutoff keeps the 1000-count point threshold. Neither setting was chosen on the labels.</p>
<h2>Features in the aligned table</h2>
<div class="tbl">{html_table(feat, "{:,.0f}", "run")}</div>
<h2>Wall time (minutes)</h2>
<div class="tbl">{html_table(wall, "{:.2f}", "run")}</div>
<p class="note">One array task per tool and run, 8 CPUs and 96 GB on one CPU generation (zen4), node-local staging of the inputs. The peak3d
column is from the 2026-10-06 rerun in which all eleven peak3d tasks started together; the other tools ran in the 55-task
array of 2026-10-02. Times include JIT warm-up and environment start-up.</p>
<h2>CPU time (minutes)</h2>
<div class="tbl">{html_table(cpu, "{:.2f}", "run")}</div>
<h2>Peak memory (GB)</h2>
<div class="tbl">{html_table(mem, "{:.2f}", "run")}</div>
<p class="note">Files: results/scores/*.tsv (full grids: 5 / 10 / 20 ppm × 0.05 / 0.1 / 0.2 min), report_2026-10-06_final.log,
benchmark_2026-10-06.md (this page as text).</p>
</div>
"""
OUT_HTML.write_text(html)
md = ["# Benchmark overview (2026-10-06, 10 ppm / 0.1 min)\n", "## Truth metrics\n", md_table(truth_t),
      "\n## IDSL003: asari with a lowered height cutoff (tuned; not part of the default comparison)\n", md_table(tuned),
      "\n## Features in the aligned table\n",
      md_table(feat, "{:,.0f}"), "\n## Wall time (min)\n", md_table(wall, "{:.2f}"), "\n## CPU time (min)\n",
      md_table(cpu, "{:.2f}"), "\n## Peak memory (GB)\n", md_table(mem, "{:.2f}")]
OUT_MD.write_text("\n".join(md) + "\n")
print(OUT_HTML, OUT_MD)
print("\n".join(md))
