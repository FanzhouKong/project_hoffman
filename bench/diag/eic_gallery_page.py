#!/usr/bin/env python3
"""Assemble several eic_overlay.py output directories into one HTML page (inline SVG figures,
per-feature apex tables). usage: eic_gallery_page.py OUT.html LABEL=DIR [LABEL=DIR ...]"""
import html
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.runs import RUNS  # noqa: E402

out = Path(sys.argv[1])
sets = [a.split("=", 1) for a in sys.argv[2:]]

CSS = """
<title>peak3d EIC Overlays</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600&family=Source+Code+Pro&display=swap">
<style>
/* layout: one reading column; each dataset a chapter, each feature a figure block with its numbers beside it */
:root { --bg:#f7f7f4; --fg:#1c1d1a; --muted:#67695f; --line:#dcdcd3; --card:#eeeee8; --accent:#1f6fc5;
        --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --warn:#b4531a; --ok:#1f7a4f;
        --sans:"Source Sans 3",system-ui,sans-serif; --mono:"Source Code Pro",ui-monospace,monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#171816; --fg:#e9e8e0; --muted:#9c9d92; --line:#34352f; --card:#20211d; --accent:#4d94e6;
        --s1:#3987e5; --s2:#d95926; --s3:#199e70; --warn:#f0925a; --ok:#5fcf8f; color-scheme:dark } }
:root[data-theme="dark"] { --bg:#171816; --fg:#e9e8e0; --muted:#9c9d92; --line:#34352f; --card:#20211d; --accent:#4d94e6;
        --s1:#3987e5; --s2:#d95926; --s3:#199e70; --warn:#f0925a; --ok:#5fcf8f; color-scheme:dark }
body { background:var(--bg); color:var(--fg); font-family:var(--sans); font-size:15px; line-height:1.5; margin:0; padding-block:28px 48px; padding-inline:16px; }
main { max-width:1180px; margin:0 auto; display:grid; gap:40px; }
h1 { font-size:1.7rem; margin:0 0 6px; text-wrap:balance; }
h2 { font-size:1.25rem; margin:0; }
h3 { font-size:1.02rem; margin:0; }
p { max-width:72ch; margin:0; }
.intro, .chapter { display:grid; gap:12px; }
.chapter { border-top:2px solid var(--line); padding-top:18px; }
.note { color:var(--muted); font-size:.92rem; }
.key { display:flex; flex-wrap:wrap; gap:10px 22px; font-size:.9rem; color:var(--muted); }
.key span::before { content:""; display:inline-block; width:18px; height:3px; border-radius:2px; margin-right:6px; vertical-align:middle; background:var(--c); }
section.feat { display:grid; gap:8px; border-top:1px solid var(--line); padding-top:14px; }
.head { display:flex; flex-wrap:wrap; align-items:center; gap:8px 14px; }
.pill { font-size:.76rem; letter-spacing:.03em; text-transform:uppercase; padding:2px 9px; border-radius:999px; border:1px solid currentColor; }
.pill.warn { color:var(--warn); } .pill.ok { color:var(--ok); }
figure { margin:0; background:var(--card); border-radius:6px; padding:10px; overflow-x:auto; }
figure svg { width:100%; height:auto; max-width:100%; display:block; }
table { border-collapse:collapse; font-family:var(--mono); font-size:.84rem; font-variant-numeric:tabular-nums; }
th, td { text-align:right; padding:3px 14px 3px 0; border-bottom:1px solid var(--line); }
th:first-child, td:first-child { text-align:left; }
th { color:var(--muted); font-weight:400; }
td.bad { color:var(--warn); font-weight:600; }
.toc { display:flex; flex-wrap:wrap; gap:6px 14px; font-size:.9rem; }
a { color:var(--accent); }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior:auto } }
</style>
"""

parts = [CSS, "<main>", '<div class="intro">', "<h1>peak3d EIC Overlays</h1>",
         "<p>For curated truth features of each run: the extracted ion chromatogram of every file alone on its native "
         "retention-time axis (top row), then all files overlaid on native RT and on the RT axis peak3d produced for "
         "that run (bottom row). The dashed line is the truth RT; the table gives each file's apex before and after "
         "correction. A tightened overlay means the files' apexes agree better after correction than before.</p>",
         '<p class="note">EICs are extracted from the raw MS1 centroids at ±5 ppm; the corrected axis is the warp table '
         "peak3d wrote for the run (rt_correction/&lt;file&gt;.tsv), with the medoid file left on its own axis.</p>"]
toc = []
for label, d in sets:
    d = Path(d)
    summ = json.loads((d / "summary.json").read_text())
    sel = pd.read_csv(d / "selected.tsv", sep="\t")
    toc.append(f'<a href="#{html.escape(label)}">{html.escape(label)}</a>')
parts.append('<p class="toc">' + " ".join(toc) + "</p></div>")

for label, d in sets:
    d = Path(d)
    summ = json.loads((d / "summary.json").read_text())
    sel = pd.read_csv(d / "selected.tsv", sep="\t")
    run = label.split()[0]
    files = [Path(f).stem for f in RUNS[run]["files"]] if run in RUNS else []
    pj = ROOT / "results/peak3d" / run / "peak3d_out/params.json"
    medoid = json.loads(pj.read_text()).get("medoid") if pj.exists() else None
    letters = list(summ[0]["files"].keys()) if summ else []
    key = "".join(f'<span style="--c:var(--s{i + 1})">file {html.escape(k)}{" (medoid)" if medoid and medoid.endswith(k) else ""}</span>'
                  for i, k in enumerate(letters[:3]))
    parts.append(f'<div class="chapter" id="{html.escape(label)}"><h2>{html.escape(label)}</h2>')
    parts.append(f'<p class="note">{len(files)} files: {html.escape(", ".join(files))}. Medoid (output axis): {html.escape(str(medoid))}. '
                 f'{len(summ)} truth features, the strongest eligible one per RT bin.</p><div class="key">{key}</div>')
    for s in summ:
        i = s["index"]
        svg = (d / f"feature_{i:02d}.svg").read_text()
        svg = re.sub(r"<\?xml[^>]*\?>\s*", "", svg)
        svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg)
        nat = [v["nat"] for v in s["files"].values()]
        cor = [v["cor"] for v in s["files"].values()]
        spread_nat = 60 * (max(nat) - min(nat)); spread_cor = 60 * (max(cor) - min(cor))
        pill = ('<span class="pill ok">overlay tightened</span>' if spread_cor <= spread_nat + 0.2 else
                '<span class="pill warn">overlay widened</span>')
        row = sel.iloc[i - 1]
        rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v['nat']:.3f}</td><td>{v['cor']:.3f}</td>"
                       f"<td{' class=bad' if abs(60 * (v['cor'] - v['nat'])) > 10 else ''}>{60 * (v['cor'] - v['nat']):+.1f}</td></tr>"
                       for k, v in s["files"].items())
        parts.append(f'<section class="feat" id="{html.escape(label)}-{i}"><div class="head"><h3>Feature {i}: m/z {s["mz"]:.4f}, RT {s["rt"]:.2f} min</h3>{pill}</div>'
                     f'<p class="note">{html.escape(str(row["truth_id"]))}, tier {html.escape(str(row["tier"]))}. Apex spread across files: '
                     f'{spread_nat:.1f} s native, {spread_cor:.1f} s corrected.</p>'
                     f"<figure>{svg}</figure>"
                     f"<table><thead><tr><th>file</th><th>apex native (min)</th><th>apex corrected (min)</th><th>shift (s)</th></tr></thead><tbody>{rows}</tbody></table></section>")
    parts.append("</div>")
parts.append("</main>")
out.write_text("\n".join(parts))
print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB")
