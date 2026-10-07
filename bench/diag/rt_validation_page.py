#!/usr/bin/env python3
"""Build results/figures/rt_validation/index.html: external validation of peak3d's RT correction."""
import base64
from pathlib import Path
import pandas as pd

D = Path(__file__).resolve().parents[2] / "results/figures/rt_validation"
S = pd.read_csv(D / "summary.tsv", sep="\t")
ORDER = ["HZV029_full", "HZV029_cert", "YEAST_12C", "YEAST_13C", "YEAST_NEG", "SZ22_12C", "SZ22_13C", "LI2018"]
LABEL = {"HZV029_full": "HZV029, 268 plasma QC injections over 17 batches; 402 certified features",
         "HZV029_cert": "HZV029, the 3 certified files; 402 certified features",
         "YEAST_12C": "yeast 2021 (NetID) 12C, credentialed tier-A features", "YEAST_13C": "yeast 2021 (NetID) 13C, credentialed tier-A features",
         "YEAST_NEG": "yeast 2021 (NetID) negative mode, 314 manually curated compounds",
         "SZ22_12C": "SZ22 E. coli 12C, credentialed tier-A features", "SZ22_13C": "SZ22 E. coli 13C, credentialed tier-A features",
         "LI2018": "MTBLS733 (Li 2018), 836 confirmed standards in 10 files"}


def img(p):
    return f'<img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" alt="{p.stem}">'


rows = []
for r in ORDER:
    s = S[S.run == r]
    if not len(s):
        continue
    s = s.iloc[0]
    rows.append(f"<tr><td>{r}</td><td class=n>{int(s.n_files)}</td><td class=n>{int(s.n_truth_used)}</td><td class=n>{s.dt_s:.2f}</td>"
                f"<td class=n>{s.spread_native_med_s:.2f} → <b>{s.spread_corr_med_s:.2f}</b></td><td class=n>{s.spread_native_p90_s:.2f} → <b>{s.spread_corr_p90_s:.2f}</b></td>"
                f"<td class=n>{s.frac_tightened:.0%}</td><td class=n>{s.file_bias_native_mad_s:.2f} → <b>{s.file_bias_corr_mad_s:.2f}</b></td></tr>")
figs = []
for r in ORDER:
    p = D / f"{r}.png"
    if p.exists():
        figs.append(f"<section><h2>{r}</h2><p class=sub>{LABEL[r]}</p>{img(p)}</section>")
li = D / "LI2018_warp_vs_truth.png"
li_html = f"<section><h2>LI2018: a compound-dependent drift that no RT-only warp can remove</h2><p class=sub>Blue: the shift each truth feature implies for the file (its consensus RT minus its native apex). Black: the shift the fitted warp applies. Gold: what is left after correction. Between 4 and 20 min the truth features split into two populations: about two thirds drift earlier by 2–11 s from SA1 to SB5 in acquisition order, one third does not move at all. The warp follows whichever population dominates the anchors in each region, so it removes most of the drift of the moving compounds up to 10 min and leaves the others displaced by 1–3 s there; overall the median absolute residual in that window is unchanged (1.36 → 1.40 s) and outside it (0.27 → 0.29 s).</p>{img(li)}</section>" if li.exists() else ""
html = f"""<title>RT Correction Validation</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1f2328;--fg2:#59636e;--line:#d8dde3;--accent:#0969da;--card:#ffffff}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#0f1216;--fg:#e6e9ee;--fg2:#9aa4b2;--line:#2a313b;--accent:#58a6ff;--card:#161b22;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#0f1216;--fg:#e6e9ee;--fg2:#9aa4b2;--line:#2a313b;--accent:#58a6ff;--card:#161b22;color-scheme:dark}}
body{{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif;max-width:1500px;margin:0 auto;padding-block:24px;padding-inline:16px}}
h1{{font-size:22px;margin:0 0 4px;text-wrap:balance}} h2{{font-size:16px;margin:28px 0 2px}} p.sub{{color:var(--fg2);margin:0 0 10px;max-width:70ch}}
p.lead{{color:var(--fg2);max-width:80ch;margin:0 0 18px}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums;margin:8px 0 12px;min-width:0}} .wrap{{overflow-x:auto}}
th,td{{padding:5px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}} th{{color:var(--fg2);font-weight:600;font-size:12px;letter-spacing:.02em}} td.n{{text-align:right}}
img{{max-width:100%;height:auto;display:block;border:1px solid var(--line);border-radius:4px;background:#fff}}
section{{min-width:0}}
</style>
<h1>External validation of peak3d's RT correction</h1>
<p class=lead>Truth lists the alignment never sees: HZV029 certified features, yeast 2021 credentialed and NetID-curated features, SZ22 credentialed features, MTBLS733 confirmed standards. Each truth feature is matched in every file (5 ppm, 0.15 min, strongest peak); the spread of its apex RT across files is 1.4826 × MAD, before (native) and after (corrected) the warp. The per-file bias is the median deviation of all truth features in that file from their cross-file medians.</p>
<div class=wrap><table><thead><tr><th>run</th><th>files</th><th>truth used</th><th>scan dt (s)</th><th>median spread (s) native → corrected</th><th>p90 spread (s)</th><th>features tightened</th><th>per-file bias MAD (s)</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
{''.join(figs)}
{li_html}
"""
(D / "index.html").write_text(html)
print("wrote", D / "index.html", f"{(D / 'index.html').stat().st_size / 1e6:.1f} MB")
