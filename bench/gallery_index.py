#!/usr/bin/env python3
"""Write data/truth/<DS>/gallery/index.html: every gallery image with its review-sheet row.
Static local page (relative image links); open it from Open OnDemand or after copying the folder.
usage: gallery_index.py DATASET
"""
import csv
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
gal = ROOT / "data/truth" / sys.argv[1] / "gallery"
rows = list(csv.DictReader((gal / "review_sheet.csv").open()))
cards = []
for r in rows:
    meta = (f"{r['truth_id']} &middot; tier {r['tier']} &middot; {html.escape(r['feature_class'])} &middot; "
            f"m/z {r['mz']} &middot; RT {r['rt_min']} min &middot; n={r['n_carbons']} &middot; "
            f"r={r['shape_r']} &middot; n(M+1)={r['n_from_M1'] or 'n/a'} &middot; 12C height {r['height12']}")
    cards.append(f'<figure id="{r["truth_id"]}"><figcaption>{meta}</figcaption>'
                 f'<img loading="lazy" src="{html.escape(r["image"])}" alt="{r["truth_id"]}"></figure>')
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{sys.argv[1]} credentialed gallery</title>
<style>
body{{font:14px/1.4 system-ui,sans-serif;margin:16px;background:#fff;color:#222}}
figure{{margin:0 0 28px;border-top:1px solid #ddd;padding-top:8px}}
figcaption{{font-weight:600;margin-bottom:4px}} img{{max-width:100%;height:auto}}
</style></head><body>
<h1>{sys.argv[1]}: credentialed truth gallery ({len(rows)} images)</h1>
<p>Record verdicts (real / not_real / unsure) in <code>review_sheet.csv</code> next to this page.
Left: 12C replicates at m/z. Middle: 13C replicates at m/z + n&times;1.00335. Right: normalized means
(dashed = natural 12C M+1, used for the tier A carbon check).</p>
{''.join(cards)}
</body></html>"""
(gal / "index.html").write_text(page)
print(f"wrote {gal / 'index.html'} ({len(rows)} images)")
