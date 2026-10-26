"""report.html: one self-contained file (styles, scripts and the results bundle inline) that opens in any
browser with no server and no install. The same viewer runs inside ``peak3d gui``."""
from __future__ import annotations

import base64
import gzip
import html
import re
from pathlib import Path

from . import bundle

STATIC = Path(__file__).with_name("static")
SCRIPTS = ("icons.js", "core.js", "plot.js", "viewer.js")       # load order; the GUI adds app.js
DATA_RE = re.compile(rb'<script id="peak3d-data" type="application/octet-stream">([A-Za-z0-9+/=\s]*)</script>')


def _inline(text: str) -> str:
    return text.replace("</", "<\\/")                      # no early end of the inline <script>


def render(blob_gz: bytes, title: str) -> str:
    page = (STATIC / "report.html").read_text(encoding="utf-8")
    css = (STATIC / "m3.css").read_text(encoding="utf-8")
    js = "\n".join(_inline((STATIC / f).read_text(encoding="utf-8")) for f in SCRIPTS)
    font = base64.b64encode((STATIC / "roboto-latin.woff2").read_bytes()).decode("ascii")
    data = base64.b64encode(blob_gz).decode("ascii")
    for key, val in (("{{TITLE}}", html.escape(title)), ("{{FONT}}", font), ("{{CSS}}", css), ("{{JS}}", js),
                     ("{{DATA}}", data)):
        page = page.replace(key, val)
    return page


def write_report(output, cache_dir=None, path=None, **limits) -> Path:
    """build the bundle for ``output`` and write <output>/report.html (or ``path``)"""
    output = Path(output)
    blob = gzip.compress(bundle.build(output, cache_dir, **limits), compresslevel=6, mtime=0)
    path = Path(path) if path else output / "report.html"
    path.write_text(render(blob, f"peak3d · {output.resolve().name}"), encoding="utf-8")
    return path


def read_bundle(report: Path) -> bytes | None:
    """the gzip'd bundle embedded in a report.html, or None"""
    m = DATA_RE.search(Path(report).read_bytes())
    return base64.b64decode(re.sub(rb"\s", b"", m.group(1))) if m else None
