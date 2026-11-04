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
SCRIPTS = ("logo.js", "icons.js", "core.js", "plot.js", "smiles-drawer.min.js", "viewer.js")   # load order; GUI adds app.js
DATA_RE = re.compile(rb'<script id="peak3d-data" type="application/octet-stream">([A-Za-z0-9+/=\s]*)</script>')


def _inline(text: str) -> str:
    return text.replace("</", "<\\/")                      # no early end of the inline <script>


def render(blob_gz: bytes, title: str) -> str:
    page = (STATIC / "report.html").read_text(encoding="utf-8")
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    js = "\n".join(_inline((STATIC / f).read_text(encoding="utf-8")) for f in SCRIPTS)
    data = base64.b64encode(blob_gz).decode("ascii")
    for key, val in (("{{TITLE}}", html.escape(title)), ("{{CSS}}", css), ("{{JS}}", js), ("{{DATA}}", data)):
        page = page.replace(key, val)
    return page


def write_report(output, cache_dir=None, path=None, **limits) -> Path:
    """build the bundle for ``output`` and write <output>/report.html (or ``path``)"""
    output = Path(output)
    blob = gzip.compress(bundle.build(output, cache_dir, **limits), compresslevel=6, mtime=0)
    path = Path(path) if path else output / "report.html"
    path.write_text(render(blob, f"Mosa · {output.resolve().name}"), encoding="utf-8")
    return path


def read_bundle(report: Path) -> bytes | None:
    """the gzip'd bundle embedded in a report.html, or None"""
    m = DATA_RE.search(Path(report).read_bytes())
    return base64.b64decode(re.sub(rb"\s", b"", m.group(1))) if m else None


def refresh_ids(output) -> bool:
    """swap the MS2, identification and annotation parts of <output>/report.html's bundle for the current files
    (after a new identity search or annotation), keeping everything else, chromatograms included (their cache is
    usually gone by then); False when there is no report to update"""
    output = Path(output)
    rep = output / "report.html"
    blob = read_bundle(rep) if rep.exists() else None
    if blob is None:
        return False
    header, arrays = bundle.unpack(gzip.decompress(blob))
    arrays = {k: v for k, v in arrays.items() if not k.startswith(bundle.MS2_PREFIXES)}
    for k in bundle.MS2_KEYS:
        header.pop(k, None)
    head, extra = bundle.ms2_ids(output, arrays["g_id"], header.get("stems"))
    arrays.update(extra)
    blob = gzip.compress(bundle.pack(bundle._clean({**header, **head}), arrays), compresslevel=6, mtime=0)
    tmp = rep.with_name(rep.name + ".tmp")
    tmp.write_text(render(blob, f"Mosa · {output.resolve().name}"), encoding="utf-8")
    tmp.replace(rep)                   # a reader never sees a half-written report
    return True
