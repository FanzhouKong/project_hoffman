"""report.html: one self-contained file (styles, scripts and the results bundle inline) that opens in any
browser with no server and no install. The same viewer runs inside ``peak3d gui``.

The bundle travels in two blocks (since 1.2.8): peak3d-data, everything but the MS2 spectra, identity-search hits
and annotations (gzip level 6), and peak3d-ms2, those (bundle.ms2_ids_blob, the blob the app sends its viewer).
A new search or annotation replaces the peak3d-ms2 block only: the main block's text is kept as it is, with no
decompressing and recompressing of the whole results. A report from before keeps all in peak3d-data (and an empty
or no peak3d-ms2); its first refresh splits it."""
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
MS2_RE = re.compile(rb'<script id="peak3d-ms2" type="application/octet-stream">([A-Za-z0-9+/=\s]*)</script>')


def _inline(text: str) -> str:
    return text.replace("</", "<\\/")                      # no early end of the inline <script>


def _page(data_b64: str, ms2_b64: str, title: str) -> str:
    """the report page around two base64 blocks (text, as stored in the file)"""
    page = (STATIC / "report.html").read_text(encoding="utf-8")
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    js = "\n".join(_inline((STATIC / f).read_text(encoding="utf-8")) for f in SCRIPTS)
    for key, val in (("{{TITLE}}", html.escape(title)), ("{{CSS}}", css), ("{{JS}}", js), ("{{MS2}}", ms2_b64),
                     ("{{DATA}}", data_b64)):
        page = page.replace(key, val)
    return page


def render(blob_gz: bytes, title: str, ms2_gz: bytes | None = None) -> str:
    """the report page for a gzip'd bundle, and the gzip'd MS2 part (bundle.ms2_ids_blob) when it travels apart"""
    return _page(base64.b64encode(blob_gz).decode("ascii"),
                 base64.b64encode(ms2_gz).decode("ascii") if ms2_gz else "", title)


def write_report(output, cache_dir=None, path=None, **limits) -> Path:
    """build the bundle for ``output`` and write <output>/report.html (or ``path``): the MS2 part as its own block"""
    output = Path(output)
    try:
        ms2 = bundle.ms2_ids_blob(output)
    except Exception:          # a damaged MS2 or identification file: one block, build() notes the error
        ms2 = None
    blob = gzip.compress(bundle.build(output, cache_dir, with_ms2=ms2 is None, **limits), compresslevel=6, mtime=0)
    path = Path(path) if path else output / "report.html"
    path.write_text(render(blob, f"Mosa · {output.resolve().name}", ms2), encoding="utf-8")
    return path


def _block(text: bytes, rx) -> bytes | None:
    m = rx.search(text)
    if not m:
        return None
    raw = re.sub(rb"\s", b"", m.group(1))
    return base64.b64decode(raw) if raw else None


def read_bundle(report: Path) -> bytes | None:
    """the gzip'd main bundle embedded in a report.html, or None"""
    return _block(Path(report).read_bytes(), DATA_RE)


def read_ms2(report: Path) -> bytes | None:
    """the gzip'd MS2 block of a report.html (bundle.ms2_ids_blob), or None (a report from before 1.2.8)"""
    return _block(Path(report).read_bytes(), MS2_RE)


def read_full(report: Path) -> tuple[dict, dict] | None:
    """(header, arrays) of a report with its MS2 block merged in, as the viewer sees it"""
    text = Path(report).read_bytes()
    blob = _block(text, DATA_RE)
    if blob is None:
        return None
    header, arrays = bundle.unpack(gzip.decompress(blob))
    ms2 = _block(text, MS2_RE)
    if ms2 is not None:
        mh, ma = bundle.unpack(gzip.decompress(ms2))
        arrays = {k: v for k, v in arrays.items() if not k.startswith(bundle.MS2_PREFIXES)}
        arrays.update(ma)
        for k in bundle.MS2_KEYS:
            if k in mh:
                header[k] = mh[k]
            else:
                header.pop(k, None)
    return header, arrays


def refresh_ids(output, ms2_gz: bytes | None = None) -> bool:
    """put the current MS2, identification and annotation parts (ms2_gz: bundle.ms2_ids_blob(output), built here
    when not given) into <output>/report.html after a new identity search or annotation, keeping everything else,
    chromatograms included (their cache is usually gone by then), and the page's code current; False when there is
    no report to update. The main block's text is reused as it is; a report from before 1.2.8 (all in one block) is
    split once"""
    output = Path(output)
    rep = output / "report.html"
    if not rep.exists():
        return False
    text = rep.read_bytes()
    m = DATA_RE.search(text)
    if m is None:
        return False
    if ms2_gz is None:
        ms2_gz = bundle.ms2_ids_blob(output)
    if MS2_RE.search(text) is not None:                 # two blocks: the main one stays as written
        data_b64 = m.group(1).decode("ascii")
    else:                                               # one block from before: without its MS2 part from now on
        header, arrays = bundle.unpack(gzip.decompress(base64.b64decode(re.sub(rb"\s", b"", m.group(1)))))
        arrays = {k: v for k, v in arrays.items() if not k.startswith(bundle.MS2_PREFIXES)}
        for k in bundle.MS2_KEYS:
            header.pop(k, None)
        main = gzip.compress(bundle.pack(bundle._clean({**header, **{k: None for k in bundle.MS2_KEYS}}), arrays),
                             compresslevel=6, mtime=0)
        data_b64 = base64.b64encode(main).decode("ascii")
    del text
    page = _page(data_b64, base64.b64encode(ms2_gz).decode("ascii"), f"Mosa · {output.resolve().name}")
    tmp = rep.with_name(rep.name + ".tmp")
    tmp.write_text(page, encoding="utf-8")
    tmp.replace(rep)                   # a reader never sees a half-written report
    return True
