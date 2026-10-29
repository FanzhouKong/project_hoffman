"""``peak3d gui``: a local web app on the Python standard library. The browser is only the window; runs
execute here as ``python -m peak3d process ...`` subprocesses, so the GUI computes exactly what the CLI
computes and a run survives a closed browser tab.

Security: the server reads folders and starts processes, so every request needs the session token
(printed in the start-up link, then kept in a SameSite cookie), the Host header must be an IP literal or
localhost (blocks DNS rebinding), and ``--root`` confines browsing, inputs and outputs to one folder."""
from __future__ import annotations

import gzip
import html
import ipaddress
import json
import mimetypes
import os
import pickle
import platform
import re
import secrets
import signal
import string
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .. import __version__
from ..estimate import VARIANT, VARIANT_SETS
from ..library import LIBRARY_SUFFIXES
from . import bundle, report

STATIC = report.STATIC
APP_SCRIPTS = (*report.SCRIPTS, "app.js")
COOKIE = "peak3d_token"
# how to start `peak3d ...`: inside the standalone app the program itself is the command line
PEAK3D_CMD = [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, "-m", "peak3d"]
# set: picker overrides stay on the command line, so a GUI run is always the variant the bar shows
SKIP_OPTIONS = {"help", "input", "output", "cache_dir", "set", "kernel"}
KERNELS = ("peak3d", "asari", "masscube")      # peak3d runs `process`; the others `external --kernel NAME`
VARIANT_NOTE = " ".join(f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in VARIANT_SETS.items())
# output files the viewer may download (by name, from a peak3d output folder)
RESULT_FILES = ("aligned_feature_table.tsv", "aligned_feature_area.tsv", "aligned_feature_height.tsv", "filled_mask.tsv",
                "presence_mask.tsv", "params.json", "ms2_spectra.msp", "ms2_map.tsv", "identifications.tsv")
RT_COLS = ("rt", "rt_min", "rt_max", "rt_sd")    # minutes in the tables; the GUI shows and exports seconds

# stage markers in the pipeline log -> (stage, progress fraction when reached)
STAGES = [("JIT warm-up", "pick", 0.05), ("picked ", "align", 0.80), ("alignment:", "group", 0.84),
          ("grouped ", "fill", 0.88), ("gap-filled", "write", 0.93), ("presence rule", "write", 0.95),
          ("report", "report", 0.97),
          # external kernels (peak3d external): the tool's own run, then conversion, chromatograms and MS2
          ("kernel: running", "pick", 0.05), ("kernel: converting", "group", 0.70),
          ("for chromatograms and MS2", "fill", 0.75), ("ms2:", "fill", 0.90), ("identify:", "fill", 0.93)]


# ------------------------------------------------------------------ options ----
def _probe(conv, text):
    try:
        return conv(text)
    except Exception:
        return None


def option_schema(command: str = "process"):
    """the options of ``peak3d <command>`` (process, or external for the other kernels) as form fields, read from
    the CLI parser so the two cannot drift apart"""
    from ..__main__ import build_parser
    ap = build_parser()
    sub = next(a for a in ap._actions if a.__class__.__name__ == "_SubParsersAction")
    if command not in sub.choices:
        return []
    fields = []
    for a in sub.choices[command]._actions:
        if a.dest in SKIP_OPTIONS or not a.option_strings:
            continue
        f = {"dest": a.dest, "flag": max(a.option_strings, key=len), "help": a.help or "",
             "default": a.default if not isinstance(a.default, Path) else str(a.default)}
        kind = a.__class__.__name__
        if kind == "_StoreTrueAction":
            f["kind"] = "flag"
        elif kind == "_AppendAction":
            f["kind"], f["default"] = "list", []
        elif a.choices:
            f["kind"], f["choices"] = "choice", list(a.choices)
        elif a.type is Path:
            f["kind"] = "path"
        elif a.type is not None:
            one = _probe(a.type, "1")
            f["kind"] = "int" if isinstance(one, int) and _probe(a.type, "0.5") is None else "float"
            if _probe(a.type, "0") is None:
                f["min_exclusive"] = 0
            if _probe(a.type, "1.5") is None and one is not None:
                f["max"] = 1
        else:
            f["kind"] = "text"
        fields.append(f)
    return fields


def options_to_argv(options: dict, schema) -> list[str]:
    """form values -> CLI arguments (only values that differ from the defaults)"""
    argv = []
    for f in schema:
        if f["dest"] not in options:
            continue
        v = options[f["dest"]]
        if f["kind"] == "flag":
            if v:
                argv.append(f["flag"])
        elif f["kind"] == "list":
            for item in v or []:
                if str(item).strip():
                    argv += [f["flag"], str(item).strip()]
        elif v is not None and str(v).strip() != "" and str(v) != str(f["default"]):
            argv += [f["flag"], str(v).strip()]
    return argv


def validate_argv(argv: list[str]) -> str | None:
    """None if the CLI accepts these arguments, else argparse's message"""
    import contextlib
    import io

    from ..__main__ import build_parser
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            build_parser().parse_args(argv)
    except SystemExit:
        lines = [ln for ln in err.getvalue().splitlines() if ln.strip()]
        return lines[-1].split("error: ", 1)[-1] if lines else "invalid options"
    return None


# ------------------------------------------------------------------ runs ----
def _mzml_files(folder: Path) -> list[Path]:
    try:
        return sorted(p for p in folder.iterdir() if p.suffix.lower() == ".mzml" and p.is_file())
    except OSError:
        return []


def command_args(kernel: str) -> list[str]:
    """the peak3d subcommand (and kernel flag) that runs this kernel"""
    return ["process"] if kernel == "peak3d" else ["external", "--kernel", kernel]


class Run:
    def __init__(self, rid: str, input_dir: Path, output: Path, argv: list[str], n_files: int, kernel: str = "peak3d"):
        self.id, self.input, self.output, self.argv, self.n_files = rid, input_dir, output, argv, n_files
        self.kernel = kernel
        self.cmd = [*PEAK3D_CMD, *command_args(kernel), "--input", str(input_dir), "--output", str(output), *argv]
        self.lines: list[str] = []
        self.state, self.stage, self.base = "running", "compile", 0.0
        self.t0, self.t1, self.returncode, self.proc = time.time(), None, None, None
        self.cancelled = False

    def start(self):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        kw = {"start_new_session": True} if os.name == "posix" else \
            {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)}
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace", bufsize=1, env=env, **kw)
        threading.Thread(target=self._follow, daemon=True).start()

    def _follow(self):
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            self.lines.append(line)
            for marker, stage, frac in STAGES:
                if marker in line and line.startswith("[peak3d]") and frac > self.base:
                    self.stage, self.base = stage, frac
        self.returncode = self.proc.wait()
        self.t1 = time.time()
        try:
            (self.output / "peak3d_gui.log").write_text("$ " + " ".join(self.cmd) + "\n" + "\n".join(self.lines) + "\n",
                                                       encoding="utf-8")
        except OSError:
            pass
        self.state = "cancelled" if self.cancelled else ("done" if self.returncode == 0 else "failed")
        self.stage = self.state

    def cancel(self):
        if self.proc is None or self.proc.poll() is not None:
            return
        self.cancelled = True
        if os.name == "posix":
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:  # the spawned pool workers are children: end the whole tree
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def progress(self) -> float:
        if self.state == "done":
            return 1.0
        if self.stage == "pick" and self.n_files and self.kernel == "peak3d":
            fd = self.output / "features"
            try:
                k = sum(1 for p in fd.glob("*.tsv") if p.name != "manifest.tsv" and not p.name.endswith(".rejected.tsv")
                        and p.stat().st_mtime >= self.t0 - 1)
            except OSError:
                k = 0
            self.picked = min(k, self.n_files)
            return 0.05 + 0.75 * self.picked / self.n_files
        return self.base

    def status(self, since: int = 0) -> dict:
        p = self.progress()
        return {"id": self.id, "kernel": self.kernel, "input": str(self.input), "output": str(self.output), "argv": self.argv,
                "command": " ".join(["peak3d", *command_args(self.kernel), "--input", _q(self.input), "--output",
                                     _q(self.output), *map(_q, self.argv)]),
                "state": self.state, "stage": self.stage, "progress": round(p, 4), "n_files": self.n_files,
                "picked": getattr(self, "picked", None), "started": self.t0,
                "elapsed": (self.t1 or time.time()) - self.t0, "returncode": self.returncode,
                "log": self.lines[since:], "log_len": len(self.lines)}


def _q(s) -> str:
    s = str(s)
    return s if re.fullmatch(r"[\w./:=+@%,\\-]+", s) else '"' + s.replace('"', '\\"') + '"'


def _seconds(v: str) -> str:
    """a retention time cell in minutes -> seconds (empty and nan stay as they are)"""
    try:
        x = float(v)
    except ValueError:
        return v
    return f"{x * 60:.6g}" if x == x else v


def _identity_columns(out: Path) -> dict:
    """MS2 count and best library hit per group id, as extra export columns (none when the output has no MS2)"""
    head = ("n_ms2", "identity", "identity_score", "identity_adduct", "identity_inchikey")
    map_path, hits_path = out / "ms2_map.tsv", out / "identifications.tsv"
    if not map_path.exists() or not (out / "ms2_spectra.msp").exists():
        return {"header": (), "rows": {}, "empty": []}
    import pandas as pd
    m = pd.read_csv(map_path, sep="\t", usecols=["group_id"])["group_id"].value_counts()
    best = {}
    if hits_path.exists():
        h = pd.read_csv(hits_path, sep="\t", dtype={"name": str, "precursor_type": str, "inchikey": str})
        h = h.fillna({"name": "", "precursor_type": "", "inchikey": ""})
        for r in h[h["rank"] == 1].itertuples():
            best[int(r.group_id)] = [r.name, f"{r.score:.4f}", r.precursor_type, r.inchikey]
    rows = {int(g): [str(int(n))] + best.get(int(g), ["", "", "", ""]) for g, n in m.items()}
    return {"header": head, "rows": rows, "empty": ["0", "", "", "", ""]}


# ------------------------------------------------------------------ app ----
class App:
    def __init__(self, root: Path | None, token: str | None, allowed_hosts=()):
        self.root = Path(root).expanduser().resolve() if root else None
        self.token = token
        self.allowed_hosts = {h.lower() for h in allowed_hosts}
        self.runs: dict[str, Run] = {}
        self.schema = option_schema()
        self.schemas = {"peak3d": self.schema, "external": option_schema("external")}
        self._bundles: dict[str, tuple] = {}
        self._ms2: dict[str, tuple] = {}
        self._searching: dict[str, threading.Lock] = {}     # one identity search per output folder at a time
        self._tables: dict[str, tuple] = {}
        self.lock = threading.Lock()

    def resolve(self, p: str | None) -> Path:
        base = self.root or Path.home()
        path = (base / Path(p).expanduser()) if p else base
        path = path.resolve()
        if self.root and not (path == self.root or self.root in path.parents):
            raise PermissionError(f"outside the allowed folder {self.root}")
        return path

    def bundle_for(self, output: Path) -> bytes:
        """gzip'd bundle: the one embedded in report.html when it is current, else built now"""
        table = output / "aligned_feature_table.tsv"
        rep = output / "report.html"
        key = (table.stat().st_mtime, rep.stat().st_mtime if rep.exists() else 0)
        hit = self._bundles.get(str(output))
        if hit and hit[0] == key:
            return hit[1]
        blob = report.read_bundle(rep) if rep.exists() and key[1] >= key[0] else None
        if blob is None:
            blob = gzip.compress(bundle.build(output), compresslevel=6, mtime=0)
        self._bundles[str(output)] = (key, blob)
        return blob


    def ms2_blob(self, output: Path) -> bytes:
        """gzip'd MS2 + identification parts of the bundle, rebuilt when those files change"""
        key = tuple((output / n).stat().st_mtime if (output / n).exists() else 0
                    for n in ("ms2_spectra.msp", "ms2_map.tsv", "identifications.tsv", "identifications.msp"))
        hit = self._ms2.get(str(output))
        if hit and hit[0] == key:
            return hit[1]
        blob = bundle.ms2_ids_blob(output)
        self._ms2[str(output)] = (key, blob)
        return blob

    def table_lines(self, path: Path):
        """header and {group_id: line} of an aligned table, kept as text so values stay exactly as written"""
        mtime = path.stat().st_mtime
        hit = self._tables.get(str(path))
        if hit and hit[0] == mtime:
            return hit[1]
        with open(path, encoding="utf-8", newline="") as fh:
            header = fh.readline()
            rows = {line.split("\t", 1)[0]: line for line in fh}
        if len(self._tables) > 3:
            self._tables.clear()
        self._tables[str(path)] = (mtime, (header, rows))
        return header, rows


class Handler(BaseHTTPRequestHandler):
    server_version = f"peak3d/{__version__}"
    app: App

    def log_message(self, fmt, *args):  # quiet: the terminal shows the start-up link and run logs only
        pass

    # ---- plumbing
    def _send(self, code, body: bytes, ctype: str, headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, default=str).encode(), "application/json")

    def _error(self, code, msg):
        self._json({"error": msg}, code)

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        if host.startswith("["):
            host = host[1:host.find("]")] if "]" in host else host
        elif host.count(":") == 1:
            host = host.split(":")[0]
        if host == "localhost" or host.endswith(".localhost") or host in self.app.allowed_hosts:
            return True
        try:
            ipaddress.ip_address(host)
            return True
        except ValueError:
            return False

    def _authed(self, query) -> bool:
        tok = self.app.token
        if tok is None:
            return True
        cookie = self.headers.get("Cookie") or ""
        got = re.search(rf"(?:^|;\s*){COOKIE}=([\w-]+)", cookie)
        cands = [got.group(1) if got else "", self.headers.get("X-Peak3d-Token") or "", (query.get("token") or [""])[0]]
        return any(secrets.compare_digest(c, tok) for c in cands if c)

    def _guard(self, query) -> bool:
        if not self._host_ok():
            self._send(403, b"Forbidden host", "text/plain")
            return False
        if not self._authed(query):
            self._send(403, ("<!doctype html><meta charset=utf-8><title>peak3d</title><body style='font:16px system-ui;"
                             "padding:40px'><h2>peak3d</h2><p>Open the link printed in the terminal (it carries the "
                             "access token).</p>").encode(), "text/html; charset=utf-8")
            return False
        return True

    # ---- routes
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if not self._guard(q):
            return
        if "token" in q and url.path == "/":
            self._send(302, b"", "text/plain", {"Location": "/", "Set-Cookie":
                       f"{COOKIE}={self.app.token}; Path=/; HttpOnly; SameSite=Strict"})
            return
        try:
            if url.path == "/":
                return self._index()
            if url.path.startswith("/static/"):
                return self._static(url.path[len("/static/"):])
            if url.path == "/api/info":
                return self._json(self._info())
            if url.path == "/api/schema":
                kernel = q.get("kernel", ["peak3d"])[0]
                return self._json(self.app.schemas["peak3d" if kernel == "peak3d" else "external"])
            if url.path == "/api/ls":
                return self._json(self._ls(q.get("path", [""])[0], q.get("ext", [""])[0]))
            if url.path == "/api/runs":
                return self._json([r.status(len(r.lines)) for r in self.app.runs.values()])
            m = re.fullmatch(r"/api/runs/(\w+)", url.path)
            if m:
                r = self.app.runs.get(m.group(1))
                if r is None:
                    return self._error(404, "no such run")
                return self._json(r.status(int(q.get("since", ["0"])[0])))
            if url.path == "/api/results/bundle":
                out = self._output(q)
                return self._send(200, self.app.bundle_for(out), "application/octet-stream")
            if url.path == "/api/results/ms2ids":
                return self._send(200, self.app.ms2_blob(self._output(q)), "application/octet-stream")
            if url.path == "/api/results/file":
                out = self._output(q)
                name = q.get("name", [""])[0]
                if name not in RESULT_FILES or not (out / name).exists():
                    return self._error(404, f"no file {name!r} in {out}")
                ctype = ("application/json" if name.endswith(".json") else "text/plain; charset=utf-8"
                         if name.endswith(".msp") else "text/tab-separated-values; charset=utf-8")
                return self._send(200, (out / name).read_bytes(), ctype,
                                  {"Content-Disposition": f'attachment; filename="{out.name}_{name}"'})
            if url.path == "/api/results/report":
                out = self._output(q)
                rep = out / "report.html"
                body = rep.read_bytes() if rep.exists() else report.render(
                    gzip.compress(bundle.build(out), mtime=0), f"peak3d · {out.name}").encode()
                disp = "attachment" if q.get("download") else "inline"
                return self._send(200, body, "text/html; charset=utf-8",
                                  {"Content-Disposition": f'{disp}; filename="peak3d_report_{out.name}.html"'})
            self._error(404, "not found")
        except PermissionError as e:
            self._error(403, str(e))
        except FileNotFoundError as e:
            self._error(404, str(e))
        except Exception as e:  # a viewer request never takes the server down
            self._error(500, f"{type(e).__name__}: {e}")

    def do_POST(self):
        url = urlparse(self.path)
        if not self._guard({}):
            return
        if self.headers.get("X-Peak3d-Token") is None and self.app.token is not None:
            return self._error(403, "missing token header")     # forms from other sites cannot set it
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._error(400, "bad JSON")
        try:
            if url.path == "/api/runs":
                return self._start(body)
            if url.path == "/api/example":
                return self._example()
            if url.path == "/api/results/table":
                return self._table(body)
            if url.path == "/api/results/reveal":
                return self._reveal(body)
            if url.path == "/api/results/identify":
                return self._identify(body)
            m = re.fullmatch(r"/api/runs/(\w+)/cancel", url.path)
            if m and m.group(1) in self.app.runs:
                self.app.runs[m.group(1)].cancel()
                return self._json({"ok": True})
            self._error(404, "not found")
        except PermissionError as e:
            self._error(403, str(e))
        except FileNotFoundError as e:
            self._error(404, str(e))
        except Exception as e:
            self._error(500, f"{type(e).__name__}: {e}")

    # ---- handlers
    def _index(self):
        page = (STATIC / "index.html").read_text(encoding="utf-8")
        tags = "\n".join(f'<script src="/static/{s}?v={__version__}"></script>' for s in APP_SCRIPTS)
        page = page.replace("{{SCRIPTS}}", tags).replace("{{VERSION}}", __version__) \
            .replace("{{VARIANT}}", html.escape(VARIANT)).replace("{{VARIANT_NOTE}}", html.escape(VARIANT_NOTE)) \
            .replace("{{TOKEN}}", self.app.token or "")       # POSTs echo it in a header; the cookie is HttpOnly
        self._send(200, page.encode(), "text/html; charset=utf-8")

    def _static(self, name):
        allowed = {p.name for p in STATIC.iterdir() if p.suffix in (".js", ".css")}
        if name not in allowed:
            return self._error(404, "not found")
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or name.endswith(".js"):
            ctype = ("text/javascript" if name.endswith(".js") else ctype) + "; charset=utf-8"
        self._send(200, (STATIC / name).read_bytes(), ctype)

    def _info(self):
        a = self.app
        try:
            from ..external import kernel_status
            kernels = kernel_status()
        except ImportError:
            kernels = {"peak3d": {"available": True, "version": __version__, "variant": VARIANT}}
        try:
            from ..library import METHODS
        except ImportError:
            METHODS = {}
        return {"version": __version__, "variant": VARIANT, "variant_sets": VARIANT_SETS, "kernels": kernels,
                "methods": METHODS,
                "platform": platform.system(), "python": platform.python_version(),
                "cpu_count": os.cpu_count() or 1, "root": str(a.root) if a.root else None,
                "start": str(a.root or Path.home()), "sep": os.sep}

    def _ls(self, p, ext=""):
        a = self.app
        if os.name == "nt" and p == "::drives" and a.root is None:
            drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
            return {"path": "", "parent": None, "dirs": drives, "n_mzml": 0, "mzml": [], "is_output": False, "files": []}
        path = a.resolve(p)
        if not path.is_dir():
            raise FileNotFoundError(f"not a folder: {path}")
        dirs, files = [], []
        suffix = LIBRARY_SUFFIXES if ext == "msp" else None            # file picking: spectral libraries only
        try:
            for e in os.scandir(path):
                if e.name.startswith("."):
                    continue
                if e.is_dir(follow_symlinks=True):
                    dirs.append(e.name)
                elif suffix and e.name.lower().endswith(suffix) and e.is_file():
                    files.append(e.name)
        except PermissionError:
            pass
        dirs.sort(key=str.lower)
        files.sort(key=str.lower)
        mz = _mzml_files(path)
        if a.root is not None and path == a.root:
            parent = None
        elif path.parent == path:
            parent = "::drives" if os.name == "nt" else None
        else:
            parent = str(path.parent)
        return {"path": str(path), "parent": parent, "dirs": dirs, "n_mzml": len(mz), "mzml": [f.name for f in mz[:8]],
                "is_output": (path / "aligned_feature_table.tsv").exists(), "has_report": (path / "report.html").exists(),
                "writable": os.access(path, os.W_OK), "files": files[:2000]}

    def _output(self, q) -> Path:
        out = self.app.resolve(q.get("path", [""])[0])
        if not (out / "aligned_feature_table.tsv").exists():
            raise FileNotFoundError(f"no peak3d results in {out}")
        return out

    def _table(self, body):
        """the rows of an aligned table for the given group ids, in that order, as in the file except retention times
        in seconds (columns renamed rt_s, ...), plus the MS2 count and best library hit when the output has them"""
        out = self._output({"path": [body.get("path") or ""]})
        name = body.get("name") or "aligned_feature_table.tsv"
        if name not in RESULT_FILES or not name.startswith("aligned_feature"):
            return self._error(400, f"not a feature table: {name!r}")
        header, rows = self.app.table_lines(out / name)
        cols = header.rstrip("\r\n").split("\t")
        eol = header[len(header.rstrip("\r\n")):] or "\n"
        rt_at = [i for i, c in enumerate(cols) if c in RT_COLS]
        extra = _identity_columns(out)
        lines = ["\t".join([f"{c}_s" if c in RT_COLS else c for c in cols] + list(extra["header"])) + eol]
        for k in (str(int(i)) for i in body.get("ids") or []):
            if k not in rows:
                continue
            v = rows[k].rstrip("\r\n").split("\t")
            for i in rt_at:
                v[i] = _seconds(v[i])
            lines.append("\t".join(v + extra["rows"].get(int(k), extra["empty"])) + eol)
        self._send(200, "".join(lines).encode("utf-8"), "text/tab-separated-values; charset=utf-8")

    def _identify(self, body):
        """identity search of this output's MS2 spectra against a .msp or .pkl library (peak3d.library.identify); rewrites
        identifications.* and refreshes report.html, then answers with the search summary"""
        from ..library import METHODS, identify
        out = self._output({"path": [body.get("path") or ""]})
        lib = self.app.resolve(body.get("library") or "")
        if lib.suffix.lower() not in LIBRARY_SUFFIXES or not lib.is_file():
            return self._error(400, f"not a spectral library (.msp or .pkl): {lib}")
        method = body.get("method") or "weighted_entropy"
        if method not in METHODS:
            return self._error(400, f"unknown similarity {method!r}")
        try:
            ppm, tol, min_score = (float(body.get(k, d)) for k, d in
                                   (("precursor_ppm", 10.0), ("fragment_tol", 0.02), ("min_score", 0.7)))
        except (TypeError, ValueError):
            return self._error(400, "tolerances and minimum score must be numbers")
        if not (ppm > 0 and tol > 0 and 0 <= min_score <= 1):
            return self._error(400, "tolerances must be > 0 and the minimum score within [0, 1]")
        with self.app.lock:
            lock = self.app._searching.setdefault(str(out), threading.Lock())
        if not lock.acquire(blocking=False):
            return self._error(409, "a search is already running for these results")
        try:
            try:
                identify(out, lib, method=method, precursor_ppm=ppm, fragment_tol=tol, min_score=min_score)
            except (pickle.UnpicklingError, ValueError, EOFError) as e:     # a .pkl that is not a library we can read
                return self._error(400, f"could not read the library {lib.name}: {e}")
            try:
                report.refresh_ids(out)
            except OSError:
                pass                              # a read-only results folder: the GUI still shows the hits
        finally:
            lock.release()
        meta = out / "identifications.json"
        self._json(json.loads(meta.read_text()) if meta.exists() else {})

    def _reveal(self, body):
        """open the folder in Finder / File Explorer / the desktop's file manager (the app runs on this computer)"""
        out = self.app.resolve(body.get("path") or "")
        if not out.is_dir():
            raise FileNotFoundError(f"not a folder: {out}")
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(out)])
            elif os.name == "nt":
                os.startfile(str(out))
            else:
                subprocess.Popen(["xdg-open", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as e:
            return self._error(500, f"could not open the folder: {e}")
        self._json({"ok": True})

    def _example(self):
        """three small synthetic runs (the `peak3d demo` data) in <root>/peak3d_example, for a first try"""
        from ..__main__ import write_demo
        out = self.app.resolve("peak3d_example")
        try:
            data = write_demo(out)
        except OSError as e:
            return self._error(403, f"cannot write example data in {out.parent}: {e.strerror or e}")
        return self._json({"input": str(data), "output": str(out / "peak3d_out"), "n_mzml": len(_mzml_files(data))})

    def _start(self, body):
        a = self.app
        kernel = body.get("kernel") or "peak3d"
        if kernel not in KERNELS:
            return self._error(400, f"unknown kernel {kernel!r}")
        if kernel != "peak3d":
            st = self._info()["kernels"].get(kernel) or {}
            if not st.get("available"):
                return self._error(400, f"{kernel} is not available here. {st.get('note') or ''}".strip())
        inp = a.resolve(body.get("input") or "")
        files = _mzml_files(inp)
        if not files:
            return self._error(400, f"no .mzML files in {inp}")
        if not body.get("output"):
            return self._error(400, "choose an output folder")
        out = a.resolve(body["output"])
        if any(r.state == "running" and r.output == out for r in a.runs.values()):
            return self._error(409, "a run is already writing to this folder")
        if (out / "aligned_feature_table.tsv").exists() and not body.get("overwrite"):
            return self._json({"error": "this folder already holds peak3d results", "confirm": "overwrite"}, 409)
        argv = options_to_argv(body.get("options") or {}, a.schemas["peak3d" if kernel == "peak3d" else "external"])
        msg = validate_argv([*command_args(kernel), "--input", str(inp), "--output", str(out), *argv])
        if msg:
            return self._error(400, msg)
        with a.lock:
            rid = f"r{len(a.runs) + 1}"
            run = Run(rid, inp, out, argv, len(files), kernel)
            a.runs[rid] = run
        run.start()
        self._json(run.status(), 201)


# ------------------------------------------------------------------ entry ----
def make_server(host="127.0.0.1", port=8765, root=None, token: str | None = "auto", allowed_hosts=()):
    """bound server (not yet serving) and the URL to open; port 0 picks a free port"""
    token = secrets.token_urlsafe(18) if token == "auto" else token
    app = App(root, token, allowed_hosts)
    handler = type("BoundHandler", (Handler,), {"app": app})
    last = None
    for p in ([port] if port == 0 else range(port, port + 20)):
        try:
            httpd = ThreadingHTTPServer((host, p), handler)
            break
        except OSError as e:
            last = e
    else:
        raise last
    httpd.daemon_threads = True
    httpd.app = app
    shown = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    url = f"http://{shown}:{httpd.server_address[1]}/" + (f"?token={token}" if token else "")
    return httpd, url


def serve(host="127.0.0.1", port=8765, root=None, token="auto", open_browser=True, allowed_hosts=()):
    httpd, url = make_server(host, port, root, token, allowed_hosts)
    print(f"[peak3d] GUI running; open this link in your browser:\n\n    {url}\n", flush=True)
    if host in ("0.0.0.0", "::"):
        print("[peak3d] listening on all interfaces; the token in the link keeps others out", flush=True)
    if httpd.app.root:
        print(f"[peak3d] folders limited to {httpd.app.root}", flush=True)
    print("[peak3d] press Ctrl+C to stop", flush=True)
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for r in httpd.app.runs.values():
            r.cancel()
        httpd.server_close()
