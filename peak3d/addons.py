"""Optional feature finders the app installs on request, each in a private Python of its own (MassCube).

MassCube is not bundled: its licence (CC BY-NC 4.0) allows non-commercial use only, and it pins numpy below 2. When
the user asks for it (and accepts that licence), install() downloads a standalone CPython build
(python-build-standalone, pinned by release and checked against its SHA-256), pip-installs the pinned MassCube
version into it, applies our one-line fix for empty MS1 scans, checks that it imports, and only then moves the
folder into place: a failed or interrupted install leaves nothing behind. external.py runs the tool with that
Python, in its own process, exactly as with PEAK3D_MASSCUBE_PYTHON.

The add-ons live in <user data>/peak3d/addons/<name>/: %LOCALAPPDATA% on Windows, ~/Library/Application Support on
macOS, $XDG_DATA_HOME or ~/.local/share on Linux; PEAK3D_ADDONS overrides."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

PBS = "https://github.com/astral-sh/python-build-standalone/releases/download/20261013/"
PYTHONS = {     # cpython 3.11.17, release 20261013: archive and SHA-256 (from the release's SHA256SUMS)
    "aarch64-apple-darwin": ("cpython-3.11.17+20261013-aarch64-apple-darwin-install_only.tar.gz",
                             "d81d757a9785850393cd695a9a686c765a6370fbdc83738589ef2ecd3e5c44b3"),
    "x86_64-apple-darwin": ("cpython-3.11.17+20261013-x86_64-apple-darwin-install_only.tar.gz",
                            "3acde87324867dab4e63329be3010130c75bfc4a3709db00866862d750a3bba3"),
    "x86_64-pc-windows-msvc": ("cpython-3.11.17+20261013-x86_64-pc-windows-msvc-install_only.tar.gz",
                               "7616ab1f74aab975bd91eb0a657a74a092a63a46cac38c5f3f9415c79fd781f5"),
    "x86_64-unknown-linux-gnu": ("cpython-3.11.17+20261013-x86_64-unknown-linux-gnu-install_only.tar.gz",
                                 "846bf682e2b67393f2affa5e5c0994e446abc3a7cc5bb699890c36346823644c"),
}
# MassCube's isotope search crashes on an MS1 scan left empty by its intensity cut; no isotopes is the right answer
_EMPTY_SCAN = ("    mzs = signals[:, 0]\n    intens = signals[:, 1]\n",
               "    # added by peak3d: the nearest MS1 scan can be empty after the intensity cut; report no isotopes\n"
               "    if signals is None or len(signals) == 0:\n"
               "        return np.empty((0, 2), dtype=float)\n\n"
               "    mzs = signals[:, 0]\n    intens = signals[:, 1]\n")
ADDONS = {
    "masscube": dict(label="MassCube", version="1.2.22", packages=["masscube==1.2.22"], module="masscube",
                     # the last numba / llvmlite with Intel-Mac wheels (newer releases build for Apple silicon only)
                     platform_packages={"x86_64-apple-darwin": ["numba==0.62.1", "llvmlite==0.45.1"]},
                     dist="masscube", licence="CC BY-NC 4.0 (non-commercial use only)",
                     licence_url="https://creativecommons.org/licenses/by-nc/4.0/",
                     homepage="https://github.com/huaxuyu/masscube", size_mb=950,
                     patches=[("masscube/feature_grouping.py", *_EMPTY_SCAN)]),
}


def home() -> Path:
    if os.environ.get("PEAK3D_ADDONS"):
        return Path(os.environ["PEAK3D_ADDONS"])
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "peak3d" / "addons"


def target() -> str | None:
    """this computer's python-build-standalone target, None when there is no build for it"""
    m = platform.machine().lower()
    if sys.platform == "darwin":
        return "aarch64-apple-darwin" if m in ("arm64", "aarch64") else "x86_64-apple-darwin"
    if os.name == "nt":
        return "x86_64-pc-windows-msvc" if m in ("amd64", "x86_64") else None
    if sys.platform.startswith("linux"):
        return "x86_64-unknown-linux-gnu" if m in ("x86_64", "amd64") else None
    return None


def _python(root: Path) -> Path:
    return root / "python" / ("python.exe" if os.name == "nt" else "bin/python3")


def python_of(name: str) -> str | None:
    """the add-on's Python when it is installed (its marker written last), else None"""
    root = home() / name
    py = _python(root)
    return str(py) if (root / "installed.json").exists() and py.exists() else None


def status(name: str) -> dict:
    a = ADDONS[name]
    root = home() / name
    info = {}
    if (root / "installed.json").exists():
        try:
            info = json.loads((root / "installed.json").read_text())
        except ValueError:
            info = {}
    return dict(name=name, label=a["label"], version=a["version"], licence=a["licence"], licence_url=a["licence_url"],
                homepage=a["homepage"], size_mb=a["size_mb"], location=str(root), supported=target() is not None,
                installed=python_of(name) is not None, installed_on=info.get("installed"))


def _download(url: str, dest: Path, sha256: str, log) -> None:
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        got, step = 0, 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            h.update(chunk)
            got += len(chunk)
            if total and got * 10 // total > step:
                step = got * 10 // total
                log(f"  {got / 1e6:.0f} of {total / 1e6:.0f} MB")
    if h.hexdigest() != sha256:
        raise RuntimeError(f"the downloaded Python does not match its published checksum ({h.hexdigest()}); not installed")


def _extract(archive: Path, dest: Path) -> None:
    with tarfile.open(archive) as t:
        if hasattr(tarfile, "data_filter"):
            t.extractall(dest, filter="data")          # no links or paths outside dest
        else:
            for m in t.getmembers():
                p = (dest / m.name).resolve()
                if not str(p).startswith(str(dest.resolve())):
                    raise RuntimeError(f"unsafe path in the archive: {m.name}")
            t.extractall(dest)


def _site_packages(py: Path) -> Path:
    out = subprocess.run([str(py), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                         capture_output=True, text=True, check=True)
    return Path(out.stdout.strip())


def _run(cmd, log) -> None:
    p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=dict(os.environ, PYTHONNOUSERSITE="1", PIP_DISABLE_PIP_VERSION_CHECK="1"))
    for line in p.stdout:
        line = line.rstrip()
        if line:
            log("  " + line)
    if p.wait() != 0:
        raise RuntimeError(f"{Path(str(cmd[0])).name} {' '.join(str(c) for c in cmd[1:4])} … failed (exit {p.returncode})")


def install(name: str, log=print, url_base: str = PBS) -> dict:
    """download and set up the add-on (see the module docstring); returns its status. Raises RuntimeError on any
    failure, after removing what it had written"""
    a = ADDONS[name]
    tgt = target()
    if tgt is None:
        raise RuntimeError(f"{a['label']} cannot be installed here: no Python build for {sys.platform} "
                           f"{platform.machine()}")
    archive_name, sha = PYTHONS[tgt]
    root = home() / name
    work = home() / f".{name}.partial"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    tic = time.time()
    try:
        log(f"Downloading Python 3.11 for {tgt} …")
        with tempfile.TemporaryDirectory(dir=home()) as tmp:
            arch = Path(tmp) / archive_name
            _download(url_base + archive_name.replace("+", "%2B"), arch, sha, log)
            log("Checksum verified; unpacking …")
            _extract(arch, work)
        py = _python(work)
        if not py.exists():
            raise RuntimeError(f"the Python archive has no {py.relative_to(work)}")
        packages = [*a["packages"], *a.get("platform_packages", {}).get(tgt, [])]
        log(f"Installing {', '.join(packages)} and the rest of its dependencies (this takes a few minutes) …")
        _run([py, "-m", "pip", "install", "--no-input", "--no-warn-script-location", *packages], log)
        site = _site_packages(py)
        for rel, old, new in a["patches"]:
            f = site / rel
            text = f.read_text(encoding="utf-8")
            if new not in text:
                if text.count(old) != 1:
                    raise RuntimeError(f"cannot apply the empty-scan fix to {rel}: unexpected file contents")
                f.write_text(text.replace(old, new), encoding="utf-8")
                log(f"Applied the empty-scan fix to {rel}")
        out = subprocess.run([str(py), "-c", f"import {a['module']}, importlib.metadata as m; print(m.version({a['dist']!r}))"],
                             capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"{a['label']} was installed but does not import: {out.stderr.strip()[-300:]}")
        (work / "installed.json").write_text(json.dumps(dict(
            name=name, version=out.stdout.strip(), packages=packages, python=archive_name, python_sha256=sha,
            licence=a["licence"], installed=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))
        shutil.rmtree(root, ignore_errors=True)
        work.replace(root)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    log(f"{a['label']} {a['version']} installed in {time.time() - tic:.0f} s ({root})")
    return status(name)


def remove(name: str) -> dict:
    shutil.rmtree(home() / name, ignore_errors=True)
    return status(name)
