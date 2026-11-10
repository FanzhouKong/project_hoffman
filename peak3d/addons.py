"""Optional tools the app installs on request: MassCube in a private Python of its own, and ThermoRawFileParser (the
Thermo .raw reader) as a self-contained program.

MassCube is not bundled: its licence (CC BY-NC 4.0) allows non-commercial use only, and it pins numpy below 2. When
the user asks for it (and accepts that licence), install() downloads a standalone CPython build
(python-build-standalone, pinned by release and checked against its SHA-256), pip-installs the pinned MassCube
version into it, applies our one-line fix for empty MS1 scans, checks that it imports, and only then writes the
marker file that makes it count as installed; a failed install removes the folder. external.py runs the tool with that
Python, in its own process, exactly as with PEAK3D_MASSCUBE_PYTHON.

ThermoRawFileParser is not bundled either: it reads .raw files with Thermo's RawFileReader libraries, whose licence
binds anyone who redistributes them (indemnity, an About-box notice, a no-redistribution agreement with every user).
The user's own computer downloads it instead, from the CompOmics release, when the user asks and accepts that
licence: the self-contained build for the system (.NET runtime inside, nothing installed system-wide), pinned by
release and checked against its SHA-256, unpacked, and run once (--version) before the marker is written.
rawfiles.py converts .raw files to mzML with it.

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
import zipfile
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
TRFP = "https://github.com/CompOmics/ThermoRawFileParser/releases/download/v.2.0.0-dev/"
TRFP_BUILDS = {  # ThermoRawFileParser 2.0.0-dev, self-contained builds: archive and SHA-256 (the release's asset digests)
    "aarch64-apple-darwin": ("ThermoRawFileParser-v.2.0.0-dev-osx-arm64.zip",
                             "6167aca5c4d44bbf35f986192bc249aba9760e138d13bba27791c2da591ac087"),
    "x86_64-apple-darwin": ("ThermoRawFileParser-v.2.0.0-dev-osx.zip",
                            "98bc806d5f45f32684d77d379c720bfd25aafe04cebcd4dc07f185fbc1942c59"),
    "x86_64-pc-windows-msvc": ("ThermoRawFileParser-v.2.0.0-dev-win.zip",
                               "c5629c42c55ff7fbfa1d0ed1ba71fbb30681ab3eb4acd0ed80042c016c6d3602"),
    "x86_64-unknown-linux-gnu": ("ThermoRawFileParser-v.2.0.0-dev-linux.zip",
                                 "19566762ce6759a93cee9aa4cef50de1e9ae2ad9078bd95826e8e733d4bb0d52"),
}
ADDONS = {
    "masscube": dict(kind="python", label="MassCube", version="1.2.22", packages=["masscube==1.2.22"], module="masscube",
                     # the last numba / llvmlite with Intel-Mac wheels (newer releases build for Apple silicon only)
                     platform_packages={"x86_64-apple-darwin": ["numba==0.62.1", "llvmlite==0.45.1"]},
                     dist="masscube", licence="CC BY-NC 4.0 (non-commercial use only)",
                     licence_url="https://creativecommons.org/licenses/by-nc/4.0/",
                     homepage="https://github.com/huaxuyu/masscube", size_mb=950,
                     patches=[("masscube/feature_grouping.py", *_EMPTY_SCAN)]),
    "thermo": dict(kind="program", label="Thermo .raw reader", tool="ThermoRawFileParser", version="2.0.0-dev",
                   builds=TRFP_BUILDS, url_base=TRFP, executable="ThermoRawFileParser",
                   # no ICU library needed (often missing on Linux): invariant globalization, but creating named
                   # cultures stays allowed, as RawFileReader makes en-US itself (invariant alone fails on every scan).
                   # LI2018 SA1 on hive: the same mzML with and without these (ICU present), 2026-11-09
                   env={"DOTNET_SYSTEM_GLOBALIZATION_INVARIANT": "1",
                        "DOTNET_SYSTEM_GLOBALIZATION_PREDEFINED_CULTURES_ONLY": "0"},
                   licence="Thermo RawFileReader licence (free to use, no redistribution); ThermoRawFileParser: Apache 2.0",
                   licence_url="https://github.com/compomics/ThermoRawFileParser/blob/master/THERMO_LICENSE",
                   notice="RawFileReader reading tool. Copyright \u00a9 2016 by Thermo Fisher Scientific, Inc. All rights reserved.",
                   homepage="https://github.com/compomics/ThermoRawFileParser", size_mb=110),
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


def _marker(name: str) -> dict:
    try:
        return json.loads((home() / name / "installed.json").read_text())
    except (OSError, ValueError):
        return {}


def program_of(name: str) -> str | None:
    """a program add-on's executable when it is installed (its marker written last), else None"""
    rel = _marker(name).get("executable")
    exe = home() / name / rel if rel else None
    return str(exe) if exe is not None and exe.is_file() else None


def _supported(a: dict) -> bool:
    tgt = target()
    return tgt is not None and (a.get("kind") != "program" or tgt in a["builds"])


def status(name: str) -> dict:
    a = ADDONS[name]
    root = home() / name
    info = _marker(name)
    installed = (program_of(name) if a.get("kind") == "program" else python_of(name)) is not None
    return dict(name=name, label=a["label"], version=a["version"], licence=a["licence"], licence_url=a["licence_url"],
                homepage=a["homepage"], size_mb=a["size_mb"], location=str(root), supported=_supported(a),
                installed=installed, installed_on=info.get("installed"), kind=a.get("kind", "python"),
                tool=a.get("tool", a["label"]), notice=a.get("notice"),
                installed_version=info.get("version") if installed else None)


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
        raise RuntimeError(f"the download does not match its published checksum ({h.hexdigest()}); not installed")


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
                         capture_output=True, encoding="utf-8", errors="replace", check=True)
    return Path(out.stdout.strip())


def _run(cmd, log) -> None:
    p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         encoding="utf-8", errors="replace",          # not the console code page (Windows)
                         env=dict(os.environ, PYTHONNOUSERSITE="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
                                  PYTHONIOENCODING="utf-8", PYTHONUTF8="1"))
    for line in p.stdout:
        line = line.rstrip()
        if line:
            log("  " + line)
    if p.wait() != 0:
        raise RuntimeError(f"{Path(str(cmd[0])).name} {' '.join(str(c) for c in cmd[1:4])} … failed (exit {p.returncode})")


def _extract_zip(archive: Path, dest: Path) -> None:
    """unpack, refusing paths outside dest; POSIX permission bits as stored (zipfile drops them: executables)"""
    dest.mkdir(parents=True, exist_ok=True)
    top = dest.resolve()
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        for m in members:
            p = (dest / m.filename).resolve()
            if p != top and top not in p.parents:
                raise RuntimeError(f"unsafe path in the archive: {m.filename}")
        z.extractall(dest)
        if os.name != "nt":
            for m in members:
                mode = (m.external_attr >> 16) & 0o777
                if mode and not m.is_dir():
                    os.chmod(dest / m.filename, mode | 0o400)


def _find_program(folder: Path, name: str) -> Path | None:
    """the program in the unpacked archive: at its top or one folder down"""
    exe = name + (".exe" if os.name == "nt" else "")
    for p in (folder / exe, *sorted(folder.glob(f"*/{exe}"))):
        if p.is_file():
            return p
    return None


def program_env(name: str) -> dict:
    """the environment a program add-on runs in: its own variables, and none of the frozen app's library path (the
    Linux app points LD_LIBRARY_PATH at its own libraries; the program must load the system's)"""
    env = dict(os.environ, **ADDONS[name].get("env", {}))
    if getattr(sys, "frozen", False):
        for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
            orig = env.pop(var + "_ORIG", None)
            if orig is not None:
                env[var] = orig
            else:
                env.pop(var, None)
    return env


def run_program(name: str, exe, *args, timeout: float | None = None) -> subprocess.CompletedProcess:
    """run a program add-on (or the same tool found elsewhere) with its environment, output captured as text"""
    env = program_env(name)
    kw = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}
    return subprocess.run([str(exe), *map(str, args)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          encoding="utf-8", errors="replace", env=env, timeout=timeout, **kw)


def _sign_for_mac(files, log) -> None:
    """ad-hoc sign (codesign --sign -, no identity needed): Apple silicon runs no unsigned code, and the macOS builds
    of ThermoRawFileParser 2.0.0-dev ship their launcher unsigned (their libraries are signed)"""
    try:
        r = subprocess.run(["codesign", "--force", "--sign", "-", *map(str, files)], capture_output=True,
                           encoding="utf-8", errors="replace")
    except OSError as e:
        log(f"codesign is not available ({e}); the program may not start")
        return
    if r.returncode != 0:
        log(f"codesign failed (exit {r.returncode}): {(r.stderr or r.stdout).strip()[-300:]}")


def _program_version(name: str, exe: Path, log) -> str:
    """run the program once (--version). On macOS its launcher is ad-hoc signed first; if it still does not start,
    every library beside it is signed too and it is tried again"""
    if sys.platform == "darwin":
        log("Signing the program for this Mac (ad hoc) …")
        _sign_for_mac([exe], log)
    r = run_program(name, exe, "--version", timeout=120)
    if r.returncode != 0 and sys.platform == "darwin":
        log("The program did not start; signing its libraries too and trying again …")
        _sign_for_mac([exe, *sorted(exe.parent.glob("*.dylib"))], log)
        r = run_program(name, exe, "--version", timeout=120)
    if r.returncode != 0:
        tail = " / ".join(ln.strip() for ln in r.stdout.splitlines()[-4:] if ln.strip())
        raise RuntimeError(f"{exe.name} was unpacked but does not run (exit {r.returncode}): {tail or 'no output'}")
    lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    return (lines[-1] if lines else "unknown")[:80]


def _install_program(name: str, a: dict, log, url_base: str | None) -> dict:
    tgt = target()
    if not _supported(a):
        raise RuntimeError(f"{a['label']} cannot be installed here: no {a['tool']} build for {sys.platform} "
                           f"{platform.machine()}")
    archive_name, sha = a["builds"][tgt]
    root = home() / name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    tic = time.time()
    try:
        log(f"Downloading {a['tool']} {a['version']} for {tgt} …")
        with tempfile.TemporaryDirectory(dir=home()) as tmp:
            arch = Path(tmp) / archive_name
            _download((url_base or a["url_base"]) + archive_name, arch, sha, log)
            log("Checksum verified; unpacking …")
            _extract_zip(arch, root / "program")
        exe = _find_program(root / "program", a["executable"])
        if exe is None:
            raise RuntimeError(f"the archive has no {a['executable']} program")
        if os.name != "nt":
            os.chmod(exe, exe.stat().st_mode | 0o755)
        version = _program_version(name, exe, log)
        (root / "installed.json").write_text(json.dumps(dict(
            name=name, tool=a["tool"], version=version, archive=archive_name, sha256=sha,
            executable=exe.relative_to(root).as_posix(), licence=a["licence"],
            installed=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    log(f"{a['tool']} {version} installed in {time.time() - tic:.0f} s ({root})")
    return status(name)


def install(name: str, log=print, url_base: str | None = None) -> dict:
    """download and set up the add-on (see the module docstring); returns its status. Raises RuntimeError on any
    failure, after removing what it had written"""
    a = ADDONS[name]
    if a.get("kind") == "program":
        return _install_program(name, a, log, url_base)
    url_base = url_base or PBS
    tgt = target()
    if tgt is None:
        raise RuntimeError(f"{a['label']} cannot be installed here: no Python build for {sys.platform} "
                           f"{platform.machine()}")
    archive_name, sha = PYTHONS[tgt]
    root = home() / name
    # in place, with the marker written last (python_of() needs it): no folder rename at the end, which Windows
    # refuses while its virus scanner still holds the files just written
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    work = root
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
                             capture_output=True, encoding="utf-8", errors="replace")
        if out.returncode != 0:
            raise RuntimeError(f"{a['label']} was installed but does not import: {out.stderr.strip()[-300:]}")
        (work / "installed.json").write_text(json.dumps(dict(
            name=name, version=out.stdout.strip(), packages=packages, python=archive_name, python_sha256=sha,
            licence=a["licence"], installed=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    log(f"{a['label']} {a['version']} installed in {time.time() - tic:.0f} s ({root})")
    return status(name)


def remove(name: str) -> dict:
    shutil.rmtree(home() / name, ignore_errors=True)
    return status(name)
