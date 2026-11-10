"""Thermo .raw input: every .raw file is converted once to a centroided, indexed mzML by ThermoRawFileParser, and
everything after that (the picker, asari, MassCube, MS2, the chromatograms) reads the mzML as for any other run.

Where the converted files go: ``mzml_converted/`` next to the .raw files, so that later runs on the same folder (the
app starts each run in a new output folder) and other tools find them; ``<output>/mzml_converted/`` when that folder
is not writable; ``--mzml-dir`` overrides both. A conversion is reused while its .raw file keeps its size and
modification time (recorded in ``.sources/<stem>.json`` beside it); a file there that this module did not write is
never overwritten.

The converter: PEAK3D_TRFP (a command, e.g. an Apptainer exec of a ThermoRawFileParser image), else the app's add-on
(addons.py: ``peak3d addon install thermo``), else ThermoRawFileParser on the PATH. It runs with Thermo's peak picking
(centroids, as msconvert's "peakPicking vendor"), every MS level, zlib-compressed arrays."""
from __future__ import annotations

import json
import os
import shlex
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MZML, RAW = ".mzml", ".raw"
CONVERTED = "mzml_converted"
SOURCES = ".sources"
INSTALL_HINT = ("install the Thermo .raw reader (in the app: the Data card; on the command line: "
                "`peak3d addon install thermo --accept-licence`), or set PEAK3D_TRFP to a ThermoRawFileParser command")


def _log(msg):
    print(f"[peak3d] {msg}", flush=True)


# ------------------------------------------------------------------ inputs ----
def input_files(folder) -> list[Path]:
    """a run's inputs in a folder, sorted: every .mzML file and every Thermo .raw file (a file; Waters .raw is a
    folder) without an mzML file of the same name next to it"""
    try:
        entries = [p for p in Path(folder).iterdir() if p.suffix.lower() in (MZML, RAW) and p.is_file()]
    except OSError:
        return []
    have = {p.stem for p in entries if p.suffix.lower() == MZML}
    return sorted(p for p in entries if p.suffix.lower() == MZML or p.stem not in have)


def is_raw(path) -> bool:
    return Path(path).suffix.lower() == RAW


def no_inputs_message(folder) -> str:
    """why a folder holds nothing to run, with a hint for the vendor formats peak3d cannot read itself"""
    folder = Path(folder)
    try:
        names = [(p.name, p.is_dir()) for p in folder.iterdir()]
    except OSError:
        names = []
    hint = ""
    if any(d and n.lower().endswith(".raw") for n, d in names):
        hint = " (Waters .raw folders cannot be read here: convert them to mzML with ProteoWizard msconvert)"
    elif any(n.lower().endswith((".d", ".wiff", ".wiff2")) for n, _ in names):
        hint = " (Agilent .d and SCIEX .wiff data cannot be read here: convert them to mzML with ProteoWizard msconvert)"
    return f"no .mzML or Thermo .raw files in {folder}{hint}"


# ------------------------------------------------------------------ converter ----
def converter() -> dict | None:
    """the ThermoRawFileParser to use: {"cmd": [...], "source": "PEAK3D_TRFP" | "add-on" | "PATH"}, or None"""
    env = os.environ.get("PEAK3D_TRFP", "").strip()
    if env:
        if Path(env).is_file():
            cmd = [env]
        else:
            cmd = shlex.split(env, posix=os.name != "nt")
            if os.name == "nt":
                cmd = [c[1:-1] if len(c) > 1 and c[0] == c[-1] == '"' else c for c in cmd]
        return dict(cmd=cmd, source="PEAK3D_TRFP")
    from .addons import program_of
    exe = program_of("thermo")
    if exe:
        return dict(cmd=[exe], source="add-on")
    for name in ("ThermoRawFileParser", "thermorawfileparser"):
        exe = shutil.which(name)
        if exe:
            return dict(cmd=[exe], source="PATH")
    return None


def _run_converter(cmd, args):
    from .addons import run_program
    return run_program("thermo", cmd[0], *cmd[1:], *args)


# ------------------------------------------------------------------ conversion ----
def _signature(raw: Path) -> dict:
    st = raw.stat()
    return dict(raw=str(raw.resolve()), size=st.st_size, mtime_ns=st.st_mtime_ns)


def _record(out_dir: Path, stem: str) -> Path:
    return out_dir / SOURCES / f"{stem}.json"


def reusable(raw: Path, out_dir: Path) -> Path | None:
    """the earlier conversion of this .raw file in out_dir, when the .raw file has not changed since"""
    dst = out_dir / f"{raw.stem}.mzML"
    try:
        rec = json.loads(_record(out_dir, raw.stem).read_text())
    except (OSError, ValueError):
        return None
    sig = _signature(raw)
    if dst.is_file() and all(rec.get(k) == sig[k] for k in ("size", "mtime_ns")) and \
            Path(rec.get("raw", "")).name == raw.name:
        return dst
    return None


def _writable(folder: Path) -> bool:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder, prefix=".peak3d-write-test-"):
            pass
        return True
    except OSError:
        return False


def target_dir(raw: Path, output, mzml_dir=None) -> Path:
    """where this .raw file's mzML goes (see the module docstring)"""
    if mzml_dir is not None:
        return Path(mzml_dir)
    beside = raw.parent / CONVERTED
    return beside if _writable(beside) else Path(output) / CONVERTED


def pending(files, output, mzml_dir=None) -> list[Path]:
    """the .raw files among these that a run into output would have to convert (no reusable conversion)"""
    out = []
    for f in map(Path, files):
        if not is_raw(f):
            continue
        places = [Path(mzml_dir)] if mzml_dir is not None else [f.parent / CONVERTED, Path(output) / CONVERTED]
        if all(reusable(f, d) is None for d in places):
            out.append(f)
    return out


def convert_one(raw: Path, out_dir: Path, cmd) -> Path:
    """convert one .raw file into out_dir/<stem>.mzML (written to a temporary folder, then moved in place) and record
    its source; raises RuntimeError with the converter's last lines when it fails"""
    raw = Path(raw)
    dst = out_dir / f"{raw.stem}.mzML"
    if dst.exists() and not _record(out_dir, raw.stem).exists():
        raise RuntimeError(f"{dst} exists and was not converted by peak3d from {raw.name}; move it away or choose "
                           f"another folder with --mzml-dir")
    (out_dir / SOURCES).mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=out_dir, prefix=f".{raw.stem}.converting-"))
    try:
        # -option=value: the form ThermoRawFileParser documents for every option. -f=2 indexed mzML, -l=3 warnings
        # and errors only; Thermo's peak picking (centroids) is on by default for every MS level
        r = _run_converter(cmd, [f"-i={raw}", f"-o={work}", "-f=2", "-l=3"])
        made = sorted(work.glob("*.mzML"))
        if r.returncode != 0 or len(made) != 1:
            tail = "\n".join(ln for ln in r.stdout.splitlines()[-8:] if ln.strip())
            raise RuntimeError(f"ThermoRawFileParser could not convert {raw.name} (exit {r.returncode})"
                               + (f":\n{tail}" if tail else ""))
        _record(out_dir, raw.stem).unlink(missing_ok=True)          # no record while the file is being replaced
        os.replace(made[0], dst)
        warnings = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        rec = dict(_signature(raw), converter=" ".join(map(str, cmd)), converted=time.strftime("%Y-%m-%d %H:%M:%S"),
                   messages=warnings[-20:])
        _record(out_dir, raw.stem).write_text(json.dumps(rec, indent=1))
        return dst
    finally:
        shutil.rmtree(work, ignore_errors=True)


def prepare(files, output, cores: int = 1, mzml_dir=None, log=_log) -> tuple[list[Path], dict]:
    """the run's files with every Thermo .raw file replaced by its mzML (same order), converting the ones not
    converted yet, several at a time; and {stem: .raw path} of the converted ones. Raises RuntimeError when .raw files
    need converting and no converter is found, or a conversion fails"""
    files = [Path(f) for f in files]
    raws = [f for f in files if is_raw(f)]
    if not raws:
        return files, {}
    stems = [f.stem for f in files]
    if len(set(stems)) != len(stems):
        raise RuntimeError("input file stems must be unique")
    by_folder = {}
    for f in raws:                               # one write test per folder
        if f.parent not in by_folder:
            by_folder[f.parent] = target_dir(f, output, mzml_dir)
    dirs = {f: by_folder[f.parent] for f in raws}
    done = {f: reusable(f, dirs[f]) for f in raws}
    todo = [f for f in raws if done[f] is None]
    if len(todo) < len(raws):
        log(f"reusing {len(raws) - len(todo)} mzML file(s) converted earlier from .raw files "
            f"({', '.join(sorted({str(dirs[f]) for f in raws if done[f] is not None}))})")
    if todo:
        conv = converter()
        if conv is None:
            raise RuntimeError(f"{len(todo)} Thermo .raw file(s) need converting to mzML and ThermoRawFileParser is "
                               f"not available: {INSTALL_HINT}")
        n = max(1, min(int(cores or 1), len(todo)))
        where = sorted({str(dirs[f]) for f in todo})
        log(f"converting {len(todo)} Thermo .raw file(s) to mzML with ThermoRawFileParser ({conv['source']}), "
            f"{n} at a time, into {', '.join(where)}")
        tic = time.perf_counter()
        count, lock = [0], threading.Lock()

        def one(f):
            t = time.perf_counter()
            out = convert_one(f, dirs[f], conv["cmd"])
            with lock:
                count[0] += 1
                log(f"converted {f.name} ({time.perf_counter() - t:.1f} s, {count[0]}/{len(todo)})")
            return out

        with ThreadPoolExecutor(n) as pool:
            futures = {f: pool.submit(one, f) for f in todo}
            errors = []
            for f, fut in futures.items():
                try:
                    done[f] = fut.result()
                except (RuntimeError, OSError) as e:      # every file is tried; all failures are reported
                    errors.append(str(e) if isinstance(e, RuntimeError) else f"{f.name}: {e}")
        if errors:
            raise RuntimeError("\n".join(errors))
        log(f"raw files converted: {len(todo)} in {time.perf_counter() - tic:.1f} s")
    return [done.get(f, f) for f in files], {f.stem: str(f.resolve()) for f in raws}
