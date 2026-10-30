"""Other feature-detection kernels (asari, MassCube) behind the same output layout as ``peak3d process``.

The tool runs with its own defaults in a subprocess of the Python that has it installed: PEAK3D_ASARI_PYTHON /
PEAK3D_MASSCUBE_PYTHON when set, else the app's own install of it (MassCube, addons.py), else this program when it
carries the package (asari is bundled with the app; it then runs as `peak3d _asari ...`, a hidden command). Its aligned table is converted to
aligned_feature_table.tsv (group_id, mz, rt, rt_min, rt_max in minutes, counts, then one column per file), and every
file is read once more by peak3d (MS1 point clouds for the chromatograms, MS2 spectra in the same pass), so MS2
mapping, the identity search and the report work exactly as after a peak3d run. Nothing of the tool's own
alignment, gap filling or annotation is re-done or second-guessed."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__

KERNELS = {"asari": dict(label="asari", package="asari", dist="asari-metabolomics", env="PEAK3D_ASARI_PYTHON",
                         quant="area", install="pip install asari-metabolomics"),
           "masscube": dict(label="MassCube", package="masscube", dist="masscube", env="PEAK3D_MASSCUBE_PYTHON",
                            quant="height", install="pip install masscube==1.2.22",
                            addon="peak3d addon install masscube --accept-licence (or the MassCube tab in the app)")}
SELF = "self"           # kernel_status()[...]["python"] when this program runs the tool itself (asari, bundled)
LAYOUT = ["group_id", "mz", "rt", "rt_min", "rt_max", "n_detected", "n_confirmed", "n_filled", "mz_ppm_spread",
          "rt_sd", "iso_offset", "charge"]
_PROBE = ("import importlib.util as u, importlib.metadata as m, sys\n"
          "if u.find_spec({pkg!r}) is None: sys.exit(3)\n"
          "try:\n    print(m.version({dist!r}))\nexcept Exception:\n    print('unknown')\n")
# MassCube sizes its worker pool from cpu_count(): point it at the cores given (as scripts/run_masscube.py does)
_MASSCUBE = ("import multiprocessing, os, sys\n"
             "n = int(sys.argv[2])\n"
             "multiprocessing.cpu_count = lambda: n\n"
             "os.cpu_count = lambda: n\n"
             "from masscube.workflows import untargeted_metabolomics_workflow\n"
             "untargeted_metabolomics_workflow(path=sys.argv[1])\n")


# ------------------------------------------------------------------ availability ----
def _probe(name: str) -> dict:
    from .addons import ADDONS, python_of
    k = KERNELS[name]
    py = os.environ.get(k["env"]) or (python_of(name) if name in ADDONS else None)
    if py:
        try:
            r = subprocess.run([py, "-c", _PROBE.format(pkg=k["package"], dist=k["dist"])], capture_output=True,
                               text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as e:
            return dict(available=False, version=None, python=py, note=f"{k['env']}={py} does not run: {e}")
        if r.returncode == 0:
            return dict(available=True, version=(r.stdout.strip().splitlines() or ["unknown"])[-1], python=py, note="")
        return dict(available=False, version=None, python=py,
                    note=f"{py} cannot import {k['package']} (install it there: {k['install']})")
    if importlib.util.find_spec(k["package"]) is not None:
        try:
            ver = importlib.metadata.version(k["dist"])
        except importlib.metadata.PackageNotFoundError:
            ver = "unknown"
        return dict(available=True, version=ver, python=SELF, note="")
    if getattr(sys, "frozen", False) and name == "asari":
        return dict(available=False, version=None, python=None,
                    note=f"asari is not part of the app on this system; set {k['env']} to a Python that has it")
    how = k.get("addon") or k["install"]
    return dict(available=False, version=None, python=None,
                note=f"{k['label']} is not installed: {how}; or set {k['env']} to a Python that has it")


def refresh() -> None:
    """probe again (after an add-on was installed or removed)"""
    kernel_status.cache_clear()


@lru_cache(maxsize=1)
def kernel_status() -> dict:
    """which kernels this installation can run: {name: {available, version, python, note}} (probed once)"""
    from .estimate import VARIANT
    out = {"peak3d": dict(available=True, version=__version__, variant=VARIANT, python=sys.executable, note="")}
    for name in KERNELS:
        out[name] = _probe(name)
    return out


def detect_polarity(path) -> str:
    """"pos" or "neg" from the first scan that states it ("pos" when none does)"""
    from pyteomics import mzml
    with mzml.read(str(path), use_index=False) as reader:
        for k, s in enumerate(reader):
            if "negative scan" in s:
                return "neg"
            if "positive scan" in s or k > 50:
                return "pos"
    return "pos"


# ------------------------------------------------------------------ conversion ----
def _stem_columns(columns, stems):
    """the tool's column for each file stem (tools add or keep '.mzML'); missing files get None"""
    by = {re.sub(r"\.mzml$", "", str(c), flags=re.I): c for c in columns}
    return {s: by.get(s) for s in stems}


def _layout(group_id, mz, rt, rt_min, rt_max, values: pd.DataFrame, iso=None, charge=None) -> pd.DataFrame:
    n = len(mz)
    V = values.to_numpy(np.float64)
    out = pd.DataFrame({"group_id": np.asarray(group_id, np.int64), "mz": np.asarray(mz, float),
                        "rt": np.asarray(rt, float), "rt_min": np.asarray(rt_min, float),
                        "rt_max": np.asarray(rt_max, float), "n_detected": (V > 0).sum(axis=1).astype(int),
                        "n_confirmed": np.zeros(n, int), "n_filled": np.zeros(n, int),
                        "mz_ppm_spread": np.full(n, np.nan), "rt_sd": np.full(n, np.nan),
                        "iso_offset": np.zeros(n, int) if iso is None else np.asarray(iso, int),
                        "charge": np.zeros(n, int) if charge is None else np.asarray(charge, int)})
    return pd.concat([out, values.reset_index(drop=True)], axis=1)


def _values(t: pd.DataFrame, stems) -> pd.DataFrame:
    cols = _stem_columns(t.columns, stems)
    return pd.DataFrame({s: (pd.to_numeric(t[c], errors="coerce").fillna(0.0).to_numpy(float) if c is not None
                             else np.zeros(len(t))) for s, c in cols.items()})


def _ids(raw, n):
    ids = pd.to_numeric(pd.Series(raw).astype(str).str.replace(r"\D", "", regex=True), errors="coerce")
    return ids.astype(np.int64).to_numpy() if ids.notna().all() and ids.is_unique else np.arange(1, n + 1)


def convert_asari(path, stems) -> pd.DataFrame:
    """asari's full_Feature_table.tsv (RT in seconds, peak areas per file) -> the aligned table layout"""
    t = pd.read_csv(path, sep="\t")
    return _layout(_ids(t["id_number"], len(t)), t["mz"], t["rtime"] / 60.0, t["rtime_left_base"] / 60.0,
                   t["rtime_right_base"] / 60.0, _values(t, stems))


def _shape_bounds(shape, rt):
    """first and last time of MassCube's peak_shape ("rt;int|rt;int|..."), around the apex rt"""
    if isinstance(shape, str) and shape:
        try:
            pts = shape.split("|")
            a, b = float(pts[0].split(";")[0]), float(pts[-1].split(";")[0])
            return min(a, rt), max(b, rt)
        except (ValueError, IndexError):
            pass
    return rt - 0.05, rt + 0.05


def _charge(adduct) -> int:
    m = re.search(r"\](\d*)[+-]$", str(adduct).strip())
    return (int(m.group(1)) if m.group(1) else 1) if m else 0


def convert_masscube(path, stems) -> pd.DataFrame:
    """MassCube's aligned_feature_table.txt (RT in minutes, peak heights per file) -> the aligned table layout"""
    t = pd.read_csv(path, sep="\t", low_memory=False)
    rt = t["RT"].to_numpy(float)
    shape = t["peak_shape"] if "peak_shape" in t else pd.Series([None] * len(t))
    b = np.array([_shape_bounds(s, r) for s, r in zip(shape, rt)], float).reshape(-1, 2)
    if "isotope_state" in t:
        iso = pd.to_numeric(t["isotope_state"].astype(str).str.extract(r"M\+(\d+)")[0], errors="coerce").fillna(0)
    elif "is_isotope" in t:
        iso = t["is_isotope"].astype(str).str.lower().eq("true").astype(int)
    else:
        iso = None
    charge = [_charge(a) for a in t["adduct"]] if "adduct" in t else None
    gid = t["feature_ID"] if "feature_ID" in t else np.arange(1, len(t) + 1)
    return _layout(_ids(gid, len(t)), t["m/z"], rt, b[:, 0], b[:, 1], _values(t, stems), iso, charge)


# ------------------------------------------------------------------ running ----
def _run(cmd, cwd: Path) -> int:
    """the tool's own output goes straight to ours (the GUI shows it in the run log)"""
    sys.stdout.flush()
    # UTF-8 output whatever the console code page: MassCube prints box-drawing characters, which Windows' cp1252
    # cannot encode (the tool then stops at its welcome banner)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return subprocess.run(cmd, cwd=str(cwd), env=env, stdout=sys.stdout.fileno(), stderr=subprocess.STDOUT).returncode


def _self_command() -> list[str]:
    """this program, as a command (the frozen app is its own executable)"""
    return [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, "-m", "peak3d"]


def run_asari_main(argv) -> None:
    """`peak3d _asari ARGS`: asari's own command line, inside this program (the app bundles asari)"""
    import types
    if "asari.__init__" not in sys.modules:
        # asari/main.py imports its version with `from .__init__ import __version__`: plain Python then runs
        # asari/__init__.py a second time as a module of that name, the frozen app's importer makes it a package and
        # the import goes in a circle. A stand-in holding the version answers that import in both
        stub = types.ModuleType("asari.__init__")
        try:
            stub.__version__ = importlib.metadata.version("asari-metabolomics")
        except importlib.metadata.PackageNotFoundError:
            stub.__version__ = "unknown"
        sys.modules["asari.__init__"] = stub
    from asari.main import main
    sys.argv = ["asari", *argv]
    main()


def _run_asari(py, input_dir: Path, output: Path, polarity: str, cores: int) -> Path:
    t0 = time.time()
    launcher = [*_self_command(), "_asari"] if py == SELF else [py, "-m", "asari.main"]
    rc = _run([*launcher, "process", "--mode", polarity, "--input", str(input_dir),
               "--output", str(output / "asari_work"), "--multicores", str(cores)], output)
    if rc != 0:
        sys.exit(f"asari failed (exit {rc}); see the log above")
    found = [p for p in output.glob("asari_work*/export/full_Feature_table.tsv") if p.stat().st_mtime >= t0 - 1]
    if not found:
        sys.exit(f"asari finished but wrote no export/full_Feature_table.tsv under {output}")
    return max(found, key=lambda p: p.stat().st_mtime)


def _run_masscube(py, files, output: Path, cores: int) -> Path:
    work = output / "masscube_work"
    shutil.rmtree(work, ignore_errors=True)          # MassCube reuses what it finds in its project folder
    (work / "data").mkdir(parents=True)
    for f in files:
        dst = work / "data" / Path(f).name
        try:
            dst.symlink_to(Path(f).resolve())
        except OSError:
            shutil.copy2(f, dst)
    if py == SELF:
        py = sys.executable
        if getattr(sys, "frozen", False):
            sys.exit("MassCube is not part of the app: install it from the MassCube tab")
    rc = _run([py, "-c", _MASSCUBE, str(work), str(max(1, cores))], work)
    if rc != 0:
        sys.exit(f"MassCube failed (exit {rc}); see the log above")
    for p in (work / "aligned_feature_table.txt", work / "project_files" / "aligned_feature_table_before_annotation.txt"):
        if p.exists():
            return p
    sys.exit(f"MassCube finished but wrote no aligned_feature_table.txt in {work}")


def _read_one(job):
    """one file's MS1 cloud (-> cache) and MS2 spectra (-> ms2/), for the chromatograms and the MS2 mapping"""
    from .io import load_cloud
    from .ms2 import Collector, from_collected, save
    path, cache_dir, ms2_dir, polarity = job
    stem = Path(path).stem
    tic = time.perf_counter()
    try:
        spectra = Collector()
        cloud = load_cloud(path, polarity, spectra)
    except Exception as e:  # a file the kernel took but peak3d cannot read: no chromatograms for it
        return dict(stem=stem, path=str(path), error=repr(e))
    cloud.save_npz(Path(cache_dir) / f"{stem}.npz")
    if spectra:
        save(from_collected(spectra, cloud.polarity), Path(ms2_dir) / f"{stem}.npz")
    return dict(stem=stem, path=str(path), n_scans=int(cloud.n_scans), n_centroids=int(cloud.n_points),
                n_ms2=len(spectra), parse_s=round(time.perf_counter() - tic, 3))


def finish(files, args, table: pd.DataFrame, kernel: str, version: str, stage: dict, polarity: str | None) -> None:
    """everything after the kernel: the aligned table, peak3d's own read of every file (chromatogram cache, MS2),
    MS2 mapping, identity search, params.json, report.html"""
    from .features import precise_mz
    from .pipeline import _write_report, log, ms2_and_identify
    output = Path(args.output)
    files = [Path(f) for f in files]
    stems = [f.stem for f in files]
    precise_mz(table).to_csv(output / "aligned_feature_table.tsv", sep="\t", index=False, float_format="%.6g")
    cache_dir, ms2_dir, features_dir = output / ".cache", output / "ms2", output / "features"
    for d in (cache_dir, ms2_dir, features_dir):
        d.mkdir(parents=True, exist_ok=True)
    for s in stems:
        (ms2_dir / f"{s}.npz").unlink(missing_ok=True)
    log(f"reading {len(files)} files for chromatograms and MS2")
    tic = time.perf_counter()
    jobs = [(str(f), str(cache_dir), str(ms2_dir), polarity) for f in files]
    n_workers = max(1, min(len(files), int(getattr(args, "cores", 1) or 1)))
    if n_workers == 1:
        rows = [_read_one(j) for j in jobs]
    else:
        with get_context("spawn").Pool(n_workers) as pool:
            rows = pool.map(_read_one, jobs, chunksize=1)
    for r in rows:
        if "error" in r:
            log(f"{r['stem']}: not read ({r['error']}); no chromatograms or MS2 from it")
    stage["read_s"] = round(time.perf_counter() - tic, 3)
    man = pd.DataFrame(rows)
    man["n_features"] = [int((table[s] > 0).sum()) if s in table else 0 for s in man["stem"]]
    man.to_csv(features_dir / "manifest.tsv", sep="\t", index=False)
    ms2_and_identify(output, args, stage)
    k = KERNELS[kernel]
    options = {key: (str(v) if isinstance(v, Path) else v) for key, v in vars(args).items() if key != "cmd"}
    options.update(quant=k["quant"], polarity=polarity)
    p = dict(kernel=kernel, kernel_label=k["label"], kernel_version=version, peak3d_version=__version__,
             command=" ".join(sys.argv), options=options, input_files=[str(f) for f in files], stage_times_s=stage,
             n_files=len(files), alignment=f"by {k['label']}", n_features_total=int(man["n_features"].sum()))
    (output / "params.json").write_text(json.dumps(p, indent=1, default=str))
    _write_report(output, cache_dir, args)
    if not getattr(args, "keep_cache", False):
        shutil.rmtree(cache_dir, ignore_errors=True)


def run_external(files, args) -> None:
    """``peak3d external``: run the kernel on --input, then finish() into --output"""
    from .pipeline import check_library, log
    check_library(args)
    kernel = args.kernel
    st = kernel_status()[kernel]
    k = KERNELS[kernel]
    if not st["available"]:
        sys.exit(f"{k['label']} is not available: {st['note']}")
    files = [Path(f).resolve() for f in files]      # absolute: the tools run in their own working folders
    stems = [f.stem for f in files]
    if len(set(stems)) != len(stems):
        sys.exit("input file stems must be unique")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    polarity = args.polarity or detect_polarity(files[0])
    stage = {}
    log(f"kernel: running {k['label']} {st['version']} on {len(files)} files ({polarity} mode)")
    tic = time.perf_counter()
    if kernel == "asari":
        src = _run_asari(st["python"], files[0].parent, output, polarity, args.cores)
    else:
        src = _run_masscube(st["python"], files, output, args.cores)
    stage["kernel_s"] = round(time.perf_counter() - tic, 3)
    log(f"kernel: {k['label']} finished in {stage['kernel_s']:.1f} s")
    log(f"kernel: converting {k['label']} output ({src.relative_to(output) if output in src.parents else src})")
    table = convert_asari(src, stems) if kernel == "asari" else convert_masscube(src, stems)
    log(f"kernel: {len(table)} features")
    finish(files, args, table, kernel, st["version"], stage, polarity)
