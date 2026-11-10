"""Thermo .raw input (rawfiles.py): which files a folder offers, conversion with a stand-in ThermoRawFileParser,
reuse, the fall-back folder, failures, the CLI and the app around it, and the add-on install (addons.py)."""
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import pytest

from peak3d import addons, external, rawfiles
from peak3d.io import load_cloud
from peak3d.synth import Peak, make_cloud, write_mzml

ROOT = Path(__file__).resolve().parents[2]

# stands in for ThermoRawFileParser: takes its -option=value arguments, "converts" a .raw file holding mzML text by
# copying it, logs each conversion, fails on a file starting with FAIL as the real one does (message, exit 1)
FAKE = r'''
import os, pathlib, shutil, sys
if "--version" in sys.argv:
    print("9.9.9-test")
    sys.exit(0)
opts = dict(a.split("=", 1) for a in sys.argv[1:] if a.startswith("-") and "=" in a)
raw, out = pathlib.Path(opts["-i"]), pathlib.Path(opts["-o"])
assert opts["-f"] == "2" and out.is_dir(), sys.argv
if os.environ.get("FAKE_TRFP_LOG"):
    with open(os.environ["FAKE_TRFP_LOG"], "a") as fh:
        fh.write(raw.name + "\n")
if raw.read_bytes().startswith(b"FAIL"):
    print("2026-11-09 ERROR Cannot read " + raw.name + ": not a RAW file")
    sys.exit(1)
shutil.copyfile(raw, out / (raw.stem + ".mzML"))
'''


def _cloud(k, n=12, n_scans=300):
    rng = np.random.default_rng(7)
    peaks = [Peak(float(m), float(r) + 0.02 * k, float(h), 0.05, iso=(0.15,)) for m, r, h in
             zip(rng.uniform(150, 900, n), rng.uniform(0.3, 1.2, n), 10 ** rng.uniform(4.8, 6.3, n))]
    cloud, _ = make_cloud(peaks, n_scans=n_scans, dt=0.005, thresh=200.0, floor=100.0, n_noise=3000, seed=k,
                          name=f"S{k}")
    return cloud


def _folder(d: Path, raw=("S0", "S1"), mzml=("S2",)) -> Path:
    """.raw files that hold mzML text (the fake converter copies them), plus real mzML files"""
    d.mkdir(parents=True, exist_ok=True)
    for k, stem in enumerate([*raw, *mzml]):
        write_mzml(_cloud(k), d / f"{stem}.mzML")
        if stem in raw:
            (d / f"{stem}.mzML").rename(d / f"{stem}.raw")
    return d


@pytest.fixture
def fake_trfp(tmp_path, monkeypatch):
    script = tmp_path / "fake_trfp.py"
    script.write_text(FAKE)
    log = tmp_path / "trfp_calls.txt"
    monkeypatch.setenv("PEAK3D_TRFP", f"{sys.executable} {script}")
    monkeypatch.setenv("FAKE_TRFP_LOG", str(log))
    calls = lambda: log.read_text().split() if log.exists() else []
    return calls


def _quiet(msg):
    pass


# ------------------------------------------------------------------ inputs ----
def test_input_files_prefer_mzml_and_skip_folders(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    for name in ("a.mzML", "a.raw", "b.RAW", "c.raw", "notes.txt"):
        (d / name).write_text("x")
    (d / "w.raw").mkdir()                              # Waters .raw is a folder
    assert [p.name for p in rawfiles.input_files(d)] == ["a.mzML", "b.RAW", "c.raw"]
    assert rawfiles.input_files(tmp_path / "missing") == []
    only_waters = tmp_path / "waters"
    (only_waters / "run1.raw").mkdir(parents=True)
    assert rawfiles.input_files(only_waters) == [] and "Waters" in rawfiles.no_inputs_message(only_waters)
    agilent = tmp_path / "agilent"
    (agilent / "run1.d").mkdir(parents=True)
    assert "Agilent" in rawfiles.no_inputs_message(agilent)
    assert rawfiles.no_inputs_message(tmp_path / "missing").startswith("no .mzML or Thermo .raw files")


# ------------------------------------------------------------------ conversion ----
def test_convert_reuse_and_reconvert_after_a_change(tmp_path, fake_trfp):
    d = _folder(tmp_path / "data")
    files = rawfiles.input_files(d)
    assert [f.name for f in files] == ["S0.raw", "S1.raw", "S2.mzML"]
    assert [f.name for f in rawfiles.pending(files, tmp_path / "out")] == ["S0.raw", "S1.raw"]
    log = []
    got, raw = rawfiles.prepare(files, tmp_path / "out", cores=2, log=log.append)
    conv = d / rawfiles.CONVERTED
    assert got == [conv / "S0.mzML", conv / "S1.mzML", d / "S2.mzML"]          # same order, mzML kept as it is
    assert raw == {"S0": str((d / "S0.raw").resolve()), "S1": str((d / "S1.raw").resolve())}
    assert sorted(fake_trfp()) == ["S0.raw", "S1.raw"]
    assert any("converting 2 Thermo .raw file(s) to mzML" in m for m in log) and sum(m.startswith("converted ") for m in log) == 2
    assert load_cloud(got[0]).n_scans == 300 and not list(conv.glob(".*converting*"))
    rec = json.loads((conv / ".sources" / "S0.json").read_text())
    assert rec["size"] == (d / "S0.raw").stat().st_size and "fake_trfp.py" in rec["converter"]
    assert rawfiles.pending(files, tmp_path / "elsewhere") == []                # found next to the .raw files

    log.clear()
    assert rawfiles.prepare(files, tmp_path / "out2", log=log.append)[0] == got     # another run: reused
    assert len(fake_trfp()) == 2 and any("reusing 2" in m for m in log)

    st = (d / "S0.raw").stat()
    os.utime(d / "S0.raw", ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))       # the .raw file changed
    assert [f.name for f in rawfiles.pending(files, tmp_path / "out")] == ["S0.raw"]
    rawfiles.prepare(files, tmp_path / "out", log=_quiet)
    assert sorted(fake_trfp()) == ["S0.raw", "S0.raw", "S1.raw"]

    assert rawfiles.prepare([d / "S2.mzML"], tmp_path / "out", log=log.append) == ([d / "S2.mzML"], {})


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="POSIX permissions; root writes anywhere")
def test_read_only_data_folder_converts_into_the_output(tmp_path, fake_trfp):
    d = _folder(tmp_path / "data", raw=("S0",), mzml=())
    d.chmod(0o555)
    try:
        got, _ = rawfiles.prepare(rawfiles.input_files(d), tmp_path / "out", log=_quiet)
        assert got == [tmp_path / "out" / rawfiles.CONVERTED / "S0.mzML"] and got[0].is_file()
        assert rawfiles.pending([d / "S0.raw"], tmp_path / "out") == []
        assert rawfiles.pending([d / "S0.raw"], tmp_path / "other_out") == [d / "S0.raw"]
    finally:
        d.chmod(stat.S_IRWXU)


def test_mzml_dir_and_a_foreign_file_is_never_overwritten(tmp_path, fake_trfp):
    d = _folder(tmp_path / "data", raw=("S0", "S1"), mzml=())
    m = tmp_path / "shared"
    got, _ = rawfiles.prepare(rawfiles.input_files(d), tmp_path / "out", mzml_dir=m, log=_quiet)
    assert got == [m / "S0.mzML", m / "S1.mzML"] and not (d / rawfiles.CONVERTED).exists()
    other = tmp_path / "theirs"
    other.mkdir()
    (other / "S0.mzML").write_text("their own conversion")
    with pytest.raises(RuntimeError, match="not converted by peak3d"):
        rawfiles.prepare([d / "S0.raw"], tmp_path / "out", mzml_dir=other, log=_quiet)
    assert (other / "S0.mzML").read_text() == "their own conversion"


def test_failure_and_missing_converter(tmp_path, fake_trfp, monkeypatch):
    d = _folder(tmp_path / "data", raw=("S0",), mzml=())
    (d / "bad.raw").write_text("FAIL: not a raw file")
    with pytest.raises(RuntimeError) as e:
        rawfiles.prepare(rawfiles.input_files(d), tmp_path / "out", cores=2, log=_quiet)
    assert "could not convert bad.raw (exit 1)" in str(e.value) and "not a RAW file" in str(e.value)
    conv = d / rawfiles.CONVERTED
    assert sorted(p.name for p in conv.iterdir()) == [".sources", "S0.mzML"]      # the good one; no leftovers
    with pytest.raises(RuntimeError, match="stems must be unique"):
        rawfiles.prepare([d / "S0.raw", tmp_path / "S0.raw"], tmp_path / "out", log=_quiet)

    monkeypatch.delenv("PEAK3D_TRFP")
    monkeypatch.setenv("PEAK3D_ADDONS", str(tmp_path / "no_addons"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty_path"))
    assert rawfiles.converter() is None
    with pytest.raises(RuntimeError, match="addon install thermo"):
        rawfiles.prepare([d / "bad.raw"], tmp_path / "out", log=_quiet)
    assert rawfiles.prepare([d / "S0.raw"], tmp_path / "out", log=_quiet)[0] == [conv / "S0.mzML"]   # converted: no converter needed


def test_one_folder_for_asari(tmp_path):
    d = _folder(tmp_path / "data", raw=(), mzml=("S0", "S1"))
    files = rawfiles.input_files(d)
    assert external._one_folder(files, tmp_path / "out") == d.resolve()
    (tmp_path / "out").mkdir()
    conv = tmp_path / "conv"
    conv.mkdir()
    (conv / "S2.mzML").write_text("x")
    (conv / "old.mzML").write_text("x")                 # a converted file that is not part of this run
    folder = external._one_folder([conv / "S2.mzML"], tmp_path / "out")
    assert folder == tmp_path / "out" / "asari_input" and [p.name for p in folder.iterdir()] == ["S2.mzML"]
    folder = external._one_folder([*files, conv / "S2.mzML"], tmp_path / "out")
    assert sorted(p.name for p in folder.iterdir()) == ["S0.mzML", "S1.mzML", "S2.mzML"]


# ------------------------------------------------------------------ command line ----
def _cli(*argv, env=None):
    return subprocess.run([sys.executable, "-m", "peak3d", *map(str, argv)], cwd=ROOT, capture_output=True, text=True,
                          env=env)


def test_cli_process_and_convert_with_raw_files(tmp_path, fake_trfp):
    d = _folder(tmp_path / "data")
    out = tmp_path / "out"
    r = _cli("process", "--input", d, "--output", out, "--cores", "2", "--no-report", env=dict(os.environ))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "converting 2 Thermo .raw file(s)" in r.stdout and "raw files converted:" in r.stdout
    t = pd.read_csv(out / "aligned_feature_table.tsv", sep="\t")
    assert {"S0", "S1", "S2"} <= set(t.columns) and len(t) > 0
    p = json.loads((out / "params.json").read_text())
    conv = d / rawfiles.CONVERTED
    assert p["input_files"] == [str(conv / "S0.mzML"), str(conv / "S1.mzML"), str(d / "S2.mzML")]
    assert set(p["raw_files"]) == {"S0", "S1"}
    r = _cli("convert", d, env=dict(os.environ))                  # converted already: reused, read back
    assert r.returncode == 0, r.stdout + r.stderr
    assert "reusing 2" in r.stdout and r.stdout.count("300 MS1 scans") == 2
    assert len(fake_trfp()) == 2
    env = {k: v for k, v in os.environ.items() if k != "PEAK3D_TRFP"}
    env.update(PEAK3D_ADDONS=str(tmp_path / "no_addons"), PATH=str(tmp_path / "empty_path"))
    (tmp_path / "new").mkdir()
    (tmp_path / "new" / "X.raw").write_text("x")
    r = _cli("process", "--input", tmp_path / "new", "--output", tmp_path / "out3", env=env)
    assert r.returncode != 0 and "addon install thermo" in r.stderr
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "run.raw").mkdir()
    r = _cli("process", "--input", tmp_path / "w", "--output", tmp_path / "out4")
    assert r.returncode != 0 and "Waters" in r.stderr


# ------------------------------------------------------------------ the app ----
def _client(url, token):
    import urllib.error
    import urllib.request
    base = url.split("?")[0].rstrip("/")

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"X-Peak3d-Token": token, "Content-Type": "application/json"},
                                     method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    return call


def test_app_lists_raw_files_and_needs_a_converter(tmp_path, fake_trfp, monkeypatch):
    from peak3d.web import server
    d = _folder(tmp_path / "data")
    monkeypatch.delenv("PEAK3D_TRFP")
    monkeypatch.setenv("PEAK3D_ADDONS", str(tmp_path / "no_addons"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty_path"))
    httpd, url = server.make_server("127.0.0.1", 0, root=tmp_path, token="t")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    call = _client(url, "t")
    try:
        code, ls = call(f"/api/ls?path={quote(str(d))}")
        assert code == 200 and (ls["n_inputs"], ls["n_raw"], ls["n_raw_pending"], ls["n_mzml"]) == (3, 2, 2, 1)
        assert ls["inputs"] == ["S0.raw", "S1.raw", "S2.mzML"]
        code, info = call("/api/info")
        assert info["raw_converter"] is None and info["addons"]["thermo"]["kind"] == "program"
        assert info["addons"]["thermo"]["installed"] is False and "Thermo" in info["addons"]["thermo"]["licence"]
        code, err = call("/api/runs", {"input": str(d), "output": str(tmp_path / "o")})
        assert code == 400 and "2 Thermo .raw file(s) need converting" in err["error"]
        assert not any(f["dest"] == "mzml_dir" for f in server.option_schema("process"))
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_run_progress_through_the_conversion(tmp_path):
    from peak3d.web.server import Run
    r = Run("r1", tmp_path, tmp_path / "o", [], n_files=3, n_convert=2)
    assert r.stage == "convert" and r.progress() == pytest.approx(0.01)
    r._take("[peak3d] converting 2 Thermo .raw file(s) to mzML with ThermoRawFileParser (add-on), 2 at a time, into x")
    r._take("[peak3d] converted S0.raw (3.0 s, 1/2)")
    assert r.stage == "convert" and r.progress() == pytest.approx(0.13)
    r._take("[peak3d] converted S1.raw (3.1 s, 2/2)")
    r._take("[peak3d] raw files converted: 2 in 3.2 s")
    assert r.stage == "compile" and r.progress() == pytest.approx(0.25) and r.status()["converted"] == 2
    r._take("[peak3d] JIT warm-up 2.0 s; picking 3 files with 1 workers x 1 threads")
    assert r.stage == "pick" and r.progress() == pytest.approx(0.25)
    r._take("[peak3d] picked 3 files in 2.0 s: 10 features")
    assert r.stage == "align" and r.progress() == pytest.approx(0.80)
    plain = Run("r2", tmp_path, tmp_path / "o2", [], n_files=3)
    assert plain.stage == "compile" and plain.status()["n_convert"] == 0
    plain._take("[peak3d] kernel: converting asari output (x)")        # not a .raw conversion
    assert plain.stage == "group" and plain.converted == 0


# ------------------------------------------------------------------ the add-on ----
def _fake_zip(tmp_path, exe_mode=0o755) -> tuple[Path, str]:
    """a ThermoRawFileParser-like release archive (flat, as the Linux build): the program prints its version"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        info = zipfile.ZipInfo("ThermoRawFileParser")
        info.external_attr = (stat.S_IFREG | exe_mode) << 16
        z.writestr(info, "#!/bin/sh\necho 2.0.0-test\n")
        z.writestr("ThermoFisher.CommonCore.RawFileReader.dll", b"\0" * 16)
        z.writestr("THERMO_LICENSE", "licence text")
    data = buf.getvalue()
    path = tmp_path / "srv" / "ThermoRawFileParser-test-linux.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


@pytest.mark.skipif(os.name == "nt", reason="the fake program is a shell script")
@pytest.mark.parametrize("mode", [0o755, 0o644])          # an archive without the exec bit still gives a program
def test_thermo_addon_install_and_found(tmp_path, monkeypatch, mode):
    arch, sha = _fake_zip(tmp_path, mode)
    monkeypatch.setenv("PEAK3D_ADDONS", str(tmp_path / "addons"))
    monkeypatch.delenv("PEAK3D_TRFP", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty_path") + os.pathsep + "/bin:/usr/bin")
    monkeypatch.setitem(addons.TRFP_BUILDS, addons.target(), (arch.name, sha))
    assert addons.status("thermo")["installed"] is False and rawfiles.converter() is None
    log = []
    st = addons.install("thermo", log=log.append, url_base=arch.parent.as_uri() + "/")
    assert st["installed"] and st["installed_version"] == "2.0.0-test" and any("Checksum verified" in m for m in log)
    exe = Path(addons.program_of("thermo"))
    assert exe.name == "ThermoRawFileParser" and os.access(exe, os.X_OK)
    info = json.loads((tmp_path / "addons" / "thermo" / "installed.json").read_text())
    assert info["sha256"] == sha and info["executable"] == "program/ThermoRawFileParser"
    assert rawfiles.converter() == {"cmd": [str(exe)], "source": "add-on"}
    assert addons.remove("thermo")["installed"] is False and rawfiles.converter() is None


@pytest.mark.skipif(os.name == "nt", reason="the fake program is a shell script")
def test_thermo_addon_bad_download_or_program_leaves_nothing(tmp_path, monkeypatch):
    arch, sha = _fake_zip(tmp_path)
    monkeypatch.setenv("PEAK3D_ADDONS", str(tmp_path / "addons"))
    monkeypatch.setitem(addons.TRFP_BUILDS, addons.target(), (arch.name, "0" * 64))
    with pytest.raises(RuntimeError, match="checksum"):
        addons.install("thermo", log=_quiet, url_base=arch.parent.as_uri() + "/")
    assert not (tmp_path / "addons" / "thermo").exists()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        info = zipfile.ZipInfo("ThermoRawFileParser")
        info.external_attr = (stat.S_IFREG | 0o755) << 16
        z.writestr(info, "#!/bin/sh\necho 'cannot start: libcoreclr missing'\nexit 3\n")
    broken = tmp_path / "srv" / "broken.zip"
    broken.write_bytes(buf.getvalue())
    monkeypatch.setitem(addons.TRFP_BUILDS, addons.target(), (broken.name, hashlib.sha256(buf.getvalue()).hexdigest()))
    with pytest.raises(RuntimeError, match="does not run .exit 3.: cannot start"):
        addons.install("thermo", log=_quiet, url_base=broken.parent.as_uri() + "/")
    assert not (tmp_path / "addons" / "thermo").exists() and addons.program_of("thermo") is None


def test_thermo_builds_cover_every_app_target():
    assert set(addons.TRFP_BUILDS) == set(addons.PYTHONS)
    assert all(len(sha) == 64 and name.endswith(".zip") for name, sha in addons.TRFP_BUILDS.values())
    assert addons.ADDONS["thermo"]["builds"] is addons.TRFP_BUILDS


@pytest.mark.skipif(os.name == "nt", reason="the fake program is a shell script")
def test_thermo_addon_on_a_mac_signs_the_launcher(tmp_path, monkeypatch):
    """macOS arm64 build of 2.0.0-dev: the launcher is unsigned (and the zip keeps no exec bits); it is ad-hoc signed
    before its first run"""
    arch, sha = _fake_zip(tmp_path, 0)
    monkeypatch.setenv("PEAK3D_ADDONS", str(tmp_path / "addons"))
    monkeypatch.setattr(addons.sys, "platform", "darwin")
    monkeypatch.setitem(addons.TRFP_BUILDS, addons.target(), (arch.name, sha))       # this machine as a Mac
    real_run, signed = subprocess.run, []

    def run(cmd, *a, **kw):
        if cmd[0] == "codesign":
            signed.append([Path(c).name for c in cmd[4:]])
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return real_run(cmd, *a, **kw)
    monkeypatch.setattr(addons.subprocess, "run", run)
    log = []
    assert addons.install("thermo", log=log.append, url_base=arch.parent.as_uri() + "/")["installed"]
    assert signed == [["ThermoRawFileParser"]] and any("Signing the program" in m for m in log)


def test_program_env_drops_the_frozen_apps_library_path(monkeypatch):
    monkeypatch.setattr(addons.sys, "frozen", True, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/app/_internal")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/opt/lib")
    monkeypatch.delenv("DYLD_LIBRARY_PATH", raising=False)
    env = addons.program_env("thermo")
    assert env["LD_LIBRARY_PATH"] == "/opt/lib" and "LD_LIBRARY_PATH_ORIG" not in env
    assert env["DOTNET_SYSTEM_GLOBALIZATION_INVARIANT"] == "1"
    assert env["DOTNET_SYSTEM_GLOBALIZATION_PREDEFINED_CULTURES_ONLY"] == "0"      # RawFileReader creates en-US
    monkeypatch.delenv("LD_LIBRARY_PATH_ORIG")
    assert "LD_LIBRARY_PATH" not in addons.program_env("thermo")
