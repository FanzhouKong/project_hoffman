"""Optional feature finders installed on request (addons.py): download check, the fix, all-or-nothing, the API."""
import hashlib
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import pytest

from peak3d import addons, external

pytestmark = pytest.mark.skipif(os.name == "nt", reason="the fake Python is a shell script")

ORIGINAL = "def find_isotope(signals):\n    mzs = signals[:, 0]\n    intens = signals[:, 1]\n    return mzs\n"


def _fake_archive(tmp_path) -> tuple[Path, str]:
    """a python-build-standalone-like archive whose python/bin/python3 runs this interpreter"""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        script = f"#!/bin/sh\nexec {sys.executable} \"$@\"\n".encode()
        info = tarfile.TarInfo("python/bin/python3")
        info.size, info.mode = len(script), 0o755
        t.addfile(info, io.BytesIO(script))
    data = buf.getvalue()
    path = tmp_path / "srv" / "cpython-test-install_only.tar.gz"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """addons home in tmp, a local archive, pip and site-packages stubbed, the import check on modules we have"""
    arch, sha = _fake_archive(tmp_path)
    monkeypatch.setenv("PEAK3D_ADDONS", str(tmp_path / "addons"))
    monkeypatch.setitem(addons.PYTHONS, addons.target(), (arch.name, sha))
    site = tmp_path / "site"
    (site / "masscube").mkdir(parents=True)
    (site / "masscube" / "feature_grouping.py").write_text(ORIGINAL)
    pip = []
    monkeypatch.setattr(addons, "_run", lambda cmd, log: pip.append([str(c) for c in cmd]))
    monkeypatch.setattr(addons, "_site_packages", lambda py: site)
    spec = dict(addons.ADDONS["masscube"], module="json", dist="pytest")       # importable by the fake Python
    monkeypatch.setitem(addons.ADDONS, "masscube", spec)
    monkeypatch.setitem(external.KERNELS, "masscube", dict(external.KERNELS["masscube"], package="json", dist="pytest"))
    monkeypatch.delenv("PEAK3D_MASSCUBE_PYTHON", raising=False)
    external.refresh()
    yield dict(base=arch.parent.as_uri() + "/", site=site, pip=pip, home=tmp_path / "addons")
    external.refresh()


def test_install_checks_downloads_patches_and_is_found(fake):
    assert addons.status("masscube")["installed"] is False and addons.python_of("masscube") is None
    log = []
    st = addons.install("masscube", log=log.append, url_base=fake["base"])
    assert st["installed"] and any("Checksum verified" in m for m in log)
    assert fake["pip"][0][-1] == "masscube==1.2.22" and "--no-input" in fake["pip"][0]
    text = (fake["site"] / "masscube" / "feature_grouping.py").read_text()
    assert "added by peak3d" in text and text.count("if signals is None") == 1
    info = json.loads((fake["home"] / "masscube" / "installed.json").read_text())
    assert info["licence"].startswith("CC BY-NC") and info["python_sha256"]
    assert not (fake["home"] / ".masscube.partial").exists()
    external.refresh()
    k = external.kernel_status()["masscube"]
    assert k["available"] and k["python"] == addons.python_of("masscube")
    addons.install("masscube", log=log.append, url_base=fake["base"])          # again: the fix is not doubled
    assert (fake["site"] / "masscube" / "feature_grouping.py").read_text().count("if signals is None") == 1
    assert addons.remove("masscube")["installed"] is False and not (fake["home"] / "masscube").exists()


def test_a_bad_download_or_unexpected_file_leaves_nothing(fake, monkeypatch):
    name, _ = addons.PYTHONS[addons.target()]
    monkeypatch.setitem(addons.PYTHONS, addons.target(), (name, "0" * 64))
    with pytest.raises(RuntimeError, match="checksum"):
        addons.install("masscube", log=lambda m: None, url_base=fake["base"])
    assert not (fake["home"] / "masscube").exists() and not (fake["home"] / ".masscube.partial").exists()
    monkeypatch.undo()
    arch, sha = _fake_archive(fake["home"].parent / "again")
    monkeypatch.setenv("PEAK3D_ADDONS", str(fake["home"]))
    monkeypatch.setitem(addons.PYTHONS, addons.target(), (arch.name, sha))
    monkeypatch.setattr(addons, "_run", lambda cmd, log: None)
    monkeypatch.setattr(addons, "_site_packages", lambda py: fake["site"])
    (fake["site"] / "masscube" / "feature_grouping.py").write_text("something else entirely\n")
    with pytest.raises(RuntimeError, match="unexpected file contents"):
        addons.install("masscube", log=lambda m: None, url_base=arch.parent.as_uri() + "/")
    assert not (fake["home"] / "masscube").exists()


def test_install_through_the_server(fake, monkeypatch, tmp_path):
    import threading
    from peak3d.web import server
    real = addons.install
    monkeypatch.setattr(addons, "install", lambda name, log=print: real(name, log=log, url_base=fake["base"]))
    httpd, url = server.make_server("127.0.0.1", 0, root=tmp_path, token="t")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    import urllib.request

    def call(path, body=None):
        req = urllib.request.Request(url.split("?")[0].rstrip("/") + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"X-Peak3d-Token": "t", "Content-Type": "application/json"},
                                     method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    try:
        code, info = call("/api/info")
        assert info["addons"]["masscube"]["installed"] is False and info["addons"]["masscube"]["licence"].startswith("CC BY-NC")
        assert call("/api/addons/install", {"name": "masscube"})[0] == 400                 # licence not accepted
        assert call("/api/addons/install", {"name": "xcms", "accept_licence": True})[0] == 400
        assert call("/api/addons/install", {"name": "masscube", "accept_licence": True}) == (200, {"state": "running"})
        for _ in range(100):
            ad = call("/api/info")[1]["addons"]["masscube"]
            if ad["job"]["state"] != "running":
                break
            time.sleep(0.1)
        assert ad["job"]["state"] == "done" and ad["installed"] and any("Checksum" in m for m in ad["job"]["log"])
        assert call("/api/info")[1]["kernels"]["masscube"]["available"]
        code, st = call("/api/addons/remove", {"name": "masscube"})
        assert code == 200 and st["installed"] is False
        assert call("/api/info")[1]["kernels"]["masscube"]["available"]      # json is importable here too
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_targets():
    assert set(addons.PYTHONS) == {"aarch64-apple-darwin", "x86_64-apple-darwin", "x86_64-pc-windows-msvc",
                                   "x86_64-unknown-linux-gnu"}
    assert all(len(sha) == 64 for _, sha in addons.PYTHONS.values())
