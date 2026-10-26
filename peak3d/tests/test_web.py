"""Browser front end: results bundle, report.html and the local GUI server (no browser needed)."""
import gzip
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import pytest

from peak3d.synth import Peak, make_cloud, write_mzml
from peak3d.web import bundle, report, server

ROOT = Path(__file__).resolve().parents[2]


def _study(d: Path, n_files=3, n=30, seed=1):
    rng = np.random.default_rng(seed)
    base = list(zip(rng.uniform(150, 900, n), rng.uniform(0.6, 3.4, n), 10 ** rng.uniform(4.5, 6.5, n)))
    d.mkdir(parents=True)
    for k in range(n_files):
        sh = (0.0, 0.04, -0.03)[k % 3]
        peaks = [Peak(float(m), float(r + sh), float(h), 0.05, iso=(0.15,)) for m, r, h in base]
        cloud, _ = make_cloud(peaks, n_scans=800, dt=0.005, thresh=200.0, floor=100.0, n_noise=8000, seed=k,
                              name=f"S{k}")
        write_mzml(cloud, d / f"S{k}.mzML")
    return d


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    """one study processed through the CLI (cache kept so chromatograms can be checked against it)"""
    tmp = tmp_path_factory.mktemp("web")
    d = _study(tmp / "data")
    out = tmp / "out"
    r = subprocess.run([sys.executable, "-m", "peak3d", "process", "--input", str(d), "--output", str(out), "--cores", "2",
                        "--keep-cache"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return d, out


# ------------------------------------------------------------------ bundle ----
def test_pack_roundtrip():
    arrays = {"a": np.arange(5, dtype=np.float64), "b": np.array([1, 2, 3], np.uint16), "c": np.array([-1, 2], np.int8),
              "d": np.zeros(0, np.float32)}
    h, back = bundle.unpack(bundle.pack({"x": 1, "s": ["p", "q"]}, arrays))
    assert h == {"x": 1, "s": ["p", "q"]}
    for k, v in arrays.items():
        assert back[k].dtype == v.dtype and np.array_equal(back[k], v)


def _eic_reference(cloud, rt, target, tol, x0, dx, n_pts):
    lo_t, hi_t = x0 - 0.5 * dx, x0 + (n_pts - 0.5) * dx
    a = max(int(np.searchsorted(rt, lo_t)) - 1, 0)
    b = min(int(np.searchsorted(rt, hi_t)) + 1, len(rt))
    vals = np.array([max([v for m, v in zip(cloud.mz[cloud.off[s]:cloud.off[s + 1]], cloud.inten[cloud.off[s]:cloud.off[s + 1]])
                          if abs(m - target) <= tol] or [0.0]) for s in range(a, b)])
    out = np.zeros(n_pts)
    seen = np.zeros(n_pts, bool)
    for s in range(a, b):
        j = int(np.floor((rt[s] - lo_t) / dx))
        if 0 <= j < n_pts:
            out[j] = max(out[j], vals[s - a]) if seen[j] else vals[s - a]
            seen[j] = True
    t = x0 + dx * np.arange(n_pts)
    fill = np.interp(t, rt[a:b], vals)
    inside = (t > rt[a]) & (t < rt[b - 1])
    out[~seen & inside] = fill[~seen & inside]
    return out


@pytest.mark.parametrize("dx", [0.0023, 0.011])     # grid finer than the scans (interpolates) and coarser (max-pools)
def test_eic_on_grid_matches_reference(dx):
    cloud, _ = make_cloud([Peak(300.0, 1.0, 1e5, 0.05), Peak(300.004, 1.3, 4e4, 0.05)], n_scans=400, dt=0.005,
                          floor=50.0, n_noise=3000, seed=3)
    rt = cloud.rt + 0.01 * np.sin(cloud.rt)            # a warped, still increasing axis
    target, tol = np.array([300.0]), np.array([300.0 * 10e-6])
    x0 = 1.0 - 24 * dx                                  # grid centred on the first peak
    got = bundle.eic_on_grid(cloud.off, cloud.mz, cloud.inten, rt, target, tol, np.array([x0]), np.array([dx]), 48)[0]
    ref = _eic_reference(cloud, rt, 300.0, 300.0 * 10e-6, x0, dx, 48)
    assert np.allclose(got, ref, rtol=1e-5, atol=1e-3)
    assert got.max() > 5e4                              # the apex is on the grid


def test_report_contents(processed):
    d, out = processed
    rep = out / "report.html"
    assert rep.exists()
    html = rep.read_text(encoding="utf-8")
    assert "<title>peak3d · out</title>" in html and "{{" not in html
    assert "<link" not in html and 'src="http' not in html and "url(http" not in html      # nothing loads from the network
    h, a = bundle.unpack(gzip.decompress(report.read_bundle(rep)))
    t = pd.read_csv(out / "aligned_feature_table.tsv", sep="\t")
    stems = ["S0", "S1", "S2"]
    assert h["n_groups"] == len(t) > 25 and h["stems"] == stems
    assert np.array_equal(a["g_id"], t["group_id"].to_numpy())
    assert np.allclose(a["g_mz"], t["mz"].to_numpy())
    # per-file values decode to the table within the log-quantisation step
    m = h["matrix"]
    q = a["m_q"].reshape(len(t), 3).astype(float)
    dec = np.where(q > 0, 10 ** (m["log10_lo"] + (q - 1) / 65534 * (m["log10_hi"] - m["log10_lo"])), 0)
    V = t[stems].to_numpy(float)
    assert np.allclose(dec, V, rtol=5e-4, atol=0)
    # detected cells agree with filled_mask (0 = detected)
    st = a["m_status"].reshape(len(t), 3)
    fm = pd.read_csv(out / "filled_mask.tsv", sep="\t").set_index("group_id").loc[t["group_id"], stems].to_numpy()
    assert np.array_equal(st == 3, fm == 0)
    # chromatograms for every group (small study), and their apex is the feature's height in that file
    e = h["eic"]
    assert e["n"] == len(t) and e["files"] == [0, 1, 2]
    tr = a["eic_q"].reshape(e["n"], 3, e["points"]).astype(float) * a["eic_scale"][:, None, None] / 65535
    det = st == 3
    apex = tr.max(axis=2)
    ratio = apex[det] / V[det]
    assert np.median(np.abs(ratio - 1)) < 0.05
    assert len(h["files"]) == 3 and {"n_features", "role", "mz_offset_ppm"} <= set(h["files"][0])
    assert h["residuals"]["n_after"] > 0


def test_report_command_without_cache(processed, tmp_path):
    _, out = processed
    dest = tmp_path / "r.html"
    r = subprocess.run([sys.executable, "-m", "peak3d", "report", str(out), "--cache-dir", str(tmp_path / "gone"),
                        "--path", str(dest)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    h, a = bundle.unpack(gzip.decompress(report.read_bundle(dest)))
    assert h["eic"] is None and "cache" in h["eic_note"] and "eic_q" not in a
    assert h["n_groups"] > 0 and "m_q" in a


def test_matrix_limit(processed):
    _, out = processed
    h, a = bundle.unpack(bundle.build(out, max_cells=10, eic_budget=0))
    assert h["matrix"] is None and "cell limit" in h["matrix_note"] and "m_q" not in a and h["eic"] is None


def test_eic_budget_keeps_most_intense(processed):
    _, out = processed
    h, a = bundle.unpack(bundle.build(out, eic_budget=3 * 48 * 5))
    assert h["eic"]["n"] == 5
    top = np.sort(np.argsort(-a["g_max"], kind="stable")[:5])
    assert np.array_equal(a["eic_row"], top)


def test_not_an_output(tmp_path):
    with pytest.raises(FileNotFoundError):
        bundle.build(tmp_path)


# ------------------------------------------------------------------ server ----
def test_schema_and_argv():
    sch = {f["dest"]: f for f in server.option_schema()}
    assert {"cores", "min_score", "polarity", "ppm", "no_rt_correction", "no_gap_fill", "quant", "no_report"} <= set(sch)
    assert "input" not in sch and "output" not in sch
    assert sch["cores"]["kind"] == "int" and sch["cores"]["min_exclusive"] == 0
    assert sch["min_score"]["kind"] == "float" and sch["min_score"]["max"] == 1
    assert sch["polarity"]["kind"] == "choice" and sch["no_gap_fill"]["kind"] == "flag"
    argv = server.options_to_argv({"cores": 4, "min_score": 0.5, "no_gap_fill": True, "polarity": None, "ppm": "7"},
                                  list(sch.values()))
    assert argv == ["--cores", "4", "--ppm", "7", "--no-gap-fill"]          # parser order, defaults left out
    assert server.validate_argv(["process", "--input", "x", "--output", "y", *argv]) is None
    assert "must be in [0, 1]" in server.validate_argv(["process", "--input", "x", "--output", "y", "--min-score", "1.5"])


class _Client:
    def __init__(self, url, token):
        self.base, self.token = url, token

    def req(self, path, body=None, token=True, host=None, method=None, cookie=False):
        headers = {"Content-Type": "application/json"}
        if cookie:
            headers["Cookie"] = f"peak3d_token={self.token}"
        if token:
            headers["X-Peak3d-Token"] = self.token
        if host:
            headers["Host"] = host
        data = json.dumps(body).encode() if body is not None else None
        rq = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            with opener.open(rq, timeout=30) as r:
                return r.status, r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, e.read(), dict(e.headers)

    def json(self, path, body=None, **kw):
        code, raw, _ = self.req(path, body, **kw)
        return code, json.loads(raw)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


@pytest.fixture()
def gui(tmp_path):
    httpd, url = server.make_server("127.0.0.1", 0, root=tmp_path, token="t0ken")
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield _Client(url.split("?")[0].rstrip("/"), "t0ken"), tmp_path, httpd
    for r in httpd.app.runs.values():
        r.cancel()
    httpd.shutdown()
    httpd.server_close()


def test_server_guards(gui):
    c, root, _ = gui
    assert c.req("/", token=False)[0] == 403
    code, _, hdr = c.req("/?token=t0ken", token=False)
    assert code == 302 and "t0ken" in hdr.get("Set-Cookie", "") and "HttpOnly" in hdr["Set-Cookie"]
    assert c.req("/api/info", host="evil.example:8765")[0] == 403          # DNS rebinding
    assert c.req("/api/info", host="localhost:1")[0] == 200
    code, page, _ = c.req("/")
    assert code == 200 and b"t0ken" in page and b"/static/app.js" in page
    assert c.req("/static/app.js")[0] == 200 and c.req("/static/../server.py")[0] == 404
    assert c.req("/static/style.css")[2]["Content-Type"].startswith("text/css")
    code, info = c.json("/api/info")
    assert info["root"] == str(root.resolve()) and info["cpu_count"] >= 1
    assert c.json("/api/ls?path=/")[0] == 403                               # outside --root
    assert c.req("/api/runs", {"input": str(root)}, token=False)[0] == 403
    code, err = c.json("/api/runs", {"input": str(root)}, token=False, cookie=True)   # cookie alone: no CSRF
    assert code == 403 and "header" in err["error"]
    assert c.req("/api/info", token=False, cookie=True)[0] == 200


def test_server_run_and_results(gui):
    c, root, httpd = gui
    d = _study(root / "data", n_files=2, n=20)
    (root / "data" / "sub").mkdir()
    code, ls = c.json(f"/api/ls?path={quote(str(d))}")
    assert code == 200 and ls["n_mzml"] == 2 and ls["dirs"] == ["sub"] and ls["parent"] == str(root.resolve())
    code, err = c.json("/api/runs", {"input": str(d / "sub"), "output": str(root / "o")})
    assert code == 400 and "no .mzML" in err["error"]
    code, err = c.json("/api/runs", {"input": str(d), "output": str(root / "o"), "options": {"min_score": 1.5}})
    assert code == 400 and "[0, 1]" in err["error"]
    code, run = c.json("/api/runs", {"input": str(d), "output": str(root / "o"), "options": {"cores": 2}})
    assert code == 201 and run["state"] == "running" and "--cores 2" in run["command"]
    t0 = time.time()
    while True:
        code, st = c.json(f"/api/runs/{run['id']}")
        if st["state"] != "running" or time.time() - t0 > 600:
            break
        time.sleep(1)
    assert st["state"] == "done", "\n".join(st["log"])
    assert st["progress"] == 1.0 and any("report:" in ln for ln in st["log"])
    assert (root / "o" / "peak3d_gui.log").exists()
    code, raw, _ = c.req(f"/api/results/bundle?path={quote(str(root / 'o'))}")
    h, a = bundle.unpack(gzip.decompress(raw))
    assert code == 200 and h["n_groups"] > 15 and h["eic"] is not None     # from the report, made while the cache existed
    code, page, hdr = c.req(f"/api/results/report?path={quote(str(root / 'o'))}&download=1")
    assert code == 200 and b'id="peak3d-data"' in page and "attachment" in hdr["Content-Disposition"]
    code, again = c.json("/api/runs", {"input": str(d), "output": str(root / "o")})
    assert code == 409 and again["confirm"] == "overwrite"
    # downloads: whole files by name, and chosen rows exactly as in the file
    out_q = quote(str(root / "o"))
    code, raw, hdr = c.req(f"/api/results/file?path={out_q}&name=aligned_feature_table.tsv")
    assert code == 200 and raw == (root / "o" / "aligned_feature_table.tsv").read_bytes()
    assert "attachment" in hdr["Content-Disposition"] and "o_aligned_feature_table.tsv" in hdr["Content-Disposition"]
    assert c.req(f"/api/results/file?path={out_q}&name=../../x")[0] == 404
    lines = (root / "o" / "aligned_feature_table.tsv").read_text().splitlines(keepends=True)
    ids = [int(lines[3].split("\t")[0]), int(lines[1].split("\t")[0]), 999999]
    code, raw, _ = c.req("/api/results/table", {"path": str(root / "o"), "ids": ids})
    assert code == 200 and raw.decode() == lines[0] + lines[3] + lines[1]
    assert c.json("/api/results/reveal", {"path": str(root.parent)})[0] == 403          # outside --root
    code, ex = c.json("/api/example", {})
    assert code == 200 and ex["n_mzml"] == 3 and Path(ex["input"]).parent == (root / "peak3d_example").resolve()
    assert c.json(f"/api/results/bundle?path={quote(str(d))}")[0] == 404
