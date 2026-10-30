"""Browser front end: results bundle, report.html and the local GUI server (no browser needed)."""
import gzip
import json
import pickle
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
RT = ("rt", "rt_min", "rt_max", "rt_sd")


def _in_seconds(header, line):
    """what /api/results/table makes of a table line: retention-time columns in seconds (header renamed *_s)"""
    cols = header.rstrip("\r\n").split("\t")
    eol = header[len(header.rstrip("\r\n")):]
    if line is None:
        return "\t".join(c + "_s" if c in RT else c for c in cols) + eol
    v = line.rstrip("\r\n").split("\t")
    for i, c in enumerate(cols):
        if c in RT and v[i] not in ("", "nan"):
            v[i] = f"{float(v[i]) * 60:.6g}"
    return "\t".join(v) + eol


def test_schema_and_argv():
    sch = {f["dest"]: f for f in server.option_schema()}
    assert {"cores", "min_score", "polarity", "ppm", "no_rt_correction", "no_gap_fill", "quant", "no_report"} <= set(sch)
    assert "input" not in sch and "output" not in sch and "set" not in sch   # GUI runs are always the named variant
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
    assert b'id="variant"' in page and b">GLRS</span>" in page and b"narrow_rule=local_median" in page
    assert c.req("/static/app.js")[0] == 200 and c.req("/static/../server.py")[0] == 404
    assert c.req("/static/style.css")[2]["Content-Type"].startswith("text/css")
    code, info = c.json("/api/info")
    assert info["root"] == str(root.resolve()) and info["cpu_count"] >= 1
    assert info["variant"] == "GLRS" and info["variant_sets"]["min_scans_rule"] == "local"
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
    # the file's own line endings (CRLF where pandas writes os.linesep, i.e. on Windows)
    lines = (root / "o" / "aligned_feature_table.tsv").read_bytes().decode("utf-8").splitlines(keepends=True)
    ids = [int(lines[3].split("\t")[0]), int(lines[1].split("\t")[0]), 999999]
    code, raw, _ = c.req("/api/results/table", {"path": str(root / "o"), "ids": ids})
    assert code == 200 and raw.decode() == "".join(_in_seconds(lines[0], ln) for ln in (None, lines[3], lines[1]))
    assert c.json("/api/results/reveal", {"path": str(root.parent)})[0] == 403          # outside --root
    code, ex = c.json("/api/example", {})
    assert code == 200 and ex["n_mzml"] == 3 and Path(ex["input"]).parent == (root / "peak3d_example").resolve()
    assert c.json(f"/api/results/bundle?path={quote(str(d))}")[0] == 404


# ------------------------------------------------------------------ MS2, identity search, kernels ----
QUERY = [(72.04, 10.0), (110.06, 100.0), (163.06, 40.0)]
LIBRARY = """NAME: Glucose
PRECURSORMZ: 180.0634
PRECURSORTYPE: [M+H]+
IONMODE: Positive
FORMULA: C6H12O6
INCHIKEY: WQZGKKKJIJFFOK-GASJEMHNSA-N
Num Peaks: 3
72.04\t10
110.06\t100
163.06\t40

NAME: Not this one
PRECURSORMZ: 180.0636
IONMODE: Positive
Num Peaks: 2
91.05\t100
119.05\t30

NAME: Far precursor
PRECURSORMZ: 250.1000
IONMODE: Positive
Num Peaks: 1
110.06\t100
"""


def _ms2_output(out: Path):
    """a two-feature output with one representative MS2 spectrum (feature 1), as process writes it"""
    out.mkdir(parents=True)
    pd.DataFrame({"group_id": [1, 2], "mz": [180.0634, 300.2], "rt": [2.0, 4.0], "rt_min": [1.95, 3.95],
                  "rt_max": [2.05, 4.05], "n_detected": [2, 1], "n_confirmed": [0, 0], "n_filled": [0, 1],
                  "mz_ppm_spread": [1.0, np.nan], "rt_sd": [0.01, np.nan], "iso_offset": [0, 0], "charge": [1, 1],
                  "A": [2e6, 5e5], "B": [1.8e6, 0.0]}).to_csv(out / "aligned_feature_table.tsv", sep="\t", index=False)
    (out / "params.json").write_text("{}")
    peaks = "".join(f"{m}\t{i}\n" for m, i in QUERY)
    (out / "ms2_spectra.msp").write_text("NAME: feature_1\nPRECURSORMZ: 180.0634\nRETENTIONTIME: 2.0\nIONMODE: Positive\n"
                                         f"Comment: group_id=1 file=A scan=11 rt_s=120.5 tic=150 n_ms2=3\nNum Peaks: 3\n{peaks}\n")
    pd.DataFrame({"group_id": [1, 1, 1], "file": ["A", "A", "B"], "scan": ["11", "12", "9"], "rt_s": [120.5, 121.0, 119.0],
                  "precursor_mz": [180.0634] * 3, "tic": [150.0, 90.0, 80.0], "n_peaks": [3, 3, 2],
                  "representative": [1, 0, 0]}).to_csv(out / "ms2_map.tsv", sep="\t", index=False)
    return out


def test_identity_search_through_the_server(gui):
    c, root, _ = gui
    out = _ms2_output(root / "res")
    (root / "libs").mkdir()
    (root / "libs" / "mini.msp").write_text(LIBRARY)
    (root / "libs" / "notes.txt").write_text("x")
    (root / "libs" / "Other.pkl").write_bytes(pickle.dumps({"not": "a library"}))
    code, ls = c.json(f"/api/ls?path={quote(str(root / 'libs'))}&ext=msp")
    assert code == 200 and ls["files"] == ["mini.msp", "Other.pkl"]
    code, info = c.json("/api/info")
    assert len(info["methods"]) == 5 and "reverse_entropy" in info["methods"] and set(info["kernels"]) >= {"peak3d", "asari", "masscube"}
    q = quote(str(out))
    # before a search: the representative spectrum only
    h, a = bundle.unpack(gzip.decompress(c.req(f"/api/results/ms2ids?path={q}")[1]))
    assert h["ms2"]["n"] == 1 and h["ms2"]["scan"] == ["11"] and h["ids"] is None and list(a["ms2_row"]) == [0]
    assert np.allclose(a["ms2_mz"], [m for m, _ in QUERY]) and a["ms2_n"][0] == 3
    bad = {"path": str(out), "library": str(root / "libs" / "notes.txt")}
    assert c.json("/api/results/identify", bad)[0] == 400
    code, err = c.json("/api/results/identify", {**bad, "library": str(root / "libs" / "Other.pkl")})
    assert code == 400 and "could not read the library Other.pkl" in err["error"]
    assert c.json("/api/results/identify", {**bad, "library": str(root / "libs" / "mini.msp"), "method": "cosine"})[0] == 400
    report.write_report(out)                                # a report from before the search (no cache: no chromatograms)
    for method in ("weighted_entropy", "entropy", "dot", "weighted_dot", "reverse_entropy"):
        code, meta = c.json("/api/results/identify", {"path": str(out), "library": str(root / "libs" / "mini.msp"),
                                                      "method": method, "min_score": 0.5})
        assert code == 200 and meta["method"] == method and meta["n_queries"] == 1 and meta["n_identified"] == 1, meta
        h, a = bundle.unpack(gzip.decompress(c.req(f"/api/results/ms2ids?path={q}")[1]))
        assert h["ids"]["name"][0] == "Glucose" and a["id_score"][0] == pytest.approx(1.0, abs=1e-4), method
        assert "Far precursor" not in h["ids"]["name"] and list(a["id_row"]) == [0] * len(a["id_row"])
        assert h["ids"]["spectra"] and np.allclose(a["id_mz"][:3], [m for m, _ in QUERY])
    # the report now carries the hits too
    rh, _ = bundle.unpack(gzip.decompress(report.read_bundle(out / "report.html")))
    assert rh["ids"]["method"] == "reverse_entropy" and rh["ms2"]["n"] == 1
    # export: seconds, MS2 count and the best hit
    code, raw, _ = c.req("/api/results/table", {"path": str(out), "ids": [1, 2]})
    rows = [ln.split("\t") for ln in raw.decode().splitlines()]
    head = rows[0]
    assert head[:5] == ["group_id", "mz", "rt_s", "rt_min_s", "rt_max_s"] and "rt_sd_s" in head
    assert head[-13:] == list(server.EXPORT_ID_COLS)
    r1 = dict(zip(head, rows[1]))
    assert rows[1][2] == "120" and r1["n_ms2"] == "3" and r1["annotation"] == "Glucose" and r1["annotation_source"] == "library"
    assert r1["annotation_adduct"] == "[M+H]+" and r1["best_hit"] == "Glucose"
    assert rows[2][-13:] == ["0", "0", "", "", "no"] + [""] * 8 and r1["reviewed"] == "no"
    # the user's annotation: unknown, then back to the library result, then a listed hit; kept apart from the search
    assert c.json("/api/results/annotate", {"path": str(out), "group_id": 1, "hit": {"spectrum": "single", "rank": 9}})[0] == 400
    assert c.json("/api/results/annotate", {"path": str(out), "group_id": 1, "hit": {"spectrum": "single", "rank": "x"}})[0] == 400
    assert c.json("/api/results/annotate", {"path": str(out), "group_id": "one", "hit": None})[0] == 400
    code, r = c.json("/api/results/annotate", {"path": str(out), "group_id": 1, "hit": "unknown"})
    assert code == 200 and r["notes"]["annotation"]["status"] == "unknown"
    h, _ = bundle.unpack(gzip.decompress(c.req(f"/api/results/ms2ids?path={q}")[1]))
    assert h["ann"] == {"0": r["notes"]}
    assert c.req(f"/api/results/report?path={q}")[0] == 200            # opening the report applies waiting refreshes
    rh, _ = bundle.unpack(gzip.decompress(report.read_bundle(out / "report.html")))
    assert rh["ann"]["0"]["annotation"]["status"] == "unknown"          # the report shows it too
    eff = pd.read_csv(out / "feature_annotations.tsv", sep="\t").set_index("group_id")
    assert eff.loc[1, "annotation"] == "unknown" and eff.loc[1, "source"] == "user" and eff.loc[1, "best_hit"] == "Glucose"
    code, r = c.json("/api/results/annotate", {"path": str(out), "group_id": 1, "hit": None})
    assert code == 200 and r["notes"] is None and not (out / "user_annotations.json").exists()
    code, r = c.json("/api/results/annotate", {"path": str(out), "group_id": 1, "hit": {"spectrum": "single", "rank": 1}})
    a = r["notes"]["annotation"]
    assert code == 200 and a["name"] == "Glucose" and a["status"] == "user" and a["confirmed"]
    # comment and stars, also on a feature without MS2; both reach the exported table
    assert c.json("/api/results/note", {"path": str(out), "group_id": 2, "stars": ["purple"]})[0] == 400
    assert c.json("/api/results/note", {"path": str(out), "group_id": 2, "comment": 5})[0] == 400
    code, r = c.json("/api/results/note", {"path": str(out), "group_id": 2, "stars": ["blue", "red"], "comment": "odd\tshape"})
    assert code == 200 and r["notes"] == {"stars": ["red", "blue"], "comment": "odd\tshape"}
    code, raw, _ = c.req("/api/results/table", {"path": str(out), "ids": [1, 2]})
    rows = [dict(zip(head, ln.split("\t"))) for ln in raw.decode().splitlines()[1:]]
    assert rows[0]["reviewed"] == "yes" and rows[1]["stars"] == "red;blue" and rows[1]["comment"] == "odd shape"
    assert rows[1]["reviewed"] == "no" and rows[1]["n_ms2"] == "0"
    assert c.req(f"/api/results/file?path={q}&name=identifications.tsv")[0] == 200
    full = bundle.unpack(bundle.build(out))[0]
    assert full["ids"]["n"] >= 1 and full["kernel"] == "peak3d"


def test_kernel_choice(gui):
    c, root, _ = gui
    ext = {f["dest"] for f in c.json("/api/schema?kernel=asari")[1]}
    assert {"cores", "polarity", "library", "similarity"} <= ext and not ext & {"kernel", "input", "output", "min_score"}
    d = _study(root / "kd", n_files=1, n=5)
    code, err = c.json("/api/runs", {"kernel": "xcms", "input": str(d), "output": str(root / "k")})
    assert code == 400 and "kernel" in err["error"]
    info = c.json("/api/info")[1]
    if not info["kernels"]["asari"]["available"]:
        code, err = c.json("/api/runs", {"kernel": "asari", "input": str(d), "output": str(root / "k")})
        assert code == 400 and "not available" in err["error"]
    assert server.command_args("masscube") == ["external", "--kernel", "masscube"] and server.command_args("peak3d") == ["process"]
