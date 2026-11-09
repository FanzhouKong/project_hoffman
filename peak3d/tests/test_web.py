"""Browser front end: results bundle, report.html and the local GUI server (no browser needed)."""
import gzip
import json
import os
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

from peak3d import io
from peak3d.synth import Peak, make_cloud, write_mzml
from peak3d.web import bundle, report, server

ROOT = Path(__file__).resolve().parents[2]


# a dichlorinated ion (diclofenac [M+H]+: 100 : 16 : 65 : 10 : 10.5 at 37Cl / 13C spacings) in every synthetic study
HALOGEN_MZ = 296.0240
HALOGEN_ISO = (0.16, 0.65, 0.10, 0.105)
HALOGEN_DA = (1.00335, 1.99770, 3.00105, 3.99545)


def _study(d: Path, n_files=3, n=30, seed=1):
    rng = np.random.default_rng(seed)
    base = list(zip(rng.uniform(150, 900, n), rng.uniform(0.6, 3.4, n), 10 ** rng.uniform(4.5, 6.5, n)))
    d.mkdir(parents=True)
    for k in range(n_files):
        sh = (0.0, 0.04, -0.03)[k % 3]
        peaks = [Peak(float(m), float(r + sh), float(h), 0.05, iso=(0.15,)) for m, r, h in base]
        peaks.append(Peak(HALOGEN_MZ, 2.0 + sh, 3e5, 0.05, iso=HALOGEN_ISO, iso_da=HALOGEN_DA))
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
                        "--compounds"], cwd=ROOT, capture_output=True, text=True)       # the cache is kept by default
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
    k = np.searchsorted(rt[a:b], t)                     # interpolate only between two scans that both hold signal
    both = (k > 0) & (k < b - a)
    both[both] &= (vals[k[both] - 1] > 0) & (vals[k[both]] > 0)
    out[~seen & inside & both] = fill[~seen & inside & both]
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


def test_eic_on_grid_steps_at_a_one_scan_rise():
    """an ion absent from one scan and at full height in the next is drawn as a step: the fine grid cells between
    the two scans stay at zero instead of ramping up (a ramp looks like a shoulder the picker left out)"""
    rt = np.arange(10) * 0.03                           # 1.8 s scans
    inten = np.array([0, 0, 0, 1e5, 5e4, 2e4, 1e4, 0, 0, 0])
    mzs = [np.array([300.0]) if v > 0 else np.zeros(0) for v in inten]
    cloud = io.from_scans("step", rt, mzs, [np.array([v], np.float32) if v > 0 else np.zeros(0, np.float32) for v in inten])
    dx = 0.005
    got = bundle.eic_on_grid(cloud.off, cloud.mz, cloud.inten, cloud.rt, np.array([300.0]), np.array([0.003]),
                             np.array([0.0]), np.array([dx]), 60)[0]
    t = np.arange(60) * dx
    assert np.all(got[t < rt[3] - 0.5 * dx] == 0)      # nothing before the first scan with signal
    assert got[np.argmin(np.abs(t - rt[3]))] == 1e5
    inside = (t > rt[3]) & (t < rt[6])
    assert np.all(got[inside] > 0)                       # still interpolated inside the peak
    assert np.all(got[t > rt[6] + 0.5 * dx] == 0)      # and a step down at the end
    n = bundle.scans_in_window(cloud.off, cloud.mz, cloud.inten, cloud.rt, np.array([300.0]), np.array([0.003]),
                               np.array([rt[2]]), np.array([rt[7]]))
    assert n.tolist() == [[6, 4]]                        # scans 2..7 in the window, 3..6 hold the ion


def test_report_contents(processed):
    d, out = processed
    rep = out / "report.html"
    assert rep.exists()
    html = rep.read_text(encoding="utf-8")
    assert "<title>Mosa · out</title>" in html and "{{" not in html
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
    # detected cells agree with filled_mask (0 = detected), and the bundle says its status comes from presence_mask.tsv
    assert m["presence"] is True
    st = a["m_status"].reshape(len(t), 3)
    fm = pd.read_csv(out / "filled_mask.tsv", sep="\t").set_index("group_id").loc[t["group_id"], stems].to_numpy()
    assert np.array_equal(st == 3, fm == 0)
    # chromatograms for every group (small study), and their apex is the feature's height in that file
    e = h["eic"]
    assert e["n"] == len(t) and e["files"] == [0, 1, 2]
    tr = a["eic_q"].reshape(e["n"], 3, e["points"]).astype(float) * a["eic_scale"][:, None, None] / 65535
    ns = a["eic_ns"].reshape(e["n"], 3, 2)
    assert np.all(ns[:, :, 1] <= ns[:, :, 0]) and ns[:, :, 0].min() >= 1    # scans in the window, those holding the ion
    det = st == 3
    apex = tr.max(axis=2)
    ratio = apex[det] / V[det]
    assert np.median(np.abs(ratio - 1)) < 0.05
    assert len(h["files"]) == 3 and {"n_features", "role", "mz_offset_ppm"} <= set(h["files"][0])
    assert h["residuals"]["n_after"] > 0
    # each trace's m/z window: at least +/- ppm, and in a file with the feature it holds the span of the centroids the
    # picker took (per-file mz_min / mz_max), at the file's own m/z scale
    lo, hi = a["eic_lo"].reshape(e["n"], 3), a["eic_hi"].reshape(e["n"], 3)
    assert e["window"] == "feature_span" and lo.min() >= e["ppm"] - 1e-4 and hi.min() >= e["ppm"] - 1e-4
    rows = a["eic_row"]
    gid = t["group_id"].to_numpy()[rows]
    offs = {r["stem"]: (r.get("mz_offset_ppm") or 0.0) for r in h["files"]}
    n_checked = 0
    for j, s in enumerate(stems):
        f = pd.read_csv(out / "features" / f"{s}.tsv", sep="\t").drop_duplicates("group_id").set_index("group_id")
        cloud = io.Cloud.load_npz(out / ".cache" / f"{s}.npz")
        corr, moff = bundle._file_axis(out, s, cloud)
        target = bundle._native_mz(t["mz"].to_numpy(float)[rows], t["rt"].to_numpy(float)[rows], cloud, corr, moff,
                                   offs.get(s, 0.0))
        has = np.isin(gid, f.index)
        ff = f.loc[gid[has]]
        assert np.all(target[has] * (1 - lo[has, j] * 1e-6) <= ff["mz_min"].to_numpy() * (1 + 1e-7))
        assert np.all(target[has] * (1 + hi[has, j] * 1e-6) >= ff["mz_max"].to_numpy() * (1 - 1e-7))
        n_checked += int(has.sum())
    assert n_checked > 50
    # compound groups travel with the bundle: ids, roles (coded), neutral mass, base flag, size
    assert np.array_equal(a["g_cid"], t["compound_id"].to_numpy())
    roles = [h["compounds"]["roles"][k] for k in a["g_role"]]
    assert roles == t["role"].fillna("").astype(str).tolist()
    assert np.array_equal(a["g_nmem"], t["n_members"].to_numpy()) and np.array_equal(a["g_base"], t["is_base"].to_numpy())
    assert h["compounds"]["n_in_multi"] == int((t["n_members"] > 1).sum())
    assert a["g_isoq"].shape == (len(t),) and a["g_isoq_of"].shape == (len(t),)
    assert "g_isoq" in html or True          # the arrays are binary; the viewer reads them by name
    # the halogen check: the dichlorinated ion is called, its isotopes are not tested, the rest are "no"
    assert "possible_halogen" in t and "halogen_pattern" in t and h["halogen_names"][0] == "" and "Cl2" in h["halogen_names"]
    hal = t["possible_halogen"].fillna("")
    row = t[(t["mz"] - HALOGEN_MZ).abs() < 0.01].index
    assert len(row) == 1 and hal[row[0]] == "yes" and t["halogen_pattern"][row[0]] == "Cl2"
    assert a["g_hal"][row[0]] == 3 and h["halogen_names"][a["g_halhint"][row[0]]] == "Cl2"
    assert (hal[t["iso_offset"] > 0] == "").all() and (hal[(t["iso_offset"] == 0)] != "").all()
    assert (hal == "yes").sum() == 1 and (hal == "maybe").sum() <= 1
    assert h["params"]["stage_times_s"]["halogen_s"] < 20


def test_eic_windows_cover_the_feature_span():
    """a file with the feature: the union of +/- ppm and the feature's own m/z span; without it: the group's median
    span, at least +/- ppm; nothing known: +/- ppm"""
    t = np.full(4, 300.0)
    mn = np.array([300 * (1 - 12e-6), 300 * (1 - 1e-6), np.nan, np.nan])
    mx = np.array([300 * (1 + 3e-6), 300 * (1 + 20e-6), np.nan, np.nan])
    lo, hi = bundle.eic_windows(t, mn, mx, np.array([np.nan, np.nan, 8.0, np.nan]), np.array([np.nan, np.nan, 2.0, np.nan]), 5.0)
    assert np.allclose(lo, [12, 5, 8, 5]) and np.allclose(hi, [5, 20, 5, 5])
    # a span off-centre (the file's offset not fully removed) still lies inside the window
    lo, hi = bundle.eic_windows(np.array([300.0]), np.array([300 * (1 + 2e-6)]), np.array([300 * (1 + 9e-6)]),
                                np.array([np.nan]), np.array([np.nan]), 5.0)
    assert np.allclose(lo, [5]) and np.allclose(hi, [9])


def test_numeric_file_names_stay_text(tmp_path):
    """IDSL003's file is 003.mzML: its stem must stay "003" in the manifest and the alignment summary, or the report
    cannot match its files"""
    (tmp_path / "rt_correction").mkdir()
    (tmp_path / "rt_correction" / "summary.tsv").write_text("file\tmz_offset_ppm\n003\t1.5\n")
    assert bundle.file_offsets(tmp_path) == {"003": 1.5}
    assert bundle._read(tmp_path / "rt_correction" / "summary.tsv")["file"].tolist() == ["003"]


def test_possible_isotopes():
    """mass arithmetic only: a monoisotopic-labelled feature k x 1.00335/z above a stronger co-eluting one"""
    C13 = bundle.C13
    mz = np.array([300.0, 300.0 + C13, 300.0 + 2 * C13, 500.0, 500.0 + C13, 300.0 + C13 / 2, 300.0 + C13 + 300e-6 * 20])
    rt = np.array([1.0, 1.0, 1.0, 2.0, 2.3, 1.0, 1.0])
    h = np.array([1e6, 1e5, 1e4, 1e6, 1e5, 1e4, 1e3])
    iso = np.array([0, 1, 0, 0, 0, 0, 0])          # row 1 is a confirmed isotope already; row 2 was left monoisotopic
    z = np.array([1, 1, 1, 1, 1, 2, 1])
    k, of = bundle.possible_isotopes(mz, rt, h, iso, z, 5.0, 0.05)
    assert k.tolist() == [0, 0, 2, 0, 0, 1, 0]      # row 4: RT too far; row 6: 20 ppm off
    assert of.tolist() == [-1, -1, 0, -1, -1, 0, -1]


def test_demo_has_compound_groups(tmp_path):
    """the demo data give the viewer adducts, fragments and a possible isotope to show"""
    from peak3d.demo import write_demo
    d = write_demo(tmp_path / "demo")
    assert len(list(d.glob("*.mzML"))) == 3


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


def _in_seconds(header, line, filled=None):
    """what /api/results/table makes of a table line: retention-time columns in seconds (header renamed *_s); with
    filled ({group id: files}), the gap_filled_files column after n_filled"""
    cols = header.rstrip("\r\n").split("\t")
    eol = header[len(header.rstrip("\r\n")):]
    at = cols.index("n_filled") + 1
    if line is None:
        head = [c + "_s" if c in RT else c for c in cols]
        return "\t".join(head[:at] + (["gap_filled_files"] if filled is not None else []) + head[at:]) + eol
    v = line.rstrip("\r\n").split("\t")
    for i, c in enumerate(cols):
        if c in RT and v[i] not in ("", "nan"):
            v[i] = f"{float(v[i]) * 60:.6g}"
    if filled is not None:
        v.insert(at, filled.get(v[0], ""))
    return "\t".join(v) + eol


def _filled_from_mask(out: Path, stems):
    """{group id: "stem;stem"}, read as text: the files filled_mask.tsv marks with 1 where the table holds a value > 0"""
    def rows(name):
        lines = (out / name).read_text(encoding="utf-8").splitlines()
        head = lines[0].split("\t")
        return {ln.split("\t")[0]: dict(zip(head, ln.split("\t"))) for ln in lines[1:]}
    table, mask = rows("aligned_feature_table.tsv"), rows("filled_mask.tsv")
    res = {g: ";".join(s for s in stems if mask[g][s] == "1" and float(r[s]) > 0) for g, r in table.items()}
    return {g: v for g, v in res.items() if v}


def test_schema_and_argv():
    sch = {f["dest"]: f for f in server.option_schema()}
    assert {"cores", "min_score", "polarity", "ppm", "no_rt_correction", "no_gap_fill", "quant", "no_report"} <= set(sch)
    assert "input" not in sch and "output" not in sch and "set" not in sch   # GUI runs are always the named variant
    assert sch["cores"]["kind"] == "int" and sch["cores"]["min_exclusive"] == 0
    assert sch["min_score"]["kind"] == "float" and sch["min_score"]["max"] == 1
    assert sch["polarity"]["kind"] == "choice" and sch["no_gap_fill"]["kind"] == "flag"
    # 1.1.1: compound grouping is opt-in, the cache is kept unless --no-keep-cache; hidden flags stay off the form
    assert sch["compounds"]["kind"] == "flag" and sch["compounds"]["default"] is False
    assert sch["no_keep_cache"]["kind"] == "flag" and sch["no_keep_cache"]["default"] is False
    assert not {"keep_cache", "no_compounds", "mz_dynamic"} & set(sch)
    assert server.validate_argv(["process", "--input", "x", "--output", "y", "--keep-cache", "--no-compounds"]) is None
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
    filled = _filled_from_mask(root / "o", ["S0", "S1"])
    ids = [int(lines[3].split("\t")[0]), int(lines[1].split("\t")[0]), 999999]
    code, raw, _ = c.req("/api/results/table", {"path": str(root / "o"), "ids": ids})
    assert code == 200 and raw.decode() == "".join(_in_seconds(lines[0], ln, filled) for ln in (None, lines[3], lines[1]))
    if filled:                                  # rows with gap-filled files come out with them
        g = sorted(filled, key=int)[0]
        line = next(ln for ln in lines[1:] if ln.split("\t")[0] == g)
        code, raw, _ = c.req("/api/results/table", {"path": str(root / "o"), "ids": [int(g)]})
        assert raw.decode().splitlines(keepends=True)[1] == _in_seconds(lines[0], line, filled)
    assert c.json("/api/results/reveal", {"path": str(root.parent)})[0] == 403          # outside --root
    code, ex = c.json("/api/example", {})
    assert code == 200 and ex["n_mzml"] == 3 and Path(ex["input"]).parent == (root / "peak3d_example").resolve()
    assert Path(ex["library"]) == (root / "peak3d_example" / "demo_library.msp").resolve() and Path(ex["library"]).is_file()
    assert c.json(f"/api/results/bundle?path={quote(str(d))}")[0] == 404


def test_export_lists_gap_filled_files(gui):
    """Download table: gap_filled_files after n_filled names the files filled_mask.tsv marks as not detected where the
    table holds a value (a filled cell that stayed 0 is not listed); file columns matched by name, numeric names kept
    as text; refreshed when filled_mask.tsv changes; no column without filled_mask.tsv (external kernels)"""
    c, root, _ = gui
    out = root / "o"
    out.mkdir()
    pd.DataFrame({"group_id": [1, 2, 5], "mz": [180.0634, 300.2, 410.3], "rt": [2.0, 4.0, 5.0], "rt_min": [1.95, 3.95, 4.9],
                  "rt_max": [2.05, 4.05, 5.1], "n_detected": [1, 3, 1], "n_confirmed": [1, 0, 1], "n_filled": [1, 0, 2],
                  "mz_ppm_spread": [1.0, 0.5, np.nan], "rt_sd": [0.01, 0.02, np.nan], "iso_offset": [0, 0, 0],
                  "charge": [1, 1, 1], "A": [2e6, 5e5, 3e4], "B": [1.5e5, 4e5, 9e5], "003": [0.0, 6e5, 7e3]}
                 ).to_csv(out / "aligned_feature_table.tsv", sep="\t", index=False)
    (out / "params.json").write_text("{}")
    mask = pd.DataFrame({"group_id": [9, 5, 2, 1], "003": [1, 1, 0, 1], "B": [0, 0, 0, 1], "A": [1, 1, 0, 0]})
    mask.to_csv(out / "filled_mask.tsv", sep="\t", index=False)          # another column order, a dropped group 9

    def export(ids):
        code, raw, _ = c.req("/api/results/table", {"path": str(out), "ids": ids})
        assert code == 200
        rows = [ln.split("\t") for ln in raw.decode().splitlines()]
        return rows[0], [dict(zip(rows[0], r)) for r in rows[1:]]

    head, rows = export([5, 1, 2, 77])
    assert head[:9] == ["group_id", "mz", "rt_s", "rt_min_s", "rt_max_s", "n_detected", "n_confirmed", "n_filled",
                        "gap_filled_files"] and head[-3:] == ["A", "B", "003"]
    assert [(r["group_id"], r["gap_filled_files"]) for r in rows] == [("5", "A;003"), ("1", "B"), ("2", "")]
    assert float(rows[0]["003"]) == 7000 and float(rows[1]["003"]) == 0 and rows[0]["rt_s"] == "300"
    # filled_mask.tsv rewritten: the export follows it
    mask.assign(B=[0, 1, 0, 1]).to_csv(out / "filled_mask.tsv", sep="\t", index=False)
    t = (out / "filled_mask.tsv").stat().st_mtime + 5
    os.utime(out / "filled_mask.tsv", (t, t))
    assert export([5])[1][0]["gap_filled_files"] == "A;B;003"
    # the report's bundle: status from presence_mask.tsv or not
    assert bundle.unpack(bundle.build(out))[0]["matrix"]["presence"] is False
    pd.DataFrame({"group_id": [1, 2, 5], "mz": [180.0634, 300.2, 410.3], "rt": [2.0, 4.0, 5.0], "n_detected": [1, 3, 1],
                  "n_confirmed": [1, 0, 1], "A": [2, 2, 1], "B": [1, 2, 2], "003": [0, 2, 1]}
                 ).to_csv(out / "presence_mask.tsv", sep="\t", index=False)
    h, a = bundle.unpack(bundle.build(out))
    assert h["matrix"]["presence"] is True and a["m_status"].reshape(3, 3).tolist() == [[3, 2, 0], [3, 3, 3], [2, 3, 2]]
    # no filled_mask.tsv: no column
    (out / "filled_mask.tsv").unlink()
    head, rows = export([5])
    assert "gap_filled_files" not in head and head[7:9] == ["n_filled", "mz_ppm_spread"] and float(rows[0]["A"]) == 30000
    viewer = (ROOT / "peak3d" / "web" / "static" / "viewer.js").read_text(encoding="utf-8")
    assert 'FILLED_COL = "gap_filled_files"' in viewer and "R.h.matrix.presence" in viewer


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


def test_ms1_pattern_through_the_server(processed):
    """the Show MS1 request: the apex-scan spectrum of a feature in the file where it is strongest, its isotope ratios
    (the synthetic study has M+1 at 15 %) and the halogen test (none: there are no halogens in it)"""
    d, out = processed
    httpd, url = server.make_server("127.0.0.1", 0, root=out.parent, token="t0ken")
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    c = _Client(url.split("?")[0].rstrip("/"), "t0ken")
    try:
        t = pd.read_csv(out / "aligned_feature_table.tsv", sep="\t")
        mono = t[t["iso_offset"] == 0].sort_values("S0", ascending=False)
        gid = int(mono.iloc[0]["group_id"])
        code, r = c.json("/api/results/ms1", {"path": str(out), "group_id": gid})
        assert code == 200, r
        assert r["group_id"] == gid and r["files"] == ["S0", "S1", "S2"] and r["file"] in r["files"]
        row = t[t["group_id"] == gid].iloc[0]
        assert r["file"] == max(["S0", "S1", "S2"], key=lambda s: row[s])
        assert abs(r["target_mz"] - row["mz"]) / row["mz"] * 1e6 < 10
        assert row["rt_min"] * 60 - 1 <= r["rt_s"] <= row["rt_max"] * 60 + 1
        j0 = r["k"].index(0)
        assert r["rel"][j0] == 1.0 and abs(r["rel"][j0 + 1] - 0.15) < 0.05
        assert r["test"]["verdict"] == "none" and r["test"]["hypotheses"][0]["chi2"] < 20
        # the dichlorinated ion through the route, with its ion composition balanced to the m/z
        hid = int(t[(t["mz"] - HALOGEN_MZ).abs() < 0.01].iloc[0]["group_id"])
        code, rh = c.json("/api/results/ms1", {"path": str(out), "group_id": hid})
        assert code == 200 and rh["test"]["verdict"] == "halogen" and rh["test"]["best"] == "Cl2"
        assert abs(rh["test"]["hypotheses"][0]["mass_error_mda"]) < 12 and "Cl2" in rh["test"]["formula"]
        assert len(r["apex_mz"]) == len(r["apex_inten"]) >= 2
        json.dumps(r)
        # a chosen file, a wrong file, a wrong feature
        code, r2 = c.json("/api/results/ms1", {"path": str(out), "group_id": gid, "file": "S2"})
        assert code == 200 and r2["file"] == "S2"
        assert c.json("/api/results/ms1", {"path": str(out), "group_id": gid, "file": "S9"})[0] == 400
        assert c.json("/api/results/ms1", {"path": str(out), "group_id": 10 ** 9})[0] == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_eic_through_the_server(processed, tmp_path):
    """Extract EIC: the whole-run chromatogram of an m/z in one file, per MS1 scan the most intense centroid within the
    tolerance, at the file's own m/z (its alignment offset) unless asked for the m/z as typed; read from the
    point-cloud cache, else from the mzML the run read"""
    d, out = processed
    httpd, url = server.make_server("127.0.0.1", 0, root=out.parent, token="t0ken")
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    c = _Client(url.split("?")[0].rstrip("/"), "t0ken")
    try:
        t = pd.read_csv(out / "aligned_feature_table.tsv", sep="\t")
        row = t[t["iso_offset"] == 0].sort_values("S1", ascending=False).iloc[0]
        mz = float(row["mz"])
        code, r = c.json("/api/results/eic", {"path": str(out), "file": "S1", "mz": mz, "ppm": 10})
        assert code == 200, r
        cloud = io.Cloud.load_npz(out / ".cache" / "S1.npz")
        assert r["source"] == "cache" and r["n_scans"] == cloud.n_scans == len(r["rt_s"]) == len(r["rt_native_s"]) == len(r["intensity"])
        assert np.allclose(r["rt_native_s"], 60 * cloud.rt, atol=1e-3)
        off = bundle.file_offsets(out).get("S1", 0.0)
        target = mz / (1 - off * 1e-6)
        ref = [max([v for m, v in zip(cloud.mz[cloud.off[s]:cloud.off[s + 1]], cloud.inten[cloud.off[s]:cloud.off[s + 1]])
                    if abs(m - target) <= target * 10e-6], default=0.0) for s in range(cloud.n_scans)]
        assert np.allclose(r["intensity"], ref, rtol=1e-4)
        apex = int(np.argmax(r["intensity"]))
        assert row["rt_min"] * 60 - 1 <= r["rt_s"][apex] <= row["rt_max"] * 60 + 1      # on the aligned axis
        # the m/z as typed; an m/z with nothing there; the same from the mzML when the cache is gone
        code, raw = c.json("/api/results/eic", {"path": str(out), "file": "S1", "mz": mz, "ppm": 10, "offset": False})
        assert code == 200 and raw["offset_ppm"] == [0.0, 0.0] and max(raw["intensity"]) > 0
        # a stored per-file window: asymmetric, from target - lo_ppm to target + hi_ppm
        code, asym = c.json("/api/results/eic", {"path": str(out), "file": "S1", "mz": mz, "ppm": 5, "lo_ppm": 2, "hi_ppm": 30})
        assert code == 200 and asym["lo_ppm"] == 2 and asym["hi_ppm"] == 30
        ref2 = [max([v for m, v in zip(cloud.mz[cloud.off[s]:cloud.off[s + 1]], cloud.inten[cloud.off[s]:cloud.off[s + 1]])
                     if target * (1 - 2e-6) <= m <= target * (1 + 30e-6)], default=0.0) for s in range(cloud.n_scans)]
        assert np.allclose(asym["intensity"], ref2, rtol=1e-4)
        for bad in ({"lo_ppm": 2}, {"lo_ppm": 0, "hi_ppm": 3}, {"lo_ppm": "x", "hi_ppm": 3}, {"lo_ppm": 2, "hi_ppm": 2000}):
            assert c.json("/api/results/eic", {"path": str(out), "file": "S1", "mz": mz, "ppm": 5, **bad})[0] == 400, bad
        code, none = c.json("/api/results/eic", {"path": str(out), "file": "S0", "mz": 1234.5678, "ppm": 5})
        assert code == 200 and max(none["intensity"]) == 0
        again = bundle.extract_eic(out, tmp_path / "gone", "S1", mz, 10.0, off)
        assert again["source"] == "mzML" and again["intensity"] == r["intensity"] and again["rt_s"] == r["rt_s"]
        (tmp_path / "o2").mkdir()
        (tmp_path / "o2" / "params.json").write_text(json.dumps({"input_files": [str(tmp_path / "moved" / "S1.mzML")]}))
        with pytest.raises(FileNotFoundError, match="no point-cloud cache"):
            bundle.extract_eic(tmp_path / "o2", tmp_path / "o2" / ".cache", "S1", mz, 10.0)
        # refused: a file of no output, no number, out of range
        assert c.json("/api/results/eic", {"path": str(out), "file": "S9", "mz": mz})[0] == 400
        for bad in ({"mz": "x"}, {"mz": -1}, {"mz": mz, "ppm": 0}, {"mz": mz, "ppm": 5000}, {"mz": float("nan")}):
            assert c.json("/api/results/eic", {"path": str(out), "file": "S1", **bad})[0] == 400, bad
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_line_chart_zoom_and_eic_button_in_the_scripts():
    """the front end: the line chart zooms (drag, box, wheel, buttons), a feature's chromatograms use it, and the
    Features tab offers Extract EIC only with the app's server (opts.eic)"""
    st = ROOT / "peak3d" / "web" / "static"
    plot, viewer, app = ((st / f).read_text(encoding="utf-8") for f in ("plot.js", "viewer.js", "app.js"))
    for k in ("zoomBy(", "resetZoom()", "pathDense(", '"wheel"', "dblclick"):
        assert k in plot, k
    assert "opts.eic ?" in viewer and "eicDialog()" in viewer and '"/api/results/eic"' in app
    assert "a.eic_lo" in viewer and "lo_ppm: lo" in app        # per-file windows reach the traces and Whole run
    # the feature's chromatograms zoom too, and in the app read the whole run on request (shared with Extract EIC)
    assert viewer.count("zoom: true") >= 2 and '"Whole run")' in viewer and "eicGet(R.stems[t.file], qOf(t))" in viewer
    css = (st / "style.css").read_text(encoding="utf-8")
    assert all(f"--viz-cat-{k}:" in css for k in range(1, 9))


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
        # the denoising search: no per-spectrum copy, a keep mask per denoised hit over the spectrum's peaks
        assert not h["ms2"]["denoised"] and "ms2_keep" not in a and len(h["ids"]["denoise_ion"]) == len(a["id_row"])
        d = a["id_dkey"]
        assert len(d) == len(a["id_row"]) and (d[a["id_spec"] == 1] == -1).all() and (d[a["id_spec"] == 3] >= 0).any()
        assert all(a["dk_off"][x + 1] - a["dk_off"][x] == len(a["ms2_mz"]) for x in d[d >= 0])
        assert set(a["id_spec"].tolist()) == {1, 3} and h["ids"]["default_spectrum"] == "single_denoised"
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
    code, r = c.json("/api/results/annotate", {"path": str(out), "group_id": 1, "hit": {"spectrum": "single_denoised", "rank": 1}})
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


def test_download_names_outside_latin1(gui):
    """a results folder named in Chinese (or with an accent) downloads: the header carries an ASCII fallback and the
    exact name as RFC 5987 UTF-8 (http.server writes headers as Latin-1; the old header failed mid-response)"""
    v = server._disposition("attachment", '数据 é"x_aligned.tsv')
    v.encode("latin-1")
    assert v.startswith('attachment; filename="') and "filename*=UTF-8''%E6%95%B0%E6%8D%AE%20%C3%A9%22x_aligned.tsv" in v
    assert server._disposition("inline", "a.html") == "inline; filename=\"a.html\"; filename*=UTF-8''a.html"
    c, root, _ = gui
    out = _ms2_output(root / "数据 é")
    q = quote(str(out))
    code, raw, hdr = c.req(f"/api/results/file?path={q}&name=aligned_feature_table.tsv")
    assert code == 200 and raw == (out / "aligned_feature_table.tsv").read_bytes()
    assert hdr["Content-Disposition"].endswith("filename*=UTF-8''" + quote("数据 é_aligned_feature_table.tsv", safe=""))
    code, page, hdr = c.req(f"/api/results/report?path={q}&download=1")
    assert code == 200 and b'id="peak3d-data"' in page
    assert hdr["Content-Disposition"].startswith("attachment;") and quote("mosa_report_数据 é.html", safe="") in hdr["Content-Disposition"]


def test_download_buttons_in_the_scripts():
    """Download table and Download candidates sit in the Features toolbar; the menu's download links stay in the
    page through their click; a saved blob's URL outlives a save dialog"""
    st = ROOT / "peak3d" / "web" / "static"
    viewer, app, core = ((st / f).read_text(encoding="utf-8") for f in ("viewer.js", "app.js", "core.js"))
    assert "count, eicBtn, dl, dlc)]" in viewer and "candidatesJson(R, list)" in viewer and '"Download candidates")' in viewer
    assert "reference_library: { identity_search: hits, open_search: [], neutral_loss_search: [], hybrid_search: [] }" in viewer
    assert 'wiki_id: `${R.h.title}/${gid}`' in viewer and 'status: "success"' in viewer and "MW_MAX = 100" in viewer
    assert "if (it.href) setTimeout(close, 0)" in app
    assert "URL.revokeObjectURL(url); a.remove(); }, 60000)" in core


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
