"""GC-MS in the app and the report: `process --mode gc` runs the GC stages, the compounds reach the viewer as a
block of their own (gcview), the server serves it with the chromatograms of a compound's ions, and report.html
carries it."""
import gzip
import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peak3d.gc import library as L
from peak3d.gc.synth import Compound, alkane, make_run, random_spectrum
from peak3d.synth import write_mzml
from peak3d.web import bundle, gcview, report, server

ROOT = Path(__file__).resolve().parents[2]
DT = 1.0 / 300.0
SCALE = {"gc_a": 1.0, "gc_b": 0.5, "gc_c": 2.0}
STATIC = ROOT / "peak3d" / "web" / "static"


def _run(*argv):
    return subprocess.run([sys.executable, "-m", "peak3d", *argv], cwd=ROOT, capture_output=True, text=True)


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    """three runs of 12 compounds and a C10-C16 alkane ladder, processed in GC-MS mode in one command, with the
    ladder found and a unit-mass library of six of the compounds searched"""
    root = tmp_path_factory.mktemp("gcview")
    rng = np.random.default_rng(5)
    comps = [alkane(n, 0.6 + 0.6 * (n - 10), 2e5) for n in range(10, 17)]
    known = []
    for k in range(12):
        m, r = random_spectrum(rng, 20)
        known.append(Compound(f"compound {k:02d}", 0.8 + 0.29 * k, 10 ** rng.uniform(4.5, 5.3), m, r))
    comps += known
    data = root / "mzml"
    data.mkdir()
    for i, stem in enumerate(SCALE):
        run = [Compound(c.name, c.rt, c.height * SCALE[stem], c.mz, c.rel, c.fwhm) for c in comps]
        cloud, _ = make_run(run, n_scans=1500, dt=DT, seed=31 + i, name=stem)
        write_mzml(cloud, data / f"{stem}.mzML")
    lib = root / "ei.msp"
    with open(lib, "w") as fh:
        for c in known[:6]:
            nm = L.nominal_mass(c.mz)
            fh.write(f"Name: {c.name}\nRetention_index: {1000.0 + 100.0 * (c.rt - 0.6) / 0.6:.1f}\nNum Peaks: {len(nm)}\n")
            fh.write("".join(f"{a} {999 * b:.1f}\n" for a, b in zip(nm, c.rel)) + "\n")
    out = root / "out"
    r = _run("process", "--input", str(data), "--output", str(out), "--mode", "gc", "--ri-ladder", "alkane",
             "--ei-library", str(lib))
    assert r.returncode == 0, r.stderr[-2000:]
    return dict(out=out, known=known, lib=lib, log=r.stdout)


def test_process_in_gc_mode_runs_the_gc_stages(study):
    out = study["out"]
    assert "[peak3d] gc: deconvolving" in study["log"] and "gc stages failed" not in study["log"]
    comp = pd.read_csv(out / "gc" / "compounds.tsv", sep="\t")
    assert len(comp) >= 12 and set(comp["n_files"]) == {3}
    assert comp["ri"].notna().all()                                 # the ladder was found
    assert sum(str(n).startswith("compound ") for n in comp["name"].fillna("")) == 6      # the library's six
    assert (out / "gc" / "hit_spectra.npz").exists()
    assert json.loads((out / "params.json").read_text())["mode"] == "gc"
    # an LC-mode run of the same command line has no gc/ (the GC options are then not used)
    assert server.validate_argv(["process", "--input", "x", "--output", "y", "--ri-ladder", "fame"]) is None


def test_block_holds_compounds_spectra_values_and_hits(study):
    out = study["out"]
    comp = pd.read_csv(out / "gc" / "compounds.tsv", sep="\t")
    tab = pd.read_csv(out / "gc" / "compound_table.tsv", sep="\t")
    hd, a = bundle.unpack(gcview.build(out))
    n = len(comp)
    assert hd["kind"] == "peak3d-gc" and hd["n"] == n and hd["stems"] == list(SCALE) and hd["digits"] == 4
    assert hd["summary"]["n_compounds"] == n and hd["summary"]["ladder"] == "alkane" and hd["summary"]["libraries"] == ["ei.msp"]
    assert hd["summary"]["picked_in"] == "gc"
    np.testing.assert_allclose(a["c_rt"], comp["rt"].values)
    np.testing.assert_allclose(a["c_qmz"], comp["quant_mz"].values)
    assert list(a["c_nfiles"]) == list(comp["n_files"]) and len(a["s_off"]) == n + 1 and a["s_off"][-1] == len(a["s_mz"])
    for k in range(n):                       # every spectrum: ascending m/z, as many ions as listed, the quantifier in it
        m = a["s_mz"][a["s_off"][k]:a["s_off"][k + 1]]
        assert len(m) == comp["n_ions"][k] and np.all(np.diff(m) > 0)
        assert np.min(np.abs(m - comp["quant_mz"][k])) < 1e-5            # the tables hold m/z to 1e-5
        ab = a["s_ab"][a["s_off"][k]:a["s_off"][k + 1]]
        assert abs(m[np.argmax(ab)] - comp["base_mz"][k]) < 1e-5
    np.testing.assert_allclose(a["q"].reshape(n, 3), tab[list(SCALE)].to_numpy(np.float32), rtol=1e-6)
    # the annotated compounds carry their name; their best hit has its library spectrum
    named = [k for k in range(n) if hd["name"][k]]
    assert len(named) == 6
    for k in named:
        j = a["h_off"][k]
        assert a["h_off"][k + 1] > j and a["h_rank"][j] == 1 and hd["hits"]["name"][j] == hd["name"][k]
        assert abs(a["h_match"][j] - a["c_match"][k]) < 1e-6 and a["h_spec"][j] >= 0
        s = a["h_spec"][j]
        lm = a["hs_mz"][a["hs_off"][s]:a["hs_off"][s + 1]]
        c = next(c for c in study["known"] if c.name == hd["name"][k])
        assert set(np.round(lm).astype(int).tolist()) == set(L.nominal_mass(c.mz).tolist())
    assert bundle.unpack(gzip.decompress(gcview.blob(out)))[0]["n"] == n


def test_trace_reads_the_ions_of_a_compound(study):
    out = study["out"]
    comp = pd.read_csv(out / "gc" / "compounds.tsv", sep="\t")
    k = int(comp["compound_id"][np.argmax(comp["height"].values)])
    row = comp[comp["compound_id"] == k].iloc[0]
    t = gcview.trace(out, out / ".cache", k)
    assert t["file"] == row["reference_file"] and sorted(t["files"]) == sorted(SCALE)
    assert t["ions"][0]["quant"] and abs(t["ions"][0]["mz"] - row["quant_mz"]) < 1e-5
    assert 1 < len(t["ions"]) <= gcview.TRACE_IONS and sum(i["quant"] for i in t["ions"]) == 1
    assert all(len(i["y"]) == len(t["rt_s"]) for i in t["ions"])
    assert t["rt_min_s"] < t["apex_s"] < t["rt_max_s"] and t["rt_s"][0] < t["rt_min_s"] and t["rt_s"][-1] > t["rt_max_s"]
    x, y = np.array(t["rt_s"]), np.array(t["ions"][0]["y"])
    assert abs(x[np.argmax(y)] - t["apex_s"]) < 1.0                      # the quantifier peaks at the component's apex
    assert abs(y.max() - t["ions"][0]["abundance"]) < 0.25 * y.max()    # ... at about its abundance
    m = t["model"]
    assert len(m["y"]) == len(m["rt_s"]) and abs(max(m["y"]) - 1.0) < 1e-6
    assert m["rt_s"][0] == pytest.approx(t["rt_min_s"]) and m["rt_s"][-1] == pytest.approx(t["rt_max_s"])
    other = next(s for s in SCALE if s != t["file"])
    t2 = gcview.trace(out, out / ".cache", k, other)
    assert t2["file"] == other and t2["ions"][0]["quant"]
    with pytest.raises(KeyError):
        gcview.trace(out, out / ".cache", 10 ** 6)
    with pytest.raises(KeyError):
        gcview.trace(out, out / ".cache", k, "no_such_file")


def test_report_carries_the_compounds(study, tmp_path):
    out = study["out"]
    rep = out / "report.html"
    gz = report.read_gc(rep)
    hd, a = bundle.unpack(gzip.decompress(gz))
    assert hd["kind"] == "peak3d-gc" and hd["n"] == len(pd.read_csv(out / "gc" / "compounds.tsv", sep="\t"))
    main = report.read_bundle(rep)
    assert report.refresh_ids(out) and report.read_gc(rep) == gz and report.read_bundle(rep) == main     # kept by a search
    assert report.refresh_gc(out) and report.read_gc(rep) == gz and report.read_bundle(rep) == main
    text = rep.read_text(encoding="utf-8")
    assert 'id="peak3d-gc"' in text and "P3.gc = {" in text and "{{GC}}" not in text
    # an LC-MS output: no block, and the page mounts without one
    lc = report.render(main, "x")
    assert '<script id="peak3d-gc" type="application/octet-stream"></script>' in lc


def _get(url, token="t0ken", body=None):
    rq = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                headers={"X-Peak3d-Token": token, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(rq, timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_compounds_through_the_server(study):
    out = study["out"]
    lc = out.parent / "lc_only"
    lc.mkdir(exist_ok=True)
    (lc / "aligned_feature_table.tsv").write_text("group_id\tmz\trt\n")
    httpd, url = server.make_server("127.0.0.1", 0, root=out.parent, token="t0ken")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = url.split("?")[0].rstrip("/")
    q = urllib.request.quote(str(out))
    try:
        code, raw = _get(f"{base}/api/results/gc?path={q}")
        assert code == 200
        hd, a = bundle.unpack(gzip.decompress(raw))
        assert hd["kind"] == "peak3d-gc" and hd["n"] == len(a["c_id"]) >= 12
        code, again = _get(f"{base}/api/results/gc?path={q}")
        assert code == 200 and again == raw                              # kept until the GC files change
        code, raw = _get(f"{base}/api/results/gc?path={urllib.request.quote(str(lc))}")
        assert code == 404 and b"no GC results" in raw                   # an LC-MS output has none
        code, raw = _get(f"{base}/api/results/gcfile?path={q}&name=compounds.tsv")
        assert code == 200 and raw == (out / "gc" / "compounds.tsv").read_bytes()
        code, raw = _get(f"{base}/api/results/gcfile?path={q}&name=spectra.msp")
        assert code == 200 and raw.startswith(b"Name: compound 0")
        assert _get(f"{base}/api/results/gcfile?path={q}&name=../params.json")[0] == 404
        k = int(a["c_id"][0])
        code, raw = _get(f"{base}/api/results/gc/trace", body={"path": str(out), "compound_id": k})
        t = json.loads(raw)
        assert code == 200 and t["ions"][0]["quant"] and len(t["ions"][0]["y"]) == len(t["rt_s"])
        assert _get(f"{base}/api/results/gc/trace", body={"path": str(out), "compound_id": 10 ** 6})[0] == 404
        assert _get(f"{base}/api/results/gc/trace", body={"path": str(out), "compound_id": "x"})[0] == 400
        code, page = _get(f"{base}/")
        assert code == 200 and b"/static/gc.js" in page
        assert _get(f"{base}/static/gc.js")[0] == 200
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_gc_options_on_the_run_form():
    sch = {f["dest"]: f for f in server.option_schema()}
    assert sch["mode"]["kind"] == "choice" and sch["mode"]["choices"] == ["lc", "gc"] and sch["mode"]["default"] == "lc"
    assert sch["ri_ladder"]["choices"] == ["alkane", "fame"] and sch["ri_ladder"]["default"] is None
    assert sch["ei_library"]["kind"] == "list" and sch["ei_library"]["flag"] == "--ei-library"
    assert sch["ri_tol"]["kind"] == "float" and sch["ri_table"]["kind"] == "path"
    argv = server.options_to_argv({"mode": "gc", "ri_ladder": "alkane", "ri_carbons": "10-16", "ei_library": ["a.msp", "b.msp"],
                                   "ri_tol": 30, "ri_type": "semi", "ri_table": None}, list(sch.values()))
    assert sorted(argv) == sorted(["--mode", "gc", "--ri-ladder", "alkane", "--ri-carbons", "10-16", "--ei-library", "a.msp",
                                   "--ei-library", "b.msp", "--ri-tol", "30"])
    assert server.validate_argv(["process", "--input", "x", "--output", "y", *argv]) is None
    from peak3d.__main__ import build_parser
    args = build_parser().parse_args(["process", "--input", "x", "--output", "y", *argv])
    assert args.ei_library == [Path("a.msp"), Path("b.msp")] and args.library is None
    g = build_parser().parse_args(["gc", "out", "--library", "a.msp", "--ri-ladder", "fame", "--ri-tol", "5"])      # as before
    assert g.library == [Path("a.msp")] and g.ri_ladder == "fame" and g.ri_tol == 5 and g.ri_type == "semi"
    assert any(marker == "gc: " and stage == "gc" for marker, stage, _ in server.STAGES)


def test_compounds_tab_in_the_scripts():
    """the front end: gc.js is part of the page and of the report, the viewer shows its tab first for a GC-MS
    output, and the app fetches the compounds and their chromatograms"""
    gc, viewer, app, plot, page = ((STATIC / f).read_text(encoding="utf-8") for f in ("gc.js", "viewer.js", "app.js", "plot.js", "report.html"))
    assert "gc.js" in report.SCRIPTS and report.SCRIPTS.index("gc.js") < report.SCRIPTS.index("viewer.js")
    assert "P3.gc = { Compounds, render, table, msp }" in gc and '"Quantifier m/z"' in gc and "P3.plot.mirror" in gc
    assert 'G ? [["compounds", "science", "Compounds"]]' in viewer and 'G ? "compounds" : "overview"' in viewer
    assert "/api/results/gc?path=" in app and '"/api/results/gc/trace"' in app and "gcTrace" in app
    assert 'mode: [["lc", "LC-MS"], ["gc", "GC-MS (EI)"]]' in app and "GC_FIELDS" in app
    assert "precLabel" in plot and "mzDigits" in plot
    assert 'id="peak3d-gc"' in page and "P3.viewer.mount(el, bundle, { gc })" in page
