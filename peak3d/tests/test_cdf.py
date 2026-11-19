"""ANDI-MS netCDF input (peak3d/cdf.py): unit-resolution files become a point cloud of nominal masses on the 13C
scale, accurate-mass files are read as they are, a folder of .cdf files runs as it is, and the GC stages report a
unit-resolution run at nominal masses."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peak3d import cdf as C
from peak3d import io
from peak3d.gc import library as L
from peak3d.gc.synth import Compound, alkane, make_run, random_spectrum
from peak3d.rawfiles import input_files, no_inputs_message
from peak3d.web import bundle, gcview, server

ROOT = Path(__file__).resolve().parents[2]
DT = 1.0 / 300.0
SCALE = {"q_a": 1.0, "q_b": 0.5, "q_c": 2.0}


def write_cdf(path, rt_s, scans, polarity=b"Positive Polarity"):
    """an ANDI-MS file: scans = list of (m/z array, intensity array), rt in seconds"""
    from scipy.io import netcdf_file
    mz = np.concatenate([m for m, _ in scans])
    it = np.concatenate([i for _, i in scans])
    idx = np.concatenate([[0], np.cumsum([len(m) for m, _ in scans])[:-1]])
    with netcdf_file(str(path), "w") as nc:
        nc.test_ionization_polarity = polarity
        nc.createDimension("scan_number", len(rt_s))
        nc.createDimension("point_number", len(mz))
        for name, dim, typ, val in (("scan_acquisition_time", "scan_number", "d", rt_s), ("scan_index", "scan_number", "i", idx),
                                    ("mass_values", "point_number", "d", mz), ("intensity_values", "point_number", "d", it)):
            v = nc.createVariable(name, typ, (dim,))
            v[:] = val


def as_unit_scans(cloud, rng, noise=0.0):
    """the scans of an accurate-mass cloud as a quadrupole export writes them: one value per nominal mass, at the
    nominal mass plus a jitter on a 0.1 Da grid"""
    scans = []
    for s in range(cloud.n_scans):
        m, i = cloud.mz[cloud.off[s]:cloud.off[s + 1]], cloud.inten[cloud.off[s]:cloud.off[s + 1]].astype(np.float64)
        nom = np.floor(m + C.BIN_SHIFT).astype(np.int64)
        u, inv = np.unique(nom, return_inverse=True)
        v = np.zeros(len(u))
        np.add.at(v, inv, i)
        if noise:
            v = v + rng.normal(0, noise, len(u))
        scans.append((u + rng.integers(-2, 4, len(u)) / 10.0, v))
    return scans


def test_unit_resolution_is_read_from_the_masses():
    rng = np.random.default_rng(0)
    nominal = rng.integers(35, 600, 5000)
    assert C.is_unit_resolution(nominal + rng.integers(-2, 4, 5000) / 10.0)              # 0.1 grid (Agilent)
    assert C.is_unit_resolution((nominal + rng.integers(-4, 8, 5000) / 20.0).astype(np.float32))    # 0.05 grid, float32
    assert not C.is_unit_resolution(nominal * (1.0 + rng.uniform(2e-4, 9e-4, 5000)))     # accurate masses
    assert not C.is_unit_resolution(np.zeros(0))


def test_unit_resolution_file_becomes_nominal_masses_on_the_13c_scale(tmp_path):
    rng = np.random.default_rng(1)
    rt = 60.0 + 0.5 * np.arange(40)
    scans = []
    for s in range(40):
        g = 1e4 * np.exp(-0.5 * ((s - 20) / 3.0) ** 2)
        m = np.array([72.8, 73.1, 73.65, 73.7, 147.2, 207.0, 300.1])     # 72.8, 73.1 and 73.65 are mass 73; 73.7 is 74
        scans.append((m, np.array([g, 2 * g, g, 0.5 * g, g, -5.0 + rng.normal(0, 30), g]) + 1.0))
    write_cdf(tmp_path / "q.CDF", rt, scans)
    cloud = io.load_cloud(tmp_path / "q.CDF")
    assert cloud.name == "q" and cloud.polarity == "pos" and cloud.n_scans == 40
    assert cloud.mass_scale == pytest.approx(io.C13) and cloud.source["kind"] == "unit resolution"
    assert cloud.source["scan_skew"] == 0.0          # one peak: no skew measured on it
    assert cloud.rt[0] == pytest.approx(1.0)         # minutes
    masses = set(C.unit_mass(cloud.mz).tolist())
    assert {73, 74, 147, 300} <= masses <= {73, 74, 147, 207, 300}
    assert np.allclose(cloud.mz / io.C13, np.rint(cloud.mz / io.C13))
    a, b = cloud.off[20], cloud.off[21]
    apex = dict(zip(C.unit_mass(cloud.mz[a:b]).tolist(), cloud.inten[a:b].tolist()))
    assert apex[73] == pytest.approx(4e4 + 3.0) and apex[74] == pytest.approx(0.5e4 + 1.0)      # one bin: summed
    assert cloud.source["noise_floor"] > 0 and 207 not in apex                                   # the noise channel is dropped


def test_accurate_mass_file_is_read_as_it_is(tmp_path):
    rt = 30.0 + 0.2 * np.arange(10)
    m = np.array([147.0659, 73.0468, 207.0329])                  # not sorted in the file
    scans = [(m, np.array([3.0, 5.0, 0.0]) * (s + 1)) for s in range(10)]
    write_cdf(tmp_path / "t.cdf", rt, scans, polarity=b"Negative Polarity")
    cloud = io.load_cloud(tmp_path / "t.cdf")
    assert cloud.mass_scale == 1.0 and cloud.source["kind"] == "accurate mass" and cloud.polarity == "neg"
    assert cloud.n_scans == 10 and cloud.n_points == 20          # the zero-intensity point is dropped
    assert np.allclose(cloud.mz[:2], [73.0468, 147.0659])
    bad = tmp_path / "x.cdf"
    from scipy.io import netcdf_file
    with netcdf_file(str(bad), "w") as nc:
        nc.createDimension("n", 1)
    with pytest.raises(ValueError, match="not an ANDI-MS file"):
        io.load_cloud(bad)


def test_a_folder_offers_its_cdf_files(tmp_path):
    for n in ("a.CDF", "b.cdf", "b.mzML", "c.raw", "c.CDF", "d.raw", "notes.txt"):
        (tmp_path / n).write_bytes(b"x")
    assert [p.name for p in input_files(tmp_path)] == ["a.CDF", "b.mzML", "c.CDF", "d.raw"]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert ".cdf" in no_inputs_message(empty)


def _run(*argv):
    return subprocess.run([sys.executable, "-m", "peak3d", *argv], cwd=ROOT, capture_output=True, text=True)


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    """three unit-resolution runs (.CDF) of 12 compounds and a C10-C16 alkane ladder, processed in GC-MS mode with
    a unit-mass library of six of the compounds"""
    root = tmp_path_factory.mktemp("cdf")
    rng = np.random.default_rng(5)
    comps = [alkane(n, 0.6 + 0.6 * (n - 10), 2e5) for n in range(10, 17)]
    known = []
    for k in range(12):
        m, r = random_spectrum(rng, 20)
        known.append(Compound(f"compound {k:02d}", 0.8 + 0.29 * k, 10 ** rng.uniform(4.5, 5.3), m, r))
    comps += known
    data = root / "cdf"
    data.mkdir()
    for i, stem in enumerate(SCALE):
        run = [Compound(c.name, c.rt, c.height * SCALE[stem], c.mz, c.rel, c.fwhm) for c in comps]
        cloud, _ = make_run(run, n_scans=1500, dt=DT, seed=31 + i, name=stem)
        write_cdf(data / f"{stem}.CDF", 60.0 * cloud.rt, as_unit_scans(cloud, rng))
    lib = root / "ei.msp"
    with open(lib, "w") as fh:
        for c in known[:6]:
            nm = L.nominal_mass(c.mz)
            fh.write(f"Name: {c.name}\nRetention_index: {1000.0 + 100.0 * (c.rt - 0.6) / 0.6:.1f}\nNum Peaks: {len(nm)}\n")
            fh.write("".join(f"{a} {999 * b:.1f}\n" for a, b in zip(nm, c.rel)) + "\n")
        c = known[6]                                 # one record with accurate masses: the library as a whole is not unit-mass
        fh.write(f"Name: {c.name}\nNum Peaks: {len(c.mz)}\n" + "".join(f"{a:.4f} {999 * b:.1f}\n" for a, b in zip(c.mz, c.rel)) + "\n")
    out = root / "out"
    r = _run("process", "--input", str(data), "--output", str(out), "--mode", "gc", "--ri-ladder", "alkane",
             "--ei-library", str(lib))
    assert r.returncode == 0, r.stderr[-2000:]
    return dict(out=out, known=known, log=r.stdout, data=data)


def test_cdf_folder_runs_and_compounds_are_at_nominal_masses(study):
    out = study["out"]
    assert "gc stages failed" not in study["log"] and "unit-resolution runs" in study["log"]
    p = json.loads((out / "params.json").read_text())
    assert p["mass_scale"] == pytest.approx(io.C13) and [Path(f).suffix for f in p["input_files"]] == [".CDF"] * 3
    fp = json.loads((out / "features" / "q_a.params.json").read_text())
    assert fp["mass_scale"] == pytest.approx(io.C13) and fp["source"]["kind"] == "unit resolution"
    comp = pd.read_csv(out / "gc" / "compounds.tsv", sep="\t")
    assert len(comp) >= 12 and set(comp["n_files"]) == {3} and comp["ri"].notna().all()
    for c in ("quant_mz", "base_mz"):
        assert np.all(comp[c] == np.rint(comp[c]))
    # every compound of the library is named, the ones with ions above m/z 209 included (where the scaled masses
    # leave the unit-mass bin), and the accurate-mass record too
    named = {str(n) for n in comp["name"].fillna("") if str(n).startswith("compound ")}
    assert named == {c.name for c in study["known"][:7]}
    assert max(float(c.mz.max()) for c in study["known"][:6]) > 300
    best = comp[comp["name"].fillna("").astype(str).str.startswith("compound ")]["match"]
    assert best.min() > 0.8
    for c in study["known"][:6]:                      # the quantifier is one of the compound's own nominal masses
        row = comp[comp["name"] == c.name].iloc[0]
        assert int(row["quant_mz"]) in set(L.nominal_mass(c.mz).tolist()) | set((L.nominal_mass(c.mz) + 1).tolist())
    with np.load(out / "gc" / "components" / "q_a.npz") as z:
        assert np.all(z["mz"] == np.rint(z["mz"]))
    assert "Num Peaks" in (out / "gc" / "spectra.msp").read_text()
    # the single ions keep the scaled m/z
    t = pd.read_csv(out / "aligned_feature_table.tsv", sep="\t")
    assert np.allclose(t["mz"] / io.C13, np.rint(t["mz"] / io.C13), atol=2e-3)


def test_viewer_block_and_chromatograms_of_a_unit_resolution_run(study):
    out = study["out"]
    hd, a = bundle.unpack(gcview.build(out))
    assert hd["digits"] == 0 and hd["summary"]["unit_mass"] is True
    assert np.all(a["s_mz"] == np.rint(a["s_mz"])) and np.all(a["c_qmz"] == np.rint(a["c_qmz"]))
    k = int(a["c_id"][int(np.argmax(a["c_height"]))])
    t = gcview.trace(out, out / ".cache", k)
    q = t["ions"][0]
    assert q["quant"] and q["mz"] == round(q["mz"])
    x, y = np.array(t["rt_s"]), np.array(q["y"])
    assert y.max() > 0 and abs(x[np.argmax(y)] - t["apex_s"]) < 1.0       # read from the cloud at the scaled m/z
    assert abs(y.max() - q["abundance"]) < 0.25 * y.max()
    assert all(max(i["y"]) > 0 for i in t["ions"])
    # the cache gone: the chromatograms come from the .CDF files the run read
    t2 = gcview.trace(out, out / "no_cache_here", k)
    assert t2["source"] == "CDF" and np.allclose(t2["ions"][0]["y"], q["y"], rtol=1e-4)


def test_other_kernels_refuse_cdf(study, tmp_path):
    r = _run("external", "--kernel", "asari", "--input", str(study["data"]), "--output", str(tmp_path / "o"))
    assert r.returncode != 0 and ("netCDF" in r.stderr or "not available" in r.stderr)
    httpd, url = server.make_server("127.0.0.1", 0, root=study["data"].parent, token=None)
    try:
        ls = httpd.RequestHandlerClass._ls(type("H", (), {"app": httpd.app})(), str(study["data"]))
        assert ls["n_cdf"] == 3 and ls["n_mzml"] == 0 and ls["n_inputs"] == 3 and ls["n_raw"] == 0
    finally:
        httpd.server_close()
