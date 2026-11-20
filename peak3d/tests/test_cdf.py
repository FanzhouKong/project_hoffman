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


def tof_unit_scans(rng, n_scans=400, masses=np.arange(85, 400), slide=0.08, jitter=3e-4, peak_every=40):
    """scans as a unit-resolution time-of-flight export writes them: one centroid per nominal mass, the centre of a
    background ion (mass defect 0.05 to 0.2 Da) and of fragments eluting every `peak_every` scans (`slide` Da above), so the
    written mass slides with their proportions; a few half-integer masses (doubly charged ions) in between"""
    scans = []
    s = np.arange(n_scans)
    phase = rng.integers(0, peak_every, len(masses))
    for k in range(n_scans):
        g = 2e5 * np.exp(-0.5 * (((k - phase) % peak_every - peak_every / 2) / 4.0) ** 2)      # the fragments
        b = np.full(len(masses), 2e4)                                                             # the background
        m = masses * 1.00047 + 0.013 + slide * g / (g + b) + rng.normal(0, jitter, len(masses))
        half = rng.random(len(masses)) < 0.04
        mm = np.concatenate([m, masses[half] + 0.55])
        ii = np.concatenate([g + b, np.full(int(half.sum()), 3e3)])
        o = np.argsort(mm)
        scans.append((mm[o], ii[o]))
    return 60.0 + 0.06 * s, scans


def flat(scans):
    off = np.concatenate([[0], np.cumsum([len(m) for m, _ in scans])])
    return np.concatenate([m for m, _ in scans]), off, np.concatenate([i for _, i in scans])


def test_one_sliding_centroid_per_nominal_mass_is_unit_resolution():
    rng = np.random.default_rng(3)
    _, scans = tof_unit_scans(rng)
    mz, off, it = flat(scans)
    assert C.near_share(mz, off) < 0.05 and C.slide_ratio(mz, it, off) > 3
    assert C.mass_kind(mz, off, it) == "one per mass" and C.is_unit_resolution(mz, off, it)
    assert not C.is_unit_resolution(mz)                         # the test needs the scans
    # a sparse accurate-mass file, one ion per nominal mass: its centroids scatter around their ions' masses
    _, steady = tof_unit_scans(rng, slide=0.0)
    mz, off, it = flat(steady)
    assert C.near_share(mz, off) < 0.05 and 0.7 < C.slide_ratio(mz, it, off) < 1.4
    assert C.mass_kind(mz, off, it) == "accurate"
    # accurate masses with several ions per nominal mass
    dense = [(np.sort(np.concatenate([m, m + 0.04, m + 0.11])), np.tile(i, 3)) for m, i in scans]
    mz, off, it = flat(dense)
    assert C.near_share(mz, off) > 0.5 and C.mass_kind(mz, off, it) == "accurate"
    # what the caller says goes
    assert C.is_unit_resolution(mz, off, it, masses="unit") and not C.is_unit_resolution(*flat(scans), masses="accurate")
    with pytest.raises(ValueError):
        C.is_unit_resolution(mz, off, it, masses="nominal")


def test_near_share_looks_inside_scans_only():
    mz = np.array([100.1, 101.1, 102.1, 100.2, 101.2, 101.4])       # scan 2 starts 1.9 Da below the end of scan 1
    off = np.array([0, 3, 6])
    assert C.near_share(mz, off) == pytest.approx(0.25)             # 101.2 / 101.4, of four neighbours
    assert C.near_share(mz[[2, 0, 1, 5, 3, 4]], off) == pytest.approx(0.25)     # unsorted scans
    assert np.isnan(C.near_share(np.array([100.0, 100.1]), np.array([0, 1, 2])))


def test_sliding_centroid_file_becomes_nominal_masses(tmp_path):
    rng = np.random.default_rng(4)
    rt, scans = tof_unit_scans(rng)
    write_cdf(tmp_path / "tof.cdf", rt, scans)
    assert io.load_cloud(tmp_path / "tof.cdf").source["masses"] == "one per mass"
    cloud = C.load_cloud(tmp_path / "tof.cdf", skew=0.0)        # as a run reads it: with the run's skew (none here)
    src = cloud.source
    assert src["kind"] == "unit resolution" and src["masses"] == "one per mass" and cloud.mass_scale == pytest.approx(io.C13)
    assert src["near_share"] < 0.05 and src["slide_ratio"] > 3
    assert np.allclose(cloud.mz / io.C13, np.rint(cloud.mz / io.C13))
    a, b = cloud.off[100], cloud.off[101]
    got = dict(zip(C.unit_mass(cloud.mz[a:b]).tolist(), cloud.inten[a:b].tolist()))
    m, i = scans[100]
    k = int(np.argmax(i))
    n = int(np.floor(m[k] + C.BIN_SHIFT))
    assert got[n] == pytest.approx(i[np.floor(m + C.BIN_SHIFT) == n].sum())      # the whole nominal mass, one value
    # told to read the masses as they are
    acc = C.load_cloud(tmp_path / "tof.cdf", masses="accurate")
    assert acc.source["kind"] == "accurate mass" and acc.mass_scale == 1.0 and acc.source["masses"].startswith("accurate (as told")
    assert acc.n_points == sum(len(m) for m, _ in scans)
    with pytest.raises(ValueError):
        C.load_cloud(tmp_path / "tof.cdf", masses="nominal")


def write_mzml_as_is(path, rt_s, scans, name=None):
    """the mzML a converter writes from an ANDI-MS file: the same centroids"""
    from peak3d.synth import write_mzml
    write_mzml(io.from_scans(name or Path(path).stem, np.asarray(rt_s) / 60.0, [m for m, _ in scans], [i for _, i in scans], "pos"),
               path)


def test_mzml_of_a_unit_resolution_run_is_read_like_its_cdf(tmp_path):
    rng = np.random.default_rng(4)
    rt, scans = tof_unit_scans(rng)
    write_cdf(tmp_path / "tof.cdf", rt, scans)
    write_mzml_as_is(tmp_path / "tof.mzML", rt, scans)
    want = C.load_cloud(tmp_path / "tof.cdf", skew=0.0)
    got = io.load_cloud(tmp_path / "tof.mzML", unit=dict(skew=0.0, masses="auto"))
    assert got.source["kind"] == "unit resolution" and got.source["masses"] == "one per mass"
    assert got.mass_scale == pytest.approx(io.C13) and got.n_scans == want.n_scans
    assert np.array_equal(got.off, want.off) and np.allclose(got.mz, want.mz) and np.allclose(got.inten, want.inten, rtol=1e-5)
    assert np.allclose(got.rt, want.rt, atol=1e-5)
    # not asked: an mzML file is read as it is (LC-MS mode), and so is one told to hold accurate masses
    plain = io.load_cloud(tmp_path / "tof.mzML")
    assert plain.source is None and plain.mass_scale == 1.0 and plain.n_points == sum(len(m) for m, _ in scans)
    told = io.load_cloud(tmp_path / "tof.mzML", unit=dict(skew=0.0, masses="accurate"))
    assert told.source["kind"] == "accurate mass" and told.mass_scale == 1.0 and np.array_equal(told.mz, plain.mz)
    # the skew pass reads the same numbers from both files
    assert C.file_skew(tmp_path / "tof.mzML") == pytest.approx(C.file_skew(tmp_path / "tof.cdf"), abs=1e-3)
    assert C.file_skew(tmp_path / "tof.mzML", masses="accurate") == (0.0, 0)


def test_accurate_mass_mzml_stays_as_it_is_in_gc_mode(tmp_path):
    rng = np.random.default_rng(6)
    base = np.sort(rng.uniform(60, 500, 150))
    base = np.sort(np.concatenate([base, base[:60] + rng.uniform(0.02, 0.2, 60)]))    # ions sharing a nominal mass
    scans = [(base * (1 + rng.normal(0, 2e-6, len(base))), rng.uniform(1e3, 1e5, len(base))) for _ in range(300)]
    rt = 0.2 * np.arange(300)
    write_mzml_as_is(tmp_path / "acc.mzML", rt, scans)
    assert C.read_mzml(tmp_path / "acc.mzML") is None             # given up after the first scans
    assert C.read_mzml(tmp_path / "acc.mzML", masses="unit") is not None
    assert C.file_skew(tmp_path / "acc.mzML") == (0.0, 0)
    plain = io.load_cloud(tmp_path / "acc.mzML")
    got = io.load_cloud(tmp_path / "acc.mzML", unit=dict(skew=0.5, masses="auto"))
    assert got.source["kind"] == "accurate mass" and got.source["near_share"] > C.NEAR_SHARE and got.mass_scale == 1.0
    assert np.array_equal(got.mz, plain.mz) and np.array_equal(got.inten, plain.inten)


def test_holes_are_bridged_only_where_the_scan_explains_them():
    S = 200
    s = np.arange(S)
    X = np.zeros((S, 6))
    X[:, 0] = 1e3                                             # the floor of the file
    X[:, 1] = 1e5 + 5e9 * np.exp(-0.5 * ((s - 100) / 12.0) ** 2)      # a very strong ion around scan 100
    X[:, 2] = 2e5 + 1e3 * s                                   # an ion with a hole under it
    X[:, 3] = 2e5                                             # one with a hole where nothing strong elutes
    X[:, 4] = 5e3                                             # one too weak to say its trace did not fade out
    X[:, 5] = 2e5                                             # one with a hole longer than allowed
    X[96:103, 2] = X[20:27, 3] = X[96:103, 4] = X[85:116, 5] = 0.0
    want = 2e5 + 1e3 * s[96:103]
    B, n = C.bridge_holes(X, max_len=20)
    assert n == 1 and B.sum() == 7 and B[96:103, 2].all()
    assert np.allclose(X[96:103, 2], want)                    # the straight line between the two sides
    assert not X[20:27, 3].any() and not X[96:103, 4].any() and not X[85:116, 5].any()
    assert C.bridge_holes(X.copy(), max_len=0)[1] == 0
    # the file's peak width: Gaussian peaks of sigma 5 scans are 11.8 scans wide at half height
    Y = np.zeros((1200, 30))
    for k in range(30):
        Y[:, k] = 1e5 * (k + 1) * np.exp(-0.5 * ((np.arange(1200) - 40 * k - 20) / 5.0) ** 2)
    assert C.peak_width(Y) == pytest.approx(11.77, abs=0.3)
    assert C.peak_width(np.zeros((50, 4))) == 0.0


def holey_run(rng, n_scans=600):
    """a unit-resolution time-of-flight run (as tof_unit_scans writes it) with a very strong ion (mass 100) eluting at
    scan 300, and three stretches without centroids: mass 250 for 6 scans across its own peak under the strong ion,
    mass 260 for 6 scans where nothing strong elutes, and nothing else"""
    masses = np.arange(85, 321)
    s = np.arange(n_scans)
    scans = []
    phase = (7 * masses) % 200                      # every ion peaks three times in the run
    phase[masses == 250] = 102
    weak = masses > 300
    for k in range(n_scans):
        g = 2e5 * np.exp(-0.5 * (((k - phase + 100) % 200 - 100) / 5.0) ** 2)
        b = np.full(len(masses), 3e3)
        g[weak], b[weak] = 0.0, 2e3
        g[masses == 100] += 2e9 * np.exp(-0.5 * ((k - 300) / 5.0) ** 2)
        m = masses * 1.00047 + 0.013 + 0.08 * g / (g + b) + rng.normal(0, 3e-4, len(masses))
        keep = np.ones(len(masses), bool)
        if 298 <= k <= 303:
            keep[masses == 250] = False
        if 100 <= k <= 105:
            keep[masses == 260] = False
        scans.append((m[keep], (g + b)[keep]))
    return 60.0 + 0.06 * s, scans


def test_a_peak_with_a_hole_is_picked_whole_flagged_and_marked_for_the_viewer(tmp_path):
    rt, scans = holey_run(np.random.default_rng(8))
    write_cdf(tmp_path / "holey.cdf", rt, scans)
    write_mzml_as_is(tmp_path / "holey.mzML", rt, scans)
    cloud = C.load_cloud(tmp_path / "holey.cdf", skew=0.0)
    src = cloud.source
    assert src["masses"] == "one per mass" and src["bridge"] is True
    assert src["bridged_holes"] == 1 and src["bridged_points"] == 6 and 20 <= src["bridge_max_scans"] <= 30
    at = np.flatnonzero(cloud.bridged)
    assert np.array_equal(np.searchsorted(cloud.off, at, side="right") - 1, np.arange(298, 304))
    assert np.all(C.unit_mass(cloud.mz[at]) == 250)
    want = 3e3 + 2e5 * np.exp(-0.5 * ((np.arange(298, 304) - 302) / 5.0) ** 2)
    assert np.all(cloud.inten[at] > 0.6 * want) and np.all(cloud.inten[at] <= 1.01 * want.max())
    assert cloud.eic(260 * io.C13, 5.0, 100, 106).max() == 0             # nothing strong there: the hole stays
    plain = C.load_cloud(tmp_path / "holey.cdf", skew=0.0, bridge=False)
    assert plain.bridged is None and plain.n_points == cloud.n_points - 6 and plain.source["bridge"] is False
    via_mzml = io.load_cloud(tmp_path / "holey.mzML", unit=dict(skew=0.0, masses="auto"))
    assert np.array_equal(np.flatnonzero(via_mzml.bridged), at)
    cloud.save_npz(tmp_path / "c.npz")                                   # the marks stay with the cached cloud
    assert np.array_equal(io.Cloud.load_npz(tmp_path / "c.npz").bridged, cloud.bridged)
    plain.save_npz(tmp_path / "p.npz")
    assert io.Cloud.load_npz(tmp_path / "p.npz").bridged is None

    # a run: the peak of mass 250 comes out whole and flagged; without bridging it does not
    (tmp_path / "in").mkdir()
    (tmp_path / "holey.mzML").rename(tmp_path / "in" / "holey.mzML")
    out = tmp_path / "out"
    r = _run("pick", str(tmp_path / "in" / "holey.mzML"), "--output", str(out), "--mode", "gc")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "1 holes filled (6 interpolated centroids" in r.stdout
    f = pd.read_csv(out / "features" / "holey.tsv", sep="\t")
    f["nom"] = C.unit_mass(f["mz"].values)
    hit = f[(f["nom"] == 250) & (f["scan_min"] <= 298) & (f["scan_max"] >= 303)]
    assert len(hit) == 1 and hit.iloc[0]["n_bridged"] == 6 and "bridged" in str(hit.iloc[0]["flags"])
    assert abs(int(hit.iloc[0]["scan_apex"]) - 302) <= 3 and 0.85 * 2.03e5 < hit.iloc[0]["height"] <= 2.04e5   # the top is cut
    rest = f.drop(hit.index)
    assert (rest["n_bridged"] == 0).all() and not rest["flags"].fillna("").str.contains("bridged").any()
    # the viewer's chromatogram of that ion says which of its points are interpolated
    e = bundle.extract_eic(out, out / ".cache", "holey", 250 * io.C13, 5.0)
    assert e["bridged"] == [298, 304] and min(e["intensity"][296:306]) > 0
    assert "bridged" in bundle.extract_eic(out, out / ".cache", "holey", 260 * io.C13, 5.0) and \
        bundle.extract_eic(out, out / ".cache", "holey", 260 * io.C13, 5.0)["bridged"] == []
    # the cache gone: the file is read again as the run read it (process records its input files; pick does not)
    (out / "params.json").write_text(json.dumps({"input_files": [str(tmp_path / "in" / "holey.mzML")]}))
    e2 = bundle.extract_eic(out, out / "no_cache_here", "holey", 250 * io.C13, 5.0)
    assert e2["source"] == "mzML" and e2["bridged"] == [298, 304]
    out2 = tmp_path / "out_plain"
    r = _run("pick", str(tmp_path / "in" / "holey.mzML"), "--output", str(out2), "--mode", "gc", "--no-bridge")
    assert r.returncode == 0, r.stderr[-2000:]
    g = pd.read_csv(out2 / "features" / "holey.tsv", sep="\t")
    assert (g["n_bridged"] == 0).all() and "holes filled" not in r.stdout
    g["nom"] = C.unit_mass(g["mz"].values)
    assert not len(g[(g["nom"] == 250) & (g["scan_min"] <= 298) & (g["scan_max"] >= 303)])
    assert "bridged" not in bundle.extract_eic(out2, out2 / ".cache", "holey", 250 * io.C13, 5.0)


def test_run_skew_needs_files_that_agree():
    assert C.study_skew([(-1.31, 80), (-1.16, 80), (-0.95, 80), (-1.2, 80), (-1.1, 80)]) == pytest.approx(-1.16)
    assert C.study_skew([(0.5, 80), (0.7, 80), (0.55, 80), (0.6, 80), (-0.6, 80)]) == pytest.approx(0.55)     # one file off
    assert C.study_skew([(-0.87, 80), (0.23, 80), (-0.35, 80), (-0.07, 80), (-0.45, 80), (0.05, 80)]) == 0.0   # scatter
    assert C.study_skew([(0.6, 80)]) == pytest.approx(0.6) and C.study_skew([(0.6, 80), (0.4, 80)]) == pytest.approx(0.5)
    assert C.study_skew([(0.6, 3), (0.5, 0)]) == 0.0            # too few peaks to measure on


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


def test_mzml_folder_of_a_unit_resolution_run_gives_the_compounds_of_the_cdf_folder(study, tmp_path):
    data = tmp_path / "mzml"
    data.mkdir()
    for f in sorted(study["data"].iterdir()):
        rt, off, mz, it = C.read_cdf(f)
        write_mzml_as_is(data / f"{f.stem}.mzML", rt, [(mz[off[s]:off[s + 1]], it[off[s]:off[s + 1]]) for s in range(len(rt))])
    out = tmp_path / "out"
    r = _run("process", "--input", str(data), "--output", str(out), "--mode", "gc", "--ri-ladder", "alkane")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "gc stages failed" not in r.stdout and "unit-resolution" in r.stdout
    p = json.loads((out / "params.json").read_text())
    assert p["mass_scale"] == pytest.approx(io.C13) and [Path(f).suffix for f in p["input_files"]] == [".mzML"] * 3
    fp = json.loads((out / "features" / "q_a.params.json").read_text())
    ref = json.loads((study["out"] / "features" / "q_a.params.json").read_text())
    assert fp["source"]["kind"] == "unit resolution" and fp["source"]["masses"] == "grid"
    assert fp["source"]["scan_skew"] == pytest.approx(ref["source"]["scan_skew"], abs=1e-3)
    assert fp["n_features"] == ref["n_features"]
    ft = pd.read_csv(out / "features" / "q_a.tsv", sep="\t")          # the table as the alignment wrote it back
    assert "rt_corr" in ft and "n_bridged" in ft and (ft["n_bridged"] == 0).all()
    a = pd.read_csv(out / "gc" / "compounds.tsv", sep="\t")
    b = pd.read_csv(study["out"] / "gc" / "compounds.tsv", sep="\t")
    assert len(a) == len(b) and np.array_equal(a["quant_mz"], b["quant_mz"]) and np.allclose(a["rt"], b["rt"], atol=1e-4)
    # the cache gone: the chromatograms come from the mzML files, read as the run read them (nominal masses)
    hd, arr = bundle.unpack(gcview.build(out))
    k = int(arr["c_id"][int(np.argmax(arr["c_height"]))])
    t = gcview.trace(out, out / ".cache", k)
    t2 = gcview.trace(out, out / "no_cache_here", k)
    assert t2["source"] == "mzML" and max(t2["ions"][0]["y"]) > 0
    assert np.allclose(t2["ions"][0]["y"], t["ions"][0]["y"], rtol=1e-4)
    # in LC-MS mode the same files are read as they are
    out_lc = tmp_path / "out_lc"
    r = _run("pick", str(data / "q_a.mzML"), "--output", str(out_lc))
    assert r.returncode == 0, r.stderr[-2000:]
    assert "source" not in json.loads((out_lc / "features" / "q_a.params.json").read_text())
