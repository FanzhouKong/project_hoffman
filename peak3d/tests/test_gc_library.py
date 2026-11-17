import numpy as np

from peak3d.gc import library as L

MSP = """Name: Decane
Formula: C10H22
MW: 142
Retention_index: SemiStdNP=1000/2/55 StdNP=1000/1/20 StdPolar=1012/3/9
Num Peaks: 6
41 300; 43 800; 57 999; 71 400;
85 250; 142 30

NAME: Twin of decane
RI: 1400
Num Peaks: 6
41 300; 43 800; 57 999; 71 400; 85 250; 142 30

Name: Lactic acid, 2TMS
InChIKey: KKEYFWRCBNTPAC-UHFFFAOYSA-N
Spectrum_type: MS1
Comments: "retention index=1066.5" "derivative=TMS"
Num Peaks: 5
73\t999
117\t650
147\t700
191\t120
219\t60
Name: No index
Num Peaks: 3
50 10
60 999
70 500

Name: Empty
Num Peaks: 0
"""


def _lib(tmp_path, text=MSP, name="lib.msp"):
    f = tmp_path / name
    f.write_text(text)
    return f


def test_read_msp_dialects(tmp_path):
    lib = L.read_msp(_lib(tmp_path))
    assert list(lib.name) == ["Decane", "Twin of decane", "Lactic acid, 2TMS", "No index"]
    assert lib.n_skipped == 1 and lib.nominal
    assert np.allclose(lib.ri[:3], [1000.0, 1400.0, 1066.5]) and np.isnan(lib.ri[3])
    assert lib.formula[0] == "C10H22" and lib.mw[0] == 142 and lib.inchikey[2].startswith("KKEY")
    m, i = lib.spectrum(0)
    assert m.tolist() == [41, 43, 57, 71, 85, 142] and i.tolist() == [300, 800, 999, 400, 250, 30]
    assert lib.spectrum(2)[0].tolist() == [73, 117, 147, 191, 219]            # an MS1-marked record is an EI spectrum
    assert L.read_msp(_lib(tmp_path), ri_type="polar").ri[0] == 1012.0


def test_nominal_mass_rule():
    # hydrogen-rich fragments are mass-sufficient by a few tenths; silicon and halogens pull below the integer
    assert L.nominal_mass([57.0699, 562.642, 72.97, 73.0468, 206.69, 206.71]).tolist() == [57, 562, 73, 73, 206, 207]


def test_accurate_spectrum_finds_its_unit_mass_record(tmp_path):
    lib = L.read_msp(_lib(tmp_path))
    q_mz = np.array([41.0386, 43.0542, 57.0699, 71.0855, 85.1012, 142.1716])
    q_int = np.array([300.0, 800, 999, 400, 250, 30])
    h = L.search(np.array([0, 6]), q_mz, q_int, lib)
    assert h["name"].iloc[0] == "Decane" and h["match"].iloc[0] > 0.999 and h["n_matched"].iloc[0] == 6
    assert set(h["name"]) >= {"Decane", "Twin of decane"}


def test_retention_index_window_removes_the_wrong_twin(tmp_path):
    lib = L.read_msp(_lib(tmp_path))
    q_mz = np.array([41.0386, 43.0542, 57.0699, 71.0855, 85.1012, 142.1716])
    q_int = np.array([300.0, 800, 999, 400, 250, 30])
    off = np.array([0, 6])
    h = L.search(off, q_mz, q_int, lib, q_ri=[1004.0], P=L.SearchParams(ri_tol=20.0))
    assert h["name"].tolist() == ["Decane"] and abs(h["delta_ri"].iloc[0] - 4.0) < 1e-9
    h = L.search(off, q_mz, q_int, lib, q_ri=[1395.0], P=L.SearchParams(ri_tol=20.0))
    assert h["name"].tolist() == ["Twin of decane"]
    h = L.search(off, q_mz, q_int, lib, q_ri=[np.nan], P=L.SearchParams(ri_tol=20.0))      # no index: no window
    assert set(h["name"]) == {"Decane", "Twin of decane"}
    # a record without an index stays a candidate unless told otherwise
    q2 = (np.array([0, 3]), np.array([50.01, 60.02, 70.03]), np.array([10.0, 999, 500]))
    assert L.search(*q2, lib, q_ri=[1000.0], P=L.SearchParams(ri_tol=20.0))["name"].tolist() == ["No index"]
    assert len(L.search(*q2, lib, q_ri=[1000.0], P=L.SearchParams(ri_tol=20.0, keep_no_ri=False))) == 0


def test_reverse_score_ignores_a_coeluting_compounds_ions(tmp_path):
    lib = L.read_msp(_lib(tmp_path))
    q_mz = np.array([73.0468, 117.0730, 147.0655, 160.1, 191.0920, 205.1, 219.1050])
    q_int = np.array([999.0, 650, 700, 500, 120, 300, 60])                     # two foreign ions (160.1, 205.1)
    h = L.search(np.array([0, 7]), q_mz, q_int, lib)
    top = h.iloc[0]
    assert top["name"] == "Lactic acid, 2TMS" and top["reverse"] > 0.999 and 0.3 < top["match"] < 0.9


def test_library_peaks_outside_the_acquired_range_do_not_count(tmp_path):
    lib = L.read_msp(_lib(tmp_path))
    q_mz = np.array([57.0699, 71.0855, 85.1012, 142.1716])                    # the run started at m/z 50
    q_int = np.array([999.0, 400, 250, 30])
    off = np.array([0, 4])
    assert L.search(off, q_mz, q_int, lib, mz_range=(50.0, 500.0))["match"].iloc[0] > 0.999
    assert L.search(off, q_mz, q_int, lib, mz_range=(35.0, 500.0))["match"].iloc[0] < 0.999


def test_accurate_mass_library_matches_within_tolerance(tmp_path):
    text = """Name: A
RetentionIndex: 1500
Num Peaks: 3
73.0468 999
147.0655 700
217.1070 300

Name: B same nominal masses
RetentionIndex: 1500
Num Peaks: 3
73.0100 999
147.1200 700
217.0200 300
"""
    lib = L.read_msp(_lib(tmp_path, text, "hr.msp"))
    assert not lib.nominal
    h = L.search(np.array([0, 3]), np.array([73.0469, 147.0653, 217.1073]), np.array([999.0, 700, 300]), lib)
    assert h["name"].tolist() == ["A"] and h["match"].iloc[0] > 0.999


def test_cache_gives_the_same_library(tmp_path):
    f = _lib(tmp_path)
    a = L.load(f, tmp_path / "cache")
    b = L.load(f, tmp_path / "cache")
    assert len(list((tmp_path / "cache").glob("ei_library_*.npz"))) == 1
    assert list(a.name) == list(b.name) and np.array_equal(a.mz, b.mz) and np.array_equal(a.ri, b.ri, equal_nan=True)
    assert a.nominal == b.nominal


def test_write_msp_round_trip(tmp_path):
    off = np.array([0, 2, 5])
    mz = np.array([57.0699, 71.0855, 73.0468, 147.0655, 217.107])
    ab = np.array([500.0, 250, 100, 70, 30])
    L.write_msp(tmp_path / "out.msp", ["one", "two"], off, mz, ab, ri=np.array([1000.0, np.nan]), rt=np.array([1.5, 2.5]))
    lib = L.read_msp(tmp_path / "out.msp")
    assert list(lib.name) == ["one", "two"] and lib.ri[0] == 1000.0 and np.isnan(lib.ri[1])
    assert np.allclose(lib.mz, mz, atol=1e-5) and np.allclose(lib.spectrum(0)[1], [999.0, 499.5])
