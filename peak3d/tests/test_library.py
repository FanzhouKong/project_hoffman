"""MSP parsing, spectrum cleaning, the five similarity scores and the identity search."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from peak3d import library
from peak3d.library import METHODS, identify, load, read_msp, write_msp_record
from peak3d.similarity import clean, similarity

MSP = '''NAME: Alanine
PRECURSORMZ: 90.0550
PRECURSORTYPE: [M+H]+
IONMODE: Positive
FORMULA: C3H7NO2
INCHIKEY: QNAYBMKLOCPYGJ-REOHCLBHSA-N
Num Peaks: 3
44.0495\t100
45.0335\t12 "annotation with spaces"
72.0444\t5

Name: Citrate (NIST style)
Precursor_type: [M-H]-
Ion_mode: N
PrecursorMZ: 191.0197 100
Spectrum_type: MS2
Num Peaks: 4
87.0 100; 111.0 40; 129.0 10;
173.0 5;

Name: an MS1 record
Spectrum_type: MS1
PrecursorMZ: 300.0
Num Peaks: 1
300.0 100

Name: no blank line before the next record
precursor_mz: 150.5
ionmode: positive
Num Peaks: 2
60.0 10
50.0 20
Name: Empty
PrecursorMZ: 200
Num Peaks: 0

NAME: No precursor
Num Peaks: 1
55.0 1
'''


def test_read_msp_dialects(tmp_path):
    p = tmp_path / "lib.msp"
    p.write_text(MSP)
    lib = read_msp(p)
    assert lib.names == ["Alanine", "Citrate (NIST style)", "no blank line before the next record", "No precursor"]
    assert lib.n_skipped == 2                                            # the MS1 record and the empty one
    assert lib.precursor_mz[0] == pytest.approx(90.055) and lib.precursor_mz[1] == pytest.approx(191.0197)
    assert math.isnan(lib.precursor_mz[3])
    assert lib.ion_mode == ["pos", "neg", "pos", ""]
    assert lib.precursor_type[:2] == ["[M+H]+", "[M-H]-"] and lib.formula[0] == "C3H7NO2"
    assert lib.inchikey[0].startswith("QNAYBMKLOCPYGJ")
    assert np.diff(lib.off).tolist() == [3, 4, 2, 1]
    a, b = lib.off[1], lib.off[2]
    assert lib.mz[a:b].tolist() == [87.0, 111.0, 129.0, 173.0] and lib.inten[a:b].tolist() == [100, 40, 10, 5]
    a, b = lib.off[2], lib.off[3]
    assert lib.mz[a:b].tolist() == [50.0, 60.0] and lib.inten[a:b].tolist() == [20, 10]   # sorted by m/z


def test_msp_roundtrip(tmp_path):
    p = tmp_path / "out.msp"
    with open(p, "w") as fh:
        write_msp_record(fh, [("NAME", "x"), ("PRECURSORMZ", 123.456), ("IONMODE", "Negative"), ("FORMULA", ""),
                              ("Comment", 'group_id=7 file="a b" scan="controllerType=0 scan=12"')],
                         np.array([50.0, 60.5]), np.array([1.0, 2.5]))
    lib = read_msp(p)
    assert lib.names == ["x"] and lib.ion_mode == ["neg"] and lib.formula == [""]
    assert lib.mz.tolist() == [50.0, 60.5] and lib.inten.tolist() == [1.0, 2.5]
    assert 'scan="controllerType=0 scan=12"' in lib.comments[0]


def test_load_is_cached_and_follows_changes(tmp_path):
    p = tmp_path / "lib.msp"
    p.write_text(MSP)
    a = load(p)
    assert load(p) is a
    p.write_text(MSP.replace("Alanine", "Alanine 2") + "\n")
    assert load(p) is not a and load(p).names[0] == "Alanine 2"


def test_read_msp_line_endings_numbers_and_colon_pairs(tmp_path):
    p = tmp_path / "lib.msp"
    p.write_bytes(b"Name: a\r\nPrecursorMZ: 1.005E+02\r\nPrecursor_type: [M-H]-\r\nNum Peaks: 3\r\n"
                  b"50.5 1.5e3 0.4\r\n.5 2\r\n70 -1\r\n\r\n"                      # CRLF; a third column; a negative peak
                  b"Name: b\rPrecursor_type: [M+2H]2+\rNum peaks: 3\r40:10 41:20 42:x 43:5\r\r"     # CR; MassBank pairs
                  b"Name: c\nAdduct: M+H\nIonization: ESI\nComments: \"SMILES=CCO\"\n80, 7; 90, nope\n")
    lib = read_msp(p)
    assert lib.names == ["a", "b", "c"] and lib.n_skipped == 0
    assert lib.precursor_mz[0] == 100.5 and np.isnan(lib.precursor_mz[1:]).all()
    assert lib.ion_mode == ["neg", "pos", ""]          # from the precursor type's charge; "M+H" has none
    assert np.diff(lib.off).tolist() == [2, 2, 1]
    assert lib.mz.tolist() == [0.5, 50.5, 40.0, 41.0, 80.0] and lib.inten.tolist() == [2, 1500, 10, 20, 7]
    assert lib.comments[2] == '"SMILES=CCO"' and lib.precursor_type[2] == "M+H"


def test_ion_mode_field_wins_over_the_precursor_type(tmp_path):
    p = tmp_path / "lib.msp"
    p.write_text("Name: a\nIon_mode: N\nPrecursor_type: [M+H]+\n50 1\n\nName: b\nPrecursor_type: [M]+*\n50 1\n")
    assert read_msp(p).ion_mode == ["neg", "pos"]


def test_load_leaves_comments_out(tmp_path):
    p = tmp_path / "lib.msp"
    p.write_text(MSP.replace("FORMULA: C3H7NO2", "FORMULA: C3H7NO2\nComment: a long comment"))
    assert read_msp(p).comments[0] == "a long comment"
    lib = load(p)
    assert lib.comments[0] == "" and lib.names[0] == "Alanine" and lib.formula[0] == "C3H7NO2"


def test_blocks_and_threads_give_the_same_library(tmp_path):
    from peak3d import msp
    p = tmp_path / "lib.msp"
    text = MSP.encode()
    p.write_bytes(text.replace(b"\n", b"\r\n") * 4 + text * 5 + text.replace(b"\n", b"\r") * 3 + text)
    one = msp.read(p, True, n_threads=1, block=1 << 30)
    many = msp.read(p, True, n_threads=3, block=97)               # blocks smaller than a record: carried over
    assert len(one[2]) == 13 * 4 and one[7] == 13 * 2
    for a, b in zip(one[:7], many[:7]):
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
    assert one[7] == many[7]


def test_empty_and_peakless_files(tmp_path):
    (tmp_path / "empty.msp").write_bytes(b"")
    (tmp_path / "none.msp").write_text("Name: x\nPrecursorMZ: 100\n\n")
    assert len(read_msp(tmp_path / "empty.msp")) == 0
    lib = read_msp(tmp_path / "none.msp")
    assert len(lib) == 0 and lib.n_skipped == 1 and lib.off.tolist() == [0]


# ------------------------------------------------------------------ cleaning ----
def test_clean_cuts_precursor_merges_close_peaks_and_drops_noise():
    m, i = clean(np.array([150.0, 198.5, 199.0]), np.array([1.0, 1.0, 1.0]), 200.0, 0.02)
    assert m.tolist() == [150.0] and i.tolist() == [1.0]                 # >= precursor - 1.6 removed
    m, i = clean(np.array([100.0, 100.03, 150.0, 160.0]), np.array([10.0, 5.0, 10.0, 0.05]), np.nan, 0.02)
    assert len(m) == 2                                                   # 0.03 Da apart: one peak; 0.05 < 1 %: gone
    assert m[0] == pytest.approx((100.0 * 10 + 100.03 * 5) / 15) and i[0] == pytest.approx(15 / 25)
    assert i.sum() == pytest.approx(1.0)


# ------------------------------------------------------------------ scores ----
A = (np.array([100.0, 200.0]), np.array([60.0, 40.0]))
B = (np.array([100.0, 300.0]), np.array([60.0, 40.0]))


@pytest.mark.parametrize("method", list(METHODS))
def test_identical_is_one_and_disjoint_is_zero(method):
    m = np.array([50.0, 80.0, 120.0, 150.0])
    it = np.array([10.0, 100.0, 30.0, 5.0])
    s, n = similarity(m, it, m, it, method)
    assert s == pytest.approx(1.0) and n == 4
    s, n = similarity(m, it, m + 3.0, it, method)
    assert s == 0.0 and n == 0


def test_two_peak_case_by_hand():
    # one shared peak of 0.6 in both: entropy 0.5 x [1.2 log2 1.2 - 2 x 0.6 log2 0.6] = 0.6 log2 2 = 0.6
    assert similarity(*A, *B, "entropy")[0] == pytest.approx(0.6)
    assert similarity(*A, *B, "dot")[0] == pytest.approx(0.36 / 0.52)
    S = -(0.6 * math.log(0.6) + 0.4 * math.log(0.4))
    w = 0.25 + 0.25 * S                                                  # S < 3: reweighted
    x = 0.6 ** w / (0.6 ** w + 0.4 ** w)
    assert similarity(*A, *B, "weighted_entropy")[0] == pytest.approx(x)  # same pattern in both: score = shared share
    a = np.array([0.6 ** 0.6 * 100 ** 3, 0.4 ** 0.6 * 200 ** 3])
    b = np.array([0.6 ** 0.6 * 100 ** 3, 0.4 ** 0.6 * 300 ** 3])
    assert similarity(*A, *B, "weighted_dot")[0] == pytest.approx(a[0] * b[0] / np.sqrt((a * a).sum() * (b * b).sum()))


def test_flat_spectra_are_not_reweighted():
    m = 100.0 + np.arange(30) * 2.0                                       # 30 equal peaks: entropy ln 30 > 3
    mb = np.concatenate([m[:20], 500.0 + np.arange(10) * 2.0])
    it = np.ones(30)
    assert similarity(m, it, mb, it, "weighted_entropy")[0] == pytest.approx(similarity(m, it, mb, it, "entropy")[0])
    assert similarity(m, it, mb, it, "entropy")[0] == pytest.approx(20 / 30)


def test_reverse_ignores_extra_query_peaks():
    lib = (np.array([100.0, 200.0]), np.array([50.0, 50.0]))
    query = (np.array([100.0, 200.0, 300.0]), np.array([50.0, 50.0, 200.0]))
    assert similarity(*query, *lib, "reverse_entropy")[0] == pytest.approx(1.0)
    assert similarity(*query, *lib, "weighted_entropy")[0] < 0.75
    # the other way round: a library peak missing from the query still costs
    assert similarity(*lib, *query, "reverse_entropy")[0] < 0.75


def test_fragment_tolerance():
    a = (np.array([100.0, 200.0]), np.array([1.0, 1.0]))
    b = (np.array([100.015, 200.015]), np.array([1.0, 1.0]))
    assert similarity(*a, *b, "entropy", fragment_tol=0.02)[1] == 2
    assert similarity(*a, *b, "entropy", fragment_tol=0.01)[1] == 0


# ------------------------------------------------------------------ identify ----
FRAG = np.array([60.04, 72.08, 86.10, 104.11])
FRAG_I = np.array([30.0, 100.0, 45.0, 10.0])


def _output(tmp_path, with_ms2=True):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame({"group_id": [1, 2], "mz": [132.1019, 250.0], "rt": [1.0, 2.0]}).to_csv(
        out / "aligned_feature_table.tsv", sep="\t", index=False)
    if with_ms2:
        with open(out / "ms2_spectra.msp", "w") as fh:
            write_msp_record(fh, [("NAME", "feature_1"), ("PRECURSORMZ", "132.101900"), ("IONMODE", "Positive"),
                                  ("Comment", 'group_id=1 file="S0" scan="scan=5" rt_s=60.000 n_ms2=3')],
                             FRAG, FRAG_I)
            write_msp_record(fh, [("NAME", "feature_2"), ("PRECURSORMZ", "250.000000"), ("IONMODE", "Positive"),
                                  ("Comment", 'group_id=2 file="S1" scan="scan=9" rt_s=120.000 n_ms2=1')],
                             np.array([80.0, 120.0]), np.array([1.0, 1.0]))
    return out


def _library(tmp_path):
    p = tmp_path / "lib.msp"
    with open(p, "w") as fh:
        write_msp_record(fh, [("NAME", "Leucine"), ("PRECURSORMZ", 132.1019), ("PRECURSORTYPE", "[M+H]+"),
                              ("IONMODE", "Positive"), ("FORMULA", "C6H13NO2")], FRAG, FRAG_I * 2)
        write_msp_record(fh, [("NAME", "Isoleucine-ish"), ("PRECURSORMZ", 132.1020), ("IONMODE", "P")],
                         FRAG, np.array([100.0, 20.0, 10.0, 5.0]))
        write_msp_record(fh, [("NAME", "Decoy"), ("PRECURSORMZ", 132.1019)], np.array([55.0, 91.0]), np.array([1.0, 1.0]))
        write_msp_record(fh, [("NAME", "Wrong mode"), ("PRECURSORMZ", 132.1019), ("IONMODE", "Negative")], FRAG, FRAG_I)
        write_msp_record(fh, [("NAME", "Too far"), ("PRECURSORMZ", 132.1050)], FRAG, FRAG_I)   # 23 ppm
    return p


def test_identify_end_to_end(tmp_path):
    out, lib = _output(tmp_path), _library(tmp_path)
    hits = identify(out, lib, "weighted_entropy", min_score=0.0, top_k=3)
    t = pd.read_csv(out / "identifications.tsv", sep="\t")
    assert len(t) == len(hits) and set(t["group_id"]) == {1}             # feature 2 has no candidate
    assert t["name"].tolist()[0] == "Leucine" and t["rank"].tolist() == list(range(1, len(t) + 1))
    assert "Wrong mode" not in set(t["name"]) and "Too far" not in set(t["name"])
    assert t["score"].iloc[0] == pytest.approx(1.0) and t["n_matched"].iloc[0] == 4
    assert t["ppm_error"].iloc[0] == pytest.approx(0.0, abs=0.01) and t["formula"].iloc[0] == "C6H13NO2"
    assert (t["score"].diff().dropna() <= 0).all()
    back = read_msp(out / "identifications.msp")
    assert back.names == t["name"].tolist() and "group_id=1 rank=1" in back.comments[0]
    info = json.loads((out / "identifications.json").read_text())
    assert info["method"] == "weighted_entropy" and info["n_queries"] == 2 and info["n_identified"] == 1
    assert info["library_n"] == 5 and info["method_label"] == METHODS["weighted_entropy"]
    # min_score drops the weak hits; top_k caps the rest
    t = identify(out, lib, "dot", min_score=0.9, top_k=1)
    assert t["name"].tolist() == ["Leucine"]


def test_identify_rewrites_stale_results(tmp_path):
    out, lib = _output(tmp_path), _library(tmp_path)
    identify(out, lib, min_score=0.0)
    (out / "ms2_spectra.msp").unlink()
    identify(out, lib, min_score=0.0)
    assert len(pd.read_csv(out / "identifications.tsv", sep="\t")) == 0
    assert (out / "identifications.msp").read_text() == ""
    assert json.loads((out / "identifications.json").read_text())["n_queries"] == 0


def test_candidate_pairs_window_and_modes():
    q_mz = np.array([100.0, 200.0])
    lib_prec = np.array([np.nan, 100.0005, 99.9995, 100.01, 200.0, 200.0])
    pq, pl = library.candidate_pairs(q_mz, np.array([1, 2], np.int8), lib_prec, np.array([0, 1, 0, 1, 1, 0], np.int8), 10)
    assert sorted(zip(pq.tolist(), pl.tolist())) == [(0, 1), (0, 2), (1, 5)]


# ------------------------------------------------------------------ MassCube / ms_entropy .pkl libraries ----
def _flash_pickle(path, records):
    """a FlashEntropySearch pickle laid out as ms_entropy 1.3 writes it (one pickled dict per spectrum in metadata,
    sorted by precursor m/z), made with stand-in classes registered under ms_entropy's module names"""
    import pickle
    import sys
    import types
    names = {"FlashEntropySearch": "ms_entropy.entropy_search.flash_entropy_search",
             "FlashEntropySearchCore": "ms_entropy.entropy_search.flash_entropy_search_core"}
    mods = {"ms_entropy", "ms_entropy.entropy_search", *names.values()}      # pickle imports the parents too
    saved = {m: sys.modules.get(m) for m in mods}
    for mod in mods:
        sys.modules[mod] = types.ModuleType(mod)
    classes = {}
    for cls, mod in names.items():
        c = type(cls, (), {"__module__": mod})
        setattr(sys.modules[mod], cls, c)
        classes[cls] = c
    try:
        recs = sorted(records, key=lambda r: r["precursor_mz"])
        blobs = [pickle.dumps(r) for r in recs]
        obj = classes["FlashEntropySearch"]()
        obj.precursor_mz_array = np.array([r["precursor_mz"] for r in recs], np.float32)
        obj.metadata_loc = np.cumsum([0] + [len(b) for b in blobs]).astype(np.uint64)
        obj.metadata = np.frombuffer(b"".join(blobs), np.uint8)
        obj.low_memory = False
        obj.entropy_search = classes["FlashEntropySearchCore"]()
        obj.entropy_search.index = [np.arange(4.0), np.arange(4, dtype=np.float32)]
        obj.entropy_search.intensity_weight = None
        path.write_bytes(pickle.dumps(obj))
    finally:
        for m, v in saved.items():
            if v is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = v


def test_read_masscube_pkl(tmp_path):
    p = tmp_path / "MassCube_DB.pkl"
    _flash_pickle(p, [
        dict(precursor_mz=181.070665, peaks=np.array([[163.06, 0.5], [85.03, 0.2], [145.05, 0.3]], np.float32),
             name="Glucose", precursor_type="[M+H]+", ion_mode="Positive", formula="C6H12O6",
             inchikey="WQZGKKKJIJFFOK-GASJEMHNSA-N", smiles="OCC1OC(O)C(O)C(O)C1O", retention_time=2.1),
        dict(precursor_mz=179.056113, peaks=np.array([[89.02, 1.0]], np.float32), name="Glucose",
             precursor_type="[M-H]-", formula="C6H12O6"),                     # ion mode from the precursor type
        dict(precursor_mz=300.0, peaks=np.zeros((0, 2), np.float32), name="no peaks")])
    lib = library.read_library(p)
    assert lib.names == ["Glucose", "Glucose"] and lib.n_skipped == 1          # sorted by precursor m/z, as stored
    assert lib.precursor_mz.tolist() == [179.056113, 181.070665]               # the records' own values, not float32
    assert lib.ion_mode == ["neg", "pos"] and lib.precursor_type == ["[M-H]-", "[M+H]+"]
    assert lib.formula == ["C6H12O6"] * 2 and lib.smiles[1].startswith("OCC1") and lib.inchikey[0] == ""
    assert np.diff(lib.off).tolist() == [1, 3]
    assert lib.mz[1:].tolist() == pytest.approx([85.03, 145.05, 163.06], abs=1e-4)   # sorted by m/z
    assert load(p) is load(p)


def test_pkl_and_msp_give_the_same_hits(tmp_path):
    out, msp_path = _output(tmp_path), _library(tmp_path)
    src = read_msp(msp_path)
    recs = [dict(precursor_mz=float(src.precursor_mz[i]), peaks=np.column_stack(
                 [src.mz[src.off[i]:src.off[i + 1]], src.inten[src.off[i]:src.off[i + 1]]]).astype(np.float32),
                 name=src.names[i], precursor_type=src.precursor_type[i], formula=src.formula[i],
                 ion_mode={"pos": "Positive", "neg": "Negative", "": ""}[src.ion_mode[i]])
            for i in range(len(src)) if np.isfinite(src.precursor_mz[i])]
    _flash_pickle(tmp_path / "lib.pkl", recs)
    for method in METHODS:
        a = identify(out, msp_path, method, min_score=0.0)
        b = identify(out, tmp_path / "lib.pkl", method, min_score=0.0)
        assert a["name"].tolist() == b["name"].tolist() and a["score"].tolist() == pytest.approx(b["score"].tolist())


def test_pkl_that_asks_for_code_is_refused(tmp_path):
    import pickle

    class Payload:
        def __reduce__(self):
            return (print, ("this must not run",))

    p = tmp_path / "evil.pkl"
    p.write_bytes(pickle.dumps({"metadata": Payload()}))
    with pytest.raises(pickle.UnpicklingError, match="builtins.print"):
        library.read_library(p)
