import numpy as np
import pytest

from peak3d.gc import ri as RI
from peak3d.gc.synth import Compound, alkane, fame, random_spectrum


def _csr(compounds):
    """spectra of the compounds as find_ladder takes them (components in elution order)"""
    compounds = sorted(compounds, key=lambda c: c.rt)
    off, mz, ab = [0], [], []
    for c in compounds:
        o = np.argsort(c.mz)
        mz.append(np.asarray(c.mz)[o])
        ab.append(np.asarray(c.rel)[o] * c.height)
        off.append(off[-1] + len(c.mz))
    return np.array([c.rt for c in compounds]), np.array(off), np.concatenate(mz), np.concatenate(ab), compounds


def test_map_interpolates_linearly_and_marks_extrapolation():
    m = RI.RIMap([2.0, 1.0, 4.0], [1100.0, 1000.0, 1200.0])            # sorted on construction
    r, ex = m.ri_at([1.0, 1.5, 3.0, 0.5, 5.0])
    assert np.allclose(r, [1000, 1050, 1150, 950, 1250])
    assert ex.tolist() == [False, False, False, True, True]
    with pytest.raises(ValueError):
        RI.RIMap([1.0], [1000.0])
    with pytest.raises(ValueError):
        RI.RIMap([1.0, 2.0], [1100.0, 1000.0])


def test_parse_carbons():
    assert RI.parse_carbons("10-13") == [10, 11, 12, 13]
    assert RI.parse_carbons("8,9,10,12-18:2") == [8, 9, 10, 12, 14, 16, 18]
    with pytest.raises(ValueError):
        RI.parse_carbons("")


def test_alkane_ladder_named_by_molecular_ions():
    rng = np.random.default_rng(0)
    lad = [alkane(n, 1.0 + 0.5 * (n - 10), 1e5) for n in range(10, 17)]
    decoys = [Compound(f"X{k}", 0.8 + 0.37 * k, 3e5, *random_spectrum(rng, 20)) for k in range(8)]
    rt, off, mz, ab, comps = _csr(lad + decoys)
    table, note = RI.find_ladder(rt, off, mz, ab, "alkane")
    assert table["carbon"].tolist() == list(range(10, 17))
    assert np.allclose(table["ri"], 100.0 * table["carbon"])
    assert np.allclose(table["rt"], [1.0 + 0.5 * (n - 10) for n in range(10, 17)])
    assert table["molecular_ion"].all() and "7 alkane" in note
    assert [comps[k].name for k in table["component"]] == [f"C{n} alkane" for n in range(10, 17)]


def test_alkane_ladder_in_unit_mass_spectra():
    """nominal masses, every ion with its 13C peak one mass above: the isotope peak of C(n)H(2n+1) sits on the
    molecular mass of the alkane with n carbons, and must not name the compound"""
    lad = [alkane(n, 1.0 + 0.5 * (n - 10), 1e5) for n in range(10, 17)]
    unit = []
    for c in lad:
        nm = np.floor(c.mz + 0.3)
        unit.append(Compound(c.name, c.rt, c.height, np.concatenate([nm, nm + 1.0]), np.concatenate([c.rel, 0.12 * c.rel])))
    rt, off, mz, ab, comps = _csr(unit)
    table, note = RI.find_ladder(rt, off, mz, ab, "alkane", unit=True)
    assert table["carbon"].tolist() == list(range(10, 17)) and table["molecular_ion"].all()
    assert RI.find_ladder(rt, off, mz, ab, "alkane")[0].empty          # accurate-mass lookup finds none of them
    assert RI._unit(57.0699) == 57.0 and RI._unit(562.64) == 562.0     # n - 0.3 <= m < n + 0.7


def test_alkane_without_molecular_ion_takes_its_place_in_the_given_ladder():
    lad = [alkane(n, 1.0 + 0.5 * (n - 10), 1e5) for n in range(10, 15)]
    c12 = lad[2]
    c12.mz, c12.rel = c12.mz[:-1], c12.rel[:-1]                      # no molecular ion for C12
    rt, off, mz, ab, _ = _csr(lad)
    table, _ = RI.find_ladder(rt, off, mz, ab, "alkane", carbons=[10, 11, 12, 13, 14])
    assert table["carbon"].tolist() == [10, 11, 12, 13, 14]
    assert table["molecular_ion"].tolist() == [True, True, False, True, True]
    only_named, _ = RI.find_ladder(rt, off, mz, ab, "alkane")       # no list given: the unnamed one is left out
    assert only_named["carbon"].tolist() == [10, 11, 13, 14]


def test_missing_member_without_labels_is_reported_uncertain():
    lad = [alkane(n, 1.0 + 0.5 * (n - 10), 1e5) for n in (10, 11, 13, 14)]
    for c in lad:
        c.mz, c.rel = c.mz[:-1], c.rel[:-1]
    rt, off, mz, ab, _ = _csr(lad)
    table, note = RI.find_ladder(rt, off, mz, ab, "alkane", carbons=[10, 11, 12, 13, 14])
    assert len(table) == 4 and "not certain" in note


def test_fame_ladder_uses_the_fiehn_index_values():
    carbons = [8, 9, 10, 12, 14, 16]
    lad = [fame(n, 2.0 + 0.4 * k, 2e5) for k, n in enumerate(carbons)]
    rng = np.random.default_rng(3)
    rt, off, mz, ab, _ = _csr(lad + [Compound("X", 2.9, 5e5, *random_spectrum(rng, 20))])
    table, _ = RI.find_ladder(rt, off, mz, ab, "fame")
    assert table["carbon"].tolist() == carbons
    assert table["ri"].tolist() == [262320, 323120, 381020, 487220, 582620, 668720]


def test_too_few_ladder_compounds_give_no_ladder():
    rt, off, mz, ab, _ = _csr([alkane(10, 1.0, 1e5), alkane(11, 1.5, 1e5)])
    table, note = RI.find_ladder(rt, off, mz, ab, "alkane")
    assert len(table) == 0 and "only 2" in note


def test_read_ladder_table(tmp_path):
    f = tmp_path / "ladder.tsv"
    f.write_text("carbon\trt_s\n10\t60\n11\t90\n12\t120\n")
    t = RI.read_ladder(f, "alkane")
    assert t["ri"].tolist() == [1000.0, 1100.0, 1200.0] and np.allclose(t["rt"], [1.0, 1.5, 2.0])
    g = tmp_path / "ladder.csv"
    g.write_text("RT,RI\n1.0,1000\n2.0,1100\n")
    assert RI.read_ladder(g)["ri"].tolist() == [1000, 1100]
    with pytest.raises(ValueError):
        RI.read_ladder(f)                                            # carbon numbers need the ladder kind
