import numpy as np
import pytest

from specdenoise import formula as fm
from specdenoise.losses import CAPS, decompose, fmass


def vec(text):
    return fm.parse(text)


def test_parse_format_mass():
    f = vec("C4H9N3O2")
    assert f[fm.C] == 4 and f[fm.H] == 9 and f[fm.N] == 3 and f[fm.O] == 2
    assert fm.format(f) == "C4H9N3O2"
    assert abs(fm.mass(f) - 131.069476) < 1e-5
    assert fm.parse("C4H9N3O2+") is None and fm.parse("[13C]C3H9N3O2") is None and fm.parse("") is None
    assert fm.format(vec("ClH")) == "HCl"


def test_adducts():
    d, n, z = fm.parse_adduct("[M+H]+")
    assert n == 1 and z == 1 and d[fm.H] == 1 and d.sum() == 1
    d, n, z = fm.parse_adduct("[M+H-H2O]+")
    assert d[fm.H] == -1 and d[fm.O] == -1
    d, n, z = fm.parse_adduct("[2M+Na]+")
    assert n == 2 and d[fm.NA] == 1
    d, n, z = fm.parse_adduct("[M-H]-")
    assert z == -1 and d[fm.H] == -1
    d, n, z = fm.parse_adduct("[M+HCOO]-")
    assert d[fm.C] == 1 and d[fm.H] == 1 and d[fm.O] == 2
    d, n, z = fm.parse_adduct("[M+2H]2+")
    assert z == 2 and d[fm.H] == 2
    assert fm.parse_adduct("[M+H+2i]+") is None
    assert fm.parse_adduct("M+H") is None
    p, z = fm.precursor_formula(vec("C4H9N3O2"), "[M+H]+")
    assert abs(fm.ion_mz(p, z) - 132.0768) < 5e-4
    p, z = fm.precursor_formula(vec("C6H12O6"), "[M+NH4]+")
    assert p[fm.N] == 1 and abs(fm.ion_mz(p, z) - 198.0972) < 5e-4
    assert fm.precursor_formula(vec("C6H12O6"), "[M+H-NH3]+") is None      # no N to lose


def test_rules():
    m = lambda t: fmass(vec(t))
    for t in ("H2O", "NH3", "CO", "CO2", "CH3", "HCl", "H3PO4", "SO3", "C2H4O2", "H", "H2", "C6H10O5", "HF", "NaCl", "H6O3", "NH5O", "CH7N"):
        assert fm.loss_ok(vec(t), m(t), CAPS), t
    for t in ("S7", "CH20", "C2O7", "H5", "N2O3S2", "C40H2", "H10O3"):
        assert not fm.loss_ok(vec(t), m(t), CAPS), t
    for t in ("C6H5", "NH4", "PO3", "HSO4", "CN", "CO", "C8H11N", "Na", "C3H8N"):
        assert fm.ion_ok(vec(t), m(t), CAPS), t
    for t in ("H3", "CH9", "C2H2N10", "P5O3S4N2"):
        assert not fm.ion_ok(vec(t), m(t), CAPS), t
    # [M+H]+ of guanidinopropionic acid is even-electron; the neutral is not an ion
    assert fm.precursor_ok(vec("C4H10N3O2"), m("C4H10N3O2"), CAPS, True)
    assert not fm.precursor_ok(vec("C4H9N3O2"), m("C4H9N3O2"), CAPS, True)
    assert fm.precursor_ok(vec("C4H9N3O2"), m("C4H9N3O2"), CAPS, False)
    # small polar [M+H]+ ions outside the Golden Rules' common ranges but inside the extended ones: taurine (H/C 4,
    # O/C 1.5), urea (H/C 5), phosphoserine (O/C 2), hypotaurine
    for t in ("C2H8NO3S", "CH5N2O", "C3H9NO6P", "C2H8NO2S"):
        assert fm.precursor_ok(vec(t), m(t), CAPS, True), t
    # beyond the extended ranges: O/C 3.5, H/C 0.05
    for t in ("C2H8NO7S", "C20HN2"):
        assert not fm.precursor_ok(vec(t), m(t), CAPS, True), t


def brute(target, tol, bounds):
    import itertools
    hits = []
    rng = [range(int(b) + 1) for b in bounds]
    for f in itertools.product(*rng):
        f = np.array(f, np.int64)
        if abs(fmass(f) - target) <= tol:
            hits.append(tuple(f))
    return sorted(hits)


def test_decompose_matches_brute_force():
    bounds = np.zeros(fm.NE, np.int64)
    bounds[[fm.C, fm.H, fm.N, fm.O, fm.S, fm.CL]] = [8, 16, 3, 4, 1, 1]
    out = np.zeros((5000, fm.NE), np.int64)
    for target in (18.010565, 60.0211, 101.1, 150.0, 77.5):
        n = decompose(target, 0.02, bounds, out)
        got = sorted(tuple(r) for r in out[:n])
        assert got == brute(target, 0.02, bounds), target


def test_decompose_water_unique():
    bounds = np.zeros(fm.NE, np.int64)
    bounds[[fm.C, fm.H, fm.N, fm.O]] = [10, 20, 5, 5]
    out = np.zeros((100, fm.NE), np.int64)
    n = decompose(18.010565, 0.002, bounds, out)
    assert n == 1 and fm.format(out[0]) == "H2O"


def test_decompose_overflow():
    bounds = np.zeros(fm.NE, np.int64)
    bounds[[fm.C, fm.H, fm.N, fm.O]] = [30, 60, 10, 10]
    out = np.zeros((3, fm.NE), np.int64)
    assert decompose(300.0, 0.3, bounds, out) == -1
