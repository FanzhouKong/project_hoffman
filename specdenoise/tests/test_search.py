import numpy as np
import pytest

from specdenoise import Params, denoise
from specdenoise import formula as fm
from specdenoise.search import IonTable, denoise_keys, ion_formula, masked_spectra, unique_keys


def _ion(formula, adduct):
    return fm.precursor_formula(fm.parse(formula), adduct)


def _spectrum(formula, adduct, losses, noise, rng, electronic=0, iso=True):
    p, z = _ion(formula, adduct)
    pmz = fm.ion_mz(p, z)
    mz = [fm.ion_mz(p - fm.parse(l), z) for l in losses] + [pmz - d for d in noise] + [pmz]
    inten = list(rng.uniform(10, 100, len(mz)))
    if iso:
        mz.append(mz[0] + fm.C13)
        inten.append(inten[0] * 0.2)
    for k in range(electronic):           # a group of identical intensities
        mz.append(pmz - 30.0 - 7.3 * k)
        inten.append(5.0)
    mz = np.array(mz) + rng.normal(0, 0.0005, len(mz))
    o = np.argsort(mz)
    return mz[o], np.array(inten)[o], pmz


def test_ion_formula_named_inferred_and_charged():
    p, z = ion_formula("C6H13NO2", "[M+H]+", 1, 132.1019)
    assert fm.format(p) == "C6H14NO2" and z == 1
    p, z = ion_formula("C6H13NO2", "", 1, 154.0838)             # no adduct: [M+Na]+ from the m/z
    assert fm.format(p) == "C6H13NNaO2" and z == 1
    p, z = ion_formula("C6H13NO2", "[M+H]+", 1, 154.0838)       # adduct contradicts the m/z: the m/z wins
    assert fm.format(p) == "C6H13NNaO2"
    p, z = ion_formula("C6H13NO2", None, 2, 130.0874)           # negative mode
    assert fm.format(p) == "C6H12NO2" and z == -1
    p, z = ion_formula("C5H12NO+", "[M]+", 1, 102.0913)         # a charged formula
    assert fm.format(p) == "C5H12NO" and z == 1
    assert ion_formula("", "[M+H]+", 1, 100.0) is None
    assert ion_formula("C6H13NO2", "", 1, float("nan")) is None    # nothing to infer from
    assert ion_formula("C6H13NO2", "", 1, 300.0) is None           # no adduct fits
    assert ion_formula("C6H13NO2", "[M+H]+", 1, float("nan"))[1] == 1   # named adduct, no m/z to check


def test_ion_table_parses_each_text_once_and_dedups():
    formula = ["C6H13NO2", "C6H13NO2", "C6H13NO2", "C5H13N3O", "", "C6H13NO2"]
    adduct = ["[M+H]+", "[M+H]+", "", "[M+H]+", "[M+H]+", "[M+Na]+"]
    mode = np.array([1, 1, 1, 1, 1, 1], np.int8)
    pmz = np.array([132.1019, 132.1019, 132.1019, 132.1131, 132.1, 154.0838])
    t = IonTable(len(formula))
    ids = t.ids(np.array([0, 1, 2, 3, 4]), formula, adduct, mode, pmz)
    assert ids[0] == ids[1] == ids[2] and ids[3] != ids[0] and ids[4] == -1
    assert len(t) == 2 and t.label(ids[0]) == "C6H14NO2+"
    ids2 = t.ids(np.array([5, 0]), formula, adduct, mode, pmz)
    assert ids2[1] == ids[0] and len(t) == 3 and t.label(ids2[0]) == "C6H13NNaO2+"
    assert t.P.shape == (3, fm.NE) and np.allclose(t.mz[ids2[0]], 154.0838, atol=1e-3)


def test_unique_keys_dedup_and_rejects():
    q_mz = np.array([132.1019, 300.0])
    ion_mz = np.array([132.1019, 132.1131, 132.2, 132.1019])
    ion_z = np.array([1, 1, 1, 2])
    pair_q = np.array([0, 0, 0, 0, 0, 0, 1])
    pair_ion = np.array([0, 0, 1, -1, 2, 3, 0])
    kq, ki, pk = unique_keys(pair_q, pair_ion, q_mz, ion_mz, ion_z)
    assert list(zip(kq, ki)) == [(0, 0), (0, 1)]
    assert pk.tolist() == [0, 0, 1, -1, -1, -1, -1]      # no formula, m/z 0.1 off, charge 2, query 300 off


def test_denoise_keys_equals_denoise_per_key():
    rng = np.random.default_rng(3)
    specs = [_spectrum("C6H13NO2", "[M+H]+", ["H2O", "CH2O2", "NH3", "C2H5NO2"], [0.5, 3.37, 11.11], rng, electronic=4),
             _spectrum("C9H8O4", "[M-H]-", ["CO2", "H2O", "C2H2O"], [7.77, 25.3], rng),
             _spectrum("C8H4BrClN2", "[M+H]+", ["HCl", "HBr", "Br"], [2.2], rng)]
    off = np.cumsum([0] + [len(s[0]) for s in specs])
    mz = np.concatenate([s[0] for s in specs])
    inten = np.concatenate([s[1] for s in specs])
    pmz = np.array([s[2] for s in specs])
    keys = [(0, "C6H13NO2", "[M+H]+"), (0, "C5H13N3O", "[M+H]+"),    # the true and an isobaric formula, same query
            (1, "C9H8O4", "[M-H]-"), (2, "C8H4BrClN2", "[M+H]+"), (1, "C10H12O3", "[M-H]-")]
    kq = np.array([k[0] for k in keys])
    P = np.array([_ion(f, a)[0] for _, f, a in keys])
    Z = np.array([_ion(f, a)[1] for _, f, a in keys])
    for params in (Params(), Params(recalibrate=False, exchange=False), Params(isotopes=False, electronic=False)):
        key_off, keep = denoise_keys(off, mz, inten, pmz, kq, P, Z, params)
        for k, (q, f, a) in enumerate(keys):
            s = slice(off[q], off[q + 1])
            r = denoise(mz[s], inten[s], pmz[q], formula=f, adduct=a, params=params)
            if abs(fm.ion_mz(*_ion(f, a)) - pmz[q]) <= 0.02:
                assert r.tier == 1
                assert (keep[key_off[k]:key_off[k + 1]] == r.keep).all(), (k, f)
    key_off, keep = denoise_keys(off, mz, inten, pmz, kq, P, Z)
    assert not keep[key_off[0]:key_off[1]].all() and keep[key_off[0]:key_off[1]].sum() > 4
    assert keep[key_off[1]:key_off[2]].sum() < keep[key_off[0]:key_off[1]].sum()     # the wrong formula keeps less


def test_denoise_keys_needs_sorted_and_handles_empty():
    with pytest.raises(ValueError):
        denoise_keys(np.array([0, 2]), np.array([100.0, 50.0]), np.ones(2), np.array([150.0]), np.array([0]),
                     _ion("C6H13NO2", "[M+H]+")[0][None], np.array([1]))
    key_off, keep = denoise_keys(np.array([0, 0]), np.zeros(0), np.zeros(0), np.array([132.1]), np.zeros(0, int),
                                 np.zeros((0, fm.NE), int), np.zeros(0, int))
    assert key_off.tolist() == [0] and len(keep) == 0


def test_masked_spectra():
    off = np.array([0, 3, 5])
    mz = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    it = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    key_q = np.array([1, 0, 1])
    key_off = np.array([0, 2, 5, 7])
    keep = np.array([True, False, True, True, False, False, True])
    d_off, d_mz, d_it = masked_spectra(off, mz, it, key_q, key_off, keep)
    assert d_off.tolist() == [0, 1, 3, 4] and d_mz.tolist() == [10.0, 1.0, 2.0, 20.0]
    assert d_it.tolist() == [8.0, 5.0, 6.0, 9.0]
