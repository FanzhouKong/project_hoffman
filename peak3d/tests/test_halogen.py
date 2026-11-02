"""MS1 isotope patterns and the halogen test (peak3d.halogen): theoretical patterns of real compounds, measured
with noise, saturation and interference; extraction from a point cloud."""
import numpy as np
import pytest

from peak3d import halogen
from peak3d.io import C13, Cloud

# formula, has a halogen?, the halogen hypothesis that should win
COMPOUNDS = [
    ("caffeine", {"C": 8, "H": 10, "N": 4, "O": 2}, False, "none"),
    ("glucose", {"C": 6, "H": 12, "O": 6}, False, "none"),
    ("cholesterol", {"C": 27, "H": 46, "O": 1}, False, "none"),
    ("PC 36:4", {"C": 44, "H": 80, "N": 1, "O": 8, "P": 1}, False, "none"),
    ("ATP", {"C": 10, "H": 16, "N": 5, "O": 13, "P": 3}, False, "none"),
    ("methionine", {"C": 5, "H": 11, "N": 1, "O": 2, "S": 1}, False, "S"),
    ("taurine", {"C": 2, "H": 7, "N": 1, "O": 3, "S": 1}, False, "S"),
    ("GSSG", {"C": 20, "H": 32, "N": 6, "O": 12, "S": 2}, False, "S2"),
    ("PFOS", {"C": 8, "H": 1, "F": 17, "O": 3, "S": 1}, False, "S"),        # fluorine is monoisotopic
    ("clofibric acid", {"C": 10, "H": 11, "Cl": 1, "O": 3}, True, "Cl"),
    ("diclofenac", {"C": 14, "H": 11, "Cl": 2, "N": 1, "O": 2}, True, "Cl2"),
    ("2,4-D", {"C": 8, "H": 6, "Cl": 2, "O": 3}, True, "Cl2"),
    ("chloramphenicol", {"C": 11, "H": 12, "Cl": 2, "N": 2, "O": 5}, True, "Cl2"),
    ("triclosan", {"C": 12, "H": 7, "Cl": 3, "O": 2}, True, "Cl3"),
    ("chlorpyrifos", {"C": 9, "H": 11, "Cl": 3, "N": 1, "O": 3, "P": 1, "S": 1}, True, "Cl3"),
    ("4-bromophenol", {"C": 6, "H": 5, "Br": 1, "O": 1}, True, "Br"),
    ("bromocriptine", {"C": 32, "H": 40, "Br": 1, "N": 5, "O": 5}, True, "Br"),
    ("dibromophenol", {"C": 6, "H": 4, "Br": 2, "O": 1}, True, "Br2"),
    ("tribromophenol", {"C": 6, "H": 3, "Br": 3, "O": 1}, True, "Br3"),
    ("bromochlorophenol", {"C": 6, "H": 4, "Br": 1, "Cl": 1, "O": 1}, True, "BrCl"),
]
MASS = dict(halogen.MONO_MASS, P=30.9737620, F=18.9984032)
ISO = dict(halogen._ISOTOPES, P=[(0.0, 1.0)], F=[(0.0, 1.0)])


def theoretical(formula, K=6):
    """(neutral monoisotopic mass, probability per offset, mean mass shift per offset) of a formula"""
    halogen._ISOTOPES.update({e: ISO[e] for e in ("P", "F")})
    try:
        p, m = halogen.pattern(formula, K)
    finally:
        for e in ("P", "F"):
            halogen._ISOTOPES.pop(e, None)
    mass = sum(MASS[e] * n for e, n in formula.items())
    return mass, p / p[0], m


def measured(formula, rng=None, rel_noise=0.0, mass_noise_da=0.0, saturate=1.0, ppm=5.0, K=6):
    """a measured pattern of a [M+H]+ ion: ratios with multiplicative noise, spacings with additive noise, M cut to
    saturate x its height (which inflates every ratio)"""
    mass, rel, shift = theoretical(formula, K)
    rel = rel.copy()
    if rng is not None and rel_noise > 0:
        rel[1:] *= np.exp(rng.normal(0, rel_noise, K - 1))
    if rng is not None and mass_noise_da > 0:
        shift = shift + rng.normal(0, mass_noise_da, K)
    rel[1:] /= saturate
    rel[rel < 0.002] = np.nan          # below detection
    shift = np.where(np.isfinite(rel), shift, np.nan)
    shift[0] = 0.0
    return rel, shift, mass + 1.00728


# ------------------------------------------------------------------ the pattern calculator ----
def test_pattern_single_elements():
    p, m = halogen.pattern({"C": 1})
    assert p[1] / p[0] == pytest.approx(0.0107 / 0.9893) and m[1] == pytest.approx(1.0033548)
    p, m = halogen.pattern({"Cl": 1})
    assert p[2] / p[0] == pytest.approx(0.2424 / 0.7576) and m[2] == pytest.approx(1.99705, abs=1e-5) and p[1] == 0
    p, m = halogen.pattern({"Br": 2})
    assert p[2] / p[0] == pytest.approx(2 * 0.4931 / 0.5069) and p[4] / p[0] == pytest.approx((0.4931 / 0.5069) ** 2)
    p, _ = halogen.pattern({"S": 1})
    assert p[2] / p[0] == pytest.approx(0.0425 / 0.9499)


def test_pattern_known_compounds():
    _, rel, shift = theoretical({"C": 8, "H": 10, "N": 4, "O": 2})       # caffeine: M+1 10.3 %, M+2 0.9 %
    assert rel[1] == pytest.approx(0.103, abs=0.004) and rel[2] == pytest.approx(0.009, abs=0.003)
    _, rel, shift = theoretical({"C": 14, "H": 11, "Cl": 2, "N": 1, "O": 2})   # diclofenac: 100 : 16 : 65 : 10 : 10.5
    assert rel[1] == pytest.approx(0.16, abs=0.01) and rel[2] == pytest.approx(0.65, abs=0.02)
    assert rel[4] == pytest.approx(0.105, abs=0.01)
    assert 1.997 < shift[2] < 1.999           # 37Cl dominates M+2, 13C2 pulls it up a little
    assert shift[2] < 2.0 < 2.0067
    # the mean shift of a pure-13C M+2 is two 13C
    _, _, shift = theoretical({"C": 30})
    assert shift[2] == pytest.approx(2 * C13, abs=1e-4)


# ------------------------------------------------------------------ the halogen test ----
@pytest.mark.parametrize("name,formula,is_hal,expect", COMPOUNDS, ids=[c[0] for c in COMPOUNDS])
def test_exact_patterns(name, formula, is_hal, expect):
    rel, shift, mz = measured(formula)
    r = halogen.halogen_test(rel, shift, mz, z=1, ppm=5.0)
    assert (r["verdict"] == "halogen") == is_hal, (name, r["verdict"], r["best"], r["delta_chi2"])
    assert r["best"] == expect, (name, r["best"], [(d["name"], round(d["chi2"], 1)) for d in r["hypotheses"][:4]])
    assert r["n_c"] == pytest.approx(formula["C"], abs=max(3, 0.1 * formula["C"]))   # N, O taken as averagine


@pytest.mark.parametrize("name,formula,is_hal,expect", COMPOUNDS, ids=[c[0] for c in COMPOUNDS])
def test_noisy_patterns(name, formula, is_hal, expect):
    """8 % ratio noise and 1 mDa spacing noise, 20 draws: the halogen call never flips"""
    rng = np.random.default_rng(7)
    for _ in range(20):
        rel, shift, mz = measured(formula, rng, rel_noise=0.08, mass_noise_da=0.001)
        r = halogen.halogen_test(rel, shift, mz, z=1, ppm=5.0)
        assert (r["verdict"] == "halogen") == is_hal, (name, r["verdict"], r["best"], r["delta_chi2"])
        if is_hal:      # the composition is a hint: BrCl and Cl3 share M+2 spacing and M+4, so only M+2 separates them
            assert r["best"] in ("Cl", "Cl2", "Cl3", "Cl4", "Br", "Br2", "Br3", "BrCl", "Br2Cl", "BrCl2"), (name, r["best"])


def test_intensities_alone_with_coarse_mass_accuracy():
    """a QTOF at 20 ppm: the M+2 spacing is barely informative, the ratios still separate Cl2 from no halogen"""
    rel, shift, mz = measured({"C": 14, "H": 11, "Cl": 2, "N": 1, "O": 2})
    r = halogen.halogen_test(rel, np.full_like(shift, np.nan), mz, ppm=20.0)
    assert r["verdict"] == "halogen" and r["best"] == "Cl2"
    rel, shift, mz = measured({"C": 8, "H": 10, "N": 4, "O": 2})
    r = halogen.halogen_test(rel, np.full_like(shift, np.nan), mz, ppm=20.0)
    assert r["verdict"] == "none"


def test_spacing_separates_sulfur_from_chlorine_when_ratios_agree():
    """a sulfur-rich molecule with a 36 % M+2 (eight S) against one chlorine: ratios alike, spacing tells them apart
    (34S at +1.9958, 37Cl at +1.9970 against 13C2 at +2.0067 on their skeletons) only through the full pattern"""
    rel, shift, mz = measured({"C": 10, "H": 20, "Cl": 1})
    r = halogen.halogen_test(rel, shift, mz, ppm=3.0)
    assert r["best"] == "Cl"
    rel, shift, mz = measured({"C": 10, "H": 20, "S": 2})
    r = halogen.halogen_test(rel, shift, mz, ppm=3.0)
    assert r["verdict"] == "none" and r["best"] == "S2"


def test_saturated_monoisotopic_peak():
    """M clipped to 60 % and 40 %: every ratio is 1.7x / 2.5x too high. Caffeine must not become a halogen call, and
    the carbon cap flags the inflated M+1 once it exceeds what the mass holds; diclofenac keeps a halogen call"""
    for sat in (0.6, 0.4):
        rel, shift, mz = measured({"C": 8, "H": 10, "N": 4, "O": 2}, saturate=sat)
        r = halogen.halogen_test(rel, shift, mz)
        assert r["verdict"] == "none"
        assert any("saturated" in f for f in r["flags"]) == (sat < 0.5)     # 2.5x M+1 is more carbon than 194 Da holds
        rel, shift, mz = measured({"C": 14, "H": 11, "Cl": 2, "N": 1, "O": 2}, saturate=sat)
        r = halogen.halogen_test(rel, shift, mz)
        assert r["verdict"] in ("halogen", "possible") and r["hypotheses"][0]["halogen"]


def test_interfered_window_is_left_out():
    """an unrelated ion in the M+2 window (not co-eluting) would look like chlorine; marked unusable it is ignored"""
    rel, shift, mz = measured({"C": 8, "H": 10, "N": 4, "O": 2})
    rel[2], shift[2] = 0.35, 1.9971
    r = halogen.halogen_test(rel, shift, mz)
    assert r["verdict"] in ("halogen", "possible")
    r = halogen.halogen_test(rel, shift, mz, usable=[True, True, False, True, True, True])
    assert r["verdict"] == "none" and any("M+2 does not co-elute" in f for f in r["flags"])


def test_weak_signal_downgrades():
    rel, shift, mz = measured({"C": 14, "H": 11, "Cl": 2, "N": 1, "O": 2})
    r = halogen.halogen_test(rel, shift, mz, noise_rel=0.3)
    assert r["verdict"] == "possible" and any("weak" in f for f in r["flags"])


def test_doubly_charged():
    """a 2+ ion of bromocriptine: the pattern on the m/z axis is spaced by C13 / 2, the test sees the neutral mass"""
    mass, rel, shift = theoretical({"C": 32, "H": 40, "Br": 1, "N": 5, "O": 5})
    mz = (mass + 2 * 1.00728) / 2
    r = halogen.halogen_test(rel, shift, mz, z=2, ppm=5.0)
    assert r["verdict"] == "halogen" and r["best"] == "Br" and r["n_c"] == pytest.approx(32, abs=2)


def test_json_clean():
    import json
    rel, shift, mz = measured({"C": 6, "H": 5, "Br": 1, "O": 1})
    rel[5] = np.nan
    json.dumps(halogen.halogen_test(rel, shift, mz))


# ------------------------------------------------------------------ extraction from a point cloud ----
def _cloud(ions, n_scans=60, dt=0.01, floor=50.0, seed=0):
    """ions: (mz, rt_min, height, fwhm_min, formula) -> a Cloud with every isotopologue of each ion as its own
    centroid (a resolved spectrum), Gaussian in time, plus a weak flat noise grid"""
    rng = np.random.default_rng(seed)
    rt = np.arange(n_scans) * dt
    mzs, vals, off = [], [], [0]
    for s in range(n_scans):
        m_s, v_s = [rng.uniform(100, 600, 40)], [rng.uniform(floor * 0.5, floor * 1.5, 40)]
        for mz, t0, hgt, fwhm, formula in ions:
            y = hgt * np.exp(-0.5 * ((rt[s] - t0) / (fwhm / 2.3548)) ** 2)
            if y < floor:
                continue
            for shift, p in _fine_pattern(formula):
                if p * y >= floor:
                    m_s.append([mz + shift])
                    v_s.append([p * y])
        m = np.concatenate(m_s)
        v = np.concatenate(v_s)
        o = np.argsort(m)
        mzs.append(m[o])
        vals.append(v[o])
        off.append(off[-1] + len(m))
    return Cloud("syn", rt, np.asarray(off, np.int64), np.concatenate(mzs), np.concatenate(vals).astype(np.float32))


def _fine_pattern(formula, cut=1e-4):
    """[(mass shift, probability)] of every isotopologue above cut, relative to the monoisotopic one"""
    peaks = {0.0: 1.0}
    for el, n in formula.items():
        iso = [(s, a) for s, a in ISO[el] if a > 0]
        for _ in range(int(n)):
            nxt = {}
            for s0, p0 in peaks.items():
                for s, a in iso:
                    k = round(s0 + s, 5)
                    nxt[k] = nxt.get(k, 0.0) + p0 * a
            peaks = {k: v for k, v in nxt.items() if v > cut * 1e-2}
    p0 = peaks[0.0]
    return [(s, p / p0) for s, p in sorted(peaks.items()) if p / p0 >= cut]


def test_extract_pattern_and_feature():
    dic = {"C": 14, "H": 11, "Cl": 2, "N": 1, "O": 2}
    caf = {"C": 8, "H": 10, "N": 4, "O": 2}
    m_dic = sum(MASS[e] * n for e, n in dic.items()) + 1.00728
    m_caf = sum(MASS[e] * n for e, n in caf.items()) + 1.00728
    cloud = _cloud([(m_dic, 0.30, 2e5, 0.06, dic), (m_caf, 0.30, 5e5, 0.06, caf),
                    (m_caf + 2 * C13 + 0.0005, 0.36, 1e6, 0.05, {"C": 10, "H": 20})])  # in caffeine's M+2 window, peaks later
    s0, s1 = 20, 40
    pat = halogen.extract_pattern(cloud, m_dic, 1, s0, s1, ppm=5.0)
    assert pat["apex_scan"] == 30 and pat["top_scans"][0] < 30 < pat["top_scans"][1]
    j0 = pat["k"].index(0)
    assert pat["rel"][j0] == 1.0
    assert pat["rel"][j0 + 2] == pytest.approx(0.65, abs=0.03)     # 37Cl and 13C2 centroids summed
    assert pat["shift"][j0 + 2] == pytest.approx(1.9977, abs=0.0005)
    assert all(pat["usable"][j0:j0 + 5])
    assert len(pat["apex_mz"]) >= 5 and min(pat["apex_mz"]) >= m_dic - 2.3 and max(pat["apex_mz"]) <= m_dic + 5.7
    res = halogen.test_feature(cloud, m_dic, 1, s0, s1, ppm=5.0)
    assert res["test"]["verdict"] == "halogen" and res["test"]["best"] == "Cl2"
    # caffeine: its M+2 window holds an ion that peaks later; the window is unusable and the call stays "none"
    res = halogen.test_feature(cloud, m_caf, 1, s0, s1, ppm=5.0)
    t = res["test"]
    assert t["verdict"] == "none" and not t["observed"]["usable"][2] and any("M+2" in f for f in t["flags"])
    # the M+2 of diclofenac, taken as a feature: the peak two units below is flagged
    res = halogen.test_feature(cloud, m_dic + 1.99705, 1, s0, s1, ppm=5.0)
    assert any("2 mass units below" in f for f in res["test"]["flags"])
    # nothing at an empty m/z
    assert halogen.extract_pattern(cloud, 777.777, 1, s0, s1, ppm=5.0) is None
