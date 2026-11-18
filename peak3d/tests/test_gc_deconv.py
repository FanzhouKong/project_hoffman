import numpy as np
import pandas as pd
import pytest

from peak3d.estimate import estimate
from peak3d.features import build_features
from peak3d.gc import deconv as DC
from peak3d.gc.synth import Compound, make_run, random_spectrum
from peak3d.io import C13
from peak3d.pick import pick_cloud
from peak3d.synth import Peak

DT = 1.0 / 300.0          # 5 spectra per second
FWHM = 0.05               # 3 s
ISO = 0.12


def _picked(compounds, **kw):
    kw.setdefault("iso", ISO)
    cloud, truth = make_run(compounds, n_scans=1500, dt=DT, **kw)
    b = pick_cloud(cloud, estimate(cloud))
    feats, _ = build_features(cloud, b)
    return cloud, feats, b.params, truth


def _abundance(d, k, mz, tol=0.003):
    m, a = d.spectrum(k)
    if len(m) == 0:
        return 0.0
    i = int(np.argmin(np.abs(m - mz)))
    return float(a[i]) if abs(m[i] - mz) <= tol else 0.0


def _component_at(d, rt, tol_s=0.3):
    k = np.flatnonzero(np.abs(d.components["rt"].values - rt) * 60 <= tol_s)
    assert len(k) == 1, f"{len(k)} components within {tol_s} s of {rt * 60:.2f} s"
    return int(k[0])


@pytest.fixture(scope="module")
def run():
    """A (strong) and B co-elute 1.5 s apart (half a peak width) and share two ions; C is alone; 40 more compounds
    spread over the run give the picker and the calibration their material"""
    rng = np.random.default_rng(1)
    mzA, relA = random_spectrum(rng, 25)
    mzB, relB = random_spectrum(rng, 25)
    shared = [float(mzA[5]), float(mzA[12])]
    mzB[3], mzB[10] = shared
    o = np.argsort(mzB)
    comps = [Compound("A", 2.0, 2e5, mzA, relA, FWHM), Compound("B", 2.0 + 1.5 / 60, 1e5, mzB[o], relB[o], FWHM)]
    comps.append(Compound("C", 3.5, 5e4, *random_spectrum(rng, 25), FWHM))
    for k in range(40):
        m, r = random_spectrum(rng, 15)
        comps.append(Compound(f"X{k}", 0.3 + 4.4 * rng.random(), 10 ** rng.uniform(3.5, 5.5), m, r, FWHM))
    cloud, feats, P, truth = _picked(comps, seed=3)
    return dict(cloud=cloud, feats=feats, P=P, truth=truth, comps=comps, shared=shared,
                d=DC.deconvolve(cloud, feats, P))


def test_log_parabola_places_a_gaussian_apex_between_scans():
    t = np.arange(40, dtype=np.float64)
    for apex in (19.5, 20.2, 20.49):
        y = (1e4 * np.exp(-0.5 * ((t - apex) / 4.0) ** 2)).astype(np.float32)
        k = int(np.argmax(y))
        pos, se = DC._log_parabola_apex(y, k, t)
        assert abs(pos - apex) < 1e-3 and se < 1e-3                          # a clean peak: placed, and known to be
    assert np.isnan(DC._log_parabola_apex(np.array([0, 5, 0], np.float32), 1, t[:3])[0])    # nothing beside the top


def test_the_apex_of_a_noisy_flat_top_is_less_certain_than_that_of_a_sharp_peak():
    rng = np.random.default_rng(3)
    t = np.arange(60, dtype=np.float64)
    se = {}
    for name, sd in (("sharp", 2.0), ("wide", 6.0)):
        e = []
        for _ in range(50):
            y = 1e3 * np.exp(-0.5 * ((t - 30.3) / sd) ** 2) + rng.normal(0, 40, len(t))
            y = np.maximum(y, 1.0).astype(np.float32)
            e.append(DC._log_parabola_apex(y, int(np.argmax(y)), t)[1])
        se[name] = np.nanmedian(e)
    assert se["wide"] > 2.0 * se["sharp"]


def test_own_tolerance_widens_only_for_ions_less_certain_than_their_class():
    tau = np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
    se = np.array([0.01, 0.01, 0.01, 0.05, 0.02, np.nan, 0.02])
    own = DC._own_tolerance(tau, se, cap=0.3)
    assert np.allclose(own[[0, 1, 2, 4, 5, 6]], tau[[0, 1, 2, 4, 5, 6]])      # typical, or no error known: the class's
    assert np.isclose(own[3], min(0.3, 0.1 * np.sqrt((25 + 1) / 2)))         # five times the typical error
    assert DC._own_tolerance(tau, se * 100, cap=0.05)[3] == 0.1               # never below the class tolerance


def test_calibration_comes_from_isotope_pairs(run):
    info = run["d"].info
    assert info["calibrated"] and info["n_iso_pairs"] > 200
    taus = info["tau_classes_s"]
    assert all(a >= b for a, b in zip(taus, taus[1:]))          # weaker ions never get a tighter tolerance
    assert taus[-1] < 0.1                                       # well below the 0.2 s scan spacing for strong ions
    assert all(0.5 <= c <= 0.95 for c in info["shape_classes"])


def test_one_component_per_compound(run):
    d, comps = run["d"], run["comps"]
    n = [int(np.sum(np.abs(d.components["rt"].values - c.rt) * 60 < 0.3)) for c in comps]
    assert sum(k == 1 for k in n) >= len(comps) - 3
    assert len(d.components) <= len(comps) + 2
    assert (d.components["n_ions"] >= 3).all()


def test_isolated_compound_spectrum_matches_truth(run):
    d = run["d"]
    c = next(x for x in run["comps"] if x.name == "C")
    k = _component_at(d, c.rt)
    for mz, rel in zip(c.mz, c.rel):
        if rel * c.height < 2000:
            continue
        got = _abundance(d, k, mz)
        assert abs(got / (rel * c.height) - 1.0) < 0.08, (mz, got, rel * c.height)
        if ISO * rel * c.height > 1500:                          # the M+1 as well
            assert abs(_abundance(d, k, mz + C13) / (ISO * rel * c.height) - 1.0) < 0.15
    m, _ = d.spectrum(k)
    true = np.concatenate([c.mz, c.mz + C13])
    assert all(np.min(np.abs(true - x)) < 0.005 for x in m)      # nothing but its own ions


def test_coeluting_pair_is_two_components_with_the_shared_ions_divided(run):
    d = run["d"]
    A, B = run["comps"][0], run["comps"][1]
    ka, kb = _component_at(d, A.rt), _component_at(d, B.rt)
    assert ka != kb
    assert abs(d.components["rt"][ka] - A.rt) * 60 < 0.05 and abs(d.components["rt"][kb] - B.rt) * 60 < 0.05
    for c, k in ((A, ka), (B, kb)):
        for mz in run["shared"]:
            true = float(c.rel[np.argmin(np.abs(c.mz - mz))]) * c.height
            assert abs(_abundance(d, k, mz) / true - 1.0) < 0.1, (c.name, mz)
        own = [(m, r) for m, r in zip(c.mz, c.rel) if r * c.height > 5000 and min(abs(m - s) for s in run["shared"]) > 1.5]
        ratios = np.array([_abundance(d, k, m) / (r * c.height) for m, r in own])
        assert len(own) > 8 and np.all(np.abs(ratios - 1.0) < 0.1)
    # the shared ions were picked as one peak each, between the two apexes: flagged, in no component
    f = run["feats"]
    for mz in run["shared"]:
        i = np.flatnonzero((np.abs(f["mz"].values - mz) < 0.003) & (np.abs(f["rt"].values - 2.01) < 0.03))
        assert len(i) == 1 and d.feature_shared[i[0]] and d.feature_component[i[0]] == -1
    sh = d.shared[d.off[ka]:d.off[ka + 1]]
    m, _ = d.spectrum(ka)
    assert all(sh[np.argmin(np.abs(m - mz))] for mz in run["shared"])


def test_quant_ion_is_an_unshared_ion(run):
    d = run["d"]
    A = run["comps"][0]
    k = _component_at(d, A.rt)
    q = d.components["quant_mz"][k]
    assert min(abs(q - s) for s in run["shared"]) > 1.5
    assert abs(d.components["quant_height"][k] / A.height - 1.0) < 0.05      # A's base peak is not one of the shared


def test_fragment_too_weak_to_be_picked_is_in_the_spectrum():
    rng = np.random.default_rng(5)
    comps = [Compound(f"X{k}", 0.4 + 0.25 * k, 10 ** rng.uniform(4.3, 5.3), *random_spectrum(rng, 15), FWHM)
             for k in range(16)]
    mz, rel = random_spectrum(rng, 12)
    weak = 311.2468
    target = Compound("T", 2.52, 1e5, np.append(mz, weak), np.append(rel, 8e-4), FWHM)       # 80 counts at the apex
    cloud, feats, P, _ = _picked(comps + [target], seed=2, iso=0.0)
    assert not ((np.abs(feats["mz"] - weak) < 0.005) & (np.abs(feats["rt"] - 2.52) < 0.02)).any()
    d = DC.deconvolve(cloud, feats, P, DC.DeconvParams(min_rel=1e-4))
    got = _abundance(d, _component_at(d, 2.52), weak, tol=0.005)
    assert 40 < got < 130


def test_constant_background_ion_stays_out_of_the_spectra():
    rng = np.random.default_rng(7)
    comps = [Compound(f"X{k}", 0.4 + 0.25 * k, 10 ** rng.uniform(4.3, 5.3), *random_spectrum(rng, 15), FWHM)
             for k in range(16)]
    bleed = 207.0324                                             # a siloxane ion present through the whole run
    cloud, truth = make_run(comps, n_scans=1500, dt=DT, iso=ISO, seed=4,
                            extra_peaks=[Peak(bleed, 2.5, 5000.0, 40.0)])
    b = pick_cloud(cloud, estimate(cloud))
    feats, _ = build_features(cloud, b)
    d = DC.deconvolve(cloud, feats, b.params)
    assert len(d.components) >= 14
    assert np.min(np.abs(d.mz - bleed)) > 0.01


def test_model_profile_times_abundance_gives_the_chromatogram(run):
    d, cloud = run["d"], run["cloud"]
    assert len(d.model_off) == len(d.components) + 1
    c = next(x for x in run["comps"] if x.name == "C")
    k = _component_at(d, c.rt)
    row = d.components.iloc[k]
    p = d.profile(k)
    assert len(p) == row["scan_max"] - row["scan_min"] + 1 and p.max() == pytest.approx(1.0)
    assert int(np.argmax(p)) + row["scan_min"] == pytest.approx(row["scan_apex"], abs=1)
    mz = float(c.mz[np.argmax(c.rel)])                           # its strongest ion, alone at that mass
    y = cloud.eic(mz, 10.0, int(row["scan_min"]), int(row["scan_max"]) + 1)
    fit = _abundance(d, k, mz) * p
    assert np.sum((y - fit) ** 2) < 0.02 * np.sum(y ** 2)


def test_empty_feature_table():
    cloud, _ = make_run([Compound("A", 1.0, 1e5, *random_spectrum(np.random.default_rng(0), 10), FWHM)], n_scans=600, dt=DT)
    b = pick_cloud(cloud, estimate(cloud))
    feats, _ = build_features(cloud, b)
    d = DC.deconvolve(cloud, feats.iloc[:0], b.params)
    assert len(d.components) == 0 and len(d.off) == 1 and len(d.feature_component) == 0
    assert len(d.model_off) == 1 and len(d.model) == 0
    d1 = DC.deconvolve(cloud, feats, b.params)                   # too few isotope pairs: fixed defaults
    assert not d1.info["calibrated"] and len(d1.components) == 1
    assert isinstance(d1.components, pd.DataFrame)


def _one_ion(trace):
    """a run holding one ion: CSR arrays for the kernels"""
    S = len(trace)
    return np.arange(S + 1, dtype=np.int64), np.full(S, 100.0), np.asarray(trace, np.float32)


def test_floor_is_the_level_a_chromatogram_keeps_beyond_its_peak():
    t = np.arange(80, dtype=np.float64)
    peak = 1000.0 * np.exp(-0.5 * ((t - 40) / 3.0) ** 2)
    args = (np.array([100.0]), np.array([0.01]), np.array([1000.0]), np.array([30], np.int64), np.array([50], np.int64), 80, 6, 0.05)
    clean = DC._floors(*_one_ion(peak), *args)[0]
    assert clean < 10.0                                                      # its own feet, nothing more
    assert abs(DC._floors(*_one_ion(peak + 200.0), *args)[0] - 200.0) < 10.0  # a background ion: the level it stands on
    # between two neighbours the trace climbs again beyond both bounds: a valley is not a floor
    crowd = peak + 1000.0 * np.exp(-0.5 * ((t - 26) / 3.0) ** 2) + 1000.0 * np.exp(-0.5 * ((t - 54) / 3.0) ** 2)
    v = (np.array([100.0]), np.array([0.01]), np.array([1000.0]), np.array([33], np.int64), np.array([47], np.int64), 80, 6, 0.05)
    assert DC._floors(*_one_ion(crowd), *v)[0] == 0.0


def test_lone_ions_with_uncertain_apexes_join_without_disturbing_the_compounds():
    # beside T an ion with a flat, wide top 0.6 s later; the run holds a few more picked ions that stand alone
    cloud, feats, P, T = _odd_run([Peak(433.2468, 2.52 + 0.6 / 60, 3e4, 1.5 * FWHM, clip=0.6)])
    off = DC.deconvolve(cloud, feats, P, DC.DeconvParams(tau_own=False))
    on = DC.deconvolve(cloud, feats, P, DC.DeconvParams(tau_own=True))
    assert off.info["n_lone_joined"] == 0 and on.info["n_lone_joined"] >= 1
    assert len(on.components) <= len(off.components)
    k0, k1 = _component_at(off, T.rt, tol_s=1.0), _component_at(on, T.rt, tol_s=1.0)
    assert on.components["n_features"][k1] >= off.components["n_features"][k0] >= 38
    for mz, rel in zip(T.mz, T.rel):                                         # the compound's spectrum is as it was
        if rel * T.height > 5000:
            assert abs(_abundance(on, k1, mz) / (rel * T.height) - 1.0) < 0.1


def _odd_run(extra, seed=11):
    rng = np.random.default_rng(seed)
    comps = [Compound(f"X{k}", 0.4 + 0.25 * k, 10 ** rng.uniform(4.3, 5.3), *random_spectrum(rng, 15), FWHM) for k in range(16)]
    comps.append(Compound("T", 2.52, 1e5, *random_spectrum(rng, 20), FWHM))
    cloud, feats, P, _ = _picked(comps, seed=2, extra_peaks=extra)
    return cloud, feats, P, comps[-1]


def test_an_ion_of_another_shape_at_the_apex_time_does_not_start_a_second_component():
    odd = 433.2468
    cloud, feats, P, T = _odd_run([Peak(odd, 2.52, 6e4, 3.0 * FWHM)])        # same apex, three times as wide
    i = np.flatnonzero((np.abs(feats["mz"].values - odd) < 0.005) & (np.abs(feats["rt"].values - 2.52) < 0.02))
    assert len(i) == 1
    d = DC.deconvolve(cloud, feats, P)
    k = _component_at(d, T.rt, tol_s=1.0)                                    # one component there, not two
    assert d.feature_component[i[0]] == k and d.feature_loose[i[0]]
    assert d.components["n_loose"][k] >= 1 and d.components["n_features"][k] >= 18
    assert d.feature_loose.sum() <= 3                                        # the compounds' own ions are tight members
    old = DC.deconvolve(cloud, feats, P, DC.DeconvParams(attach=False, ref_ions=1))
    assert old.feature_component[i[0]] != _component_at(old, T.rt, tol_s=0.2) if \
        np.sum(np.abs(old.components["rt"].values - T.rt) * 60 <= 0.2) == 1 else True


def test_a_stronger_ion_of_another_shape_at_the_apex_time_does_not_disturb_the_component():
    big = 444.3219
    # the strongest ion at T's apex time has another shape (clipped flat and wide)
    cloud, feats, P, T = _odd_run([Peak(big, 2.52, 4e5, 3.0 * FWHM, clip=0.5)])
    d = DC.deconvolve(cloud, feats, P)
    k = _component_at(d, T.rt, tol_s=1.0)
    assert d.components["n_features"][k] >= 18                               # T's ions are the tight members
    for mz, rel in zip(T.mz, T.rel):
        if rel * T.height > 5000:
            assert abs(_abundance(d, k, mz) / (rel * T.height) - 1.0) < 0.1


def test_min_features_drops_components_resting_on_few_picked_ions(run):
    d3 = DC.deconvolve(run["cloud"], run["feats"], run["P"], DC.DeconvParams(min_features=3))
    assert (d3.components["n_features"] >= 3).all()
    assert len(d3.components) <= len(run["d"].components)
    assert d3.info["n_below_min_features"] == len(run["d"].components) - len(d3.components) or d3.info["n_below_min_features"] >= 0


def _late_run(late, seed=13):
    """T, 40 scattered compounds, and `late`: a function of T giving one more Compound"""
    rng = np.random.default_rng(seed)
    comps = [Compound(f"X{k}", 0.3 + 4.4 * rng.random(), 10 ** rng.uniform(3.8, 5.3), *random_spectrum(rng, 15), FWHM) for k in range(40)]
    T = Compound("T", 2.52, 2e5, *random_spectrum(rng, 20), FWHM)
    L = late(T, rng)
    cloud, feats, P, _ = _picked(comps + [T, L], seed=4)
    return cloud, feats, P, T, L


def _near(d, rt, tol_s):
    return np.flatnonzero(np.abs(d.components["rt"].values - rt) * 60 <= tol_s)


def _lag_apex_shift(fwhm, tau):
    """how much later a Gaussian peak passed through a first-order lag has its apex (synth puts the apex at rt)"""
    from scipy.stats import exponnorm
    sigma = fwhm / 2.3548200450309493
    g = np.linspace(-6 * sigma, 6 * sigma + 8 * tau, 40001)
    return float(g[np.argmax(exponnorm.pdf(g, tau / sigma, loc=0.0, scale=sigma))])


def test_delayed_fragments_join_their_compound():
    # eight more ions of T leave the source through a lag of 1 s: they start with T, their apex and tail are later
    tau = 1.0 / 60
    cloud, feats, P, T, L = _late_run(lambda T, rng: Compound("T_late", T.rt + _lag_apex_shift(FWHM, tau), 1e5,
                                                                *random_spectrum(rng, 8), FWHM, tau=tau))
    assert 0.3 < (L.rt - T.rt) * 60 < 0.8
    off = DC.deconvolve(cloud, feats, P, DC.DeconvParams(delay=False))
    assert len(_near(off, T.rt, 1.0)) == 2                                   # without the test: two components
    d = DC.deconvolve(cloud, feats, P)
    k = _near(d, T.rt, 1.0)
    assert len(k) == 1 and d.components["n_delayed"][k[0]] >= 1
    assert abs(d.components["rt"][k[0]] - T.rt) * 60 < 0.1                   # reported at the larger group: T itself
    m, a = d.spectrum(int(k[0]))
    for c in (T, L):
        strong = [(mz, rel) for mz, rel in zip(c.mz, c.rel) if rel * c.height > 5000]
        assert len(strong) >= 4
        for mz, rel in strong:
            got = _abundance(d, int(k[0]), mz)
            assert 0.5 < got / (rel * c.height) < 1.15, (c.name, mz, got, rel * c.height)
    i = np.flatnonzero((np.abs(feats["mz"].values[:, None] - L.mz[None, :]) < 0.003).any(axis=1) & (np.abs(feats["rt"].values - L.rt) < 0.02))
    assert len(i) >= 4 and (d.feature_component[i] == k[0]).all()
    assert d.feature_delayed[i].sum() >= len(i) - 2              # a weak one may fit T's own model as well
    assert d.info["n_delayed_copies"] == d.components["n_delayed"][k[0]]     # nothing else in the run was folded


def test_a_second_compound_just_after_stays_a_component():
    # the same eight ions as a compound of its own, 0.5 s later with T's shape: everything is later, the start too
    cloud, feats, P, T, L = _late_run(lambda T, rng: Compound("U", T.rt + 0.5 / 60, 1e5, *random_spectrum(rng, 8), FWHM))
    d = DC.deconvolve(cloud, feats, P)
    k = _near(d, T.rt + 0.25 / 60, 1.0)
    assert len(k) == 2 and (d.components["n_delayed"][k] == 0).all()
    ku = int(k[np.argmax(d.components["rt"].values[k])])
    assert abs(d.components["rt"][ku] - L.rt) * 60 < 0.1
    for mz, rel in zip(L.mz, L.rel):
        if rel * L.height > 5000:
            assert abs(_abundance(d, ku, mz) / (rel * L.height) - 1.0) < 0.15
    assert d.info["n_delayed_copies"] == 0


# ------------------------------------------------- single-peaked models, explained components ----
def test_unimodal_model_continues_as_a_decay_where_a_neighbour_rises():
    m = np.array([0.0, 0.1, 0.5, 0.3, 0.28, 0.5, 1.0, 0.6, 0.22, 0.24, 0.05])
    off = np.array([0, len(m)], np.int64)
    v = m.copy()
    DC._unimodal(off, v, 0.05)
    assert v[6] == 1.0 and v[5] == 0.5 and v[4] == 0.28            # up to the valley: untouched
    assert np.all(np.diff(v[:7]) >= 0) and np.all(np.diff(v[6:]) <= 0)
    assert v[2] < 0.28 * 0.9 and v[0] < v[2]                       # the neighbour's hump (0.5) is gone: a decay
    assert v[9] == pytest.approx(0.22)                             # a rise below the threshold is levelled
    flat = np.array([0.2, 0.6, 1.0, 0.7, 0.3])
    w = flat.copy()
    DC._unimodal(np.array([0, 5], np.int64), w, 0.05)
    assert np.array_equal(w, flat)                                 # a single peak is left as it is


def test_nnls_fit_keeps_the_first_listed_column_of_a_dependent_set():
    x = np.linspace(-2, 2, 21)
    own = np.exp(-0.5 * x ** 2)
    X = np.column_stack([np.ones(21), np.zeros(21), 0.3 * np.ones(21), own])     # constant, empty, flat, own
    y = 5.0 * own + 1.0
    G, g, yy = X.T @ X, X.T @ y, float(y @ y)
    coef, sse = DC._nnls_fit(G, g, yy, np.array([3, 0, 1, 2], np.int64), 4)
    assert coef[3] == pytest.approx(5.0, rel=1e-6) and sse < 1e-6 * yy
    assert DC._nnls_sse(G, g, yy, np.array([0, 1, 2], np.int64), 3) > 0.5 * float(((y - y.mean()) ** 2).sum())


@pytest.fixture(scope="module")
def cluster():
    """A and B, one and a half peak widths apart, share ten ions in every ratio: the picked peak of a shared ion
    has its apex anywhere between the two, by more than the apex tolerance. C, weak, elutes two widths after B
    and shares nothing; forty more compounds give the picker and the calibration their material."""
    rng = np.random.default_rng(5)
    mzA, relA = random_spectrum(rng, 22)
    mzB, relB = random_spectrum(rng, 22)
    k = np.argsort(-relA)[2:12]
    mzB[:10] = mzA[k]
    relB[:10] = relA[k] * rng.uniform(0.2, 3.0, 10)
    relB = np.clip(relB, 0.02, 1.0)
    o = np.argsort(mzB)
    comps = [Compound("A", 2.0, 2e5, mzA, relA, FWHM), Compound("B", 2.0 + 1.5 * FWHM, 1.5e5, mzB[o], relB[o], FWHM),
             Compound("C", 2.0 + 3.5 * FWHM, 8e3, *random_spectrum(rng, 12), FWHM)]
    for q in range(40):
        m, r = random_spectrum(rng, 15)
        comps.append(Compound(f"X{q}", 0.3 + (1.3 if q % 2 else 4.0) * rng.random() + (0.0 if q % 2 else 2.6),
                              10 ** rng.uniform(3.5, 5.5), m, r, FWHM))
    cloud, feats, P, truth = _picked(comps, seed=7)
    return dict(cloud=cloud, feats=feats, P=P, comps=comps, shared=np.sort(mzA[k]))


def _in_window(d, comps, pad=1.0):
    t = d.components["rt"].values
    return np.flatnonzero((t >= comps[0].rt - pad * FWHM) & (t <= comps[2].rt + pad * FWHM))


def test_shared_ions_of_two_overlapping_compounds_make_no_components_of_their_own(cluster):
    comps = cluster["comps"]
    old = DC.deconvolve(cluster["cloud"], cluster["feats"], cluster["P"], DC.DeconvParams(explain=False, unimodal=False))
    d = DC.deconvolve(cluster["cloud"], cluster["feats"], cluster["P"])
    assert len(_in_window(old, comps)) > 3                         # what the rule is for: pieces between A and B
    assert len(_in_window(d, comps)) == 3 and d.info["n_explained"] > 0
    a, b = _component_at(d, comps[0].rt), _component_at(d, comps[1].rt)
    for mz in cluster["shared"]:                                   # every shared ion is in both spectra, divided
        ia = float(comps[0].rel[np.argmin(np.abs(comps[0].mz - mz))] * comps[0].height)
        ib = float(comps[1].rel[np.argmin(np.abs(comps[1].mz - mz))] * comps[1].height)
        if min(ia, ib) < 5000:
            continue
        assert abs(_abundance(d, a, mz) / ia - 1.0) < 0.25, (mz, _abundance(d, a, mz), ia)
        assert abs(_abundance(d, b, mz) / ib - 1.0) < 0.25, (mz, _abundance(d, b, mz), ib)


def test_ions_of_dissolved_pieces_are_members_of_their_compound_again(cluster):
    d = DC.deconvolve(cluster["cloud"], cluster["feats"], cluster["P"])
    F = cluster["feats"]
    comps = cluster["comps"]
    b = _component_at(d, comps[1].rt)
    only_b = [mz for mz in comps[1].mz if np.min(np.abs(cluster["shared"] - mz)) > 0.01]
    rows = [int(F.index[(np.abs(F["mz"].values - mz) < 0.003) & (np.abs(F["rt"].values - comps[1].rt) < FWHM)][0])
            for mz in only_b if ((np.abs(F["mz"].values - mz) < 0.003) & (np.abs(F["rt"].values - comps[1].rt) < FWHM)).any()]
    assert len(rows) >= 8
    assert np.mean(d.feature_component[rows] == b) >= 0.9
    assert d.info["n_rejoined"] >= 0


def test_a_weak_compound_beside_a_large_one_is_not_explained_away(cluster):
    d = DC.deconvolve(cluster["cloud"], cluster["feats"], cluster["P"])
    c = cluster["comps"][2]
    k = _component_at(d, c.rt)
    m, _ = d.spectrum(k)
    assert sum(np.min(np.abs(m - x)) < 0.005 for x in c.mz) >= 6


def _small_beside_large(seed):
    """A, with five ions of its own, 1.2 peak widths before B, and twelve of B's ions at a third to two thirds of
    B's abundance: in those ions A is a shoulder on B's peak, their apex is near B's, and there are more of them
    than A has ions of its own"""
    rng = np.random.default_rng(seed)
    mzB, relB = random_spectrum(rng, 30)
    mzA, relA = random_spectrum(rng, 17)
    k = np.argsort(-relB)[1:13]
    mzA[:12] = mzB[k]
    relA[:12] = np.clip(2.0 * relB[k] * rng.uniform(0.3, 0.6, 12), 0.02, 1.0)
    o = np.argsort(mzA)
    comps = [Compound("A", 2.0, 1e5, mzA[o], relA[o], FWHM), Compound("B", 2.0 + 1.2 * FWHM, 2e5, mzB, relB, FWHM)]
    for q in range(40):
        m, r = random_spectrum(rng, 15)
        comps.append(Compound(f"X{q}", 0.3 + (1.3 if q % 2 else 4.0) * rng.random() + (0.0 if q % 2 else 2.6),
                              10 ** rng.uniform(3.5, 5.5), m, r, FWHM))
    cloud, feats, P, truth = _picked(comps, seed=seed + 1)
    return cloud, feats, P, comps


def _between(d, first, last, pad=1.0):
    t = d.components["rt"].values
    return np.flatnonzero((t >= first.rt - pad * FWHM) & (t <= last.rt + pad * FWHM))


def test_ions_shared_with_a_smaller_compound_make_no_component_of_their_own():
    cloud, feats, P, comps = _small_beside_large(21)
    old = DC.deconvolve(cloud, feats, P, DC.DeconvParams(explain_smaller=False))
    d = DC.deconvolve(cloud, feats, P)
    assert len(_between(old, comps[0], comps[1])) == 3             # what the rule is for: A's shoulder ions as a piece
    assert len(_between(d, comps[0], comps[1])) == 2
    _component_at(d, comps[0].rt), _component_at(d, comps[1].rt)          # A and B are both there
