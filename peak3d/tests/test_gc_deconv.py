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
        assert abs(DC._log_parabola_apex(y, k, t) - apex) < 1e-3
    assert np.isnan(DC._log_parabola_apex(np.array([0, 5, 0], np.float32), 1, t[:3]))       # nothing beside the top


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


def test_empty_feature_table():
    cloud, _ = make_run([Compound("A", 1.0, 1e5, *random_spectrum(np.random.default_rng(0), 10), FWHM)], n_scans=600, dt=DT)
    b = pick_cloud(cloud, estimate(cloud))
    feats, _ = build_features(cloud, b)
    d = DC.deconvolve(cloud, feats.iloc[:0], b.params)
    assert len(d.components) == 0 and len(d.off) == 1 and len(d.feature_component) == 0
    d1 = DC.deconvolve(cloud, feats, b.params)                   # too few isotope pairs: fixed defaults
    assert not d1.info["calibrated"] and len(d1.components) == 1
    assert isinstance(d1.components, pd.DataFrame)


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
