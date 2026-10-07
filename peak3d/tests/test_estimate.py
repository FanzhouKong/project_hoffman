import numpy as np

from peak3d.estimate import estimate, fwhm_at, strongest_distinct
from peak3d.synth import Peak, make_cloud


def _study(seed=0, n=300, fwhm=0.06, sig_a=0.6, sig_b=40.0, dt=0.005):
    rng = np.random.default_rng(seed)
    peaks = [Peak(float(m), float(r), float(h), fwhm)
             for m, r, h in zip(rng.uniform(100, 900, n), rng.uniform(0.5, 9.5, n), 10 ** rng.uniform(4, 6.5, n))]
    return make_cloud(peaks, n_scans=2000, dt=dt, thresh=300.0, floor=150.0, n_noise=60000, seed=seed,
                      sig_a=sig_a, sig_b=sig_b)


def test_strongest_distinct_are_distinct_in_mz():
    cloud, _ = _study()
    idx = strongest_distinct(cloud, 200)
    mz = np.sort(cloud.mz[idx])
    assert len(idx) == 200
    assert np.all(np.diff(mz) / mz[1:] * 1e6 >= 20.0)


def test_fwhm_at_recovers_width():
    cloud, truth = make_cloud([Peak(300.0, 2.0, 1e6, 0.08)], n_scans=1000, dt=0.005, thresh=0.0)
    scan = int(np.argmax(cloud.eic(300.0, 5.0)))
    w, n_above, k = fwhm_at(cloud, 300.0, scan)
    assert abs(w - 0.08) / 0.08 < 0.05
    assert k == scan


def test_estimate_recovers_sigma_fwhm_and_rules():
    cloud, _ = _study(seed=1, fwhm=0.06, sig_a=0.6, sig_b=40.0)
    P = estimate(cloud)
    assert abs(P.dt - 0.005) < 1e-9
    assert abs(P.fwhm_med - 0.06) / 0.06 < 0.15
    assert abs(P.sig_a - 0.6) / 0.6 < 0.3
    assert abs(P.sig_b - 40.0) / 40.0 < 0.3
    # FWHM = 12 scans -> K capped at 3, min_scans capped at 5
    assert P.K == 3 and P.min_scans == 5
    assert 5.0 <= P.tol_max_ppm <= 30.0


def test_estimate_rules_for_narrow_peaks():
    cloud, _ = _study(seed=2, fwhm=0.02, dt=0.005)   # 4 scans per FWHM
    P = estimate(cloud)
    assert P.K == 2 and P.min_scans == 3
