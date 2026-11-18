"""bench/gc/cdf_to_mzml.py: unit-resolution CDF -> centroid mzML (nominal masses, noise floor, scan skew)."""
import numpy as np
import pytest

from bench.gc import cdf_to_mzml as C


def _write_cdf(path, rt, scans):
    """scans: list of (m/z array, intensity array)"""
    from scipy.io import netcdf_file
    mz = np.concatenate([m for m, _ in scans])
    it = np.concatenate([i for _, i in scans])
    idx = np.concatenate([[0], np.cumsum([len(m) for m, _ in scans])[:-1]])
    with netcdf_file(str(path), "w") as nc:
        nc.createDimension("scan_number", len(rt))
        nc.createDimension("point_number", len(mz))
        for name, dim, typ, val in (("scan_acquisition_time", "scan_number", "d", rt), ("scan_index", "scan_number", "i", idx),
                                    ("mass_values", "point_number", "d", mz), ("intensity_values", "point_number", "d", it)):
            v = nc.createVariable(name, typ, (dim,))
            v[:] = val


def _run(n_scans=200, skew=0.0, masses=np.arange(40, 301, 20), dt=0.5, noise=0.0, seed=0):
    """peaks of one compound per 40 scans, every mass sampled skew x m / 1000 scans late"""
    rng = np.random.default_rng(seed)
    s = np.arange(n_scans, dtype=float)
    X = np.zeros((n_scans, len(masses)))
    for c in range(20, n_scans - 10, 40):
        for j, m in enumerate(masses):
            t = s + skew * m / 1000.0
            X[:, j] += 1e4 * (1 + j % 3) * np.exp(-0.5 * ((t - c - 0.3) / 2.0) ** 2)
    if noise:
        X += rng.normal(0, noise, X.shape)
    return s * dt + 60.0, masses, X


def test_unit_matrix_bins_and_sums(tmp_path):
    rt = np.array([10.0, 10.5, 11.0])
    scans = [(np.array([72.8, 73.1, 73.69, 73.7, 100.0]), np.array([5.0, 7.0, 1.0, 2.0, -3.0]))] * 3
    _write_cdf(tmp_path / "a.CDF", rt, scans)
    r, off, mz, it = C.read_cdf(tmp_path / "a.CDF")
    masses, X = C.unit_matrix(r, off, mz, it)
    assert masses[0] == 73 and masses[-1] == 100
    assert X[0, 0] == 13.0            # 72.8, 73.1 and 73.69 are mass 73
    assert X[0, 1] == 2.0             # 73.7 is mass 74
    assert X[0, -1] == 0.0            # a negative point adds nothing
    assert C.unit_matrix(r, off, mz, it, clip=False)[1][0, -1] == -3.0


def test_mass_scale_round_trip():
    n = np.arange(35, 801)
    assert np.array_equal(C.unit_mass(n * C.C13), n)
    assert np.allclose(np.diff(n * C.C13), C.C13)


def test_noise_floor_from_negative_points():
    rng = np.random.default_rng(1)
    it = rng.normal(0, 50, 100000)
    assert C.noise_floor(it) == pytest.approx(150, rel=0.05)
    assert C.noise_floor(np.abs(it) + 150) == 0.0


@pytest.mark.parametrize("skew", [-1.2, 0.0, 0.8])
def test_skew_measured_and_removed(skew):
    rt, masses, X = _run(skew=skew)
    k, n = C.measure_skew(X, masses)
    assert n >= 4
    assert k == pytest.approx(skew, abs=0.08)
    k2, _ = C.estimate_skew(C.deskew(X, masses, k), masses)
    assert abs(k2) < 0.05


def test_convert_writes_scaled_nominal_masses(tmp_path):
    rt, masses, X = _run(skew=-1.0)
    scans = [(masses + 0.1, X[s]) for s in range(len(rt))]
    _write_cdf(tmp_path / "b.CDF", rt, scans)
    cloud, n_in, k, n_k, floor = C.convert(tmp_path / "b.CDF", skew="auto")
    assert floor == 0.0 and k == pytest.approx(-1.0, abs=0.08)
    assert set(C.unit_mass(cloud.mz)) <= set(masses)
    assert cloud.rt[0] == pytest.approx(1.0)         # minutes
    # after the correction every mass peaks in the same scan
    apex = []
    for m in masses[[0, -1]]:
        y = cloud.eic(m * C.C13, 5.0)
        apex.append(int(np.argmax(y[:40])))
    assert apex[0] == apex[1]
