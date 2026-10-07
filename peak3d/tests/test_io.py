import numpy as np
import pytest

from peak3d.io import Cloud, from_scans
from peak3d.synth import Peak, make_cloud, permute_within_scans


def test_from_scans_sorts_and_drops_zeros():
    rts = [0.0, 0.1, 0.2]
    mzs = [[300.0, 100.0, 200.0], [], [150.0, 150.5]]
    ins = [[1.0, 0.0, 3.0], [], [2.0, np.nan]]
    c = from_scans("t", rts, mzs, ins)
    assert c.n_scans == 3 and c.n_points == 3
    assert list(c.off) == [0, 2, 2, 3]
    assert list(c.mz) == [200.0, 300.0, 150.0]
    assert c.scan_index().tolist() == [0, 0, 2]


def test_validate_rejects_bad_clouds():
    with pytest.raises(ValueError):
        from_scans("t", [0.0, 0.0], [[100.0], [100.0]], [[1.0], [1.0]])  # non-increasing rt
    c = from_scans("t", [0.0, 0.1], [[100.0, 101.0], [100.0]], [[1.0, 1.0], [1.0]])
    c.mz[1] = 99.0  # break within-scan order
    with pytest.raises(ValueError):
        c.validate()


def test_eic_is_per_scan_max_within_ppm():
    c = from_scans("t", [0.0, 0.1, 0.2],
                   [[100.0000, 100.0003, 200.0], [100.0001], [300.0]],
                   [[5.0, 7.0, 1.0], [2.0], [9.0]])
    e = c.eic(100.0, 5.0)
    assert e.tolist() == [7.0, 2.0, 0.0]
    e = c.eic(100.0, 5.0, 1, 3)
    assert e.tolist() == [2.0, 0.0]
    a = c.eic_argmax(100.0, 5.0)
    assert a.tolist() == [1, 3, -1]


def test_npz_roundtrip(tmp_path):
    cloud, _ = make_cloud([Peak(200.0, 1.0, 1e5, 0.05)], n_scans=100, dt=0.02, floor=50, n_noise=200)
    p = tmp_path / "c.npz"
    cloud.save_npz(p)
    d = Cloud.load_npz(p)
    assert d.name == cloud.name and d.polarity == cloud.polarity
    assert np.array_equal(d.mz, cloud.mz) and np.array_equal(d.inten, cloud.inten)
    assert np.array_equal(d.off, cloud.off) and np.array_equal(d.rt, cloud.rt)


def test_permutation_within_scans_is_invisible():
    cloud, _ = make_cloud([Peak(200.0, 1.0, 1e5, 0.05)], n_scans=100, dt=0.02, floor=50, n_noise=500, seed=3)
    d = permute_within_scans(cloud, seed=7)
    assert np.array_equal(d.mz, cloud.mz) and np.array_equal(d.inten, cloud.inten)
