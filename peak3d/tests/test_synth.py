import numpy as np

from peak3d.synth import Peak, make_cloud, profile


def test_profile_apex_and_fwhm():
    t = np.linspace(0, 2, 20001)
    for pk in (Peak(200.0, 1.0, 1.0, 0.1), Peak(200.0, 1.0, 1.0, 0.1, tau=0.05)):
        y = profile(t, pk)
        assert abs(t[np.argmax(y)] - 1.0) < 1e-3
        assert abs(y.max() - 1.0) < 1e-6
    y = profile(t, Peak(200.0, 1.0, 1.0, 0.1))
    above = t[y >= 0.5]
    assert abs((above[-1] - above[0]) - 0.1) < 1e-3


def test_truth_matches_sampled_points():
    cloud, truth = make_cloud([Peak(200.0, 1.0, 1e6, 0.05, iso=(0.1,))], n_scans=400, dt=0.005, thresh=0.0)
    assert len(truth) == 2
    m = truth.iloc[0]
    e = cloud.eic(200.0, 5.0)
    assert abs(e.max() - m.height) < 1e-3 * m.height
    assert abs(cloud.rt[np.argmax(e)] - m.rt) < 1e-9
    assert truth.iloc[1].mz > 201.0


def test_dropout_and_threshold_remove_points():
    full, _ = make_cloud([Peak(200.0, 1.0, 1e4, 0.05)], n_scans=400, dt=0.005, thresh=0.0)
    thr, _ = make_cloud([Peak(200.0, 1.0, 1e4, 0.05)], n_scans=400, dt=0.005, thresh=500.0)
    drop, _ = make_cloud([Peak(200.0, 1.0, 1e4, 0.05)], n_scans=400, dt=0.005, thresh=0.0, dropout_scale=5e3)
    assert thr.n_points < full.n_points and drop.n_points < full.n_points
    assert thr.inten.min() >= 500.0


def test_scan_gaps_remove_grid_points():
    c, _ = make_cloud([Peak(200.0, 1.0, 1e4, 0.05)], n_scans=100, dt=0.01, drop_scans=[10, 11, 12])
    assert c.n_scans == 97
    assert np.isclose(np.diff(c.rt).max(), 0.04)
