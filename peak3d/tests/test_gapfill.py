import numpy as np

from peak3d.gapfill import fill_file
from peak3d.synth import Peak, _trapezoid, make_cloud
from peak3d.warp import TableWarp


def test_fill_window_is_inverse_warped_and_zero_when_absent():
    # the file's native RT runs 0.1 min late: a consensus group at 2.0 sits at native 2.1
    cloud, _ = make_cloud([Peak(400.0, 2.1, 1e5, 0.05)], n_scans=800, dt=0.005, thresh=0.0)
    warp = TableWarp.global_shift(cloud.rt, -0.1)   # native -> corrected
    h, a, _ = fill_file(cloud, warp, np.array([400.0, 400.0, 500.0]), np.array([2.0, 2.6, 2.0]),
                        np.array([0.03, 0.03, 0.03]), ppm_tol=5.0)
    assert h[0] == np.float32(1e5) and a[0] > 0
    assert h[1] == 0 and a[1] == 0           # right m/z, wrong time
    assert h[2] == 0 and a[2] == 0           # nothing at that m/z


def test_fill_area_is_trapezoid_over_present_scans():
    cloud, truth = make_cloud([Peak(400.0, 2.0, 1e5, 0.05)], n_scans=800, dt=0.005, thresh=0.0)
    h, a, _ = fill_file(cloud, TableWarp.identity(cloud.rt), np.array([400.0]), np.array([2.0]), np.array([0.2]), 5.0)
    e = cloud.eic(400.0, 5.0)
    keep = e > 0
    assert abs(a[0] - _trapezoid(e[keep], cloud.rt[keep])) / a[0] < 1e-3
