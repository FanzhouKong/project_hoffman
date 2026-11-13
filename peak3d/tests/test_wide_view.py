"""The ion followed through an m/z wander (kernels.wide_view) and the two gates built on it: wide_background (a stretch
of a continuous ion read at a shifted m/z) and plateau_edge (the last piece of a plateau before it drops)."""
from dataclasses import replace

import numpy as np

from peak3d import kernels as K
from peak3d.estimate import estimate
from peak3d.features import build_features
from peak3d.io import Cloud
from peak3d.pick import pick_cloud
from peak3d.synth import Peak, make_cloud

DT = 0.005
FW = 0.05
BG = dict(n_scans=800, dt=DT, thresh=200.0, floor=100.0, n_noise=30000, sig_a=0.5, sig_b=30.0)


def _cloud(points, n_scans=400):
    """a cloud from (scan, m/z, intensity) rows"""
    p = np.asarray(points, float)
    o = np.lexsort((p[:, 1], p[:, 0]))
    p = p[o]
    off = np.searchsorted(p[:, 0], np.arange(n_scans + 1)).astype(np.int64)
    return Cloud("t", DT * np.arange(n_scans, dtype=np.float64), off, p[:, 1].copy(), p[:, 2].astype(np.float32))


def _view(cloud, mz, apex, lo, hi, tol_n=6.0, tol_w=30.0, near=8, far=24):
    q = np.array([mz])
    a = lambda v: np.array([v], np.int64)
    return K.wide_view(cloud.off, cloud.mz, cloud.inten, q, q * tol_n * 1e-6, q * tol_w * 1e-6, a(apex), a(lo), a(hi), a(near), a(far))[0]


def test_wide_view_sees_a_continuous_ion_through_its_wander():
    """an ion at a constant 1e5 in every scan, read 18 ppm high in scans 100-120: within 6 ppm of the shifted m/z the
    stretch has nothing beside it; followed within 30 ppm it is the same level on both sides"""
    M = 200.0
    pts = [(s, M * (1 + (18e-6 if 100 <= s <= 120 else 0.0)), 1e5) for s in range(400)]
    c = _cloud(pts)
    w = _view(c, M * (1 + 18e-6), 110, 100, 120)
    assert w[K.WV_BG] == 1e5 and w[K.WV_LEVEL] == 1e5 and w[K.WV_DIP] == 1e5 and w[K.WV_FOREIGN] == 0.0
    n = _view(c, M * (1 + 18e-6), 110, 100, 120, tol_w=6.0)          # wide = narrow: the usual view
    assert n[K.WV_BG] == 0.0 and n[K.WV_LEVEL] == 0.0


def test_wide_view_plateau_edge_and_a_real_peak():
    """a plateau that ends at scan 200: the piece before the end has the plateau on one side with no dip on the way,
    nothing on the other. A Gaussian peak has neither. One empty scan on the way is not a dip"""
    M = 200.0
    c = _cloud([(s, M, 1e5) for s in range(201) if s != 185])
    w = _view(c, M, 195, 190, 200)
    assert w[K.WV_LEVEL] == 1e5 and w[K.WV_BG] == 0.0 and w[K.WV_DIP] == 1e5
    g = _cloud([(s, M, 1e5 * np.exp(-0.5 * ((s - 200) / 5.0) ** 2) + 1.0) for s in range(400)])
    w = _view(g, M, 200, 190, 210)
    assert w[K.WV_LEVEL] < 0.05 * 1e5 and w[K.WV_BG] < 0.05 * 1e5
    # a peak before a bigger, flat-topped one (a saturated neighbour): the level beyond is above its apex, but the trace
    # falls by more than 20 % in between
    two = _cloud([(s, M, 1e5 * np.exp(-0.5 * ((s - 200) / 3.0) ** 2) + (3e5 if s >= 208 else 0.0) + 1.0) for s in range(400)])
    w = _view(two, M, 200, 194, 206)
    assert w[K.WV_LEVEL] > 1e5 and w[K.WV_DIP] < 0.8 * 1e5


def test_wide_view_reports_a_second_ion_in_the_window():
    """another ion 20 ppm away, present in every scan at 60 % of the peak: it fills the wide window, and the share of
    the peak's scans holding it says so (the gates then leave the candidate alone)"""
    M = 200.0
    pts = [(s, M, 1e5 * np.exp(-0.5 * ((s - 200) / 5.0) ** 2) + 1.0) for s in range(400)]
    pts += [(s, M * (1 + 20e-6), 6e4) for s in range(400)]
    w = _view(_cloud(pts), M, 200, 190, 210)
    assert w[K.WV_FOREIGN] == 1.0 and w[K.WV_BG] == 6e4


def _with_ion(cloud, mz_of_scan, inten_of_scan):
    """the cloud plus one centroid per scan (m/z and intensity as functions of the scan index; intensity 0 = none)"""
    S = cloud.n_scans
    sc = np.repeat(np.arange(S), np.diff(cloud.off))
    add = [(s, mz_of_scan(s), inten_of_scan(s)) for s in range(S) if inten_of_scan(s) > 0]
    pts = np.vstack([np.column_stack([sc, cloud.mz, cloud.inten]), np.asarray(add, float)])
    o = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[o]
    return Cloud(cloud.name, cloud.rt, np.searchsorted(pts[:, 0], np.arange(S + 1)).astype(np.int64), pts[:, 1].copy(),
                 pts[:, 2].astype(np.float32), cloud.polarity)


def _features(cloud, min_score=None, tol_max_ppm=None, **flags):
    P = estimate(cloud, overrides=flags)
    if tol_max_ppm is not None:
        P = replace(P, tol_max_ppm=tol_max_ppm)
    feats, rej = build_features(cloud, pick_cloud(cloud, P), min_score=min_score, keep_rejected=True)
    return feats, rej, P


def _near(feats, mz, lo, hi, ppm=3.0):
    return feats[(np.abs(feats["mz"] - mz) <= ppm * 1e-6 * mz) & (feats["rt"] >= lo) & (feats["rt"] <= hi)]


# In real files these pieces pass the score (ragged edges, a falling flank); on a noise-free synthetic plateau they are
# flat and fail it, so the tests keep every gated candidate (min_score 0) to see the gates themselves. The linking cap
# is set to 15 ppm: the estimate of a file without m/z scatter is the 5 ppm floor, below the 9 ppm shift used here.
KEEP = dict(min_score=0.0, tol_max_ppm=15.0)
OFF = dict(wide_background=False, plateau_edge=False)
M_BG, SHIFT = 250.1, 9e-6


def _background(base, shifted, last_scan=None, seed=3):
    jit = 0.3e-6 * np.random.default_rng(seed).standard_normal(base.n_scans)
    return _with_ion(base, lambda s: M_BG * (1 + jit[s] + (SHIFT if s in shifted else 0.0)),
                     lambda s: 2e5 if last_scan is None or s <= last_scan else 0.0)


def test_wide_background_rejects_a_shifted_stretch_of_a_background_ion():
    """a background ion at a constant 2e5 whose measured m/z reads 9 ppm high for 30 scans (as while a saturating
    compound elutes). Without the gate that stretch is a candidate with nothing beside it; with the gate it is
    rejected, and the real peaks are untouched"""
    base, _ = make_cloud([Peak(400.2, 2.0, 1e6, FW), Peak(500.3, 1.0, 5e4, FW)], **BG)
    cloud = _background(base, set(range(300, 330)))
    off, _, P = _features(cloud, **OFF, **KEEP)
    assert 3.0 * P.sig_ppm(np.array([2e5]))[0] + 1.0 < 1e6 * SHIFT < P.tol_max_ppm          # the case it is built for
    piece = _near(off, M_BG * (1 + SHIFT), 300 * DT, 330 * DT)
    assert len(piece) == 1 and piece["ridge_ratio"].iloc[0] == 0.0, "the shifted stretch must pass without the gate"
    on, rej, _ = _features(cloud, wide_background=True, plateau_edge=False, **KEEP)
    assert len(_near(on, M_BG * (1 + SHIFT), 300 * DT, 330 * DT)) == 0
    r = rej[rej["reason"] == "gate:wide_background"]
    assert len(r) == 1 and abs(r["mz"].iloc[0] / M_BG - 1 - SHIFT) < 2e-6 and r["background_wide"].iloc[0] > 0.9
    for mz, rt in ((400.2, 2.0), (500.3, 1.0)):
        a, b = _near(off, mz, rt - 0.1, rt + 0.1, 5.0), _near(on, mz, rt - 0.1, rt + 0.1, 5.0)
        assert len(a) == len(b) == 1 and a["height"].iloc[0] == b["height"].iloc[0] and a["area"].iloc[0] == b["area"].iloc[0]
    assert len(on) == len(off) - 1
    edge_only, _, _ = _features(cloud, wide_background=False, plateau_edge=True, **KEEP)       # a plateau on both sides and no drop: also an edge
    assert len(_near(edge_only, M_BG * (1 + SHIFT), 300 * DT, 330 * DT)) == 0


def test_plateau_edge_rejects_the_last_piece_of_a_plateau():
    """a background ion at a constant 2e5 that stops at scan 400 (the sample front arrives); ten scans before the last
    eleven are read 9 ppm high. The last piece has the plateau on one side and nothing on the other: a candidate
    without the gate, rejected with it; wide_background alone keeps it (one side is clean)"""
    base, _ = make_cloud([Peak(400.2, 2.0, 1e6, FW), Peak(500.3, 1.0, 5e4, FW)], **BG)
    cloud = _background(base, set(range(380, 390)), last_scan=400, seed=4)
    off, _, _ = _features(cloud, **OFF, **KEEP)
    assert len(_near(off, M_BG, 390 * DT, 401 * DT)) == 1, "the last piece of the plateau must pass without the gate"
    bg_only, _, _ = _features(cloud, wide_background=True, plateau_edge=False, **KEEP)
    assert len(_near(bg_only, M_BG, 390 * DT, 401 * DT)) == 1
    on, rej, _ = _features(cloud, wide_background=False, plateau_edge=True, **KEEP)
    assert len(_near(on, M_BG, 390 * DT, 401 * DT)) == 0
    r = rej[(rej["reason"] == "gate:plateau_edge") & (np.abs(rej["mz"] / M_BG - 1) < 2e-6)]
    assert len(r) == 1 and r["plateau_level"].iloc[0] > 0.9 and r["plateau_dip"].iloc[0] > 0.9 and r["background_wide"].iloc[0] == 0.0
    for mz, rt in ((400.2, 2.0), (500.3, 1.0)):
        assert len(_near(on, mz, rt - 0.1, rt + 0.1, 5.0)) == 1


def test_gates_are_on_by_default_and_leave_clean_peaks_alone():
    cloud, _ = make_cloud([Peak(400.2, 2.0, 1e6, FW, iso=(0.2,)), Peak(500.3, 1.0, 5e4, FW), Peak(300.1, 3.0, 2e5, FW, tau=0.05)], **BG)
    a, _, P = _features(cloud, **OFF)
    assert P.wide_background is False and P.plateau_edge is False
    b, _, Pb = _features(cloud)
    assert Pb.wide_background is True and Pb.plateau_edge is True
    assert len(a) == len(b) and np.array_equal(a["mz"].values, b["mz"].values) and np.array_equal(a["height"].values, b["height"].values)
