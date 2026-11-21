"""Unit-resolution GC-MS runs (2026-11-20): peak-shaped fills of holes at a peak's top, the picker's checks of a
peak's own trace (fill support, top in a hole, detector overload), pieces a hole cut apart joined, and the GC
stage's second pass that joins a compound split across files."""
import numpy as np
import pandas as pd

from peak3d import cdf as C
from peak3d import features as FT
from peak3d import kernels as K
from peak3d.estimate import Params
from peak3d.gc import pipeline as GP
from peak3d.io import Cloud


def gauss(t, a, mu, sd):
    return a * np.exp(-0.5 * ((t - mu) / sd) ** 2)


def test_peak_fill_puts_the_top_inside_the_hole():
    t = np.arange(60, dtype=float)
    y = gauss(t, 1000.0, 30.0, 5.0) + 10.0
    y[27:34] = 0.0
    v = C.peak_fill(y, 27, 34)
    assert v is not None
    assert 29 <= 27 + int(np.argmax(v)) <= 31            # the vertex is inside, near the true apex
    assert v.max() > max(y[26], y[34])                   # above both sides: not a straight ramp
    assert np.allclose(v, gauss(t[27:34], 1000.0, 30.0, 5.0) + 10.0, rtol=0.15)


def test_peak_fill_leaves_a_ramp_to_the_line():
    y = np.linspace(100.0, 500.0, 40)
    y[15:20] = 0.0
    assert C.peak_fill(y, 15, 20) is None                # rises on both sides: no top in the hole
    X = np.stack([y, np.full(40, 1e6), np.full(40, 5.0)], axis=1)   # a strong ion (the 1 % rule holds), a weak one (floor)
    B, n = C.bridge_holes(X, 10)
    assert n == 1 and B[15:20, 0].all()
    assert np.allclose(np.diff(X[14:21, 0]), np.diff(X[14:21, 0])[0])     # the straight line


def test_bridge_holes_fills_a_hidden_top_with_a_peak():
    t = np.arange(80, dtype=float)
    X = np.stack([gauss(t, 2000.0, 40.0, 6.0) + 50.0, np.full(80, 1e7)], axis=1)
    X[37:44, 0] = 0.0
    B, n = C.bridge_holes(X, 20)
    assert n == 1 and B[37:44, 0].all()
    assert 37 <= int(np.argmax(X[:, 0])) <= 43           # the apex lies in the filled stretch


def _cloud(traces, rt, filled=None, scale=C.C13):
    """a unit-resolution cloud: traces = {nominal mass: values per scan (0 = no centroid)}"""
    masses = sorted(traces)
    mzs, its, fl = [], [], []
    for s in range(len(rt)):
        m = [k * scale for k in masses if traces[k][s] > 0]
        i = [traces[k][s] for k in masses if traces[k][s] > 0]
        f = [bool(filled[k][s]) if filled and k in filled else False for k in masses if traces[k][s] > 0]
        mzs.append(np.array(m)); its.append(np.array(i, np.float32)); fl.append(np.array(f, bool))
    off = np.concatenate([[0], np.cumsum([len(m) for m in mzs])]).astype(np.int64)
    c = Cloud("x", np.asarray(rt, float), off, np.concatenate(mzs), np.concatenate(its).astype(np.float32))
    c.mass_scale = scale
    c.bridged = np.concatenate(fl) if filled else None
    return c


def test_unit_trace_tells_a_fill_made_ramp_from_a_peak_with_a_filled_top():
    S = 60
    rt = np.arange(S) / 600.0
    t = np.arange(S, dtype=float)
    peak = gauss(t, 5e5, 30.0, 4.0) + 1e4
    fill_top = np.zeros(S, bool); fill_top[28:33] = True
    ramp = np.zeros(S); ramp[20] = 4e4; ramp[21:30] = np.linspace(4.2e4, 6e4, 9); ramp[30] = 6.2e4
    fill_ramp = np.zeros(S, bool); fill_ramp[21:30] = True
    c = _cloud({100: peak, 200: ramp}, rt, filled={100: fill_top, 200: fill_ramp})
    U = K.unit_trace(c.off, c.mz, c.inten, c.rt, c.bridged, np.array([100 * C.C13, 200 * C.C13]), np.full(2, 0.3),
                     np.array([15, 20]), np.array([45, 30]), np.array([30, 30]))
    assert U[0, K.UT_CLIMB_L] > 0.3 * 5e5 and U[0, K.UT_CLIMB_R] > 0.3 * 5e5 and U[0, K.UT_HOLE_APEX] == 1
    assert U[1, K.UT_NMEAS] == 2 and U[1, K.UT_CLIMB_R] == 0        # two lone centroids and a line between them
    P = Params(fwhm_med=4.0 / 600.0, fwhm_scans=9.4, min_scans=5)
    cand = pd.DataFrame({"mz": [100 * C.C13, 200 * C.C13], "scan_lo": [15, 20], "scan_hi": [45, 30], "scan_apex": [30, 30],
                         "height": [peak[30], 6.2e4], "noise": [1e3, 1e3], "n_bridged": [5, 9]})
    U2 = FT.unit_checks(cand, c, P)
    assert U2["fill_ok"].tolist() == [True, False]
    assert U2["hole_apex"].tolist()[0]


def test_overload_scans_and_saturated_flag():
    S = 600
    rt = np.arange(S) / 600.0
    t = np.arange(S, dtype=float)
    traces = {m: np.full(S, 1e4 + m) for m in range(50, 90)}           # 40 masses in every scan
    traces[300] = np.minimum(gauss(t, 2e9, 300.0, 5.0), 1e9) + 10.0      # a huge peak, clipped at the top ...
    traces[310] = gauss(t, 4e8, 300.0, 5.0) + 10.0                      # and one of its ions, a clean Gaussian
    for m in range(70, 90):                                             # ... during which half the masses vanish
        traces[m][290:311] = 0.0
    c = _cloud(traces, rt)
    over, base = FT.overload_scans(c)
    assert over[295:306].all() and not over[:250].any()
    P = Params(fwhm_med=5.0 / 600.0, fwhm_scans=11.8, min_scans=5)
    cand = pd.DataFrame({"mz": [300 * C.C13, 60 * C.C13, 310 * C.C13], "scan_lo": [280, 280, 280],
                         "scan_hi": [320, 320, 320], "scan_apex": [300, 300, 300], "height": [1e9, 1e4 + 60, 4e8],
                         "noise": [10.0, 10.0, 10.0], "n_bridged": [0, 0, 0]})
    # the clipped top is flagged; a weak ion and a strong but undistorted one are not
    assert FT.unit_checks(cand, c, P)["n_sat_unit"].tolist() == [1, 0, 0]


def test_join_hole_pieces_makes_one_peak_of_two():
    S = 60
    rt = np.arange(S) / 600.0
    t = np.arange(S, dtype=float)
    y = gauss(t, 3e5, 30.0, 4.0) + 5e3
    y[28:33] = 0.0                                                       # an unfilled hole across the top
    c = _cloud({120: y}, rt)
    P = Params(fwhm_med=4.0 / 600.0, fwhm_scans=9.4)
    cand = pd.DataFrame({"mz": [120 * C.C13] * 2, "scan_lo": [15, 33], "scan_hi": [27, 45], "scan_apex": [27, 33],
                         "rt_min": [rt[15], rt[33]], "rt_max": [rt[27], rt[45]], "height": [y[27], y[33]],
                         "area": [1.0, 2.0], "area_bsub": [1.0, 2.0], "n_scans": [13, 13], "n_points": [13, 13],
                         "n_gaps": [0, 0], "n_bridged": [0, 0], "hole_apex": [True, True]})
    keep = FT.join_hole_pieces(cand, np.array([True, True]), c, P)
    assert keep.sum() == 1
    k = int(np.flatnonzero(keep)[0])
    assert cand.loc[k, "scan_lo"] == 15 and cand.loc[k, "scan_hi"] == 45 and cand.loc[k, "n_gaps"] == 5
    assert cand.loc[k, "area"] == 3.0
    # a real valley between them (the ion present) is not a hole: nothing joined
    y2 = y.copy(); y2[28:33] = 1e3
    c2 = _cloud({120: y2}, rt)
    cand2 = cand.copy()
    assert FT.join_hole_pieces(cand2, np.array([True, True]), c2, P).sum() == 2


def test_merge_split_joins_complementary_rows_only():
    # four components (one per file): compounds 0 (files 0, 1) and 1 (files 2, 3) are one compound whose spectra
    # differ by an ion each; compound 2 (file 2 only) has another spectrum at the same time and quantifier
    base_mz = np.array([73.0, 147.0, 205.0, 217.0, 319.0])
    spec = [(base_mz, np.array([30, 100, 40, 60, 20.0])), (base_mz, np.array([32, 100, 38, 58, 22.0])),
            (np.append(base_mz, 400.0), np.array([28, 100, 41, 61, 19.0, 90.0])),
            (np.array([73.0, 147.0, 260.0, 333.0, 450.0]), np.array([100, 60, 80, 90, 70.0]))]   # shares 73, 147 only
    mz = np.concatenate([m for m, _ in spec])
    ab = np.concatenate([a for _, a in spec])
    off = np.concatenate([[0], np.cumsum([len(m) for m, _ in spec])]).astype(np.int64)
    w = ab ** 0.6 * mz / 100.0
    gid = np.array([0, 0, 1, 2])
    rt = np.array([10.0, 10.001, 10.002, 10.002])
    file_of = np.array([0, 1, 2, 3])
    tic = np.array([300.0, 290.0, 280.0, 400.0])
    quant = np.array([147.0, 147.0, 147.0, 147.0])
    new, G, joins = GP._merge_split(gid, 3, rt, file_of, tic, quant, off, mz, w, 0.01, 4, 0.003, 10.0)
    assert joins == 1 and G == 2 and new[0] == new[1] == new[2] and new[3] != new[0]
    # the same pair sharing a file stays apart
    new, G, joins = GP._merge_split(gid, 3, rt, np.array([0, 1, 1, 3]), tic, quant, off, mz, w, 0.01, 4, 0.003, 10.0)
    assert joins == 0 and G == 3
