"""eic_noise="own": a candidate's chromatogram prominence judged against its own trace's noise."""
from dataclasses import replace

import numpy as np
import pandas as pd

from peak3d import kernels as K
from peak3d.estimate import Params, parse_overrides, prepare
from peak3d.features import build_features, eic_check
from peak3d.io import from_scans
from peak3d.pick import pick_cloud

S, DT = 600, 0.01                 # scans, minutes per scan
G_SD = 3.0                        # scans; FWHM about 7 scans, like the MassCube synthetic signals


def gauss(n, mu, sd=G_SD):
    x = np.arange(n)
    return np.exp(-0.5 * ((x - mu) / sd) ** 2)


def cloud_of(signals, seed=0):
    """signals: (m/z, first scan, 50 intensities); one centroid per signal per scan where its intensity > 0"""
    per_mz, per_in = [[] for _ in range(S)], [[] for _ in range(S)]
    for mz, s0, y in signals:
        for i, v in enumerate(y):
            if v > 0:
                per_mz[s0 + i].append(mz)
                per_in[s0 + i].append(v)
    return from_scans("T", np.arange(S) * DT, per_mz, per_in)


def noisy_background(rng, n=60, noise=0.1):
    """n single peaks of height 1e6 with Gaussian noise of noise x 1e6 per point, clipped at 0, m/z 1600-1606"""
    out = []
    for k in range(n):
        y = 1e6 * gauss(50, 24.5) + rng.normal(0, noise * 1e6, 50)
        out.append((1600.0 + 0.1 * k, int(rng.integers(10, S - 60)), np.maximum(y, 0)))
    return out


def double(ratio=4, sep=15.0):
    y = ratio * gauss(50, 24.5 - sep / 2) + gauss(50, 24.5 + sep / 2)
    return y / y.max() * 1e6


def _feats(cloud, P):
    feats, _ = build_features(cloud, pick_cloud(cloud, P))
    return feats


def test_default_unchanged_and_settable():
    assert Params().eic_noise == "max"
    assert parse_overrides(["eic_noise=own"]) == {"eic_noise": "own"}


def test_own_noise_reads_the_trace_not_the_neighbourhood():
    rng = np.random.default_rng(1)
    clean = (1650.0, 100, 1e6 * gauss(50, 24.5))
    noisy = (1650.5, 300, 1e6 * gauss(50, 24.5) + 3e5 + rng.normal(0, 5e4, 50))    # on a raised base: nothing clipped
    short = (1651.0, 450, 1e6 * gauss(8, 3.5, 1.5))
    cl = cloud_of([clean, noisy, short])
    q = np.array([1650.0, 1650.5, 1651.0])
    E = K.eic_shape(cl.off, cl.mz, cl.inten, q, q * 5e-6, np.array([124, 324, 453]), np.array([110, 310, 450]),
                    np.array([140, 340, 457]), 80, K.gaussian_kernel(3.5))
    assert E[0, K.ES_OWN] < 1e3                          # a clean peak reads near 0 however sharp
    assert 3.5e4 < E[1, K.ES_OWN] < 6.5e4                # the noisy trace reads its own noise (sd 5e4)
    assert E[2, K.ES_OWN] == -1                          # 8 scans of signal: too few to measure
    assert E[0, K.ES_NOISE] == 0                         # the old trace estimate: empty scans around read 0


def _candidate(cl, mz, apex, lo, hi, cell_noise):
    return pd.DataFrame({"mz": [mz], "sig_apex": [0.5], "scan_apex": [apex], "scan_lo": [lo], "scan_hi": [hi],
                         "noise": [cell_noise], "height": [float(cl.eic(mz, 5.0, apex, apex + 1)[0])]})


def test_clean_small_peak_of_a_close_double_passes_against_its_own_noise():
    y = double(ratio=4, sep=12.25)                       # the MassCube synthetic double at resolution 1.0, ratio 4:1
    s0 = 200
    cl = cloud_of([(1603.05, s0, y)])
    k2 = 25 + int(np.argmax(y[25:]))                      # small apex
    v = int(np.argmax(y[:25])) + int(np.argmin(y[int(np.argmax(y[:25])):k2]))   # valley
    df = _candidate(cl, 1603.05, s0 + k2, s0 + v, s0 + 49, cell_noise=3e4)   # the synthetic file's cell noise
    P = Params(fwhm_scans=8.6)                           # and its median peak width
    assert eic_check(df, cl, P)["eic_class"].iloc[0] == "lowsnr"
    assert eic_check(df, cl, replace(P, eic_noise="own"))["eic_class"].iloc[0] == "ok"


def test_own_noise_adds_fewer_extra_peaks_on_noisy_singles():
    rng = np.random.default_rng(3)
    cl = cloud_of(noisy_background(rng, n=300, noise=0.1))
    P = prepare(cl)
    mzs = 1600.0 + 0.1 * np.arange(300)

    def extras(f):
        return sum(max(0, int((np.abs(f["mz"].values - m) < 0.01).sum()) - 1) for m in mzs)
    assert extras(_feats(cl, replace(P, eic_noise="own"))) < extras(_feats(cl, P))


# merge_test="own": whether a trace holds two peaks is judged against that trace's own noise

def authors_double(ratio, res):
    """the MassCube synthetic double exactly as the authors' notebook builds it: 50 points on x = 0..4,
    Gaussians of sd 1/4, means 2 -/+ res / 2, the first ratio x the second, scaled to 1e6"""
    x = np.linspace(0, 4, 50)
    y = ratio * np.exp(-0.5 * ((x - (2 - res / 2)) / 0.25) ** 2) + np.exp(-0.5 * ((x - (2 + res / 2)) / 0.25) ** 2)
    return y / y.max() * 1e6


def _widths(rng, n=30):
    """single clean peaks of the same width, so the file's median FWHM is the synthetic one"""
    return [(1700.0 + 0.1 * k, int(rng.integers(10, S - 60)), 1e6 * gauss(50, 24.5, 3.0625)) for k in range(n)]


def test_merge_own_default_unchanged():
    assert Params().merge_test == "frac"
    assert parse_overrides(["merge_test=own"]) == {"merge_test": "own"}


def test_clean_close_double_kept_apart_by_its_own_noise():
    rng = np.random.default_rng(5)
    y = authors_double(4, 1.0)
    k1, k2 = int(np.argmax(y[:25])), 25 + int(np.argmax(y[25:]))
    assert 0.5 < y[k1 + int(np.argmin(y[k1:k2]))] / y[k2] < 0.51      # just above the fixed 0.5 rule
    cl = cloud_of(_widths(rng) + [(1603.05, 250, y)])
    P = prepare(cl)
    near = lambda f: f[np.abs(f["mz"].values - 1603.05) < 0.01]
    assert len(near(_feats(cl, P))) == 1                                   # "frac": merged
    b = pick_cloud(cl, replace(P, merge_test="own"))
    f, _ = build_features(cl, b)
    two = near(f).sort_values("rt")
    assert len(two) == 2 and b.shoulder.sum() >= 1
    assert abs(two["scan_apex"].iloc[0] - (250 + k1)) <= 1 and abs(two["scan_apex"].iloc[1] - (250 + k2)) <= 1


def test_noisy_single_peaks_are_not_split():
    rng = np.random.default_rng(6)
    singles = [(1650.0 + 0.1 * k, int(rng.integers(10, S - 60)), 3e5 + 1e6 * gauss(50, 24.5, 3.0625) + rng.normal(0, 5e4, 50))
               for k in range(40)]
    cl = cloud_of(_widths(rng) + singles)
    P = prepare(cl)

    def extras(f):
        return sum(max(0, int((np.abs(f["mz"].values - m) < 0.01).sum()) - 1) for m, _, _ in singles)
    # a 3-sd dip test splits a few percent of noisy peaks by chance: at most one more split in these 40
    assert extras(_feats(cl, replace(P, merge_test="own"))) <= extras(_feats(cl, P)) + 1


def test_noisy_close_double_kept_apart_and_its_small_peak_kept():
    rng = np.random.default_rng(7)
    sigs = []
    for k in range(10):                                  # ten 4:1 doubles at resolution 1.0 with 2 % noise, clipped at 0
        y = np.maximum(authors_double(4, 1.0) + rng.normal(0, 0.02e6, 50), 0)
        sigs.append((1603.05 + 0.1 * k, 40 + 50 * k, y))
    cl = cloud_of(_widths(rng) + sigs)
    # under narrow_rule "local_median" the frac rule already keeps 7 of these apart and "own" adds a third piece to one
    P = prepare(cl, {"narrow_rule": "median"})
    counts = lambda f: [int((np.abs(f["mz"].values - m) < 0.01).sum()) for m, _, _ in sigs]
    own = counts(_feats(cl, replace(P, merge_test="own")))
    assert sum(c == 2 for c in own) >= 8 and sum(c == 2 for c in own) > sum(c == 2 for c in counts(_feats(cl, P)))
