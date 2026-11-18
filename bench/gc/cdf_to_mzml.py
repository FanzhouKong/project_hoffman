#!/usr/bin/env python3
"""ANDI-MS netCDF (.CDF) of a unit-resolution GC-MS run -> centroid mzML that peak3d reads.

The DeepCPR benchmark runs (Yu et al., Anal. Chem. 2026, doi 10.1021/acs.analchem.6c04440; amino-acid and fatty-acid
standard mixtures) are quadrupole data: one value per nominal mass, written with a mass jitter of 0.1-0.3 Da.
peak3d expects accurate masses, so each scan is reduced to nominal masses first:

- nominal mass n holds every point with n - 0.3 <= m < n + 0.7 (the unit-mass rule of peak3d/gc/library.py and of
  the NIST libraries); the intensities of one scan in one bin are summed, non-positive values dropped.
- the m/z written is n x --mass-scale. The default scale is the 13C - 12C distance (1.0033548): M and M+1 of a
  fragment are then exactly one 13C apart, as on an accurate-mass instrument, so the picker annotates them as an
  isotope pair and the deconvolution can calibrate its apex-time and shape rules on them. Divide by the scale to
  get the nominal mass back (unit_mass()). --mass-scale 1 writes plain integers.
- --noise-floor T: points below T are dropped before anything else. Some exports hold every nominal mass of every
  scan, noise included, with the baseline subtracted so that noise is negative as often as positive (Shimadzu
  GCMSsolution: 30 % of the points of the fatty-acid runs are negative); others are thresholded by the instrument
  (the Agilent amino-acid runs: nothing below 150 counts). "auto" (the default) sets T to 3 x the root mean
  square of the file's negative intensities, which is 0, i.e. nothing dropped, for a file without negative points.
- --deskew K: a quadrupole measures the masses of one scan one after the other, so the chromatogram of each mass is
  sampled at its own times, t_scan + K x dt x m / 1000 (K = scans per 1000 Da; K > 0 when the scan runs from low
  to high mass, K < 0 from high to low; |K| x mass range / 1000 = the share of the scan interval spent scanning).
  Every mass chromatogram is moved to the scan times by cubic interpolation, the middle of the mass range staying
  where it is. --deskew auto measures K in the file (estimate_skew): at the strongest peaks of the total ion
  current, the apex of every ion to a fraction of a scan against its mass. Default 0: nothing is moved.

    python -I bench/gc/cdf_to_mzml.py IN.CDF OUT.mzML [--mass-scale S] [--deskew K|auto] [--name STEM]

Prints one tab-separated summary line: file, scans, input points, output centroids, m/z range, skew, seconds."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

C13 = 1.0033548378
BIN_SHIFT = 0.3


def read_cdf(path):
    """(rt in seconds, scan offsets, m/z, intensity) of an ANDI-MS file"""
    from scipy.io import netcdf_file
    with netcdf_file(str(path), "r", mmap=False) as nc:
        v = nc.variables
        rt = np.array(v["scan_acquisition_time"][:], np.float64)
        idx = np.array(v["scan_index"][:], np.int64)
        mz = np.array(v["mass_values"][:], np.float64)
        it = np.array(v["intensity_values"][:], np.float64)
    off = np.concatenate([idx, [len(mz)]]).astype(np.int64)
    return rt, off, mz, it


def unit_matrix(rt, off, mz, it, clip=True):
    """(nominal masses, scans x masses matrix): intensities summed per nominal mass, negatives as zero (clip)"""
    nom = np.floor(mz + BIN_SHIFT).astype(np.int64)
    lo, hi = int(nom.min()), int(nom.max())
    X = np.zeros((len(rt), hi - lo + 1), np.float64)
    scan = np.repeat(np.arange(len(rt)), np.diff(off))
    np.add.at(X, (scan, nom - lo), np.maximum(it, 0.0) if clip else it)
    return np.arange(lo, hi + 1), X


def _apex(y, k):
    """sub-scan apex position of y around its maximum k: vertex of the parabola through the logarithms"""
    if k <= 0 or k >= len(y) - 1 or y[k - 1] <= 0 or y[k + 1] <= 0 or y[k] <= 0:
        return np.nan
    a, b, c = np.log(y[k - 1]), np.log(y[k]), np.log(y[k + 1])
    den = a - 2 * b + c
    return k + 0.5 * (a - c) / den if den < 0 else np.nan


def estimate_skew(X, masses, n_peaks=80, min_ions=8, min_span=80.0):
    """K (scans per 1000 Da; see the module text) and the number of peaks it rests on. At each of the strongest
    peaks of the total ion current, every ion of at least 2 % of the peak's base ion whose own maximum lies within
    one scan of the peak gets a sub-scan apex; the slope of apex against mass (weights: square root of the
    height) is taken per peak, and K is the median of the peaks' slopes."""
    tic = X.sum(axis=1)
    S = len(tic)
    order = np.argsort(-tic)
    taken = np.zeros(S, bool)
    slopes = []
    for s in order:
        if len(slopes) >= n_peaks or tic[s] <= 0:
            break
        if taken[max(0, s - 8):s + 9].any() or s < 3 or s >= S - 3:
            continue
        if not (tic[s] >= tic[s - 1] and tic[s] >= tic[s + 1]):
            continue
        taken[s] = True
        row = X[s]
        js = np.flatnonzero(row >= 0.02 * row.max())
        m, t, w = [], [], []
        for j in js:
            y = X[s - 3:s + 4, j]
            k = int(np.argmax(y))
            if abs(k - 3) > 1:
                continue
            a = _apex(y, k)
            if np.isfinite(a):
                m.append(masses[j]); t.append(a); w.append(np.sqrt(y[k]))
        if len(m) < min_ions or max(m) - min(m) < min_span:
            continue
        m, t, w = np.array(m, float), np.array(t), np.array(w)
        mm, tm = np.average(m, weights=w), np.average(t, weights=w)
        den = float((w * (m - mm) ** 2).sum())
        if den > 0:
            slopes.append(1000.0 * float((w * (m - mm) * (t - tm)).sum()) / den)
    # an ion sampled d scans late shows its apex d scans early: the sampling delay is minus the apex slope
    return (-float(np.median(slopes)) if slopes else 0.0), len(slopes)


def measure_skew(X, masses, rounds=3):
    """estimate_skew, repeated on the corrected matrix until the remainder is small: (K, peaks)"""
    k, n = estimate_skew(X, masses)
    for _ in range(rounds - 1):
        dk, n = estimate_skew(deskew(X, masses, k), masses)
        k += dk
        if abs(dk) < 0.02:
            break
    return k, n


def deskew(X, masses, k):
    """every mass chromatogram moved from its own sampling times to the scan times (see the module text)"""
    if k == 0.0 or X.shape[0] < 3:
        return X
    from scipy.interpolate import CubicSpline
    s = np.arange(X.shape[0], dtype=np.float64)
    mid = 0.5 * (masses[0] + masses[-1])
    out = np.empty_like(X)
    for j, m in enumerate(masses):
        if not X[:, j].any():
            out[:, j] = 0.0
            continue
        # the value sampled at s + d is wanted at s: the chromatogram read at s - d. A cubic keeps the peak's
        # height and apex (a straight line between samples flattens it and moves the apex by less than d);
        # where it undershoots next to a peak, or the channel had nothing on either side, the value is 0.
        d = k * (m - mid) / 1000.0
        y = CubicSpline(s, X[:, j])(np.clip(s - d, 0.0, s[-1]))
        lo = np.floor(np.clip(s - d, 0.0, s[-1])).astype(np.int64)
        hi = np.minimum(lo + 1, len(s) - 1)
        y[(X[lo, j] <= 0) & (X[hi, j] <= 0)] = 0.0
        out[:, j] = np.maximum(y, 0.0)
    return out


def unit_mass(mz, scale=C13):
    """nominal masses of m/z values written with --mass-scale"""
    return np.rint(np.asarray(mz, np.float64) / scale).astype(np.int64)


def noise_floor(it):
    """3 x rms of the negative intensities (0 without any)"""
    neg = it[it < 0]
    return 3.0 * float(np.sqrt(np.mean(neg ** 2))) if len(neg) else 0.0


def convert(src, scale=C13, skew=0.0, name=None, floor="auto"):
    """(Cloud, input points, K used, peaks K rests on, noise floor); skew: K, or "auto" to measure it in the file"""
    from peak3d.io import from_scans
    rt, off, mz, it = read_cdf(src)
    floor = noise_floor(it) if floor == "auto" else float(floor)
    it = np.where(it >= floor, it, 0.0)
    masses, X = unit_matrix(rt, off, mz, it)
    n_skew = 0
    if skew == "auto":
        skew, n_skew = measure_skew(X, masses)
    skew = float(skew)
    if skew != 0.0:
        # interpolating next to an empty sample makes values below anything the input held (the instrument's own
        # threshold, or the noise floor above): they are dropped again
        least = float(X[X > 0].min()) if (X > 0).any() else 0.0
        X = deskew(X, masses, skew)
        X[X < least] = 0.0
    mzs, its = [], []
    m_out = masses * scale
    for s in range(len(rt)):
        k = np.flatnonzero(X[s] > 0)
        mzs.append(m_out[k])
        its.append(X[s, k])
    cloud = from_scans(name or Path(src).stem, rt / 60.0, mzs, its, "pos")
    return cloud, len(mz), skew, n_skew, floor


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path, nargs="?", help="omit with --measure-skew")
    ap.add_argument("--noise-floor", default="auto", help="counts, or auto")
    ap.add_argument("--measure-skew", action="store_true", help="print the file's K and stop")
    ap.add_argument("--mass-scale", type=float, default=C13)
    ap.add_argument("--deskew", default="0", help="scans per 1000 Da, or auto")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    if a.measure_skew:
        rt, off, mz, it = read_cdf(a.src)
        floor = noise_floor(it) if a.noise_floor == "auto" else float(a.noise_floor)
        masses, X = unit_matrix(rt, off, mz, np.where(it >= floor, it, 0.0))
        k, n = measure_skew(X, masses)
        print(f"{k:.4f}\t{n}\t{a.src.name}", flush=True)
        return
    from peak3d.synth import write_mzml
    tic = time.perf_counter()
    cloud, n_in, k, n_k, floor = convert(a.src, a.mass_scale, a.deskew if a.deskew == "auto" else float(a.deskew),
                                         a.name or a.dst.stem, a.noise_floor)
    cloud.validate()
    tmp = a.dst.with_name(a.dst.name + ".part")
    write_mzml(cloud, tmp)
    tmp.replace(a.dst)
    print(f"{a.src.name}\t{cloud.n_scans}\t{n_in}\t{cloud.n_points}\t{cloud.mz.min():.3f}-{cloud.mz.max():.3f}\t"
          f"skew {k:.4f} scans/1000 Da ({n_k} peaks)\tfloor {floor:.0f}\t{time.perf_counter() - tic:.1f}", flush=True)


if __name__ == "__main__":
    main()
