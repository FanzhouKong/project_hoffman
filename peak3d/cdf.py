"""ANDI-MS netCDF (.cdf) input, the usual export of GC-MS instruments (scipy reads it: netCDF classic).

Two kinds of file come in this container:

- unit-resolution scans (quadrupoles): one value per nominal mass, written with a mass jitter of 0.1-0.3 Da on a
  0.05 or 0.1 Da grid. The picker works on accurate masses, so each scan is reduced to nominal masses first:
  * nominal mass n holds every point with n - 0.3 <= m < n + 0.7 (the unit-mass rule of gc/library.py and of the
    NIST libraries); the intensities of one scan in one bin are summed;
  * points below 3 x the root mean square of the file's negative intensities are dropped (some exports hold every
    nominal mass of every scan, noise included, with the baseline subtracted so that noise is negative as often as
    positive; nothing is dropped from a file without negative points);
  * the scan skew is taken out: a quadrupole measures the masses of one scan one after the other, so the
    chromatogram of each mass is sampled at its own times, t_scan + K x dt x m / 1000 (K: scans per 1000 Da).
    K is measured (estimate_skew: the apex of every ion of the strongest peaks, to a fraction of a scan, against
    its mass) and every mass chromatogram is moved to the scan times by cubic interpolation, the middle of the
    mass range staying where it is. A run uses one K for all its files, the median of the files' own (study_skew):
    one study is one instrument method, and a single file's value can be off (0.5 to 0.7 in four of the five
    fatty-acid runs of the DeepCPR paper, -0.6 in the fifth);
  * the m/z of the point cloud is n x 1.0033548 (the 13C - 12C distance): M and M+1 of a fragment are then exactly
    one 13C apart, as on an accurate-mass instrument, so the picker annotates them as an isotope pair and the GC
    deconvolution can calibrate its rules on them. Cloud.mass_scale records the factor; the GC stages divide by
    it, so compounds, spectra and library searches are at nominal masses. The feature tables of single ions keep
    the scaled m/z.
- accurate-mass centroids (time-of-flight instruments): read as they are.

Which of the two a file is, is read from its masses (is_unit_resolution)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import C13, Cloud, from_scans

CDF = ".cdf"
BIN_SHIFT = 0.3            # nominal mass n holds n - BIN_SHIFT <= m < n + 1 - BIN_SHIFT
GRID = 0.05                # Da: unit-resolution exports write their masses on this grid or a coarser one
GRID_SHARE = 0.98          # ... at least this share of them
SKEW_MIN_PEAKS = 10        # a scan skew measured on fewer peaks is not applied
SKEW_MAX_SPAN = 1.2        # ... nor one that would have the mass range take more than this many scan intervals


def is_cdf(path) -> bool:
    return Path(path).suffix.lower() == CDF


def read_cdf(path):
    """(rt in seconds, scan offsets, m/z, intensity) of an ANDI-MS file"""
    from scipy.io import netcdf_file
    with netcdf_file(str(path), "r", mmap=False) as nc:
        v = nc.variables
        missing = [k for k in ("scan_acquisition_time", "scan_index", "mass_values", "intensity_values") if k not in v]
        if missing:
            raise ValueError(f"{Path(path).name}: not an ANDI-MS file (no {', '.join(missing)})")
        rt = np.array(v["scan_acquisition_time"][:], np.float64)
        idx = np.array(v["scan_index"][:], np.int64)
        mz = np.array(v["mass_values"][:], np.float64)
        it = np.array(v["intensity_values"][:], np.float64)
    off = np.concatenate([idx, [len(mz)]]).astype(np.int64)
    return rt, off, mz, it


def polarity(path) -> str:
    """"pos" or "neg" from the file's test_ionization_polarity (electron ionisation: positive when not given)"""
    from scipy.io import netcdf_file
    with netcdf_file(str(path), "r", mmap=False) as nc:
        p = getattr(nc, "test_ionization_polarity", b"")
    p = p.decode("ascii", "replace") if isinstance(p, bytes) else str(p)
    return "neg" if "neg" in p.lower() else "pos"


def is_unit_resolution(mz) -> bool:
    """whether these masses are unit-resolution values: written on a 0.05 Da grid (or coarser). Accurate masses
    fall on such a grid by chance only (about 1 in 12)."""
    mz = np.asarray(mz, np.float64)
    if len(mz) == 0:
        return False
    return float(np.mean(np.abs(mz / GRID - np.round(mz / GRID)) < 0.04)) >= GRID_SHARE


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
    """nominal masses of m/z values written with a mass scale"""
    return np.rint(np.asarray(mz, np.float64) / scale).astype(np.int64)


def noise_floor(it):
    """3 x rms of the negative intensities (0 without any)"""
    neg = it[it < 0]
    return 3.0 * float(np.sqrt(np.mean(neg ** 2))) if len(neg) else 0.0


def file_skew(path) -> tuple[float, int]:
    """(K, peaks it rests on) measured in one file; (0, 0) for an accurate-mass file"""
    rt, off, mz, it = read_cdf(path)
    if not is_unit_resolution(mz) or len(rt) < 8:
        return 0.0, 0
    masses, X = unit_matrix(rt, off, mz, np.where(it >= noise_floor(it), it, 0.0))
    k, n = measure_skew(X, masses)
    return float(k), int(n)


def study_skew(values) -> float:
    """one K for the files of a run, from their (K, peaks): the median of the files measured on at least
    SKEW_MIN_PEAKS peaks; 0 when there is none"""
    ks = [k for k, n in values if n >= SKEW_MIN_PEAKS]
    return float(np.median(ks)) if ks else 0.0


def unit_cloud(name, rt, off, mz, it, scale=C13, skew="auto", floor="auto", pol="pos"):
    """the Cloud of a unit-resolution run (module text) and what was done: (Cloud, {noise_floor, scan_skew,
    skew_peaks, n_points_in}). skew: K, or "auto" to measure it in the file (not applied when it rests on fewer
    than SKEW_MIN_PEAKS peaks or is larger than a scan can be); floor: counts, or "auto"."""
    floor = noise_floor(it) if floor == "auto" else float(floor)
    it = np.where(it >= floor, it, 0.0)
    masses, X = unit_matrix(rt, off, mz, it)
    n_skew = 0
    if skew == "auto":
        skew, n_skew = measure_skew(X, masses)
        if n_skew < SKEW_MIN_PEAKS or abs(skew) * (masses[-1] - masses[0]) / 1000.0 > SKEW_MAX_SPAN:
            skew = 0.0
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
    cloud = from_scans(name, rt / 60.0, mzs, its, pol)
    cloud.mass_scale = float(scale)
    return cloud, dict(noise_floor=float(floor), scan_skew=skew, skew_peaks=int(n_skew), n_points_in=int(len(mz)))


def load_cloud(path, name: str | None = None, skew="auto") -> Cloud:
    """Read an ANDI-MS file into a Cloud (module text). cloud.source says what was done: kind ("unit resolution"
    or "accurate mass"), and for a unit-resolution file its noise floor and scan skew. skew: the K to take out of a
    unit-resolution file (a run's study_skew), or "auto" to measure it in this file alone."""
    path = Path(path)
    rt, off, mz, it = read_cdf(path)
    if len(rt) > 1 and np.any(np.diff(rt) <= 0):
        raise ValueError(f"{path.name}: scan times are not increasing")
    name = name or path.stem
    pol = polarity(path)
    if is_unit_resolution(mz):
        cloud, info = unit_cloud(name, rt, off, mz, it, skew=skew, pol=pol)
        cloud.source = dict(kind="unit resolution", mass_scale=cloud.mass_scale, **info)
    else:
        cloud = from_scans(name, rt / 60.0, [mz[off[s]:off[s + 1]] for s in range(len(rt))],
                           [it[off[s]:off[s + 1]] for s in range(len(rt))], pol)
        cloud.source = dict(kind="accurate mass", n_points_in=int(len(mz)))
    return cloud
