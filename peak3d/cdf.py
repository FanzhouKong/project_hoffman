"""Unit-resolution GC-MS runs, and ANDI-MS netCDF (.cdf) input, the usual export of GC-MS instruments (scipy reads
it: netCDF classic).

Two kinds of file come as .cdf, and as the mzML files converted from them or from the vendor's data:

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
- accurate-mass centroids (high-resolution time-of-flight and Orbitrap instruments): read as they are.

Which of the two a file is, is read from its masses (mass_kind). Unit-resolution files come in two forms:
  * "grid": the masses sit on a 0.05 Da grid or a coarser one (quadrupole exports);
  * "one per mass": each scan holds one centroid per nominal mass, written to many decimals (unit-resolution
    time-of-flight instruments). Such a value is the centre of everything the instrument saw in that nominal mass, so
    it is no accurate mass: it slides between the mass of the background ion and that of an eluting fragment as their
    proportions change (by 0.05 to 0.15 Da over one chromatographic peak in the 22 cerebrospinal-fluid runs this was
    met on: 17 scans/s, m/z 84-800), and moves again when the detector saturates, while from one scan to the next it
    moves by under 1 mDa. Read as accurate masses, the picker measures that scan-to-scan steadiness, follows each ion
    within a few ppm and cuts a peak wherever its centroid slides away. Two things together make a file this kind:
    - almost no two centroids of one scan are closer than 0.5 Da (near_share at most NEAR_SHARE: 3 to 5 % of the
      neighbours in those runs, half-integer masses of doubly charged ions; 39 to 78 % in accurate-mass files of two
      Q-TOF and three Orbitrap instruments, GC and LC), so nothing the instrument resolved is lost in nominal bins;
    - the masses slide (slide_ratio at least SLIDE_RATIO: the centroid of a nominal mass moves 4 to 6 times further
      over 8 scans than over one in those runs, 0.8 to 1.3 times in the accurate-mass files, where a centroid
      scatters around one mass): a sparse accurate-mass file, one ion per nominal mass, stays accurate.
    These files are binned to nominal masses like the grid form.
load_cloud(masses="unit" / "accurate") overrides the test (`peak3d process --masses`).

mzML files get the same treatment in GC-MS mode (io.load_cloud(unit=...) -> unit_from_scans): the masses of every file
are tested, a unit-resolution file is reduced to nominal masses with the run's scan skew taken out, and an
accurate-mass file is left as it was read. In LC-MS mode an mzML file is never tested (`--masses unit` forces it).

Holes (bridge_holes). In the runs of the "one per mass" kind above, a spectrum recorded while a very strong compound
elutes lacks many of its weaker ions: with a base peak above 1e8 counts a scan holds 100 to 190 centroids where its
neighbours hold 420, an ion at 0.03 to 0.1 % of the base peak is missing in 8 to 16 % of such scans and one below
0.03 % in 22 to 38 %, one above 1 % never; ions lighter than every strong ion of the scan stay (1 % missing against
10 % for heavier ones, which reach the detector after the strong ion). The trace of such an ion stops at full height,
stays empty for about a peak width and comes back at full height: ~3,000 such holes per file, and 9 % of the picked
peaks ended at one. ChromaTOF writes the same centroids to .cdf and to mzML, so the holes are in the export. A hole
is filled with the straight line between the two centroids around it when
  - both of them are at least BRIDGE_FLANK x the file's intensity floor (a trace that ends there did not fade out),
  - every filled value stays under BRIDGE_RATIO of the scan's strongest ion (the scan shows why the ion is missing),
  - the hole is at most BRIDGE_WIDTHS peak widths long (peak_width: the file's own; a longer line would stand for a
    whole peak).
The filled points are marked (Cloud.bridged): a peak that holds any is flagged `bridged` with their number in
`n_bridged`, and the app draws those stretches of a chromatogram dashed. Under a filled stretch the height is a
guess: a peak whose apex lies in a hole comes out too low. bridge=False (`--no-bridge`) leaves the holes.

One scan skew per run is taken out only when the run's files agree on it (study_skew): a scanning instrument shows the
same K in every file of a method (-1.31 to -0.95 and, four files of five, 0.5 to 0.7 in the two quadrupole sets of
the DeepCPR paper), a time-of-flight instrument has none and the measurement returns scatter around zero (-0.87 to
+0.23 over the 22 runs above, the two halves of one file disagreeing in sign)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import C13, Cloud, from_scans

CDF = ".cdf"
BIN_SHIFT = 0.3            # nominal mass n holds n - BIN_SHIFT <= m < n + 1 - BIN_SHIFT
GRID = 0.05                # Da: unit-resolution exports write their masses on this grid or a coarser one
GRID_SHARE = 0.98          # ... at least this share of them
NEAR_DA = 0.5              # two centroids of one scan closer than this are two ions of one nominal mass
NEAR_SHARE = 0.10          # a file with at most this share of such neighbours holds one centroid per nominal mass ...
SLIDE_LAG = 8              # ... when its masses slide: the centroid of a nominal mass moves, over this many scans,
SLIDE_RATIO = 2.0          # at least this many times further than from one scan to the next (medians, in ppm)
SLIDE_FLOOR = 10.0         # on centroids of at least this many times the file's intensity floor (its 1st percentile)
SLIDE_MIN_PAIRS = 1000     # and on at least this many pairs of centroids
SKEW_AGREE = 2.0           # a run's skew is at least this many robust sd of its files' values away from 0
SKEW_AGREE_FILES = 3       # ... asked of runs with at least this many measured files
MASSES = ("auto", "unit", "accurate")
BRIDGE_FLANK = 10.0        # a hole is filled when the centroids on both sides are at least this many times the floor,
BRIDGE_RATIO = 0.01        # every filled value is under this share of its scan's strongest ion,
BRIDGE_WIDTHS = 2.0        # and the hole is at most this many peak widths (FWHM) long
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


def _scan_sorted(mz, off):
    """(scan of every point, order that sorts the points by scan, then mass)"""
    scan = np.repeat(np.arange(len(off) - 1), np.diff(off))
    return scan, np.lexsort((mz, scan))


def near_share(mz, off) -> float:
    """share of the neighbouring centroids within one scan that are closer than NEAR_DA (nan without neighbours)"""
    mz = np.asarray(mz, np.float64)
    off = np.asarray(off, np.int64)
    if len(mz) < 2:
        return float("nan")
    scan = np.repeat(np.arange(len(off) - 1), np.diff(off))
    inside = scan[1:] == scan[:-1]
    if not inside.any():
        return float("nan")
    gaps = np.diff(mz)[inside]
    if (gaps < 0).any():                    # scans not sorted by mass (a Cloud's are): sort them
        scan, o = _scan_sorted(mz, off)
        gaps = np.diff(mz[o])[scan[o][1:] == scan[o][:-1]]
    return float(np.mean(gaps < NEAR_DA))


def slide_ratio(mz, it, off, lag=SLIDE_LAG) -> float:
    """how many times further the centroid of a nominal mass moves over `lag` scans than from one scan to the next:
    the median |difference| (ppm) of the pairs `lag` scans apart over that of consecutive scans, each nominal mass
    of a scan standing with its strongest centroid, pairs of at least SLIDE_FLOOR x the file's intensity floor.
    About 1 for accurate masses (a centroid scatters around its ion's mass), several for the centre of a whole
    nominal mass (module text); nan when there are too few pairs or the masses do not move at all."""
    mz = np.asarray(mz, np.float64)
    it = np.asarray(it, np.float64)
    off = np.asarray(off, np.int64)
    if len(mz) < 2 * SLIDE_MIN_PAIRS:
        return float("nan")
    scan = np.repeat(np.arange(len(off) - 1), np.diff(off))
    nom = np.floor(mz + BIN_SHIFT).astype(np.int64)
    width = int(nom.max() - nom.min()) + 1
    key = scan * width + (nom - nom.min())
    o = np.lexsort((it, key))
    last = np.r_[key[o][1:] != key[o][:-1], True]            # the strongest centroid of each scan x nominal mass
    k, m, y = key[o][last], mz[o][last], it[o][last]
    floor = SLIDE_FLOOR * float(np.percentile(it, 1))

    def step(n):
        want = k + n * width
        j = np.minimum(np.searchsorted(k, want), len(k) - 1)
        ok = (k[j] == want) & (np.minimum(y, y[j]) >= floor)
        return np.abs(m[j[ok]] - m[ok]) / m[ok]

    one, far = step(1), step(lag)
    if min(len(one), len(far)) < SLIDE_MIN_PAIRS:
        return float("nan")
    base = float(np.median(one))
    return float(np.median(far)) / base if base > 0 else float("nan")


def mass_measures(mz, off=None, it=None) -> dict:
    """{kind, near_share, slide_ratio}: what the masses of a file are (module text) and the two measurements behind
    it (None where not taken). kind: "grid" (unit resolution, written on a 0.05 Da grid or coarser; accurate masses
    fall on such a grid by chance only, about 1 in 12), "one per mass" (unit resolution, one sliding centroid per
    nominal mass; needs the scan offsets and the intensities) or "accurate"."""
    mz = np.asarray(mz, np.float64)
    out = dict(kind="accurate", near_share=None, slide_ratio=None)
    if len(mz) == 0:
        return out
    if float(np.mean(np.abs(mz / GRID - np.round(mz / GRID)) < 0.04)) >= GRID_SHARE:
        out["kind"] = "grid"
        return out
    if off is None or it is None:
        return out
    share = near_share(mz, off)
    if not np.isfinite(share):
        return out
    out["near_share"] = round(share, 4)
    if share <= NEAR_SHARE:
        ratio = slide_ratio(mz, it, off)
        if np.isfinite(ratio):
            out["slide_ratio"] = round(ratio, 2)
            if ratio >= SLIDE_RATIO:
                out["kind"] = "one per mass"
    return out


def mass_kind(mz, off=None, it=None) -> str:
    return mass_measures(mz, off, it)["kind"]


def is_unit_resolution(mz, off=None, it=None, masses="auto") -> bool:
    """whether these masses are unit-resolution values (mass_kind), or what the caller says they are (masses:
    "unit" or "accurate")"""
    if masses not in MASSES:
        raise ValueError(f"masses must be one of {', '.join(MASSES)}")
    if masses != "auto":
        return masses == "unit" and len(mz) > 0
    return mass_kind(mz, off, it) != "accurate"


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


SNIFF_SCANS = 200          # an mzML file whose first scans (this many with centroids) show accurate masses is not read on


def read_mzml(path, masses="auto", polarity=None):
    """(rt in minutes or seconds as written, scan offsets, m/z, intensity) of the MS1 scans of an mzML file, for
    file_skew. masses "auto": None as soon as the first SNIFF_SCANS scans with centroids show accurate masses
    (neither on the grid nor one centroid per nominal mass), so that an accurate-mass file costs a fraction of a
    second here; "accurate": None without reading."""
    if masses == "accurate":
        return None
    from pyteomics import mzml
    from .io import _scan_rt_minutes
    want = {"pos": "positive scan", "neg": "negative scan"}.get(polarity)
    rts, mzs, its = [], [], []
    filled, sniffed = 0, masses != "auto"
    with mzml.read(str(path), use_index=False) as reader:
        for s in reader:
            if s.get("ms level") != 1 or (want is not None and want not in s):
                continue
            m = np.asarray(s["m/z array"], np.float64)
            rts.append(_scan_rt_minutes(s))
            mzs.append(m)
            its.append(np.asarray(s["intensity array"], np.float64))
            filled += len(m) > 0
            if not sniffed and filled >= SNIFF_SCANS:
                sniffed = True
                mz = np.concatenate(mzs)
                off = np.concatenate([[0], np.cumsum([len(a) for a in mzs])])
                share = near_share(mz, off)
                if mass_kind(mz) != "grid" and np.isfinite(share) and share > NEAR_SHARE:
                    return None
    if not rts:
        return None
    off = np.concatenate([[0], np.cumsum([len(a) for a in mzs])]).astype(np.int64)
    return np.asarray(rts, np.float64), off, np.concatenate(mzs), np.concatenate(its)


def file_skew(path, masses="auto", polarity=None) -> tuple[float, int]:
    """(K, peaks it rests on) measured in one .cdf or mzML file; (0, 0) for an accurate-mass file"""
    read = read_cdf(path) if is_cdf(path) else read_mzml(path, masses, polarity)
    if read is None:
        return 0.0, 0
    rt, off, mz, it = read
    if not is_unit_resolution(mz, off, it, masses) or len(rt) < 8:
        return 0.0, 0
    masses, X = unit_matrix(rt, off, mz, np.where(it >= noise_floor(it), it, 0.0))
    k, n = measure_skew(X, masses)
    return float(k), int(n)


def study_skew(values) -> float:
    """one K for the files of a run, from their (K, peaks): the median of the files measured on at least
    SKEW_MIN_PEAKS peaks; 0 when there is none, and 0 when the files do not agree on one (SKEW_AGREE_FILES files or
    more whose median is closer to 0 than SKEW_AGREE x their robust sd: module text)"""
    ks = np.array([k for k, n in values if n >= SKEW_MIN_PEAKS], np.float64)
    if not len(ks):
        return 0.0
    med = float(np.median(ks))
    if len(ks) >= SKEW_AGREE_FILES and abs(med) < SKEW_AGREE * 1.4826 * float(np.median(np.abs(ks - med))):
        return 0.0
    return med


def peak_width(X, n_peaks=200) -> float:
    """the file's typical peak width in scans (FWHM): at the strongest peaks of the total ion current (as
    estimate_skew takes them), the width at half height of the scan's strongest ion; the median, 0 when fewer than
    10 peaks could be measured"""
    tic = X.sum(axis=1)
    S = len(tic)
    taken = np.zeros(S, bool)
    widths = []
    for s in np.argsort(-tic):
        if len(widths) >= n_peaks or tic[s] <= 0:
            break
        if taken[max(0, s - 8):s + 9].any() or s < 3 or s >= S - 3:
            continue
        if not (tic[s] >= tic[s - 1] and tic[s] >= tic[s + 1]):
            continue
        taken[s] = True
        y = X[:, int(np.argmax(X[s]))]
        half = 0.5 * y[s]
        a = s
        while a > 0 and y[a - 1] >= half:
            a -= 1
        b = s
        while b < S - 1 and y[b + 1] >= half:
            b += 1
        if a == 0 or b == S - 1 or b - a > 200:
            continue
        left = a - (y[a] - half) / (y[a] - y[a - 1]) if y[a] > y[a - 1] else a
        right = b + (y[b] - half) / (y[b] - y[b + 1]) if y[b] > y[b + 1] else b
        widths.append(right - left)
    return float(np.median(widths)) if len(widths) >= 10 else 0.0


def bridge_holes(X, max_len, flank=BRIDGE_FLANK, ratio=BRIDGE_RATIO):
    """fill the holes of a scans x masses matrix in place (module text): (mask of the filled cells, holes filled).
    max_len: the longest hole filled, in scans."""
    S, M = X.shape
    B = np.zeros(X.shape, bool)
    max_len = int(max_len)
    if max_len < 1 or S < 3 or not (X > 0).any():
        return B, 0
    base = X.max(axis=1)
    level = flank * float(np.percentile(X[X > 0], 1))
    n = 0
    for j in range(M):
        y = X[:, j]
        z = y <= 0
        if z.all() or not z.any():
            continue
        d = np.diff(np.concatenate([[0], z.view(np.int8), [0]]))
        a, b = np.flatnonzero(d == 1), np.flatnonzero(d == -1)        # the empty scans a .. b - 1
        ok = (a > 0) & (b < S) & (b - a <= max_len)
        a, b = a[ok], b[ok]
        ok = np.minimum(y[a - 1], y[b]) >= level
        for s0, s1 in zip(a[ok], b[ok]):
            left, right = y[s0 - 1], y[s1]
            v = left + (right - left) * np.arange(1, s1 - s0 + 1) / (s1 - s0 + 1)
            if np.any(v > ratio * base[s0:s1]):
                continue
            y[s0:s1] = v
            B[s0:s1, j] = True
            n += 1
    return B, n


def unit_cloud(name, rt, off, mz, it, scale=C13, skew="auto", floor="auto", pol="pos", bridge=True):
    """the Cloud of a unit-resolution run (module text) and what was done: (Cloud, {noise_floor, scan_skew,
    skew_peaks, n_points_in, bridge, and with it bridged_holes, bridged_points, bridge_max_scans}). skew: K, or
    "auto" to measure it in the file (not applied when it rests on fewer than SKEW_MIN_PEAKS peaks or is larger than
    a scan can be); floor: counts, or "auto"; bridge: fill the holes (bridge_holes; Cloud.bridged marks the points)."""
    floor = noise_floor(it) if floor == "auto" else float(floor)
    it = np.where(it >= floor, it, 0.0)
    masses, X = unit_matrix(rt, off, mz, it)
    n_skew = 0
    if skew == "auto":
        skew, n_skew = measure_skew(X, masses)
        if n_skew < SKEW_MIN_PEAKS or abs(skew) * (masses[-1] - masses[0]) / 1000.0 > SKEW_MAX_SPAN:
            skew = 0.0
    skew = float(skew)
    B, bridged = None, dict(bridge=bool(bridge))
    if bridge:
        max_len = int(BRIDGE_WIDTHS * peak_width(X))
        B, n_holes = bridge_holes(X, max_len)
        bridged.update(bridged_holes=int(n_holes), bridge_max_scans=max_len)
    if skew != 0.0:
        # interpolating next to an empty sample makes values below anything the input held (the instrument's own
        # threshold, or the noise floor above): they are dropped again
        least = float(X[X > 0].min()) if (X > 0).any() else 0.0
        X = deskew(X, masses, skew)
        X[X < least] = 0.0
    mzs, its, marks = [], [], []
    m_out = masses * scale
    for s in range(len(rt)):
        k = np.flatnonzero(X[s] > 0)
        mzs.append(m_out[k])
        its.append(X[s, k])
        if B is not None:
            marks.append(B[s, k])
    cloud = from_scans(name, rt / 60.0, mzs, its, pol)
    cloud.mass_scale = float(scale)
    if B is not None:
        mark = np.concatenate(marks) if marks else np.zeros(0, bool)
        if len(mark) == cloud.n_points and mark.any():
            cloud.bridged = mark
        bridged["bridged_points"] = int(mark.sum()) if len(mark) == cloud.n_points else 0
    return cloud, dict(noise_floor=float(floor), scan_skew=skew, skew_peaks=int(n_skew), n_points_in=int(len(mz)),
                       **bridged)


def load_cloud(path, name: str | None = None, skew="auto", masses="auto", bridge=True) -> Cloud:
    """Read an ANDI-MS file into a Cloud (module text). cloud.source says what was done: kind ("unit resolution"
    or "accurate mass"), masses (what the file's masses are, mass_kind; with masses="unit" / "accurate" what was
    told and what the file shows), near_share, slide_ratio (where measured), and for a unit-resolution file its
    noise floor, scan skew and what was bridged. skew: the K to take out of a unit-resolution file (a run's
    study_skew), or "auto" to measure it in this file alone; bridge: fill the holes of a unit-resolution run."""
    path = Path(path)
    rt, off, mz, it = read_cdf(path)
    if len(rt) > 1 and np.any(np.diff(rt) <= 0):
        raise ValueError(f"{path.name}: scan times are not increasing")
    name = name or path.stem
    pol = polarity(path)
    if masses not in MASSES:
        raise ValueError(f"masses must be one of {', '.join(MASSES)}")
    seen = mass_measures(mz, off, it)
    kind = seen.pop("kind")
    seen = dict(masses=kind if masses == "auto" else f"{masses} (as told; read from the file: {kind})", **seen)
    if is_unit_resolution(mz, off, it, masses) if masses != "auto" else kind != "accurate":
        cloud, info = unit_cloud(name, rt, off, mz, it, skew=skew, pol=pol, bridge=bridge)
        cloud.source = dict(kind="unit resolution", mass_scale=cloud.mass_scale, **seen, **info)
    else:
        cloud = from_scans(name, rt / 60.0, [mz[off[s]:off[s + 1]] for s in range(len(rt))],
                           [it[off[s]:off[s + 1]] for s in range(len(rt))], pol)
        cloud.source = dict(kind="accurate mass", **seen, n_points_in=int(len(mz)))
    return cloud


def unit_from_scans(cloud: Cloud, mzs, intens, skew="auto", masses="auto", bridge=True) -> Cloud:
    """what load_cloud does for a .cdf file, for a run read from mzML: `cloud` as io.load_cloud built it, mzs and
    intens the scans' arrays as the file holds them (negative intensities included: the noise floor of a
    baseline-subtracted export is measured on them). The masses are tested on the cloud; a unit-resolution run comes
    back as a new Cloud at nominal masses (unit_cloud), an accurate-mass run as the same cloud. cloud.source as in
    load_cloud."""
    if masses not in MASSES:
        raise ValueError(f"masses must be one of {', '.join(MASSES)}")
    seen = mass_measures(cloud.mz, cloud.off, cloud.inten)
    kind = seen.pop("kind")
    seen = dict(masses=kind if masses == "auto" else f"{masses} (as told; read from the file: {kind})", **seen)
    unit = kind != "accurate" if masses == "auto" else masses == "unit"
    if not unit or cloud.n_points == 0:
        cloud.source = dict(kind="accurate mass", **seen, n_points_in=int(cloud.n_points))
        return cloud
    mz = np.concatenate([np.asarray(m, np.float64) for m in mzs])
    it = np.concatenate([np.asarray(i, np.float64) for i in intens])
    off = np.concatenate([[0], np.cumsum([len(m) for m in mzs])]).astype(np.int64)
    bad = ~(np.isfinite(mz) & np.isfinite(it))
    if bad.any():
        mz, it = np.where(bad, mz[~bad][0] if (~bad).any() else 0.0, mz), np.where(bad, 0.0, it)
    out, info = unit_cloud(cloud.name, cloud.rt * 60.0, off, mz, it, skew=skew, pol=cloud.polarity, bridge=bridge)
    out.source = dict(kind="unit resolution", mass_scale=out.mass_scale, **seen, **info)
    return out
