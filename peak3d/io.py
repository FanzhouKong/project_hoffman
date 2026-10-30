"""Point-cloud I/O.

An MS1 run is one scan-major CSR point cloud: the centroids of scan ``s`` are
``mz[off[s]:off[s+1]]`` (ascending m/z) with intensities ``inten[off[s]:off[s+1]]``.
Nothing is binned, smoothed or resampled; the picker works on the centroids as acquired.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

C13 = 1.0033548378  # 13C - 12C


@njit(cache=True)
def _lower_bound(a, lo, hi, v):
    """first index i in [lo, hi) with a[i] >= v (hi if none)"""
    while lo < hi:
        m = (lo + hi) >> 1
        if a[m] < v:
            lo = m + 1
        else:
            hi = m
    return lo


@njit(cache=True)
def _upper_bound(a, lo, hi, v):
    """first index i in [lo, hi) with a[i] > v (hi if none)"""
    while lo < hi:
        m = (lo + hi) >> 1
        if a[m] <= v:
            lo = m + 1
        else:
            hi = m
    return lo


@njit(cache=True)
def eic_kernel(off, mz, inten, target, tol, s0, s1):
    """per-scan max intensity within [target - tol, target + tol] for scans s0..s1-1"""
    out = np.zeros(s1 - s0, np.float32)
    for s in range(s0, s1):
        a = off[s]
        b = off[s + 1]
        lo = _lower_bound(mz, a, b, target - tol)
        hi = _upper_bound(mz, lo, b, target + tol)
        best = np.float32(0.0)
        for p in range(lo, hi):
            if inten[p] > best:
                best = inten[p]
        out[s - s0] = best
    return out


@njit(cache=True)
def eic_argmax_kernel(off, mz, inten, target, tol, s0, s1):
    """index of the most intense centroid within tolerance per scan (-1 if none)"""
    out = np.full(s1 - s0, -1, np.int64)
    for s in range(s0, s1):
        a = off[s]
        b = off[s + 1]
        lo = _lower_bound(mz, a, b, target - tol)
        hi = _upper_bound(mz, lo, b, target + tol)
        best = np.float32(0.0)
        k = -1
        for p in range(lo, hi):
            if inten[p] > best:
                best = inten[p]
                k = p
        out[s - s0] = k
    return out


@dataclass
class Cloud:
    name: str
    rt: np.ndarray        # (S,) float64, minutes, strictly increasing
    off: np.ndarray       # (S+1,) int64 CSR offsets
    mz: np.ndarray        # (N,) float64, ascending within each scan
    inten: np.ndarray     # (N,) float32, > 0
    polarity: str = ""    # "pos", "neg" or "" (unknown)

    @property
    def n_scans(self) -> int:
        return len(self.rt)

    @property
    def n_points(self) -> int:
        return len(self.mz)

    def scan_index(self) -> np.ndarray:
        """scan of every centroid, int32[N]"""
        return np.repeat(np.arange(self.n_scans, dtype=np.int32), np.diff(self.off))

    def eic(self, mz: float, ppm: float, s0: int = 0, s1: int | None = None) -> np.ndarray:
        """max intensity per scan within +/- ppm of mz, for scans s0..s1-1 (dense float32)"""
        if s1 is None:
            s1 = self.n_scans
        return eic_kernel(self.off, self.mz, self.inten, float(mz), float(mz) * ppm * 1e-6, int(s0), int(s1))

    def eic_argmax(self, mz: float, ppm: float, s0: int = 0, s1: int | None = None) -> np.ndarray:
        if s1 is None:
            s1 = self.n_scans
        return eic_argmax_kernel(self.off, self.mz, self.inten, float(mz), float(mz) * ppm * 1e-6, int(s0), int(s1))

    def validate(self) -> None:
        S, N = self.n_scans, self.n_points
        if self.off.shape != (S + 1,) or self.off[0] != 0 or self.off[-1] != N:
            raise ValueError("CSR offsets do not match rt/mz lengths")
        if np.any(np.diff(self.off) < 0):
            raise ValueError("CSR offsets must be non-decreasing")
        if S > 1 and np.any(np.diff(self.rt) <= 0):
            raise ValueError("scan times must be strictly increasing")
        if len(self.inten) != N:
            raise ValueError("mz and inten lengths differ")
        if N and (not np.all(np.isfinite(self.inten)) or np.any(self.inten <= 0)):
            raise ValueError("intensities must be finite and > 0")
        if N and not np.all(np.isfinite(self.mz)):
            raise ValueError("m/z values must be finite")
        if N > 1:
            d = np.diff(self.mz)
            b = self.off[1:-1] - 1          # d[b] crosses a scan boundary and may be negative
            b = b[(b >= 0) & (b < N - 1)]
            d[b] = 0.0
            if np.any(d < 0):
                raise ValueError("m/z must be ascending within each scan")

    def save_npz(self, path) -> None:
        np.savez(path, rt=self.rt, off=self.off, mz=self.mz, inten=self.inten,
                 name=np.array(self.name), polarity=np.array(self.polarity))

    @classmethod
    def load_npz(cls, path) -> "Cloud":
        z = np.load(path, allow_pickle=False)
        return cls(str(z["name"]), z["rt"], z["off"], z["mz"], z["inten"], str(z["polarity"]))


def from_scans(name: str, rts, mzs, intens, polarity: str = "") -> Cloud:
    """Build a Cloud from per-scan arrays. Drops non-positive/non-finite centroids and sorts
    each scan by m/z; scans are kept even when empty (their rt stays on the grid)."""
    rt = np.asarray(rts, dtype=np.float64)
    if len(rt) == 0:
        return Cloud(name, rt, np.zeros(1, np.int64), np.zeros(0), np.zeros(0, np.float32), polarity)
    mz_parts, in_parts, counts = [], [], np.zeros(len(rt), np.int64)
    for k, (m, i) in enumerate(zip(mzs, intens)):
        m = np.asarray(m, dtype=np.float64)
        i = np.asarray(i, dtype=np.float32)
        keep = np.isfinite(i) & (i > 0) & np.isfinite(m)
        m, i = m[keep], i[keep]
        if len(m) > 1 and np.any(np.diff(m) < 0):
            o = np.argsort(m, kind="stable")
            m, i = m[o], i[o]
        mz_parts.append(m)
        in_parts.append(i)
        counts[k] = len(m)
    off = np.zeros(len(rt) + 1, np.int64)
    np.cumsum(counts, out=off[1:])
    mz = np.concatenate(mz_parts) if mz_parts else np.zeros(0)
    inten = np.concatenate(in_parts) if in_parts else np.zeros(0, np.float32)
    c = Cloud(name, rt, off, mz, inten, polarity)
    c.validate()
    return c


def _scan_rt_minutes(spec) -> float:
    t = spec["scanList"]["scan"][0]["scan start time"]
    unit = getattr(t, "unit_info", None)
    t = float(t)
    if unit == "second":
        return t / 60.0
    if unit == "minute":
        return t
    return t  # unknown unit: caller applies the seconds guard on the whole run


def _precursor_mz(spec) -> float:
    """the selected ion's m/z of an MS2 spectrum, else its isolation window target (nan if neither is given)"""
    try:
        prec = spec["precursorList"]["precursor"][0]
    except (KeyError, IndexError, TypeError):
        return float("nan")
    try:
        return float(prec["selectedIonList"]["selectedIon"][0]["selected ion m/z"])
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    try:
        return float(prec["isolationWindow"]["isolation window target m/z"])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _isolation_window(spec):
    """(lower, upper) m/z bounds of an MS2 spectrum's isolation window, (nan, nan) when the file does not say"""
    try:
        w = spec["precursorList"]["precursor"][0]["isolationWindow"]
        t = float(w["isolation window target m/z"])
        return t - float(w["isolation window lower offset"]), t + float(w["isolation window upper offset"])
    except (KeyError, IndexError, TypeError, ValueError):
        return float("nan"), float("nan")


ISOTOPE_DA = 1.003355          # 13C - 12C
PRECURSOR_PPM, PRECURSOR_DA = 10.0, 0.003


def precursor_in_ms1(mz, inten, prec, lo, hi, ppm: float = PRECURSOR_PPM, da: float = PRECURSOR_DA):
    """(intensity, purity) of an MS2 precursor in the MS1 scan before it: the strongest MS1 peak within max(ppm,
    da) of the precursor m/z (0 when there is none), and that ion's share of all MS1 intensity in the isolation
    window [lo, hi], counting its own 13C isotopes (charge 1) as its own; purity is nan when the window is unknown"""
    if mz is None or not np.isfinite(prec):
        return float("nan"), float("nan")
    tol = max(prec * ppm * 1e-6, da)
    near = np.abs(mz - prec) <= tol
    p_int = float(inten[near].max()) if near.any() else 0.0
    if not (np.isfinite(lo) and np.isfinite(hi)) or p_int <= 0:
        return p_int, float("nan")
    win = (mz >= lo) & (mz <= hi)
    total = float(inten[win].sum())
    own = p_int
    for k in (1, 2, 3):
        if prec + k * ISOTOPE_DA > hi:
            break
        iso = win & (np.abs(mz - (prec + k * ISOTOPE_DA)) <= tol)
        if iso.any():
            own += float(inten[iso].max())
    return p_int, (min(1.0, own / total) if total > 0 else float("nan"))


def load_cloud(path, polarity: str | None = None, ms2: list | None = None) -> Cloud:
    """Read MS1 centroids of an mzML file into a Cloud.

    polarity: None = require a single polarity in the file (raise on polarity switching),
    "pos"/"neg" = keep only that polarity.
    ms2: a list to collect the file's centroided MS2 spectra into, in the same pass (the parser decodes them
    anyway): (scan id, rt, precursor m/z, polarity, precursor intensity, precursor purity, m/z array, intensity
    array) with rt in minutes by the same rule as the MS1 scans; precursor intensity and purity are measured in the
    last MS1 scan before the spectrum (precursor_in_ms1). Empty and profile-mode MS2 spectra are skipped and
    counted in ``ms2.n_skipped`` when the collector has that attribute (ms2.Collector)."""
    from pyteomics import mzml

    want = {"pos": "positive scan", "neg": "negative scan"}.get(polarity)
    rts, mzs, intens = [], [], []
    seen = set()
    n_ms1 = n_profile = 0
    found = []
    last_mz = last_int = None      # the last MS1 scan kept, for the precursor of the MS2 spectra after it
    with mzml.read(str(path), use_index=False) as reader:
        for s in reader:
            level = s.get("ms level")
            if level != 1:
                if ms2 is not None and level == 2 and (want is None or want in s):
                    m = s.get("m/z array")
                    if "profile spectrum" in s or m is None or len(m) == 0:
                        if hasattr(ms2, "n_skipped"):
                            ms2.n_skipped += 1
                        continue
                    pol = "neg" if "negative scan" in s else ("pos" if "positive scan" in s else "")
                    prec = _precursor_mz(s)
                    p_int, purity = precursor_in_ms1(last_mz, last_int, prec, *_isolation_window(s))
                    found.append((str(s.get("id", "")), _scan_rt_minutes(s), prec, pol, p_int, purity,
                                  np.asarray(m, dtype=np.float64), np.asarray(s["intensity array"], dtype=np.float32)))
                continue
            n_ms1 += 1
            n_profile += "profile spectrum" in s
            pol = "neg" if "negative scan" in s else ("pos" if "positive scan" in s else "")
            seen.add(pol)
            if want is not None and want not in s:
                continue
            rts.append(_scan_rt_minutes(s))
            mzs.append(s["m/z array"])
            intens.append(s["intensity array"])
            last_mz, last_int = mzs[-1], intens[-1]
    if n_ms1 and n_profile == n_ms1:
        raise ValueError(f"{path}: MS1 spectra are profile mode; peak3d needs centroided MS1 "
                         f"(convert with msconvert --filter \"peakPicking vendor msLevel=1\")")
    pols = seen - {""}
    if polarity is None and len(pols) > 1:
        raise ValueError(f"{path}: both polarities present; pass polarity='pos' or 'neg'")
    rt = np.asarray(rts, dtype=np.float64)
    seconds = bool(len(rt) and rt.max() > 200)   # a run longer than 200 min is implausible: these are seconds
    if seconds:
        rt = rt / 60.0
    if ms2 is not None:
        ms2.extend((sid, t / 60.0 if seconds else t, *rest) for sid, t, *rest in found)
    pol = polarity or (next(iter(pols)) if pols else "")
    return from_scans(Path(path).stem, rt, mzs, intens, pol)
