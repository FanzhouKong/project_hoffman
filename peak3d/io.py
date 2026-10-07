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


def load_cloud(path, polarity: str | None = None) -> Cloud:
    """Read MS1 centroids of an mzML file into a Cloud.

    polarity: None = require a single polarity in the file (raise on polarity switching),
    "pos"/"neg" = keep only that polarity."""
    from pyteomics import mzml

    want = {"pos": "positive scan", "neg": "negative scan"}.get(polarity)
    rts, mzs, intens = [], [], []
    seen = set()
    with mzml.read(str(path), use_index=False) as reader:
        for s in reader:
            if s.get("ms level") != 1:
                continue
            pol = "neg" if "negative scan" in s else ("pos" if "positive scan" in s else "")
            seen.add(pol)
            if want is not None and want not in s:
                continue
            rts.append(_scan_rt_minutes(s))
            mzs.append(s["m/z array"])
            intens.append(s["intensity array"])
    pols = seen - {""}
    if polarity is None and len(pols) > 1:
        raise ValueError(f"{path}: both polarities present; pass polarity='pos' or 'neg'")
    rt = np.asarray(rts, dtype=np.float64)
    if len(rt) and rt.max() > 200:  # a run longer than 200 min is implausible: these are seconds
        rt = rt / 60.0
    pol = polarity or (next(iter(pols)) if pols else "")
    return from_scans(Path(path).stem, rt, mzs, intens, pol)
