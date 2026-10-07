"""Minimal in-memory MS1 access for building tool-independent truth sets.

An MS1 run is held as flat numpy arrays sorted by m/z, so an extracted ion
chromatogram (EIC) for any m/z window is one searchsorted plus a scatter-max.
Nothing here detects peaks; it only reads centroids.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyteomics import mzml

C13 = 1.0033548378  # 13C - 12C


@dataclass
class Run:
    name: str
    rt: np.ndarray        # (n_scans,) minutes
    mz: np.ndarray        # all centroids, sorted by m/z
    inten: np.ndarray
    scan: np.ndarray      # scan index of each centroid

    @property
    def n_scans(self):
        return len(self.rt)

    def eic(self, mz, ppm=5.0, lo=0, hi=None):
        """max intensity per scan within +/- ppm of mz (dense, length n_scans)"""
        tol = mz * ppm * 1e-6
        a, b = np.searchsorted(self.mz, [mz - tol, mz + tol])
        out = np.zeros(self.n_scans)
        if b > a:
            np.maximum.at(out, self.scan[a:b], self.inten[a:b])
        return out

    def scan_window(self, rt, half_width):
        return (int(np.searchsorted(self.rt, rt - half_width)),
                int(np.searchsorted(self.rt, rt + half_width)))


def load_run(path):
    rts, mzs, ints, scans = [], [], [], []
    k = 0
    with mzml.read(str(path)) as reader:
        for s in reader:
            if s.get("ms level") != 1:
                continue
            m, i = s["m/z array"], s["intensity array"]
            keep = i > 0
            rts.append(float(s["scanList"]["scan"][0]["scan start time"]))
            mzs.append(m[keep]); ints.append(i[keep]); scans.append(np.full(keep.sum(), k, np.int32))
            k += 1
    mz = np.concatenate(mzs); inten = np.concatenate(ints).astype(float); scan = np.concatenate(scans)
    o = np.argsort(mz, kind="stable")
    rt = np.asarray(rts)
    # pyteomics reports minutes for Thermo/msconvert mzML; guard against seconds
    if rt.max() > 200:
        rt = rt / 60.0
    return Run(Path(path).stem, rt, mz[o], inten[o], scan[o])
