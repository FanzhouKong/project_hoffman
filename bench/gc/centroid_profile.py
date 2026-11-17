#!/usr/bin/env python3
"""Profile-mode mzML -> centroid mzML, for runs that exist only as profile data (the MRH GC-Orbitrap set).

peak3d reads centroided MS1. A vendor centroider is not available for an mzML file, so each profile spectrum is
reduced here: every local maximum of the intensity is one centroid, its m/z and height the vertex of the parabola
through the logarithms of the maximum and its two neighbours (exact for a Gaussian peak, any sample spacing); a
maximum with a zero neighbour keeps its own sample. This is also how the MRH ground truth was made ("closest local
maxima" of the raw data), so truth and centroids describe the same points. Nothing is smoothed or thresholded
unless --min-intensity is given.

    python -I bench/gc/centroid_profile.py IN.mzML OUT.mzML [--min-intensity X]

Prints one tab-separated summary line: file, MS1 scans, profile points, centroids, seconds."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from peak3d.io import _scan_rt_minutes, from_scans      # noqa: E402
from peak3d.synth import write_mzml                    # noqa: E402


@njit(cache=True)
def centroid_scan(mz, it, min_intensity):
    n = len(mz)
    out_m = np.empty(n // 2 + 1, np.float64)
    out_i = np.empty(n // 2 + 1, np.float32)
    k = 0
    for p in range(n):
        y = it[p]
        if y <= 0 or y < min_intensity:
            continue
        yl = it[p - 1] if p > 0 else 0.0
        yr = it[p + 1] if p < n - 1 else 0.0
        if not (y > yl and y >= yr):
            continue
        m = mz[p]
        h = y
        if yl > 0 and yr > 0:
            x0 = mz[p - 1] - m
            x2 = mz[p + 1] - m
            l0 = np.log(yl / y)
            l2 = np.log(yr / y)
            den = x0 * x2 * (x0 - x2)
            if den != 0.0:
                a = (l0 * x2 - l2 * x0) / den
                b = (l2 * x0 * x0 - l0 * x2 * x2) / den
                if a < 0.0:
                    xv = -b / (2.0 * a)
                    if x0 < xv < x2:
                        m += xv
                        h = y * np.exp(-b * b / (4.0 * a))
        out_m[k] = m
        out_i[k] = h
        k += 1
    return out_m[:k], out_i[:k]


def centroid_file(src, min_intensity: float = 0.0):
    """(Cloud of centroids, number of profile points) of the MS1 spectra of a profile mzML"""
    from pyteomics import mzml
    rts, mzs, its = [], [], []
    n_prof = 0
    pols = set()
    with mzml.read(str(src), use_index=False) as reader:
        for s in reader:
            if s.get("ms level") != 1:
                continue
            m = np.asarray(s["m/z array"], np.float64)
            i = np.asarray(s["intensity array"], np.float64)
            n_prof += len(m)
            if "profile spectrum" in s:
                m, i = centroid_scan(m, i, float(min_intensity))
            pols.add("neg" if "negative scan" in s else "pos")
            rts.append(_scan_rt_minutes(s))
            mzs.append(m)
            its.append(i)
    rt = np.asarray(rts, np.float64)
    if len(rt) and rt.max() > 200:          # seconds without a unit (load_cloud's rule)
        rt = rt / 60.0
    return from_scans(Path(src).stem, rt, mzs, its, pols.pop() if len(pols) == 1 else "pos"), n_prof


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--min-intensity", type=float, default=0.0)
    a = ap.parse_args()
    tic = time.perf_counter()
    cloud, n_prof = centroid_file(a.src, a.min_intensity)
    cloud.validate()
    tmp = a.dst.with_name(a.dst.name + ".part")
    write_mzml(cloud, tmp)
    tmp.replace(a.dst)
    print(f"{a.src.name}\t{cloud.n_scans}\t{n_prof}\t{cloud.n_points}\t{time.perf_counter() - tic:.1f}", flush=True)


if __name__ == "__main__":
    main()
