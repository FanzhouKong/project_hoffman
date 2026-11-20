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

- --as-is: none of the above. The centroids are written with the masses and intensities the file holds (points of
  zero or negative intensity left out, the scans sorted by mass), as a vendor-neutral converter would write them.
  This is the mzML to give peak3d since 1.4.0b3: in GC-MS mode it tests the masses of every mzML file itself and
  reduces a unit-resolution run to nominal masses, with one scan skew for the whole run (peak3d/cdf.py).

    python -I bench/gc/cdf_to_mzml.py IN.CDF OUT.mzML [--as-is] [--mass-scale S] [--deskew K|auto] [--name STEM]

Prints one tab-separated summary line: file, scans, input points, output centroids, m/z range, skew, seconds."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# the reader, the nominal-mass reduction and the skew correction are peak3d's own since 1.4.0b2 (peak3d/cdf.py, which
# reads .cdf files directly); this script keeps the benchmark's conversion to mzML with its explicit settings
from peak3d.cdf import (BIN_SHIFT, C13, _apex, deskew, estimate_skew, measure_skew, noise_floor, read_cdf,  # noqa: E402,F401
                        unit_mass, unit_matrix)


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


def convert_as_is(src, name=None):
    """(Cloud, input points): the file's centroids as they are"""
    from peak3d.cdf import polarity
    from peak3d.io import from_scans
    rt, off, mz, it = read_cdf(src)
    n = len(rt)
    cloud = from_scans(name or Path(src).stem, rt / 60.0, [mz[off[s]:off[s + 1]] for s in range(n)],
                       [it[off[s]:off[s + 1]] for s in range(n)], polarity(src))
    return cloud, len(mz)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path, nargs="?", help="omit with --measure-skew")
    ap.add_argument("--noise-floor", default="auto", help="counts, or auto")
    ap.add_argument("--measure-skew", action="store_true", help="print the file's K and stop")
    ap.add_argument("--mass-scale", type=float, default=C13)
    ap.add_argument("--deskew", default="0", help="scans per 1000 Da, or auto")
    ap.add_argument("--name", default=None)
    ap.add_argument("--as-is", action="store_true", help="write the file's centroids as they are (see above)")
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
    if a.as_is:
        cloud, n_in = convert_as_is(a.src, a.name or a.dst.stem)
        cloud.validate()
        tmp = a.dst.with_name(a.dst.name + ".part")
        write_mzml(cloud, tmp)
        tmp.replace(a.dst)
        print(f"{a.src.name}\t{cloud.n_scans}\t{n_in}\t{cloud.n_points}\t{cloud.mz.min():.4f}-{cloud.mz.max():.4f}\t"
              f"as is\t{time.perf_counter() - tic:.1f}", flush=True)
        return
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
