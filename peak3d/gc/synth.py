"""Synthetic GC-EI runs with known compounds: the test oracle of the GC stages.

A compound is a list of fragment ions (m/z, relative abundance) that share one apex time and one peak shape; every
ion carries an M+1 isotope peak. Two compounds may hold the same m/z (a shared ion): the centroider-like merge of
synth.make_cloud sums the two signals scan by scan, as an instrument does."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..synth import Peak, make_cloud

ELECTRON = 0.00054858
H, C12, O16 = 1.00782503207, 12.0, 15.99491461956


@dataclass
class Compound:
    name: str
    rt: float                 # apex, minutes
    height: float             # of the base peak
    mz: np.ndarray            # fragment m/z
    rel: np.ndarray           # relative abundance, base peak 1
    fwhm: float = 0.05        # minutes
    tau: float = 0.0          # tailing (EMG), minutes
    ri: float = float("nan")  # retention index of the library entry (tests)


def random_spectrum(rng, n_ions: int = 20, mz_lo: float = 50.0, mz_hi: float = 400.0, min_sep: float = 0.5):
    """n_ions fragment m/z at least min_sep apart with CH-like mass defects, relative abundances log-uniform in
    [0.02, 1] with one base peak"""
    nominal = np.sort(rng.choice(np.arange(int(mz_lo), int(mz_hi)), size=n_ions, replace=False)).astype(np.float64)
    mz = nominal * (1.0 + rng.uniform(2e-4, 9e-4, n_ions))
    keep = np.concatenate([[True], np.diff(mz) >= min_sep])
    mz = mz[keep]
    rel = np.exp(rng.uniform(np.log(0.02), 0.0, len(mz)))
    rel[rng.integers(len(mz))] = 1.0
    return mz, rel / rel.max()


def alkane(n_c: int, rt: float, height: float, fwhm: float = 0.05) -> Compound:
    """n-alkane CnH2n+2: the CkH2k+1+ series (m/z 57, 71, 85, ...) and a weak molecular ion"""
    ks = np.arange(4, min(n_c, 12))
    mz = [C12 * k + H * (2 * k + 1) - ELECTRON for k in ks]
    rel = [1.0, 0.75, 0.55] + [0.3 * 0.7 ** i for i in range(len(ks) - 3)]
    mz.append(C12 * n_c + H * (2 * n_c + 2) - ELECTRON)
    rel.append(0.03)
    return Compound(f"C{n_c} alkane", rt, height, np.array(mz), np.array(rel[:len(mz) - 1] + [0.03]), fwhm,
                    ri=100.0 * n_c)


def fame(n_c: int, rt: float, height: float, fwhm: float = 0.05) -> Compound:
    """methyl ester of the saturated Cn fatty acid: m/z 74.0362 (McLafferty), 87.0441, and the molecular ion"""
    mz = [74.03623, 87.04406, 101.05971, 143.10666, C12 * (n_c + 1) + H * (2 * n_c + 2) + 2 * O16 - ELECTRON]
    return Compound(f"C{n_c} FAME", rt, height, np.array(mz), np.array([1.0, 0.65, 0.08, 0.12, 0.1]), fwhm)


def make_run(compounds, n_scans: int = 1500, dt: float = 1.0 / 300.0, iso: float = 0.12, seed: int = 0,
             thresh: float = 20.0, inoise_c: float = 1.0, inoise_r: float = 0.01, floor: float = 0.0,
             n_noise: int = 0, name: str = "gc", extra_peaks=(), **kw):
    """(Cloud, truth) of a run holding the compounds; truth rows carry the compound name of each ion.
    dt 1/300 min = 5 spectra per second; every ion has an M+1 at `iso` of its height. extra_peaks: synth.Peak
    objects added as they are (a background ion: one very wide peak), compound name ""."""
    peaks, owner = list(extra_peaks), [""] * len(extra_peaks)
    for c in compounds:
        for m, r in zip(c.mz, c.rel):
            peaks.append(Peak(float(m), c.rt, c.height * float(r), c.fwhm, c.tau, iso=(iso,) if iso > 0 else ()))
            owner.append(c.name)
    cloud, truth = make_cloud(peaks, n_scans=n_scans, dt=dt, mz_range=(40.0, 600.0), thresh=thresh, seed=seed,
                              inoise_c=inoise_c, inoise_r=inoise_r, floor=floor, n_noise=n_noise, name=name,
                              merge_ppm=kw.pop("merge_ppm", 10.0), **kw)
    truth["compound"] = [owner[p] for p in truth["peak_id"]]
    return cloud, truth
