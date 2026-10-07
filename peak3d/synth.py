"""Synthetic point clouds with known truth: the test oracle for peak3d.

Peaks are Gaussian or exponentially modified Gaussian profiles sampled on the scan grid,
with an intensity-dependent m/z error ``sqrt(a^2 + b^2 / I)`` ppm, a centroider-like
intensity threshold, optional random dropout of weak centroids, flat-top clipping, sparse
floor noise and removed scans (acquisition gaps). Everything is seeded and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import exponnorm

from pathlib import Path

from .io import C13, Cloud, from_scans

FWHM_TO_SIGMA = 1.0 / 2.3548200450309493


@dataclass
class Peak:
    mz: float
    rt: float            # apex time, minutes
    height: float
    fwhm: float          # minutes
    tau: float = 0.0     # EMG tail constant in minutes; 0 = Gaussian
    z: int = 1
    iso: tuple = ()      # relative heights of M+1, M+2, ... (same shape, at mz + n*C13/z)
    clip: float = 0.0    # if > 0: saturate the profile at clip * height (flat top)


def profile(t: np.ndarray, pk: Peak) -> np.ndarray:
    """unit-height profile of ``pk`` at times t, apex exactly at pk.rt"""
    sigma = pk.fwhm * FWHM_TO_SIGMA
    if pk.tau <= 0:
        y = np.exp(-0.5 * ((t - pk.rt) / sigma) ** 2)
    else:
        # exponnorm's mode is right of loc; find the shift on a fine grid and re-centre
        K = pk.tau / sigma
        g = np.linspace(-6 * sigma, 6 * sigma + 8 * pk.tau, 4001)
        f = exponnorm.pdf(g, K, loc=0.0, scale=sigma)
        shift = g[np.argmax(f)]
        y = exponnorm.pdf(t - pk.rt + shift, K, loc=0.0, scale=sigma) / f.max()
    if pk.clip > 0:
        y = np.minimum(y, pk.clip)
    return y


def make_cloud(peaks, n_scans: int = 600, dt: float = 0.005, t0: float = 0.0,
               mz_range=(100.0, 1000.0), sig_a: float = 0.5, sig_b: float = 30.0,
               thresh: float = 0.0, dropout_scale: float = 0.0, floor: float = 0.0,
               n_noise: int = 0, drop_scans=(), seed: int = 0, name: str = "synthetic",
               merge_ppm: float = 3.0, inoise_c: float = 0.0, inoise_r: float = 0.0,
               mz_outlier_frac: float = 0.0, mz_outlier_k: float = 5.0, sat_I: float = 0.0, sat_ppm: float = 0.0):
    """Return (Cloud, truth DataFrame).

    sig_a, sig_b: m/z error model in ppm, sd = sqrt(sig_a^2 + sig_b^2 / I)
    inoise_c, inoise_r: intensity noise, sd = sqrt(inoise_c * I + (inoise_r * I)^2) added to every
                  sampled point before the threshold (0, 0 = exact intensities, no random draws)
    thresh:       centroider threshold; sampled points below it are not emitted
    dropout_scale: if > 0, a point of intensity I is dropped with probability exp(-I / scale)
    floor, n_noise: sparse noise points with median intensity ``floor`` (log-normal, sd 0.7)
    drop_scans:   scan indices removed from the grid (MS1 acquisition gaps)
    merge_ppm:    centroids of one scan closer than this are summed into one (what a
                  centroider does with unresolved signals); 0 disables
    mz_outlier_frac, mz_outlier_k: this fraction of peak centroids gets an m/z error of
                  +/- mz_outlier_k sd instead of a Gaussian draw (heavy-tailed scatter); 0 disables
    sat_I, sat_ppm: detector saturation: a peak centroid above sat_I is shifted by
                  sat_ppm * (1 - sat_I / I) ppm; 0 disables
    truth columns: peak_id, iso_n, z, mz, rt, height, area, fwhm, n_sampled
    """
    rng = np.random.default_rng(seed)
    rt = t0 + dt * np.arange(n_scans, dtype=np.float64)
    if len(drop_scans):
        rt = np.delete(rt, np.asarray(drop_scans, int))
    S = len(rt)
    per_scan_mz = [[] for _ in range(S)]
    per_scan_in = [[] for _ in range(S)]
    rows = []
    for pid, pk in enumerate(peaks):
        shape = profile(rt, pk)
        for n, ratio in enumerate((1.0,) + tuple(pk.iso)):
            mz0 = pk.mz + n * C13 / pk.z
            I = pk.height * ratio * shape
            k = int(np.argmax(I))
            rows.append(dict(peak_id=pid, iso_n=n, z=pk.z, mz=mz0, rt=float(rt[k]), height=float(I[k]),
                             area=float(np.trapz(I, rt)), fwhm=pk.fwhm,
                             n_sampled=int(np.count_nonzero(I >= max(thresh, 1e-12)))))
            if inoise_c > 0 or inoise_r > 0:
                I = I + rng.standard_normal(S) * np.sqrt(inoise_c * I + (inoise_r * I) ** 2)
                I = np.maximum(I, 0.0)
            keep = I >= max(thresh, 1e-12)
            if dropout_scale > 0:
                keep &= rng.random(S) >= np.exp(-I / dropout_scale)
            for s in np.flatnonzero(keep):
                sd_ppm = np.sqrt(sig_a ** 2 + sig_b ** 2 / I[s])
                dev = sd_ppm * rng.standard_normal()
                if mz_outlier_frac > 0 and rng.random() < mz_outlier_frac:
                    dev = sd_ppm * mz_outlier_k * (1.0 if rng.random() < 0.5 else -1.0)
                if sat_I > 0 and I[s] > sat_I:
                    dev += sat_ppm * (1.0 - sat_I / I[s])
                per_scan_mz[s].append(mz0 * (1.0 + 1e-6 * dev))
                per_scan_in[s].append(I[s])
    if n_noise and floor > 0:
        sc = rng.integers(0, S, n_noise)
        lo, hi = np.log(mz_range[0]), np.log(mz_range[1])
        nm = np.exp(rng.uniform(lo, hi, n_noise))
        ni = floor * np.exp(0.7 * rng.standard_normal(n_noise))
        for s, m, i in zip(sc, nm, ni):
            per_scan_mz[s].append(m)
            per_scan_in[s].append(i)
    mz_arrays, in_arrays = [], []
    for m, i in zip(per_scan_mz, per_scan_in):
        m, i = np.asarray(m, dtype=np.float64), np.asarray(i, dtype=np.float64)
        if merge_ppm > 0 and len(m) > 1:
            o = np.argsort(m, kind="stable")
            m, i = m[o], i[o]
            grp = np.concatenate([[0], np.cumsum(np.diff(m) / m[1:] * 1e6 > merge_ppm)])
            tot = np.bincount(grp, weights=i)
            m = np.bincount(grp, weights=i * m) / tot
            i = tot
        mz_arrays.append(m)
        in_arrays.append(i)
    cloud = from_scans(name, rt, mz_arrays, in_arrays, "pos")
    truth = pd.DataFrame(rows, columns=["peak_id", "iso_n", "z", "mz", "rt", "height", "area", "fwhm", "n_sampled"])
    return cloud, truth


def permute_within_scans(cloud: Cloud, seed: int = 0) -> Cloud:
    """Same centroids, each scan's points handed to from_scans in a random order."""
    rng = np.random.default_rng(seed)
    mzs, ins = [], []
    for s in range(cloud.n_scans):
        a, b = cloud.off[s], cloud.off[s + 1]
        o = rng.permutation(b - a)
        mzs.append(cloud.mz[a:b][o])
        ins.append(cloud.inten[a:b][o])
    return from_scans(cloud.name, cloud.rt, mzs, ins, cloud.polarity)


def write_mzml(cloud: Cloud, path) -> None:
    """Write a minimal centroid MS1 mzML (zlib + base64, 64-bit m/z, 32-bit intensity, scan
    start time in minutes) that pyteomics reads back into the same Cloud. Test helper only."""
    import base64
    import zlib

    def enc(a):
        return base64.b64encode(zlib.compress(a.tobytes())).decode()

    pol = {"pos": '<cvParam cvRef="MS" accession="MS:1000130" name="positive scan" value=""/>',
           "neg": '<cvParam cvRef="MS" accession="MS:1000129" name="negative scan" value=""/>'}.get(cloud.polarity, "")
    out = ['<?xml version="1.0" encoding="utf-8"?>',
           '<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">',
           '<cvList count="1"><cv id="MS" fullName="PSI-MS" URI="https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"/></cvList>',
           '<fileDescription><fileContent><cvParam cvRef="MS" accession="MS:1000579" name="MS1 spectrum" value=""/>'
           '<cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum" value=""/></fileContent></fileDescription>',
           '<run id="run">', f'<spectrumList count="{cloud.n_scans}">']
    for s in range(cloud.n_scans):
        a, b = cloud.off[s], cloud.off[s + 1]
        mz = np.ascontiguousarray(cloud.mz[a:b], dtype="<f8")
        it = np.ascontiguousarray(cloud.inten[a:b], dtype="<f4")
        out.append(
            f'<spectrum index="{s}" id="scan={s + 1}" defaultArrayLength="{b - a}">'
            '<cvParam cvRef="MS" accession="MS:1000579" name="MS1 spectrum" value=""/>'
            '<cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="1"/>'
            '<cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum" value=""/>' + pol +
            f'<scanList count="1"><scan><cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="{cloud.rt[s]:.6f}" unitCvRef="UO" unitAccession="UO:0000031" unitName="minute"/></scan></scanList>'
            '<binaryDataArrayList count="2">'
            f'<binaryDataArray encodedLength="{len(enc(mz))}"><cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>'
            '<cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>'
            '<cvParam cvRef="MS" accession="MS:1000514" name="m/z array" value=""/>'
            f'<binary>{enc(mz)}</binary></binaryDataArray>'
            f'<binaryDataArray encodedLength="{len(enc(it))}"><cvParam cvRef="MS" accession="MS:1000521" name="32-bit float" value=""/>'
            '<cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>'
            '<cvParam cvRef="MS" accession="MS:1000515" name="intensity array" value=""/>'
            f'<binary>{enc(it)}</binary></binaryDataArray>'
            '</binaryDataArrayList></spectrum>')
    out += ['</spectrumList>', '</run>', '</mzML>']
    Path(path).write_text("\n".join(out))
