"""Synthetic noise for MS/MS spectra, with continuous intensities (the Poisson(λ)/100 intensities of Kong et al.
2025 are quantised, which the identical-intensity rule detects on its own).

add_noise(mz, inten, pmz, n, kind, scale, rng, table=None, bounds=None):
  kind "uniform":  m/z uniform in [50, pmz - 1.6]                       (electronic-like)
  kind "formula":  m/z = pmz - (a random plausible loss mass), i.e. the ion sits where some formula could
                   (chemical-like, what a formula-only test cannot remove)
  intensity: lognormal, median = scale x base peak, sigma 0.7 (continuous)
Returns (mz, inten, is_noise) sorted by m/z."""
from __future__ import annotations

import numpy as np

from .losses import MAX_TABLE, BIN


def random_loss_masses(n, maxloss, table, rng):
    """random masses from the plausible-loss bitmap (table = cumulative counts per BIN) up to maxloss"""
    kmax = min(int(maxloss / BIN), len(table) - 2)
    total = table[kmax + 1]
    if total <= 0:
        return rng.uniform(1.0, maxloss, n)
    picks = rng.integers(1, total + 1, n)
    k = np.searchsorted(table, picks, side="left") - 1
    return (k + rng.uniform(0, 1, n)) * BIN


def add_noise(mz, inten, pmz, n, kind, scale, rng, table=None):
    mz = np.asarray(mz, float); inten = np.asarray(inten, float)
    n = int(np.ceil(n))
    if n <= 0:
        order = np.argsort(mz)
        return mz[order], inten[order], np.zeros(len(mz), bool)[order]
    if kind == "uniform":
        nmz = rng.uniform(50.0, pmz - 1.6, n)
    elif kind == "formula":
        losses = random_loss_masses(n, min(pmz - 50.0, MAX_TABLE), table, rng)
        nmz = pmz - losses
        nmz = np.where(nmz < 50.0, rng.uniform(50.0, pmz - 1.6, n), nmz)
    else:
        raise ValueError(kind)
    nint = inten.max() * scale * rng.lognormal(0.0, 0.7, n)
    amz = np.concatenate([mz, nmz]); ait = np.concatenate([inten, nint])
    isn = np.concatenate([np.zeros(len(mz), bool), np.ones(n, bool)])
    order = np.argsort(amz)
    return amz[order], ait[order], isn[order]
