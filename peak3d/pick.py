"""Per-file 3D pick: Cloud -> basin table (numpy arrays). No pandas here."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import kernels as K
from .estimate import Params, estimate
from .io import Cloud

MAX_WIDTH_FWHM = 6.0   # bounds walk stops this many median FWHMs from the apex
MIN_SEP_FWHM = 0.8     # same-ion apexes closer than this (in FWHMs) are one peak with a dip
TRACE_MIN_SCANS = 5    # a merged basin this long is a real ion's trace (for the noise surface)
SIDEBAND_PPM = 100.0   # an isolated centroid this close in m/z to a trace point ...
SIDEBAND_RATIO = 10.0  # ... at least this many times stronger, in the same scan, is a sideband


@dataclass
class Basins:
    stats: np.ndarray          # (n, kernels.NCOL) float64, columns kernels.COLS
    persistence: np.ndarray    # (n,) apex - saddle to the first higher surviving basin
    noise: np.ndarray          # (n,) noise surface at the basin's apex cell
    basin_id: np.ndarray       # (n,) id into lab
    lab: np.ndarray            # (N,) int32 final basin id per centroid (all basins, not only selected)
    params: Params
    n_basins_initial: int
    n_after_merge: int
    timings: dict = field(default_factory=dict)
    noise_surface: np.ndarray | None = None   # (n_bands, n_blocks) the surface the S/N used

    def col(self, name):
        return self.stats[:, K.C[name]]


def _merge(P, nbr, lab, eff, smooth, apex_int, apex_scan, apex_mz, apex_sig, noise_b, B):
    """persistence merge of the initial basins; returns (lab2, B2, surv, persistence per survivor)"""
    if P.far_frac > 0:
        ea, eb, es, es_s = K.saddle_edges(nbr, lab, eff, smooth)
        apex_s = K.basin_max(lab, smooth, B)
    else:
        ea, eb, es = K.saddle_edges(nbr, lab, eff)
        es_s, apex_s = es, apex_int
    uf, pers_saddle = K.persistence_merge(ea, eb, es, apex_int, noise_b, P.rel_frac, P.k_noise,
                                          apex_scan, apex_mz, apex_sig, MIN_SEP_FWHM * P.fwhm_scans,
                                          es_s, apex_s, P.far_fwhm * P.fwhm_scans, P.far_frac)
    surv = np.flatnonzero(uf == np.arange(B, dtype=uf.dtype))
    remap = np.full(B, -1, np.int32)
    remap[surv] = np.arange(len(surv), dtype=np.int32)
    lab2 = remap[uf[lab]]
    persistence = apex_int[surv].astype(np.float64) - np.where(pers_saddle[surv] > 0, pers_saddle[surv], 0.0)
    return lab2, len(surv), surv, persistence, len(es)


def pick_cloud(cloud: Cloud, params: Params | None = None) -> Basins:
    t = {}
    tic = time.perf_counter()
    P = params if params is not None else estimate(cloud)
    t["estimate"] = time.perf_counter() - tic

    off, mz, inten, rt = cloud.off, cloud.mz, cloud.inten, cloud.rt
    N = cloud.n_points

    tic = time.perf_counter()
    sig = K.per_point_sigma(inten, P.sig_a, P.sig_b)
    nbr = K.find_neighbors(off, mz, inten, sig, P.K, P.tol_max_ppm, P.same_scan_k, P.same_scan_floor_ppm)
    t["neighbors"] = time.perf_counter() - tic

    tic = time.perf_counter()
    eff = K.effective_intensity(nbr, inten)
    parent = K.link_steepest(nbr, eff, inten)
    label = K.resolve_roots(parent)
    lab, roots = K.dense_labels(label)
    B = len(roots)
    smin, smax, npts = K.basin_extent(lab, off, B)
    t["link"] = time.perf_counter() - tic

    # noise surface, first pass: centroids whose basin before the merge spans <= 2 scans
    tic = time.perf_counter()
    mz_lo, mz_hi = float(mz.min()), float(mz.max()) * (1 + 1e-9)
    cell_args = (rt, mz_lo, mz_hi, P.n_mz_bands, P.rt_block_min, P.noise_cell_min_points, P.floor)
    med, _ = K.noise_cell_medians(off, mz, inten, lab, smin, smax, *cell_args)
    n_blocks = med.shape[1]
    scan_of = cloud.scan_index()
    lmz_lo = np.log(mz_lo)
    lmz_w = (np.log(mz_hi) - lmz_lo) / P.n_mz_bands + 1e-12
    band = np.clip(((np.log(mz[roots]) - lmz_lo) / lmz_w).astype(np.int64), 0, P.n_mz_bands - 1)
    blk = np.clip(((rt[scan_of[roots]] - rt[0]) / P.rt_block_min).astype(np.int64), 0, n_blocks - 1)
    apex_int = eff[roots].astype(np.float32)
    t["noise"] = time.perf_counter() - tic

    tic = time.perf_counter()
    apex_scan = scan_of[roots].astype(np.int64)
    apex_mz = mz[roots]
    apex_sig = sig[roots].astype(np.float64)
    smooth = K.smooth_traces(nbr, eff, P.n_smooth) if P.far_frac > 0 else None
    margs = (P, nbr, lab, eff, smooth, apex_int, apex_scan, apex_mz, apex_sig)
    lab2, B2, surv, persistence, n_edges = _merge(*margs, med[band, blk], B)
    t["merge"] = time.perf_counter() - tic

    if P.noise_post_merge:
        # second pass: before the merge every ripple maximum on a ragged real trace is its own
        # one- or two-scan basin and would count as noise; after it, the centroids that still
        # form no 3D structure are the noise, minus the sidebands of intense ions
        tic = time.perf_counter()
        smin2, smax2, _ = K.basin_extent(lab2, off, B2)
        isolated = K.short_basin_mask(lab2, smin2, smax2, 2)
        trace = ~K.short_basin_mask(lab2, smin2, smax2, TRACE_MIN_SCANS - 1)
        side = K.sideband_mask(off, mz, inten, isolated, trace, SIDEBAND_PPM, SIDEBAND_RATIO)
        med, _ = K.cell_medians(off, mz, inten, isolated & ~side, *cell_args)
        t["noise2"] = time.perf_counter() - tic
        tic = time.perf_counter()
        lab2, B2, surv, persistence, n_edges = _merge(*margs, med[band, blk], B)
        t["merge2"] = time.perf_counter() - tic
    noise2 = med[band, blk][surv]

    tic = time.perf_counter()
    smin2, smax2, npts2 = K.basin_extent(lab2, off, B2)
    sel = np.flatnonzero((smax2 - smin2 + 1 >= P.min_scans) & (npts2 >= P.min_scans))
    order, boff = K.sort_by_label(lab2, B2)
    stats = K.basin_stats(sel, order, boff, scan_of, nbr, mz, inten, eff, rt, noise2, P.fwhm_med, MAX_WIDTH_FWHM,
                          P.noise_c, P.noise_r, P.k_valley, P.rep_raw)
    t["stats"] = time.perf_counter() - tic
    t["n_edges"] = int(n_edges)
    return Basins(stats=stats, persistence=persistence[sel], noise=noise2[sel], basin_id=sel, lab=lab2,
                  params=P, n_basins_initial=int(B), n_after_merge=int(B2), timings=t, noise_surface=med)
