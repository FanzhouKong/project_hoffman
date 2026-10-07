"""Per-file parameter estimation. Nothing here is tuned per dataset: every value is read off
the file itself (scan spacing, peak width, m/z error model, intensity floor) and written to
the params sidecar so a reviewer can see what the file told the picker."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .io import Cloud

SEED_PPM = 20.0        # EIC width used only for the strongest ions during estimation
SEED_MIN_SEP_PPM = 20.0


@dataclass
class Params:
    # measured
    dt: float = float("nan")            # median MS1 spacing, minutes
    span: float = float("nan")          # last - first scan time, minutes
    floor: float = float("nan")         # 1st percentile of all intensities
    fwhm_med: float = float("nan")      # median FWHM of strong ions, minutes
    fwhm_scans: float = float("nan")
    sig_a: float = 0.5                  # m/z error model: sd_ppm(I) = sqrt(a^2 + b^2 / I)
    sig_b: float = 0.0
    n_seeds: int = 0
    n_fwhm_seeds: int = 0
    n_sigma_points: int = 0
    # derived
    tol_max_ppm: float = 10.0
    K: int = 2
    min_scans: int = 3
    # linking: same-scan neighbours within same_scan_k x the pair tolerance, or within same_scan_floor_ppm, are
    # split pieces of one centroid (their intensities are summed into the surface height); a factor of 2 let a
    # weak piece's sigma reach a second ion 10-20 ppm away. rep_raw: a scan's representative point for the m/z
    # statistics and the apex is the basin point with the highest raw intensity, not the first point with the
    # highest summed one, so a second ion linked in as a split partner cannot stand for the scan
    same_scan_k: float = 1.0
    same_scan_floor_ppm: float = 5.0
    rep_raw: bool = True
    # fixed
    rel_frac: float = 0.5      # a valley must fall below half the lower apex for two peaks to be two
    far_fwhm: float = 2.0      # ... unless the apexes are at least this many median FWHMs apart: then
    far_frac: float = 0.2      # a valley this far below the lower apex on the smoothed trace suffices (0 = off)
    smooth_fwhm: float = 0.5   # smoothing kernel for that test, FWHM in median peak FWHMs
    n_smooth: int = 1          # derived: [1, 2, 1] passes giving that kernel
    noise_post_merge: bool = True  # noise surface from centroids still isolated after the merge (no sidebands)
    snr_floor_baseline: bool = True  # S/N above the lowest level the trace reaches at its bounds
    ripple_k: float = 0.0      # apex must rise this many scan-to-scan ripples above the ion's own level (0 = off)
    ridge_w_cap: float = 2.0   # side windows (ridge / background / far-level tests) use w = min(max(own FWHM,
                               # median FWHM), ridge_w_cap x median FWHM): a 10x-wide ridge segment is judged
                               # beside itself, not 20-60 widths away (inf = windows scale with the feature)
    # intensity-dependent noise: expected scan-to-scan sd of an ion current at level I is
    # sqrt(cell_floor^2 + noise_c * I + noise_r^2 * I^2); c and r fitted per file (fit_intensity_noise)
    noise_c: float = 0.0
    noise_r: float = 0.0
    noise_s0: float = float("nan")   # the fit's own floor term (diagnostic; the cell surface is used at run time)
    n_noise_pts: int = 0
    n_noise_bins: int = 0
    k_valley: float = 3.0      # a valley inside a feature is significant at this many sd of the level difference
    max_valleys: int = -1      # gate: reject a candidate with more significant valleys than this (-1 = off)
    score_loc_rip: bool = True  # rule score includes the localisation (1 - far_level) and unimodality (1 / tv_ratio) terms
    # 1D chromatogram check of every gated candidate (kernels.eic_shape): smoothed EIC must peak inside the
    # bounds, with prominence >= eic_k_snr x max(trace residual noise, cell noise); same-ion duplicates on one
    # smoothed maximum are reduced to the strongest
    eic_check: bool = True
    eic_smooth_fwhm: float = 0.5    # smoothing kernel FWHM in median peak FWHMs (kills 1-3 scan spikes, keeps doublets >= 1 FWHM apart)
    eic_half_win_fwhm: float = 10.0  # window half-width in median FWHMs
    eic_k_snr: float = 3.0
    eic_dup_fwhm: float = 0.25      # same-ion maxima closer than this (in FWHMs) are one peak
    far_level_max: float = 0.5      # gate: reject when the ion's typical level 2-6 widths out on the cleaner side exceeds
                                    # this fraction of the apex (a bump on a tail or ridge; 1.0 = off)
    k_noise: float = 3.0
    min_snr: float = 3.0
    min_prominence_rel: float = 0.2
    min_score: float = 0.5     # on the 7-term score; coverage first, the chromatogram check and the cross-file
                               # presence rule remove what the score lets through
    n_mz_bands: int = 16
    rt_block_min: float = 1.0
    noise_cell_min_points: int = 50

    def sig_ppm(self, inten):
        return np.sqrt(self.sig_a ** 2 + self.sig_b ** 2 / np.asarray(inten, dtype=np.float64))

    def to_dict(self):
        d = asdict(self)
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in d.items()}


def strongest_distinct(cloud: Cloud, n: int = 2000, min_sep_ppm: float = SEED_MIN_SEP_PPM,
                       pool: int = 60000):
    """indices of the n most intense centroids that are mutually >= min_sep_ppm apart in m/z
    (greedy, strongest first): one seed per strong ion, at its apex"""
    N = cloud.n_points
    if N == 0:
        return np.zeros(0, np.int64)
    pool = min(pool, N)
    cand = np.argpartition(-cloud.inten, pool - 1)[:pool] if pool < N else np.arange(N)
    cand = cand[np.argsort(-cloud.inten[cand], kind="stable")]
    kept_mz = np.empty(n)
    kept = []
    k = 0
    for p in cand:
        m = cloud.mz[p]
        if k:
            srt = kept_mz[:k]
            # kept_mz is kept sorted by insertion below
            i = np.searchsorted(srt, m)
            tol = m * min_sep_ppm * 1e-6
            if (i < k and srt[i] - m < tol) or (i > 0 and m - srt[i - 1] < tol):
                continue
            kept_mz[i + 1:k + 1] = kept_mz[i:k]
            kept_mz[i] = m
        else:
            kept_mz[0] = m
        kept.append(p)
        k += 1
        if k >= n:
            break
    return np.asarray(kept, np.int64)


def fwhm_at(cloud: Cloud, mz: float, scan: int, ppm: float = SEED_PPM, window: int | None = None):
    """FWHM (minutes) of the EIC apex nearest to ``scan``; None when no half-max crossing is
    found on either side inside the window or fewer than 3 scans sit above half max."""
    S = cloud.n_scans
    if window is None:
        window = max(60, S // 20)
    s0, s1 = max(0, scan - window), min(S, scan + window + 1)
    raw = cloud.eic(mz, ppm, s0, s1).astype(np.float64)
    e = raw
    if len(raw) >= 3:  # 3-point mean: width estimation only, so scan-to-scan ripple does not bias it low
        e = np.convolve(raw, np.ones(3) / 3.0, mode="same")
    rt = cloud.rt[s0:s1]
    k = int(np.argmax(e))
    top = e[k]
    if top <= 0:
        return None
    half = 0.5 * top
    # left crossing: last scan >= half walking left, interpolate to the first below
    i = k
    while i - 1 >= 0 and e[i - 1] >= half:
        i -= 1
    if i == 0:
        return None
    tl = rt[i - 1] + (rt[i] - rt[i - 1]) * (half - e[i - 1]) / (e[i] - e[i - 1])
    j = k
    while j + 1 < len(e) and e[j + 1] >= half:
        j += 1
    if j == len(e) - 1:
        return None
    tr = rt[j] + (rt[j + 1] - rt[j]) * (e[j] - half) / (e[j] - e[j + 1])
    # the smoothed trace must rest on real points: at least 3 raw non-zero scans above half max
    if j - i + 1 < 3 or np.count_nonzero(raw[i:j + 1]) < 3:
        return None
    return float(tr - tl), int(j - i + 1), int(s0 + k)


def trace_deviations(cloud: Cloud, seed_idx: np.ndarray, scan_of_seed: np.ndarray, reach_scans: np.ndarray,
                     ppm: float = SEED_PPM):
    """Per strong ion: the contiguous run of scans around its apex (most intense centroid within +/- ppm
    per scan). Returns lists of (intensities, ppm deviations from the trace's intensity-weighted mean)."""
    I_all, dev_all = [], []
    S = cloud.n_scans
    for p, s, reach in zip(seed_idx, scan_of_seed, reach_scans):
        m0 = cloud.mz[p]
        s0, s1 = max(0, s - int(reach)), min(S, s + int(reach) + 1)
        idx = cloud.eic_argmax(m0, ppm, s0, s1)
        k = s - s0
        lo = k
        while lo - 1 >= 0 and idx[lo - 1] >= 0:
            lo -= 1
        hi = k
        while hi + 1 < len(idx) and idx[hi + 1] >= 0:
            hi += 1
        idx = idx[lo:hi + 1]
        if len(idx) < 4:
            continue
        mzs, ins = cloud.mz[idx], cloud.inten[idx].astype(np.float64)
        mw = np.average(mzs, weights=ins)
        I_all.append(ins)
        dev_all.append((mzs - mw) / mw * 1e6)
    return I_all, dev_all


def fit_sigma(cloud: Cloud, seed_idx: np.ndarray, scan_of_seed: np.ndarray, reach_scans: np.ndarray,
              ppm: float = SEED_PPM, a_floor: float = 0.3, traces=None):
    """Fit sd_ppm(I)^2 = a^2 + b^2 / I from the m/z scatter of each strong ion's own trace: the
    contiguous run of scans around its apex (most intense centroid within +/- ppm per scan),
    so the tails contribute the low-intensity points the model needs."""
    I_all, dev_all = traces if traces is not None else trace_deviations(cloud, seed_idx, scan_of_seed, reach_scans, ppm)
    if not I_all:
        return a_floor, 0.0, 0
    I = np.concatenate(I_all)
    dev = np.concatenate(dev_all)
    n = len(I)
    lg = np.log10(I)
    edges = np.arange(np.floor(lg.min() * 2) / 2, lg.max() + 0.5, 0.5)   # half-decade bins
    b = np.clip(np.searchsorted(edges, lg, side="right") - 1, 0, len(edges) - 2)
    xs, ys = [], []
    for k in range(len(edges) - 1):
        sel = b == k
        if sel.sum() < 20:
            continue
        d = dev[sel]
        sd = 1.4826 * np.median(np.abs(d - np.median(d)))
        if sd > 0:
            xs.append(1.0 / np.median(I[sel]))
            ys.append(sd * sd)
    if len(xs) < 2:
        sd = 1.4826 * np.median(np.abs(dev - np.median(dev))) if n else a_floor
        return float(max(sd, a_floor)), 0.0, n
    A = np.column_stack([np.ones(len(xs)), xs])
    coef, *_ = np.linalg.lstsq(A, np.asarray(ys), rcond=None)
    a2 = max(coef[0], a_floor ** 2)
    b2 = max(coef[1], 0.0)
    return float(np.sqrt(a2)), float(np.sqrt(b2)), n


def fit_intensity_noise(cloud: Cloud, seed_idx: np.ndarray, scan_of_seed: np.ndarray, reach_scans: np.ndarray,
                        ppm: float = SEED_PPM, top_frac: float = 0.1, min_pts: int = 7, min_bin: int = 30):
    """Fit sd(I)^2 = s0^2 + c * I + r^2 * I^2, the scan-to-scan noise of an ion current at level I, from
    the strongest ions' traces: the contiguous scans >= top_frac of each apex, each point compared with
    the 5-point local quadratic (Savitzky-Golay) through its neighbours, apex scan excluded. A local
    quadratic follows any smooth peak shape (tailing, shoulders), so unlike a global Gaussian fit its
    residual is noise, not asymmetry; the residual sd of white noise is sqrt(18/35) of the noise sd,
    which is undone. Residuals are binned by half-decade of level, a robust sd taken per bin, and the
    variance law fitted to the bins by non-negative least squares weighted by 1 / variance (every bin
    counts in relative terms). Returns (s0, c, r, n_points, n_bins); c = r = 0 when fewer than two bins
    are available, so the cell floor alone describes the noise."""
    from scipy.optimize import nnls
    S = cloud.n_scans
    sg = np.array([-3.0, 12.0, 17.0, 12.0, -3.0]) / 35.0
    corr = 1.0 / np.sqrt(1.0 - 17.0 / 35.0)
    lev_all, res_all = [], []
    for p, s, reach in zip(seed_idx, scan_of_seed, reach_scans):
        s0, s1 = max(0, s - int(reach)), min(S, s + int(reach) + 1)
        e = cloud.eic(cloud.mz[p], ppm, s0, s1).astype(np.float64)
        if len(e) < min_pts:
            continue
        k = int(np.argmax(e))
        thr = max(top_frac * e[k], 1e-9)
        lo = k
        while lo - 1 >= 0 and e[lo - 1] >= thr:
            lo -= 1
        hi = k
        while hi + 1 < len(e) and e[hi + 1] >= thr:
            hi += 1
        n = hi - lo + 1
        if n < min_pts:
            continue
        y = e[lo:hi + 1]
        fit = np.convolve(y, sg[::-1], mode="valid")          # fitted value at points 2 .. n-3
        j = np.arange(2, n - 2)
        keep = (j != k - lo) & (fit > 0)
        if not keep.any():
            continue
        res_all.append((y[j][keep] - fit[keep]) * corr)
        lev_all.append(fit[keep])
    if not res_all:
        return float("nan"), 0.0, 0.0, 0, 0
    lev = np.concatenate(lev_all)
    res = np.concatenate(res_all)
    lg = np.log10(lev)
    edges = np.arange(np.floor(lg.min() * 2) / 2, lg.max() + 0.5, 0.5)
    b = np.clip(np.searchsorted(edges, lg, side="right") - 1, 0, len(edges) - 2)
    xs, vs = [], []
    for kb in range(len(edges) - 1):
        sel = b == kb
        if sel.sum() < min_bin:
            continue
        d = res[sel]
        sd = 1.4826 * np.median(np.abs(d - np.median(d)))
        if sd > 0:
            xs.append(float(np.median(lev[sel])))
            vs.append(sd * sd)
    xs, vs = np.asarray(xs), np.asarray(vs)
    if len(xs) < 2:
        return float("nan"), 0.0, 0.0, int(len(res)), int(len(xs))
    cols = [np.ones(len(xs)), xs] + ([xs * xs] if len(xs) >= 3 else [])
    A = np.column_stack(cols) / vs[:, None]                 # rows weighted by 1 / variance -> target 1
    scale = A.max(axis=0)
    x, _ = nnls(A / scale, np.ones(len(xs)))
    x = x / scale
    s0 = float(np.sqrt(x[0]))
    c = float(x[1])
    r = float(np.sqrt(x[2])) if len(x) > 2 else 0.0
    return s0, c, r, int(len(res)), int(len(xs))


def estimate(cloud: Cloud, n_seeds: int = 2000) -> Params:
    P = Params()
    S = cloud.n_scans
    if S < 2 or cloud.n_points == 0:
        raise ValueError("cannot estimate parameters: fewer than 2 scans or no centroids")
    P.dt = float(np.median(np.diff(cloud.rt)))
    P.span = float(cloud.rt[-1] - cloud.rt[0])
    P.floor = float(np.percentile(cloud.inten, 1))
    seeds = strongest_distinct(cloud, n_seeds)
    P.n_seeds = int(len(seeds))
    scan_of = np.searchsorted(cloud.off, seeds, side="right") - 1
    widths = []
    reach = np.full(len(seeds), 8.0)
    for i, (p, s) in enumerate(zip(seeds, scan_of)):
        r = fwhm_at(cloud, cloud.mz[p], int(s))
        if r is not None:
            widths.append(r[0])
            reach[i] = max(8.0, 3.0 * r[0] / P.dt)
    P.n_fwhm_seeds = len(widths)
    if widths:
        P.fwhm_med = float(np.median(widths))
    else:  # no measurable width: assume 4 scans and say so in the sidecar
        P.fwhm_med = 4 * P.dt
    P.fwhm_scans = P.fwhm_med / P.dt
    reach = np.where(reach == 8.0, max(8.0, 3.0 * P.fwhm_scans), reach)
    traces = trace_deviations(cloud, seeds, scan_of, reach)
    P.sig_a, P.sig_b, P.n_sigma_points = fit_sigma(cloud, seeds, scan_of, reach, traces=traces)
    P.noise_s0, P.noise_c, P.noise_r, P.n_noise_pts, P.n_noise_bins = fit_intensity_noise(cloud, seeds, scan_of, reach)
    P.tol_max_ppm = float(np.clip(4 * P.sig_ppm(P.floor), 5.0, 30.0))
    P.K = int(np.clip(round(0.5 * P.fwhm_scans), 1, 3))
    P.min_scans = int(np.clip(round(0.5 * P.fwhm_scans), 3, 5))
    sm_sigma = P.smooth_fwhm * P.fwhm_scans / 2.3548
    P.n_smooth = int(np.clip(round(2.0 * sm_sigma ** 2), 1, 50))
    return P
