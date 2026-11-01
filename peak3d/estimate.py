"""Per-file parameter estimation. Nothing here is tuned per dataset: every value is read off
the file itself (scan spacing, peak width, m/z error model, intensity floor) and written to
the params sidecar so a reviewer can see what the file told the picker."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import numpy as np

from .io import Cloud

SEED_PPM = 20.0        # EIC width used only for the strongest ions during estimation
SEED_MIN_SEP_PPM = 20.0


# the benchmark name of the Params defaults (manifest/peak3d_variants.tsv) and the switches that make it on top of the
# base picker: G scan_gain, L link_da 2 mDa, R narrow_rule local_median, S min_scans_rule local. The GUI shows it and
# params.json records it; rename it whenever a default change moves the picker's output
VARIANT = "GLRS"
VARIANT_SETS = {"scan_gain": True, "link_da": 0.002, "narrow_rule": "local_median", "min_scans_rule": "local"}


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
    # link_da: absolute floor (Da) on that same-scan link and on the same-apex duplicate test, for low m/z, where an
    # ion's split pieces and side peaks sit a few mDa away (tens of ppm) in every scan; it governs below
    # link_da / (same_scan_floor_ppm x 1e-6) (m/z 400 at 2 mDa). 0 = ppm only. The feature keeps the m/z of its
    # strongest centroid (rep_raw), so absorbed pieces do not move it. Default 2 mDa since 2026-10-22 (variant GL2: +3.6 on
    # the MassCube EIC labels, no truth lost; 3 mDa lost a curated YEAST_NEG ion)
    link_da: float = 0.002
    rep_raw: bool = True
    # fixed
    rel_frac: float = 0.5      # a valley must fall below half the lower apex for two peaks to be two
    far_fwhm: float = 2.0      # ... unless the apexes are at least this many median FWHMs apart: then
    far_frac: float = 0.2      # a valley this far below the lower apex on the smoothed trace suffices (0 = off)
    smooth_fwhm: float = 0.5   # smoothing kernel for that test, FWHM in median peak FWHMs
    n_smooth: int = 1          # derived: [1, 2, 1] passes giving that kernel
    # merge_test "frac": the rel_frac / far rules above; "noise": the lower apex survives when the dip from its
    # smoothed apex to the smoothed saddle exceeds k_valley sd of that difference under the file's noise law
    # (cell floor, noise_c, noise_r; scaled by the noise reduction of the smoothing). No fraction is fixed.
    # "noise_raw": the same test on the raw apex and saddle with the raw noise law (smoothing at the file's median
    # width fills the valley of a doublet narrower than that width). "own": two apexes of one ion are two peaks when the
    # dip (raw apex to raw saddle) exceeds k_valley x sqrt(2) x the noise of that ion's own trace, else the "frac" rules
    # decide (kernels.persistence_merge); the lower apex is then a shoulder, whose chromatogram S/N is judged against
    # the noise of a smoothed difference on that same trace (features.eic_check), not against the neighbourhood
    merge_test: str = "frac"
    # scan_gain: divide every MS1 scan by the gain all its ions share (estimate.scan_gains), then refit the floor and
    # the noise law on the corrected cloud (default since 2026-10-21; prepare() applies it, pick_cloud alone does not);
    # gain_sd / n_gain_scans: robust sd of the log gains and the scans they rest on
    scan_gain: bool = True
    gain_sd: float = 0.0
    n_gain_scans: int = 0
    # width_rule "fwhm": a candidate needs FWHM >= 0.3 x the file's median FWHM and min_scans (3-5, from the median
    # width); "scans": no width test against the file, 3 scans with signal (rise, apex, fall) and no width term in the score;
    # "time": the "fwhm" tests with the median width counted in each candidate's local scans (features.add_local_width),
    # i.e. minimum scans = half the median width / local scan interval (3-5), FWHM + one local interval >= 0.3 x the median
    # width, and the score, side windows and chromatogram check in local scans: data-dependent runs change their MS1
    # interval along the gradient, so one file-wide scan count means different times
    width_rule: str = "fwhm"
    # narrow_rule "median": "too narrow" = FWHM < 0.3 x the file's median FWHM; "local": FWHM (+ the local scan interval
    # under width_rule "time") below the narrow_q quantile of the FWHM of the narrow_k trusted peaks eluting nearest in
    # RT (trusted: pass every other gate, S/N >= 10, Gaussian r2 >= 0.9; the "median" rule when fewer than 30), so a peak
    # is judged against peaks of its own part of the gradient (features.local_narrow_threshold); "local_median": FWHM
    # (+ the local scan interval under "time") below narrow_frac x the median FWHM of the narrow_k candidates eluting
    # nearest in RT that pass every other gate (features.local_median_threshold; the "median" rule when fewer than 30);
    # "off": no "too narrow" test (the scan minimum and the score's scan term stay). Default "local_median" since
    # 2026-10-28 (variant GLRS: synthetic doubles +2 pt, MassCube EIC labels +1, no truth lost)
    narrow_rule: str = "local_median"
    narrow_frac: float = 0.3
    narrow_q: float = 0.02
    narrow_k: int = 100
    # min_scans_rule "file": a candidate needs min_scans scans with signal, half the file's median FWHM counted at the
    # file's median MS1 interval (3-5, one number for the whole file); "local" (with width_rule "fwhm"): the same count at
    # the candidate's own MS1 interval (min_scans_local, features.add_local_width), since a data-dependent run's MS1
    # interval can stretch from 0.4 to 3 s along the gradient. Only the minimum changes; the width test, score and
    # chromatogram check stay as they are (width_rule "time" changes all of them). min_scans_pick (derived): the basin
    # pre-filter in pick_cloud, the lowest local minimum anywhere in the file (= min_scans when the interval is uniform).
    # Default "local" since 2026-10-28 (variant GLRS: keeps 3-4-scan peaks where a DDA run samples MS1 slowly)
    min_scans_rule: str = "local"
    min_scans_pick: int = 3
    # bridge_scans: the walk that sets a peak's bounds (kernels._walk) ends at two consecutive scans at or below the low
    # threshold, a scan without a centroid of the ion counting as low. With bridge_scans > 0 it crosses a hole of up to
    # that many scans without a centroid when the scans on both sides of it hold at least bridge_frac x the apex: a
    # dropout inside the peak body (the centroider folding the ion into a saturated neighbour), not a return to the
    # baseline, which falls before it vanishes. 0 = off
    bridge_scans: int = 0
    bridge_frac: float = 0.5
    # satellite_merge: co-eluting candidates with the same elution profile and within 2 x sat_da of each other are the
    # centroider's pieces of one ion (side peaks lie within sat_da of it) and are folded into the strongest (m/z =
    # height-weighted mean). sat_da is learned per file from the side peaks of its strongest ions (0 = too few found,
    # nothing is merged); n_sat = how many were seen
    satellite_merge: bool = False
    sat_da: float = 0.0
    n_sat: int = 0
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
    # eic_noise "max": prominence vs eic_k_snr x max(trace residual, cell noise), raw-point noise against a smoothed
    # prominence; "law": vs eic_k_snr sd of the smoothed (top - base) difference under the file's noise law with the
    # cell noise as floor; "law_trace": the same with the trace's own residual noise as floor; "own": prominence vs
    # eic_k_snr x the trace's own scan-to-scan noise wherever it has signal (kernels.eic_shape ES_OWN; the cell noise
    # when it has too few scans with signal), so a peak is judged against its own noise, not its neighbourhood's
    eic_noise: str = "max"
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
    return fit_noise_law(np.concatenate(lev_all), np.concatenate(res_all), min_bin)


def fit_noise_law(lev: np.ndarray, res: np.ndarray, min_bin: int = 30):
    """sd(I)^2 = s0^2 + c * I + r^2 * I^2 from noise residuals ``res`` at levels ``lev``: residuals binned by
    half-decade of level, a robust sd per bin, the law fitted to the bins by non-negative least squares
    weighted by 1 / variance. Returns (s0, c, r, n_points, n_bins); c = r = 0 with fewer than two bins."""
    from scipy.optimize import nnls
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


def _loo_weights(rt: np.ndarray) -> np.ndarray:
    """(n_scans, 4): weights of scans j-2, j-1, j+1, j+2 whose sum predicts a value at scan j by the quadratic
    through those four points in actual time (rows without four neighbours are 0)"""
    S = len(rt)
    W = np.zeros((S, 4))
    if S < 5:
        return W
    j = np.arange(2, S - 2)
    x = np.stack([rt[j - 2], rt[j - 1], rt[j + 1], rt[j + 2]], axis=1) - rt[j][:, None]
    X = np.stack([np.ones_like(x), x, x * x], axis=2)                       # (n, 4, 3)
    Xt = np.transpose(X, (0, 2, 1))
    W[j] = np.linalg.solve(Xt @ X, Xt)[:, 0, :]                             # first row of (X'X)^-1 X'
    return W


def scan_gains(cloud: Cloud, P: Params, n_seeds: int = 2000, top_frac: float = 0.1, min_traces: int = 3):
    """Multiplicative gain of every MS1 scan shared by all its ions (AGC / injection-time error, spray flicker),
    read off the strongest ions' traces (the contiguous scans >= top_frac of each apex). Each point's log
    intensity minus the quadratic through its four neighbours in time is that scan's log gain minus the same
    combination of the neighbours' log gains, plus the ion's own noise; the median over the traces covering a
    scan (at least min_traces) gives one equation per scan, and the banded system is solved for all log gains
    at once (a scan-by-scan median alone keeps most of the neighbours' gains). The operator barely sees slow
    components, so the solve is the MAP estimate under gains ~ N(0, sd_g^2): each equation weighted by its
    measurement noise (the pooled spread of the residuals around their scan medians, / sqrt(n traces), x the
    median's sqrt(pi / 2)), ridge 1 / sd_g, and sd_g^2 = (variance of the scan medians - their measurement
    variance) / mean(1 + sum of squared neighbour weights). A file without gain jitter beyond its noise gets
    gains of 1. Returns (gains with median 1, robust sd of the log gains, number of scans with an equation)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr
    S = cloud.n_scans
    if S < 5:
        return np.ones(S), 0.0, 0
    W = _loo_weights(cloud.rt)
    seeds, scan_of = seed_ions(cloud, n_seeds)
    reach = int(max(8, 3 * P.fwhm_scans))
    sc, res = [], []
    for p, s in zip(seeds, scan_of):
        s0, s1 = max(0, int(s) - reach), min(S, int(s) + reach + 1)
        e = cloud.eic(cloud.mz[p], SEED_PPM, s0, s1).astype(np.float64)
        k = int(np.argmax(e))
        thr = max(top_frac * e[k], 1e-9)
        lo, hi = k, k
        while lo - 1 >= 0 and e[lo - 1] >= thr:
            lo -= 1
        while hi + 1 < len(e) and e[hi + 1] >= thr:
            hi += 1
        if hi - lo + 1 < 5:
            continue
        le = np.log(e[lo:hi + 1])
        j = np.arange(2, hi - lo - 1)
        g = s0 + lo + j
        nb = np.stack([le[j - 2], le[j - 1], le[j + 1], le[j + 2]], axis=1)
        sc.append(g)
        res.append(le[j] - np.sum(W[g] * nb, axis=1))
    if not sc:
        return np.ones(S), 0.0, 0
    sc, res = np.concatenate(sc), np.concatenate(res)
    o = np.argsort(sc, kind="stable")
    sc, res = sc[o], res[o]
    rows = np.flatnonzero(np.bincount(sc, minlength=S) >= min_traces)
    rows = rows[(rows >= 2) & (rows < S - 2)]
    if len(rows) < 5:
        return np.ones(S), 0.0, int(len(rows))
    bounds = np.searchsorted(sc, np.concatenate([rows, rows + 1]))
    lo_i, hi_i = bounds[:len(rows)], bounds[len(rows):]
    n = hi_i - lo_i
    m = np.array([np.median(res[a:b]) for a, b in zip(lo_i, hi_i)])
    dev = np.concatenate([res[a:b] - mj for a, b, mj in zip(lo_i, hi_i, m)])
    sd_r = 1.4826 * np.median(np.abs(dev))
    sd_n = np.sqrt(np.pi / 2) * sd_r / np.sqrt(n)              # measurement sd of each scan median
    var_m = (1.4826 * np.median(np.abs(m - np.median(m)))) ** 2
    var_g = (var_m - np.mean(sd_n ** 2)) / np.mean(1.0 + np.sum(W[rows] ** 2, axis=1))
    if not np.isfinite(var_g) or var_g <= 0 or sd_r <= 0:
        return np.ones(S), 0.0, int(len(rows))
    wt = 1.0 / sd_n
    cols = np.stack([rows, rows - 2, rows - 1, rows + 1, rows + 2], axis=1)
    vals = np.concatenate([np.ones((len(rows), 1)), -W[rows]], axis=1) * wt[:, None]
    A = csr_matrix((vals.ravel(), (np.repeat(np.arange(len(rows)), 5), cols.ravel())), shape=(len(rows), S))
    lg = lsqr(A, m * wt, damp=1.0 / np.sqrt(var_g), atol=1e-8, btol=1e-8)[0]
    lg -= np.median(lg[rows])
    sd = float(1.4826 * np.median(np.abs(lg[rows] - np.median(lg[rows]))))
    return np.exp(lg), sd, int(len(rows))


SAT_SEARCH_DA = 0.005   # side peaks are looked for within this distance of a strong ion
SAT_R = 0.9             # a side peak follows its ion's elution profile at least this closely (Pearson r)


def profile_r(cloud: Cloud, mz_a: float, mz_b: float, half_da: float, s0: int, s1: int) -> float:
    """Pearson r of two ions' traces (per-scan maxima within +/- half_da of each m/z) over scans s0..s1-1 where both
    have signal; 0 with fewer than 5 such scans"""
    a = cloud.eic(mz_a, half_da / mz_a * 1e6, s0, s1).astype(np.float64)
    b = cloud.eic(mz_b, half_da / mz_b * 1e6, s0, s1).astype(np.float64)
    m = (a > 0) & (b > 0)
    if m.sum() < 5 or a[m].std() == 0 or b[m].std() == 0:
        return 0.0
    return float(np.corrcoef(a[m], b[m])[0, 1])


def satellite_offsets(cloud: Cloud, seeds: np.ndarray, scan_of: np.ndarray, reach: np.ndarray, P: Params,
                      min_found: int = 20):
    """m/z distances (Da) of the side peaks a centroider reports next to strong ions: in each strong ion's apex scan,
    every weaker centroid within SAT_SEARCH_DA, beyond the ion's own split pieces, whose trace follows the ion's
    elution profile (r >= SAT_R). Returns (95th percentile of those distances, how many), or (0, how many) when fewer
    than min_found were seen."""
    off, mz, inten = cloud.off, cloud.mz, cloud.inten
    S = cloud.n_scans
    found = []
    for p, s, rch in zip(seeds, scan_of, reach):
        s = int(s)
        m0 = float(mz[p])
        a, b = off[s], off[s + 1]
        lo, hi = np.searchsorted(mz[a:b], [m0 - SAT_SEARCH_DA, m0 + SAT_SEARCH_DA])
        if inten[a + lo:a + hi].max() > inten[p]:     # the seed is itself a side peak of a stronger ion
            continue
        own =max(P.same_scan_floor_ppm, 3.0 * float(P.sig_ppm(inten[p]))) * m0 * 1e-6
        s0, s1 = max(0, s - int(rch)), min(S, s + int(rch) + 1)
        for q in range(a + lo, a + hi):
            d = abs(float(mz[q]) - m0)
            if q == p or d <= own or inten[q] > inten[p]:
                continue
            if profile_r(cloud, m0, float(mz[q]), 0.4 * d, s0, s1) >= SAT_R:
                found.append(d)
    if len(found) < min_found:
        return 0.0, len(found)
    return float(np.percentile(found, 95)), len(found)


def seed_ions(cloud: Cloud, n_seeds: int = 2000):
    """the strongest distinct ions (estimate's seeds) and their apex scans"""
    seeds = strongest_distinct(cloud, n_seeds)
    return seeds, np.searchsorted(cloud.off, seeds, side="right") - 1


def apply_scan_gains(cloud: Cloud, gains: np.ndarray) -> None:
    """divide every scan's intensities by its gain, in place"""
    cloud.inten /= np.repeat(gains, np.diff(cloud.off)).astype(cloud.inten.dtype)


def prepare(cloud: Cloud, overrides: dict | None = None) -> Params:
    """the picker's parameters for this cloud (estimate), and with scan_gain on: every scan divided by its gain, in
    place, then the floor and noise law refit on the corrected cloud. Call before pick_cloud."""
    P = estimate(cloud, overrides=overrides)
    if P.scan_gain:
        g, g_sd, g_n = scan_gains(cloud, P)
        apply_scan_gains(cloud, g)
        P = refit_after_gains(cloud, P)
        P.gain_sd, P.n_gain_scans = g_sd, g_n
    return P


def refit_after_gains(cloud: Cloud, P: Params, n_seeds: int = 2000) -> Params:
    """after apply_scan_gains: the values a scan-wide gain changes, measured again on the corrected cloud (the
    intensity floor and the noise law); widths and the m/z error model do not depend on a few-percent gain"""
    from dataclasses import replace
    seeds, scan_of = seed_ions(cloud, n_seeds)
    reach = np.full(len(seeds), max(8.0, 3.0 * P.fwhm_scans))
    s0, c, r, n_pts, n_bins = fit_intensity_noise(cloud, seeds, scan_of, reach)
    return replace(P, floor=float(np.percentile(cloud.inten, 1)), noise_s0=s0, noise_c=c, noise_r=r,
                   n_noise_pts=n_pts, n_noise_bins=n_bins)


MODES = {"merge_test": ("frac", "noise", "noise_raw", "own"), "eic_noise": ("max", "law", "law_trace", "own"),
         "width_rule": ("fwhm", "scans", "time"), "narrow_rule": ("median", "local", "local_median", "off"),
         "min_scans_rule": ("file", "local")}


def parse_overrides(items) -> dict:
    """["name=value", ...] (the CLI's --set) -> {name: typed value} for Params fields; unknown names,
    unparsable values and unknown modes raise ValueError"""
    types = {f.name: f.type for f in fields(Params)}
    out = {}
    for it in items or ():
        name, sep, val = str(it).partition("=")
        name, val = name.strip(), val.strip()
        if not sep or name not in types:
            raise ValueError(f"--set {it!r}: expected NAME=VALUE with NAME a picker parameter")
        t = types[name]
        if t == "bool":
            if val.lower() not in ("true", "false", "1", "0"):
                raise ValueError(f"--set {it!r}: expected true or false")
            out[name] = val.lower() in ("true", "1")
        elif t == "int":
            out[name] = int(val)
        elif t == "float":
            out[name] = float(val)
        else:
            out[name] = val
        if name in MODES and out[name] not in MODES[name]:
            raise ValueError(f"--set {it!r}: {name} must be one of {', '.join(MODES[name])}")
    return out


def variant_label(overrides: dict | None = None) -> str:
    """VARIANT, plus every override that moves a field off its default ("GLRS + merge_test=noise")"""
    d = Params()
    extra = [f"{k}={v}" for k, v in sorted((overrides or {}).items()) if getattr(d, k) != v]
    return " + ".join([VARIANT, " ".join(extra)]) if extra else VARIANT


def estimate(cloud: Cloud, n_seeds: int = 2000, overrides: dict | None = None) -> Params:
    """overrides: fixed Params fields to change before estimation (derived values follow them); a field
    the file's own measurements set cannot be overridden"""
    overrides = dict(overrides or {})
    P = Params(**overrides)
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
    P.min_scans = 3 if P.width_rule in ("scans", "time") else int(np.clip(round(0.5 * P.fwhm_scans), 3, 5))
    P.min_scans_pick = P.min_scans
    if P.min_scans_rule == "local" and P.width_rule == "fwhm":
        # the slowest local MS1 interval (median of the 4 intervals around a scan, as features.add_local_width takes it)
        d = np.diff(cloud.rt)
        dt_hi = float(np.median(np.lib.stride_tricks.sliding_window_view(d, 4), axis=1).max()) if len(d) >= 4 else float(d.max())
        P.min_scans_pick = int(np.clip(round(0.5 * P.fwhm_med / dt_hi), 3, 5)) if dt_hi > 0 else 3
    sm_sigma = P.smooth_fwhm * P.fwhm_scans / 2.3548
    P.n_smooth = int(np.clip(round(2.0 * sm_sigma ** 2), 1, 50))
    if P.satellite_merge:
        P.sat_da, P.n_sat = satellite_offsets(cloud, seeds, scan_of, reach, P)
    for k, v in overrides.items():
        if getattr(P, k) != v:
            raise ValueError(f"{k} is measured from the file and cannot be set")
    return P
