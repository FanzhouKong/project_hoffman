"""The denoising API.

denoise(mz, inten, precursor_mz, formula=None, adduct=None, charge=None, params=Params()) -> Result with a keep
mask and a reason code per ion:
  0 kept by the loss rule, 1 precursor ion, 2 kept as an isotopologue of a kept ion, 3 removed: no permitted
  loss, 4 removed: above the precursor, 5 removed: electronic noise (identical intensities), 6 untouched
  (charge > 1 or unparsable formula: the spectrum is passed through, Result.applied is False)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import formula as fm
from .losses import candidate_precursors, explain_t0, explain_t1, iso_rescue, loss_table

REASONS = ("loss", "precursor", "isotope", "forbidden", "above_precursor", "electronic", "untouched")


@dataclass
class Params:
    tol_ppm: float = 10.0        # fragment m/z tolerance, ppm ...
    tol_abs: float = 0.003       # ... with this floor (Da)
    tol_precursor: float = 0.003 # T0: precursor m/z tolerance (Da) for candidate formulas
    top_k: int = 5               # T0: losses permitted by candidates ranked <= top_k (ties share a rank)
    margin: float | None = None  # T0: instead, candidates scoring >= best - margin (capped by top_k) when set
    prune_margin: float = 0.1    # T0: candidates that cannot come within this of the best score are dropped early
    tier: str = "auto"           # "auto": T1 when a usable formula is given, else T0; "t0": ignore the formula;
                                 # "free": formula-free, keep every ion whose loss is some plausible formula
    elements: tuple = ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I")   # T0 candidate element set
    element_caps: dict = None    # T0: per-element count limits overriding the Golden Rules caps (default below)
    even_electron: bool = True   # T0: precursor is an even-electron ion
    electronic: bool = True      # remove groups of >= electronic_n ions with one intensity (+-0.1 %)
    electronic_n: int = 4
    isotopes: bool = True
    recalibrate: bool = True     # T1: measure losses from the given precursor m/z (fragments share its calibration
                                 # offset) when it lies within 20 mDa of the formula's m/z
    precursor_peak: bool = False # shift every m/z by the offset of the observed precursor peak (strongest peak within
                                 # max(20 mDa, 20 ppm) of precursor_mz, >= 2 % of the base peak) from precursor_mz.
                                 # Off: on NIST23 and the dilution series it picked wrong peaks and moved edge ions
                                 # out of tolerance (T1 similarity -0.01 on [M+Cl]-, -0.03 on 0.4 pmol spectra)
    exchange: bool = True        # T1: halide -> OH exchange products (losses of p + H2O carrying a halogen)
    max_candidates: int = 200000
    max_losses: int = 50000
    max_scored: int = 20000      # T0: with more candidates than this the spectrum gets the formula-free rule only


@dataclass
class Result:
    keep: np.ndarray
    reason: np.ndarray
    applied: bool
    tier: int
    candidates: np.ndarray | None = None      # T0: candidate precursor formulas, best first
    scores: np.ndarray | None = None          # T0: explained-intensity fraction per candidate
    rank: np.ndarray | None = None            # T0: per-ion best explaining rank (0 = none)
    best: np.ndarray | None = None            # T0: per-ion score of the best explaining candidate (-1 = none)
    weak: np.ndarray | None = None            # T0: per-ion formula-free test (some plausible loss exists)
    flag: np.ndarray | None = None            # per-ion 0 fragment, 1 precursor, 2 above precursor
    live: np.ndarray | None = None            # per-ion survived the electronic step
    err: np.ndarray | None = None             # T1: per-ion best mass error (inf = none)
    overflow: bool = False                    # T0: too many candidates, formula-free rule applied instead
    offset: float = 0.0                       # m/z shift applied (observed precursor peak - precursor_mz)
    _ctx: tuple | None = None

    def variant(self, top_k=5, margin=None, free=False):
        """keep mask of another T0 decision rule from the same analysis (electronic, precursor, above-precursor and
        isotopologue handling as in the original call)"""
        if self.tier != 0 or self._ctx is None:
            raise ValueError("variant() needs a tier-0 result")
        mz, inten, params, elem = self._ctx
        if free or self.overflow:
            ok = self.weak
        elif margin is not None:
            top = self.scores[0] if len(self.scores) else 0.0
            ok = (self.rank > 0) & (self.rank <= top_k) & (self.best >= top - margin)
        else:
            ok = (self.rank > 0) & (self.rank <= top_k)
        keep = ((self.flag == 1) | ((self.flag == 0) & ok)) & self.live
        reason = np.where(self.flag == 1, 1, np.where(self.flag == 2, 4, np.where(ok, 0, 3))).astype(np.int8)
        reason[~self.live] = 5
        if params.isotopes:
            iso_rescue(mz, inten, keep, reason, params.tol_abs, params.tol_ppm, elem)
        return keep

    @property
    def reasons(self):
        return [REASONS[r] for r in self.reason]


def electronic_mask(inten, n_min=4, rel=0.001):
    """True for ions belonging to a group of >= n_min ions whose intensities agree within rel"""
    order = np.argsort(inten)
    s = inten[order]
    mask = np.zeros(len(inten), bool)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] <= s[i] * (1 + rel):
            j += 1
        if j - i + 1 >= n_min:
            mask[order[i:j + 1]] = True
        i = j + 1
    return mask


DEFAULT_CAPS = {"F": 6, "Cl": 4, "Br": 2, "I": 2, "Si": 2, "Na": 1, "K": 1}   # realistic for small molecules; the
                                                                            # Golden Rules caps (F 34, Cl 12 ...) make the
                                                                            # candidate space explode above ~500 Da


def bounds_for(elements, m, element_caps=None):
    b = np.zeros(fm.NE, np.int64)
    c = fm.caps(m)
    caps = dict(DEFAULT_CAPS, **(element_caps or {}))
    for e in elements:
        b[fm.IDX[e]] = min(c[fm.IDX[e]], caps.get(e, 10 ** 6))
    return b


def _iso_allow(elem_present):
    return np.array([True, bool(elem_present[fm.S]), bool(elem_present[fm.CL]), bool(elem_present[fm.BR])])


def denoise(mz, inten, precursor_mz, formula=None, adduct=None, charge=None, params=Params()):
    mz = np.asarray(mz, np.float64)
    inten = np.asarray(inten, np.float64)
    order = np.argsort(mz, kind="stable")
    if np.any(np.diff(order) < 0):
        r = denoise(mz[order], inten[order], precursor_mz, formula, adduct, charge, params)
        inv = np.empty_like(order); inv[order] = np.arange(len(order))
        r.keep = r.keep[inv]; r.reason = r.reason[inv]
        if r.rank is not None:
            r.rank = r.rank[inv]
        if r.err is not None:
            r.err = r.err[inv]
        return r
    n = len(mz)
    offset = 0.0
    if params.precursor_peak and n:
        win = max(0.02, 20e-6 * precursor_mz)
        near = np.flatnonzero(np.abs(mz - precursor_mz) <= win)
        near = near[inten[near] >= 0.02 * inten.max()]
        if len(near):                 # several peaks in the window: the one closest to precursor_mz is the precursor
            offset = float(mz[near[np.argmin(np.abs(mz[near] - precursor_mz))]] - precursor_mz)
            mz = mz - offset
    keep = np.ones(n, bool)
    reason = np.zeros(n, np.int8)
    if params.electronic:
        m = electronic_mask(inten, params.electronic_n)
        keep[m] = False
        reason[m] = 5
    live = keep.copy()
    p = None
    if formula is not None and params.tier == "auto":
        f = fm.parse(formula)
        pa = fm.precursor_formula(f, adduct) if (f is not None and adduct) else None
        if pa is None and f is not None and charge in (1, -1):
            pa = (f + (1 if charge == 1 else -1) * np.eye(fm.NE, dtype=np.int64)[fm.H], charge)
        if pa is not None:
            p, charge = pa
            if abs(fm.ion_mz(p, charge) - precursor_mz) > 0.02:
                p = None                      # record and formula disagree: fall back to T0
    if charge is None:
        charge = 1
    if abs(charge) != 1:
        reason[:] = 6
        return Result(keep, reason, False, 0)
    if p is not None:
        work = np.zeros((params.max_losses, fm.NE), np.int64)
        err = np.zeros(n)
        err_x = np.zeros(n)
        nexp = np.zeros(n, np.int64)
        flag = np.zeros(n, np.int64)
        pmz_ref = precursor_mz if params.recalibrate else fm.ion_mz(p, charge)
        explain_t1(mz, pmz_ref, charge, p, params.tol_abs, params.tol_ppm, work, err, nexp, flag, err_x)
        ok = np.isfinite(err) | (params.exchange & np.isfinite(err_x))
        if offset != 0.0:             # the peak taken as the precursor may be another ion: keep the shift only if it
            err0 = np.zeros(n); err_x0 = np.zeros(n); nexp0 = np.zeros(n, np.int64); flag0 = np.zeros(n, np.int64)
            explain_t1(mz + offset, pmz_ref, charge, p, params.tol_abs, params.tol_ppm, work, err0, nexp0, flag0, err_x0)
            ok0 = np.isfinite(err0) | (params.exchange & np.isfinite(err_x0))
            if inten[live & (flag0 == 0) & ok0].sum() > inten[live & (flag == 0) & ok].sum():   # explains more intensity
                mz = mz + offset; offset = 0.0
                err, err_x, nexp, flag, ok = err0, err_x0, nexp0, flag0, ok0
        reason[live & (flag == 1)] = 1
        reason[live & (flag == 2)] = 4
        reason[live & (flag == 0) & ~ok] = 3
        keep &= (flag == 1) | ((flag == 0) & ok)
        keep[~live] = False
        if params.isotopes:
            iso_rescue(mz, inten, keep, reason, params.tol_abs, params.tol_ppm, _iso_allow(p > 0))
        return Result(keep, reason, True, 1, err=err, offset=offset)
    bounds = bounds_for(params.elements, precursor_mz * abs(charge), params.element_caps)
    cands = np.zeros((params.max_candidates, fm.NE), np.int64)
    nc = candidate_precursors(precursor_mz, charge, params.tol_precursor, bounds, params.even_electron, cands)
    overflow = nc < 0 or nc > params.max_scored
    if nc < 0:
        nc = params.max_candidates
    if overflow:
        nc = 0
    work = np.zeros((params.max_losses, fm.NE), np.int64)
    rank = np.zeros(n, np.int64)
    nexp = np.zeros(n, np.int64)
    flag = np.zeros(n, np.int64)
    score = np.zeros(max(nc, 1))
    best = np.zeros(n)
    explain_t0(mz, inten, precursor_mz, charge, cands, nc, params.tol_abs, params.tol_ppm, params.tol_precursor,
               loss_table(bounds_for(params.elements, 1e9, params.element_caps)), work, rank, nexp, flag, score, best, params.prune_margin)
    if params.tier == "free" or overflow:
        ok = nexp > 0
    elif params.margin is not None:
        ok = (rank > 0) & (rank <= params.top_k) & (best >= (score[:nc].max() if nc else 0) - params.margin)
    else:
        ok = (rank > 0) & (rank <= params.top_k)
    reason[live & (flag == 1)] = 1
    reason[live & (flag == 2)] = 4
    reason[live & (flag == 0) & ~ok] = 3
    keep &= (flag == 1) | ((flag == 0) & ok)
    keep[~live] = False
    order = np.argsort(-score[:nc], kind="stable")
    top = cands[:nc][order][: max(1, int(np.sum(score[:nc] >= (score[:nc][order[min(params.top_k, nc) - 1]] if nc else 0))))]
    elem = _iso_allow((top > 0).any(axis=0) if nc else np.zeros(fm.NE, bool))
    if params.isotopes:
        iso_rescue(mz, inten, keep, reason, params.tol_abs, params.tol_ppm, elem)
    return Result(keep, reason, True, 0, candidates=cands[:nc][order], scores=score[:nc][order], rank=rank, best=best,
                  weak=nexp > 0, flag=flag, live=live, overflow=overflow, offset=offset, _ctx=(mz, inten, params, elem))
