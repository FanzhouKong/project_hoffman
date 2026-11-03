"""Bounded mass decomposition and the loss / precursor-candidate machinery (numba).

decompose(target, tol, bounds, out) enumerates every count vector f with 0 <= f <= bounds and
|mass(f) - target| <= tol by depth-first search over the elements in formula.ELEMENTS order (heaviest first);
hydrogen, the last element, is solved directly, and a branch is cut as soon as the remaining mass cannot be
reached with the remaining elements. It fills out[:n] and returns n (negative n: out was too small)."""
from __future__ import annotations

import os

import numpy as np
from numba import njit

from .formula import (BR, C, CL, ELECTRON, F, H, I, MASSES, NE, O, _CAPS, hmax, ion_ok, loss_ok, precursor_ok)

CAPS = _CAPS
MH2O = 2 * MASSES[H] + MASSES[O]


@njit(cache=True)
def fmass(f):
    m = 0.0
    for e in range(NE):
        m += f[e] * MASSES[e]
    return m


@njit(cache=True)
def decompose(target, tol, bounds, out):
    ne = NE
    maxrest = np.zeros(ne)
    for i in range(ne - 2, -1, -1):
        maxrest[i] = maxrest[i + 1] + bounds[i + 1] * MASSES[i + 1]
    counts = np.zeros(ne, np.int64)
    partial = np.zeros(ne + 1)
    cap = out.shape[0]
    n = 0
    i = 0
    counts[0] = -1
    mh = MASSES[ne - 1]
    while i >= 0:
        if i == ne - 1:
            r = target - partial[i]
            h = int(np.floor(r / mh + 0.5))
            if h >= 0 and h <= bounds[i] and abs(r - h * mh) <= tol:
                if n >= cap:
                    return -1
                for e in range(ne - 1):
                    out[n, e] = counts[e]
                out[n, ne - 1] = h
                n += 1
            i -= 1
            continue
        counts[i] += 1
        if counts[i] > bounds[i]:
            counts[i] = -1
            i -= 1
            continue
        partial[i + 1] = partial[i] + counts[i] * MASSES[i]
        r = target - partial[i + 1]
        if r < -tol:
            counts[i] = -1
            i -= 1
            continue
        if r > maxrest[i] + tol:
            continue
        i += 1
        counts[i] = -1
    return n


@njit(cache=True)
def candidate_precursors(pmz, charge, tol, bounds, even_electron, out):
    """precursor ion formulas whose m/z lies within tol of pmz, obeying precursor_ok; returns n (or -1)"""
    target = pmz * abs(charge) + charge * ELECTRON       # mass of the atoms
    n = decompose(target, tol, bounds, out)
    if n < 0:
        return n
    k = 0
    for j in range(n):
        f = out[j]
        if precursor_ok(f, fmass(f), CAPS, even_electron):
            if k != j:
                out[k, :] = f
            k += 1
    return k


@njit(cache=True)
def _t1_loss_ok(lo, ml, caps):
    """T1 loss rule: organic losses as loss_ok; a carbon-free loss is any sub-formula of the known precursor with at
    least one heavy atom (or H..H4) and hydrogens within the saturated bound (Na2HPO4, Na3PO3, NaCl, H3PO4 ...)"""
    if lo[C] > 0:
        return loss_ok(lo, ml, caps)
    heavy = 0
    for e in range(NE - 1):
        heavy += lo[e]
    if heavy == 0:
        return lo[H] >= 1 and lo[H] <= 4
    return lo[H] <= hmax(lo, 2)


@njit(cache=True)
def _t1_ion_ok(fr, mf, caps):
    """T1 fragment rule: organic ions as ion_ok; a carbon-free fragment ion is any sub-formula of the known precursor
    with a heavy atom and hydrogens within the saturated bound (Na2PO3+, Na3SO3+, Na3H2P2O7+, K+ ...)"""
    if fr[C] > 0:
        return ion_ok(fr, mf, caps)
    heavy = 0
    for e in range(NE - 1):
        heavy += fr[e]
    return heavy >= 1 and fr[H] <= hmax(fr, 2)


@njit(cache=True)
def explain_t1(mz, pmz_theo, charge, p, tol_abs, tol_ppm, work, err, nexp, flag, err_x):
    """Tier 1: one known precursor ion formula p (its exact m/z pmz_theo). For every ion: err = smallest
    |observed - theoretical| over the sub-formula losses L of p with loss_ok(L) and ion_ok(p - L), within
    the tolerance max(tol_abs, tol_ppm * mz); nexp = number of such losses; flag 1 = precursor ion itself,
    2 = above the precursor. err = inf where nothing explains the ion. err_x: the same over losses of
    p + H2O that carry a halogen (halide -> hydroxyl exchange in the source or the cell; inf when p has no
    halogen)."""
    frag = np.zeros(NE, np.int64)
    px = p.copy()
    px[O] += 1
    px[H] += 2
    halo = p[F] + p[CL] + p[BR] + p[I] > 0
    # the precursor is known: a sub-formula may carry as many atoms of an element as the precursor does, even past
    # the Golden Rules caps (written for neutral molecules: Na, K <= 1, which [M-H+2Na]+ fragments exceed)
    caps = CAPS.copy()
    for r in range(caps.shape[0]):
        for e in range(NE):
            if px[e] > caps[r, e]:
                caps[r, e] = px[e]
    for i in range(mz.shape[0]):
        tol = max(tol_abs, tol_ppm * 1e-6 * mz[i])
        d = pmz_theo - mz[i]
        err[i] = np.inf
        err_x[i] = np.inf
        nexp[i] = 0
        flag[i] = 0
        if abs(d) <= tol:
            flag[i] = 1
            err[i] = abs(d)
            continue
        if d < -tol:
            flag[i] = 2
            continue
        n = decompose(d, tol, p, work)
        if n < 0:
            n = work.shape[0]
        best = np.inf
        k = 0
        for j in range(n):
            lo = work[j]
            ml = fmass(lo)
            if not _t1_loss_ok(lo, ml, caps):
                continue
            for e in range(NE):
                frag[e] = p[e] - lo[e]
            if not _t1_ion_ok(frag, fmass(frag), caps):
                continue
            k += 1
            e_ = abs(d - ml)
            if e_ < best:
                best = e_
        err[i] = best
        nexp[i] = k
        if halo and not (best <= tol):
            n = decompose(d + MH2O, tol, px, work)
            if n < 0:
                n = work.shape[0]
            bx = np.inf
            for j in range(n):
                lo = work[j]
                if lo[F] + lo[CL] + lo[BR] + lo[I] == 0:
                    continue
                ml = fmass(lo)
                if not _t1_loss_ok(lo, ml, caps):
                    continue
                for e in range(NE):
                    frag[e] = px[e] - lo[e]
                if not _t1_ion_ok(frag, fmass(frag), caps):
                    continue
                e_ = abs(d + MH2O - ml)
                if e_ < bx:
                    bx = e_
            err_x[i] = bx


MAX_TABLE = 800.0        # formula-free test: below this loss mass a bitmap of plausible loss masses is used;
BIN = 2e-4               # above it the test passes (plausible formulas are dense at any useful tolerance). Bitmap bin (Da).
_TABLES = {}


@njit(cache=True)
def _mark_losses(maxmass, bounds, bitmap):
    """bitmap[k] = 1 when some formula within bounds passing loss_ok has mass in bin k (width BIN)"""
    ne = NE
    counts = np.zeros(ne, np.int64)
    partial = np.zeros(ne + 1)
    f = np.zeros(ne, np.int64)
    i = 0
    counts[0] = -1
    mh = MASSES[ne - 1]
    nb = bitmap.shape[0]
    while i >= 0:
        if i == ne - 1:
            r = maxmass - partial[i]
            hmax_ = min(bounds[i], int(np.floor(r / mh)))
            for e in range(ne - 1):
                f[e] = counts[e]
            for h in range(hmax_ + 1):
                f[ne - 1] = h
                m = partial[i] + h * mh
                if m > 0 and loss_ok(f, m, CAPS):
                    k = int(m / BIN)
                    if k < nb:
                        bitmap[k] = 1
            i -= 1
            continue
        counts[i] += 1
        if counts[i] > bounds[i]:
            counts[i] = -1
            i -= 1
            continue
        partial[i + 1] = partial[i] + counts[i] * MASSES[i]
        if partial[i + 1] > maxmass:
            counts[i] = -1
            i -= 1
            continue
        i += 1
        counts[i] = -1


def loss_table(bounds):
    """cumulative count of plausible-loss bins up to MAX_TABLE for this element bound vector (cached per process and
    on disk under $SPECDENOISE_CACHE, else specdenoise/.cache)"""
    key = tuple(int(x) for x in bounds)
    if key not in _TABLES:
        cache = os.environ.get("SPECDENOISE_CACHE") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
        path = os.path.join(cache, "losses_" + "_".join(map(str, key)) + f"_{int(MAX_TABLE)}_{BIN:g}.npy")
        if os.path.exists(path):
            _TABLES[key] = np.load(path)
        else:
            bitmap = np.zeros(int(MAX_TABLE / BIN) + 1, np.uint8)
            _mark_losses(MAX_TABLE, np.asarray(bounds, np.int64), bitmap)
            cum = np.concatenate([[0], np.cumsum(bitmap, dtype=np.int64)]).astype(np.int32)
            try:
                os.makedirs(cache, exist_ok=True)
                tmp = f"{path[:-4]}.{os.getpid()}.tmp.npy"     # ends in .npy: np.save would append it otherwise
                np.save(tmp, cum)
                os.replace(tmp, path)
            except OSError:
                pass
            _TABLES[key] = cum
    return _TABLES[key]


@njit(cache=True)
def exists_loss(d, tol, table):
    """formula-free test: some plausible loss formula has mass within tol of d"""
    if d > MAX_TABLE:
        return True
    lo = max(0, int((d - tol) / BIN))
    hi = min(table.shape[0] - 1, int((d + tol) / BIN) + 1)
    return table[hi] - table[lo] > 0


@njit(cache=True)
def _explained_by(cands, c, cmass, fragmass, tol, work, frag):
    """does candidate c explain a fragment ion of atom mass fragmass: a sub-formula loss within tol passing
    loss_ok whose remainder passes ion_ok. The lighter side (loss or fragment) is decomposed, bounded by the
    candidate; the other side is the remainder."""
    target = cmass[c] - fragmass
    if target <= fragmass:
        n = decompose(target, tol, cands[c], work)
        if n < 0:
            n = work.shape[0]
        for j in range(n):
            lo = work[j]
            ml = fmass(lo)
            if not loss_ok(lo, ml, CAPS):
                continue
            for e in range(NE):
                frag[e] = cands[c, e] - lo[e]
            if ion_ok(frag, cmass[c] - ml, CAPS):
                return True
        return False
    n = decompose(fragmass, tol, cands[c], work)
    if n < 0:
        n = work.shape[0]
    for j in range(n):
        fr = work[j]
        mf = fmass(fr)
        if not ion_ok(fr, mf, CAPS):
            continue
        for e in range(NE):
            frag[e] = cands[c, e] - fr[e]
        if loss_ok(frag, cmass[c] - mf, CAPS):
            return True
    return False


@njit(cache=True)
def explain_t0(mz, inten, pmz, charge, cands, ncand, tol_abs, tol_ppm, tol_p, table, work, rank, nexp, flag, score,
               best_score, prune_margin):
    """Tier 0: precursor candidates cands[:ncand] (from candidate_precursors), scored by the intensity of the ions
    they explain. A candidate explains an ion when the ion's atom mass differs from the candidate's by a
    sub-formula loss (decomposition bounded by the candidate, within the ion tolerance) that passes loss_ok and
    leaves a fragment passing ion_ok. Ions are visited by decreasing intensity; after each ion a candidate whose
    score cannot reach best - prune_margin (fraction of the total intensity) even if it explained everything left is
    dropped. rank[i] = 1 + number of candidates scoring strictly more than the best candidate explaining ion i
    (0 = none); best_score[i] = that candidate's final score (-1 = none); nexp[i] = 1 when a plausible loss formula
    exists at all (formula-free test against table); flag 1 = precursor ion, 2 = above the precursor."""
    nion = mz.shape[0]
    nc = ncand
    frag = np.zeros(NE, np.int64)
    cmass = np.zeros(nc)
    alive = np.ones(nc, np.bool_)
    for c in range(nc):
        cmass[c] = fmass(cands[c])
        score[c] = 0.0
    for i in range(nion):
        rank[i] = 0
        nexp[i] = 0
        flag[i] = 0
        best_score[i] = -1.0
    total = 0.0
    for i in range(nion):
        tol = max(tol_abs, tol_ppm * 1e-6 * mz[i])
        d = pmz - mz[i]
        if abs(d) <= tol + tol_p:
            flag[i] = 1
        elif d < -(tol + tol_p):
            flag[i] = 2
        else:
            total += inten[i]
    order = np.argsort(-inten)
    expl = np.zeros((nc, nion), np.bool_)
    remaining = total
    best = 0.0
    n_alive = nc
    for oi in range(nion):
        i = order[oi]
        if flag[i] != 0:
            continue
        tol = max(tol_abs, tol_ppm * 1e-6 * mz[i])
        if exists_loss(pmz - mz[i], tol + tol_p, table):
            nexp[i] = 1
        remaining -= inten[i]
        if n_alive == 0:
            continue
        fragmass = mz[i] * abs(charge) + charge * ELECTRON
        for c in range(nc):
            if alive[c] and _explained_by(cands, c, cmass, fragmass, tol, work, frag):
                expl[c, i] = True
                score[c] += inten[i]
                if score[c] > best:
                    best = score[c]
        for c in range(nc):
            if alive[c] and score[c] + remaining < best - prune_margin * total:
                alive[c] = False
                n_alive -= 1
    for c in range(nc):
        score[c] = score[c] / total if total > 0 else 0.0
    for i in range(nion):
        if flag[i] != 0:
            continue
        b = -1.0
        for c in range(nc):
            if expl[c, i] and score[c] > b:
                b = score[c]
        best_score[i] = b
        if b < 0:
            continue
        r = 1
        for c in range(nc):
            if score[c] > b:
                r += 1
        rank[i] = r


ISO_SPACING = np.array([1.0033548378, 1.9957959, 1.9970499, 1.9979535])     # 13C, 34S, 37Cl, 81Br


@njit(cache=True)
def iso_rescue(mz, inten, keep, reason, tol_abs, tol_ppm, allow):
    """keep (reason 2) every rejected ion (reason 3, or 4 = above the precursor) that sits one isotope spacing
    above a kept ion of at least its intensity; allow[k] enables spacing k (34S/37Cl/81Br only when the
    precursor holds that element). mz must be sorted ascending: partners are found by binary search and ions
    are visited in m/z order, so a chain of isotopologues resolves in one pass."""
    n = mz.shape[0]
    for i in range(n):
        if keep[i] or (reason[i] != 3 and reason[i] != 4):
            continue
        tol = max(tol_abs, tol_ppm * 1e-6 * mz[i])
        for k in range(4):
            if not allow[k]:
                continue
            target = mz[i] - ISO_SPACING[k]
            j = np.searchsorted(mz, target - tol)
            while j < n and mz[j] <= target + tol:
                if keep[j] and inten[j] >= inten[i]:
                    keep[i] = True
                    reason[i] = 2
                    break
                j += 1
            if keep[i]:
                break


@njit(cache=True)
def iso_rescue_batch(off, mz, inten, keep, reason, tol_abs, tol_ppm, slack, allow):
    """iso_rescue over a packed set of spectra: off[i]:off[i+1] delimit spectrum i, slack[i] is added to its
    tolerance, allow[i] is its (4,) enable vector"""
    for i in range(off.shape[0] - 1):
        a, b = off[i], off[i + 1]
        if b - a < 2:
            continue
        all_kept = True
        for j in range(a, b):
            if not keep[j]:
                all_kept = False
                break
        if all_kept:
            continue
        iso_rescue(mz[a:b], inten[a:b], keep[a:b], reason[a:b], tol_abs + slack[i], tol_ppm, allow[i])
