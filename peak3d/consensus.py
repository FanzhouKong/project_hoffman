"""One MS2 spectrum per feature from all the spectra mapped to it (any file), weighted by precursor abundance.

The more of the precursor ion there was when a spectrum was taken, the more of its fragment ions were counted, so
each spectrum is weighted by its precursor's intensity in the MS1 scan before it (io.precursor_in_ms1). Fragments
of the compound come back in every good spectrum; noise ions do not, so a peak needs a second spectrum to stay and
noise cannot pile up as spectra are added.

1. Each spectrum is cleaned as for the search (similarity.clean: the precursor region cut, peaks under 1 % of
   the strongest dropped, normalised to sum 1).
2. Spectra are left out when their precursor was absent from the MS1 scan before them or under
   MIN_PRECURSOR_SHARE of the strongest precursor among them; at most MAX_SPECTRA are kept, most abundant first.
   Optional (off by default): purity < min_purity (the precursor a minor ion of its isolation window).
3. Optional agreement gate (off by default, min_similarity = 0): the medoid maximises its own weight plus the
   weighted entropy similarity of the others to it; spectra less similar to it than min_similarity are left out; of
   two spectra that disagree, the more abundant one is kept alone. Off because heavy noise makes good spectra look
   unlike each other and the gate then falls back to one spectrum, noise and all; the vote alone keeps noise out
   (results/ms2_consensus_2026-10-29).
4. Peaks within the fragment tolerance are matched across spectra, strongest (weight x share) first, at most one
   per spectrum. Every spectrum that holds the peak votes for it with its weight; a spectrum without it votes
   against only when the peak, at its share so far, would have stood DETECT_MARGIN times above the weakest peak
   that spectrum could show (its lowest recorded intensity, or the 1 % cut); weaker spectra abstain.
5. A peak stays when at least MIN_PRESENT spectra hold it and they carry at least VOTE of the voting weight. With
   protect=True every peak of the spectrum with the most abundant precursor stays as well.
6. A kept peak's intensity is the weighted mean of its share over the voters (0 for those without it); its m/z
   the weighted mean of theirs. The result is normalised to sum 1.

A feature with one usable spectrum keeps that spectrum (cleaned)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit

from .similarity import NOISE, PRECURSOR_MARGIN, clean_raw, score

MIN_PURITY = 0.0               # off: recorded in ms2_map.tsv, not used, until tested across files
MIN_PRECURSOR_SHARE = 0.01
MAX_SPECTRA = 20
MIN_SIMILARITY = 0.0           # off (see 3.)
MIN_PRESENT = 2
VOTE = 0.5
DETECT_MARGIN = 1.5


@dataclass
class Consensus:
    mz: np.ndarray                     # ascending
    inten: np.ndarray                  # sums to 1
    used: np.ndarray                   # indices of the input spectra merged
    top: int                           # index of the input spectrum with the most abundant precursor among them
    status: list = field(default_factory=list)   # per input spectrum: used, purity, no_precursor, weak, beyond,
    n_dropped: int = 0                 # empty, dissimilar; peaks the vote removed
    n_protected: int = 0               # peaks kept only because the top spectrum holds them


@njit(cache=True)
def _merge(pm, pi, ps, w, floor, tol, min_present, vote, margin, protect):
    """the vote of steps 4-6 over one feature's pooled peaks (pm ascending; pi shares; ps spectrum index):
    (m/z, intensity, peaks dropped, peaks kept only by protection), m/z ascending, intensity not normalised"""
    n = len(pm)
    ns = len(w)
    assigned = np.zeros(n, np.bool_)
    order = np.argsort(-(w[ps] * pi))
    out_m = np.empty(n)
    out_i = np.empty(n)
    best = np.empty(ns, np.int64)
    nc = 0
    dropped = 0
    by_protect = 0
    for k in order:
        if assigned[k]:
            continue
        c = pm[k]
        lo = np.searchsorted(pm, c - tol)
        hi = np.searchsorted(pm, c + tol, side="right")
        best[:] = -1
        for j in range(lo, hi):
            if assigned[j]:
                continue
            s = ps[j]
            if best[s] < 0 or abs(pm[j] - c) < abs(pm[best[s]] - c):
                best[s] = j
        wp = 0.0
        num = 0.0
        mzn = 0.0
        npres = 0
        held = False
        for s in range(ns):
            j = best[s]
            if j >= 0:
                assigned[j] = True
                wp += w[s]
                num += w[s] * pi[j]
                mzn += w[s] * pi[j] * pm[j]
                npres += 1
                if s == protect:
                    held = True
        share = num / wp
        wv = wp
        for s in range(ns):
            if best[s] < 0 and share >= margin * floor[s]:
                wv += w[s]
        voted = npres >= min_present and wp >= vote * wv
        if voted or held:
            out_m[nc] = mzn / num
            out_i[nc] = num / wv
            nc += 1
            if not voted:
                by_protect += 1
        else:
            dropped += 1
    o = np.argsort(out_m[:nc])
    return out_m[:nc][o], out_i[:nc][o], dropped, by_protect


def _clean(mz, inten, prec, tol):
    """(m/z, share, floor share) of one spectrum: cleaned, and the smallest share it could have shown"""
    mz = np.asarray(mz, np.float64)
    inten = np.asarray(inten, np.float64)
    m, i = clean_raw(mz, inten, float(prec), float(tol))
    if not len(m):
        return m, i, 1.0
    total = i.sum()
    raw = inten[(inten > 0) & np.isfinite(inten) & (mz < prec - PRECURSOR_MARGIN if prec > 0 else True)]
    floor = max(NOISE * i.max(), raw.min() if len(raw) else 0.0) / total
    return m, i / total, floor


def build(spectra, prec_int, purity, prec_mz: float, tol: float = 0.02, protect: bool = False, weight=None,
          min_purity: float = MIN_PURITY, max_spectra: int = MAX_SPECTRA, min_similarity: float = MIN_SIMILARITY,
          min_present: int = MIN_PRESENT, vote: float = VOTE) -> Consensus | None:
    """the consensus spectrum of one feature (module docstring). spectra: [(m/z, intensity), ...]; prec_int and
    purity per spectrum (nan = not measured); prec_mz: the feature's m/z, for cleaning. weight: per-spectrum
    weights instead of the precursor intensity (comparisons only). Returns None when no spectrum is usable."""
    n = len(spectra)
    prec_int = np.asarray(prec_int, np.float64)
    purity = np.asarray(purity, np.float64)
    status = ["used"] * n
    cleaned = [_clean(m, i, prec_mz, tol) for m, i in spectra]
    known = np.isfinite(prec_int) & (prec_int > 0)
    measured = bool(known.any())       # else (not recorded, or never found in MS1) no precursor gates, TIC weights
    for k in range(n):
        if not len(cleaned[k][0]):
            status[k] = "empty"
        elif measured and not known[k]:
            status[k] = "no_precursor"
        elif np.isfinite(purity[k]) and purity[k] < min_purity:
            status[k] = "purity"
    tic = np.array([float(np.sum(i)) for _, i in spectra])
    if weight is None:
        weight = np.where(known, prec_int, 0.0) if measured else tic
    weight = np.asarray(weight, np.float64)
    ok = [k for k in range(n) if status[k] == "used"]
    if not ok:
        return None
    if known[ok].any():
        top_int = prec_int[ok][known[ok]].max()
        for k in ok:
            if known[k] and prec_int[k] < MIN_PRECURSOR_SHARE * top_int:
                status[k] = "weak"
    ok = sorted((k for k in ok if status[k] == "used"), key=lambda k: -weight[k])
    for k in ok[max_spectra:]:
        status[k] = "beyond"
    ok = ok[:max_spectra]
    if len(ok) > 1 and min_similarity > 0:
        sim = np.eye(len(ok))
        for a in range(len(ok)):
            for b in range(a + 1, len(ok)):
                ma, ia, _ = cleaned[ok[a]]
                mb, ib, _ = cleaned[ok[b]]
                sim[a, b] = sim[b, a] = score(ma, ia, mb, ib, 0, float(tol))[0]
        w = weight[ok]
        med = int(np.argmax(sim @ w))
        keep = [ok[a] for a in range(len(ok)) if sim[a, med] >= min_similarity]
        if len(ok) == 2 and len(keep) < 2:
            keep = [ok[0]]                         # two that disagree: the more abundant one
        for k in ok:
            if k not in keep:
                status[k] = "dissimilar"
        ok = sorted(keep, key=lambda k: -weight[k])
    top = max(ok, key=lambda k: (prec_int[k] if known[k] else -1.0, weight[k]))
    if len(ok) == 1:
        m, i, _ = cleaned[ok[0]]
        return Consensus(m, i, np.array(ok), top, status)
    pm = np.concatenate([cleaned[k][0] for k in ok])
    pi = np.concatenate([cleaned[k][1] for k in ok])
    ps = np.concatenate([np.full(len(cleaned[k][0]), a, np.int64) for a, k in enumerate(ok)])
    o = np.argsort(pm, kind="stable")
    floor = np.array([cleaned[k][2] for k in ok])
    m, i, dropped, by_protect = _merge(pm[o], pi[o], ps[o], weight[ok] / weight[ok].sum(), floor, float(tol),
                                       int(min_present), float(vote), DETECT_MARGIN,
                                       ok.index(top) if protect else -1)
    if not len(m):                                 # nothing confirmed: the most abundant spectrum alone
        for k in ok:
            if k != top:
                status[k] = "unconfirmed"
        m, i, _ = cleaned[top]
        return Consensus(m, i, np.array([top]), top, status, dropped, 0)
    return Consensus(m, i / i.sum(), np.array(ok), top, status, dropped, by_protect)
