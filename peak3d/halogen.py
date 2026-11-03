"""The MS1 isotope pattern of one feature, and a test for halogens (Cl, Br) in it.

A chlorine atom adds a peak two mass units above the monoisotopic one at 32 % of its height (37Cl), a bromine at 97 %
(81Br). Carbon, nitrogen, oxygen and hydrogen leave M+2 at a few percent for a small molecule, sulfur adds 4.4 % per
atom. ``halogen_test`` fits the observed pattern (M+1 .. M+4 relative to M) under elemental hypotheses - no halogen,
S1, S2, Cl1-4, Br1-3, BrCl, Br2Cl, BrCl2 - each with the carbon count chosen to reproduce M+1 and the rest of the
skeleton scaled like an average organic molecule, and compares the fits (a chi-square with a 15 % relative tolerance).
Where the mass accuracy allows, the spacing of M+2 above M adds evidence: 37Cl and 81Br sit 0.0097 and 0.0088 Da
below 13C2, and 34S 0.0109 Da below. The call is "halogen" when the best halogen hypothesis beats the best
halogen-free one by 9 chi-square units (3 sigma), "possible" above 4, otherwise "none".

The composition reported with each hypothesis is a pattern model, not an identification: the carbon count comes
from M+1, and nitrogen, oxygen and hydrogen are then chosen so that the ion's monoisotopic mass equals the measured
m/z times the charge (hydrogen is isotopically almost silent, so this does not move the pattern).

``extract_pattern`` reads the pattern of a feature out of a point cloud: the apex scan on the feature's own trace,
the intensity in a window at each M+k over the top half of the peak, and whether each window co-elutes with M.
``test_groups`` does the same for many features of one file at once (the pipeline's possible_halogen column).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from numba import njit

from .io import C13, Cloud, _lower_bound, _upper_bound

ELECTRON = 0.00054858
VERDICTS = ("", "no", "maybe", "yes")      # possible_halogen column: not tested, none, possible, halogen

# (mass shift above the lightest isotope, abundance) per nominal offset; IUPAC representative values
_ISOTOPES = {
    "C": [(0.0, 0.9893), (1.0033548, 0.0107)],
    "H": [(0.0, 0.999885), (1.0062767, 0.000115)],
    "N": [(0.0, 0.99636), (0.9970349, 0.00364)],
    "O": [(0.0, 0.99757), (1.0042169, 0.00038), (2.0042460, 0.00205)],
    "S": [(0.0, 0.9499), (0.9993878, 0.0075), (1.9957959, 0.0425), (0.0, 0.0), (3.9950096, 0.0001)],
    "Cl": [(0.0, 0.7576), (0.0, 0.0), (1.9970499, 0.2424)],
    "Br": [(0.0, 0.5069), (0.0, 0.0), (1.9979535, 0.4931)],
    "P": [(0.0, 1.0)],
}
MONO_MASS = {"C": 12.0, "H": 1.0078250, "N": 14.0030740, "O": 15.9949146, "S": 31.9720711, "Cl": 34.9688527,
             "Br": 78.9183376, "P": 30.9737620}
# the average organic molecule (Senko's averagine), atoms per Da; sulfur is a hypothesis of its own here
AVERAGINE = {"C": 4.9384 / 111.1254, "H": 7.7583 / 111.1254, "N": 1.3577 / 111.1254, "O": 1.4773 / 111.1254}
HYPOTHESES = [("none", {}), ("S", {"S": 1}), ("S2", {"S": 2}),
              ("Cl", {"Cl": 1}), ("Cl2", {"Cl": 2}), ("Cl3", {"Cl": 3}), ("Cl4", {"Cl": 4}),
              ("Br", {"Br": 1}), ("Br2", {"Br": 2}), ("Br3", {"Br": 3}),
              ("BrCl", {"Br": 1, "Cl": 1}), ("Br2Cl", {"Br": 2, "Cl": 1}), ("BrCl2", {"Br": 1, "Cl": 2})]
K_FIT = 4             # M+1 .. M+4 enter the fit
REL_TOL = 0.15        # relative tolerance on an isotope ratio (instrument + ion statistics)
ABS_TOL = 0.01        # absolute tolerance on a ratio, so tiny peaks never dominate
CLIP = 4.0            # a spacing term counts at most 4 sigma: a skewed centroid of a half-resolved doublet
MASS_SLACK = 0.03     # Da; a composition further off the ion mass (the carbons M+1 demands leave no room) is penalised
MIN_M2 = 0.12         # a halogen call needs at least this much M+2 (one Cl gives 0.32)
R_MIN = 0.6           # an isotope window must co-elute with M this well to enter the fit
STRONG, POSSIBLE = 9.0, 4.0


@njit(cache=True)
def _conv(pa, ma, pb, mb):
    """the pattern of two parts together: probability per nominal offset, and the mean mass shift of that offset"""
    K = len(pa)
    p = np.zeros(K)
    m = np.zeros(K)
    for i in range(K):
        if pa[i] == 0:
            continue
        for j in range(K - i):
            if pb[j] == 0:
                continue
            w = pa[i] * pb[j]
            p[i + j] += w
            m[i + j] += w * (ma[i] + mb[j])
    for k in range(K):
        if p[k] > 0:
            m[k] /= p[k]
    return p, m


@lru_cache(maxsize=4096)
def _power(el: str, n: int, K: int):
    """the pattern of n atoms of one element (by repeated squaring), cached: the same counts recur across features"""
    shifts, probs = np.zeros(K), np.zeros(K)
    for k, (sh, a) in enumerate(_ISOTOPES[el][:K]):
        shifts[k], probs[k] = sh, a
    q, mq = np.zeros(K), np.zeros(K)
    q[0] = 1.0
    while n:
        if n & 1:
            q, mq = _conv(q, mq, probs, shifts)
        n >>= 1
        if n:
            probs, shifts = _conv(probs, shifts, probs, shifts)
    return q, mq


def pattern(composition: dict, K: int = 6):
    """the isotope pattern of a composition {element: count}: (probability, mean mass shift) per offset 0..K-1"""
    p = np.zeros(K)
    p[0] = 1.0
    m = np.zeros(K)
    for el, n in composition.items():
        if n > 0:
            q, mq = _power(el, int(n), K)
            p, m = _conv(p, m, q, mq)
    return p, m


@njit(cache=True)
def _balance(left, n_c, n_avg, o_avg, m_h, m_n, m_o, m_p):
    """N near its averagine share, O and P (monoisotopic, so free for the pattern) whatever the mass needs, H the
    remainder with 0 <= H <= 2C + N + 3: the combination closest to the mass left, nearest averagine on ties.
    Returns (n, o, h, p) or (-1, ...) when nothing fits"""
    h_max_base = 2 * n_c + 3
    best_n, best_o, best_h, best_p = -1, 0, 0, 0
    best_score = 1e300
    for n_n in range(max(0, n_avg - 6), n_avg + 7):
        h_max = h_max_base + n_n
        for n_p in range(0, 4):
            heavy = left - m_n * n_n - m_p * n_p
            if heavy < -0.5:
                break
            o_lo = max(0, int(math.ceil((heavy - m_h * (h_max + 0.5)) / m_o)))
            o_hi = int(math.floor(heavy / m_o)) + 1
            for n_o in range(o_lo, o_hi + 1):
                h_mass = heavy - m_o * n_o
                n_h = int(round(h_mass / m_h))
                if n_h < 0 or n_h > h_max:
                    continue
                err = abs(h_mass - m_h * n_h)
                score = err + 0.001 * (abs(n_n - n_avg) + abs(n_o - o_avg) + 2 * n_p)
                if score < best_score:
                    best_n, best_o, best_h, best_p, best_score = n_n, n_o, n_h, n_p, score
    return best_n, best_o, best_h, best_p


def _skeleton(mass: float, extra: dict, rel1: float):
    """the organic part of an ion of this monoisotopic mass carrying the extra atoms: the carbon count chosen so the
    predicted M+1 matches the observed one (at most what the mass holds), nitrogen and oxygen near averagine
    proportions and hydrogen filling the rest, so the composition's mass is the measured one to a few mDa.
    Returns (composition, carbon_capped)."""
    rest = mass - sum(MONO_MASS[e] * n for e, n in extra.items())
    if rest < 12.0:
        return None, False
    avg = {e: rest * f for e, f in AVERAGINE.items()}
    r1 = {e: _ISOTOPES[e][1][1] / _ISOTOPES[e][0][1] if len(_ISOTOPES[e]) > 1 else 0.0 for e in _ISOTOPES}
    other = r1["H"] * avg["H"] + r1["N"] * avg["N"] + r1["O"] * avg["O"] + sum(r1[e] * n for e, n in extra.items())
    cap = int(rest // 12.0)
    n_c = int(round((rel1 - other) / r1["C"])) if np.isfinite(rel1) else int(round(avg["C"]))
    capped = n_c > cap
    n_c = min(max(n_c, 0), cap)
    left = rest - 12.0 * n_c
    n_avg = int(round(avg["N"]))
    o_avg = int(round(avg["O"]))
    best = _balance(left, n_c, n_avg, o_avg, MONO_MASS["H"], MONO_MASS["N"], MONO_MASS["O"], MONO_MASS["P"])
    if best[0] < 0:             # the carbons alone are near the mass: shed N and O and take what hydrogen fits
        n_h = max(int(round(left / MONO_MASS["H"])), 0)
        best = (0, 0, min(n_h, 2 * n_c + 3), 0)
    comp = {"C": n_c, "H": best[2], "N": best[0], "O": best[1]}
    if best[3]:
        comp["P"] = best[3]
    comp.update(extra)
    return comp, capped


def formula(comp: dict) -> str:
    order = ["C", "H", "N", "O", "P", "S", "Cl", "Br"]
    return "".join(f"{e}{comp[e] if comp[e] != 1 else ''}" for e in order if comp.get(e, 0) > 0)


def mass_of(comp: dict) -> float:
    return sum(MONO_MASS[e] * n for e, n in comp.items())


def halogen_test(rel, shift, mz: float, z: int = 1, ppm: float = 5.0, noise_rel: float = 0.0, usable=None,
                 polarity: str = "", quick: bool = False) -> dict:
    """Test one isotope pattern for halogens.

    rel:   intensity of M+k relative to M for k = 0 .. K-1 (rel[0] = 1; nan = not measured)
    shift: (m/z of M+k - m/z of M) * z in Da, nan when unknown; from the same scans as rel
    mz:    m/z of M; z its charge; ppm the run's mass tolerance; noise_rel the noise floor relative to M
    usable: per k, whether the window is clean (co-elutes with M); unusable ones are left out of the fit
    polarity: "pos" / "neg" / "": the ion's composition is balanced to the ion mass (the electrons it lost or gained)
    quick: when M+2 is too small for any halogen call (or not usable), fit the halogen-free hypotheses only
    """
    rel = np.asarray(rel, float)
    shift = np.asarray(shift, float)
    K = len(rel)
    z = max(int(z), 1)
    use = np.ones(K, bool) if usable is None else np.asarray(usable, bool)
    use &= np.isfinite(rel)
    sign = 1 if polarity == "pos" else -1 if polarity == "neg" else 0
    mass = mz * z + sign * z * ELECTRON      # the ion's atoms (the composition is the ion's: [M+H]+ carries that H)
    sig_da = max(ppm * mz * 1e-6 * z * 0.71, 0.0005)   # one centroid difference, ppm read as ~2 sigma per centroid
    rel1 = rel[1] if K > 1 and use[1] else np.nan
    flags = []
    out = []
    no_call = K <= 2 or not use[2] or not (rel[2] >= MIN_M2)
    for name, extra in HYPOTHESES:
        if quick and no_call and ("Cl" in extra or "Br" in extra):
            continue
        comp, capped = _skeleton(mass, extra, rel1)
        if comp is None:
            continue
        p, m = pattern(comp, K)
        pred = p / p[0]
        err = mass_of(comp) - mass
        # no sane composition near the ion mass under this hypothesis: its halogens plus the carbons M+1 demands
        # overshoot the mass, or leave too little for anything else
        chi2 = min(((abs(err) - MASS_SLACK) / 0.02) ** 2, CLIP ** 2) if abs(err) > MASS_SLACK else 0.0
        for k in range(1, min(K_FIT, K - 1) + 1):
            if not use[k]:
                continue
            sd = math.sqrt((REL_TOL * max(pred[k], rel[k])) ** 2 + ABS_TOL ** 2 + noise_rel ** 2)
            chi2 += ((rel[k] - pred[k]) / sd) ** 2
            if k >= 2 and np.isfinite(shift[k]) and rel[k] >= 0.03 and p[k] > 0:   # where the observed M+k sits
                r = min(abs(shift[k] - m[k]) / sig_da, CLIP)
                chi2 += r * r
        out.append({"name": name, "halogen": any(e in extra for e in ("Cl", "Br")), "chi2": float(chi2),
                    "n_c": int(comp["C"]), "capped": bool(capped), "formula": formula(comp),
                    "mass_error_mda": float(1e3 * err),
                    "rel": [float(v) for v in pred], "shift": [float(v) for v in m]})
    out.sort(key=lambda d: d["chi2"])
    best = out[0] if out else None
    hal = [d for d in out if d["halogen"]]
    free = [d for d in out if not d["halogen"]]
    delta = (free[0]["chi2"] - hal[0]["chi2"]) if hal and free else -math.inf
    none = next((d for d in out if d["name"] == "none"), None)
    m2_excess = float(rel[2] - none["rel"][2]) if none is not None and K > 2 and use[2] else float("nan")
    verdict = "none"
    if hal and best is not None and best["halogen"] and not no_call:
        verdict = "halogen" if delta >= STRONG else "possible" if delta >= POSSIBLE else "none"
    if K > 1 and use[1] and best is not None and best["capped"]:
        flags.append("M+1 is higher than the mass allows: a saturated M or an interference; the carbon count is capped")
    if noise_rel > 0.1:
        flags.append("weak signal: M is within 10x of the noise floor")
        if verdict == "halogen":
            verdict = "possible"
    for k in range(1, K):
        if not use[k] and np.isfinite(rel[k]):
            flags.append(f"M+{k} does not co-elute with M (an interference); left out of the fit")
    if use[2] and K > 2 and rel[2] > 0.08 and verdict == "none" and best is not None and best["name"].startswith("S"):
        flags.append(f"M+2 is explained by sulfur ({best['name']})")
    return {"verdict": verdict, "best": best["name"] if best else None, "formula": best["formula"] if best else None,
            "ion": "+" if sign > 0 else "-" if sign < 0 else "",
            "n_c": best["n_c"] if best else None, "delta_chi2": float(delta) if np.isfinite(delta) else None,
            "m2_excess": None if not np.isfinite(m2_excess) else m2_excess,
            "observed": {"rel": [None if not np.isfinite(v) else float(v) for v in rel],
                         "shift": [None if not np.isfinite(v) else float(v) for v in shift],
                         "usable": [bool(u) for u in use]},
            "hypotheses": out, "flags": flags, "sigma_da": float(sig_da)}


def _corr(a, b):
    if len(a) < 3:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / d if d > 0 else float("nan")


@njit(cache=True)
def _window_sums(off, mz, inten, s0, s1, lo, hi):
    """per scan s0..s1-1 and window j: the intensity sum within [lo[j], hi[j]] and the intensity-weighted m/z sum"""
    n, K = s1 - s0, len(lo)
    I = np.zeros((n, K))
    M = np.zeros((n, K))
    for s in range(s0, s1):
        a0, b0 = off[s], off[s + 1]
        for j in range(K):
            a = _lower_bound(mz, a0, b0, lo[j])
            b = _upper_bound(mz, a0, b0, hi[j])
            for i in range(a, b):
                I[s - s0, j] += inten[i]
                M[s - s0, j] += inten[i] * mz[i]
    return I, M


def _windows(mz, z, ppm, k_lo, k_hi):
    ks = np.arange(k_lo, k_hi + 1)
    tol = mz * ppm * 1e-6
    half = (tol + 0.008 * np.abs(ks)) / z      # 15N at +0.997, 2H at +1.006, 34S at +1.996, 13C2 at +2.007 ...
    centre = mz + ks * C13 / z
    return ks, centre - half, centre + half


def _summarise(I, M, j0):
    """(rel, shift, r, usable, apex, lo, hi, inten, mzk) of window sums over the scans, or None without M"""
    trace = I[:, j0]
    if trace.max() <= 0:
        return None
    n = len(trace)
    apex = int(np.argmax(trace))
    lo = apex
    while lo > 0 and trace[lo - 1] >= 0.5 * trace[apex]:
        lo -= 1
    hi = apex
    while hi < n - 1 and trace[hi + 1] >= 0.5 * trace[apex]:
        hi += 1
    top = slice(lo, hi + 1)
    inten = I[top].sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mzk = M[top].sum(axis=0) / inten
    r = np.array([_corr(I[:, j], trace) for j in range(I.shape[1])])
    rel = inten / inten[j0]
    usable = np.isfinite(rel) & (np.isnan(r) | (r >= R_MIN) | (rel < 0.02))
    with np.errstate(invalid="ignore"):
        shift = (mzk - mzk[j0])
    return rel, shift, r, usable, apex, lo, hi, inten, mzk


def extract_pattern(cloud: Cloud, mz: float, z: int, s0: int, s1: int, ppm: float, k_lo: int = -2, k_hi: int = 5,
                    view: tuple = (-2.3, 5.7)) -> dict | None:
    """the isotope pattern of the ion at mz (charge z) over scans s0..s1-1: a window at each M+k, k_lo..k_hi, wide
    enough for every isotopologue of that offset; the intensity summed over the top half of the peak on the M trace,
    the intensity-weighted m/z, and the correlation of each window's trace with M's. Also the apex scan's centroids
    within view (in Da / z around mz) for display. None when nothing is found at M."""
    z = max(int(z), 1)
    s0, s1 = max(int(s0), 0), min(int(s1), cloud.n_scans)
    if s1 <= s0:
        return None
    ks, lo_e, hi_e = _windows(mz, z, ppm, k_lo, k_hi)
    I, M = _window_sums(cloud.off, cloud.mz, cloud.inten, s0, s1, lo_e, hi_e)
    j0 = int(np.where(ks == 0)[0][0])
    got = _summarise(I, M, j0)
    if got is None:
        return None
    rel, shift, r, usable, apex, lo, hi, inten, mzk = got
    shift = shift * z
    sa = s0 + apex
    m = cloud.mz[cloud.off[sa]:cloud.off[sa + 1]]
    v = cloud.inten[cloud.off[sa]:cloud.off[sa + 1]]
    a, b = np.searchsorted(m, mz + view[0] / z), np.searchsorted(m, mz + view[1] / z, side="right")
    noise = float(np.percentile(v, 10)) if len(v) else 0.0
    return {"k": ks.tolist(), "inten": inten.tolist(), "rel": [None if not np.isfinite(x) else float(x) for x in rel],
            "mz": [None if not np.isfinite(x) else float(x) for x in mzk],
            "shift": [None if not np.isfinite(x) else float(x) for x in shift],
            "r": [None if not np.isfinite(x) else float(x) for x in r], "usable": usable.tolist(),
            "apex_scan": sa, "top_scans": [s0 + lo, s0 + hi], "n_scans": s1 - s0, "noise": noise,
            "apex_mz": m[a:b].astype(float).tolist(), "apex_inten": v[a:b].astype(float).tolist()}


def test_feature(cloud: Cloud, mz: float, z: int, s0: int, s1: int, ppm: float) -> dict | None:
    """extract_pattern followed by halogen_test on the M .. M+5 part of it"""
    pat = extract_pattern(cloud, mz, z, s0, s1, ppm)
    if pat is None:
        return None
    j0 = pat["k"].index(0)
    rel = np.array([np.nan if v is None else v for v in pat["rel"][j0:]])
    shift = np.array([np.nan if v is None else v for v in pat["shift"][j0:]])
    noise_rel = pat["noise"] / max(pat["inten"][j0] / max(pat["top_scans"][1] - pat["top_scans"][0] + 1, 1), 1e-12)
    res = halogen_test(rel, shift, mz, z, ppm, noise_rel=min(noise_rel, 1.0), usable=pat["usable"][j0:],
                       polarity=cloud.polarity)
    below = [(k, v) for k, v in zip(pat["k"], pat["rel"]) if k < 0 and v is not None and v >= 0.5]
    for k, v in below:
        res["flags"].append(f"a co-eluting peak {-k} mass unit{'s' if k < -1 else ''} below at {v:.2f}x M: this may "
                            f"not be the monoisotopic peak" + (" (a 37Cl / 81Br isotopologue?)" if k == -2 else ""))
    pat["test"] = res
    return pat


def test_groups(cloud: Cloud, corr, mz, rt, rt_lo, rt_hi, z, ppm: float, noise=0.0):
    """the halogen verdict of many features in one file: for each, the scans between its aligned bounds (corr: the
    file's corrected scan times, same length as cloud.rt; at least 3 scans around rt), the pattern of M..M+5 and the
    test. Returns (verdict codes into VERDICTS, best hypothesis names) aligned with the inputs."""
    n = len(mz)
    code = np.zeros(n, np.uint8)
    best = np.array([""] * n, dtype=object)
    if n == 0:
        return code, best
    corr = np.asarray(corr, np.float64)
    noise = np.broadcast_to(np.asarray(noise, np.float64), (n,))      # the noise floor per feature, or one for all
    S = cloud.n_scans
    s0 = np.searchsorted(corr, rt_lo)
    s1 = np.searchsorted(corr, rt_hi, side="right")
    c = np.clip(np.searchsorted(corr, rt), 1, max(S - 2, 1))
    s0 = np.clip(np.minimum(s0, c - 1), 0, S)
    s1 = np.clip(np.maximum(s1, c + 2), 0, S)
    levels = {"none": 1, "possible": 2, "halogen": 3}
    for i in range(n):
        if s1[i] <= s0[i]:
            continue
        zi = max(int(z[i]), 1)
        ks, lo_e, hi_e = _windows(float(mz[i]), zi, ppm, 0, 5)
        I, M = _window_sums(cloud.off, cloud.mz, cloud.inten, int(s0[i]), int(s1[i]), lo_e, hi_e)
        got = _summarise(I, M, 0)
        if got is None:
            continue
        rel, shift, r, usable, apex, lo, hi, inten, mzk = got
        noise_rel = noise[i] / max(inten[0] / (hi - lo + 1), 1e-12) if noise[i] > 0 else 0.0
        res = halogen_test(rel, shift * zi, float(mz[i]), zi, ppm, noise_rel=min(noise_rel, 1.0), usable=usable,
                           polarity=cloud.polarity, quick=True)
        code[i] = levels[res["verdict"]]
        best[i] = res["best"] or ""
    return code, best
