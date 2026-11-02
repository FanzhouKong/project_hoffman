"""The MS1 isotope pattern of one feature, and a test for halogens (Cl, Br) in it.

A chlorine atom adds a peak two mass units above the monoisotopic one at 32 % of its height (37Cl), a bromine at 97 %
(81Br). Carbon, nitrogen, oxygen and hydrogen leave M+2 at a few percent for a small molecule, sulfur adds 4.4 % per
atom. ``halogen_test`` fits the observed pattern (M+1 .. M+4 relative to M) under elemental hypotheses - no halogen,
S1, S2, Cl1-4, Br1-3, BrCl, Br2Cl, BrCl2 - each with the carbon count chosen to reproduce M+1 and the rest of the
skeleton scaled like an average organic molecule, and compares the fits (a chi-square with a 15 % relative tolerance).
Where the mass accuracy allows, the spacing of M+2 above M adds evidence: 37Cl and 81Br sit 0.0097 and 0.0088 Da
below 13C2, and 34S 0.0109 Da below. The call is "halogen" when the best halogen hypothesis beats the best
halogen-free one by 9 chi-square units (3 sigma), "possible" above 4, otherwise "none".

``extract_pattern`` reads the pattern of a feature out of a point cloud: the apex scan on the feature's own trace,
the intensity in a window at each M+k over the top half of the peak, and whether each window co-elutes with M.
"""
from __future__ import annotations

import math

import numpy as np

from .io import C13, Cloud

# (mass shift above the lightest isotope, abundance) per nominal offset; IUPAC representative values
_ISOTOPES = {
    "C": [(0.0, 0.9893), (1.0033548, 0.0107)],
    "H": [(0.0, 0.999885), (1.0062767, 0.000115)],
    "N": [(0.0, 0.99636), (0.9970349, 0.00364)],
    "O": [(0.0, 0.99757), (1.0042169, 0.00038), (2.0042460, 0.00205)],
    "S": [(0.0, 0.9499), (0.9993878, 0.0075), (1.9957959, 0.0425), (0.0, 0.0), (3.9950096, 0.0001)],
    "Cl": [(0.0, 0.7576), (0.0, 0.0), (1.9970499, 0.2424)],
    "Br": [(0.0, 0.5069), (0.0, 0.0), (1.9979535, 0.4931)],
}
MONO_MASS = {"C": 12.0, "H": 1.0078250, "N": 14.0030740, "O": 15.9949146, "S": 31.9720711, "Cl": 34.9688527,
             "Br": 78.9183376}
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
MIN_M2 = 0.12         # a halogen call needs at least this much M+2 (one Cl gives 0.32)
R_MIN = 0.6           # an isotope window must co-elute with M this well to enter the fit
STRONG, POSSIBLE = 9.0, 4.0


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
    nz = p > 0
    m[nz] /= p[nz]
    return p, m


def pattern(composition: dict, K: int = 6):
    """the isotope pattern of a composition {element: count}: (probability, mean mass shift) per offset 0..K-1"""
    p = np.zeros(K)
    p[0] = 1.0
    m = np.zeros(K)
    for el, n in composition.items():
        shifts, probs = np.zeros(K), np.zeros(K)
        for k, (s, a) in enumerate(_ISOTOPES[el][:K]):
            shifts[k], probs[k] = s, a
        q, mq = np.zeros(K), np.zeros(K)        # the element's n atoms, by repeated squaring
        q[0] = 1.0
        n = int(n)
        while n:
            if n & 1:
                q, mq = _conv(q, mq, probs, shifts)
            probs, shifts = _conv(probs, shifts, probs, shifts)
            n >>= 1
        p, m = _conv(p, m, q, mq)
    return p, m


def _skeleton(mass: float, extra: dict, rel1: float):
    """the organic part of a molecule of this neutral mass carrying the extra atoms: averagine proportions, with the
    carbon count chosen so the predicted M+1 matches the observed one. Returns (composition, n_c_capped)."""
    rest = mass - sum(MONO_MASS[e] * n for e, n in extra.items())
    if rest < 12.0:
        return None, False
    comp = {e: int(round(rest * f)) for e, f in AVERAGINE.items() if e != "C"}
    comp.update(extra)
    r1 = {e: _ISOTOPES[e][1][1] / _ISOTOPES[e][0][1] if len(_ISOTOPES[e]) > 1 else 0.0 for e in _ISOTOPES}
    other = sum(r1[e] * n for e, n in comp.items())
    cap = int(rest // 12.0)
    n_c = int(round((rel1 - other) / r1["C"])) if np.isfinite(rel1) else int(round(rest * AVERAGINE["C"]))
    capped = n_c > cap
    comp["C"] = min(max(n_c, 0), cap)
    # hydrogens cannot outnumber what the carbons can carry; keep the count plausible for the mass
    comp["H"] = min(comp["H"], 2 * comp["C"] + comp.get("N", 0) + 3)
    return comp, capped


def formula(comp: dict) -> str:
    order = ["C", "H", "N", "O", "S", "Cl", "Br"]
    return "".join(f"{e}{comp[e] if comp[e] != 1 else ''}" for e in order if comp.get(e, 0) > 0)


def halogen_test(rel, shift, mz: float, z: int = 1, ppm: float = 5.0, noise_rel: float = 0.0, usable=None) -> dict:
    """Test one isotope pattern for halogens.

    rel:   intensity of M+k relative to M for k = 0 .. K-1 (rel[0] = 1; nan = not measured)
    shift: (m/z of M+k - m/z of M) * z in Da, nan when unknown; from the same scans as rel
    mz:    m/z of M; z its charge; ppm the run's mass tolerance; noise_rel the noise floor relative to M
    usable: per k, whether the window is clean (co-elutes with M); unusable ones are left out of the fit
    """
    rel = np.asarray(rel, float)
    shift = np.asarray(shift, float)
    K = len(rel)
    z = max(int(z), 1)
    use = np.ones(K, bool) if usable is None else np.asarray(usable, bool)
    use &= np.isfinite(rel)
    mass = mz * z            # the neutral mass to within an adduct: enough for the element counts
    sig_da = max(ppm * mz * 1e-6 * z * 0.71, 0.0005)   # one centroid difference, ppm read as ~2 sigma per centroid
    rel1 = rel[1] if K > 1 and use[1] else np.nan
    flags = []
    out = []
    for name, extra in HYPOTHESES:
        comp, capped = _skeleton(mass, extra, rel1)
        if comp is None:
            continue
        p, m = pattern(comp, K)
        pred = p / p[0]
        chi2 = 0.0
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
                    "rel": [float(v) for v in pred], "shift": [float(v) for v in m]})
    out.sort(key=lambda d: d["chi2"])
    best = out[0] if out else None
    hal = [d for d in out if d["halogen"]]
    free = [d for d in out if not d["halogen"]]
    delta = (free[0]["chi2"] - hal[0]["chi2"]) if hal and free else -math.inf
    none = next((d for d in out if d["name"] == "none"), None)
    m2_excess = float(rel[2] - none["rel"][2]) if none is not None and K > 2 and use[2] else float("nan")
    verdict = "none"
    if hal and best is not None and best["halogen"] and use[2] and rel[2] >= MIN_M2:
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
            "n_c": best["n_c"] if best else None, "delta_chi2": float(delta) if np.isfinite(delta) else None,
            "m2_excess": None if not np.isfinite(m2_excess) else m2_excess,
            "observed": {"rel": [None if not np.isfinite(v) else float(v) for v in rel],
                         "shift": [None if not np.isfinite(v) else float(v) for v in shift],
                         "usable": [bool(u) for u in use]},
            "hypotheses": out, "flags": flags, "sigma_da": float(sig_da)}


def _corr(a, b):
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


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
    ks = np.arange(k_lo, k_hi + 1)
    tol = mz * ppm * 1e-6
    half = (tol + 0.008 * np.abs(ks)) / z       # 15N at +0.997, 2H at +1.006, 34S at +1.996, 13C2 at +2.007 ...
    centre = mz + ks * C13 / z
    n = s1 - s0
    I = np.zeros((n, len(ks)))
    M = np.zeros((n, len(ks)))
    for s in range(s0, s1):
        m = cloud.mz[cloud.off[s]:cloud.off[s + 1]]
        v = cloud.inten[cloud.off[s]:cloud.off[s + 1]].astype(np.float64)
        a = np.searchsorted(m, centre - half)
        b = np.searchsorted(m, centre + half, side="right")
        for j in range(len(ks)):
            if b[j] > a[j]:
                w = v[a[j]:b[j]]
                I[s - s0, j] = w.sum()
                M[s - s0, j] = (m[a[j]:b[j]] * w).sum() / w.sum()
    j0 = int(np.where(ks == 0)[0][0])
    trace = I[:, j0]
    if trace.max() <= 0:
        return None
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
        mzk = (I[top] * M[top]).sum(axis=0) / inten
    r = np.array([_corr(I[:, j], trace) for j in range(len(ks))])
    rel = inten / inten[j0]
    usable = np.isfinite(rel) & (np.isnan(r) | (r >= R_MIN) | (rel < 0.02))
    sa = s0 + apex
    m = cloud.mz[cloud.off[sa]:cloud.off[sa + 1]]
    v = cloud.inten[cloud.off[sa]:cloud.off[sa + 1]]
    a, b = np.searchsorted(m, mz + view[0] / z), np.searchsorted(m, mz + view[1] / z, side="right")
    noise = float(np.percentile(v, 10)) if len(v) else 0.0
    return {"k": ks.tolist(), "inten": inten.tolist(), "rel": [None if not np.isfinite(x) else float(x) for x in rel],
            "mz": [None if not np.isfinite(x) else float(x) for x in mzk],
            "shift": [None if not (np.isfinite(mzk[j]) and np.isfinite(mzk[j0])) else float((mzk[j] - mzk[j0]) * z)
                      for j in range(len(ks))],
            "r": [None if not np.isfinite(x) else float(x) for x in r], "usable": usable.tolist(),
            "apex_scan": sa, "top_scans": [s0 + lo, s0 + hi], "n_scans": n, "noise": noise,
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
    res = halogen_test(rel, shift, mz, z, ppm, noise_rel=min(noise_rel, 1.0), usable=pat["usable"][j0:])
    below = [(k, v) for k, v in zip(pat["k"], pat["rel"]) if k < 0 and v is not None and v >= 0.5]
    for k, v in below:
        res["flags"].append(f"a co-eluting peak {-k} mass unit{'s' if k < -1 else ''} below at {v:.2f}x M: this may "
                            f"not be the monoisotopic peak" + (" (a 37Cl / 81Br isotopologue?)" if k == -2 else ""))
    pat["test"] = res
    return pat
