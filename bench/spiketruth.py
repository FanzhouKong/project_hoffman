#!/usr/bin/env python3
"""Tool-independent truth for the PR002464 spiked standards: every chromatographic peak the 96
standards produce (m/z, RT, bounds), built from the raw centroided mzML of one study.

Design (bench/pr2464_design.py -> data/raw/PR002464/design): the 96 standards come in 24 mixtures of
four. Mixture k is in spike solution A at amount a and in B at amount b, log2(a/b) = +-1..4 (eight
signatures, 12 standards each), so a vial with A uL of spike A and B uL of spike B holds a*A + b*B.
Matrix-free spike vials: C (A+B 1:1 at 1000/500/200/100 uL; vials 9-12) and E (A:B 1:0, 0:1, 1:3,
3:1, 1:1 at 1000 uL; vials 18-22). Every vial is injected once per batch (4 injections).

Steps
 1. discovery on the strong vials (9, 18, 19): EIC of every adduct of every standard, max over the
    per-vial median traces -> local maxima above MIN_HEIGHT_X x floor and SNR_MIN x the local
    baseline (running 20th percentile: a median sits on the tail of strongly tailing peaks)
    = candidate (ion, RT)
 2. per-file RT offsets from the strongest candidates (spike-free files: interpolated in
    injection order within the batch)
 3. tests on the matrix-free vials only. "noise" = median read in the blanks (>= the floor);
    reads in the E and C series are noise-subtracted
    present   apex with >= MIN_SCANS contiguous scans in >= MIN_REPS injections of the vial where
              the ion is stronger (18 = A only, 19 = B only)
    specific  median read in process + solvent blanks <= MAX_BLANK x that vial's height
    design    E series (A:B 1:0, 0:1, 1:3, 3:1, 1:1) fits alpha A + beta B with free alpha, beta >= 0
              (mean relative residual <= MAX_RESID), A and B differ (|log2(h18/h19)| >= R_AB_MIN)
              and the C series falls with dilution (log-log slope >= MIN_DIL_SLOPE, when >= 3
              levels are above noise). Together: spike-derived. Carry-over, background and
              internal standards (present in process blanks) fail
    signature log2(h18/h19) within SIG_TOL of the standard's log2FC (a one-sided bound when the
              weak vial is at noise), and the fixed-ratio fit a A + b B within MAX_RESID
    tier A = spike-derived and signature: the peak belongs to this standard
    tier B = spike-derived only: a real spike peak at this standard's ion mass whose ratio says it
             comes from a mixture with another signature (label uncertain)
 4. each standard's tier-A peak groups: ions within RT_GROUP of each other. A group needs one of
    the standard's PRIMARY ions; groups of dimers / rare adducts alone drop to tier B (a one-sided
    signature at a rare adduct mass is satisfied by any strongly A- or B-rich ion). The group with
    the largest summed height is the standard's main peak (rt_rank 1), others rt_rank 2, 3 ...
    (tier B: rt_rank 0)
Known gap: peaks whose RT moves with load (acesulfame in RP: 2.70 min at 1/8 load, 3.04 min at
full load) fail the fixed-RT design test and are left out.
 5. co-eluting ions: every other m/z cluster at a main peak's apex in the dominant vial whose
    profile matches the standard's strongest ion (r >= R_MIN in >= MIN_REPS injections) and that
    passes step 3 -> isotopes, other adducts, multimers, in-source fragments, impurities.
    Annotated by mass difference. Flags (kept in the truth, for the scorer to decide):
    "ft_artifact?" within ARTIFACT_DA of a co-eluting truth ion >= 1/ARTIFACT_RATIO times stronger
    (Orbitrap sidebands / ringing); "broad" FWHM > BROAD_X x the median (overload smears, bands)
 6. the same discovery and tests on formula masses shifted by MASS_DECOYS estimate how many
    unrelated ions pass by chance

Matrix vials (plasma G 26-30, plasma + yeast I 34-38, QC) are measured and reported (presence,
fit) but never tested: matrix effects and endogenous compounds would confound the signature.

All thresholds are fixed here and written to params.json.
usage: spiketruth.py STUDY [--cpus N]          STUDY = ST004500 | ST004501 | ST004502 | ST004504
"""
import argparse
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import percentile_filter
from scipy.optimize import nnls
from scipy.signal import find_peaks

_trapezoid = getattr(np, "trapezoid", None) or np.trapz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.msdata import C13, load_run  # noqa: E402
from bench.mz_registry import mz_test_files  # noqa: E402

BASE = ROOT / "data/raw/PR002464"
STUDIES = {  # study -> polarity, instrument class, chromatography
    "ST004500": ("neg", "orbitrap", "RP"),
    "ST004504": ("pos", "orbitrap", "RP"),
    "ST004501": ("pos", "orbitrap", "HILIC"),
    "ST004502": ("neg", "orbitrap", "HILIC"),
}

# ------------------------------------------------------------- parameters ----
PPM = {"orbitrap": 5.0, "qtof": 20.0}   # EIC half-width
CLUSTER_PPM = {"orbitrap": 3.0, "qtof": 10.0}   # gap that splits centroid m/z clusters (sweep)
MIN_SCANS = 4        # contiguous nonzero scans around an apex
LOCAL_MAX_SCANS = 3  # an apex must be the maximum of +-3 scans
MIN_REPS = 3         # of the 4 injections of the dominant vial
PROM_FRAC = 0.3      # discovery: local maximum prominence >= 0.3 x its height
MIN_HEIGHT_X = 10.0  # discovery and sweep: height >= 10 x intensity floor (1st pct of centroids)
SNR_MIN = 3.0        # discovery: height >= 3 x the trace's running 20th percentile over +-BASE_WIN_MIN
BASE_WIN_MIN = 1.0    # wide enough for broad bands (hispolon: FWHM ~40 s)
MAX_BLANK = 0.1      # blank median read / dominant vial median height
R_AB_MIN = 0.4       # spike-derived: |log2(h18/h19)| >= 0.4 (every signature is >= 1)
MIN_DIL_SLOPE = 0.5  # spike-derived: C-series log-log slope (1 = proportional), when measurable
SIG_TOL = 0.75       # |log2(h18/h19) - log2FC|; signatures are >= 1 apart
MAX_RESID = 0.3      # E-series fits: sum|h - fit| / sum h
R_MIN = 0.8          # co-eluting ion vs the standard's strongest ion, profile correlation
ARTIFACT_DA, ARTIFACT_RATIO = 0.05, 0.05
BROAD_X = 3.0        # flag "broad": FWHM > 3 x the study's median tier-A FWHM (overload smears, bands)
APEX_TOL_MIN = 0.05  # apex search half-window around the expected RT (after offsets)
OFFSET_WIN_MIN = 0.3 # apex search half-window while offsets are not yet known
RT_GROUP_MIN = 0.1   # passing ions of one standard closer than this are one peak group
SNIP_MIN = 0.5       # EIC half-window kept per candidate
MZ_LO, MZ_HI = 50.0, 1200.0   # acquisition range (Methods_20230824.docx)
MASS_DECOYS = (-0.02, 0.02)
PRIMARY = {"pos": {"[M+H]+", "[M+NH4]+", "[M+Na]+", "[M+H-H2O]+"},
           "neg": {"[M-H]-", "[M+HCOO]-", "[M+Cl]-", "[M-H-H2O]-"}}
N_ANCHORS = 150      # strongest candidates used for RT offsets

PROTON = 1.00727646688
ELEMENTS = {"C": 12.0, "H": 1.00782503207, "N": 14.0030740048, "O": 15.99491461956,
            "S": 31.97207100, "P": 30.97376163, "F": 18.99840322, "Cl": 34.96885268,
            "Br": 78.9183371, "I": 126.904473, "Na": 22.9897692809, "K": 38.96370668}
# (name, n molecules, mass added, charge): m/z = (n M + delta) / z
ADDUCTS = {
    "pos": [("[M+H]+", 1, PROTON, 1), ("[M+NH4]+", 1, 18.0338257, 1), ("[M+Na]+", 1, 22.9892207, 1),
            ("[M+K]+", 1, 38.9631579, 1), ("[M+H-H2O]+", 1, PROTON - 18.0105647, 1),
            ("[M+H-2H2O]+", 1, PROTON - 36.0211294, 1), ("[M+2H]2+", 1, 2 * PROTON, 2),
            ("[M+CH3OH+H]+", 1, 33.0334912, 1), ("[M+ACN+H]+", 1, 42.0338257, 1),
            ("[2M+H]+", 2, PROTON, 1), ("[2M+NH4]+", 2, 18.0338257, 1), ("[2M+Na]+", 2, 22.9892207, 1)],
    "neg": [("[M-H]-", 1, -PROTON, 1), ("[M+Cl]-", 1, 34.9694021, 1), ("[M+HCOO]-", 1, 44.9982030, 1),
            ("[M+CH3COO]-", 1, 59.0138531, 1), ("[M-H-H2O]-", 1, -PROTON - 18.0105647, 1),
            ("[M+Na-2H]-", 1, 20.9746662, 1), ("[M-2H]2-", 1, -2 * PROTON, 2),
            ("[2M-H]-", 2, -PROTON, 1), ("[2M+HCOO]-", 2, 44.9982030, 1)],
}
# mass differences used to annotate co-eluting ions relative to a truth ion of the same standard
ISOTOPE_STEPS = [("M+1", C13), ("M+2", 2 * C13), ("M+3", 3 * C13), ("34S", 1.9957959),
                 ("37Cl", 1.9970499), ("41K", 1.9981186), ("81Br", 1.9979534)]


def formula_mass(f):
    m = 0.0
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", f):
        m += ELEMENTS[el] * (int(n) if n else 1)
    return m


# ------------------------------------------------------------- design ----
def study_files(st, mz_test=False):
    """MS1 injections of a study with their composition (uL) and class. The injections whose m/z calibration is
    off (bench/mz_registry.py) are left out unless mz_test: the truth is built without them."""
    meta = pd.read_csv(BASE / st / "2-sample_metadata/sample_metadata.csv")
    meta = meta[(meta["ms_level"] == 1) & (meta["sample_type"] != "conditioning.meoh")].copy()
    meta["vial"] = meta["vial"].astype(int)   # conditioning injections use letter vials ("aa")
    # the centroid folder is 3-data_mzML_centroid in most packages, 3-mzML_centroid in ST004502
    d = BASE / st
    paths = {p.stem: p for sub in d.glob("3-*centroid") for p in sub.rglob("*.mzML")}
    meta["path"] = [str(paths.get(Path(s).stem, "")) for s in meta["sample_name"]]
    missing = meta[meta["path"] == ""]
    if len(missing):
        sys.exit(f"{len(missing)} mzML files missing under {d}, e.g. {missing['sample_name'].iloc[0]}")
    meta["file"] = [Path(p).stem for p in meta["path"]]
    if not mz_test:
        meta = meta[~meta["file"].isin(mz_test_files(st))]
    return classify_files(meta)


def classify_files(meta):
    m = meta.rename(columns={"[spikeA]": "A", "[spikeB]": "B"}).copy()
    matrix = m[["SRM 1950", "SRM 8231", "Yeast 12C", "Yeast 13C"]].sum(axis=1) > 0
    spike = (m["A"] + m["B"]) > 0
    m["cls"] = np.select(
        [m["sample_type"].str.endswith("blank"), spike & ~matrix, spike & matrix, matrix],
        ["blank", "spike", "spike_matrix", "matrix"], "other")
    return m.reset_index(drop=True)


def standards_table():
    s = pd.read_csv(BASE / "design/standards.tsv", sep="\t")
    s["mass"] = [formula_mass(f) for f in s["formula"]]
    bad = (s["mass"] - s["monoisotopic_mass"]).abs() > 1e-3
    if bad.any():
        sys.exit(f"formula mass disagrees with PubChem for standards {s.loc[bad, 'std'].tolist()}")
    return s


def ion_table(stds, mode, shift=0.0):
    """every adduct of every standard inside the scan range; shift > 0 gives mass decoys"""
    rows = []
    for _, r in stds.iterrows():
        for name, n, delta, z in ADDUCTS[mode]:
            mz = (n * r["mass"] + delta) / z + shift
            if MZ_LO <= mz <= MZ_HI:
                rows.append(dict(std=int(r["std"]), ion=name, mz=mz, log2FC=float(r["log2FC"]),
                                 a=float(r["spikeA_vol"]), b=float(r["spikeB_vol"])))
    return pd.DataFrame(rows)


# --------------------------------------------------------- EIC primitives ----
def eic_win(run, mz, ppm, s0, s1):
    """max intensity per scan in [s0, s1) within ppm of mz, and the m/z of that centroid"""
    tol = mz * ppm * 1e-6
    a, b = np.searchsorted(run.mz, [mz - tol, mz + tol])
    e = np.zeros(s1 - s0)
    mzs = np.zeros(s1 - s0)
    if b > a:
        sc = run.scan[a:b]
        k = (sc >= s0) & (sc < s1)
        sc, it, mm = sc[k] - s0, run.inten[a:b][k], run.mz[a:b][k]
        o = np.argsort(it, kind="stable")     # ascending: the strongest centroid of a scan wins
        e[sc[o]] = it[o]
        mzs[sc[o]] = mm[o]
    return e, mzs


def measure(e, rts, t, tol):
    """apex of the EIC e (times rts) within tol of t. Returns dict or None when absent: no signal
    there or fewer than MIN_SCANS contiguous nonzero scans around the apex. Bounds: walk out on the
    3-point smoothed profile to 2 % of the apex or the first valley."""
    near = np.abs(rts - t) <= tol
    if not near.any() or e[near].max() <= 0:
        return None
    k = np.flatnonzero(near)[np.argmax(e[near])]
    # a real apex is the highest point within +-LOCAL_MAX_SCANS, also past the window edge;
    # otherwise the window only cuts into the flank of a larger peak nearby
    if e[k] < e[max(0, k - LOCAL_MAX_SCANS):k + LOCAL_MAX_SCANS + 1].max():
        return None
    lo = k
    while lo > 0 and e[lo - 1] > 0:
        lo -= 1
    hi = k
    while hi < len(e) - 1 and e[hi + 1] > 0:
        hi += 1
    if hi - lo + 1 < MIN_SCANS:
        return None
    h = e[k]
    sm = np.convolve(e, np.ones(3) / 3, mode="same")
    l = k
    while l > lo and sm[l - 1] > 0.02 * h and sm[l - 1] <= sm[l] * 1.02:
        l -= 1
    r = k
    while r < hi and sm[r + 1] > 0.02 * h and sm[r + 1] <= sm[r] * 1.02:
        r += 1
    seg = e[l:r + 1]
    area = float(_trapezoid(seg, rts[l:r + 1])) if r > l else 0.0
    return dict(height=float(h), apex_rt=float(rts[k]), left=float(rts[l]), right=float(rts[r]),
                area=area, fwhm=fwhm(e, rts, k), n_scans=int(hi - lo + 1))


def fwhm(e, rts, k):
    h = e[k] / 2
    l = k
    while l > 0 and e[l - 1] > h:
        l -= 1
    r = k
    while r < len(e) - 1 and e[r + 1] > h:
        r += 1
    tl = rts[l] if l == 0 else np.interp(h, [e[l - 1], e[l]], [rts[l - 1], rts[l]])
    tr = rts[r] if r == len(e) - 1 else np.interp(h, [e[r + 1], e[r]], [rts[r + 1], rts[r]])
    return float(tr - tl)


def corr_on_grid(p1, t1, p2, t2, t, half):
    g = np.linspace(t - half, t + half, 25)
    a = np.interp(g, t1, p1, left=0, right=0)
    b = np.interp(g, t2, p2, left=0, right=0)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------- workers ----
def w_discover(args):
    """dense EICs of all ions, resampled on the reference time grid"""
    path, mzs, ppm, grid = args
    run = load_run(path)
    out = np.zeros((len(mzs), len(grid)), np.float32)
    for i, m in enumerate(mzs):
        e, _ = eic_win(run, m, ppm, 0, run.n_scans)
        out[i] = np.interp(grid, run.rt, e, left=0, right=0)
    floor = float(np.percentile(run.inten, 1))
    return out, floor


def w_measure(args):
    """apex statistics of every candidate (mz, expected RT) in one file"""
    path, mzs, ts, ppm, tol = args
    run = load_run(path)
    n = len(mzs)
    res = {k: np.full(n, np.nan) for k in ("height", "apex_rt", "left", "right", "area", "fwhm", "mz_obs", "n_scans")}
    near_max = np.zeros(n)   # plain max within tol (no scan-count rule): the blank / matrix read-out
    for i, (m, t) in enumerate(zip(mzs, ts)):
        s0, s1 = run.scan_window(t, SNIP_MIN)
        if s1 <= s0:
            continue
        e, mm = eic_win(run, m, ppm, s0, s1)
        rts = run.rt[s0:s1]
        near = np.abs(rts - t) <= tol
        near_max[i] = e[near].max() if near.any() else 0.0
        p = measure(e, rts, t, tol)
        if p is None:
            continue
        for k in ("height", "apex_rt", "left", "right", "area", "fwhm", "n_scans"):
            res[k][i] = p[k]
        k = np.flatnonzero(rts == p["apex_rt"])[0]
        lo, hi = max(0, k - 1), min(len(e), k + 2)
        w = e[lo:hi]
        res["mz_obs"][i] = float(np.average(mm[lo:hi][w > 0], weights=w[w > 0]))
    res["near_max"] = near_max
    return res


def w_sweep(args):
    """m/z clusters at each target apex in one dominant-vial file, with their profile correlation
    to the target's reference ion. targets: (key, expected apex t, reference ion m/z, fwhm)"""
    path, targets, ppm, cluster_ppm, floor = args
    run = load_run(path)
    out = []
    for key, t, ref_mz, fw in targets:
        half = max(2 * fw, 0.1)
        s0, s1 = run.scan_window(t, half)
        if s1 - s0 < MIN_SCANS:
            continue
        rts = run.rt[s0:s1]
        eref, _ = eic_win(run, ref_mz, ppm, s0, s1)
        pref = measure(eref, rts, t, APEX_TOL_MIN)
        if pref is None:
            continue
        ta = pref["apex_rt"]
        # centroids within +-1 scan of the reference apex, clustered by m/z
        ka = int(np.flatnonzero(rts == ta)[0]) + s0
        sel = (run.scan >= ka - 1) & (run.scan <= ka + 1)
        mz, it = run.mz[sel], run.inten[sel]   # run.mz is sorted, so mz is too
        if len(mz) == 0:
            continue
        cuts = np.flatnonzero(np.diff(mz) / mz[1:] * 1e6 > cluster_ppm) + 1
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(mz)]):
            if it[a:b].max() < MIN_HEIGHT_X * floor:
                continue
            cm = float(np.average(mz[a:b], weights=it[a:b]))
            e, _ = eic_win(run, cm, ppm, s0, s1)
            p = measure(e, rts, ta, APEX_TOL_MIN)
            if p is None:
                continue
            r = corr_on_grid(eref, rts, e, rts, ta, max(fw, 3 * float(np.median(np.diff(rts)))))
            out.append((key, cm, p["height"], p["apex_rt"] - ta, r, path))
    return out


# ------------------------------------------------------------- discovery ----
def discover(pool, files, ions, ppm, grid):
    """candidates (ion row, RT) from the per-vial median traces of the strong vials"""
    strong = files[files["vial"].isin([9, 18, 19])]
    res = pool.map(w_discover, [(p, ions["mz"].values, ppm, grid) for p in strong["path"]])
    floor = float(np.median([r[1] for r in res]))
    stack = np.stack([r[0] for r in res])                     # file x ion x grid
    traces = np.stack([np.median(stack[(strong["vial"] == v).values], axis=0) for v in (9, 18, 19)])
    trace = traces.max(axis=0)                                 # ion x grid
    w = 2 * int(BASE_WIN_MIN / float(np.median(np.diff(grid)))) + 1
    base = percentile_filter(trace, 20, size=(1, w), mode="nearest")
    rows = []
    for i in range(len(ions)):
        sm = np.convolve(trace[i], np.ones(3) / 3, mode="same")
        pk, pr = find_peaks(sm, prominence=0, height=MIN_HEIGHT_X * floor)
        for k, prom in zip(pk, pr["prominences"]):
            if prom < PROM_FRAC * sm[k] or sm[k] < SNR_MIN * base[i, k]:
                continue
            lo, hi = max(0, k - 3), min(len(grid), k + 4)
            if np.count_nonzero(trace[i, lo:hi]) < MIN_SCANS:
                continue
            rows.append(dict(ion_row=i, rt=float(grid[k]), trace_h=float(trace[i, k])))
    cand = pd.DataFrame(rows, columns=["ion_row", "rt", "trace_h"])
    cand = cand.join(ions.reset_index(drop=True), on="ion_row")
    return cand, floor


def rt_offsets(pool, files, cand, ppm):
    """per-file RT offset (min): median apex shift of the strongest candidates in spike files;
    spike-free files take the offset interpolated in injection order within their batch"""
    top = cand.sort_values("trace_h", ascending=False).drop_duplicates("mz").head(N_ANCHORS)
    has = files["cls"].isin(["spike", "spike_matrix"]).values
    res = pool.map(w_measure, [(p, top["mz"].values, top["rt"].values, ppm, OFFSET_WIN_MIN)
                               for p in files.loc[has, "path"]])
    shifts = np.stack([r["apex_rt"] for r in res]) - top["rt"].values      # file x anchor
    # anchors as a whole move together; per anchor, centre on its median so a mis-placed
    # discovery RT does not bias every file
    shifts -= np.nanmedian(shifts, axis=0)
    off = np.full(len(files), np.nan)
    off[has] = np.nanmedian(shifts, axis=1)
    off = pd.Series(off, index=files.index)
    for _, g in files.groupby("batch"):
        g = g.sort_values("injection_order")
        known = off[g.index].notna().values
        if known.any():
            off[g.index] = np.interp(g["injection_order"], g["injection_order"][known], off[g.index][known])
    return off.fillna(0.0)


def measure_all(pool, files, cand, off, ppm, tol):
    """matrices (candidate x file) of apex statistics, expected RT = candidate RT + file offset"""
    args = [(p, cand["mz"].values, cand["rt"].values + o, ppm, tol) for p, o in zip(files["path"], off)]
    res = pool.map(w_measure, args)
    return {k: np.stack([r[k] for r in res], axis=1) for k in res[0]}


# ------------------------------------------------------------------ tests ----
def vial_median(M, files, vial, present_only=False):
    cols = (files["vial"] == vial).values
    x = M[:, cols]
    if present_only:
        return np.nanmedian(x, axis=1)
    return np.median(np.nan_to_num(x), axis=1)


def evaluate(cand, meas, files, floor):
    """step 3 on every candidate; returns cand with test columns, tier A / B / "" (rejected)"""
    H, near = meas["height"], meas["near_max"]
    c = cand.copy()
    fc = c["log2FC"].values
    blank = (files["cls"] == "blank").values
    noise = np.maximum(np.median(near[:, blank], axis=1), floor)
    w18, w19 = vial_median(near, files, 18), vial_median(near, files, 19)
    r = np.log2(np.maximum(w18, noise) / np.maximum(w19, noise))
    # the vial where the ion is stronger carries presence and specificity
    dom = np.where(r >= 0, 18, 19)
    h18, h19 = vial_median(H, files, 18), vial_median(H, files, 19)
    hdom = np.where(dom == 18, h18, h19)
    n18 = np.sum(~np.isnan(H[:, (files["vial"] == 18).values]), axis=1)
    n19 = np.sum(~np.isnan(H[:, (files["vial"] == 19).values]), axis=1)
    c["dominant_vial"] = dom
    c["n_present_dom"] = np.where(dom == 18, n18, n19)
    c["height_dom"] = hdom
    c["noise"] = noise
    c["blank_frac"] = np.median(near[:, blank], axis=1) / np.maximum(hdom, 1)
    mat = (files["cls"] == "matrix").values
    c["matrix_frac"] = np.median(near[:, mat], axis=1) / np.maximum(hdom, 1) if mat.any() else np.nan
    weak_below = np.where(dom == 18, w19, w18) <= 2 * noise
    c["log2_18_19"] = r
    c["weak_at_noise"] = weak_below
    # E series on noise-subtracted vial medians: free fit alpha A + beta B, and the fixed a:b fit
    E = [18, 19, 20, 21, 22]
    vinfo = files.drop_duplicates("vial").set_index("vial")
    A = np.array([vinfo.loc[v, "A"] for v in E], float)
    B = np.array([vinfo.loc[v, "B"] for v in E], float)
    h = np.maximum(np.stack([vial_median(near, files, v) for v in E], axis=1) - noise[:, None], 0)
    M = np.stack([A, B], axis=1)
    free = np.full(len(c), np.nan)
    for i in range(len(c)):
        if h[i].sum() > 0:
            coef, _ = nnls(M, h[i])
            free[i] = np.abs(h[i] - M @ coef).sum() / h[i].sum()
    c["e_free_resid"] = free
    x = c["a"].values[:, None] * A[None] + c["b"].values[:, None] * B[None]
    k = (h * x).sum(1) / (x * x).sum(1)
    c["e_fit_resid"] = np.abs(h - k[:, None] * x).sum(1) / np.maximum(h.sum(1), 1)
    # C-series dilution (vials 9-12): log-log slope of height vs amount, on levels above noise
    Cv = [9, 10, 11, 12]
    hc = np.stack([vial_median(near, files, v) for v in Cv], axis=1) - noise[:, None]
    amt = np.log2(np.array([vinfo.loc[v, "A"] + vinfo.loc[v, "B"] for v in Cv]))
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log2(np.where(hc > noise[:, None], hc, np.nan))
    ok = ~np.isnan(y)
    slope = np.full(len(c), np.nan)
    for i in np.flatnonzero(ok.sum(1) >= 3):
        slope[i] = np.polyfit(amt[ok[i]], y[i, ok[i]], 1)[0]
    c["c_dilution_slope"] = slope
    c["pass_present"] = c["n_present_dom"] >= MIN_REPS
    c["pass_blank"] = c["blank_frac"] <= MAX_BLANK
    c["pass_design"] = (c["e_free_resid"] <= MAX_RESID) & (np.abs(r) >= R_AB_MIN) \
        & ~(c["c_dilution_slope"] < MIN_DIL_SLOPE)
    c["spike_derived"] = c["pass_present"] & c["pass_blank"] & c["pass_design"]
    sig = np.where(weak_below, (np.sign(r) == np.sign(fc)) & (np.abs(r) >= np.abs(fc) - SIG_TOL),
                   np.abs(r - fc) <= SIG_TOL)
    c["pass_signature"] = sig & (c["e_fit_resid"] <= MAX_RESID)
    c["tier"] = np.where(c["spike_derived"] & c["pass_signature"], "A",
                         np.where(c["spike_derived"], "B", ""))
    c["passed"] = c["tier"] != ""
    # consensus position over present matrix-free spike injections
    sp = (files["cls"] == "spike").values
    with np.errstate(all="ignore"):
        c["rt_apex"] = np.nanmedian(meas["apex_rt"][:, sp], axis=1)
        c["rt_left"] = np.nanmedian(meas["left"][:, sp], axis=1)
        c["rt_right"] = np.nanmedian(meas["right"][:, sp], axis=1)
        c["fwhm_s"] = np.nanmedian(meas["fwhm"][:, sp], axis=1) * 60
        c["mz_obs"] = np.nanmedian(meas["mz_obs"][:, sp], axis=1)
        c["n_present_spike"] = np.sum(~np.isnan(H[:, sp]), axis=1)
        spm = (files["cls"] == "spike_matrix").values
        c["n_present_spike_matrix"] = np.sum(~np.isnan(H[:, spm]), axis=1)
    c["mz_err_ppm"] = (c["mz_obs"] - c["mz"]) / c["mz"] * 1e6
    return c


def group_peaks(ok):
    """step 4: per standard, passing ions within RT_GROUP_MIN form one peak group; rank groups by
    summed dominant height"""
    ok = ok.sort_values(["std", "rt_apex"]).copy()
    gid = np.zeros(len(ok), int)
    g = -1
    prev_std, prev_t = None, -1e9
    for i, (s, t) in enumerate(zip(ok["std"], ok["rt_apex"])):
        if s != prev_std or t - prev_t > RT_GROUP_MIN:
            g += 1
        gid[i] = g
        prev_std, prev_t = s, t
    ok["group"] = gid
    tot = ok.groupby("group")["height_dom"].transform("sum")
    ok["group_height"] = tot
    rank = ok.drop_duplicates("group")[["std", "group", "group_height"]] \
        .sort_values(["std", "group_height"], ascending=[True, False])
    rank["rt_rank"] = rank.groupby("std").cumcount() + 1
    return ok.merge(rank[["group", "rt_rank"]], on="group")


def dedupe(cand, ppm, tol):
    """one row per (m/z, RT) peak; when several labels pass for the same peak keep the strongest
    and list the others"""
    # tier A first, then formula ions ("formula" > "coelution"), then main-RT peaks, then the strongest
    cand = cand.sort_values(["tier", "source", "rt_rank", "height_dom"],
                            ascending=[True, False, True, False]).reset_index(drop=True)
    keep, others = [], {}
    taken = np.zeros(len(cand), bool)
    mz, rt = cand["mz_obs"].fillna(cand["mz"]).values, cand["rt_apex"].values
    for i in range(len(cand)):
        if taken[i]:
            continue
        same = np.flatnonzero(~taken & (np.abs(mz - mz[i]) <= mz[i] * ppm * 1e-6) & (np.abs(rt - rt[i]) <= tol))
        taken[same] = True
        keep.append(i)
        others[i] = ";".join(f"{cand.loc[j, 'std']}:{cand.loc[j, 'ion']}" for j in same if j != i)
    out = cand.loc[keep].copy()
    out["also_matches"] = [others[i] for i in keep]
    return out


# ----------------------------------------------------------- annotation ----
def annotate(ion_mz, std_mass, mode, truth_ions):
    """name a co-eluting ion of a standard: isotope of a stronger truth ion of the same standard
    (truth_ions: (m/z, name) of those), an adduct of M, else in-source fragment (below M) /
    unannotated"""
    for t_mz, t_name in truth_ions:
        for nm, d in ISOTOPE_STEPS:
            for z in (1, 2):
                if abs(ion_mz - (t_mz + d / z)) <= 5e-6 * ion_mz + 0.002:
                    return f"{t_name} {nm}" + ("" if z == 1 else " (z=2)"), "isotope"
    for name, n, delta, z in ADDUCTS[mode]:
        if abs(ion_mz - (n * std_mass + delta) / z) <= 5e-6 * ion_mz:
            return name, "adduct"
    neutral = ion_mz - (PROTON if mode == "pos" else -PROTON)
    if neutral < std_mass - 1:
        return f"fragment (M-{std_mass - neutral:.4f})", "fragment"
    return f"unannotated (M+{neutral - std_mass:.4f})", "unannotated"


# ------------------------------------------------------------------- main ----
def build(st, files, stds, mode, instrument, out, cpus):
    ppm, cluster_ppm = PPM[instrument], CLUSTER_PPM[instrument]
    out.mkdir(parents=True, exist_ok=True)
    ref = load_run(files.loc[files["vial"] == 9, "path"].iloc[0])
    grid = ref.rt
    dt = float(np.median(np.diff(grid)))
    tol = max(APEX_TOL_MIN, 3 * dt)
    ions = ion_table(stds, mode)
    decoys = pd.concat([ion_table(stds, mode, d).assign(decoy=f"mass{d:+.2f}") for d in MASS_DECOYS])
    all_ions = pd.concat([ions.assign(decoy=""), decoys], ignore_index=True)
    print(f"[{st}] {len(files)} MS1 files, scan dt {dt * 60:.2f} s, {len(ions)} ions "
          f"(+{len(decoys)} decoys), apex tol {tol * 60:.1f} s", flush=True)

    with mp.Pool(cpus) as pool:
        cand, floor = discover(pool, files, all_ions, ppm, grid)
        print(f"[{st}] intensity floor {floor:.0f}; {len(cand)} candidates "
              f"({(cand['decoy'] == '').sum()} target)", flush=True)
        off = rt_offsets(pool, files, cand[cand["decoy"] == ""], ppm)
        print(f"[{st}] RT offsets (s): median |off| {np.median(np.abs(off)) * 60:.2f}, "
              f"max {np.max(np.abs(off)) * 60:.2f}", flush=True)
        meas = measure_all(pool, files, cand, off.values, ppm, tol)
        ev = evaluate(cand, meas, files, floor)
        tgt = ev[ev["decoy"] == ""]
        ok = group_peaks(tgt[tgt["tier"] == "A"])
        has_primary = ok.groupby("group")["ion"].transform(lambda x: x.isin(PRIMARY[mode]).any()).astype(bool)
        demoted = ok[~has_primary].drop(columns=["group", "group_height", "rt_rank"]).assign(tier="B")
        ok = group_peaks(ok[has_primary].drop(columns=["group", "group_height", "rt_rank"]))
        okB = pd.concat([tgt[tgt["tier"] == "B"], demoted]).assign(rt_rank=0)
        print(f"[{st}] formula ions: tier A {len(ok)} on {ok['std'].nunique()} standards, "
              f"tier B {len(okB)}", flush=True)

        # 5. co-eluting ions around each standard's main peak
        main = ok[ok["rt_rank"] == 1].sort_values("height_dom", ascending=False).drop_duplicates("std")
        sweep_rows = []
        for vial in (18, 19):
            m = main[main["dominant_vial"] == vial]
            if not len(m):
                continue
            fl = files[files["vial"] == vial]
            args = [(p, [(int(s), t + o, mz, fw / 60) for s, t, mz, fw in
                         zip(m["std"], m["rt_apex"], m["mz"], m["fwhm_s"])], ppm, cluster_ppm, floor)
                    for p, o in zip(fl["path"], off[fl.index])]
            for res in pool.map(w_sweep, args):
                sweep_rows += res
        sw = pd.DataFrame(sweep_rows, columns=["std", "mz", "height", "dt", "r", "file"])
        new = sweep_candidates(sw, pd.concat([ok, okB]), main, stds, ppm)
        print(f"[{st}] co-eluting candidates: {len(new)}", flush=True)
        if len(new):
            meas2 = measure_all(pool, files, new, off.values, ppm, tol)
            ev2 = evaluate(new, meas2, files, floor)
        else:
            ev2 = new
    # assemble
    form = pd.concat([ok, okB], ignore_index=True)
    form["source"] = "formula"
    form["annotation_class"] = "formula_ion"
    form["annotation"] = form["ion"]
    co = ev2[ev2["passed"]].copy() if len(ev2) else ev2
    if len(co):
        co["source"] = "coelution"
        co["rt_rank"] = np.where(co["tier"] == "A", 1, 0)
        mass = stds.set_index("std")["mass"]
        co = co.sort_values("height_dom", ascending=False).reset_index(drop=True)
        co["annotation"], co["annotation_class"] = "", ""
        for i, (s, m) in enumerate(zip(co["std"], co["mz_obs"])):
            # isotopes are looked up among the standard's formula ions and its stronger co-eluting ions
            refs = ok.loc[(ok["std"] == s) & (ok["rt_rank"] == 1), ["mz_obs", "ion"]].values.tolist()
            prev = co.iloc[:i]
            prev = prev[prev["std"] == s]
            refs += [(mm, nm if cl == "adduct" else f"{nm.split(' (')[0]} @{mm:.4f}")
                     for mm, nm, cl in prev[["mz_obs", "annotation", "annotation_class"]].values]
            co.loc[i, ["annotation", "annotation_class"]] = annotate(m, mass[s], mode, refs)
    truth = dedupe(pd.concat([form, co], ignore_index=True), ppm, tol)
    # standards whose main peaks co-elute and share a signature cannot be told apart
    truth = flag_shared(truth, main)
    truth = flag_artifacts(truth)
    broad = truth["fwhm_s"] > BROAD_X * truth.loc[truth["tier"] == "A", "fwhm_s"].median()
    truth["flags"] = [";".join(x for x in (f, "broad" if b else "") if x) for f, b in zip(truth["flags"], broad)]
    truth = truth.merge(stds[["std", "name", "formula"]], on="std")
    truth = truth.sort_values(["std", "tier", "rt_rank", "height_dom"], ascending=[True, True, True, False])
    truth.insert(0, "truth_id", [f"{st}_{i:04d}" for i in range(len(truth))])
    cols = ["truth_id", "std", "name", "formula", "tier", "source", "annotation", "annotation_class", "flags",
            "ion", "mz", "mz_obs", "mz_err_ppm", "rt_apex", "rt_left", "rt_right", "fwhm_s", "rt_rank",
            "shared_with", "also_matches", "log2FC", "log2_18_19", "weak_at_noise", "dominant_vial",
            "height_dom", "noise", "n_present_dom", "n_present_spike", "n_present_spike_matrix", "blank_frac",
            "matrix_frac", "e_free_resid", "e_fit_resid", "c_dilution_slope"]
    truth[cols].to_csv(out / "spiked_truth.tsv", sep="\t", index=False)

    # per-file read-out of every truth peak (for scoring tools file by file and for quantification)
    write_per_file(out, truth, files, off, ppm, tol, cpus)
    ev.to_csv(out / "all_tested_candidates.tsv", sep="\t", index=False)
    if len(ev2):
        ev2.to_csv(out / "coelution_tested.tsv", sep="\t", index=False)
    files.assign(rt_offset_s=off.values * 60).drop(columns=["path"]) \
        .to_csv(out / "files.tsv", sep="\t", index=False)
    summary = standards_summary(stds, ev[ev["decoy"] == ""], truth)
    summary.to_csv(out / "standards_summary.tsv", sep="\t", index=False)
    qc = decoy_qc(ev)
    qc.to_csv(out / "decoy_qc.tsv", sep="\t", index=False)
    params = dict(study=st, mode=mode, instrument=instrument, n_files=len(files), ppm=ppm,
                  cluster_ppm=cluster_ppm, MIN_SCANS=MIN_SCANS, MIN_REPS=MIN_REPS, PROM_FRAC=PROM_FRAC,
                  MIN_HEIGHT_X=MIN_HEIGHT_X, SNR_MIN=SNR_MIN, BASE_WIN_MIN=BASE_WIN_MIN, MAX_BLANK=MAX_BLANK,
                  LOCAL_MAX_SCANS=LOCAL_MAX_SCANS, R_AB_MIN=R_AB_MIN, MIN_DIL_SLOPE=MIN_DIL_SLOPE, SIG_TOL=SIG_TOL, MAX_RESID=MAX_RESID,
                  R_MIN=R_MIN, ARTIFACT_DA=ARTIFACT_DA, ARTIFACT_RATIO=ARTIFACT_RATIO, BROAD_X=BROAD_X,
                  apex_tol_s=tol * 60, offset_win_s=OFFSET_WIN_MIN * 60,
                  rt_group_s=RT_GROUP_MIN * 60, mass_decoys=MASS_DECOYS, floor=floor,
                  adducts=[a[0] for a in ADDUCTS[mode]], primary=sorted(PRIMARY[mode]), n_anchor=N_ANCHORS)
    (out / "params.json").write_text(json.dumps(params, indent=2))
    print(qc.to_string(index=False))
    print(truth.groupby(["tier", "source", "annotation_class"]).size().to_string())
    a = truth[truth["tier"] == "A"]
    print(f"[{st}] {len(truth)} truth peaks: tier A {len(a)} on {a['std'].nunique()} of {len(stds)} standards "
          f"({(a['rt_rank'] == 1).sum()} at the main RT), tier B {(truth['tier'] == 'B').sum()}", flush=True)
    return truth


def sweep_candidates(sw, ok, main, stds, ppm):
    """co-eluting clusters seen with r >= R_MIN in >= MIN_REPS injections and not already truth"""
    if not len(sw):
        return pd.DataFrame()
    sw = sw[sw["r"] >= R_MIN].sort_values(["std", "mz"])
    rows = []
    for s, g in sw.groupby("std"):
        mz = g["mz"].values
        cuts = np.flatnonzero(np.diff(mz) / mz[1:] * 1e6 > ppm) + 1
        f = g["file"].values
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(mz)]):
            n = len(set(f[a:b]))
            if n < MIN_REPS:
                continue
            rows.append(dict(std=s, mz=float(np.median(mz[a:b])), n_reps=n))
    new = pd.DataFrame(rows)
    if not len(new):
        return new
    mrow = main.set_index("std")
    new["rt"] = new["std"].map(mrow["rt_apex"])
    known = ok[["std", "mz_obs", "rt_apex"]].values
    keep = []
    for s, m, t in zip(new["std"], new["mz"], new["rt"]):
        hit = (np.abs(known[:, 1] - m) <= m * ppm * 1e-6) & (np.abs(known[:, 2] - t) <= RT_GROUP_MIN)
        keep.append(not hit.any())
    new = new[keep].reset_index(drop=True)
    s = stds.set_index("std")
    new["ion"] = "coeluting"
    new["log2FC"] = new["std"].map(s["log2FC"])
    new["a"] = new["std"].map(s["spikeA_vol"])
    new["b"] = new["std"].map(s["spikeB_vol"])
    new["decoy"] = ""
    return new


def flag_artifacts(truth):
    """truth peaks within ARTIFACT_DA of a >= 1/ARTIFACT_RATIO times stronger co-eluting truth ion
    (any standard): Orbitrap sidebands / ringing look exactly like real co-eluting ions"""
    flags = [""] * len(truth)
    t = truth.reset_index(drop=True)
    mz, rt, h = t["mz_obs"].values, t["rt_apex"].values, t["height_dom"].values
    for i in range(len(t)):
        near = (np.abs(rt - rt[i]) <= RT_GROUP_MIN) & (np.abs(mz - mz[i]) <= ARTIFACT_DA) \
            & (h * ARTIFACT_RATIO >= h[i])
        near[i] = False
        if near.any():
            flags[i] = "ft_artifact?"
    t["flags"] = flags
    return t


def flag_shared(truth, main):
    """standards whose main peaks are within RT_GROUP_MIN and share log2FC: their co-eluting ions
    could belong to either"""
    m = main[["std", "rt_apex", "log2FC"]].values
    shared = {}
    for s, t, fc in m:
        o = m[(m[:, 0] != s) & (np.abs(m[:, 1] - t) <= RT_GROUP_MIN) & (m[:, 2] == fc)]
        shared[int(s)] = ";".join(str(int(x)) for x in o[:, 0])
    truth["shared_with"] = [shared.get(int(s), "") if src == "coelution" else ""
                            for s, src in zip(truth["std"], truth["source"])]
    return truth


def write_per_file(out, truth, files, off, ppm, tol, cpus):
    t = truth.reset_index(drop=True)
    with mp.Pool(cpus) as pool:
        meas = measure_all(pool, files, t.assign(mz=t["mz_obs"].fillna(t["mz"]), rt=t["rt_apex"]),
                           off.values, ppm, tol)
    rows = []
    for j, f in enumerate(files["file"]):
        for i, tid in enumerate(t["truth_id"]):
            rows.append((tid, f, meas["height"][i, j], meas["area"][i, j], meas["apex_rt"][i, j],
                         meas["near_max"][i, j]))
    pd.DataFrame(rows, columns=["truth_id", "file", "height", "area", "apex_rt", "near_max"]) \
        .to_csv(out / "per_file.tsv.gz", sep="\t", index=False, float_format="%.6g")


def standards_summary(stds, tgt, truth):
    rows = []
    for _, s in stds.iterrows():
        c = tgt[tgt["std"] == s["std"]]
        tr = truth[(truth["std"] == s["std"]) & (truth["tier"] == "A") & (truth["rt_rank"] == 1)]
        if len(tr):
            best = tr.sort_values("height_dom", ascending=False).iloc[0]
            status, why = "found", ""
            rt, ion, h = best["rt_apex"], best["annotation"], best["height_dom"]
        else:
            status, rt, ion, h = "not_found", np.nan, "", np.nan
            if not len(c):
                why = "no candidate peak at any adduct"
            else:
                fails = [k for k in ("pass_present", "pass_blank", "pass_design", "pass_signature")
                         if not c[k].any()]
                why = ("no candidate passes: " + ",".join(fails)) if fails else \
                    ("tier B only" if (c["tier"] == "B").any() else "no single candidate passes all")
        rows.append(dict(std=s["std"], name=s["name"], formula=s["formula"], log2FC=s["log2FC"],
                         xlogp=s["xlogp"], status=status, reason=why, main_rt=rt, main_ion=ion,
                         main_height=h, n_tierA=int(((truth["std"] == s["std"]) & (truth["tier"] == "A")).sum()),
                         n_tierB=int(((truth["std"] == s["std"]) & (truth["tier"] == "B")).sum()),
                         n_candidates=len(c)))
    return pd.DataFrame(rows)


def decoy_qc(ev):
    rows = []
    for name, g in ev.groupby(ev["decoy"].replace("", "target")):
        rows.append(dict(search=name, candidates=len(g), present=int(g["pass_present"].sum()),
                         present_specific=int((g["pass_present"] & g["pass_blank"]).sum()),
                         tier_A=int((g["tier"] == "A").sum()), tier_B=int((g["tier"] == "B").sum())))
    qc = pd.DataFrame(rows)
    tgt = qc[qc["search"] == "target"]
    for t in ("tier_A", "tier_B"):
        if len(tgt) and tgt[t].iloc[0]:
            qc[f"{t}_per_target"] = qc[t] / tgt[t].iloc[0]
    return qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=list(STUDIES))
    ap.add_argument("--cpus", type=int, default=4)
    a = ap.parse_args()
    mode, instrument, _ = STUDIES[a.study]
    build(a.study, study_files(a.study), standards_table(), mode, instrument,
          ROOT / "data/truth/PR002464" / a.study, a.cpus)


if __name__ == "__main__":
    main()
