#!/usr/bin/env python3
"""Tool-independent 12C/13C credentialed truth set, built from raw centroided mzML.

Steps (see README "Credentialed truth"):
 1. per-file RT offsets vs 12C replicate 1 (measured on strong ions / their labeled partners)
 2. permissive candidate seeds from 12C runs: ppm-clustered centroids -> EIC -> local maxima
 3. strict pairing per seed: 13C partner at m + n*1.00335 that co-elutes, has the same
    peak shape (r >= R_MIN), and is present in >= MIN_REPS of 3 runs on both sides
 3b. label specificity: the seed m must be (nearly) absent from the 13C runs and the
    partner absent from the 12C runs - removes background ions present in both
 4. carbon-count check from the natural 12C M+1/M ratio -> tier A (consistent),
    tier B (M+1 too weak to test); testable-but-inconsistent candidates are rejected
 5. the same pipeline with decoys (wrong mass spacing; right spacing at the wrong RT)
    estimates the truth set's own false-credential rate
 6. feature classes: isotope peak / adduct of another truth feature, else "primary"

All thresholds are fixed here, before any tool is scored, and written to params.json.
usage: credtruth.py DATASET [--cpus N]     DATASET = SZ22 | YEAST
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.msdata import C13, load_run  # noqa: E402
from bench.runs import RUNS  # noqa: E402

# ------------------------------------------------------------- parameters ----
PPM = 5.0            # m/z tolerance for every EIC
CLUSTER_PPM = 3.0    # gap that splits centroid m/z clusters when seeding
MIN_SCANS = 4        # nonzero scans needed around an apex (seeding and presence)
R_MIN = 0.8          # 12C vs 13C peak-shape correlation
MIN_REPS = 2         # of 3 replicates, on each side
MAX_CROSS = 0.2      # label specificity: unlabeled m in 13C runs <= 0.2 x 12C height, and
                     # m + n*13C in 12C runs <= 0.2 x 13C height (+ natural M+n for n=1)
NAT13C = 0.010816    # natural 13C/12C per carbon (1.07% / 98.93%)
CARBON_TESTABLE = 10 # carbon check only if the expected M+1 is >= 10 x intensity floor
FWHM_S = {"SZ22": 1.4, "YEAST": 4.0}   # measured by cred_explore.py (median, top ions)
RT_DECOY_MIN = {"SZ22": 1.0, "YEAST": 2.0}
MASS_DECOYS = (-0.02, 0.02)


def rt_tol_min(ds, dt):
    # apex agreement: half a peak width, but never under 3 scans
    return max(0.5 * FWHM_S[ds] / 60, 3 * dt)


# ------------------------------------------------------------ data access ----
R12 = R13 = None
OFF12 = OFF13 = None
CFG = {}


def eic_win(run, mz, s0, s1):
    tol = mz * PPM * 1e-6
    a, b = np.searchsorted(run.mz, [mz - tol, mz + tol])
    out = np.zeros(s1 - s0)
    if b > a:
        sc = run.scan[a:b]
        k = (sc >= s0) & (sc < s1)
        np.maximum.at(out, sc[k] - s0, run.inten[a:b][k])
    return out


def window(run, t, half):
    return int(np.searchsorted(run.rt, t - half)), int(np.searchsorted(run.rt, t + half))


def apex_in(run, mz, t, tol, half):
    """apex near t (within tol) of the EIC at mz; returns (height, apex_rt, profile, rts)
    or None if absent (no signal there or fewer than MIN_SCANS nonzero scans around it)"""
    s0, s1 = window(run, t, half)
    e = eic_win(run, mz, s0, s1)
    rts = run.rt[s0:s1]
    near = np.abs(rts - t) <= tol
    if not near.any() or e[near].max() <= 0:
        return None
    k = np.flatnonzero(near)[np.argmax(e[near])]
    lo, hi = max(0, k - MIN_SCANS), min(len(e), k + MIN_SCANS + 1)
    run_len = 1
    j = k - 1
    while j >= lo and e[j] > 0:
        run_len += 1; j -= 1
    j = k + 1
    while j < hi and e[j] > 0:
        run_len += 1; j += 1
    if run_len < MIN_SCANS:
        return None
    return e[k], rts[k], e, rts


def height_near(runs, offs, mz, t, tol):
    """median over replicates of the max EIC intensity within tol of t (0 if absent)"""
    hs = []
    for r, o in zip(runs, offs):
        s0, s1 = window(r, t + o, tol)
        hs.append(eic_win(r, mz, s0, s1).max() if s1 > s0 else 0.0)
    return float(np.median(hs))


def corr_on_grid(p1, t1, p2, t2, t, half):
    g = np.linspace(t - half, t + half, 25)
    a = np.interp(g, t1, p1, left=0, right=0)
    b = np.interp(g, t2, p2, left=0, right=0)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ------------------------------------------------------------------ seeds ----
def seeds_from(run):
    mz, inten, scan = run.mz, run.inten, run.scan
    gaps = np.diff(mz) / mz[1:] * 1e6
    cuts = np.flatnonzero(gaps > CLUSTER_PPM) + 1
    bounds = np.concatenate([[0], cuts, [len(mz)]])
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < MIN_SCANS:
            continue
        e = np.zeros(run.n_scans)
        np.maximum.at(e, scan[a:b], inten[a:b])
        if np.count_nonzero(e) < MIN_SCANS:
            continue
        sm = np.convolve(e, np.ones(3) / 3, mode="same")
        pk, pr = find_peaks(sm, prominence=0)
        for k, prom in zip(pk, pr["prominences"]):
            if prom < 0.3 * sm[k]:
                continue
            lo, hi = max(0, k - 3), min(run.n_scans, k + 4)
            if np.count_nonzero(e[lo:hi]) < MIN_SCANS:
                continue
            sel = (scan[a:b] >= lo) & (scan[a:b] < hi)
            w = inten[a:b][sel]
            out.append((float(np.average(mz[a:b][sel], weights=w)), float(run.rt[k]), float(e[lo:hi].max())))
    return out


def merge_seeds(lists, tol_rt):
    """greedy: strongest first; drop weaker seeds within PPM and tol_rt of a kept one"""
    arr = pd.DataFrame([x for l in lists for x in l], columns=["mz", "rt", "h"]) \
        .sort_values("h", ascending=False).values
    idx = np.argsort(arr[:, 0])
    smz = arr[idx, 0]
    used = np.zeros(len(arr), bool)
    keep = []
    for i in range(len(arr)):
        if used[i]:
            continue
        m, t = arr[i, 0], arr[i, 1]
        keep.append(arr[i])
        lo, hi = np.searchsorted(smz, [m - m * PPM * 1e-6, m + m * PPM * 1e-6])
        for j in idx[lo:hi]:
            if abs(arr[j, 1] - t) <= tol_rt:
                used[j] = True
    return pd.DataFrame(keep, columns=["mz", "rt", "h"])


# ---------------------------------------------------------------- pairing ----
def presence(runs, offs, mz, t, tol, half):
    """apex per replicate at reference time t; stops early once MIN_REPS is out of reach.
    Returns list of (height, apex_rt_ref, profile, rts_ref) or None per replicate."""
    out, missing = [], 0
    for r, o in zip(runs, offs):
        p = apex_in(r, mz, t + o, tol, half)
        if p is None:
            missing += 1
            if len(runs) - missing < MIN_REPS:
                return None
            out.append(None)
        else:
            out.append((p[0], p[1] - o, p[2], p[3] - o))  # times back on the reference axis
    return out


def test_seed(args):
    """Full credentialing test of one seed for one spacing / RT shift; returns dict or None"""
    m, t, step, rt_shift = args
    tol, half, dt = CFG["tol"], CFG["half"], CFG["dt"]
    shape_half = 2 * FWHM_S[CFG["ds"]] / 60 + 3 * dt
    p12 = presence(R12, OFF12, m, t, tol, half)
    if p12 is None:
        return None
    ok12 = [p for p in p12 if p]
    i_best = int(np.argmax([p[0] if p else -1 for p in p12]))
    best12 = p12[i_best]
    t12 = float(np.median([p[1] for p in ok12]))
    cands = []
    for n in range(1, int(m // 12) + 1):
        # decoy RT shift: look for the partner rt_shift minutes away from the 12C apex
        p13 = presence(R13, OFF13, m + n * step, t12 + rt_shift, tol, half)
        if p13 is None:
            continue
        ok13 = [p for p in p13 if p]
        b13 = max(ok13, key=lambda p: p[0])
        r = corr_on_grid(best12[2], best12[3], b13[2], b13[3] - rt_shift, best12[1], shape_half)
        if r < R_MIN:
            continue
        # label specificity: background ions show up at m (and at the "partner") in both runs
        h12 = float(np.median([p[0] for p in ok12]))
        h13 = float(np.median([p[0] for p in ok13]))
        carry = height_near(R13, OFF13, m, t12 + rt_shift, tol) / h12
        natural = (n * NAT13C if n == 1 else 0.0) * h12
        leak = max(0.0, height_near(R12, OFF12, m + n * step, t12 + rt_shift, tol) - natural) / h13
        if carry <= MAX_CROSS and leak <= MAX_CROSS:
            cands.append((h13, r, n, ok13, carry, leak))
    if not cands:
        return None
    # with ~99% labeling the fully labeled species dominates the 13C run; partially
    # labeled M+(N-1) etc. co-elute with the same shape but are far weaker
    cands.sort(key=lambda c: (-c[0], -c[1]))
    h13_best, r, n, ok13, carry, leak = cands[0]
    ambiguous = len(cands) > 1 and cands[1][0] >= 0.5 * h13_best
    h12 = float(np.median([p[0] for p in ok12]))
    h13 = float(np.median([p[0] for p in ok13]))
    # carbon-count check on the strongest 12C replicate: natural M+1 / M ~ n x 1.08%
    # M+1 is read as a plain maximum near the apex (no scan-count rule: Orbitrap centroid
    # thresholding clips low isotope peaks to a few scans); its shape is checked only when
    # it has >= MIN_SCANS points
    i0 = best12[0]
    rr, oo = R12[i_best], OFF12[i_best]
    s0, s1 = window(rr, best12[1] + oo, half)
    e1 = eic_win(rr, m + C13, s0, s1)
    near = np.abs(rr.rt[s0:s1] - (best12[1] + oo)) <= tol
    i1 = float(e1[near].max()) if near.any() else 0.0
    if n * NAT13C * i0 < CARBON_TESTABLE * CFG["floor"]:
        n_iso, carbon = np.nan, "untestable"
    else:
        n_iso = i1 / i0 / NAT13C
        lo_, hi_ = n - max(1.0, 0.15 * n), n + max(2.0, 0.30 * n)
        shape_ok = True
        if np.count_nonzero(e1) >= MIN_SCANS:
            shape_ok = corr_on_grid(best12[2], best12[3], e1, rr.rt[s0:s1] - oo, best12[1], shape_half) >= 0.7
        carbon = "consistent" if (shape_ok and lo_ <= n_iso <= hi_) else "inconsistent"
    return dict(mz=m, rt=t12, n=n, r=r, ambiguous=ambiguous,
                reps12=len(ok12), reps13=len(ok13), height12=h12, height13=h13,
                log10_ratio=float(np.log10(h13 / h12)), n_iso=n_iso, carbon=carbon,
                unlabeled_in_13C=carry, partner_in_12C=leak,
                n_cands=len(cands))


def init_worker(ds, f12, f13, off12, off13, cfg):
    global R12, R13, OFF12, OFF13, CFG
    R12 = [load_run(f) for f in f12]; R13 = [load_run(f) for f in f13]
    OFF12, OFF13, CFG = off12, off13, cfg


def run_pairing(pool, seeds, step, rt_shift):
    args = [(m, t, step, rt_shift) for m, t in zip(seeds["mz"], seeds["rt"])]
    res = pool.map(test_seed, args, chunksize=200)
    return pd.DataFrame([r for r in res if r])


# ---------------------------------------------------------------- classes ----
def classify(df, tol):
    df = df.sort_values("mz").reset_index(drop=True)
    mz, rt, n, h = df["mz"].values, df["rt"].values, df["n"].values, df["height12"].values
    cls = np.array(["primary"] * len(df), dtype=object)
    rel = np.array([""] * len(df), dtype=object)

    def find(target, i):
        lo, hi = np.searchsorted(mz, [target - target * PPM * 1e-6, target + target * PPM * 1e-6])
        return [j for j in range(lo, hi) if abs(rt[j] - rt[i]) <= tol]

    for i in range(len(df)):
        # natural-isotope peak: parent at m - k*13C, co-eluting, stronger, one more carbon per step
        for k in (1, 2):
            for j in find(mz[i] - k * C13, i):
                if h[j] > h[i] and n[j] - n[i] == k:
                    cls[i], rel[i] = f"isotope_M+{k}", f"{mz[j]:.4f}"
        if cls[i] != "primary":
            continue
        for name, d in (("Na", 21.981944), ("K", 37.955882), ("NH4", 17.026549)):
            for j in find(mz[i] - d, i):
                if n[j] == n[i]:
                    cls[i], rel[i] = f"adduct_{name}", f"{mz[j]:.4f}"
    df["feature_class"] = cls
    df["related_to_mz"] = rel
    return df


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(FWHM_S))
    ap.add_argument("--cpus", type=int, default=4)
    a = ap.parse_args()
    ds = a.dataset
    out = ROOT / "data/truth" / ds
    out.mkdir(parents=True, exist_ok=True)
    f12, f13 = RUNS[f"{ds}_12C"]["files"], RUNS[f"{ds}_13C"]["files"]
    runs12 = [load_run(f) for f in f12]
    runs13 = [load_run(f) for f in f13]
    dt = float(np.median(np.diff(runs12[0].rt)))
    tol = rt_tol_min(ds, dt)
    half = max(4 * FWHM_S[ds] / 60, 10 * dt)
    floor = float(np.median([np.percentile(r.inten, 1) for r in runs12]))
    print(f"[{ds}] scan dt {dt * 60:.2f}s, apex tol {tol * 60:.2f}s, window +/-{half * 60:.1f}s, "
          f"intensity floor {floor:.0f}", flush=True)

    # 1. RT offsets: strongest ions of 12C rep1, apex in each file (13C: labeled partner)
    ref = runs12[0]
    top = merge_seeds([seeds_from(ref)], tol).nlargest(300, "h")
    def offset(run, labeled):
        d = []
        for m, t in zip(top["mz"], top["rt"]):
            best = None
            ns = range(1, int(m // 12) + 1) if labeled else [0]
            for n in ns:
                p = apex_in(run, m + n * C13, t, 0.5, 0.75)  # wide: offsets not known yet
                if p and (best is None or p[0] > best[0]):
                    best = p
            if best and best[0] > 20 * floor:
                d.append(best[1] - t)
        d = np.asarray(d)
        return float(np.median(d[np.abs(d) < 0.25])) if len(d) else 0.0
    off12 = [0.0] + [offset(r, False) for r in runs12[1:]]
    off13 = [offset(r, True) for r in runs13]
    print(f"[{ds}] RT offsets (s) 12C {np.round(np.array(off12) * 60, 2)} 13C {np.round(np.array(off13) * 60, 2)}", flush=True)

    # 2. seeds from all 12C replicates
    seeds = merge_seeds([seeds_from(r) for r in runs12], tol)
    seeds = seeds[(seeds["mz"] >= 60)].reset_index(drop=True)
    print(f"[{ds}] {len(seeds)} candidate seeds", flush=True)

    cfg = dict(ds=ds, tol=tol, half=half, dt=dt, floor=floor)
    params = dict(dataset=ds, files_12C=f12, files_13C=f13, PPM=PPM, CLUSTER_PPM=CLUSTER_PPM,
                  MIN_SCANS=MIN_SCANS, R_MIN=R_MIN, MIN_REPS=MIN_REPS, MAX_CROSS=MAX_CROSS, NAT13C=NAT13C,
                  FWHM_s=FWHM_S[ds], apex_tol_s=tol * 60, window_half_s=half * 60, floor=floor,
                  rt_offsets_12C_s=[o * 60 for o in off12], rt_offsets_13C_s=[o * 60 for o in off13],
                  carbon_window="n - max(1, 0.15n) <= n_iso <= n + max(2, 0.30n), M+1 shape r>=0.7",
                  carbon_testable=f"expected M+1 >= {CARBON_TESTABLE} x floor", partner_choice="most intense passing 13C partner",
                  mass_decoys=MASS_DECOYS, rt_decoy_min=RT_DECOY_MIN[ds], n_seeds=len(seeds))
    (out / "params.json").write_text(json.dumps(params, indent=2))
    seeds.to_csv(out / "seeds.tsv", sep="\t", index=False)

    with mp.Pool(a.cpus, initializer=init_worker, initargs=(ds, f12, f13, off12, off13, cfg)) as pool:
        target = run_pairing(pool, seeds, C13, 0.0)
        decoys = {}
        for d in MASS_DECOYS:
            decoys[f"mass{d:+.2f}"] = run_pairing(pool, seeds, C13 + d, 0.0)
        for s in (-RT_DECOY_MIN[ds], RT_DECOY_MIN[ds]):
            decoys[f"rt{s:+.1f}min"] = run_pairing(pool, seeds, C13, s)

    def tiers(df):
        df = df.copy()
        df["tier"] = np.where(df["carbon"] == "consistent", "A",
                              np.where(df["carbon"] == "untestable", "B", "rejected"))
        return df

    target = tiers(target)
    target["seed_id"] = np.arange(len(target))
    truth = classify(target[target["tier"] != "rejected"].copy(), tol)
    truth.insert(0, "truth_id", [f"{ds}_{i:05d}" for i in range(len(truth))])
    truth.to_csv(out / "credentialed.tsv", sep="\t", index=False)
    target.to_csv(out / "all_tested_pairs.tsv", sep="\t", index=False)

    rows = []
    for name, df in [("target", target)] + list(decoys.items()):
        df = tiers(df) if len(df) else df
        cnt = {t: int((df["tier"] == t).sum()) if len(df) else 0 for t in ("A", "B", "rejected")}
        rows.append(dict(search=name, **cnt))
    qc = pd.DataFrame(rows)
    pd.concat([tiers(df).assign(search=k) for k, df in decoys.items() if len(df)]) \
        .to_csv(out / "decoy_pairs.tsv", sep="\t", index=False)
    dec = qc[qc["search"] != "target"]
    for t in ("A", "B"):
        tgt = int(qc.loc[qc["search"] == "target", t].iloc[0])
        qc.loc[qc["search"] == "target", f"est_false_rate_{t}"] = (
            dec[t].max() / tgt if tgt else np.nan)  # worst decoy, per tier
    qc.to_csv(out / "decoy_qc.tsv", sep="\t", index=False)
    print(qc.to_string(index=False))
    print(truth.groupby(["tier", "feature_class"]).size().to_string())


if __name__ == "__main__":
    main()
