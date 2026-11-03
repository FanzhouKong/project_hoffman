#!/usr/bin/env python3
"""Biological-matrix truth for PR002464 (Orbitrap studies): plasma peaks and credentialed yeast peaks.

Needs the spiked-standard truth of the same study first (bench/spiketruth.py): its files.tsv holds the
per-injection RT offsets used here, and its peaks are cross-referenced.

plasma STUDY  every peak that behaves like a plasma metabolite (data/truth/PR002464/<study>/plasma_truth.tsv)
 1. seeds: local maxima of ppm-clustered centroids (bench/credtruth.py seeds_from) in the 4 injections of
    vials 13, 14 and 17 (1000 uL plasma: SRM 1950, SRM 8231, 1:1), merged, >= MIN_HEIGHT_X x floor
 2. measured in every plasma, yeast, spike and blank injection at the seed RT + the injection's offset
 3. tests (same rules as the spike truth where they apply):
    present   apex (local max of +-3 scans, >= 4 contiguous scans) in all but one injection of vial 17 (3 of 4)
    specific  median read in the 24 blanks <= MAX_BLANK x the vial-17 height
    dilution  plasma 1000 / 500 / 200 / 100 uL (vials 1-4, SRM 1950 : 8231 1:1): noise-subtracted heights,
              >= 3 levels above noise, log-log slope and R^2 (1 = proportional)
    mixing    SRM 1950 : SRM 8231 at 1:0, 0:1, 1:3, 3:1, 1:1 (vials 13-17) fits alpha f + beta (1 - f),
              alpha, beta >= 0 (mean relative residual <= MAX_RESID)
    tier A: slope within 1 +- SLOPE_TOL_A and R^2 >= R2_A; tier B: within 1 +- SLOPE_TOL_B and R^2 >= R2_B
 4. null: the same tests with the dilution amounts permuted (two orders far from monotone) count how many
    seeds pass by chance
 5. annotation: natural isotope of a stronger truth peak; also seen in yeast-only / spike-only vials;
    same m/z and RT as a spiked-standard truth peak; flags ft_artifact? / broad as in the spike truth;
    known common metabolites ([M+H]+ / [M-H]-) as a sanity check

yeast STUDY   checks the PR002464-specific design adds to the credentialed yeast pairs that
              `credtruth.py PR2464_<study>` built (vial 23 12C vs vial 24 U-13C):
    in vial 25 (12C : 13C 1:1) both partners present and co-eluting; median blank read of the 12C peak;
    dilution slope of both partners over the yeast series (vials 5-8: 12C + 13C 1:1 at 1000 / 500 / 200 /
    100 uL). Written to data/truth/PR002464/<study>/yeast_truth.tsv (all credentialed peaks, the checks as
    columns, `confirmed` = present in vial 25 on both sides and absent from the blanks).
usage: biotruth.py {plasma,yeast} STUDY [--cpus N]
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import spiketruth as stt  # noqa: E402
from bench.cred_gallery import KNOWN, mass as formula_mass_simple  # noqa: E402
from bench.credtruth import merge_seeds, seeds_from  # noqa: E402
from bench.msdata import C13, load_run  # noqa: E402

TRUTH = ROOT / "data/truth/PR002464"

# ------------------------------------------------------------- parameters ----
MIN_HEIGHT_X = 10.0   # seeds: height >= 10 x intensity floor (as spike discovery)
SEED_RT_MERGE = 0.05  # seeds from different injections within 3 s and 5 ppm are one seed
MAX_MISSING = 1       # injections of a vial (17, 25) a peak may miss, at least 2 holding it: 3 of 4, and 2 of the
                      # 2-3 left when a vial lost injections to the m/z registry (bench/mz_registry.py)
MAX_BLANK = 0.1
MAX_RESID = 0.3
MIN_LEVELS = 3        # dilution levels above noise needed for a slope
SLOPE_TOL_A, R2_A = 0.25, 0.95
SLOPE_TOL_B, R2_B = 0.5, 0.80
ALSO_IN = 0.1         # "also in yeast / spike vials": their median read >= 10 % of the plasma height
PLASMA_DIL = [1, 2, 3, 4]          # vials: 1000, 500, 200, 100 uL plasma (1:1)
PLASMA_MIX = [13, 14, 15, 16, 17]  # SRM 1950 share 1, 0, 0.25, 0.75, 0.5
YEAST_DIL = [5, 6, 7, 8]           # yeast 12C + 13C 1:1 at 1000 / 500 / 200 / 100 uL
PERMUTED = ([200, 1000, 100, 500], [500, 100, 1000, 200])   # vial 1..4 amounts, far from monotone


def min_reps(n):
    """injections of a vial with n of them that must hold a peak"""
    return max(2, n - MAX_MISSING)


def offsets(study, files):
    o = pd.read_csv(TRUTH / study / "files.tsv", sep="\t").set_index("file")["rt_offset_s"] / 60
    return files["file"].map(o).fillna(0.0).values


def w_seeds(args):
    """seeds of one injection above MIN_HEIGHT_X x its floor, on the offset-free time axis"""
    path, off = args
    run = load_run(path)
    floor = float(np.percentile(run.inten, 1))
    return [(m, t - off, h) for m, t, h in seeds_from(run) if h >= MIN_HEIGHT_X * floor], floor


def measure_all(pool, files, off, mzs, rts, ppm, tol):
    res = pool.map(stt.w_measure, [(p, mzs, rts + o, ppm, tol) for p, o in zip(files["path"], off)])
    return {k: np.stack([r[k] for r in res], axis=1) for k in res[0]}


def vmed(M, files, vials, absent_zero=True):
    """per-candidate median over the injections of each vial -> (cand x len(vials))"""
    out = []
    for v in vials:
        x = M[:, (files["vial"] == v).values]
        out.append(np.median(np.nan_to_num(x), axis=1) if absent_zero else np.nanmedian(x, axis=1))
    return np.stack(out, axis=1)


def loglog(h, amt, noise):
    """slope and R^2 of log2 height vs log2 amount over levels with h > noise; NaN if < MIN_LEVELS"""
    n = len(h)
    slope, r2, nlev = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, int)
    x = np.log2(np.asarray(amt, float))
    for i in range(n):
        ok = h[i] > noise[i]
        nlev[i] = ok.sum()
        if nlev[i] < MIN_LEVELS:
            continue
        y = np.log2(h[i, ok])
        p = np.polyfit(x[ok], y, 1)
        resid = y - np.polyval(p, x[ok])
        tot = ((y - y.mean()) ** 2).sum()
        slope[i], r2[i] = p[0], 1 - (resid ** 2).sum() / tot if tot > 0 else 0.0
    return slope, r2, nlev


def dilution_pass(slope, r2, tol, r2min):
    return (np.abs(slope - 1) <= tol) & (r2 >= r2min)


# ----------------------------------------------------------------- plasma ----
def build_plasma(study, cpus):
    mode, instrument, _ = stt.STUDIES[study]
    out = TRUTH / study
    files = stt.study_files(study)
    files["off"] = offsets(study, files)
    tol = json.loads((out / "params.json").read_text())["apex_tol_s"] / 60   # the spike truth's
    spike = pd.read_csv(out / "spiked_truth.tsv", sep="\t")
    return plasma_core(study, files, out, mode, stt.PPM[instrument], tol, spike, cpus)


def plasma_core(study, files, out, mode, ppm, tol, spike, cpus):
    """files: study_files() + "off" (min); spike: the spiked-standard truth (cross-reference)"""
    files = files.copy()
    files["plasma"] = files["SRM 1950"] + files["SRM 8231"]
    seedf = files[files["vial"].isin([13, 14, 17])]
    keep = files["vial"].isin(PLASMA_DIL + PLASMA_MIX + [5, 6, 7, 8, 23, 24, 25, 9, 10, 11, 12, 18, 19, 20, 21, 22]) \
        | (files["cls"] == "blank")
    mf = files[keep].reset_index(drop=True)
    with mp.Pool(cpus) as pool:
        res = pool.map(w_seeds, list(zip(seedf["path"], seedf["off"])))
        floor = float(np.median([r[1] for r in res]))
        seeds = merge_seeds([r[0] for r in res], SEED_RT_MERGE)
        seeds = seeds[(seeds["h"] >= MIN_HEIGHT_X * floor) & (seeds["mz"] >= 50)].reset_index(drop=True)
        print(f"[{study} plasma] floor {floor:.0f}; {len(seeds)} seeds from {len(seedf)} injections", flush=True)
        meas = measure_all(pool, mf, mf["off"].values, seeds["mz"].values, seeds["rt"].values, ppm, tol)
    c = evaluate_plasma(seeds, meas, mf, floor)
    # nulls: dilution amounts permuted
    hd = np.maximum(vmed(meas["near_max"], mf, PLASMA_DIL) - c["noise"].values[:, None], 0)
    base = c["pass_present"] & c["pass_blank"] & c["pass_mixing"]
    nulls = []
    for perm in PERMUTED:
        sl, r2, _ = loglog(hd, perm, c["noise"].values)
        nulls.append(dict(order=str(perm), tier_A=int((base & dilution_pass(sl, r2, SLOPE_TOL_A, R2_A)).sum()),
                          tier_AB=int((base & dilution_pass(sl, r2, SLOPE_TOL_B, R2_B)).sum())))
    truth = c[c["tier"] != ""].copy()
    truth = annotate_plasma(truth, spike)
    truth = truth.sort_values(["tier", "height_17"], ascending=[True, False])
    truth.insert(0, "truth_id", [f"{study}_P{i:05d}" for i in range(len(truth))])
    cols = ["truth_id", "tier", "mz_obs", "rt_apex", "rt_left", "rt_right", "fwhm_s", "height_17", "noise",
            "n_present_17", "blank_frac", "dil_slope", "dil_r2", "dil_levels", "mix_resid", "log2_1950_8231",
            "also_in_yeast", "also_in_spike", "spike_truth", "annotation", "flags"]
    truth[cols].to_csv(out / "plasma_truth.tsv", sep="\t", index=False)
    c.to_csv(out / "plasma_all_tested.tsv.gz", sep="\t", index=False, float_format="%.6g")
    tgt = dict(order="true", tier_A=int((c["tier"] == "A").sum()), tier_AB=int((c["tier"] != "").sum()))
    qc = pd.DataFrame([tgt] + nulls)
    qc.to_csv(out / "plasma_null_qc.tsv", sep="\t", index=False)
    kc = known_check(truth, mode)
    kc.to_csv(out / "plasma_known_metabolites.tsv", sep="\t", index=False)
    (out / "plasma_params.json").write_text(json.dumps(dict(
        study=study, ppm=ppm, apex_tol_s=tol * 60, floor=floor, n_seeds=len(seeds), MIN_HEIGHT_X=MIN_HEIGHT_X,
        SEED_RT_MERGE=SEED_RT_MERGE, MAX_MISSING=MAX_MISSING, MAX_BLANK=MAX_BLANK, MAX_RESID=MAX_RESID,
        MIN_LEVELS=MIN_LEVELS, tier_A=dict(slope_tol=SLOPE_TOL_A, r2=R2_A), tier_B=dict(slope_tol=SLOPE_TOL_B, r2=R2_B),
        ALSO_IN=ALSO_IN, dilution_vials=PLASMA_DIL, mixing_vials=PLASMA_MIX, permuted=PERMUTED,
        n_files=len(mf)), indent=2))
    print(qc.to_string(index=False))
    print(truth.groupby(["tier", "annotation"]).size().to_string() if len(truth) else "no truth")
    print(f"[{study} plasma] {len(truth)} truth peaks: tier A {(truth['tier'] == 'A').sum()}, "
          f"tier B {(truth['tier'] == 'B').sum()}; known metabolites matched {kc['metabolite'].nunique() if len(kc) else 0}",
          flush=True)
    return truth


def evaluate_plasma(seeds, meas, files, floor):
    H, near = meas["height"], meas["near_max"]
    c = seeds.copy()
    blank = (files["cls"] == "blank").values
    noise = np.maximum(np.median(near[:, blank], axis=1), floor)
    c["noise"] = noise
    h17 = vmed(H, files, [17])[:, 0]
    c["height_17"] = h17
    c["n_present_17"] = np.sum(~np.isnan(H[:, (files["vial"] == 17).values]), axis=1)
    c["blank_frac"] = np.median(near[:, blank], axis=1) / np.maximum(h17, 1)
    amt = [files.loc[files["vial"] == v, "plasma"].iloc[0] for v in PLASMA_DIL]
    hd = np.maximum(vmed(near, files, PLASMA_DIL) - noise[:, None], 0)
    c["dil_slope"], c["dil_r2"], c["dil_levels"] = loglog(hd, amt, noise)
    f = np.array([files.loc[files["vial"] == v, "SRM 1950"].iloc[0] / files.loc[files["vial"] == v, "plasma"].iloc[0]
                  for v in PLASMA_MIX])
    hm = np.maximum(vmed(near, files, PLASMA_MIX) - noise[:, None], 0)
    M = np.stack([f, 1 - f], axis=1) * 1000
    resid = np.full(len(c), np.nan)
    for i in range(len(c)):
        if hm[i].sum() > 0:
            coef, _ = nnls(M, hm[i])
            resid[i] = np.abs(hm[i] - M @ coef).sum() / hm[i].sum()
    c["mix_resid"] = resid
    with np.errstate(divide="ignore", invalid="ignore"):
        c["log2_1950_8231"] = np.log2(np.maximum(hm[:, 0], noise) / np.maximum(hm[:, 1], noise))
    yeast = files["vial"].isin([5, 6, 7, 8, 23, 24, 25]).values
    spike = (files["cls"] == "spike").values
    c["also_in_yeast"] = np.median(near[:, yeast], axis=1) >= ALSO_IN * np.maximum(h17, 1)
    c["also_in_spike"] = np.median(near[:, spike], axis=1) >= ALSO_IN * np.maximum(h17, 1)
    c["pass_present"] = c["n_present_17"] >= min_reps(int((files["vial"] == 17).sum()))
    c["pass_blank"] = c["blank_frac"] <= MAX_BLANK
    c["pass_mixing"] = c["mix_resid"] <= MAX_RESID
    ok = c["pass_present"] & c["pass_blank"] & c["pass_mixing"]
    a = ok & dilution_pass(c["dil_slope"], c["dil_r2"], SLOPE_TOL_A, R2_A)
    b = ok & ~a & dilution_pass(c["dil_slope"], c["dil_r2"], SLOPE_TOL_B, R2_B)
    c["tier"] = np.where(a, "A", np.where(b, "B", ""))
    pv = files["vial"].isin(PLASMA_MIX + PLASMA_DIL).values
    with np.errstate(all="ignore"):
        c["rt_apex"] = np.nanmedian(meas["apex_rt"][:, pv], axis=1)
        c["rt_left"] = np.nanmedian(meas["left"][:, pv], axis=1)
        c["rt_right"] = np.nanmedian(meas["right"][:, pv], axis=1)
        c["fwhm_s"] = np.nanmedian(meas["fwhm"][:, pv], axis=1) * 60
        c["mz_obs"] = np.nanmedian(meas["mz_obs"][:, pv], axis=1)
    return c


def annotate_plasma(t, sp):
    t = t.sort_values("height_17", ascending=False).reset_index(drop=True)
    mz, rt, h = t["mz_obs"].values, t["rt_apex"].values, t["height_17"].values
    ann = ["" for _ in range(len(t))]
    for i in range(len(t)):
        for k in (1, 2):
            j = np.flatnonzero((np.abs(mz - (mz[i] - k * C13)) <= mz[i] * 5e-6) & (np.abs(rt - rt[i]) <= 0.05)
                               & (h > h[i]))
            if len(j):
                ann[i] = f"isotope M+{k} of {mz[j[0]]:.4f}"
                break
    t["annotation"] = [a.split(" of ")[0] if a else "" for a in ann]
    t["annotation"] = t["annotation"].replace("", "primary")
    hit = []
    for m, r in zip(mz, rt):
        k = sp[(np.abs(sp["mz_obs"] - m) <= m * 5e-6) & (np.abs(sp["rt_apex"] - r) <= 0.05)]
        hit.append(k["truth_id"].iloc[0] if len(k) else "")
    t["spike_truth"] = hit
    t["source"] = "plasma"
    t = stt.flag_artifacts(t.rename(columns={"height_17": "height_dom"})).rename(columns={"height_dom": "height_17"})
    broad = t["fwhm_s"] > 3.0 * t.loc[t["tier"] == "A", "fwhm_s"].median()
    t["flags"] = [";".join(x for x in (f, "broad" if b else "") if x) for f, b in zip(t["flags"], broad)]
    return t


def known_check(truth, mode):
    sign = 1 if mode == "pos" else -1
    rows = []
    for name, f in KNOWN.items():
        target = formula_mass_simple(f) + sign * stt.PROTON
        k = truth[np.abs(truth["mz_obs"] - target) <= target * 5e-6]
        for _, r in k.iterrows():
            rows.append(dict(metabolite=name, formula=f, mz_expected=round(target, 5), truth_id=r["truth_id"],
                             tier=r["tier"], rt=round(r["rt_apex"], 3), height_17=r["height_17"]))
    return pd.DataFrame(rows, columns=["metabolite", "formula", "mz_expected", "truth_id", "tier", "rt", "height_17"])


# ------------------------------------------------------------------ yeast ----
def build_yeast(study, cpus):
    mode, instrument, _ = stt.STUDIES[study]
    ppm = stt.PPM[instrument]
    out = TRUTH / study
    cred = pd.read_csv(out / "yeast_cred/credentialed.tsv", sep="\t")
    files = stt.study_files(study)
    files["off"] = offsets(study, files)
    tol = json.loads((out / "params.json").read_text())["apex_tol_s"] / 60
    mf = files[files["vial"].isin([25] + YEAST_DIL) | (files["cls"] == "blank")].reset_index(drop=True)
    m12 = cred["mz"].values
    m13 = cred["mz"].values + cred["n"].values * C13
    rts = cred["rt"].values
    # credtruth RTs are on the axis of its 12C replicate 1 (the first vial-23 injection); the spike-truth
    # offsets are injection RT minus the offset-free RT, so subtract that injection's offset
    f23 = files[files["vial"] == 23].iloc[0]
    rts = rts - f23["off"]
    with mp.Pool(cpus) as pool:
        a = measure_all(pool, mf, mf["off"].values, np.concatenate([m12, m13]), np.concatenate([rts, rts]), ppm, tol)
    n = len(cred)
    H12, H13 = a["height"][:n], a["height"][n:]
    N12, N13 = a["near_max"][:n], a["near_max"][n:]
    v25 = (mf["vial"] == 25).values
    blank = (mf["cls"] == "blank").values
    t = cred.copy()
    t["n_mixed_12C"] = np.sum(~np.isnan(H12[:, v25]), axis=1)
    t["n_mixed_13C"] = np.sum(~np.isnan(H13[:, v25]), axis=1)
    with np.errstate(all="ignore"):
        t["mixed_dt_s"] = np.nanmedian(a["apex_rt"][:n][:, v25] - a["apex_rt"][n:][:, v25], axis=1) * 60
    h25 = np.median(np.nan_to_num(H12[:, v25]), axis=1)
    t["blank_frac"] = np.median(N12[:, blank], axis=1) / np.maximum(h25, 1)
    floor = 100.0
    noise12 = np.maximum(np.median(N12[:, blank], axis=1), floor)
    noise13 = np.maximum(np.median(N13[:, blank], axis=1), floor)
    amt = [1000, 500, 200, 100]
    for lab, N, noise in (("12C", N12, noise12), ("13C", N13, noise13)):
        hd = np.maximum(vmed(N, mf, YEAST_DIL) - noise[:, None], 0)
        t[f"dil_slope_{lab}"], t[f"dil_r2_{lab}"], _ = loglog(hd, amt, noise)
    need = min_reps(int(v25.sum()))
    t["confirmed"] = (t["n_mixed_12C"] >= need) & (t["n_mixed_13C"] >= need) & (t["blank_frac"] <= MAX_BLANK)
    t["mz_13C"] = m13
    t.to_csv(out / "yeast_truth.tsv", sep="\t", index=False)
    s = t.groupby("tier")["confirmed"].agg(["sum", "count"])
    print(s.to_string())
    print(f"[{study} yeast] {len(t)} credentialed peaks, {int(t['confirmed'].sum())} confirmed in vial 25 and "
          f"absent from blanks; median dilution slope 12C {np.nanmedian(t['dil_slope_12C']):.2f}", flush=True)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["plasma", "yeast"])
    ap.add_argument("study", choices=[s for s in stt.STUDIES if s != "ST004504"])
    ap.add_argument("--cpus", type=int, default=4)
    a = ap.parse_args()
    (build_plasma if a.what == "plasma" else build_yeast)(a.study, a.cpus)


if __name__ == "__main__":
    main()
