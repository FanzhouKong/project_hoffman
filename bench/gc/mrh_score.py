#!/usr/bin/env python3
"""Score a peak3d output of the MRH GC-Orbitrap set (Zenodo 18596747) against its ground truth.

Truth: groundtruth_corrected_<level>.csv, one per injection (NEWSTD_<level>.mzML), 1,065 rows = the expected EI
ions of 59 standards (INTERFLAB.csv: compound, nominal RT, up to 20 fragment m/z each). The authors moved every
expected ion to the closest local maximum of that file's raw data and recorded it (mz_matched, rt_matched,
intensity_matched); "matched" is False where they found no data.

The window of that search was wide (0.1 Da, and up to 12 s from the nominal RT), so a recorded point is not always
the expected ion: at 0.1 ng/mL 70 % of the recorded m/z are more than 10 ppm from the expected m/z, and the ions
of one compound are recorded up to 24 s apart. Each truth row is therefore screened, without looking at any picker
output:
  mass_ok   recorded m/z within --ppm of the expected m/z
  rt_ok     recorded RT within --rt-win of the compound's consensus RT in that file (the densest group of its
            mass_ok ions; a compound needs --min-ions of them to have one)
  raw_peak  on the centroided run, the ion (within --ppm) is present in at least 5 consecutive scans around the
            recorded point (needs the output's .cache)
  raw_apex  the recorded point is a top of that trace: walking away from it on either side, the trace falls below
            0.7 x its height before it exceeds 1.5 x (a point on the flank of a peak whose apex lay outside the
            authors' window fails: DBDPE, recorded 6 s before its apex at a tenth of the height)
  responds  the ion's intensity follows the concentration: the log-log slope of intensity against ng/mL over the
            neighbouring levels (two each side, median of the pairwise slopes) is at least 0.5. With intensity =
            background + k x concentration that slope is the compound's share of the recorded intensity, so a
            row passes when the compound is at least half of it. Column-bleed ions on the list fail at every
            level (m/z 207.032 and 281.051 of the siloxanes: the same intensity from 0.1 to 1000 ng/mL), a real
            fragment fails at the levels where it is under the background. Not estimable (fewer than three
            levels) counts as passed, and so do the internal standards (Category IS: one amount in every level).
  clean     all five: this ion co-elutes with its compound at the expected mass, is a chromatographic peak with
            its top at the recorded point, and comes from the compound
Scores are given for all recorded rows and for the clean ones; only the clean ones are evidence about the picker.

Ion level (the picker). A truth point is found when a picked feature lies within --ppm in m/z and --rt-tol
seconds (apex) of it. A miss is explained from the rejected candidates (`peak3d process --keep-rejected`): the
reason of the closest rejected candidate, "inside" when a picked feature of that m/z spans the point but has its
apex elsewhere, else "no candidate".

Compound level (the deconvolution, `peak3d gc`). For a compound with a consensus RT, the components within
--rt-win of it are candidates; best = the largest share of the compound's clean ions found in one component's
spectrum, any = the share found in at least one. A compound whose ions sit in one component has best = any.

    python bench/gc/mrh_score.py --output results/gc_mrh/peak3d --truth data/raw/MRH_GC_ORBITRAP --out results/gc_mrh/score
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LEVELS = ["0_1", "0_5", "1", "5", "10", "25", "50", "100", "250", "500", "1000"]
MIN_RUN = 5                # consecutive scans that make a chromatographic peak (raw_peak)
APEX_RISE, APEX_FALL = 1.5, 0.7     # raw_apex: no rise above 1.5 x before a fall below 0.7 x
RESPONSE_MIN = 0.5         # responds: least log-log slope of intensity against concentration
RESPONSE_SPAN = 2          # levels on each side that enter the slope


def level_value(level: str) -> float:
    return float(level.replace("_", "."))


def load_truth(path) -> pd.DataFrame:
    t = pd.read_csv(path)
    out = pd.DataFrame({
        "compound": t["CompoundName"].astype(str).str.strip(),
        "category": t["Category"].astype(str),
        "rt_nominal": t["Retention Time"].astype(float),
        "mz_expected": t["m/z"].astype(float),
        "matched": t["matched"].astype(str).str.lower().eq("true"),
        "mz": pd.to_numeric(t["mz_matched"], errors="coerce"),
        "rt": pd.to_numeric(t["rt_matched"], errors="coerce"),
        "intensity": pd.to_numeric(t["intensity_matched"], errors="coerce"),
    })
    out["matched"] &= out["mz"].notna() & out["rt"].notna()
    # the list holds some compounds twice under one name (two nominal RTs): the entry is name + nominal RT
    out["entry"] = out["compound"] + " @" + out["rt_nominal"].map(lambda v: f"{v:g}")
    out["ppm"] = (out["mz"] / out["mz_expected"] - 1.0) * 1e6
    return out


def consensus_rt(rt, win: float, min_ions: int) -> float:
    """centre (median) of the densest group of values within +/- win of one of them; nan below min_ions"""
    rt = np.sort(np.asarray(rt, np.float64))
    if len(rt) < min_ions:
        return float("nan")
    n = np.array([np.sum(np.abs(rt - x) <= win) for x in rt])
    k = int(np.argmax(n))
    if n[k] < min_ions:
        return float("nan")
    return float(np.median(rt[np.abs(rt - rt[k]) <= win]))


def screen(truth: pd.DataFrame, ppm: float = 10.0, rt_win_s: float = 1.5, min_ions: int = 3) -> pd.DataFrame:
    """adds mass_ok, rt_consensus (per entry), rt_ok"""
    t = truth.copy()
    t["mass_ok"] = t["matched"] & (t["ppm"].abs() <= ppm)
    cons = {e: consensus_rt(g.loc[g["mass_ok"], "rt"].values, rt_win_s / 60.0, min_ions) for e, g in t.groupby("entry")}
    t["rt_consensus"] = t["entry"].map(cons)
    t["rt_ok"] = t["mass_ok"] & ((t["rt"] - t["rt_consensus"]).abs() * 60.0 <= rt_win_s)
    return t


def raw_peak(truth: pd.DataFrame, cloud, ppm: float = 10.0):
    """(peak, apex) per truth row. peak: the ion is present in >= MIN_RUN consecutive scans that include, or lie
    next to, the recorded scan. apex: the recorded point is a top of the trace (module docstring)."""
    peak = np.zeros(len(truth), bool)
    apex = np.zeros(len(truth), bool)
    S = cloud.n_scans
    for k, (ok, m, r) in enumerate(zip(truth["matched"].values, truth["mz"].values, truth["rt"].values)):
        if not ok:
            continue
        s = int(np.clip(np.searchsorted(cloud.rt, r), 0, S - 1))
        if s > 0 and abs(cloud.rt[s - 1] - r) < abs(cloud.rt[s] - r):
            s -= 1
        lo, hi = max(0, s - 3 * MIN_RUN), min(S, s + 3 * MIN_RUN + 1)
        v = np.asarray(cloud.eic(float(m), ppm, lo, hi), np.float64)
        y = v > 0
        c = s - lo
        best = 0
        for start in (c - 1, c, c + 1):                      # the run through the recorded scan or a neighbour
            if 0 <= start < len(y) and y[start]:
                a = start
                while a > 0 and y[a - 1]:
                    a -= 1
                b = start
                while b < len(y) - 1 and y[b + 1]:
                    b += 1
                best = max(best, b - a + 1)
        peak[k] = best >= MIN_RUN
        y0 = float(v[max(0, c - 1):c + 2].max())
        top = y0 > 0
        for step in (-1, 1):
            j = c + step
            while top and 0 <= j < len(v) and v[j] >= APEX_FALL * y0:
                top = v[j] <= APEX_RISE * y0
                j += step
        apex[k] = top
    return peak, apex


def dose_response(rows: pd.DataFrame) -> np.ndarray:
    """Per row of the truth tables of all levels together (columns level, entry, mz_expected, intensity, matched,
    mass_ok, rt_ok): the log-log slope of intensity against concentration for that ion over the levels within
    RESPONSE_SPAN positions of the row's own, median of the pairwise slopes; NaN with fewer than three levels."""
    out = np.full(len(rows), np.nan)
    use = (rows["matched"] & rows["mass_ok"] & rows["rt_ok"]).values & (rows["intensity"].values > 0)
    pos = rows["level"].astype(str).map({lv: i for i, lv in enumerate(LEVELS)}).values
    x = np.log10(np.array([level_value(str(lv)) for lv in rows["level"]]))
    y = np.log10(np.where(use, rows["intensity"].values, 1.0).astype(np.float64))
    sub = rows[use].assign(_i=np.flatnonzero(use))
    for _, g in sub.groupby(["entry", "mz_expected"], sort=False):
        idx = g["_i"].values
        for i in idx:
            near = idx[np.abs(pos[idx] - pos[i]) <= RESPONSE_SPAN]
            if len(np.unique(pos[near])) < 3:
                continue
            a, b = np.triu_indices(len(near), 1)
            dx = x[near[b]] - x[near[a]]
            ok = dx != 0
            out[i] = float(np.median((y[near[b]] - y[near[a]])[ok] / dx[ok]))
    return out


def match_features(t_mz, t_rt, f_mz, f_rt, f_lo, f_hi, ppm: float = 10.0, rt_tol_s: float = 2.0):
    """per truth point: (index of the closest feature in RT within ppm, or -1; its RT distance in s; found;
    inside = a feature within ppm spans the point but none has its apex within rt_tol_s)"""
    o = np.argsort(f_mz, kind="stable")
    smz = np.asarray(f_mz, np.float64)[o]
    n = len(t_mz)
    idx = np.full(n, -1, np.int64)
    d_rt = np.full(n, np.nan)
    found = np.zeros(n, bool)
    inside = np.zeros(n, bool)
    for k in range(n):
        if not np.isfinite(t_mz[k]):
            continue
        a = np.searchsorted(smz, t_mz[k] * (1 - ppm * 1e-6))
        b = np.searchsorted(smz, t_mz[k] * (1 + ppm * 1e-6), side="right")
        if b <= a:
            continue
        c = o[a:b]
        d = np.abs(np.asarray(f_rt)[c] - t_rt[k]) * 60.0
        j = int(np.argmin(d))
        idx[k], d_rt[k] = c[j], d[j]
        found[k] = d[j] <= rt_tol_s
        inside[k] = (not found[k]) and bool(np.any((np.asarray(f_lo)[c] <= t_rt[k]) & (np.asarray(f_hi)[c] >= t_rt[k])))
    return idx, d_rt, found, inside


def score_compounds(t: pd.DataFrame, comps: pd.DataFrame, off, mz, ppm: float, rt_win_s: float, min_ions: int):
    """one row per truth entry with >= min_ions clean ions: n_clean, n_components (candidates holding >= 2 of
    them), best, any (shares of the clean ions), component (the best one), d_rt_s"""
    rows = []
    c_rt = comps["rt"].values if len(comps) else np.zeros(0)
    for e, g in t[t["clean"]].groupby("entry"):
        ions = np.unique(np.round(g["mz"].values, 4))
        if len(ions) < min_ions:
            continue
        rt0 = float(g["rt_consensus"].iloc[0])
        cand = np.flatnonzero(np.abs(c_rt - rt0) * 60.0 <= rt_win_s)
        have = np.zeros((len(cand), len(ions)), bool)
        for q, c in enumerate(cand):
            m = mz[off[c]:off[c + 1]]
            if len(m):
                k = np.clip(np.searchsorted(m, ions), 1, len(m) - 1) if len(m) > 1 else np.zeros(len(ions), np.int64)
                near = np.minimum(np.abs(m[k] - ions), np.abs(m[np.maximum(k - 1, 0)] - ions))
                have[q] = near <= ions * ppm * 1e-6
        n_in = have.sum(axis=1) if len(cand) else np.zeros(0, np.int64)
        b = int(np.argmax(n_in)) if len(cand) else -1
        rows.append(dict(entry=e, rt_consensus=rt0, n_clean=len(ions), n_candidates=len(cand),
                         n_components=int((n_in >= 2).sum()), best=float(n_in[b] / len(ions)) if b >= 0 else 0.0,
                         any=float(have.any(axis=0).mean()) if len(cand) else 0.0,
                         component=int(comps["component_id"].values[cand[b]]) if b >= 0 else -1,
                         d_rt_s=float((c_rt[cand[b]] - rt0) * 60.0) if b >= 0 else np.nan,
                         intensity=float(g["intensity"].max())))
    return pd.DataFrame(rows, columns=["entry", "rt_consensus", "n_clean", "n_candidates", "n_components", "best", "any",
                                       "component", "d_rt_s", "intensity"])


def score_level(level: str, t: pd.DataFrame, output: Path, ppm: float, rt_tol_s: float, rt_win_s: float, min_ions: int):
    """t: this level's screened truth (screen) with the response_slope and responds columns"""
    from peak3d.io import Cloud
    stem = f"NEWSTD_{level}"
    t = t.copy()
    feats = pd.read_csv(output / "features" / f"{stem}.tsv", sep="\t")
    cache = output / ".cache" / f"{stem}.npz"
    t["raw_peak"], t["raw_apex"] = raw_peak(t, Cloud.load_npz(cache), ppm) if cache.exists() else (True, True)
    t["clean"] = t["mass_ok"] & t["rt_ok"] & t["raw_peak"] & t["raw_apex"] & t["responds"]
    idx, d_rt, found, inside = match_features(t["mz"].values, t["rt"].values, feats["mz"].values, feats["rt_apex"].values,
                                              feats["rt_min"].values, feats["rt_max"].values, ppm, rt_tol_s)
    found &= t["matched"].values
    t["found"], t["feature_id"], t["d_rt_s"] = found, np.where(found, idx, -1), d_rt
    t["feature_height"] = np.where(found, feats["height"].values[np.maximum(idx, 0)], np.nan) if len(feats) else np.nan
    reason = np.where(found, "found", np.where(inside, "inside another peak", "no candidate"))
    rej_file = output / "features" / f"{stem}.rejected.tsv"
    rej = pd.read_csv(rej_file, sep="\t", usecols=lambda c: c in ("mz", "rt", "rt_apex", "rt_min", "rt_max", "reason")) \
        if rej_file.exists() else pd.DataFrame()
    if len(rej):
        r_rt = rej["rt_apex"].values if "rt_apex" in rej else rej["rt"].values
        ri, _, rfound, _ = match_features(t["mz"].values, t["rt"].values, rej["mz"].values, r_rt, rej["rt_min"].values,
                                          rej["rt_max"].values, ppm, rt_tol_s)
        take = ~found & rfound & t["matched"].values
        reason = np.where(take, "rejected " + rej["reason"].astype(str).values[np.maximum(ri, 0)], reason)
    t["outcome"] = np.where(t["matched"].values, reason, "no data in truth")
    # crowding: other entries whose consensus RT lies within one median peak width
    pj = output / "features" / f"{stem}.params.json"
    fwhm = float(json.loads(pj.read_text()).get("fwhm_med", np.nan)) if pj.exists() else np.nan
    cons = t.drop_duplicates("entry").set_index("entry")["rt_consensus"]
    near = {e: int(np.sum(np.abs(cons.values - v) <= fwhm) - 1) if np.isfinite(v) and np.isfinite(fwhm) else -1
            for e, v in cons.items()}
    t["n_coeluting"] = t["entry"].map(near)
    comp_rows = pd.DataFrame()
    cdir = output / "gc" / "components"
    if (cdir / f"{stem}.npz").exists():
        comps = pd.read_csv(cdir / f"{stem}.tsv", sep="\t")
        with np.load(cdir / f"{stem}.npz") as z:
            comp_rows = score_compounds(t, comps, z["off"], z["mz"], ppm, rt_win_s, min_ions)
        comp_rows.insert(0, "level", level)
    info = dict(level=level, ng_ml=level_value(level), n_features=int(len(feats)), fwhm_s=round(60 * fwhm, 3))
    return t, comp_rows, info


def _recall(t):
    return float(t["found"].mean()) if len(t) else float("nan")


def summarise(rows: pd.DataFrame, comp: pd.DataFrame, infos: list) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_level = []
    for info in infos:
        t = rows[rows["level"] == info["level"]]
        m = t[t["matched"]]
        c = t[t["clean"]].drop_duplicates(["mz", "rt"])
        located = m[m["mass_ok"] & m["rt_ok"]]
        d = dict(info, rows=len(t), recorded=len(m), mass_ok=int(m["mass_ok"].sum()), rt_ok=int(m["rt_ok"].sum()),
                 not_a_peak=int((~located["raw_peak"]).sum()), not_at_apex=int((located["raw_peak"] & ~located["raw_apex"]).sum()),
                 no_response=int((~located["responds"]).sum()), clean=int(t["clean"].sum()), clean_points=len(c), recall_recorded=_recall(m), recall_clean=_recall(c),
                 recall_not_clean=_recall(m[~m["clean"]]))
        if len(comp):
            k = comp[comp["level"] == info["level"]]
            d.update(compounds=len(k), compounds_with_component=int((k["best"] > 0).sum()),
                     median_best=float(k["best"].median()) if len(k) else np.nan,
                     share_best_ge_0_8=float((k["best"] >= 0.8).mean()) if len(k) else np.nan,
                     median_any=float(k["any"].median()) if len(k) else np.nan)
        by_level.append(d)
    clean = rows[rows["clean"]].drop_duplicates(["level", "mz", "rt"])
    misses = clean[~clean["found"]].groupby(["level", "outcome"]).size().rename("n").reset_index()
    clean = clean.assign(decade=np.floor(np.log10(clean["intensity"].clip(lower=1))).astype(int),
                         crowd=np.minimum(clean["n_coeluting"], 2))
    strata = pd.concat([
        clean.groupby("decade")["found"].agg(["size", "mean"]).reset_index().assign(by="intensity decade (10^x)")
             .rename(columns={"decade": "class"}),
        clean.groupby("crowd")["found"].agg(["size", "mean"]).reset_index().assign(by="other standards within one peak width")
             .rename(columns={"crowd": "class"})], ignore_index=True)[["by", "class", "size", "mean"]]
    return pd.DataFrame(by_level), misses, strata.rename(columns={"size": "n", "mean": "recall"})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True, help="peak3d output folder of the MRH files")
    ap.add_argument("--truth", type=Path, default=Path("data/raw/MRH_GC_ORBITRAP"))
    ap.add_argument("--out", type=Path, required=True, help="folder for the score tables")
    ap.add_argument("--ppm", type=float, default=10.0)
    ap.add_argument("--rt-tol", type=float, default=2.0, help="feature apex vs truth point, seconds")
    ap.add_argument("--rt-win", type=float, default=1.5, help="truth ion vs its compound's consensus RT, seconds")
    ap.add_argument("--min-ions", type=int, default=3)
    a = ap.parse_args(argv)
    levels = [lv for lv in LEVELS if (a.output / "features" / f"NEWSTD_{lv}.tsv").exists()
              and (a.truth / f"groundtruth_corrected_{lv}.csv").exists()]
    if not levels:
        sys.exit(f"no NEWSTD_<level> feature tables in {a.output / 'features'} with a truth file in {a.truth}")
    truth = pd.concat([screen(load_truth(a.truth / f"groundtruth_corrected_{lv}.csv"), a.ppm, a.rt_win, a.min_ions)
                       .assign(level=lv) for lv in levels], ignore_index=True)
    truth.insert(0, "level", truth.pop("level"))
    truth["response_slope"] = dose_response(truth)
    # the five internal standards (Category "IS") are at one amount in every level: flat by design
    truth["responds"] = ~(truth["response_slope"].values < RESPONSE_MIN) | (truth["category"].astype(str).values == "IS")
    parts, comps, infos = [], [], []
    for lv in levels:
        t, c, info = score_level(lv, truth[truth["level"] == lv], a.output, a.ppm, a.rt_tol, a.rt_win, a.min_ions)
        parts.append(t); comps.append(c); infos.append(info)
    rows = pd.concat(parts, ignore_index=True)
    comp = pd.concat(comps, ignore_index=True) if any(len(c) for c in comps) else pd.DataFrame()
    by_level, misses, strata = summarise(rows, comp, infos)
    a.out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(a.out / "truth_rows.tsv", sep="\t", index=False, float_format="%.6g")
    by_level.to_csv(a.out / "by_level.tsv", sep="\t", index=False, float_format="%.4g")
    misses.to_csv(a.out / "misses_clean.tsv", sep="\t", index=False)
    strata.to_csv(a.out / "strata_clean.tsv", sep="\t", index=False, float_format="%.4g")
    if len(comp):
        comp.to_csv(a.out / "compounds.tsv", sep="\t", index=False, float_format="%.4g")
    (a.out / "params.json").write_text(json.dumps(dict(output=str(a.output), truth=str(a.truth), ppm=a.ppm, rt_tol_s=a.rt_tol,
                                                       rt_win_s=a.rt_win, min_ions=a.min_ions, levels=levels), indent=1))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(by_level.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(strata.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(misses.pivot_table(index="outcome", columns="level", values="n", fill_value=0).reindex(columns=levels).to_string())


if __name__ == "__main__":
    main()
