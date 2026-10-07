# cross-tool feature comparison, part 1: for a run, every tool's aligned features are matched against every other
# tool (10 ppm / 0.1 min). Per feature: how many tools report it, which, its intensity, whether it is an M+1 / M+2
# isotopologue of another feature in the same table, a truth hit where a truth list exists, a 13C partner where a 13C
# run exists (reference = union of every tool's 13C features; chance level from decoy spacings), and, on runs of <= 3
# files, a raw-data replicate check: a peak at the position in the OTHER injections (local maximum of the 5 ppm EIC,
# >= 3 x the picker's cell noise, >= 4 contiguous scans, relative prominence >= 0.2 within 2 FWHM).
# Two position lists are built across tools and de-duplicated: "consensus missed" (reported by >= 2 of the other tools,
# no peak3d feature within tolerance) and "any missed" (>= 1 other tool), plus peak3d's own single-tool features.
# usage: crosstool_overlap.py RUN [RUN ...]   (reads mzML for runs of <= 3 files -> srun)
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/diag"))
import bench.score as S  # noqa: E402
from bench.runs import RUNS, TOOLS  # noqa: E402
from bench.score import hits, load  # noqa: E402
from fix_eval import eval_truth  # noqa: E402
from miss_common import noise_lookup, to_native  # noqa: E402
from peak3d.estimate import estimate  # noqa: E402
from peak3d.io import load_cloud  # noqa: E402
from peak3d.pick import pick_cloud  # noqa: E402

PPM, RT_TOL = 10.0, 0.1
C13 = S.C13
OUT = ROOT / os.environ.get("CROSSTOOL_OUT", "results/diag_crosstool")
MIN_CONTIG = int(os.environ.get("CROSSTOOL_MIN_CONTIG", "4"))


def matcher_bool(feats, q_mz, q_rt):
    return hits(feats, q_mz, q_rt, PPM, RT_TOL)


def dedupe(df):
    """greedy: strongest first, drop positions within tolerance of a kept one"""
    df = df.sort_values("intensity", ascending=False).reset_index(drop=True)
    keep = np.zeros(len(df), bool)
    kept_mz, kept_rt = [], []
    for i, (m, r) in enumerate(zip(df["mz"].values, df["rt"].values)):
        if kept_mz:
            km = np.array(kept_mz)
            kr = np.array(kept_rt)
            if np.any((np.abs(km - m) / m * 1e6 <= PPM) & (np.abs(kr - r) <= RT_TOL)):
                continue
        keep[i] = True
        kept_mz.append(m)
        kept_rt.append(r)
    return df[keep].reset_index(drop=True)


def raw_peak(c, mz, rt_nat, ppm, fwhm_min, floor, cell, tol_min=RT_TOL):
    """(ok, apex, snr_cell, contig, prom_rel): a peak at the position in this file"""
    half = 0.5
    s0 = int(max(0, np.searchsorted(c.rt, rt_nat - half)))
    s1 = int(min(c.n_scans, np.searchsorted(c.rt, rt_nat + half)))
    e = c.eic(mz, ppm, s0, s1).astype(np.float64)
    r = c.rt[s0:s1]
    if len(e) < 3:
        return False, 0.0, 0.0, 0, 0.0
    win = np.abs(r - rt_nat) <= tol_min
    if not win.any() or e[win].max() <= 0:
        return False, 0.0, 0.0, 0, 0.0
    sm = np.convolve(e, [0.25, 0.5, 0.25], mode="same")
    idx = np.flatnonzero(win)
    lm = [i for i in idx if 0 < i < len(sm) - 1 and sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1] and e[i] > 0]
    if not lm:
        return False, float(e[idx].max()), float(e[idx].max() / max(cell, 1e-9)), 0, 0.0
    i = max(lm, key=lambda k: e[k])
    h = e[i]
    lo = i
    while lo - 1 >= 0 and e[lo - 1] > 0:
        lo -= 1
    hi = i
    while hi + 1 < len(e) and e[hi + 1] > 0:
        hi += 1
    contig = hi - lo + 1
    side = max(2, int(round(2 * fwhm_min / max(np.median(np.diff(c.rt)), 1e-6))))
    left = e[max(0, i - side):i]
    right = e[i + 1:i + 1 + side]
    prom = h - max(left.min() if len(left) else 0.0, right.min() if len(right) else 0.0)
    prom_rel = prom / h
    snr_cell = h / max(cell, floor, 1e-9)
    ok = snr_cell >= 3 and contig >= MIN_CONTIG and prom_rel >= 0.2
    return bool(ok), float(h), float(snr_cell), int(contig), float(prom_rel)


def main(run):
    files = RUNS[run]["files"]
    stems = [Path(f).stem for f in files]
    N = len(files)
    out = OUT / run
    out.mkdir(parents=True, exist_ok=True)
    tools = [t for t in TOOLS if load(t, run) is not None]
    F = {}
    for t in tools:
        feats, samples = load(t, run)
        f = feats.copy()
        f["intensity"] = samples.max(axis=1).values if len(samples.columns) else np.nan
        F[t] = f
    if run == "IDSL003":
        lab = pd.read_csv(S.IDSL_TRUTH)
        lab = lab[lab["Manual Curation"] == "TP"]
        truth = pd.DataFrame({"mz": lab["m/z"].values, "rt": lab["RT(min)"].values})
    else:
        truth = eval_truth(run)
        if truth is not None and "tp" in truth:
            truth = None
    # 13C reference: union of every tool's 13C features
    ref13 = None
    if run.endswith("12C"):
        run13 = run.replace("12C", "13C")
        parts = [load(t, run13)[0] for t in tools if load(t, run13) is not None]
        ref13 = pd.concat(parts, ignore_index=True)
    # per-tool annotation
    for t in tools:
        f = F[t]
        for u in tools:
            f[f"in_{u}"] = True if u == t else matcher_bool(F[u][["mz", "rt"]], f["mz"].values, f["rt"].values)
        f["n_tools"] = f[[f"in_{u}" for u in tools]].sum(axis=1).astype(int)
        f["n_others_nonp3d"] = f[[f"in_{u}" for u in tools if u not in (t, "peak3d")]].sum(axis=1).astype(int)
        # isotopologue: a feature C13/z below at the same RT in the same table (z = 1, 2)
        iso = np.zeros(len(f), bool)
        for z in (1, 2):
            iso |= matcher_bool(f[["mz", "rt"]], f["mz"].values - C13 / z, f["rt"].values)
        f["is_isotopologue"] = iso
        f["truth"] = matcher_bool(truth, f["mz"].values, f["rt"].values) if truth is not None else np.nan
        if ref13 is not None:
            f["cred"] = S.credentialed(f[["mz", "rt"]], ref13, C13)
            f["cred_decoy"] = np.mean([S.credentialed(f[["mz", "rt"]], ref13, C13 + d).astype(float) for d in (-0.035, -0.02, 0.02, 0.035)], axis=0)
    # cross-tool position lists
    others = [t for t in tools if t != "peak3d"]
    rows = []
    for t in others:
        f = F[t]
        m = ~f["in_peak3d"]
        rows.append(f[m].assign(tool=t))
    missed = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    missed_any = dedupe(missed) if len(missed) else missed
    missed_cons = dedupe(missed[missed["n_others_nonp3d"] >= 1]) if len(missed) else missed
    p3d_only = F["peak3d"][F["peak3d"]["n_tools"] == 1].copy() if "peak3d" in F else pd.DataFrame()
    cons_ref = F["peak3d"][F["peak3d"]["n_tools"] >= 3].copy() if "peak3d" in F else pd.DataFrame()
    # replicate check on runs of <= 3 files
    if N <= 3 and "peak3d" in F:
        d = ROOT / "results/peak3d" / run / "peak3d_out"
        clouds, P, look, W = [], [], [], []
        for f, s in zip(files, stems):
            c = load_cloud(f)
            p = estimate(c)
            b = pick_cloud(c, p)
            clouds.append(c)
            P.append(p)
            look.append(noise_lookup(c, p, b))
            W.append(pd.read_csv(d / "rt_correction" / f"{s}.tsv", sep="\t"))
            print(run, s, "loaded", flush=True)
        ppm_eic = 10.0 if run == "IDSL003" else 5.0

        def rep_cols(df, label):
            n_ok = np.zeros(len(df), int)
            apex = np.zeros(len(df))
            snr = np.zeros(len(df))
            prom = np.zeros(len(df))
            for i, (m, r) in enumerate(zip(df["mz"].values, df["rt"].values)):
                oks, hs, ss, ps = [], [], [], []
                for k in range(N):
                    rt_nat = to_native(W[k], r)
                    ok, h, sc, ct, pr = raw_peak(clouds[k], m, rt_nat, ppm_eic, P[k].fwhm_med, P[k].floor, look[k](m, rt_nat))
                    oks.append(ok)
                    hs.append(h)
                    ss.append(sc)
                    ps.append(pr)
                n_ok[i] = sum(oks)
                apex[i] = max(hs)
                snr[i] = max(ss)
                prom[i] = float(np.median(ps))
                if i % 2000 == 0:
                    print(run, label, i, "of", len(df), flush=True)
            df = df.copy()
            df["rep_files"] = n_ok
            df["raw_apex"] = apex
            df["raw_snr_cell"] = snr
            df["raw_prom_rel"] = prom
            return df
        p3d_only = rep_cols(p3d_only, "peak3d-only")
        missed_cons = rep_cols(missed_cons, "consensus-missed")
        rng = np.random.default_rng(0)
        cons_ref = rep_cols(cons_ref.iloc[rng.choice(len(cons_ref), size=min(2000, len(cons_ref)), replace=False)], "consensus-ref")
        missed_any = rep_cols(missed_any.iloc[rng.choice(len(missed_any), size=min(3000, len(missed_any)), replace=False)], "any-missed sample")
    for t in tools:
        F[t].to_csv(out / f"features_{t}.tsv", sep="\t", index=False, float_format="%.6g")
    p3d_only.to_csv(out / "peak3d_only.tsv", sep="\t", index=False, float_format="%.6g")
    missed_cons.to_csv(out / "missed_consensus.tsv", sep="\t", index=False, float_format="%.6g")
    missed_any.to_csv(out / "missed_any.tsv", sep="\t", index=False, float_format="%.6g")
    cons_ref.to_csv(out / "consensus_ref.tsv", sep="\t", index=False, float_format="%.6g")
    # summary
    summ = {"run": run, "n_files": N, "tools": tools, "counts": {t: int(len(F[t])) for t in tools}}
    summ["pairwise_recall"] = {t: {u: round(float(F[t][f"in_{u}"].mean()), 3) for u in tools} for t in tools}
    summ["by_n_tools"] = {t: F[t]["n_tools"].value_counts().sort_index().to_dict() for t in tools}
    summ["n_missed_any"] = int(len(missed_any)) if N > 3 else int(len(dedupe(missed))) if len(missed) else 0
    summ["n_missed_consensus"] = int(len(missed_cons))
    summ["n_peak3d_only"] = int(len(p3d_only))

    def ev(df):
        r = {"n": int(len(df))}
        if len(df) == 0:
            return r
        r["median_intensity"] = float(np.nanmedian(df["intensity"]))
        r["isotopologue"] = round(float(df["is_isotopologue"].mean()), 3)
        if truth is not None:
            r["truth"] = round(float(df["truth"].mean()), 3)
            r["n_truth"] = int(df["truth"].sum())
        if ref13 is not None:
            r["cred_net"] = round(float(df["cred"].mean() - df["cred_decoy"].mean()), 3)
        if "rep_files" in df:
            r["replicated_2of3"] = round(float((df["rep_files"] >= 2).mean()), 3)
            r["replicated_any"] = round(float((df["rep_files"] >= 1).mean()), 3)
            r["raw_apex_median"] = float(np.median(df["raw_apex"]))
        return r
    summ["evidence"] = {"peak3d_only": ev(p3d_only), "missed_consensus": ev(missed_cons), "missed_any": ev(missed_any),
                        "peak3d_consensus_ref": ev(cons_ref)}
    for t in tools:
        f = F[t]
        summ["evidence"][f"{t}_single_tool"] = ev(f[f["n_tools"] == 1])
        summ["evidence"][f"{t}_all"] = ev(f)
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    for r in sys.argv[1:]:
        main(r)
