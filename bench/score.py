#!/usr/bin/env python3
"""Score tool outputs in results/<tool>/<run>/ against each run's ground truth.

Writes results/scores/*.tsv and prints a summary. Every feature a tool reports is
scored (isotopes included): both tools report isotopic peaks as features, and the
truth sets label real chromatographic peaks whether or not they are monoisotopic.

usage: python bench/score.py [SCORER ...]   (run with envs/masscube/bin/python: needs pandas/scipy;
       SCORER = idsl, hzv029, yeast_neg, credentialing, credtruth, ratio, spike, runtime; default all)
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.runs import CRED_PAIRS, LOCAL_TOOLS, RUNS, TOOLS, local_tools  # noqa: E402

RES = ROOT / "results"
OUT = RES / "scores"
C13 = 1.0033548  # 13C - 12C mass difference


# ---------------------------------------------------------------- loading ----
FALLBACK = {}  # (tool, run) -> note, for runs scored from a partial output


def load(tool, run):
    """Return (features DataFrame[mz, rt] with rt in minutes, sample intensity matrix) or None.

    If MassCube crashed after alignment (no DONE), its pre-annotation aligned table is
    used instead and the run is recorded in FALLBACK."""
    d = RES / tool / run
    stems = [Path(f).stem for f in RUNS[run]["files"]]
    if tool == "asari" or tool.startswith("asari_"):  # asari_* = tuned variants (scripts/32_run_asari_tuned.sbatch)
        if not (d / "DONE").exists():
            return None
        t = pd.read_csv(next(d.glob("asari_out*/export/full_Feature_table.tsv")), sep="\t")
        feats = pd.DataFrame({"mz": t["mz"], "rt": t["rtime"] / 60.0})
    elif tool == "idslipa":
        if not (d / "DONE").exists():
            return None
        o = d / "idslipa_out"
        g = o / "peak_alignment/peak_height_gapfilled.csv"
        if g.exists():  # aligned + gap-filled, like the other tools' tables
            t = pd.read_csv(g)
            feats = pd.DataFrame({"mz": t["mz"], "rt": t["RT"]})
        else:  # single-file run: the file's own peaklist
            t = pd.read_csv(next(o.glob("peaklists/peaklist_*.csv")))
            feats = pd.DataFrame({"mz": t["m/z 12C"], "rt": t["retentionTimeApex"]})
            t[stems[0]] = t["PeakHeight"]
    elif tool == "peak3d":
        if not (d / "DONE").exists():
            return None
        t = pd.read_csv(d / "peak3d_out/aligned_feature_table.tsv", sep="\t")
        feats = pd.DataFrame({"mz": t["mz"], "rt": t["rt"]})
    elif tool == "masscube":
        f = d / "aligned_feature_table.txt"
        if not (d / "DONE").exists():
            f = d / "project_files/aligned_feature_table_before_annotation.txt"
            if not f.exists():
                return None
            FALLBACK[(tool, run)] = "crashed after alignment; pre-annotation table used"
        t = pd.read_csv(f, sep="\t", low_memory=False)
        feats = pd.DataFrame({"mz": t["m/z"], "rt": t["RT"]})
    elif tool in LOCAL_TOOLS:
        x = local_tools.load(tool, d)
        if x is None:
            return None
        feats, t = x
    else:
        raise ValueError(f"unknown tool {tool!r}")
    cols = [c for c in t.columns if re.sub(r"\.mzML$", "", str(c)) in stems]
    samples = t[cols].copy()
    samples.columns = [re.sub(r"\.mzML$", "", str(c)) for c in cols]
    return feats.reset_index(drop=True), samples.reset_index(drop=True).astype(float)


def matcher(feats):
    """Return f(mz, rt, ppm, rt_tol) -> bool, using a sorted m/z index."""
    order = np.argsort(feats["mz"].values)
    mz = feats["mz"].values[order]
    rt = feats["rt"].values[order]

    def hit(q_mz, q_rt, ppm, rt_tol):
        tol = q_mz * ppm * 1e-6
        lo, hi = np.searchsorted(mz, q_mz - tol), np.searchsorted(mz, q_mz + tol, side="right")
        return bool(np.any(np.abs(rt[lo:hi] - q_rt) <= rt_tol))
    return hit


def hits(feats, q_mz, q_rt, ppm, rt_tol):
    h = matcher(feats)
    return np.array([h(m, r, ppm, rt_tol) for m, r in zip(q_mz, q_rt)])


# ---------------------------------------------------------------- truths -----
GRID_PPM, GRID_RT = (5, 10, 20), (0.05, 0.1, 0.2)


# cleaned IDSL.IPA labels (2026-10-05): the 17,004 of the 20,000 curated m/z-RT pairs whose TP/TN call a blind visual
# EIC check confirmed; flagged, questionable and self-conflicting labels removed (bench/diag/idsl_clean_truth.py,
# data/truth/IDSL003/params.json). Scores on the original 20,000: results/scores/idsl_original_labels.tsv
IDSL_TRUTH = ROOT / "data/truth/IDSL003/labels_clean.csv"


def score_idsl():
    lab = pd.read_csv(IDSL_TRUTH)
    is_tp = (lab["Manual Curation"] == "TP").values
    rows = []

    def add(name, detected, ppm="", rt=""):
        tp, fn = int((detected & is_tp).sum()), int((~detected & is_tp).sum())
        fp, tn = int((detected & ~is_tp).sum()), int((~detected & ~is_tp).sum())
        rows.append(dict(tool=name, ppm=ppm, rt_tol_min=rt, TP=tp, FN=fn, FP=fp, TN=tn,
                         recall=tp / (tp + fn), FPR=fp / (fp + tn),
                         precision=tp / (tp + fp) if tp + fp else np.nan,
                         F1=2 * tp / (2 * tp + fp + fn)))

    # reference tools, as reported by the dataset authors
    for c in ["IDSL.IPA", "XCMS", "MZMINE", "MSDIAL"]:
        add(f"{c} (published)", lab[c].isin(["TP", "FP"]).values)
    # TUNED asari rows (lowered peak-height cutoff; IDSL003 only), next to the default-settings tools
    for tool in TOOLS + ["asari_autoheight", "asari_h1000"]:
        x = load(tool, "IDSL003")
        if x is None:
            continue
        for ppm in GRID_PPM:
            for rt in GRID_RT:
                add(tool, hits(x[0], lab["m/z"].values, lab["RT(min)"].values, ppm, rt), ppm, rt)
    return pd.DataFrame(rows)


def score_hzv029():
    p = next((ROOT / "data/raw/ASARI_DATA/x").glob("*/data/hzv029_manual_certified.txt"))
    tr = pd.read_csv(p, sep="\t")
    rows = []
    for tool in TOOLS:
        x = load(tool, "HZV029_cert")
        if x is None:
            continue
        for ppm in GRID_PPM:
            for rt in GRID_RT:
                h = hits(x[0], tr["moverz"].values, tr["RT_minutes"].values, ppm, rt)
                rows.append(dict(tool=tool, ppm=ppm, rt_tol_min=rt, n_truth=len(tr),
                                 found=int(h.sum()), recall=h.mean(), n_features=len(x[0])))
    return pd.DataFrame(rows)


def score_yeastneg():
    """Recall on the 314 NetID manually curated peaks (Confidence TRUE), yeast neg mode.
    Also split by NetID class: metabolites vs 'artifacts' (isotopes/adducts/fragments)."""
    tr = pd.read_csv(ROOT / "data/raw/NETID_YEAST/paper_SI/manual_curation_yeast_neg.tsv", sep="\t")
    tr = tr[tr["Confidence"].astype(str).str.upper() == "TRUE"].reset_index(drop=True)
    is_met = tr["class"].isin(["Metabolite", "Putative Metabolite"]).values
    rows = []
    for tool in TOOLS:
        x = load(tool, "YEAST_NEG")
        if x is None:
            continue
        for ppm in GRID_PPM:
            for rt in GRID_RT:
                h = hits(x[0], tr["medMz"].values, tr["medRt"].values, ppm, rt)
                rows.append(dict(tool=tool, ppm=ppm, rt_tol_min=rt, n_truth=len(tr),
                                 found=int(h.sum()), recall=h.mean(),
                                 recall_metabolites=h[is_met].mean(), recall_artifacts=h[~is_met].mean(),
                                 n_features=len(x[0])))
    return pd.DataFrame(rows)


def credentialed(f12, f13, step, ppm=5, rt_tol=0.1, n_min=1, n_max=60):
    """12C features with a partner in the 13C run at m/z + n*step (n carbons) and same RT.
    With step=C13 this counts real labeled partners; with a non-physical step it
    estimates how many 'partners' turn up by chance."""
    h = matcher(f13)
    out = np.zeros(len(f12), bool)
    for i, (m, r) in enumerate(zip(f12["mz"].values, f12["rt"].values)):
        nmax = min(n_max, int(m // 12))  # no more carbons than the mass allows
        out[i] = any(h(m + n * step, r, ppm, rt_tol) for n in range(n_min, nmax + 1))
    return out


def score_cred():
    rows = []
    for name, (r12, r13) in CRED_PAIRS.items():
        for tool in TOOLS:
            a, b = load(tool, r12), load(tool, r13)
            if a is None or b is None:
                continue
            f12, f13 = a[0], b[0]
            tgt = credentialed(f12, f13, C13).sum()
            # four decoy spacings, 20-35 mDa per carbon off the 13C spacing
            dec = np.mean([credentialed(f12, f13, C13 + d).sum() for d in (-0.035, -0.02, 0.02, 0.035)])
            rows.append(dict(dataset=name, tool=tool, n_features_12C=len(f12),
                             n_features_13C=len(f13), credentialed=int(tgt),
                             decoy_expected=round(dec, 1), credentialed_net=round(tgt - dec, 1),
                             frac_credentialed_net=(tgt - dec) / len(f12)))
    return pd.DataFrame(rows)


def score_credtruth(screened=False):
    """Tool-independent credentialed truth (data/truth/<DS>/credentialed.tsv, bench/credtruth.py):
    12C features with a co-eluting, same-shape U-13C partner at m/z + n x 1.00335 in >= 2/3 runs
    on each side. Per tool: recall of the truth features in the 12C run (all, tier A/B, primary
    features only), recall of their 13C partners in the 13C run, and the share of the tool's 12C
    features that are credentialed truth features (a lower bound on its share of real labelled
    metabolites: the truth set holds only what passed every test).
    screened=True scores the entries that pass the seed prominence screen (credentialed_prominence.tsv,
    column keep: the seed apex rises at least 20 % above the deeper side minimum within 3 FWHM, median
    over the 12C injections; bench/diag/credtruth_prominence.py, 2026-10-06). The builder seeds every
    local maximum without a prominence test, so a bump on the tail of a bigger peak of the same ion was
    credentialed whenever its 13C partner showed the same tail (107 of 3,992 YEAST, 3 of 625 SZ22)."""
    rows = []
    for name, (r12, r13) in CRED_PAIRS.items():
        if screened:
            t = pd.read_csv(ROOT / f"data/truth/{name}/credentialed_prominence.tsv", sep="\t")
            t = t[t["keep"].astype(bool)].reset_index(drop=True)
        else:
            t = pd.read_csv(ROOT / f"data/truth/{name}/credentialed.tsv", sep="\t")
        partner = t["mz"].values + t["n"].values * C13
        tierA = (t["tier"] == "A").values
        prim = (t["feature_class"] == "primary").values
        for tool in TOOLS:
            a, b = load(tool, r12), load(tool, r13)
            if a is None or b is None:
                continue
            f12, f13 = a[0], b[0]
            for ppm in GRID_PPM:
                for rt_tol in GRID_RT:
                    h12 = hits(f12, t["mz"].values, t["rt"].values, ppm, rt_tol)
                    h13 = hits(f13, partner, t["rt"].values, ppm, rt_tol)
                    own = hits(pd.DataFrame({"mz": t["mz"], "rt": t["rt"]}), f12["mz"].values, f12["rt"].values,
                               ppm, rt_tol)
                    rows.append(dict(dataset=name, tool=tool, ppm=ppm, rt_tol_min=rt_tol, n_truth=len(t),
                                     n_features_12C=len(f12), n_features_13C=len(f13),
                                     recall_12C=h12.mean(), recall_A=h12[tierA].mean(), recall_B=h12[~tierA].mean(),
                                     recall_primary=h12[prim].mean(), recall_13C_partner=h13.mean(),
                                     recall_both=(h12 & h13).mean(), truth_share_of_features=own.mean()))
    return pd.DataFrame(rows)


def score_ratio():
    des = pd.read_csv(RES / "inputs/BM21_design.tsv", sep="\t")
    des = des[des["plasma_fraction"].notna()]
    rows = []
    for run, an in [("BM21_HILIC", "HILIC"), ("BM21_RP", "RP")]:
        d = des[des["analysis"] == an]
        frac = d.set_index("sample")["plasma_fraction"]
        for tool in TOOLS:
            x = load(tool, run)
            if x is None:
                continue
            feats, S = x
            cols = [c for c in S.columns if c in frac.index]
            M = S[cols].fillna(0).values
            p = frac[cols].values
            present = (M > 0).mean(axis=1)
            rho = np.array([spearmanr(v, p).statistic if v.std() > 0 else np.nan for v in M])
            good = np.abs(rho) >= 0.9
            for thr in (0.0, 0.5):
                keep = present > thr if thr else np.ones(len(M), bool)
                rows.append(dict(run=run, tool=tool, n_samples=len(cols),
                                 min_presence=f">{thr:.0%}" if thr else "all",
                                 n_features=int(keep.sum()),
                                 n_ratio_consistent=int((good & keep).sum()),
                                 frac_ratio_consistent=(good & keep).sum() / max(keep.sum(), 1)))
    return pd.DataFrame(rows)


def score_spike():
    """MTBLS733 standards: detection recall on the 836 confirmed true features, and
    SB/SA fold-change accuracy (expected ratio a\\b = SB:SA, per the study protocol)."""
    t = pd.read_csv(ROOT / "data/raw/LI2018_QE/m_MTBLS733_mass_spectrometry_v2_maf.tsv", sep="\t")
    ab = t["Compound concentration ratio"].str.split("\\\\", expand=True).astype(float)
    t["exp_log2"] = np.log2(ab[0] / ab[1])
    t["group"] = np.where(t["exp_log2"] == 0, "constant", "differential")
    rows = []
    for tool in TOOLS:
        x = load(tool, "LI2018")
        if x is None:
            continue
        feats, S = x
        sa = [c for c in S.columns if c.startswith("SA")]
        sb = [c for c in S.columns if c.startswith("SB")]
        order = np.argsort(feats["mz"].values)
        mz, rt = feats["mz"].values[order], feats["rt"].values[order]
        for ppm in GRID_PPM:
            for rt_tol in GRID_RT:
                obs = np.full(len(t), np.nan)
                found = np.zeros(len(t), bool)
                for i, (m, r) in enumerate(zip(t["mass_to_charge"], t["retention_time"])):
                    lo = np.searchsorted(mz, m - m * ppm * 1e-6)
                    hi = np.searchsorted(mz, m + m * ppm * 1e-6, side="right")
                    k = lo + np.flatnonzero(np.abs(rt[lo:hi] - r) <= rt_tol)
                    if not len(k):
                        continue
                    found[i] = True
                    j = order[k[np.argmin(np.abs(rt[k] - r))]]  # closest in RT
                    a, b = S.loc[j, sa].mean(), S.loc[j, sb].mean()
                    if a > 0 and b > 0:
                        obs[i] = np.log2(b / a)
                err = obs - t["exp_log2"].values
                row = dict(tool=tool, ppm=ppm, rt_tol_min=rt_tol, n_truth=len(t),
                           recall=found.mean(),
                           recall_constant=found[t["group"] == "constant"].mean(),
                           recall_differential=found[t["group"] == "differential"].mean(),
                           n_features=len(feats))
                d = (t["group"] == "differential").values & ~np.isnan(obs)
                c = (t["group"] == "constant").values & ~np.isnan(obs)
                row.update(
                    diff_quantified=int(d.sum()),
                    diff_median_abs_log2_err=np.median(np.abs(err[d])) if d.any() else np.nan,
                    diff_within_2fold=(np.abs(err[d]) <= 1).mean() if d.any() else np.nan,
                    diff_right_direction=(np.sign(obs[d]) == np.sign(t["exp_log2"].values[d])).mean()
                    if d.any() else np.nan,
                    const_quantified=int(c.sum()),
                    const_false_2fold=(np.abs(obs[c]) > 1).mean() if c.any() else np.nan)
                rows.append(row)
    return pd.DataFrame(rows)


def score_runtime():
    rows = []
    for tool in TOOLS:
        for run in RUNS:
            t = RES / tool / run / "time.txt"
            if not (RES / tool / run / "DONE").exists() or not t.exists():
                continue
            txt = t.read_text()
            g = lambda k: re.search(rf"{k}: (.+)", txt).group(1).strip()  # noqa: E731
            wall = g(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\)")
            secs = sum(float(x) * 60 ** i for i, x in enumerate(reversed(wall.split(":"))))
            x = load(tool, run)
            rows.append(dict(tool=tool, run=run, n_files=len(RUNS[run]["files"]),
                             wall_min=round(secs / 60, 2),
                             cpu_min=round((float(g("User time \\(seconds\\)")) +
                                            float(g("System time \\(seconds\\)"))) / 60, 2),
                             max_rss_gb=round(int(g(r"Maximum resident set size \(kbytes\)")) / 2**20, 2),
                             n_features=len(x[0]) if x else np.nan))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    only = set(sys.argv[1:])  # optional scorer names, e.g. `score.py idsl`; default all
    for name, fn in [("idsl", score_idsl), ("hzv029", score_hzv029), ("yeast_neg", score_yeastneg), ("credentialing", score_cred),
                     ("credtruth", score_credtruth),
                     ("credtruth_screened", lambda: score_credtruth(screened=True)),
                     ("ratio", score_ratio), ("spike", score_spike), ("runtime", score_runtime)]:
        if only and name not in only:
            continue
        df = fn()
        df.to_csv(OUT / f"{name}.tsv", sep="\t", index=False, quoting=csv.QUOTE_MINIMAL,
                  lineterminator="\n")
        print(f"\n== {name} ==")
        print(df.round(3).to_string(index=False) if len(df) else "(no finished runs)")
    if FALLBACK:
        print("\nscored from partial output:")
        for (t, r), note in FALLBACK.items():
            print(f"  {t} {r}: {note}")


if __name__ == "__main__":
    main()
