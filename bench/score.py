#!/usr/bin/env python3
"""Score tool outputs in results/<tool>/<run>/ against each run's ground truth.

Writes results/scores/*.tsv and prints a summary. Every feature a tool reports is
scored (isotopes included): both tools report isotopic peaks as features, and the
truth sets label real chromatographic peaks whether or not they are monoisotopic.

usage: python bench/score.py [SCORER ...]   (run with envs/masscube/bin/python: needs pandas/scipy;
       SCORER = idsl, hzv029, yeast_neg, credentialing, credtruth, ratio, spike, mc_synth, mc_eic, pr2464_spike,
       pr2464_yeast, pr2464_plasma, runtime; default all)
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
    elif tool == "peak3d" or tool.startswith("peak3d_"):  # peak3d_* = variants (scripts/42_peak3d_variant.sbatch)
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


# MassCube paper benchmark (Yu et al. 2025; truth built by bench/masscube_truth.py). As in the authors'
# notebooks, a signal is correct when the tool reports exactly n_peaks features in its window; the
# authors' own results (MassCube 1.0, MS-DIAL 4.9, MZmine 3.90, xcms 4.0) are rescored as "(published)"
MC_TRUTH = ROOT / "data/truth/MASSCUBE"


def window_counts(feats, q_mz, mz_tol, q_rt=None, rt_tol=None):
    """features with |dm/z| < mz_tol (scalar or per query, Da) and, if q_rt is given, |dRT| < rt_tol"""
    order = np.argsort(feats["mz"].values)
    mz, rt = feats["mz"].values[order], feats["rt"].values[order]
    lo = np.searchsorted(mz, q_mz - mz_tol, side="right")
    hi = np.searchsorted(mz, q_mz + mz_tol, side="left")
    if q_rt is None:
        return hi - lo
    return np.array([int(np.sum(np.abs(rt[a:b] - r) < rt_tol)) for a, b, r in zip(lo, hi, q_rt)])


def count_summary(cnt, k):
    """accuracy (count == k) plus how the wrong ones fail: missed (0), merged (0 < count < k), extra (> k)"""
    n = len(cnt) or np.nan  # no labels of this class (some files have no double peaks): NaN, not 0
    return dict(n=len(cnt), correct=int((cnt == k).sum()), accuracy=(cnt == k).sum() / n,
                missed=(cnt == 0).sum() / n, merged=((cnt > 0) & (cnt < k)).sum() / n, extra=(cnt > k).sum() / n)


def mc_rows(name, cnt, n_peaks, found=None, **key):
    """one row: single / double accuracy, their mean, and the failure fractions; with `found` (per signal: both
    labelled peaks found, other features nearby allowed; synth_found_both, or count >= 2 for the EIC labels)
    also double_found and its mean with single_acc"""
    row = dict(tool=name, **key)
    for k, cls in ((1, "single"), (2, "double")):
        s = count_summary(cnt[n_peaks == k], k)
        row.update({f"{cls}_n": s["n"], f"{cls}_acc": s["accuracy"], f"{cls}_missed": s["missed"],
                    f"{cls}_extra": s["extra"]} | ({"double_merged": s["merged"]} if k == 2 else {}))
    row["avg_acc"] = (row["single_acc"] + row["double_acc"]) / 2
    if found is not None:
        row["double_found"] = found[n_peaks == 2].mean() if (n_peaks == 2).any() else np.nan
        row["avg_found"] = (row["single_acc"] + row["double_found"]) / 2
    return row


def synth_found_both(feats, t, mz_tol=0.01):
    """per double signal: each labelled apex (truth apex1_rt, apex2_rt) has a feature of its own (|dm/z| < mz_tol,
    |dRT| < half the apex separation, so one feature cannot serve both). Other features at the m/z do not count
    against it; merged into one or missed -> False. Single signals (no apex2_rt) -> False."""
    a1, a2 = t["apex1_rt"].values, t["apex2_rt"].values
    half = (a2 - a1) / 2
    order = np.argsort(feats["mz"].values)
    mz, rt = feats["mz"].values[order], feats["rt"].values[order]
    q = t["mz"].values
    lo = np.searchsorted(mz, q - mz_tol, side="right")
    hi = np.searchsorted(mz, q + mz_tol, side="left")
    return np.array([bool(np.any(np.abs(rt[a:b] - p) < h) and np.any(np.abs(rt[a:b] - s) < h))
                     for a, b, p, s, h in zip(lo, hi, a1, a2, half)])


def score_mc_synth():
    """27,000 inserted Gaussian peaks; m/z window 0.01 Da over the whole run. The paper's accuracy is the
    mean over the six noise levels (per-level table: mc_synth_by_noise.tsv); *_pooled = correct / n.
    double_found: a double signal is a hit when both labelled peaks are found, whatever else is reported at its
    m/z (synth_found_both); tools run here only, the published outputs give counts, not positions."""
    t = pd.read_csv(MC_TRUTH / "synth_peaks.tsv", sep="\t")
    n_peaks, noise = t["n_peaks"].values, t["noise"].values
    counts = {f"{c[4:]} (published)": t[c].values for c in t.columns if c.startswith("pub_")}
    found = {}
    for tool in TOOLS:
        x = load(tool, "MC_SYNTH")
        if x is not None:
            counts[tool] = window_counts(x[0], t["mz"].values, 0.01)
            found[tool] = synth_found_both(x[0], t)
    rows, detail = [], []
    for name, cnt in counts.items():
        f = found.get(name)
        levels = [mc_rows(name, cnt[noise == z], n_peaks[noise == z], None if f is None else f[noise == z], noise=z)
                  for z in sorted(set(noise))]
        detail += levels
        row = mc_rows(name, cnt, n_peaks, f)
        for cls in ("single", "double"):
            row[f"{cls}_pooled"] = row[f"{cls}_acc"]
            row[f"{cls}_acc"] = np.mean([r[f"{cls}_acc"] for r in levels])
        row["avg_acc"] = (row["single_acc"] + row["double_acc"]) / 2
        if f is not None:
            row["double_found_pooled"] = row["double_found"]
            row["double_found"] = np.mean([r["double_found"] for r in levels])
            row["avg_found"] = (row["single_acc"] + row["double_found"]) / 2
        rows.append(row)
    pd.DataFrame(detail).to_csv(OUT / "mc_synth_by_noise.tsv", sep="\t", index=False)
    return pd.DataFrame(rows)


def score_mc_eic():
    """722 labeled EICs on eight files (one run each). rule 'paper': |dm/z| < 0.002 Da (single) / 0.005 Da
    (double), |dRT| < 0.2 min; rule 'wide': m/z window max(paper, 10 ppm), because the label m/z are
    MassCube's own feature m/z. Rows pool the eight files (per file: mc_eic_by_file.tsv).
    double_found: a double label is a hit when >= 2 features lie in its window (neighbouring real peaks are
    allowed; a label has one RT, so the two peaks cannot be told apart by position)."""
    lab = pd.read_csv(MC_TRUTH / "eic_labels.tsv", sep="\t")
    pub = pd.read_csv(MC_TRUTH / "published_features.tsv", sep="\t")
    n_peaks, files = lab["n_peaks"].values, lab["file"].values
    tol = np.where(n_peaks == 1, 0.002, 0.005)
    rules = {"paper": tol, "wide": np.maximum(tol, lab["mz"].values * 10e-6)}
    feats = {f"{t} (published)": dict(tuple(g.groupby("file"))) for t, g in pub.groupby("tool")}
    for tool in TOOLS:
        per = {f: x[0] for f in sorted(set(files)) if (x := load(tool, f"MC_{f}")) is not None}
        if per:
            feats[tool] = per
    rows, detail = [], []
    for name, per in feats.items():
        for rule, mz_tol in rules.items():
            cnt = np.full(len(lab), -1)
            for f, d in per.items():
                m = files == f
                cnt[m] = window_counts(d, lab["mz"].values[m], mz_tol[m], lab["rt"].values[m], 0.2)
            for f in sorted(per):
                m = files == f
                detail.append(mc_rows(name, cnt[m], n_peaks[m], cnt[m] >= 2, rule=rule, file=f,
                                      n_features=len(per[f])))
            done = cnt >= 0
            rows.append(mc_rows(name, cnt[done], n_peaks[done], cnt[done] >= 2, rule=rule, n_files=len(per)))
    pd.DataFrame(detail).to_csv(OUT / "mc_eic_by_file.tsv", sep="\t", index=False)
    return pd.DataFrame(rows)


# PR002464 (Broeckling/Du/Patti): one run per Orbitrap study (PR2464_<study>, all 224 MS1 injections). Truth tables in
# data/truth/PR002464/<study>/ from bench/spiketruth.py (spiked standards), bench/credtruth.py + biotruth.py (yeast
# 12C/13C) and biotruth.py (plasma). peak3d runs here as the GLR variant (narrow_rule=local_median,
# scripts/42_peak3d_variant.sbatch -> results/peak3d_GLR/); a default peak3d run would be scored too if present.
PR2464_TRUTH = ROOT / "data/truth/PR002464"
PR2464_EXTRA = ["peak3d_GLR"]
PR2464_RUNS = {r: r.removeprefix("PR2464_") for r in RUNS if r.startswith("PR2464_")}
# chance-match control: the truth list moved by these m/z offsets (Da) at the same RT; a tool reporting many features
# matches some of these by accident, at about the rate it matches the truth by accident
PR2464_DECOY_DA = (-0.05, 0.05)


def truthy(s):
    """bool column written by pandas (True/False), whether read back as bool or as text"""
    return s.astype(str).str.lower().eq("true")


def nearest(feats, q_mz, q_rt, ppm, rt_tol):
    """row index of the feature closest in RT among those within ppm and rt_tol of each query; -1 if none"""
    order = np.argsort(feats["mz"].values)
    mz, rt = feats["mz"].values[order], feats["rt"].values[order]
    out = np.full(len(q_mz), -1)
    for i, (m, r) in enumerate(zip(q_mz, q_rt)):
        lo, hi = np.searchsorted(mz, m - m * ppm * 1e-6), np.searchsorted(mz, m + m * ppm * 1e-6, side="right")
        k = lo + np.flatnonzero(np.abs(rt[lo:hi] - r) <= rt_tol)
        if len(k):
            out[i] = order[k[np.argmin(np.abs(rt[k] - r))]]
    return out


def vial_medians(S, files, vials):
    """median intensity over each vial's injections (missing = 0), features x vials, as the truth builders read
    heights (median of the 4 injections, robust to one lost injection)"""
    v = files.set_index("file")["vial"]
    cols = [c for c in S.columns if c in v.index]
    M = S[cols].fillna(0).values
    return np.stack([np.median(M[:, [i for i, c in enumerate(cols) if v[c] == x]], axis=1) for x in vials], axis=1)


def log_slopes(H, amt, min_levels=3):
    """log2-log2 slope of each row of H on amt over the levels with H > 0 (>= min_levels, else NaN), as the truth
    builders fit dilution series"""
    x = np.log2(np.asarray(amt, float))
    out = np.full(len(H), np.nan)
    for i, h in enumerate(H):
        ok = h > 0
        if ok.sum() >= min_levels:
            out[i] = np.polyfit(x[ok], np.log2(h[ok]), 1)[0]
    return out


def dilution_cols(slope, raw):
    """tool dilution slopes vs the truth builder's slopes from the raw data (peaks with both)"""
    ok = np.isfinite(slope) & np.isfinite(raw)
    d = np.abs(slope[ok] - raw[ok])
    return dict(dil_quantified=int(ok.sum()), dil_slope_median=np.median(slope[ok]) if ok.any() else np.nan,
                dil_median_abs_err_vs_raw=np.median(d) if ok.any() else np.nan,
                dil_within_025_of_raw=(d <= 0.25).mean() if ok.any() else np.nan)


def pr2464_study(run):
    """files.tsv of the spike truth (file, vial, amounts) and the amount of each vial in a dilution series"""
    files = pd.read_csv(PR2464_TRUTH / PR2464_RUNS[run] / "files.tsv", sep="\t")
    return files, files.groupby("vial").first()


def pr2464_grid(run, truth_fn, row_fn):
    """rows for every tool with a finished run x ppm x RT tolerance; truth_fn() -> truth table, row_fn(t, feats, S,
    idx, ppm, rt_tol) -> dict of scores, idx = matched feature row per truth row (-1 = not found)"""
    rows = []
    t = truth_fn()
    for tool in TOOLS + PR2464_EXTRA:
        x = load(tool, run)
        if x is None:
            continue
        feats, S = x
        for ppm in GRID_PPM:
            for rt_tol in GRID_RT:
                idx = nearest(feats, t["mz"].values, t["rt"].values, ppm, rt_tol)
                dec = [nearest(feats, t["mz"].values + d, t["rt"].values, ppm, rt_tol) >= 0 for d in PR2464_DECOY_DA]
                row = dict(study=PR2464_RUNS[run], tool=tool, ppm=ppm, rt_tol_min=rt_tol, n_truth=len(t),
                           n_features=len(feats), recall=(idx >= 0).mean(), decoy_match=np.mean(dec))
                rows.append(row | row_fn(t, feats, S, idx, ppm, rt_tol))
    return rows


def score_pr2464_spike():
    """Spiked-standard truth (spiked_truth.tsv): recall (all, tier A / B, tier A without ft_artifact?/broad flags,
    formula ions); A/B ratio of tier-A peaks whose weak side is above noise: tool log2(vial 18 / vial 19) vs the
    designed log2(a/b) (vial 18 = mix A only, 19 = mix B only); dilution slope over vials 9-12 vs the raw-data slope."""
    rows = []
    for run in PR2464_RUNS:
        files, vinfo = pr2464_study(run)
        C = [9, 10, 11, 12]
        amt = [vinfo.loc[v, "A"] + vinfo.loc[v, "B"] for v in C]

        def truth():
            t = pd.read_csv(PR2464_TRUTH / PR2464_RUNS[run] / "spiked_truth.tsv", sep="\t", keep_default_na=False,
                            na_values=[""])
            return t.assign(mz=t["mz_obs"].fillna(t["mz"]), rt=t["rt_apex"], flags=t["flags"].fillna(""),
                            weak_at_noise=truthy(t["weak_at_noise"]))

        def score(t, feats, S, idx, ppm, rt_tol):
            h = idx >= 0
            A = (t["tier"] == "A").values
            out = dict(n_tierA=int(A.sum()), recall_A=h[A].mean(), recall_B=h[~A].mean(),
                       recall_A_unflagged=h[A & (t["flags"] == "").values].mean(),
                       recall_formula_ion=h[(t["annotation_class"] == "formula_ion").values].mean())
            V = np.full((len(t), 6), np.nan)
            V[h] = vial_medians(S.iloc[idx[h]], files, [18, 19] + C)
            ok = A & ~t["weak_at_noise"].values
            q = ok & h & (V[:, 0] > 0) & (V[:, 1] > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                err = np.log2(V[:, 0] / V[:, 1]) - t["log2FC"].values
            exp = t["log2FC"].values[q]
            out.update(ratio_eligible=int(ok.sum()),
                       ratio_quantified=int(q.sum()),
                       ratio_median_abs_log2_err=np.median(np.abs(err[q])) if q.any() else np.nan,
                       ratio_within_2fold=(np.abs(err[q]) <= 1).mean() if q.any() else np.nan,
                       ratio_right_direction=(np.sign(err[q] + exp) == np.sign(exp)).mean() if q.any() else np.nan)
            sl = np.full(len(t), np.nan)
            sl[A & h] = log_slopes(V[A & h, 2:], amt)
            return out | dilution_cols(sl, t["c_dilution_slope"].values)
        rows += pr2464_grid(run, truth, score)
    return pd.DataFrame(rows)


def score_pr2464_yeast():
    """Credentialed yeast truth (yeast_truth.tsv; 12C vial 23 vs U-13C vial 24): recall of the 12C peaks (all, tier
    A / B, primary, confirmed in the 12C:13C 1:1 vial 25), of their 13C partners (m/z mz_13C, same RT) and of both;
    12C dilution slope over vials 5-8 (12C 500 / 250 / 100 / 50 uL) vs the raw-data slope."""
    rows = []
    for run in PR2464_RUNS:
        files, vinfo = pr2464_study(run)
        D = [5, 6, 7, 8]
        amt = [vinfo.loc[v, "Yeast 12C"] for v in D]

        def truth():
            t = pd.read_csv(PR2464_TRUTH / PR2464_RUNS[run] / "yeast_truth.tsv", sep="\t")
            return t.assign(confirmed=truthy(t["confirmed"]))

        def score(t, feats, S, idx, ppm, rt_tol):
            h = idx >= 0
            h13 = nearest(feats, t["mz_13C"].values, t["rt"].values, ppm, rt_tol) >= 0
            A = (t["tier"] == "A").values
            out = dict(n_tierA=int(A.sum()), recall_A=h[A].mean(), recall_B=h[~A].mean(),
                       recall_primary=h[(t["feature_class"] == "primary").values].mean(),
                       recall_confirmed=h[t["confirmed"].values].mean(),
                       recall_13C_partner=h13.mean(), recall_both=(h & h13).mean())
            sl = np.full(len(t), np.nan)
            sl[h] = log_slopes(vial_medians(S.iloc[idx[h]], files, D), amt)
            return out | dilution_cols(sl, t["dil_slope_12C"].values)
        rows += pr2464_grid(run, truth, score)
    return pd.DataFrame(rows)


def score_pr2464_plasma():
    """Plasma truth (plasma_truth.tsv): recall (all, tier A / B, tier A without flags); dilution slope over vials 1-4
    (1000 / 500 / 200 / 100 uL plasma) vs the raw-data slope; SRM 1950 / SRM 8231 ratio (vial 13 / vial 14) vs the
    raw-data ratio."""
    rows = []
    for run in PR2464_RUNS:
        files, vinfo = pr2464_study(run)
        D = [1, 2, 3, 4]
        amt = [vinfo.loc[v, "SRM 1950"] + vinfo.loc[v, "SRM 8231"] for v in D]

        def truth():
            t = pd.read_csv(PR2464_TRUTH / PR2464_RUNS[run] / "plasma_truth.tsv", sep="\t", keep_default_na=False,
                            na_values=[""])
            return t.assign(mz=t["mz_obs"], rt=t["rt_apex"], flags=t["flags"].fillna(""))

        def score(t, feats, S, idx, ppm, rt_tol):
            h = idx >= 0
            A = (t["tier"] == "A").values
            out = dict(n_tierA=int(A.sum()), recall_A=h[A].mean(), recall_B=h[~A].mean(),
                       recall_A_unflagged=h[A & (t["flags"] == "").values].mean())
            V = np.full((len(t), 6), np.nan)
            V[h] = vial_medians(S.iloc[idx[h]], files, D + [13, 14])
            sl = np.full(len(t), np.nan)
            sl[h] = log_slopes(V[h, :4], amt)
            q = h & (V[:, 4] > 0) & (V[:, 5] > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                err = np.abs(np.log2(V[:, 4] / V[:, 5]) - t["log2_1950_8231"].values)
            return out | dilution_cols(sl, t["dil_slope"].values) | dict(
                mix_quantified=int(q.sum()), mix_median_abs_log2_err_vs_raw=np.median(err[q]) if q.any() else np.nan,
                mix_within_2fold_of_raw=(err[q] <= 1).mean() if q.any() else np.nan)
        rows += pr2464_grid(run, truth, score)
    return pd.DataFrame(rows)


def score_runtime():
    rows = []
    for tool in TOOLS + PR2464_EXTRA:
        for run in RUNS:
            if tool in PR2464_EXTRA and run not in PR2464_RUNS:
                continue
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
                     ("ratio", score_ratio), ("spike", score_spike),
                     ("mc_synth", score_mc_synth), ("mc_eic", score_mc_eic),
                     ("pr2464_spike", score_pr2464_spike), ("pr2464_yeast", score_pr2464_yeast),
                     ("pr2464_plasma", score_pr2464_plasma), ("runtime", score_runtime)]:
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
