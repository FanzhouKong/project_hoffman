#!/usr/bin/env python3
"""Do asari / MassCube miss PR002464 truth peaks, or remove them with a sample-presence filter?

asari has no presence filter (features are peaks on the composite track summed over all samples, and the benchmark
reads export/full_Feature_table.tsv), so only MassCube is tested. MassCube 1.2.22 keeps an aligned feature only if
it was detected in more than detection_rate_cutoff (0.1) of the non-blank files, or its highest intensity is >= 10 x
ms1_abs_int_tol (50,000 for Orbitrap -> 500,000); our runs have no sample_table.csv, so all 224 injections count.

Each truth peak (spike / yeast / plasma, as scored by bench/score.py) is looked up in MassCube's 224 per-file feature
lists (results/masscube/<run>/single_files/) and in the aligned table. A truth peak MassCube did not report is
  never_picked      no per-file feature in any injection: missed at peak detection, before any presence filter
  presence_filter   picked in 1-22 injections, max per-file height < 500,000: the detection-rate rule removes it
  lost_other        picked in >= 23 injections or above 500,000, yet absent (alignment / quality filter / merging)
"Picked in a file" = a per-file feature within the ppm / RT tolerance, a proxy for MassCube's own alignment.
Also: the designed number of injections holding each truth source, and (spike truth, from per_file.tsv.gz) in how
many injections the raw truth peak reaches MassCube's per-file floor of 50,000.

Writes results/pr2464_presence_filter/{summary.tsv,summary.md,per_peak.tsv.gz}.
usage: pr2464_presence_filter.py [PPM RT_TOL_MIN ...]   (pairs; default 5 0.1 and 20 0.2)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.score import PR2464_RUNS, PR2464_TRUTH, RES, load, nearest  # noqa: E402

OUT = RES / "pr2464_presence_filter"
RATE_CUTOFF, ORBI_INT_TOL = 0.1, 50_000   # MassCube 1.2.22 defaults (params.py; Orbitrap ms1_abs_int_tol)
HIGH_KEEP = 10 * ORBI_INT_TOL              # kept regardless of detection rate at or above this height
CLASSES = ["reported", "never_picked", "presence_filter", "lost_other"]


def truth_sets(study):
    """truth tables reduced to mz / rt (minutes), as bench/score.py scores them"""
    d = PR2464_TRUTH / study
    s = pd.read_csv(d / "spiked_truth.tsv", sep="\t", keep_default_na=False, na_values=[""])
    y = pd.read_csv(d / "yeast_truth.tsv", sep="\t")
    p = pd.read_csv(d / "plasma_truth.tsv", sep="\t", keep_default_na=False, na_values=[""])
    return {"spike": pd.DataFrame({"truth_id": s["truth_id"], "mz": s["mz_obs"].fillna(s["mz"]), "rt": s["rt_apex"]}),
            "yeast": pd.DataFrame({"truth_id": y["truth_id"], "mz": y["mz"], "rt": y["rt"]}),
            "plasma": pd.DataFrame({"truth_id": p["truth_id"], "mz": p["mz_obs"], "rt": p["rt_apex"]})}


def single_files(run):
    """MassCube's per-file features: mz, rt (min), height, file index"""
    fs = sorted((RES / "masscube" / run / "single_files").glob("*.txt"))
    parts = []
    for i, f in enumerate(fs):
        t = pd.read_csv(f, sep="\t", usecols=["m/z", "RT", "peak_height"])
        parts.append(pd.DataFrame({"mz": t["m/z"], "rt": t["RT"], "h": t["peak_height"], "file": i}))
    return pd.concat(parts, ignore_index=True), len(fs)


def per_file_hits(sf, q_mz, q_rt, ppm, rt_tol):
    """number of distinct files with a per-file feature within ppm and rt_tol, and the max height among them"""
    o = np.argsort(sf["mz"].values)
    mz, rt, h, fi = (sf[c].values[o] for c in ["mz", "rt", "h", "file"])
    n, hmax = np.zeros(len(q_mz), int), np.zeros(len(q_mz))
    for i, (m, r) in enumerate(zip(q_mz, q_rt)):
        lo, hi = np.searchsorted(mz, m - m * ppm * 1e-6), np.searchsorted(mz, m + m * ppm * 1e-6, side="right")
        k = lo + np.flatnonzero(np.abs(rt[lo:hi] - r) <= rt_tol)
        if len(k):
            n[i], hmax[i] = len(np.unique(fi[k])), h[k].max()
    return n, hmax


def design_presence(study):
    """injections (of all) holding each truth source by design"""
    f = pd.read_csv(PR2464_TRUTH / study / "files.tsv", sep="\t")
    return {"spike": int((f["[spike]"] > 0).sum()), "spike_only": int(((f["[spike]"] > 0) & (f["cls"] == "spike")).sum()),
            "mix_A": int((f["A"] > 0).sum()), "mix_B": int((f["B"] > 0).sum()),
            "yeast": int((f["Yeast 12C"] > 0).sum()), "plasma": int((f["[plasma]"] > 0).sum()), "all": len(f)}


def main():
    a = [float(x) for x in sys.argv[1:]] or [5, 0.1, 20, 0.2]
    grid = list(zip(a[::2], a[1::2]))
    OUT.mkdir(parents=True, exist_ok=True)
    rows, peaks, md = [], [], []
    for run, study in PR2464_RUNS.items():
        feats, _ = load("masscube", run)
        sf, nfiles = single_files(run)
        min_n = int(np.floor(RATE_CUTOFF * nfiles)) + 1   # detection_rate > cutoff
        al = pd.read_csv(RES / "masscube" / run / "aligned_feature_table.txt", sep="\t", usecols=["detection_rate"])
        design = design_presence(study)
        md.append(f"## {study}\n\ninjections {nfiles}; designed presence {design}; MassCube keeps a feature picked in "
                  f">= {min_n} files or above {HIGH_KEEP:,}; aligned features {len(al):,}, of which "
                  f"{(al['detection_rate'] <= RATE_CUTOFF).sum():,} at detection_rate <= {RATE_CUTOFF} (high-intensity "
                  f"exemption)\n")
        pf = pd.read_csv(PR2464_TRUTH / study / "per_file.tsv.gz", sep="\t", usecols=["truth_id", "height"])
        raw50 = pf.assign(ok=pf["height"] >= ORBI_INT_TOL).groupby("truth_id")["ok"].sum()
        for name, t in truth_sets(study).items():
            for ppm, rt_tol in grid:
                rep = nearest(feats, t["mz"].values, t["rt"].values, ppm, rt_tol) >= 0
                n, hmax = per_file_hits(sf, t["mz"].values, t["rt"].values, ppm, rt_tol)
                cls = np.where(rep, "reported", np.where(n == 0, "never_picked",
                               np.where((n < min_n) & (hmax < HIGH_KEEP), "presence_filter", "lost_other")))
                row = dict(study=study, truth=name, ppm=ppm, rt_tol_min=rt_tol, n_truth=len(t))
                row |= {c: int((cls == c).sum()) for c in CLASSES}
                row |= dict(recall=rep.mean(), recall_if_no_presence_filter=(rep | (cls == "presence_filter")).mean(),
                            picked_any_file=(n > 0).mean(), median_files_picked_if_any=np.median(n[n > 0]) if (n > 0).any()
                            else np.nan)
                if name == "spike":
                    r50 = raw50.reindex(t["truth_id"]).fillna(0).values
                    row |= dict(raw_ge50k_in_ge_min_files=(r50 >= min_n).mean(), raw_ge50k_in_no_file=(r50 == 0).mean())
                rows.append(row)
                peaks.append(t.assign(study=study, truth=name, ppm=ppm, rt_tol_min=rt_tol, reported=rep, n_files_picked=n,
                                      max_height_picked=hmax, cls=cls))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.tsv", sep="\t", index=False)
    pd.concat(peaks).to_csv(OUT / "per_peak.tsv.gz", sep="\t", index=False)
    with open(OUT / "summary.md", "w") as fh:
        fh.write("# MassCube on PR002464: missed or removed by the detection-rate filter?\n\n" + "\n".join(md) + "\n")
        try:
            fh.write(df.round(3).to_markdown(index=False) + "\n")
        except ImportError:   # to_markdown needs tabulate
            fh.write("```\n" + df.round(3).to_string(index=False) + "\n```\n")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\n".join(md))
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
