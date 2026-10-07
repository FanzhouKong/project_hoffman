#!/usr/bin/env python3
"""Validate peak3d's RT correction on finished benchmark runs.

For every results/peak3d/<run>/ with a DONE file: anchor residual MAD before/after per file,
fraction of files within one scan after correction, fraction of groups detected in >= 80 % of
files, held-out anchor check (if the run was processed with --holdout), and - for SZ22/YEAST -
agreement of the pairwise file shifts with the constant offsets measured independently in
data/truth/<DS>/params.json. Pass --uncorrected DIR to compare with an `align --no-rt-correction`
re-run of the same features (presence fraction with and without correction).

usage: rt_validate.py [RUN ...] [--uncorrected RUN=DIR ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.runs import RUNS  # noqa: E402

RES = ROOT / "results/peak3d"


def presence80(table_path):
    t = pd.read_csv(table_path, sep="\t", usecols=["n_detected"])
    return t


def validate_run(run, uncorrected=None):
    d = RES / run
    out = d / "peak3d_out"
    if not (d / "DONE").exists() or not out.exists():
        return None
    p = json.loads((out / "params.json").read_text())
    s = pd.read_csv(out / "rt_correction/summary.tsv", sep="\t")
    n_files = len(s)
    row = dict(run=run, n_files=n_files, medoid=p.get("medoid"), iterations=p.get("n_iterations"),
               rt_tol_s=60 * p["rt_tol_min"] if p.get("rt_tol_min") else np.nan,
               pooled_mad_s=p.get("pooled_anchor_mad_s"))
    others = s[s["role"] == "other"]
    if len(others):
        dt_s = 60 * np.median(np.diff(pd.read_csv(out / f"rt_correction/{s['file'].iloc[0]}.tsv", sep="\t")["rt_native"]))
        row.update(dt_s=dt_s, mad_before_med_s=others["mad_before_s"].median(), mad_after_med_s=others["mad_after_s"].median(),
                   frac_files_within_1dt=(others["mad_after_s"] <= dt_s).mean(),
                   max_abs_shift_p50_s=others["max_abs_shift_s"].median(), max_abs_shift_max_s=others["max_abs_shift_s"].max(),
                   n_flagged=int((~others["flag"].str.startswith("ok")).sum()),
                   flags=";".join(sorted(set(f.split(":")[0] for f in others["flag"] if not f.startswith("ok")))))
    t = presence80(out / "aligned_feature_table.tsv")
    row.update(n_groups=len(t), frac_groups_80pct=(t["n_detected"] >= 0.8 * n_files).mean(),
               frac_groups_all=(t["n_detected"] == n_files).mean())
    if uncorrected and Path(uncorrected).exists():
        tu = presence80(Path(uncorrected) / "aligned_feature_table.tsv")
        row.update(uncorrected_n_groups=len(tu), uncorrected_frac_groups_80pct=(tu["n_detected"] >= 0.8 * n_files).mean())
    h = out / "qc/holdout.tsv"
    if h.exists():
        ho = pd.read_csv(h, sep="\t")
        row.update(holdout_out_over_in=(ho["mad_out_s"] / ho["mad_in_s"].clip(lower=1e-6)).median(),
                   holdout_out_over_none=(ho["mad_out_s"] / ho["mad_none_s"].clip(lower=1e-6)).median())
    return row


def check_truth_offsets(run, ds):
    """pairwise file shifts vs the constant offsets in data/truth/<ds>/params.json"""
    d = RES / run / "peak3d_out"
    pj = ROOT / "data/truth" / ds / "params.json"
    if not (RES / run / "DONE").exists() or not pj.exists():
        return None
    tp = json.loads(pj.read_text())
    key = "rt_offsets_12C_s" if run.endswith("12C") else "rt_offsets_13C_s"
    files = [Path(f).stem for f in tp["files_12C" if run.endswith("12C") else "files_13C"]]
    truth = dict(zip(files, tp[key]))
    s = pd.read_csv(d / "rt_correction/summary.tsv", sep="\t")
    shifts = {}
    for f in s["file"]:
        w = pd.read_csv(d / f"rt_correction/{f}.tsv", sep="\t")
        shifts[f] = 60 * float(np.mean(w["rt_corr"] - w["rt_native"]))
    rows = []
    fs = [f for f in files if f in shifts]
    for i in range(len(fs)):
        for j in range(i + 1, len(fs)):
            # truth offsets are apex shifts of file relative to rep 1: file_j - file_i native time of the same ion
            d_truth = truth[fs[j]] - truth[fs[i]]
            # peak3d shift maps native -> consensus; a file that elutes later has a more negative shift
            d_ours = -(shifts[fs[j]] - shifts[fs[i]])
            rows.append(dict(run=run, file_i=fs[i], file_j=fs[j], truth_dt_s=d_truth, peak3d_dt_s=d_ours,
                             abs_err_s=abs(d_truth - d_ours)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=list(RUNS))
    ap.add_argument("--uncorrected", nargs="*", default=[], help="RUN=DIR pairs of align --no-rt-correction outputs")
    a = ap.parse_args()
    unc = dict(x.split("=", 1) for x in a.uncorrected)
    rows = [r for r in (validate_run(run, unc.get(run)) for run in a.runs) if r]
    pd.set_option("display.width", 250, "display.max_columns", 40)
    if rows:
        df = pd.DataFrame(rows)
        print("== RT correction summary ==")
        print(df.round(3).to_string(index=False))
        (ROOT / "results/scores").mkdir(exist_ok=True)
        df.to_csv(ROOT / "results/scores/rt_correction.tsv", sep="\t", index=False)
    else:
        print("no finished peak3d runs")
    try:
        sys.path.insert(0, str(ROOT / "bench/diag"))
        from apex_spread import spread_stats
        sp = [spread_stats(r) for r in a.runs if (RES / r / "DONE").exists()]
        sp = pd.DataFrame([x for x in sp if x and x.get("n_in_all_files")])
        if len(sp):
            print("\n== truth-feature apex spread across files, native -> corrected (the alignment must tighten this) ==")
            print(sp.round(2).to_string(index=False))
            sp.to_csv(ROOT / "results/scores/rt_truth_spread.tsv", sep="\t", index=False)
    except Exception as e:  # diagnostic only
        print("truth-spread table skipped:", repr(e))
    parts = [x for x in (check_truth_offsets(r, ds) for ds in ("SZ22", "YEAST") for r in (f"{ds}_12C", f"{ds}_13C"))
             if x is not None and len(x)]
    if parts:
        off = pd.concat(parts, ignore_index=True)
        print("\n== pairwise file shifts vs independent truth offsets (s) ==")
        print(off.round(3).to_string(index=False))
        off.to_csv(ROOT / "results/scores/rt_offsets_check.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
