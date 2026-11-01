#!/usr/bin/env python3
"""Does each PR002464 injection contain what its sample metadata says?

For every MS1 injection of a study, the amount of five components is estimated from marker peaks of
the truth sets built for that study (the truth of ST004500 is used for ST004504, same column):
  spike A   tier-A spike formula ions with log2FC >= 3 (reference: vial 18, 1000 uL spike A)
  spike B   tier-A spike formula ions with log2FC <= -3 (reference: vial 19)
  plasma    tier-A plasma peaks not seen in yeast or spike vials (reference: vial 17, 1000 uL plasma)
  yeast 12C confirmed tier-A yeast peaks (reference: vial 23, 1000 uL)
  yeast 13C their 13C partners (reference: vial 24)
markers: the strongest, at least MIN_DRT apart in RT (one per compound);
estimate = median over the markers of (height in the injection / median height in the reference vial),
heights read as the plain maximum within +-6 s of the marker RT. Expected = the metadata amount / 1000 uL.
An injection is flagged when a component expected present reads < 1/4 of it (or > 4x), or a component
expected absent reads > 0.25.
Writes results/pr2464_label_audit/<study>.tsv and a summary.
usage: pr2464_label_audit.py STUDY [--markers-from STUDY] [--cpus N]
"""
import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench import spiketruth as stt  # noqa: E402
from bench.msdata import load_run  # noqa: E402

TRUTH = ROOT / "data/truth/PR002464"
OUT = ROOT / "results/pr2464_label_audit"
N_MARK = 25
WIN = 0.1     # min, generous: offsets of spike-free injections are interpolated
MIN_DRT = 0.1 # min between markers of one component


def markers(src):
    sp = pd.read_csv(TRUTH / src / "spiked_truth.tsv", sep="\t")
    f = sp[(sp["tier"] == "A") & (sp["source"] == "formula") & (sp["rt_rank"] == 1)]
    pa = pd.read_csv(TRUTH / src / "plasma_truth.tsv", sep="\t")
    pa = pa[(pa["tier"] == "A") & ~pa["also_in_yeast"] & ~pa["also_in_spike"] & (pa["annotation"] == "primary")]
    ye = pd.read_csv(TRUTH / src / "yeast_truth.tsv", sep="\t")
    ye = ye[(ye["tier"] == "A") & ye["confirmed"]]
    rows = []
    for comp, d, mzc, rtc, hc, ref in (
            ("spikeA", f[f["log2FC"] >= 3], "mz_obs", "rt_apex", "height_dom", 18),
            ("spikeB", f[f["log2FC"] <= -3], "mz_obs", "rt_apex", "height_dom", 19),
            ("plasma", pa, "mz_obs", "rt_apex", "height_17", 17),
            ("yeast12C", ye, "mz", "rt", "height12", 23),
            ("yeast13C", ye, "mz_13C", "rt", "height13", 24)):
        # one marker per compound: the strongest ion, then only ions >= MIN_DRT from every kept marker
        # (several ions of one compound would let a single compound decide the estimate)
        kept = []
        for m, r in zip(*d.sort_values(hc, ascending=False)[[mzc, rtc]].values.T):
            if all(abs(r - k[1]) >= MIN_DRT for k in kept):
                kept.append((m, r))
            if len(kept) == N_MARK:
                break
        rows += [dict(comp=comp, mz=m, rt=r, ref=ref) for m, r in kept]
    return pd.DataFrame(rows)


def w_read(args):
    path, mzs, rts = args
    run = load_run(path)
    out = np.zeros(len(mzs))
    for i, (m, t) in enumerate(zip(mzs, rts)):
        s0, s1 = run.scan_window(t, WIN)
        if s1 > s0:
            e, _ = stt.eic_win(run, m, 5.0, s0, s1)
            out[i] = e.max()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study")
    ap.add_argument("--markers-from", default=None)
    ap.add_argument("--cpus", type=int, default=8)
    a = ap.parse_args()
    src = a.markers_from or a.study
    mk = markers(src)
    files = stt.study_files(a.study)
    with mp.Pool(a.cpus) as pool:
        H = np.stack(pool.map(w_read, [(p, mk["mz"].values, mk["rt"].values) for p in files["path"]]), axis=1)
    exp = {"spikeA": files["A"] / 1000, "spikeB": files["B"] / 1000,
           "plasma": (files["SRM 1950"] + files["SRM 8231"]) / 1000,
           "yeast12C": files["Yeast 12C"] / 1000, "yeast13C": files["Yeast 13C"] / 1000}
    rows = files[["file", "batch", "injection_order", "vial", "sample_type"]].copy()
    flags = [[] for _ in range(len(files))]
    for comp in exp:
        sel = (mk["comp"] == comp).values
        ref = (files["vial"] == mk.loc[sel, "ref"].iloc[0]).values
        refh = np.median(H[sel][:, ref], axis=1)
        good = refh > 0
        ratio = H[sel][good] / refh[good, None]
        est = np.median(ratio, axis=0)
        rows[f"{comp}_obs"] = est.round(3)
        rows[f"{comp}_exp"] = exp[comp].values
        for i, (o, e) in enumerate(zip(est, exp[comp].values)):
            if e > 0.04 and (o < e / 4 or o > e * 4):
                flags[i].append(f"{comp} {o:.2f} vs {e:.2f}")
            elif e == 0 and o > 0.25:
                flags[i].append(f"{comp} {o:.2f} vs 0")
    rows["flags"] = ["; ".join(f) for f in flags]
    OUT.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT / f"{a.study}.tsv", sep="\t", index=False)
    bad = rows[rows["flags"] != ""]
    print(f"[{a.study}] markers from {src}: " + ", ".join(f"{c} {int((mk['comp'] == c).sum())}" for c in exp))
    print(f"[{a.study}] {len(bad)} of {len(rows)} injections disagree with their metadata")
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120); pd.set_option("display.max_rows", 300)
    print(bad[["file", "vial", "sample_type", "flags"]].to_string(index=False))


if __name__ == "__main__":
    main()
