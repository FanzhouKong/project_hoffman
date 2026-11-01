#!/usr/bin/env python3
"""Why ST004504 (RP pos) is left out of the PR002464 truth: spike content of every injection against its metadata.

Markers: per spiked standard, the primary-ion discovery candidate (all_tested_candidates.tsv of the spike truth, no
decoys) with the highest median height over the QC injections (vials 49-56, every material at 250 uL; their labels
agree with their contents in every study); kept when present (> 5 x the study floor) in >= 90 % of QC injections;
one per standard, >= MIN_DRT apart in RT, the N_MARK strongest. The same procedure on ST004500 (same RP column,
negative mode, labels consistent) is the contrast.
Per injection: height = maximum within +-WIN of the marker RT + the injection's offset (5 ppm EIC); spike index =
median over markers of height / the marker's median QC height. Expected index from the metadata = median over markers
of (a A + b B) / (250 a + 250 b), a and b the standard's amounts in spike mixes A and B, A and B the uL in the vial.
Also tests whether shifting the labels by k injections within a batch (k = -3..3) explains the observed pattern.
Writes results/pr2464_st004504/{markers,injections}_<study>.tsv and summary.json.
usage: pr2464_st004504_evidence.py [--cpus N]
"""
import argparse
import json
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
OUT = ROOT / "results/pr2464_st004504"
STUDIES = {"ST004500": "neg", "ST004504": "pos"}
N_MARK, MIN_DRT, WIN, PPM = 20, 0.1, 0.1, 5.0
QC_VIALS = range(49, 57)


def heights(args):
    path, off, mzs, rts = args
    run = load_run(path)
    out = np.zeros(len(mzs))
    for i, (m, t) in enumerate(zip(mzs, rts)):
        s0, s1 = run.scan_window(t + off, WIN)
        if s1 > s0:
            out[i] = stt.eic_win(run, m, PPM, s0, s1)[0].max()
    return out


def klass(r):
    if r["sample_type"] in ("QC", "test.QC"):
        return "QC"
    if r["sample_type"].endswith("blank"):
        return r["sample_type"].replace(".", " ")
    return {"matrix": "plasma / yeast only", "spike": "spike only", "spike_matrix": "spike + matrix"}.get(r["cls"], r["cls"])


def study(st, pool):
    mode = STUDIES[st]
    files = stt.study_files(st)
    off = pd.read_csv(TRUTH / st / "files.tsv", sep="\t").set_index("file")["rt_offset_s"] / 60
    files["off"] = files["file"].map(off).fillna(0.0)
    params = json.loads((TRUTH / st / "params.json").read_text())
    c = pd.read_csv(TRUTH / st / "all_tested_candidates.tsv", sep="\t", keep_default_na=False, na_values=[""])
    c = c[(c["decoy"].fillna("") == "") & c["ion"].isin(stt.PRIMARY[mode])].copy()
    c["mz_use"] = c["mz_obs"].fillna(c["mz"])
    c["rt_use"] = c["rt_apex"].fillna(c["rt"])
    c = c[np.isfinite(c["mz_use"]) & np.isfinite(c["rt_use"])].reset_index(drop=True)
    H = np.stack(pool.map(heights, [(p, o, c["mz_use"].values, c["rt_use"].values)
                                    for p, o in zip(files["path"], files["off"])]), axis=1)   # candidates x files
    qc = files["vial"].isin(QC_VIALS).values
    c["qc_median"] = np.median(H[:, qc], axis=1)
    c["qc_present"] = (H[:, qc] > 5 * params["floor"]).mean(axis=1)
    c = c[c["qc_present"] >= 0.9].sort_values("qc_median", ascending=False)
    keep = []
    for i, r in c.iterrows():                      # one per standard, distinct RT, strongest first
        if r["std"] in {c.loc[k, "std"] for k in keep}:
            continue
        if any(abs(r["rt_use"] - c.loc[k, "rt_use"]) < MIN_DRT for k in keep):
            continue
        keep.append(i)
        if len(keep) == N_MARK:
            break
    mk = c.loc[keep]
    Hm = H[mk.index.values] / mk["qc_median"].values[:, None]                        # relative to QC
    a, b = mk["a"].values[:, None], mk["b"].values[:, None]
    exp = (a * files["A"].values[None] + b * files["B"].values[None]) / (250 * a + 250 * b)
    inj = files[["file", "batch", "injection_order", "vial", "sample_type", "cls", "A", "B"]].copy()
    inj["class"] = [klass(r) for _, r in files.iterrows()]
    inj["spike_index"] = np.median(Hm, axis=0)
    inj["expected"] = np.median(exp, axis=0)
    inj["n_markers_found"] = (Hm > 0.05).sum(axis=0)
    OUT.mkdir(parents=True, exist_ok=True)
    mk[["std", "ion", "mz_use", "rt_use", "a", "b", "qc_median", "qc_present"]].to_csv(OUT / f"markers_{st}.tsv", sep="\t",
                                                                                      index=False)
    inj.to_csv(OUT / f"injections_{st}.tsv", sep="\t", index=False)
    return inj, mk


def summarize(inj):
    nonqc = inj["class"] != "QC"
    absent = nonqc & (inj["expected"] == 0)
    present = nonqc & (inj["expected"] > 0)
    s = dict(n_markers=None,
             labelled_without_spike=int(absent.sum()),
             without_spike_reading_ge_025=int((absent & (inj["spike_index"] >= 0.25)).sum()),
             labelled_with_spike=int(present.sum()),
             with_spike_reading_lt_005=int((present & (inj["spike_index"] < 0.05)).sum()),
             qc=int((~nonqc).sum()),
             qc_within_2fold=int((~nonqc & inj["spike_index"].between(0.5, 2)).sum()))
    by = {}
    for k, g in inj.groupby("class"):
        by[k] = dict(n=len(g), reading_ge_025=int((g["spike_index"] >= 0.25).sum()),
                     median_index=float(g["spike_index"].median()))
    s["by_class"] = by
    # does shifting the labels by k injections (within a batch, in injection order) explain the observations?
    shifts = {}
    for k in range(-3, 4):
        agree = []
        for _, g in inj.sort_values("injection_order").groupby("batch"):
            e = (g["expected"].values > 0)
            o = (g["spike_index"].values >= 0.25)
            if k > 0:
                e, o = e[:-k], o[k:]
            elif k < 0:
                e, o = e[-k:], o[:k]
            agree.append((e == o))
        shifts[k] = float(np.concatenate(agree).mean())
    s["label_shift_agreement"] = shifts
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpus", type=int, default=8)
    a = ap.parse_args()
    summ = {}
    with mp.Pool(a.cpus) as pool:
        for st in STUDIES:
            inj, mk = study(st, pool)
            s = summarize(inj)
            s["n_markers"] = len(mk)
            summ[st] = s
            print(f"[{st}] {len(mk)} markers; {json.dumps(s, indent=1)}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
