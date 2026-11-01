#!/usr/bin/env python3
"""Recall of each tool by the intensity of the truth peak.

Truth sets with a tool-independent intensity per peak:
  HZV029_cert   certified features, column Intensity of the published list
  SZ22_12C, YEAST_12C   credentialed truth, height12 (raw 12C apex height, bench/credtruth.py)
  PR2464_spike  spiked-standard truth, height_dom (raw apex height, median over the injections of the vial, 18 or 19,
                where the ion is stronger; bench/spiketruth.py)
  PR2464_yeast  yeast credentialed truth, height12 (bench/biotruth.py)
  PR2464_plasma plasma truth, height_17 (raw apex height, median over the vial-17 injections; bench/biotruth.py)
The PR2464 sets pool the three Orbitrap studies; a tool counts only on the studies it has a finished run for (IDSL.IPA:
ST004501). Each set is cut into equal-count intensity bins over all its peaks; a truth peak is found when the tool's
aligned table has a feature within 10 ppm / 0.1 min (bench.score.hits, the benchmark rule). peak3d is scored as its
current defaults (peak3d_GL2), as in the scorecard, and on PR002464 as its GLR variant (the only peak3d run there).
Writes results/scores/recall_by_intensity.tsv (column run = truth set).

usage: recall_by_intensity.py   (run on a compute node; reads the tools' feature tables only)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench import score as S  # noqa: E402
from bench.runs import CRED_PAIRS, LOCAL_TOOLS  # noqa: E402

PPM, RT_TOL = 10, 0.1
OTHERS = ["asari", "masscube", "idslipa", *LOCAL_TOOLS]
N_BINS = {"HZV029_cert": 5, "SZ22_12C": 5, "YEAST_12C": 8, "PR2464_spike": 8, "PR2464_yeast": 8, "PR2464_plasma": 8}


def truths():
    """(truth set, label, peak3d key, [(run, mz, rt, intensity), ...])"""
    p = next((ROOT / "data/raw/ASARI_DATA/x").glob("*/data/hzv029_manual_certified.txt"))
    t = pd.read_csv(p, sep="\t")
    yield "HZV029_cert", "certified features", "peak3d_GL2", [
        ("HZV029_cert", t["moverz"].values, t["RT_minutes"].values, t["Intensity"].values)]
    for name, (r12, _) in CRED_PAIRS.items():
        t = pd.read_csv(ROOT / f"data/truth/{name}/credentialed.tsv", sep="\t")
        yield r12, "credentialed peaks", "peak3d_GL2", [(r12, t["mz"].values, t["rt"].values, t["height12"].values)]
    for tr, fname, mzc, rtc, hc in [("spike", "spiked_truth.tsv", "mz_obs", "rt_apex", "height_dom"),
                                    ("yeast", "yeast_truth.tsv", "mz", "rt", "height12"),
                                    ("plasma", "plasma_truth.tsv", "mz_obs", "rt_apex", "height_17")]:
        parts = []
        for run, study in S.PR2464_RUNS.items():
            t = pd.read_csv(S.PR2464_TRUTH / study / fname, sep="\t")
            mz = t[mzc].fillna(t["mz"]).values if tr == "spike" else t[mzc].values
            parts.append((run, mz, t[rtc].values, t[hc].values))
        yield f"PR2464_{tr}", f"{tr} truth peaks", "peak3d_GLR", parts


def main():
    rows = []
    for name, label, p3d, parts in truths():
        n = N_BINS[name]
        allh = np.concatenate([p[3] for p in parts])
        edges = np.quantile(allh, np.linspace(0, 1, n + 1))
        ball = np.clip(np.searchsorted(edges, allh, side="right") - 1, 0, n - 1)
        med = [float(np.median(allh[ball == k])) for k in range(n)]  # bin position over all peaks of the set
        for tool in [p3d] + OTHERS:
            inten, hit, used = [], [], []
            for run, mz, rt, h in parts:
                x = S.load(tool, run)
                if x is None:
                    continue
                inten.append(h); hit.append(S.hits(x[0], mz, rt, PPM, RT_TOL)); used.append(run)
            if not used:
                print(f"{name} {tool}: no output", flush=True)
                continue
            inten, hit = np.concatenate(inten), np.concatenate(hit)
            b = np.clip(np.searchsorted(edges, inten, side="right") - 1, 0, n - 1)
            print(f"{name} {tool} ({len(used)} runs): {int(hit.sum())} / {len(hit)} = {hit.mean():.3f}", flush=True)
            for k in range(n):
                m = b == k
                rows.append(dict(run=name, truth=label, tool=tool, runs=",".join(used), bin=k, lo=edges[k],
                                 hi=edges[k + 1], median=med[k], n=int(m.sum()), found=int(hit[m].sum()),
                                 recall=round(float(hit[m].mean()), 4) if m.any() else np.nan))
    out = ROOT / "results/scores/recall_by_intensity.tsv"
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
