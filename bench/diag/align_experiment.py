# diagnostic: re-run alignment offline on saved features with variants of the fit rules; score by the
# spread of truth-feature apexes across files (native vs corrected)
import sys, glob, json
import numpy as np, pandas as pd
ROOT = "/quobyte/metabolomicsgrp/fanzhou/hoffmann"
sys.path.insert(0, ROOT)
from peak3d import align as AL, warp as WP
from peak3d.align import AlignParams, align_run
sys.path.insert(0, f"{ROOT}/bench/diag")
from apex_spread import truth_table

def load(run):
    d = f"{ROOT}/results/peak3d/{run}/peak3d_out"
    stems = sorted(p.split("/")[-1][:-4] for p in glob.glob(f"{d}/features/*.tsv") if not p.endswith(("manifest.tsv", "rejected.tsv")))
    feats = {s: pd.read_csv(f"{d}/features/{s}.tsv", sep="\t") for s in stems}
    grids = {s: pd.read_csv(f"{d}/rt_correction/{s}.tsv", sep="\t")["rt_native"].values for s in stems}
    return feats, grids

def spread(run, feats, res):
    tr = truth_table(run).reset_index(drop=True)
    stems = list(feats)
    nat = np.full((len(tr), len(stems)), np.nan); cor = np.full_like(nat, np.nan)
    for j, s in enumerate(stems):
        f = feats[s]; o = np.argsort(f.mz.values); smz = f.mz.values[o]; w = res.files[s].warp
        for i, (m, t) in enumerate(zip(tr.mz.values, tr.rt.values)):
            lo = np.searchsorted(smz, m * (1 - 5e-6)); hi = np.searchsorted(smz, m * (1 + 5e-6), side="right")
            k = o[lo:hi]; k = k[np.abs(f.rt.values[k] - t) <= 0.15]
            if len(k):
                kk = k[np.argmax(f.height.values[k])]; nat[i, j] = f.rt.values[kk]; cor[i, j] = w.forward(np.array([f.rt.values[kk]]))[0]
    ok = np.isfinite(nat).sum(axis=1) == len(stems)
    sn = 60 * (nat[ok].max(axis=1) - nat[ok].min(axis=1)); sc = 60 * (cor[ok].max(axis=1) - cor[ok].min(axis=1))
    return ok.sum(), np.median(sn), np.percentile(sn, 90), np.median(sc), np.percentile(sc, 90), (sc < sn - 1e-9).mean(), (sc > sn + 1e-9).mean()

variants = {"quantile bins, 12/knot, need 3 (default)": dict(APK=12, NEED=3),
            "quantile bins, 8/knot": dict(APK=8, NEED=3),
            "quantile bins, need 2": dict(APK=12, NEED=2)}
for run in sys.argv[1:]:
    feats, grids = load(run)
    print(f"\n== {run} ({len(feats)} files) ==")
    for name, v in variants.items():
        WP.ANCHORS_PER_KNOT, AL.CONSENSUS_MIN_FILES = v["APK"], v["NEED"]
        res = align_run(feats, grids, AlignParams())
        n, mn, pn, mc, pc, b, w = spread(run, feats, res)
        ks = [res.files[s].diag.K for s in feats if res.files[s].role == "other"]
        print(f"  {name:32s} truth in all files {n:4d} | spread median {mn:.2f} -> {mc:.2f} s, p90 {pn:.2f} -> {pc:.2f} | tightened {b:.0%} widened {w:.0%} | knots {ks}")
