# diagnostic: per RT bin, what the anchors say about a file's shift vs what the truth features say
import sys, glob
import numpy as np, pandas as pd
ROOT = "/quobyte/metabolomicsgrp/fanzhou/hoffmann"
run, stem_sub = sys.argv[1], sys.argv[2]      # e.g. YEAST_12C 12C14N-c
A = pd.read_csv(f"{ROOT}/results/peak3d/{run}/peak3d_out/rt_correction/anchors.tsv", sep="\t")
S = pd.read_csv(f"{ROOT}/results/peak3d/{run}/peak3d_out/rt_correction/summary.tsv", sep="\t")
medoid = S.loc[S.role == "medoid", "file"].iloc[0]
stem = [f for f in S.file if stem_sub in f][0]
a = A[A.file == stem]
if run.startswith("YEAST_1"):
    tr = pd.read_csv(f"{ROOT}/data/truth/YEAST/credentialed.tsv", sep="\t"); tr = tr[(tr.tier == "A") & (~tr.ambiguous.astype(bool))]
elif run.startswith("SZ22"):
    tr = pd.read_csv(f"{ROOT}/data/truth/SZ22/credentialed.tsv", sep="\t"); tr = tr[(tr.tier == "A") & (~tr.ambiguous.astype(bool))]
fm = pd.read_csv(f"{ROOT}/results/peak3d/{run}/peak3d_out/features/{medoid}.tsv", sep="\t")
ff = pd.read_csv(f"{ROOT}/results/peak3d/{run}/peak3d_out/features/{stem}.tsv", sep="\t")
def find(f, m, t):
    k = np.flatnonzero((np.abs(f.mz.values - m) / m * 1e6 <= 5) & (np.abs(f.rt.values - t) <= 0.15))
    return k[np.argmax(f.height.values[k])] if len(k) else -1
rows = []
for m, t in zip(tr.mz.values, tr.rt.values):
    i, j = find(fm, m, t), find(ff, m, t)
    if i >= 0 and j >= 0:
        rows.append(dict(rt=fm.rt.values[i], d_truth=60 * (fm.rt.values[i] - ff.rt.values[j]), h=ff.height.values[j]))
T = pd.DataFrame(rows)
w = pd.read_csv(f"{ROOT}/results/peak3d/{run}/peak3d_out/rt_correction/{stem}.tsv", sep="\t")
print(f"{run} {stem} vs medoid {medoid}: {len(a)} matched anchors, {len(T)} truth features found in both")
print(" RT bin      anchors: n  wsum  median(consensus-native) s | truth: n  median(medoid-file) s  | warp shift s")
edges = np.linspace(0, max(w.rt_native.max(), 1), 11)
for lo, hi in zip(edges[:-1], edges[1:]):
    aa = a[(a.rt_native >= lo) & (a.rt_native < hi)]; tt = T[(T.rt >= lo) & (T.rt < hi)]
    ws = w[(w.rt_native >= lo) & (w.rt_native < hi)]
    print(f" {lo:5.1f}-{hi:5.1f}   {len(aa):3d}  {aa.cluster_w.sum() if len(aa) else 0:5.1f}  {aa.residual_before_s.median() if len(aa) else float('nan'):+6.2f}            |  {len(tt):3d}   {tt.d_truth.median() if len(tt) else float('nan'):+6.2f}                | {60 * (ws.rt_corr - ws.rt_native).mean() if len(ws) else float('nan'):+6.2f}")
