import sys
sys.path.insert(0, "/quobyte/metabolomicsgrp/fanzhou/hoffmann"); sys.path.insert(0, "/quobyte/metabolomicsgrp/fanzhou/hoffmann/peak3d/tests")
import numpy as np
from test_align import _truth, _warps, _file, GRID, T, DT
from peak3d import anchors as AN, warp as WP
from peak3d.align import AlignParams, align_run
truth = _truth(); W = _warps()
feats = {k: _file(truth, W[k], n_decoy=100, seed=i) for i, k in enumerate(W)}
orig = AN.cluster_weights
def run(label):
    res = align_run(feats, {k: GRID for k in W}, AlignParams())
    med = res.medoid; inside = (truth["rt"] > 0.6) & (truth["rt"] < T - 0.6)
    errs = {}
    for k in W:
        if k == med: continue
        fa = res.files[k]
        corr = fa.warp.forward(W[k](truth["rt"].values[inside])); target = W[med](truth["rt"].values[inside])
        errs[k] = (round(60 * np.median(np.abs(corr - target)), 2), fa.diag.K, fa.diag.n, fa.diag.interp, round(float(fa.anchor_w.sum()), 1), len(fa.anchor_w))
    print(f"{label:40s} medoid {med} | per file (median err s, K, n matched, interp, weight sum, n anchors): {errs}")
run("cluster weights ON (current)")
AN.cluster_weights = lambda rt, width: np.ones(len(rt)); run("cluster weights OFF")
AN.cluster_weights = orig
WP.ANCHORS_PER_KNOT = 8; run("weights ON, 8 anchors/knot"); WP.ANCHORS_PER_KNOT = 12
import peak3d.warp as W2
