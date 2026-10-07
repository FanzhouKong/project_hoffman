#!/usr/bin/env python3
"""Offline check of peak3d's RT correction against the external truth lists, without rerunning the
pipeline: re-runs align_run on the published per-file feature tables (results/peak3d/RUN/peak3d_out/
features) and scan grids (rt_correction/<stem>.tsv), then applies each variant's warps to the truth
features' native apexes saved by rt_harm_diag.py (results/rt_harm/RUN.npz). Nothing is written to
results/peak3d.
usage: rt_gate_eval.py OUT.tsv RUN [RUN ...]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from peak3d.align import AlignParams, align_run  # noqa: E402
from peak3d.features import COLUMNS  # noqa: E402


def load(run):
    d = ROOT / "results/peak3d" / run / "peak3d_out"
    z = np.load(ROOT / f"results/rt_harm/{run}.npz")
    stems = [str(s) for s in z["stems"]]
    feats = {s: pd.read_csv(d / "features" / f"{s}.tsv", sep="\t", usecols=COLUMNS) for s in stems}
    grids = {s: pd.read_csv(d / "rt_correction" / f"{s}.tsv", sep="\t", usecols=["rt_native"])["rt_native"].values for s in stems}
    return stems, feats, grids, z["nat"], z["cor"], float(z["dt"])


def metrics(nat, cor, dt):
    """per truth feature: SD across files (s) and the 1.4826 x MAD spread shown on the validation page"""
    sd = lambda a: 60 * np.nanstd(a, 1, ddof=1)
    mads = lambda a: 60 * 1.4826 * np.nanmedian(np.abs(a - np.nanmedian(a, 1, keepdims=True)), 1)
    sn, sc, mn, mc = sd(nat), sd(cor), mads(nat), mads(cor)
    dsd = sc - sn
    return dict(sd_nat=np.median(sn), sd_cor=np.median(sc), sd_mean_nat=np.mean(sn), sd_mean_cor=np.mean(sc),
                wid_any=np.mean(dsd > 1e-9), wid_025=np.mean(dsd > 0.25), wid_scan=np.mean(dsd > 60 * dt),
                tig_025=np.mean(dsd < -0.25), tig_scan=np.mean(dsd < -60 * dt),
                mad_nat=np.median(mn), mad_cor=np.median(mc), mad_wid=np.mean(mc > mn + 1e-9), mad_tig=np.mean(mc < mn - 1e-9))


VARIANTS = [("gate_off", AlignParams(gate=False)), ("gate_on", AlignParams())]
SHOW = "gate_on"

if __name__ == "__main__":
    out = Path(sys.argv[1])
    rows = []
    pd.set_option("display.width", 250, "display.max_columns", 40)
    for run in sys.argv[2:]:
        stems, feats, grids, nat, cor_pub, dt = load(run)
        rows.append(dict(run=run, variant="published", **metrics(nat, cor_pub, dt)))
        for name, P in VARIANTS:
            tic = time.perf_counter()
            res = align_run({s: f.copy() for s, f in feats.items()}, grids, P)
            el = time.perf_counter() - tic
            cor = np.column_stack([res.files[s].warp.forward(nat[:, j]) for j, s in enumerate(stems)])
            if name == "gate_off":
                diff = 60 * np.nanmax(np.abs(cor - cor_pub))
                print(f"{run}: gate_off reproduces the published rt_corr to {diff:.3g} s (max over truth cells)")
            rows.append(dict(run=run, variant=name, align_s=el, **metrics(nat, cor, dt)))
            np.save(ROOT / f"results/rt_harm/{run}__{name}.npy", cor)
            if name == SHOW:
                g = pd.DataFrame([dict(file=s[-14:], n_val=fa.gate["n_val"], lam=" ".join(f"{v:.2f}" for v in fa.gate["lam"]),
                                       nodes=" ".join(f"{v:.1f}" for v in fa.gate["nodes"]), dev_nat=fa.gate["dev_native_s"],
                                       dev_fit=fa.gate["dev_full_s"], dev_applied=fa.gate["dev_gated_s"])
                                  for s, fa in res.files.items() if fa.gate])
                if len(g) <= 12:
                    print(g.round(2).to_string(index=False))
                else:
                    print(g[["n_val", "dev_nat", "dev_fit", "dev_applied"]].describe().round(2).to_string())
        print(pd.DataFrame(rows[-1 - len(VARIANTS):]).round(3).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False, float_format="%.4f")
