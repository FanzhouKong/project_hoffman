#!/usr/bin/env python3
"""PR002464 scores pooled over the three Orbitrap studies, for the scorecard (10 ppm / 0.1 min, bench/score.py rules).

Per tool and truth (spike / yeast / plasma): recall and chance-match (decoy) rate pooled over the truth peaks of the
studies the tool ran on (from results/scores/pr2464_*.tsv). Spike A/B ratio error against the design, recomputed per
peak as score_pr2464_spike does (tool log2(vial 18 / vial 19) vs designed log2(a/b), tier-A peaks whose weak side is
above noise) and pooled: over the peaks the tool quantified (scope "own"), and over the peaks that every tool with all
three studies quantified (scope "common"; IDSL.IPA ran on ST004501 only and is left out of it), since a tool that
reports only strong peaks gets the easy ratios. peak3d runs here as the GLR variant (PR2464_EXTRA).
Writes results/scores/pr2464_pooled.tsv.

usage: pr2464_pooled.py   (run on a compute node)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench import score as S  # noqa: E402

PPM, RT_TOL = 10, 0.1
TRUTHS = ["spike", "yeast", "plasma"]


def pooled_recall():
    rows = []
    for tr in TRUTHS:
        t = pd.read_csv(ROOT / f"results/scores/pr2464_{tr}.tsv", sep="\t")
        t = t[(t["ppm"] == PPM) & (t["rt_tol_min"] == RT_TOL)]
        for tool, g in t.groupby("tool"):
            n = g["n_truth"].sum()
            rows.append(dict(tool=tool, truth=tr, metric="recall", scope=",".join(g["study"]), n=int(n),
                             value=(g["recall"] * g["n_truth"]).sum() / n))
            rows.append(dict(tool=tool, truth=tr, metric="decoy_match", scope=",".join(g["study"]), n=int(n),
                             value=(g["decoy_match"] * g["n_truth"]).sum() / n))
    R = pd.DataFrame(rows)
    for tool, g in R[R["metric"] == "decoy_match"].groupby("tool"):  # all truth peaks together
        rows.append(dict(tool=tool, truth="all", metric="decoy_match", scope=g["scope"].iloc[0], n=int(g["n"].sum()),
                         value=(g["value"] * g["n"]).sum() / g["n"].sum()))
    return rows


def ratio_errors():
    """per-peak log2 ratio error of every tool on every study's spike truth (NaN = not quantified)"""
    out = []
    for run, study in S.PR2464_RUNS.items():
        files, _ = S.pr2464_study(run)
        t = pd.read_csv(S.PR2464_TRUTH / study / "spiked_truth.tsv", sep="\t", keep_default_na=False, na_values=[""])
        mz, rt = t["mz_obs"].fillna(t["mz"]).values, t["rt_apex"].values
        ok = (t["tier"] == "A").values & ~S.truthy(t["weak_at_noise"]).values
        for tool in S.TOOLS + S.PR2464_EXTRA:
            x = S.load(tool, run)
            if x is None:
                continue
            feats, M = x
            idx = S.nearest(feats, mz, rt, PPM, RT_TOL)
            h = idx >= 0
            V = np.full((len(t), 2), np.nan)
            V[h] = S.vial_medians(M.iloc[idx[h]], files, [18, 19])
            q = ok & h & (V[:, 0] > 0) & (V[:, 1] > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                err = np.where(q, np.log2(V[:, 0] / V[:, 1]) - t["log2FC"].values, np.nan)
            out.append(pd.DataFrame(dict(study=study, truth_id=t["truth_id"], tool=tool, eligible=ok, err=err)))
            print(f"{study} {tool}: quantified {int(q.sum())}, median |err| {np.nanmedian(np.abs(err)):.3f}", flush=True)
    return pd.concat(out, ignore_index=True)


def main():
    rows = pooled_recall()
    E = ratio_errors()
    W = E.pivot_table(index=["study", "truth_id"], columns="tool", values="err", aggfunc="first")
    full = [c for c in W.columns if E[E["tool"] == c]["study"].nunique() == len(S.PR2464_RUNS)]
    common = W[full].notna().all(axis=1)
    print("tools with all studies:", full, "| peaks quantified by all of them:", int(common.sum()), flush=True)
    for tool in W.columns:
        e = W[tool].dropna().abs()
        scope = ",".join(sorted(E[E["tool"] == tool]["study"].unique()))
        rows.append(dict(tool=tool, truth="spike", metric="ratio_median_abs_log2_err", scope=scope, n=len(e),
                         value=e.median()))
        rows.append(dict(tool=tool, truth="spike", metric="ratio_within_2fold", scope=scope, n=len(e),
                         value=(e <= 1).mean()))
        if tool in full:
            c = W.loc[common, tool].abs()
            rows.append(dict(tool=tool, truth="spike", metric="ratio_median_abs_log2_err_common", scope="common",
                             n=len(c), value=c.median()))
    out = ROOT / "results/scores/pr2464_pooled.tsv"
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False, float_format="%.4f")
    print(pd.DataFrame(rows).to_string())
    print("wrote", out)


if __name__ == "__main__":
    main()
