#!/usr/bin/env python3
"""Score the deconvolution on the MRH set at the level of picked ions, without rewriting an output: every file's
features are deconvolved in-process (peak3d.gc.deconv) and compared with the screened truth of bench/gc/mrh_score.py
(its truth_rows.tsv: clean rows and the feature each was found as).

  completeness  per truth compound with >= 3 found clean ions: the share of those ions' features that sit in one
                component (the compound's main component); ions in no component count against it
  separation    per pair of truth compounds of one file whose consensus retention times differ by at most two
                median peak widths and that share no feature: separated when their main components differ; by
                distance in peak widths
  size          components per file, and how many rest on one picked ion

    python bench/gc/deconv_eval.py --output results/gc_mrh/peak3d --score results/gc_mrh/score [--set tau_k=5] [--cores 11]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LEVELS = ["0_1", "0_5", "1", "5", "10", "25", "50", "100", "250", "500", "1000"]
BINS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]


def compound_features(t: pd.DataFrame, min_ions: int = 3) -> dict:
    """entry -> (consensus RT, sorted feature ids of its found clean ions); entries with the same features once"""
    out, seen = {}, set()
    for e, g in t[t["clean"] & t["found"]].groupby("entry"):
        fid = tuple(sorted(set(g["feature_id"].astype(int))))
        if len(fid) >= min_ions and fid not in seen:
            seen.add(fid)
            out[e] = (float(g["rt_consensus"].iloc[0]), np.array(fid, np.int64))
    return out


def evaluate(feature_component: np.ndarray, comps: dict, fwhm_min: float):
    """(per-compound rows, per-pair rows) for one file"""
    rows, main = [], {}
    for e, (rt, fid) in comps.items():
        c = feature_component[fid]
        vals, cnt = np.unique(c[c >= 0], return_counts=True)
        m = int(vals[np.argmax(cnt)]) if len(vals) else -1
        main[e] = m
        rows.append(dict(entry=e, rt=rt, n=len(fid), main=m, main_share=float(np.mean(c == m)) if m >= 0 else 0.0,
                         n_components=int(len(vals)), unassigned=int((c < 0).sum())))
    pairs = []
    names = sorted(comps, key=lambda e: comps[e][0])
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = abs(comps[b][0] - comps[a][0]) / fwhm_min
            if d > BINS[-1]:
                break
            if len(np.intersect1d(comps[a][1], comps[b][1])):
                continue
            pairs.append(dict(a=a, b=b, d_fwhm=d, separated=bool(main[a] != main[b] or main[a] < 0)))
    return rows, pairs


def _one(job):
    from peak3d.gc import deconv as DC
    from peak3d.io import Cloud
    output, score, lv, sets = job
    stem = f"NEWSTD_{lv}"
    P = DC.params_from_dict(json.loads((output / "features" / f"{stem}.params.json").read_text()))
    cloud = Cloud.load_npz(output / ".cache" / f"{stem}.npz")
    F = pd.read_csv(output / "features" / f"{stem}.tsv", sep="\t", low_memory=False)
    import time
    t0 = time.perf_counter()
    d = DC.deconvolve(cloud, F, P, DC.DeconvParams(**sets))
    sec = time.perf_counter() - t0
    T = pd.read_csv(score / "truth_rows.tsv", sep="\t", dtype={"level": str})
    rows, pairs = evaluate(d.feature_component, compound_features(T[T["level"] == lv]), P.fwhm_med)
    nf = d.components["n_features"].values
    info = dict(level=lv, features=len(F), components=len(d.components), one_feature=int((nf <= 1).sum()),
                assigned=int((d.feature_component >= 0).sum()), seconds=round(sec, 2))
    return info, [dict(level=lv, **r) for r in rows], [dict(level=lv, **p) for p in pairs]


def parse_sets(items):
    from peak3d.gc.deconv import DeconvParams
    types = {f.name: f.type for f in fields(DeconvParams)}
    out = {}
    for it in items or ():
        k, _, v = it.partition("=")
        if k not in types:
            raise SystemExit(f"--set {it}: not a DeconvParams field")
        t = types[k]
        out[k] = (v.lower() in ("1", "true")) if t == "bool" else int(v) if t == "int" else float(v) if t == "float" else v
    return out


def summarise(infos, rows, pairs):
    R, Pp, I = pd.DataFrame(rows), pd.DataFrame(pairs), pd.DataFrame(infos)
    by = R.groupby("level").agg(compounds=("entry", "size"), main_share=("main_share", "mean"),
                                whole=("main_share", lambda v: float((v >= 1.0).mean())),
                                ge_0_9=("main_share", lambda v: float((v >= 0.9).mean()))).reindex([lv for lv in LEVELS if lv in set(R["level"])])
    by = by.join(I.set_index("level")[["features", "components", "one_feature", "assigned", "seconds"]])
    total = dict(compounds=len(R), main_share=R["main_share"].mean(), whole=float((R["main_share"] >= 1.0).mean()),
                 ge_0_9=float((R["main_share"] >= 0.9).mean()), features=int(I["features"].sum()), components=int(I["components"].sum()),
                 one_feature=int(I["one_feature"].sum()), assigned=int(I["assigned"].sum()), seconds=float(I["seconds"].sum()))
    by.loc["all"] = total
    sep = pd.DataFrame()
    if len(Pp):
        Pp["bin"] = pd.cut(Pp["d_fwhm"], BINS, include_lowest=True)
        sep = Pp.groupby("bin", observed=False)["separated"].agg(pairs="size", separated="mean").reset_index()
    return by, sep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--score", type=Path, required=True, help="folder of mrh_score.py for that output (truth_rows.tsv)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="a DeconvParams field (repeatable)")
    ap.add_argument("--levels", nargs="*", default=None)
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help="folder for compounds.tsv / pairs.tsv / by_level.tsv")
    a = ap.parse_args(argv)
    sets = parse_sets(a.set)
    levels = [lv for lv in (a.levels or LEVELS) if (a.output / "features" / f"NEWSTD_{lv}.tsv").exists()]
    jobs = [(a.output, a.score, lv, sets) for lv in levels]
    if a.cores > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(min(a.cores, len(jobs))) as ex:
            res = list(ex.map(_one, jobs))
    else:
        res = [_one(j) for j in jobs]
    infos = [r[0] for r in res]
    rows = [x for r in res for x in r[1]]
    pairs = [x for r in res for x in r[2]]
    by, sep = summarise(infos, rows, pairs)
    pd.set_option("display.width", 220)
    print(f"deconvolution of {a.output} with {sets or 'defaults'}")
    print(by.to_string(float_format=lambda v: f"{v:.3f}"))
    print()
    print(sep.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(a.out / "compounds.tsv", sep="\t", index=False, float_format="%.5g")
        pd.DataFrame(pairs).to_csv(a.out / "pairs.tsv", sep="\t", index=False, float_format="%.5g")
        by.to_csv(a.out / "by_level.tsv", sep="\t", float_format="%.4g")


if __name__ == "__main__":
    main()
