#!/usr/bin/env python3
"""Junk and duplicate counts of a GC-mode output of unit-resolution runs without truth (the LECO CSF folders):
features and their flags, same-file pairs of one nominal mass with apexes <= 1 s apart in two aligned rows, aligned
rows, compounds and compound pairs with the same quantifier <= 1 s apart (in disjoint files / sharing a file).

    python bench/gc/diag/csf_junk_counts.py OUTPUT [OUTPUT ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def counts(out: Path) -> dict:
    d = {}
    pj = json.loads((out / "params.json").read_text())
    S = float(pj.get("mass_scale") or 1.0)
    F = []
    for p in sorted((out / "features").glob("*.tsv")):
        if p.name == "manifest.tsv" or p.name.endswith(".rejected.tsv"):
            continue
        t = pd.read_csv(p, sep="\t", usecols=lambda c: c in ("mz", "rt_apex", "height", "flags", "group_id", "group_kept"))
        t["file"] = p.stem
        F.append(t)
    F = pd.concat(F, ignore_index=True)
    fl = F["flags"].fillna("").astype(str)
    d["features"] = len(F)
    for k in ("on_background", "bridged", "hole_apex", "saturated"):
        d[f"flag_{k}"] = int(fl.str.contains(k).sum())
    F["nom"] = np.rint(F["mz"] / S).astype(np.int64)
    F = F[F["group_id"] >= 0]
    n = 0
    for _, g in F.groupby(["file", "nom"]):
        if len(g) < 2:
            continue
        g = g.sort_values("rt_apex")
        r, gid = g["rt_apex"].to_numpy(), g["group_id"].to_numpy()
        n += int(np.sum((np.diff(r) * 60 <= 1.0) & (gid[1:] != gid[:-1])))
    d["ion_pairs_1s"] = n
    A = pd.read_csv(out / "aligned_feature_table.tsv", sep="\t", usecols=[0, 1, 2])
    d["aligned_rows"] = len(A)
    d["saturated_cells"] = int(pj.get("stage_times_s", {}).get("n_saturated_cells", -1))
    C = pd.read_csv(out / "gc" / "compounds.tsv", sep="\t")
    CT = pd.read_csv(out / "gc" / "compound_table.tsv", sep="\t")
    files = [c for c in CT.columns if c not in ("compound_id", "rt", "ri", "quant_mz", "name")]
    fs = {int(r.compound_id): set(f for f in files if pd.notna(r[f])) for _, r in CT.iterrows()}
    d["compounds"] = len(C)
    d["compounds_2files"] = int((C["n_files"] == 2).sum())
    dis = sh = 0
    for q, g in C.sort_values(["quant_mz", "rt"]).groupby("quant_mz"):
        r, ids = g["rt"].to_numpy(), g["compound_id"].to_numpy()
        for i in np.flatnonzero(np.diff(r) * 60 <= 1.0):
            if fs[ids[i]] & fs[ids[i + 1]]:
                sh += 1
            else:
                dis += 1
    d["cmp_pairs_1s_disjoint"] = dis
    d["cmp_pairs_1s_sharing"] = sh
    gp = json.loads((out / "gc" / "params.json").read_text())
    d["split_joins"] = gp.get("stage_times_s", {}).get("split_joins", "")
    return d


if __name__ == "__main__":
    rows = {Path(p).parent.name + "/" + Path(p).name: counts(Path(p)) for p in sys.argv[1:]}
    print(pd.DataFrame(rows).to_string())
