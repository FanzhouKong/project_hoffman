#!/usr/bin/env python3
"""Score the GC compound rows (the same compound across files, gc/compounds.tsv) on the MRH truth.

In every file, a truth compound with >= --min-ions found clean ions (bench/gc/mrh_score.py, truth_rows.tsv) has a main
component (the component holding most of those ions' features, gc/components/<file>.features.tsv); that component's
compound row (gc/components/<file>.tsv, compound_id; -1 = in no reported row) is the compound's row in that file.

  split     per truth compound: the rows its file cells fall in; one row is right, more is a compound split
            across rows (cells_main = share of its cells in its most common row; cells_none = cells in no row)
  mixed     per row: the truth compounds whose cells fall in it, counting only pairs whose consensus retention
            times differ by more than one median peak width (co-eluting entries of the truth list share a component
            by design); two or more is two compounds made one
Entries whose found ions are the same features in every file are counted once (as bench/gc/deconv_eval.py does).

    python bench/gc/mrh_compound_rows.py --output results/.../mrh --score results/.../mrh_score [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--score", type=Path, required=True)
    ap.add_argument("--min-ions", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    tr = pd.read_csv(a.score / "truth_rows.tsv", sep="\t")
    tr = tr[tr["clean"] & tr["found"]]
    gp = json.loads((a.output / "gc" / "params.json").read_text())
    fw_s = 60.0 * float(np.median([f["fwhm_med"] for f in gp["files"]]))
    cells = []
    for level, g in tr.groupby("level"):
        stem = f"NEWSTD_{level}"
        fc = pd.read_csv(a.output / "gc" / "components" / f"{stem}.features.tsv", sep="\t").set_index("feature_id")
        comp = pd.read_csv(a.output / "gc" / "components" / f"{stem}.tsv", sep="\t").set_index("component_id")
        seen = set()
        for e, h in g.groupby("entry"):
            fid = tuple(sorted(set(h["feature_id"].astype(int))))
            if len(fid) < a.min_ions or fid in seen:
                continue
            seen.add(fid)
            c = fc.loc[list(fid), "component_id"].to_numpy()
            c = c[c >= 0]
            if not len(c):
                cells.append(dict(entry=e, level=level, rt=float(h["rt_consensus"].iloc[0]), component=-1, row=-1))
                continue
            vals, cnt = np.unique(c, return_counts=True)
            m = int(vals[np.argmax(cnt)])
            row = int(comp.loc[m, "compound_id"]) if m in comp.index and np.isfinite(comp.loc[m, "compound_id"]) else -1
            cells.append(dict(entry=e, level=level, rt=float(h["rt_consensus"].iloc[0]), component=m, row=row))
    C = pd.DataFrame(cells)
    per = []
    for e, g in C.groupby("entry"):
        r = g["row"].to_numpy()
        rr = r[r >= 0]
        vals, cnt = np.unique(rr, return_counts=True)
        per.append(dict(entry=e, rt=float(g["rt"].iloc[0]), cells=len(g), cells_none=int((r < 0).sum()),
                        rows=len(vals), cells_main=float(cnt.max() / len(g)) if len(cnt) else 0.0))
    E = pd.DataFrame(per)
    mixed = []
    for row, g in C[C["row"] >= 0].groupby("row"):
        ents = g.groupby("entry")["rt"].first().sort_values()
        rts = 60.0 * ents.to_numpy()
        far = np.any(np.abs(rts[:, None] - rts[None, :]) > fw_s)
        mixed.append(dict(row=row, entries=len(ents), far=bool(far), names="; ".join(ents.index)))
    Mx = pd.DataFrame(mixed)
    n_far = int(Mx["far"].sum()) if len(Mx) else 0
    print(f"truth compounds {len(E)}, cells {len(C)} | in one row {int((E['rows'] == 1).sum())}, split over 2+ rows "
          f"{int((E['rows'] >= 2).sum())} (rows per compound mean {E['rows'].mean():.2f}), cells in the main row "
          f"{(E['cells_main'] * E['cells']).sum() / E['cells'].sum():.3f}, cells in no row {int(E['cells_none'].sum())} | "
          f"rows holding 2+ compounds more than a peak width apart: {n_far}")
    if a.out is not None:
        a.out.mkdir(parents=True, exist_ok=True)
        C.to_csv(a.out / "cells.tsv", sep="\t", index=False)
        E.to_csv(a.out / "compounds.tsv", sep="\t", index=False)
        Mx.to_csv(a.out / "rows.tsv", sep="\t", index=False)
        (a.out / "summary.json").write_text(json.dumps(dict(
            compounds=len(E), cells=len(C), one_row=int((E["rows"] == 1).sum()), split=int((E["rows"] >= 2).sum()),
            rows_mean=float(E["rows"].mean()), cells_main=float((E["cells_main"] * E["cells"]).sum() / E["cells"].sum()),
            cells_none=int(E["cells_none"].sum()), mixed_far=n_far), indent=1))


if __name__ == "__main__":
    main()
