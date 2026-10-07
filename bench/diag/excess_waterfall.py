# diagnostic: how far would each candidate fix bring peak3d's aligned feature count down, and what
# would it cost in truth recall? Cumulative steps on the existing aligned table (post hoc, no rerun):
#   rounding   merge complementary same-ion group pairs split only by the %.6g m/z round trip
#   eic        drop groups whose detected members are mostly EIC-junk (junk_classes.py classes)
#   narrow     drop groups whose members mostly hold fewer than max(6, half a median FWHM) present scans
#   presence   drop groups detected (not gap-filled) in fewer than max(2, 10 % of the files)
# Needs results/diag_excess/<RUN>_classes.tsv and <RUN>_groups_cls.tsv (junk_classes.py).
# usage: excess_waterfall.py RUN [RUN ...]
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/diag"))
from bench.runs import LOCAL_TOOLS  # noqa: E402
from bench.score import load  # noqa: E402
from junk_classes import match_groups, truth_table  # noqa: E402


def rounding_absorbed(t, det):
    """groups that a full-precision m/z would have merged into a partner (one per pair)"""
    mz, lo, hi = t["mz"].values, t["rt_min"].values, t["rt_max"].values
    nd = det.sum(1)
    step = np.where(mz < 100, 1e-4, np.where(mz < 1000, 1e-3, 1e-2))
    o = np.argsort(mz)
    smz = mz[o]
    absorbed = np.zeros(len(t), bool)
    for i in range(len(t)):
        if absorbed[i]:
            continue
        a = np.searchsorted(smz, mz[i] + 0.5 * step[i])
        b = np.searchsorted(smz, mz[i] + 1.5 * step[i])
        for j in o[a:b]:
            if absorbed[j] or abs(mz[j] - mz[i]) / mz[i] * 1e6 <= 5.0:
                continue
            if min(hi[i], hi[j]) <= max(lo[i], lo[j]) or (det[i] & det[j]).any():
                continue
            absorbed[j if nd[j] <= nd[i] else i] = True
            break
    return absorbed


def main(run):
    d = ROOT / "results/peak3d" / run / "peak3d_out"
    t = pd.read_csv(d / "aligned_feature_table.tsv", sep="\t")
    m = pd.read_csv(d / "filled_mask.tsv", sep="\t").set_index("group_id").loc[t["group_id"]]
    stems = list(m.columns)
    nf = len(stems)
    det = (m.values == 0) & (t[stems].values > 0)
    G = pd.read_csv(ROOT / f"results/diag_excess/{run}_groups_cls.tsv", sep="\t", usecols=["group_id", "junk", "supported"])
    A = pd.read_csv(ROOT / f"results/diag_excess/{run}_classes.tsv", sep="\t",
                    usecols=["group_id", "cls", "n_scans", "n_scans_rel", "rep_frac"])
    fwhm_scans = A["n_scans"] / A["n_scans_rel"]
    narrow = A.assign(nar=A["n_scans"] < np.maximum(6.0, 0.5 * fwhm_scans)).groupby("group_id")["nar"].mean() > 0.5
    rep = A.groupby("group_id")["rep_frac"].mean()
    t = t.merge(G, on="group_id", how="left")
    t["narrow"] = t["group_id"].map(narrow).fillna(False).values.astype(bool)
    t["rep"] = t["group_id"].map(rep).values
    t["junk"] = t["junk"].fillna(False).astype(bool)
    need = max(2, int(np.ceil(0.1 * nf))) if nf >= 3 else 1
    keep = np.ones(len(t), bool)
    steps = [("current", keep.copy())]
    keep &= ~rounding_absorbed(t, det)
    steps.append(("+ full-precision m/z", keep.copy()))
    keep &= ~t["junk"].values
    steps.append(("+ EIC re-check", keep.copy()))
    keep &= ~t["narrow"].values
    steps.append(("+ >= max(6, FWHM/2) scans", keep.copy()))
    if need > 1:
        keep &= det.sum(1) >= need
        steps.append((f"+ detected in >= {need} files", keep.copy()))
    T = truth_table(run)
    tm = match_groups(T, t["mz"].values, t["rt"].values) if T is not None else None
    print(f"\n== {run}: {nf} files")
    others = {k: load(k, run) for k in ("asari", "masscube", *LOCAL_TOOLS)}
    print("   other tools: " + ", ".join(f"{k} {len(v[0])}" for k, v in others.items() if v is not None))
    for name, k in steps:
        line = f"  {name:28s} {k.sum():7d}  supported {t['supported'].values[k].mean():.2f}"
        if t["rep"].notna().any():
            r = t["rep"].values[k]
            line += f"  replicable {np.nanmean(r >= 0.5):.2f}"
        if tm is not None:
            h = np.array([len(x) > 0 and k[x].any() for x in tm])
            if "tp" in T:
                tp = T["tp"].values
                line += f"  TP {int((h & tp).sum())} FP {int((h & ~tp).sum())}"
            else:
                line += f"  recall {h.mean():.3f} ({int(h.sum())}/{len(T)})"
        print(line)


if __name__ == "__main__":
    for r in sys.argv[1:]:
        main(r)
