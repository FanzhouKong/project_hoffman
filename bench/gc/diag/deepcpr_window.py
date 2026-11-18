#!/usr/bin/env python3
"""One retention-time window of a DeepCPR benchmark run, in numbers: the raw ion traces, what DeepCPR reports
there, and what Mosa picked, rejected and made of it. For judging a disagreement on the raw data.

    python bench/gc/diag/deepcpr_window.py SET STEM T0 T1 [--mosa mosa_dsk] [--ions 55,67,...] [--top 10]
SET: aa or fa; T0, T1: seconds."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from bench.gc.cdf_to_mzml import C13, unit_mass                                    # noqa: E402
from bench.gc.deepcpr_score import load_deepcpr, load_mosa, observed, r2           # noqa: E402

M = Path("/quobyte/metabolomicsgrp/fanzhou/hoffmann")
np.set_printoptions(linewidth=260, suppress=True)


def top_ions(spec, masses, n=8):
    o = np.argsort(-spec)[:n]
    return " ".join(f"{masses[j]}:{100 * spec[j] / max(spec.max(), 1e-12):.0f}" for j in o if spec[j] > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set"); ap.add_argument("stem"); ap.add_argument("t0", type=float); ap.add_argument("t1", type=float)
    ap.add_argument("--mosa", default="mosa_dsk")
    ap.add_argument("--ions", default="")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--no-traces", action="store_true")
    g = ap.parse_args()
    raw = M / "data/raw/DEEPCPR_GC/data" / ("amino acids" if g.set == "aa" else "fatty acids") / f"{g.stem}.CDF"
    rt, masses, Xb, _ = observed(raw)
    R = M / "results/gc_deepcpr"
    D = load_deepcpr(R / "deepcpr" / g.set, g.stem, masses, rt)
    A = load_mosa(R / g.mosa / g.set, g.stem, masses, rt)
    a, b = np.searchsorted(rt, [g.t0, g.t1])
    print(f"{g.stem} {rt[a]:.1f}-{rt[b - 1]:.1f} s, scans {a}-{b - 1}, dt {np.median(np.diff(rt)):.3f} s")
    seg = Xb[a:b]
    ions = [int(x) for x in g.ions.split(",") if x] or [int(masses[j]) for j in np.argsort(-seg.max(axis=0))[:g.top]]
    if not g.no_traces:
        print("raw traces, % of each ion's maximum in the window (baseline removed); time of every 2nd scan:")
        print("            t =", " ".join(f"{t:5.1f}" for t in rt[a:b:2]))
        for m in ions:
            y = seg[:, m - masses[0]]
            print(f"  m/z {m:3d} max {y.max():9.0f} @ {rt[a + int(np.argmax(y))]:6.1f}", (100 * y / max(y.max(), 1)).round(0).astype(int))
    print("DeepCPR components:")
    T = pd.read_csv(R / "deepcpr" / g.set / "single" / f"{g.stem}.csv")
    for j in np.flatnonzero((D["rt"] >= g.t0) & (D["rt"] <= g.t1)):
        print(f"  rt {D['rt'][j]:7.2f} area {D['area'][j]:.2e} COM {T['COM'].iloc[j]} R2 {T['R2'].iloc[j]:.3f} SNR {T['SNR'].iloc[j]:5.1f} | {top_ions(D['spec'][j], masses)}")
    comp = pd.read_csv(R / g.mosa / g.set / "gc/components" / f"{g.stem}.tsv", sep="\t")
    print("Mosa components:")
    own = np.zeros_like(seg)
    for c in np.flatnonzero((A["rt"] >= g.t0 - 3) & (A["rt"] <= g.t1 + 3)):
        lo, p = A["profs"][c]
        x0, x1 = max(lo, a), min(lo + len(p), b)
        if x1 > x0:
            own[x0 - a:x1 - a] += np.outer(p[x0 - lo:x1 - lo], A["spec"][c])
        if g.t0 <= A["rt"][c] <= g.t1:
            r = comp.iloc[c]
            print(f"  #{c} rt {A['rt'][c]:7.2f} [{rt[int(r.scan_min)]:.1f}-{rt[int(r.scan_max)]:.1f}] area {A['area'][c]:.2e} feats {int(r.n_features)} loose {int(r.n_loose)} "
                  f"delayed {int(r.n_delayed)} ions {int(r.n_ions)} shared {int(r.n_shared_ions)} model_r {r.model_r:.2f} tol {r.apex_tol_s:.2f}s | {top_ions(A['spec'][c], masses)}")
    print(f"Mosa own model on the window: R2 {r2(seg, own):.3f};  per ion (data max, model max, R2):")
    for m in ions:
        y, f = seg[:, m - masses[0]], own[:, m - masses[0]]
        print(f"  m/z {m:3d} data {y.max():9.0f} model {f.max():9.0f} R2 {r2(y, f):6.2f}   model %:", (100 * f / max(y.max(), 1)).round(0).astype(int))
    F = pd.read_csv(R / g.mosa / g.set / "features" / f"{g.stem}.tsv", sep="\t")
    FC = pd.read_csv(R / g.mosa / g.set / "gc/components" / f"{g.stem}.features.tsv", sep="\t")
    F = F.merge(FC, on="feature_id")
    F["nom"] = unit_mass(F["mz"].values, C13)
    w = F[(F["rt"] * 60 >= g.t0) & (F["rt"] * 60 <= g.t1) & F["nom"].isin(ions)].sort_values(["nom", "rt"])
    print("Mosa picked features of these ions (rt s, bounds, height, S/N, component, model_r, shared, flags):")
    for _, f in w.iterrows():
        print(f"  m/z {int(f.nom):3d} rt {f.rt * 60:7.2f} [{f.rt_min * 60:.1f}-{f.rt_max * 60:.1f}] h {f.height:9.0f} snr {f.snr:6.1f} comp {int(f.component_id):4d} r {f.model_r:.2f} "
              f"shared {int(f.shared)} {f.flags if isinstance(f.flags, str) else ''}")
    rj = R / g.mosa / g.set / "features" / f"{g.stem}.rejected.tsv"
    if rj.exists():
        J = pd.read_csv(rj, sep="\t", low_memory=False)
        J["nom"] = unit_mass(J["mz"].values, C13)
        w = J[(J["rt"] * 60 >= g.t0) & (J["rt"] * 60 <= g.t1) & J["nom"].isin(ions)].sort_values(["nom", "rt"])
        print("Mosa rejected candidates of these ions:")
        for _, f in w.iterrows():
            print(f"  m/z {int(f.nom):3d} rt {f.rt * 60:7.2f} [{f.rt_min * 60:.1f}-{f.rt_max * 60:.1f}] h {f.height:9.0f} snr {f.snr:6.1f} gate {f.gate} reason {f.reason} flags {f.flags if isinstance(f.flags, str) else ''}")


if __name__ == "__main__":
    main()
