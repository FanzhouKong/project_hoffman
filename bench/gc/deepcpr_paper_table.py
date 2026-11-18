#!/usr/bin/env python3
"""Mosa beside the figures of Yu et al. (Anal. Chem. 2026, 98, 32686), computed the paper's way on its two
standard-mixture sets.

The paper scores each target compound at each level: R2 = explained variance of the resolved model against the
measured data of the segment the compound lies in, r = correlation of its resolved peak area with the nominal
level over the five runs; then means over compounds (Figure 3a, b), over the compound x level cases with a
wavelet S/N of 3-10 (Figure 3c), and over the 15 FAMEs in six co-elution clusters (Figure 3d).

Here the targets are taken from a DeepCPR run of the same files (the paper's compound tables are in its
Supporting Information): DeepCPR components matched across the five files whose area rises with the level
(log-log slope >= 0.5). For every target and file:
  DeepCPR  R2 = the R2 it writes for the component (single/<file>.csv), S/N and COM (components in the segment)
           from the same row;
  Mosa     the component with the same spectrum at that time (cosine >= --min-cos on square-root intensities of
           the ions above 1 % of the base peak, within --rt-tol s); R2 = Mosa's own model (profile x spectrum of
           all its components) against the data of DeepCPR's segment, on this scorer's baseline. A target without
           such a component is "not found": it has no area, and the R2 of its segment still counts.
r for Mosa needs the compound at all five levels.

    python bench/gc/deepcpr_paper_table.py --raw DIR --levels stem=level,... --deepcpr DIR --mosa DIR --out FILE.tsv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.gc.deepcpr_score import (cosine, deepcpr_mass_offset, load_deepcpr, load_mosa, match_series,   # noqa: E402
                                    observed, root_spectra, window_scores)


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if np.all(np.isfinite(y)) and np.std(y) > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--levels", required=True)
    ap.add_argument("--deepcpr", type=Path, required=True)
    ap.add_argument("--mosa", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rt-tol", type=float, default=1.5)
    ap.add_argument("--series-rt-tol", type=float, default=3.0)
    ap.add_argument("--min-cos", type=float, default=0.7)
    g = ap.parse_args()
    levels = {k: float(v) for k, v in (p.split("=") for p in g.levels.split(","))}
    stems = sorted(levels, key=lambda s: levels[s])
    x = np.array([levels[s] for s in stems])
    obs = {s: observed(g.raw / f"{s}.CDF") for s in stems}
    lo, hi = min(o[1][0] for o in obs.values()), max(o[1][-1] for o in obs.values())
    for s, (rt, masses, Xb, wins) in list(obs.items()):
        full = np.zeros((Xb.shape[0], hi - lo + 1))
        full[:, masses[0] - lo:masses[-1] - lo + 1] = Xb
        obs[s] = (rt, np.arange(lo, hi + 1), full, wins)
    rt, masses, Xb, _ = obs[stems[-1]]
    off = deepcpr_mass_offset(g.deepcpr, stems[-1], masses, rt, Xb)
    D, Mo, tab, segs = {}, {}, {}, {}
    for s in stems:
        rt, masses, Xb, _ = obs[s]
        D[s] = load_deepcpr(g.deepcpr, s, masses, rt, mass_offset=off)
        Mo[s] = load_mosa(g.mosa, s, masses, rt)
        tab[s] = pd.read_csv(g.deepcpr / "single" / f"{s}.csv")
        G = pd.read_csv(g.deepcpr / "seg" / f"{s}.csv")
        s0 = np.searchsorted(rt, G["RT_st"].values * 60.0 - 1e-6)
        s1 = np.minimum(np.searchsorted(rt, G["RT_ed"].values * 60.0 - 1e-6), len(rt))
        wm = window_scores(rt, Xb, list(zip(s0, np.maximum(s1, s0 + 3))), Mo[s])
        wd = window_scores(rt, Xb, list(zip(s0, np.maximum(s1, s0 + 3))), D[s])
        segs[s] = dict(st=G["RT_st"].values * 60.0, ed=G["RT_ed"].values * 60.0, mosa_own=wm["r2_own"].values,
                       mosa_spec=wm["r2_spectra"].values, deep_spec=wd["r2_spectra"].values)
    rows = []
    for k, row in enumerate(match_series(D, stems, stems[-1], g.series_rt_tol, g.min_cos)):
        if len(row) < len(stems):
            continue
        y = np.array([D[s]["area"][row[s]] for s in stems])
        if np.any(y <= 0) or np.polyfit(np.log(x), np.log(y), 1)[0] < 0.5:
            continue
        ym = np.full(len(stems), np.nan)
        for q, s in enumerate(stems):
            j = row[s]
            t = D[s]["rt"][j]
            seg = int(np.argmin(np.where((segs[s]["st"] - 0.5 <= t) & (t <= segs[s]["ed"] + 0.5), 0.0, 1.0)
                                + 1e-9 * np.abs(0.5 * (segs[s]["st"] + segs[s]["ed"]) - t)))
            A = Mo[s]
            near = np.flatnonzero(np.abs(A["rt"] - t) <= g.rt_tol)
            ra, rb = root_spectra(A["spec"][near]), root_spectra(D[s]["spec"][j:j + 1])[0]
            cs = np.array([cosine(a, rb) for a in ra])
            found = len(cs) > 0 and cs.max() >= g.min_cos
            if found:
                ym[q] = A["area"][near[int(np.argmax(cs))]]
            rows.append(dict(target=k, rt=round(float(D[stems[-1]]["rt"][row[stems[-1]]]), 2), file=s, level=levels[s],
                             base_mz=int(masses[np.argmax(D[s]["spec"][j])]),
                             com=int(tab[s]["COM"].iloc[j]), snr=float(tab[s]["SNR"].iloc[j]),
                             r2_deepcpr=float(tab[s]["R2"].iloc[j]), r2_deepcpr_this_scorer=float(segs[s]["deep_spec"][seg]),
                             r2_mosa=float(segs[s]["mosa_own"][seg]), r2_mosa_spectra=float(segs[s]["mosa_spec"][seg]),
                             mosa_found=bool(found), mosa_cos=float(cs.max()) if len(cs) else 0.0,
                             area_deepcpr=float(y[q]), area_mosa=float(ym[q])))
        for r_ in rows[-len(stems):]:
            r_["r_deepcpr"] = pearson(x, y)
            r_["r_mosa"] = pearson(x, ym)
    T = pd.DataFrame(rows)
    g.out.parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(g.out, sep="\t", index=False, float_format="%.5g")
    per = T.groupby("target").agg(com=("com", "max"), r_deepcpr=("r_deepcpr", "first"), r_mosa=("r_mosa", "first"),
                                  found=("mosa_found", "sum"))

    def line(name, cases, comp):
        if not len(cases):
            return f"{name}: none"
        return (f"{name}: {comp['r_deepcpr'].size} compounds, {len(cases)} compound x level cases | "
                f"R2 DeepCPR {cases['r2_deepcpr'].mean():.3f} (this scorer's measure {cases['r2_deepcpr_this_scorer'].mean():.3f}), "
                f"Mosa own model {cases['r2_mosa'].mean():.3f} (median {cases['r2_mosa'].median():.3f}), "
                f"Mosa spectra refit {cases['r2_mosa_spectra'].mean():.3f} | Mosa found {int(cases['mosa_found'].sum())}/{len(cases)} cases, "
                f"{int((comp['found'] == len(stems)).sum())}/{len(comp)} compounds at all levels | "
                f"r DeepCPR {comp['r_deepcpr'].mean():.3f}, Mosa {comp['r_mosa'].mean():.3f} "
                f"(on the {int(comp['r_mosa'].notna().sum())} it has at all levels; DeepCPR on those "
                f"{comp.loc[comp['r_mosa'].notna(), 'r_deepcpr'].mean():.3f})")

    print(line("all targets", T, per))
    low = T[(T["snr"] >= 3) & (T["snr"] <= 10)]
    print(line("S/N 3-10 cases", low, per.loc[sorted(set(low["target"]))]))
    clu = per[per["com"] >= 2]
    print(line("co-eluting (>= 2 components in DeepCPR's segment)", T[T["target"].isin(clu.index)], clu))
    iso = per[per["com"] < 2]
    print(line("alone in their segment", T[T["target"].isin(iso.index)], iso))


if __name__ == "__main__":
    main()
