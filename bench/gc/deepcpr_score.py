#!/usr/bin/env python3
"""Deconvolution benchmark on the DeepCPR example runs, in the manner of Yu et al. (Anal. Chem. 2026,
doi 10.1021/acs.analchem.6c04440): a standard mixture injected at five levels, unit-resolution GC-MS.

The paper reports, per tool, the explained variance (R2) of the resolved model and the correlation (r) of resolved
peak area with concentration. This scorer computes both in one way for every tool, from what each tool wrote:

  components of a file = apex time, area, spectrum at nominal masses (+ the chromatographic profile when exported)

  observed data   X: scans x nominal masses from the CDF (bench/gc/cdf_to_mzml.unit_matrix), minus a baseline per mass
                  (rolling minimum of the 5-scan median over --baseline-s, smoothed, raised to the middle of the
                  noise; nothing clipped): the same X for every tool.
  windows         stretches of the run where the baseline-corrected total ion current is above 5 sd of its
                  scan-to-scan noise, widened by 3 scans; found from the data, not from any tool.
  R2 spectra      per window, the scans are fitted with the spectra of the tool's components whose apex lies in the
                  window (non-negative least squares, one scan at a time); R2 = 1 - var(X - fit) / var(X), the
                  definition in DeepCPR's code (_reconstruction_r2). A window without a component scores 0.
                  This asks only for spectra, so it is the same test for every tool; it is generous (the profiles
                  are free), and more components always fit better: read it with the component count.
  R2 own          the same with the tool's own profiles and abundances (X_fit = sum profile x spectrum); only for
                  tools that export profiles (Mosa). DeepCPR's own figure is the R2 column it writes per segment.
  linearity       components are matched across the five files (apex within --rt-tol seconds, spectrum cosine at
                  least --min-cos on square-root intensities of the ions above 1 % of the base peak, greedy from the largest
                  area in the reference file); r = Pearson correlation of
                  area with the level, for every series found at all five levels. Constant compounds (column
                  bleed, reagent peaks) are series too: "responding" series have a log-log slope of area on level
                  of at least 0.5, and r is summarised over those.
  segments        with --segments PATH (a DeepCPR result folder) both R2 are also computed on the segments DeepCPR
                  itself cut and resolved (seg/<file>.csv), next to the R2 it reports for each: its own figure on
                  its own baseline (airPLS), so the two are close relatives, not the same number.
  library         with --library (an EI .msp, read at nominal masses) every component is searched with peak3d's own
                  scores (peak3d/gc/library.py: weights intensity^0.6 x m/z^3, match and reverse); the best hit per
                  component is written, and the summary counts the compounds named at match >= 0.7.

    python -I bench/gc/deepcpr_score.py --raw DIR --levels LMJ-05=5,LMJ-10=10,... --out DIR \
        --tool mosa=mosa:RESULTS/mosa/aa --tool deepcpr=deepcpr:RESULTS/deepcpr/aa
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, minimum_filter1d, uniform_filter1d
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.gc.cdf_to_mzml import C13, read_cdf, unit_mass, unit_matrix      # noqa: E402


# ---------------------------------------------------------------- data ------
def observed(cdf, baseline_s=60.0):
    """(rt seconds, nominal masses, baseline-corrected matrix, windows [(s0, s1)])"""
    rt, off, mz, it = read_cdf(cdf)
    masses, X = unit_matrix(rt, off, mz, it, clip=False)       # noise stays symmetric: the baseline takes its mean
    dt = float(np.median(np.diff(rt)))
    w = max(5, int(round(baseline_s / dt)) | 1)
    sm = median_filter(X, size=(5, 1), mode="nearest")
    base = uniform_filter1d(minimum_filter1d(sm, size=w, axis=0, mode="nearest"), size=w, axis=0, mode="nearest")
    Xb = X - base
    Xb -= np.median(Xb, axis=0)            # the rolling minimum sits below the noise: back to its middle
    tic = Xb.sum(axis=1)
    d = np.diff(tic)
    sd = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)
    above = tic > 5.0 * sd + np.median(tic)
    wins = []
    s = 0
    S = len(tic)
    while s < S:
        if above[s]:
            e = s
            while e + 1 < S and (above[e + 1] or above[min(e + 2, S - 1)] or above[min(e + 3, S - 1)]):
                e += 1
            a, b = max(0, s - 3), min(S, e + 4)
            if wins and a <= wins[-1][1]:
                wins[-1] = (wins[-1][0], b)
            elif e - s + 1 >= 3:
                wins.append((a, b))
            s = e + 1
        else:
            s += 1
    return rt, masses, Xb, wins


# ---------------------------------------------------------------- tools -----
def load_mosa(out, stem, masses, rt, scale=C13):
    """components of a peak3d gc output: area = sum of the spectrum x sum of the model profile x scan spacing"""
    out = Path(out)
    c = pd.read_csv(out / "gc" / "components" / f"{stem}.tsv", sep="\t")
    z = np.load(out / "gc" / "components" / f"{stem}.npz")
    K, M = len(c), len(masses)
    spec = np.zeros((K, M))
    nom = unit_mass(z["mz"], scale) - masses[0]
    profs = []
    dt = float(np.median(np.diff(rt)))
    area = np.zeros(K)
    has_model = "model" in z.files
    for k in range(K):
        a, b = int(z["off"][k]), int(z["off"][k + 1])
        j = nom[a:b]
        ok = (j >= 0) & (j < M)
        np.add.at(spec[k], j[ok], z["abundance"][a:b][ok])
        if has_model:
            p = z["model"][int(z["model_off"][k]):int(z["model_off"][k + 1])].astype(np.float64)
            profs.append((int(c["scan_min"].iloc[k]), p))
            area[k] = spec[k].sum() * p.sum() * dt
        else:
            area[k] = float(c["tic"].iloc[k])
    return dict(rt=c["rt"].values * 60.0, area=area, spec=spec, profs=profs if has_model else None,
                quant=c["quant_height"].values.astype(float), own_r2=None)


def deepcpr_mass_offset(out, stem, masses, rt, Xb):
    """the .msp files number the masses from 1 by matrix column: the offset (0 or 1) to the nominal masses is the
    one at which the 25 largest components resemble the raw scans at their apex"""
    best, best_cos = 0, -1.0
    for off in (0, 1):
        c = load_deepcpr(out, stem, masses, rt, mass_offset=off)
        cs = []
        for k in np.argsort(-c["area"])[:25]:
            s = int(np.clip(np.searchsorted(rt, c["rt"][k]), 1, len(rt) - 2))
            cs.append(cosine(Xb[s - 1:s + 2].sum(axis=0), c["spec"][k]))
        m = float(np.median(cs)) if cs else 0.0
        if m > best_cos:
            best, best_cos = off, m
    return best


def load_deepcpr(out, stem, masses, rt, mass_offset=0):
    """components of a DeepCPR result folder (single/<stem>.csv + ms/<stem>/<i>.msp, i in table order)"""
    out = Path(out)
    f = out / "single" / f"{stem}.csv"
    try:
        c = pd.read_csv(f)
    except pd.errors.EmptyDataError:
        c = pd.DataFrame(columns=["rt", "peak area", "R2"])
    K, M = len(c), len(masses)
    spec = np.zeros((K, M))
    for k in range(K):
        vals = []
        for line in (out / "ms" / stem / f"{k}.msp").read_text().splitlines():
            p = line.split()
            if len(p) == 2 and p[0][0].isdigit():
                vals.append((int(float(p[0])), float(p[1])))
        for m, v in vals:
            j = m - mass_offset - masses[0]
            if 0 <= j < M and v > 0:
                spec[k, j] = v
    return dict(rt=c["rt"].values.astype(float) * 60.0, area=c["peak area"].values.astype(float), spec=spec, profs=None,
                quant=None, own_r2=c["R2"].values.astype(float) if "R2" in c else None)


LOADERS = {"mosa": load_mosa, "deepcpr": load_deepcpr}


# ---------------------------------------------------------------- scores ----
def r2(obs, fit):
    v = np.var(obs, ddof=1)
    return float(1.0 - np.var(fit - obs, ddof=1) / v) if v > 0 else float("nan")


def window_scores(rt, Xb, wins, comp):
    """per window: R2 with free profiles (spectra only), R2 with the tool's own profiles, components"""
    rows = []
    apex = np.searchsorted(rt, comp["rt"])
    for s0, s1 in wins:
        obs = Xb[s0:s1]
        k = np.flatnonzero((apex >= s0) & (apex < s1))
        row = dict(s0=s0, s1=s1, rt0=rt[s0], rt1=rt[s1 - 1], n_comp=len(k), signal=float((obs ** 2).sum()))
        if len(k) == 0:
            row.update(r2_spectra=0.0, r2_own=0.0 if comp["profs"] is not None else np.nan)
            rows.append(row)
            continue
        Sp = comp["spec"][k]
        nz = Sp.sum(axis=1) > 0
        Sp = Sp[nz]
        fit = np.zeros_like(obs)
        if len(Sp):
            A = Sp.T
            for s in range(obs.shape[0]):
                coef, _ = nnls(A, obs[s])
                fit[s] = A @ coef
        row["r2_spectra"] = r2(obs, fit)
        if comp["profs"] is not None:
            own = np.zeros_like(obs)
            for c in range(len(comp["rt"])):          # every component reaching into the window, not only its own
                lo, p = comp["profs"][c]
                a, b = max(lo, s0), min(lo + len(p), s1)
                if b > a:
                    own[a - s0:b - s0] += np.outer(p[a - lo:b - lo], comp["spec"][c])
            row["r2_own"] = r2(obs, own)
        else:
            row["r2_own"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def root_spectra(spec, min_rel=0.01):
    """square roots of the spectra (rows), ions below min_rel of the row's base peak left out: MCR spectra hold a
    value at every mass, and under a square root the hundreds of noise-level ones would outweigh the real ions"""
    top = spec.max(axis=1, keepdims=True) if spec.size else np.zeros((len(spec), 1))
    return np.sqrt(np.where(spec >= min_rel * top, spec, 0.0))


def match_series(comps, stems, ref, rt_tol, min_cos):
    """components matched across files: rows = series (from the reference file, largest area first), columns = files"""
    R = comps[ref]
    order = np.argsort(-R["area"])
    used = {s: np.zeros(len(comps[s]["rt"]), bool) for s in stems}
    root = {s: root_spectra(comps[s]["spec"]) for s in stems}
    rows = []
    for i in order:
        if R["area"][i] <= 0:
            continue
        row = {ref: int(i)}
        for s in stems:
            if s == ref:
                continue
            C = comps[s]
            cand = np.flatnonzero((np.abs(C["rt"] - R["rt"][i]) <= rt_tol) & ~used[s])
            best, bc = -1, min_cos
            for j in cand:
                cs = cosine(root[ref][i], root[s][j])
                if cs >= bc:
                    best, bc = int(j), cs
            if best >= 0:
                row[s] = best
                used[s][best] = True
        rows.append(row)
    return rows


def linearity(comps, stems, levels, rt_tol, min_cos, key="area"):
    ref = max(stems, key=lambda s: levels[s])
    x = np.array([levels[s] for s in stems], float)
    out = []
    for row in match_series(comps, stems, ref, rt_tol, min_cos):
        y = np.array([comps[s][key][row[s]] if s in row else np.nan for s in stems], float)
        n = int(np.isfinite(y).sum())
        rec = dict(rt=float(comps[ref]["rt"][row[ref]]), n_levels=n, area_ref=float(comps[ref]["area"][row[ref]]),
                   ref_component=int(row[ref]), base_mz=int(np.argmax(comps[ref]["spec"][row[ref]])), r=np.nan,
                   slope_loglog=np.nan)
        rec.update({f"y_{s}": v for s, v in zip(stems, y)})
        if n == len(stems) and np.all(y > 0):
            rec["r"] = float(np.corrcoef(x, y)[0, 1])
            rec["slope_loglog"] = float(np.polyfit(np.log(x), np.log(y), 1)[0])
        out.append(rec)
    return pd.DataFrame(out)


def library_hits(comp, masses, prep, mz_range):
    """best library hit per component: DataFrame(component, name, inchikey, match, reverse)"""
    from peak3d.gc import library as L
    K = len(comp["rt"])
    offs, mzs, its = [0], [], []
    for k in range(K):
        j = np.flatnonzero(comp["spec"][k] > 0)
        mzs.append(masses[j].astype(float)); its.append(comp["spec"][k][j]); offs.append(offs[-1] + len(j))
    H = L.search(np.array(offs, np.int64), np.concatenate(mzs) if mzs else np.zeros(0),
                 np.concatenate(its) if its else np.zeros(0), prep, mz_range=mz_range)
    H = H[H["rank"] == 1].set_index("query")
    out = pd.DataFrame(dict(component=np.arange(K), rt=comp["rt"], area=comp["area"]))
    for c in ("name", "inchikey", "match", "reverse"):
        out[c] = H[c].reindex(out["component"]).values
    return out


def raw_dose(obs, stems, levels, n_peaks=40, tol_s=3.0):
    """tool-free check of the levels: apex heights of the strongest single-mass peaks of the top-level file, read
    in every file within tol_s; median ratio to the top level per file"""
    ref = max(stems, key=lambda s: levels[s])
    rt, masses, Xb, _ = obs[ref]
    flat = np.argsort(-Xb, axis=None)
    picked = []
    for f in flat:
        s, j = np.unravel_index(f, Xb.shape)
        if all(abs(rt[s] - rt[a]) > 2 * tol_s for a, _ in picked):
            picked.append((s, j))
        if len(picked) >= n_peaks:
            break
    res = {}
    for st in stems:
        r2_, m2, X2, _ = obs[st]
        ratios = []
        for s, j in picked:
            a, b = np.searchsorted(r2_, [rt[s] - tol_s, rt[s] + tol_s])
            jj = masses[j] - m2[0]
            if 0 <= jj < X2.shape[1] and b > a:
                ratios.append(X2[a:b, jj].max() / Xb[s, j])
        res[st] = float(np.median(ratios))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="folder of the set's .CDF files")
    ap.add_argument("--levels", required=True, help="stem=level,... (relative concentrations)")
    ap.add_argument("--tool", action="append", required=True, metavar="NAME=KIND:PATH", help="kind: mosa or deepcpr")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rt-tol", type=float, default=3.0, help="seconds")
    ap.add_argument("--min-cos", type=float, default=0.7)
    ap.add_argument("--baseline-s", type=float, default=60.0)
    ap.add_argument("--segments", type=Path, default=None, help="DeepCPR result folder whose segments are scored too")
    ap.add_argument("--library", type=Path, default=None, help="EI .msp searched at nominal masses")
    ap.add_argument("--mass-scale", type=float, default=C13, help="--mass-scale of the mzML the Mosa run read")
    a = ap.parse_args()
    levels = {k: float(v) for k, v in (p.split("=") for p in a.levels.split(","))}
    stems = sorted(levels, key=lambda s: levels[s])
    a.out.mkdir(parents=True, exist_ok=True)
    obs = {s: observed(a.raw / f"{s}.CDF", a.baseline_s) for s in stems}
    lo, hi = min(o[1][0] for o in obs.values()), max(o[1][-1] for o in obs.values())
    for s, (rt, masses, Xb, wins) in list(obs.items()):       # one mass axis for the set
        full = np.zeros((Xb.shape[0], hi - lo + 1))
        full[:, masses[0] - lo:masses[-1] - lo + 1] = Xb
        obs[s] = (rt, np.arange(lo, hi + 1), full, wins)
    dose = raw_dose(obs, stems, levels)
    summary = dict(levels=levels, raw_height_ratio_to_top_level=dose, tools={},
                   windows={s: len(obs[s][3]) for s in stems})
    lines = []
    prep = None
    if a.library:
        import dataclasses
        from peak3d.gc import library as Lib
        prep = Lib.prepare(dataclasses.replace(Lib.load(a.library, a.out / ".library_cache"), nominal=True),
                           Lib.SearchParams(top_k=1))
    for spec in a.tool:
        name, rest = spec.split("=", 1)
        kind, path = rest.split(":", 1)
        comps = {}
        wtabs = []
        kw = dict(scale=a.mass_scale)
        if kind == "deepcpr":
            rt, masses, Xb, wins = obs[stems[-1]]
            kw = dict(mass_offset=deepcpr_mass_offset(path, stems[-1], masses, rt, Xb))
        for s in stems:
            rt, masses, Xb, wins = obs[s]
            comps[s] = LOADERS[kind](path, s, masses, rt, **kw)
            w = window_scores(rt, Xb, wins, comps[s])
            w.insert(0, "file", s)
            wtabs.append(w)
        W = pd.concat(wtabs, ignore_index=True)
        seg_summary = {}
        if a.segments is not None:
            stabs = []
            for s in stems:
                rt, masses, Xb, wins = obs[s]
                G = pd.read_csv(a.segments / "seg" / f"{s}.csv")
                s0 = np.searchsorted(rt, G["RT_st"].values * 60.0 - 1e-6)
                s1 = np.minimum(np.searchsorted(rt, G["RT_ed"].values * 60.0 - 1e-6), len(rt))
                ok = s1 - s0 >= 3
                w = window_scores(rt, Xb, list(zip(s0[ok], s1[ok])), comps[s])
                w.insert(0, "file", s)
                w["r2_reported"] = G["R2"].values[ok]
                stabs.append(w)
            G = pd.concat(stabs, ignore_index=True)
            G.to_csv(a.out / f"{name}.segments.tsv", sep="\t", index=False, float_format="%.5g")
            seg_summary = dict(segments=int(len(G)), seg_without_component=int((G["n_comp"] == 0).sum()),
                               seg_r2_spectra_mean=float(G["r2_spectra"].mean()),
                               seg_r2_spectra_median=float(G["r2_spectra"].median()),
                               seg_r2_own_mean=float(G["r2_own"].mean()) if G["r2_own"].notna().any() else None,
                               seg_r2_own_median=float(G["r2_own"].median()) if G["r2_own"].notna().any() else None,
                               seg_r2_reported_mean=float(G["r2_reported"].mean()),
                               seg_r2_reported_median=float(G["r2_reported"].median()))
        W.to_csv(a.out / f"{name}.windows.tsv", sep="\t", index=False, float_format="%.5g")
        L = linearity(comps, stems, levels, a.rt_tol, a.min_cos)
        L["base_mz"] += int(obs[stems[0]][1][0])
        L.to_csv(a.out / f"{name}.series.tsv", sep="\t", index=False, float_format="%.6g")
        full = L[L["n_levels"] == len(stems)]
        resp = full[full["slope_loglog"] >= 0.5]
        hit_summary = {}
        if prep is not None:
            hs = []
            for s in stems:
                rt, masses, Xb, wins = obs[s]
                h = library_hits(comps[s], masses, prep, (float(masses[0]), float(masses[-1])))
                h.insert(0, "file", s)
                hs.append(h)
            Hh = pd.concat(hs, ignore_index=True)
            Hh.to_csv(a.out / f"{name}.hits.tsv", sep="\t", index=False, float_format="%.6g")
            top = Hh[Hh["file"] == stems[-1]]
            good = top[top["match"] >= 0.7]
            hit_summary = dict(named_ge_0_7_top_level=int(len(good)),
                               distinct_compounds_ge_0_7_top_level=int(good["inchikey"].str[:14].nunique()),
                               named_ge_0_7_all_files=int((Hh["match"] >= 0.7).sum()),
                               match_median_named=float(good["match"].median()) if len(good) else None)
        own = np.concatenate([c["own_r2"] for c in comps.values() if c["own_r2"] is not None] or [np.zeros(0)])
        t = dict(
            components_per_file={s: int(len(comps[s]["rt"])) for s in stems},
            windows=int(len(W)), windows_without_component=int((W["n_comp"] == 0).sum()),
            r2_spectra_mean=float(W["r2_spectra"].mean()), r2_spectra_median=float(W["r2_spectra"].median()),
            r2_spectra_signal_weighted=float(np.average(W["r2_spectra"], weights=W["signal"])),
            r2_own_mean=float(W["r2_own"].mean()) if W["r2_own"].notna().any() else None,
            r2_own_median=float(W["r2_own"].median()) if W["r2_own"].notna().any() else None,
            r2_own_signal_weighted=float(np.average(W["r2_own"], weights=W["signal"])) if W["r2_own"].notna().all() else None,
            r2_reported_by_tool_mean=float(own.mean()) if len(own) else None,
            series_in_reference=int(len(L)), series_all_levels=int(len(full)),
            r_mean=float(full["r"].mean()) if len(full) else None,
            r_median=float(full["r"].median()) if len(full) else None,
            r_ge_0_99=int((full["r"] >= 0.99).sum()), r_ge_0_95=int((full["r"] >= 0.95).sum()),
            slope_loglog_median=float(full["slope_loglog"].median()) if len(full) else None,
            responding_series=int(len(resp)),
            responding_r_mean=float(resp["r"].mean()) if len(resp) else None,
            responding_r_median=float(resp["r"].median()) if len(resp) else None,
            responding_r_ge_0_99=int((resp["r"] >= 0.99).sum()),
            **hit_summary, **seg_summary,
        )
        summary["tools"][name] = t
        lines.append(f"{name}: comps/file {list(t['components_per_file'].values())}  windows {t['windows']} "
                     f"(none: {t['windows_without_component']})  R2 spectra mean {t['r2_spectra_mean']:.3f} "
                     f"median {t['r2_spectra_median']:.3f} weighted {t['r2_spectra_signal_weighted']:.3f}  "
                     f"R2 own mean {t['r2_own_mean']}  series {t['series_all_levels']}/{t['series_in_reference']} "
                     f"responding {t['responding_series']}: r mean {t['responding_r_mean']} "
                     f"median {t['responding_r_median']} >=0.99: {t['responding_r_ge_0_99']}  {hit_summary} {seg_summary}")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("raw height ratio to the top level:", {k: round(v, 3) for k, v in dose.items()})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
