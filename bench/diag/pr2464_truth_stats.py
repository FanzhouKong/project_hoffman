#!/usr/bin/env python3
"""Summary statistics of the PR002464 ground truth (spiked standards, credentialed yeast, plasma dilution/mixing) for
the three Orbitrap studies, for reporting. Reads only data/truth/PR002464/<study>/ and the design export; writes
results/scores/pr2464_truth_stats.md and .tsv (one row per statistic, one column per study) and .json (the table plus
the log10 height histograms of each truth set and the reasons spike peaks fall into tier B, for the methods page).

Overlap between the three truth sets: two peaks are the same when within 5 ppm and 3 s (the truth builders' own
tolerances); "distinct truth peaks" counts connected groups over all three sets.
usage: pr2464_truth_stats.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRUTH = ROOT / "data/truth/PR002464"
OUT = ROOT / "results/scores"
STUDIES = {"ST004500": "RP neg", "ST004501": "HILIC pos", "ST004502": "HILIC neg"}
PPM, RT_S = 5.0, 3.0
HBINS = np.round(np.arange(2.5, 8.01, 0.25), 2)   # log10 height bin edges for the histograms


def n_pct(k, n):
    return f"{k:,} ({100 * k / n:.0f} %)" if n else "0"


def span(a, b, fmt="{:.2f}"):
    """a range: en dash between non-negative ends, "to" when either end is negative"""
    return f"{fmt.format(a)}{' to ' if min(a, b) < 0 else '–'}{fmt.format(b)}"


def med_range(x, fmt="{:.2f}"):
    x = pd.Series(x).dropna()
    return f"{fmt.format(x.median())} ({span(x.min(), x.max(), fmt)})" if len(x) else "-"


def iqr(x, fmt="{:.2f}"):
    x = pd.Series(x).dropna()
    return f"{fmt.format(x.median())} [{span(x.quantile(.25), x.quantile(.75), fmt)}]" if len(x) else "-"


def sci(v):
    return f"{v:.1e}".replace("e+0", "e").replace("e+", "e")


def pairs(mz, rt):
    """index pairs (i, j) within PPM and RT_S (rt in minutes)"""
    o = np.argsort(mz)
    m, t = mz[o], rt[o]
    out = []
    for a in range(len(m)):
        b = a + 1
        while b < len(m) and m[b] - m[a] <= m[a] * PPM * 1e-6:
            if abs(t[b] - t[a]) * 60 <= RT_S:
                out.append((o[a], o[b]))
            b += 1
    return out


def components(n, edges):
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in edges:
        parent[find(i)] = find(j)
    return np.array([find(i) for i in range(n)])


def tier_b_reasons(sp):
    """why spike-derived peaks are tier B: the builder's signature test (spiketruth.py) replayed per peak"""
    b = sp[sp["tier"] == "B"]
    r, fc = b["log2_18_19"], b["log2FC"]
    weak = b["weak_at_noise"].astype(str).eq("True")
    sig = np.where(weak, (np.sign(r) == np.sign(fc)) & (r.abs() >= fc.abs() - 0.75), (r - fc).abs() <= 0.75)
    fit = (b["e_fit_resid"] <= 0.3).values
    return {"ratio_off": int((~sig).sum()), "fit_fails": int((sig & ~fit).sum()),
            "no_primary": int((sig & fit).sum()), "coelution": int((b["source"] == "coelution").sum()),
            "total": len(b)}


def study_stats(st):
    d = TRUTH / st
    files = pd.read_csv(d / "files.tsv", sep="\t")
    sp = pd.read_csv(d / "spiked_truth.tsv", sep="\t", keep_default_na=False, na_values=[""])
    ss = pd.read_csv(d / "standards_summary.tsv", sep="\t")
    ye = pd.read_csv(d / "yeast_truth.tsv", sep="\t")
    yq = pd.read_csv(d / "yeast_cred/decoy_qc.tsv", sep="\t").set_index("search")
    yk = pd.read_csv(d / "yeast_cred/known_metabolite_check.tsv", sep="\t")
    seeds_y = re.search(r"seeds tested: (\d+)", (d / "yeast_cred/qc_report.md").read_text())
    pl = pd.read_csv(d / "plasma_truth.tsv", sep="\t", keep_default_na=False, na_values=[""])
    pn = pd.read_csv(d / "plasma_null_qc.tsv", sep="\t")
    pk = pd.read_csv(d / "plasma_known_metabolites.tsv", sep="\t")
    pp = json.loads((d / "plasma_params.json").read_text())
    r = {}

    # --- study
    r["Study: MS1 injections"] = f"{len(files)}"
    r["Study: injections by type"] = "; ".join(f"{k} {v}" for k, v in files["sample_type"].value_counts().items())
    r["Study: vials x batches"] = f"{files['vial'].nunique()} x {files['batch'].nunique()}"

    # --- spiked standards
    A = sp["tier"] == "A"
    flags = sp["flags"].fillna("")
    r["Spike: truth peaks"] = f"{len(sp):,}"
    r["Spike: tier A / tier B"] = f"{A.sum():,} / {(~A).sum():,}"
    found = ss[ss["status"] == "found"]   # the builder's call: the standard has a tier-A peak group
    r["Spike: standards found (of 96; >= 1 tier-A peak)"] = f"{len(found)}"
    r["Spike: tier-A peaks per found standard, median (range)"] = med_range(found["n_tierA"].values, "{:.0f}")
    for cls in ["formula_ion", "isotope", "adduct", "fragment", "unannotated"]:
        r[f"Spike: class {cls}"] = n_pct(int((sp["annotation_class"] == cls).sum()), len(sp))
    r["Spike: flagged ft_artifact? (sideband candidate)"] = n_pct(int(flags.str.contains("ft_artifact").sum()), len(sp))
    r["Spike: flagged broad"] = n_pct(int(flags.str.contains("broad").sum()), len(sp))
    r["Spike: weak side at noise (one-sided ratio)"] = n_pct(int(sp["weak_at_noise"].astype(str).eq("True").sum()), len(sp))
    fi = sp["annotation_class"] == "formula_ion"
    r["Spike: formula ions |m/z error| ppm, median [IQR]"] = iqr(sp.loc[fi, "mz_err_ppm"].abs())
    ok = A & ~sp["weak_at_noise"].astype(str).eq("True")
    r["Spike: tier A |observed - designed log2(A/B)|, median [IQR]"] = iqr(
        (sp.loc[ok, "log2_18_19"] - sp.loc[ok, "log2FC"]).abs())
    r["Spike: tier A dilution slope (vials 9-12), median [IQR]"] = iqr(sp.loc[A, "c_dilution_slope"])
    r["Spike: designed |log2(A/B)| of truth peaks: 1 / 2 / 3 / 4"] = " / ".join(
        f"{int((sp['log2FC'].abs() == k).sum()):,}" for k in (1, 2, 3, 4))
    r["Spike: height (dominant vial), median [IQR]"] = iqr(sp["height_dom"], "{:.2e}")
    r["Spike: height < 1e5"] = n_pct(int((sp["height_dom"] < 1e5).sum()), len(sp))
    r["Spike: FWHM s, median [IQR]"] = iqr(sp["fwhm_s"], "{:.1f}")
    r["Spike: m/z range"] = span(sp["mz_obs"].min(), sp["mz_obs"].max(), "{:.1f}")
    r["Spike: RT range min"] = span(sp["rt_apex"].min(), sp["rt_apex"].max())

    # --- yeast credentialing
    Ay = ye["tier"] == "A"
    conf = ye["confirmed"].astype(str).eq("True")
    r["Yeast: seeds tested"] = f"{int(seeds_y.group(1)):,}" if seeds_y else "-"
    r["Yeast: credentialed peaks"] = f"{len(ye):,}"
    r["Yeast: tier A / tier B"] = f"{Ay.sum():,} / {(~Ay).sum():,}"
    r["Yeast: confirmed in 12C:13C 1:1 vial 25 and absent from blanks"] = n_pct(int(conf.sum()), len(ye))
    for cls in ["primary", "isotope_M+1", "isotope_M+2", "adduct_Na", "adduct_K", "adduct_NH4"]:
        r[f"Yeast: class {cls}"] = n_pct(int((ye["feature_class"] == cls).sum()), len(ye))
    r["Yeast: carbons (13C label count), median (range)"] = med_range(ye["n"], "{:.0f}")
    r["Yeast: 12C/13C shape correlation r, median [IQR]"] = iqr(ye["r"])
    t = yq.loc["target"]
    dec = yq.drop(index="target")
    r["Yeast: worst of 4 decoy searches (mass +-0.02 Da, RT +-1 min), tier A / B"] = \
        f"{dec['A'].max():,.0f} / {dec['B'].max():,.0f}"
    r["Yeast: estimated false rate tier A / tier B (worst decoy / target)"] = \
        f"{100 * t['est_false_rate_A']:.1f} % / {100 * t['est_false_rate_B']:.1f} %"
    agrees = yk["agrees"].astype(str).eq("True")
    r["Yeast: known metabolites matched / carbon count agrees"] = f"{yk['metabolite'].nunique()} / {int(agrees.sum())} of {len(yk)} matches"
    r["Yeast: 12C dilution slope (vials 5-8), median [IQR]"] = iqr(ye["dil_slope_12C"])
    r["Yeast: 12C height (vial 23), median [IQR]"] = iqr(ye["height12"], "{:.2e}")
    r["Yeast: 12C height < 1e5"] = n_pct(int((ye["height12"] < 1e5).sum()), len(ye))
    r["Yeast: m/z range"] = span(ye["mz"].min(), ye["mz"].max(), "{:.1f}")
    r["Yeast: RT range min"] = span(ye["rt"].min(), ye["rt"].max())

    # --- plasma
    Ap = pl["tier"] == "A"
    pf = pl["flags"].fillna("")
    r["Plasma: seeds tested"] = f"{pp['n_seeds']:,}"
    r["Plasma: truth peaks"] = f"{len(pl):,}"
    r["Plasma: tier A / tier B"] = f"{Ap.sum():,} / {(~Ap).sum():,}"
    for a in ["primary", "isotope M+1", "isotope M+2"]:
        r[f"Plasma: annotation {a}"] = n_pct(int((pl["annotation"] == a).sum()), len(pl))
    r["Plasma: flagged ft_artifact? / broad"] = (f"{int(pf.str.contains('ft_artifact').sum()):,} / "
                                                 f"{int(pf.str.contains('broad').sum()):,}")
    r["Plasma: dilution slope (vials 1-4), median [IQR]"] = iqr(pl["dil_slope"])
    r["Plasma: dilution R^2, median [IQR]"] = iqr(pl["dil_r2"], "{:.3f}")
    r["Plasma: log2(SRM 1950 / SRM 8231), median [IQR]"] = iqr(pl["log2_1950_8231"])
    r["Plasma: |log2(SRM 1950 / SRM 8231)| >= 1"] = n_pct(int((pl["log2_1950_8231"].abs() >= 1).sum()), len(pl))
    true = pn[pn["order"] == "true"].iloc[0]
    perm = pn[pn["order"] != "true"]
    r["Plasma: permuted-dilution passes A / A+B (max over permutations)"] = \
        f"{perm['tier_A'].max():,} / {perm['tier_AB'].max():,} (vs {true['tier_A']:,} / {true['tier_AB']:,})"
    r["Plasma: known metabolites matched"] = f"{pk['metabolite'].nunique()}"
    r["Plasma: also seen in yeast-only / spike-only vials"] = (
        f"{int(pl['also_in_yeast'].astype(str).eq('True').sum()):,} / "
        f"{int(pl['also_in_spike'].astype(str).eq('True').sum()):,}")
    r["Plasma: height (vial 17), median [IQR]"] = iqr(pl["height_17"], "{:.2e}")
    r["Plasma: height < 1e5"] = n_pct(int((pl["height_17"] < 1e5).sum()), len(pl))
    r["Plasma: m/z range"] = span(pl["mz_obs"].min(), pl["mz_obs"].max(), "{:.1f}")
    r["Plasma: RT range min"] = span(pl["rt_apex"].min(), pl["rt_apex"].max())

    # --- overlap between the three sets (5 ppm, 3 s)
    sets = {"spike": (sp["mz_obs"].fillna(sp["mz"]).values, sp["rt_apex"].values),
            "yeast": (ye["mz"].values, ye["rt"].values),
            "plasma": (pl["mz_obs"].values, pl["rt_apex"].values)}
    lab = np.concatenate([[k] * len(v[0]) for k, v in sets.items()])
    mz = np.concatenate([v[0] for v in sets.values()])
    rt = np.concatenate([v[1] for v in sets.values()])
    e = pairs(mz, rt)
    comp = components(len(mz), e)
    r["All: truth peaks summed over the three sets"] = f"{len(mz):,}"
    r["All: distinct truth peaks (5 ppm, 3 s)"] = f"{len(set(comp)):,}"
    member = pd.DataFrame({"c": comp, "s": lab}).drop_duplicates()
    k = member.groupby("c")["s"].apply(lambda s: "+".join(sorted(s)))
    for combo in ["plasma+spike", "plasma+yeast", "spike+yeast", "plasma+spike+yeast"]:
        r[f"All: distinct peaks shared {combo}"] = f"{int((k == combo).sum()):,}"
    hist = {name: np.histogram(np.log10(np.clip(h, 10 ** HBINS[0], 10 ** HBINS[-1] - 1)), HBINS)[0].tolist()
            for name, h in (("spike", sp["height_dom"]), ("yeast", ye["height12"]), ("plasma", pl["height_17"]))}
    return r, hist, tier_b_reasons(sp)


def main():
    cols, hists, tierb = {}, {}, {}
    for st, lc in STUDIES.items():
        cols[f"{st} ({lc})"], hists[st], tierb[st] = study_stats(st)
        print(f"[{st}] done", flush=True)
    df = pd.DataFrame(cols)
    df.index.name = "statistic"
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "pr2464_truth_stats.tsv", sep="\t")
    # across studies: standards found
    found = {}
    for st in STUDIES:
        ss = pd.read_csv(TRUTH / st / "standards_summary.tsv", sep="\t")
        found[st] = set(ss.loc[ss["status"] == "found", "std"])
    anyf = set.union(*found.values())
    allf = set.intersection(*found.values())
    lines = ["# PR002464 ground truth: summary statistics", "",
             "Orbitrap ID-X studies of Metabolomics Workbench PR002464 (ST004504, RP pos, excluded: file contents do not "
             "match the deposited sample metadata). Built by bench/spiketruth.py, bench/credtruth.py and "
             "bench/biotruth.py; this table by bench/diag/pr2464_truth_stats.py.", "",
             f"Standards (96 spiked in 24 mixtures): {len(anyf)} found (a tier-A peak group) in at least one study, "
             f"{len(allf)} in all three.", "",
             ""]
    try:
        md = df.to_markdown()
    except ImportError:
        md = "| statistic | " + " | ".join(df.columns) + " |\n|" + "---|" * (len(df.columns) + 1) + "\n" + \
             "\n".join(f"| {i} | " + " | ".join(str(v) for v in row) + " |" for i, row in df.iterrows())
    lines[-1] = md
    (OUT / "pr2464_truth_stats.md").write_text("\n".join(lines) + "\n")
    (OUT / "pr2464_truth_stats.json").write_text(json.dumps(dict(
        studies=STUDIES, stats={c: df[c].to_dict() for c in df.columns}, height_bins_log10=HBINS.tolist(),
        height_hist=hists, tier_b=tierb, standards=dict(found_any=len(anyf), found_all=len(allf),
                                                       found={s: len(found[s]) for s in STUDIES})), indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
