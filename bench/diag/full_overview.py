# the full benchmark of chosen peak3d versions next to every other tool: runs bench/score.py's scorers on those tools,
# writes their tables to results/diag_double/full_<TAG>/ and an overview.md (truth recall, quantification, IDSL003,
# credentialing, ratio consistency, replicate junk share from tool_junk.py, the MassCube synthetic and EIC benchmarks with
# the published references, run time over the runs every tool finished, features per run; PR002464 for the tools run there).
# usage: full_overview.py TAG TOOL=LABEL ...   (TOOL: a results/ directory, e.g. peak3d_GLRS=peak3d GLRS; the PR002464
# scorers also take bench/score.py PR2464_EXTRA, the peak3d version run there)
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import bench.score as S  # noqa: E402

tag = sys.argv[1]
LABEL = dict(a.split("=", 1) for a in sys.argv[2:])
TOOLS = list(LABEL)
for t in S.PR2464_EXTRA:                       # labels for the peak3d version scored on PR002464 only
    LABEL.setdefault(t, t.replace("peak3d_", "peak3d ") + " (PR002464 only)")
OUT = ROOT / "results/diag_double" / f"full_{tag}"
JUNK = ROOT / "results/diag_double/tool_junk_peak3d_GL2_peak3d_GLRS_asari_masscube_idslipa_eb.tsv"
JUNK_ALIAS = {"peak3d": "peak3d_GL2"}          # results/peak3d is the GL2 default run

S.TOOLS[:] = TOOLS
S.OUT = OUT
OUT.mkdir(parents=True, exist_ok=True)
sc = {}
for name, fn in [("hzv029", S.score_hzv029), ("yeast_neg", S.score_yeastneg), ("spike", S.score_spike),
                 ("credtruth", S.score_credtruth), ("credtruth_screened", lambda: S.score_credtruth(screened=True)),
                 ("idsl", S.score_idsl), ("ratio", S.score_ratio), ("credentialing", S.score_cred),
                 ("mc_synth", S.score_mc_synth), ("mc_eic", S.score_mc_eic), ("runtime", S.score_runtime),
                 ("pr2464_spike", S.score_pr2464_spike), ("pr2464_yeast", S.score_pr2464_yeast),
                 ("pr2464_plasma", S.score_pr2464_plasma)]:
    x = fn()
    df = x[0] if isinstance(x, tuple) else x
    df.to_csv(OUT / f"{name}.tsv", sep="\t", index=False)
    sc[name] = df
    print(f"{name}: {len(df)} rows", flush=True)


def pick(df, col, tool, **flt):
    d = df[df["tool"] == tool]
    for k, v in flt.items():
        d = d[d[k] == v]
    return float(d[col].iloc[0]) if len(d) else np.nan


T = {}
std = dict(ppm=10, rt_tol_min=0.1)
for t in TOOLS:
    r = {}
    r["HZV029 found /402"] = pick(sc["hzv029"], "found", t, **std)
    r["YEAST_NEG found /314"] = pick(sc["yeast_neg"], "found", t, **std)
    r["LI2018 recall"] = pick(sc["spike"], "recall", t, **std)
    r["LI2018 log2 err"] = pick(sc["spike"], "diff_median_abs_log2_err", t, **std)
    r["LI2018 within 2x"] = pick(sc["spike"], "diff_within_2fold", t, **std)
    for c, n in (("F1", "F1"), ("precision", "prec"), ("recall", "recall")):
        r[f"IDSL003 {n}"] = pick(sc["idsl"], c, t, **std)
    for ds in ("SZ22", "YEAST"):
        r[f"{ds} cred recall"] = pick(sc["credtruth"], "recall_12C", t, dataset=ds, **std)
        r[f"{ds} cred scr"] = pick(sc["credtruth_screened"], "recall_12C", t, dataset=ds, **std)
    for ds in ("SZ22", "YEAST"):
        r[f"cred frac {ds}"] = pick(sc["credentialing"], "frac_credentialed_net", t, dataset=ds)
    for run, n in (("BM21_HILIC", "HILIC"), ("BM21_RP", "RP")):
        r[f"BM21 ratio {n}"] = pick(sc["ratio"], "frac_ratio_consistent", t, run=run, min_presence="all")
    T[LABEL[t]] = r
O = pd.DataFrame(T)

J = pd.read_csv(JUNK, sep="\t")
jrows = {}
for t in TOOLS:
    jt = J[J["tool"] == JUNK_ALIAS.get(t, t)]
    if jt.empty:
        continue
    d = {f"junk {r.run} (%)": round(100 * r.junk / r.features, 2) for r in jt.itertuples()}
    d["junk mean of 4 runs (%)"] = round(100 * (jt.junk / jt.features).mean(), 2)
    d["not reproduced, mean (%)"] = round(100 * (jt.not_reproduced / jt.features).mean(), 2)
    jrows[LABEL[t]] = d
JT = pd.DataFrame(jrows)

syn, eic = sc["mc_synth"], sc["mc_eic"]
eic = eic[eic["rule"] == "paper"]
names = list(dict.fromkeys(list(syn["tool"]) + list(eic["tool"])))
M = []
for n in names:
    s, e = syn[syn["tool"] == n], eic[eic["tool"] == n]
    M.append({"tool": LABEL.get(n, n),
              "synth single": s["single_acc"].iloc[0] if len(s) else np.nan, "synth double": s["double_acc"].iloc[0] if len(s) else np.nan,
              "synth avg": s["avg_acc"].iloc[0] if len(s) else np.nan, "EIC single": e["single_acc"].iloc[0] if len(e) else np.nan,
              "EIC double": e["double_acc"].iloc[0] if len(e) else np.nan, "EIC avg": e["avg_acc"].iloc[0] if len(e) else np.nan})
M = pd.DataFrame(M)
num = M.columns[1:]
M[num] = M[num] * 100                          # the scorers report fractions

PR = {"pr2464_spike": ["n_features", "recall", "recall_A", "decoy_match", "ratio_within_2fold", "dil_within_025_of_raw"],
      "pr2464_yeast": ["recall", "recall_A", "recall_confirmed", "decoy_match", "dil_within_025_of_raw"],
      "pr2464_plasma": ["recall", "recall_A", "decoy_match", "dil_within_025_of_raw", "mix_within_2fold_of_raw"]}
prmd = []
for name, cols in PR.items():
    d = sc[name]
    d = d[(d["ppm"] == 10) & (d["rt_tol_min"] == 0.1)]
    if d.empty:
        continue
    rows = {}
    for (study, tool), g in d.groupby(["study", "tool"]):
        for c in cols:
            rows.setdefault((study, c), {})[LABEL.get(tool, tool)] = g[c].iloc[0]
    X = pd.DataFrame(rows).T
    X = X[list(dict.fromkeys(c for c in [LABEL.get(t, t) for t in TOOLS + list(S.PR2464_EXTRA)] if c in X.columns))]
    prmd += [f"### {name} (10 ppm / 0.1 min)", "", X.to_markdown(floatfmt=".3f"), ""]

rt = sc["runtime"]
rt = rt[~rt["run"].str.startswith(("MC_", "PR2464_"))]
core = sorted(set.intersection(*[set(rt.loc[rt["tool"] == t, "run"]) for t in TOOLS]))
R = rt[rt["run"].isin(core)].groupby("tool").agg(wall=("wall_min", "sum"), cpu=("cpu_min", "sum"), rss=("max_rss_gb", "max"),
                                                 runs=("run", "size"), feats=("n_features", "sum"))
R.index = [LABEL[t] for t in R.index]
R = R.loc[[LABEL[t] for t in TOOLS if LABEL[t] in R.index]]
FP = rt[rt["run"].isin(core)].pivot_table(index="run", columns="tool", values="n_features").rename(columns=LABEL)
FP = FP[[LABEL[t] for t in TOOLS if LABEL[t] in FP.columns]]

md = [f"# Benchmark overview: {', '.join(LABEL[t] for t in TOOLS)} ({tag}, 10 ppm / 0.1 min)", "",
      O.to_markdown(floatfmt=".3f"), "",
      "## Replicate junk share (tool_junk.py: not reproduced in either other injection, or fails the trace check)", "",
      JT.to_markdown(floatfmt=".2f"), "",
      "## MassCube benchmark (%)", "", M.to_markdown(index=False, floatfmt=".1f"), "",
      f"## Time, {len(core)} core runs (minutes; max RSS GB)", "", R.to_markdown(floatfmt=".2f"), "",
      "## Features per run", "", FP.to_markdown(floatfmt=".0f"), "",
      "## PR002464 (only the tools run there; peak3d as bench/score.py PR2464_EXTRA)", ""] + prmd
(OUT / "overview.md").write_text("\n".join(md))
print("\n".join(md))
print(f"-> {OUT / 'overview.md'}")
