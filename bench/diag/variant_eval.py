# diagnostic: score peak3d variants (results/peak3d_<variant>/, scripts/42_peak3d_variant.sbatch) with the benchmark's
# own scorers and apply the pre-stated adoption rule (results/diag_double/ADOPTION_RULE.md) against variant "base".
# Also checks that "base" reproduces results/peak3d byte for byte, and runs the replicate-junk yardstick
# (version_junk.py: one noise reference for every variant) on the four replicate runs.
# Writes results/diag_double/scores/*.tsv, variant_eval.tsv (one row per rule line) and variant_eval.md.
# usage: variant_eval.py [--no-junk] [VARIANT ...]   (default: every variant in manifest/peak3d_variants.tsv; the first
#        listed is the reference the rule compares against; run on a compute node, SLURM_CPUS_PER_TASK workers)
import csv
import filecmp
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/diag"))
import bench.score as S  # noqa: E402
from bench.runs import RUNS  # noqa: E402

OUT = ROOT / "results/diag_double"
VARIANTS = [a for a in sys.argv[1:] if not a.startswith("--")] or \
    [r["variant"] for r in csv.DictReader((ROOT / "manifest/peak3d_variants.tsv").open(), delimiter="\t")]
TOOLS = [f"peak3d_{v}" for v in VARIANTS]
JUNK_RUNS = ["HZV029_cert", "YEAST_NEG", "SZ22_12C", "YEAST_12C"]


def identity():
    """every base* variant (defaults) vs the benchmark's peak3d: per-file feature tables and the aligned table"""
    rows = []
    for v in [v for v in VARIANTS if v.startswith("base")]:
        for run in RUNS:
            a, b = ROOT / "results/peak3d" / run / "peak3d_out", ROOT / f"results/peak3d_{v}" / run / "peak3d_out"
            if not (b.parent / "DONE").exists() or not (a.parent / "DONE").exists():
                continue
            names = ["aligned_feature_table.tsv"] + [f"features/{Path(f).stem}.tsv" for f in RUNS[run]["files"]]
            diff = [n for n in names if not filecmp.cmp(a / n, b / n, shallow=False)]
            rows.append(dict(variant=v, run=run, files=len(names), different=len(diff),
                             first_different=diff[0] if diff else ""))
    return pd.DataFrame(rows)


def scores():
    S.TOOLS[:] = TOOLS
    S.OUT = OUT / ("scores_" + "_".join(VARIANTS))
    S.OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, fn in [("hzv029", S.score_hzv029), ("yeast_neg", S.score_yeastneg), ("spike", S.score_spike),
                     ("credtruth", S.score_credtruth), ("credtruth_screened", lambda: S.score_credtruth(screened=True)),
                     ("idsl", S.score_idsl), ("ratio", S.score_ratio), ("credentialing", S.score_cred),
                     ("mc_synth", S.score_mc_synth), ("mc_eic", S.score_mc_eic), ("runtime", S.score_runtime)]:
        df = fn()
        df.to_csv(S.OUT / f"{name}.tsv", sep="\t", index=False)
        out[name] = df
    return out


def junk():
    import version_junk as vj
    versions = {v: str(ROOT / f"results/peak3d_{v}") for v in VARIANTS}
    jobs = [(r, k, versions) for r in JUNK_RUNS for k in range(len(RUNS[r]["files"]))]
    from peak3d.pipeline import _warm_jit
    _warm_jit()
    with get_context("spawn").Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))) as pool:
        res = pool.map(vj.one, jobs, chunksize=1)
    rows = []
    for run in JUNK_RUNS:
        for v in VARIANTS:
            A = pd.concat([o[v] for r, s, o in res if r == run])
            bad = (A["rep"] == 0) | (A["cls"] != "ok")
            rows.append(dict(run=run, variant=v, features=len(A), junk=int(bad.sum()), junk_share=float(bad.mean())))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / ("scores_" + "_".join(VARIANTS)) / "version_junk.tsv", sep="\t", index=False)
    return T


def head(df):
    return df[(df["ppm"] == 10) & (df["rt_tol_min"] == 0.1)] if "ppm" in df else df


def metrics(sc, jk):
    """{variant: {metric: value}} at the headline points"""
    m = {v: {} for v in VARIANTS}

    def put(name, df, col, key=None, filt=None):
        d = head(df)
        if filt is not None:
            d = d[filt(d)]
        for v in VARIANTS:
            x = d[d["tool"] == f"peak3d_{v}"]
            if len(x):
                m[v][name] = float(x[col].iloc[0]) if key is None else float(x.set_index(key)[col].iloc[0])
    put("hzv029_recall", sc["hzv029"], "recall")
    put("yeastneg_recall", sc["yeast_neg"], "recall")
    put("li2018_recall", sc["spike"], "recall")
    put("li2018_log2err", sc["spike"], "diff_median_abs_log2_err")
    put("li2018_within2", sc["spike"], "diff_within_2fold")
    for ds in ("SZ22", "YEAST"):
        put(f"credtruth_{ds}", sc["credtruth"], "recall_12C", filt=lambda d, ds=ds: d["dataset"] == ds)
        put(f"credtruth_scr_{ds}", sc["credtruth_screened"], "recall_12C", filt=lambda d, ds=ds: d["dataset"] == ds)
    put("idsl_F1", sc["idsl"], "F1")
    for run in ("BM21_HILIC", "BM21_RP"):
        put(f"ratio_{run}", sc["ratio"], "frac_ratio_consistent",
            filt=lambda d, run=run: (d["run"] == run) & (d["min_presence"] == "all"))
    put("cred_frac_YEAST", sc["credentialing"], "frac_credentialed_net", filt=lambda d: d["dataset"] == "YEAST")
    put("synth_single", sc["mc_synth"], "single_acc")
    put("synth_double", sc["mc_synth"], "double_acc")
    put("synth_avg", sc["mc_synth"], "avg_acc")
    put("eic_avg", sc["mc_eic"], "avg_acc", filt=lambda d: d["rule"] == "paper")
    put("eic_single", sc["mc_eic"], "single_acc", filt=lambda d: d["rule"] == "paper")
    put("eic_double", sc["mc_eic"], "double_acc", filt=lambda d: d["rule"] == "paper")
    rt = sc["runtime"]
    for v in VARIANTS:
        x = rt[rt["tool"] == f"peak3d_{v}"]
        m[v]["cpu_min_total"] = float(x["cpu_min"].sum())
        m[v]["n_runs_timed"] = len(x)
        m[v]["features_total"] = float(x["n_features"].sum())
    if jk is not None:
        for _, r in jk.iterrows():
            m[r["variant"]][f"junk_{r['run']}"] = r["junk_share"]
    return m


# (metric, direction, tolerance): pass if variant >= base - tol (direction +1) or <= base + tol (direction -1)
RULE = [("synth_double", "gt", 0.0), ("synth_single", +1, 0.01), ("eic_avg", +1, 0.005),
        ("hzv029_recall", +1, 1 / 402 + 1e-9), ("yeastneg_recall", +1, 1 / 314 + 1e-9),
        ("li2018_recall", +1, 0.005), ("li2018_log2err", -1, 0.01), ("li2018_within2", +1, 0.01),
        ("credtruth_SZ22", +1, 0.005), ("credtruth_YEAST", +1, 0.005),
        ("credtruth_scr_SZ22", +1, 0.005), ("credtruth_scr_YEAST", +1, 0.005), ("idsl_F1", +1, 0.005),
        ("ratio_BM21_HILIC", +1, 0.005), ("ratio_BM21_RP", +1, 0.005)] + \
       [(f"junk_{r}", -1, 0.005) for r in JUNK_RUNS] + [("cred_frac_YEAST", +1, 0.01), ("cpu_min_total", "ratio", 1.10)]


def decide(m):
    rows = []
    base = m[VARIANTS[0]]
    for name, d, tol in RULE:
        row = dict(metric=name, base=base.get(name, np.nan))
        for v in VARIANTS[1:]:
            x, b = m[v].get(name, np.nan), base.get(name, np.nan)
            if np.isnan(x) or np.isnan(b):
                ok = None
            elif d == "gt":
                ok = x > b
            elif d == "ratio":
                ok = x <= tol * b
            else:
                ok = (x >= b - tol) if d > 0 else (x <= b + tol)
            row[v] = x
            row[f"{v}_pass"] = ok
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    junk_on = "--no-junk" not in sys.argv[1:]
    idt = identity()
    print("== base vs results/peak3d (byte identity) ==")
    print(idt.to_string(index=False))
    sc = scores()
    jk = junk() if junk_on else None
    m = metrics(sc, jk)
    D = decide(m)
    tag = "_".join(VARIANTS)
    D.to_csv(OUT / f"variant_eval_{tag}.tsv", sep="\t", index=False)
    passed = {v: bool(D[f"{v}_pass"].fillna(False).all()) for v in VARIANTS[1:]}
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print("\n== adoption rule (results/diag_double/ADOPTION_RULE.md) ==")
    print(D.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\npasses every line:", passed)
    extra = pd.DataFrame(m).T[["features_total", "n_runs_timed", "eic_single", "eic_double", "synth_avg"]]
    print(extra.to_string())
    with (OUT / f"variant_eval_{tag}.md").open("w") as fh:
        fh.write("# peak3d noise-law variants vs base\n\n## base identity\n\n" + idt.to_markdown(index=False) +
                 "\n\n## adoption rule\n\n" + D.to_markdown(index=False, floatfmt=".4f") +
                 f"\n\npasses every line: {passed}\n\n" + extra.to_markdown(floatfmt=".4f") + "\n")


if __name__ == "__main__":
    main()
