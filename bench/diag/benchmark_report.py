# report: every tool plus earlier peak3d versions side by side, from results/scores and score snapshots
# (10 ppm, 0.1 min unless stated). usage: benchmark_report.py LABEL=SCORES_DIR ...  (the current dir is "now")
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NOW = ROOT / "results/scores"
snaps = dict(a.split("=", 1) for a in sys.argv[1:])
pd.set_option("display.width", 220, "display.max_columns", 30)


def table(name, label_dirs):
    out = []
    for label, d in [("now", NOW)] + [(k, Path(v)) for k, v in label_dirs.items()]:
        p = Path(d) / f"{name}.tsv"
        if not p.exists():
            alt = ROOT / f"results/diag_excess/{name}_{label}.tsv"     # snapshot scored by score_snapshot.py
            if not alt.exists():
                continue
            t = pd.read_csv(alt, sep="\t")
        else:
            t = pd.read_csv(p, sep="\t")
            if label != "now":
                t = t[t["tool"] == "peak3d"]
                t = t.assign(tool=t["tool"] + f" ({label})")
        out.append(t)
    t = pd.concat(out, ignore_index=True)
    if "ppm" in t:
        t = t[(t["ppm"] == 10) & (t["rt_tol_min"] == 0.1)]
    return t


f = lambda v: f"{v:.3f}"   # noqa: E731
rt = table("runtime", snaps)
print("== number of features (aligned table rows) ==")
print(rt.pivot_table(index="run", columns="tool", values="n_features", aggfunc="first").astype("Int64").to_string())
print("\n== wall time (min) ==")
print(rt.pivot_table(index="run", columns="tool", values="wall_min", aggfunc="first").round(2).to_string())
ct = table("credtruth", snaps)
print("\n== credentialed truth (data/truth; recall of the 12C truth features, of their 13C partners, share of the tool's 12C features that are truth) ==")
print(ct[["dataset", "tool", "n_truth", "n_features_12C", "recall_12C", "recall_A", "recall_B", "recall_primary",
          "recall_13C_partner", "recall_both", "truth_share_of_features"]].sort_values(["dataset", "tool"]).to_string(index=False, float_format=f))
for name, cols in (("hzv029", ["found", "recall", "n_features"]),
                   ("yeast_neg", ["found", "recall", "n_features"]),
                   ("spike", ["recall", "n_features", "diff_median_abs_log2_err", "diff_within_2fold", "const_false_2fold"]),
                   ("idsl", ["TP", "FP", "recall", "precision", "F1"])):
    t = table(name, snaps)
    print(f"\n== {name} ==")
    print(t[["tool"] + cols].sort_values("tool").to_string(index=False, float_format=f))
t = table("ratio", snaps)
print("\n== ratio (BM21, all features) ==")
print(t[t["min_presence"] == "all"][["run", "tool", "n_features", "n_ratio_consistent", "frac_ratio_consistent"]]
      .sort_values(["run", "tool"]).to_string(index=False, float_format=f))
