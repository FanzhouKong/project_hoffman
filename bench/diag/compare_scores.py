# diagnostic: peak3d before / after a rerun, side by side with the other tools, from two score dirs
# usage: compare_scores.py OLD_SCORES_DIR NEW_SCORES_DIR
import sys
from pathlib import Path

import pandas as pd

old, new = Path(sys.argv[1]), Path(sys.argv[2])
pd.set_option("display.width", 200)


def at_default(df):
    return df[(df["ppm"] == 10) & (df["rt_tol_min"] == 0.1)] if "ppm" in df else df


def show(name, cols, key=("tool",)):
    a, b = pd.read_csv(old / f"{name}.tsv", sep="\t"), pd.read_csv(new / f"{name}.tsv", sep="\t")
    a, b = at_default(a), at_default(b)
    pa = a[a["tool"] == "peak3d"].assign(tool="peak3d (Oct 2)")
    out = pd.concat([pa, b])[list(key) + cols]
    print(f"\n== {name}" + (" (10 ppm, 0.1 min)" if "ppm" in a else ""))
    print(out.sort_values(list(key)[:-1] + ["tool"] if len(key) > 1 else ["tool"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


show("hzv029", ["found", "recall", "n_features"])
show("yeast_neg", ["found", "recall", "recall_metabolites", "n_features"])
show("spike", ["recall", "n_features", "diff_median_abs_log2_err", "diff_within_2fold", "const_false_2fold"])
show("idsl", ["TP", "FP", "recall", "precision", "F1"])
show("ratio", ["n_features", "n_ratio_consistent", "frac_ratio_consistent"], key=("run", "min_presence", "tool"))
show("credentialing", ["n_features_12C", "credentialed_net", "frac_credentialed_net"], key=("dataset", "tool"))
show("runtime", ["n_files", "wall_min", "max_rss_gb", "n_features"], key=("run", "tool"))
