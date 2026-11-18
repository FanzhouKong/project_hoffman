#!/usr/bin/env python3
"""Two tools side by side on one set of the DeepCPR benchmark, from the tables bench/gc/deepcpr_score.py wrote
(<tool>.series.tsv, <tool>.hits.tsv with --library): which library compounds each tool names in the top-level file,
and, for the compounds both name, the match score and the linearity of each.

A compound = the first block of the InChIKey of a component's best library hit (match >= --min-match) in the
top-level file; when a tool has several components with the same hit, the one with the largest area stands for it
(the others are counted as duplicates: a compound split in two, or two isomers the library cannot tell apart).

    python -I bench/gc/deepcpr_compare.py SCORE_DIR TOOL_A TOOL_B [--min-match 0.7]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def named(score, tool, top, min_match):
    H = pd.read_csv(score / f"{tool}.hits.tsv", sep="\t")
    S = pd.read_csv(score / f"{tool}.series.tsv", sep="\t")
    H = H[(H["file"] == top) & (H["match"] >= min_match)].copy()
    H["key"] = H["inchikey"].astype(str).str[:14]
    H = H.merge(S[["ref_component", "n_levels", "r", "slope_loglog"]], left_on="component", right_on="ref_component",
                how="left")
    dup = int(len(H) - H["key"].nunique())
    H = H.sort_values("area", ascending=False).drop_duplicates("key")
    return H.set_index("key"), dup


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("score", type=Path)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--min-match", type=float, default=0.7)
    g = ap.parse_args()
    levels = json.loads((g.score / "summary.json").read_text())["levels"]
    top = max(levels, key=levels.get)
    A, dup_a = named(g.score, g.a, top, g.min_match)
    B, dup_b = named(g.score, g.b, top, g.min_match)
    both = A.index.intersection(B.index)
    T = pd.DataFrame(dict(name=A.loc[both, "name"], rt_a=A.loc[both, "rt"], rt_b=B.loc[both, "rt"],
                          match_a=A.loc[both, "match"], match_b=B.loc[both, "match"],
                          reverse_a=A.loc[both, "reverse"], reverse_b=B.loc[both, "reverse"],
                          levels_a=A.loc[both, "n_levels"], levels_b=B.loc[both, "n_levels"],
                          r_a=A.loc[both, "r"], r_b=B.loc[both, "r"],
                          slope_a=A.loc[both, "slope_loglog"], slope_b=B.loc[both, "slope_loglog"])).sort_values("rt_a")
    T.to_csv(g.score / f"compare_{g.a}_vs_{g.b}.tsv", sep="\t", float_format="%.5g")
    only_a = A.loc[A.index.difference(B.index), ["name", "rt", "area", "match", "r"]].sort_values("rt")
    only_b = B.loc[B.index.difference(A.index), ["name", "rt", "area", "match", "r"]].sort_values("rt")
    only_a.to_csv(g.score / f"compare_only_{g.a}.tsv", sep="\t", float_format="%.5g")
    only_b.to_csv(g.score / f"compare_only_{g.b}.tsv", sep="\t", float_format="%.5g")
    n5 = T[(T["levels_a"] == len(levels)) & (T["levels_b"] == len(levels))]
    resp = n5[(n5["slope_a"] >= 0.5) | (n5["slope_b"] >= 0.5)]
    out = dict(top_level_file=top, min_match=g.min_match,
               named={g.a: int(len(A)), g.b: int(len(B))}, duplicates={g.a: dup_a, g.b: dup_b},
               named_by_both=int(len(both)), only={g.a: int(len(only_a)), g.b: int(len(only_b))},
               match_median_on_common={g.a: float(T["match_a"].median()), g.b: float(T["match_b"].median())},
               match_higher={g.a: int((T["match_a"] > T["match_b"] + 0.01).sum()),
                             g.b: int((T["match_b"] > T["match_a"] + 0.01).sum())},
               common_at_all_levels_in_both=int(len(n5)), common_responding=int(len(resp)),
               r_mean_common_responding={g.a: float(resp["r_a"].mean()), g.b: float(resp["r_b"].mean())},
               r_median_common_responding={g.a: float(resp["r_a"].median()), g.b: float(resp["r_b"].median())},
               all_levels_among_common={g.a: int((T["levels_a"] == len(levels)).sum()),
                                        g.b: int((T["levels_b"] == len(levels)).sum())})
    (g.score / f"compare_{g.a}_vs_{g.b}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"tables: {g.score}/compare_{g.a}_vs_{g.b}.tsv, compare_only_{g.a}.tsv, compare_only_{g.b}.tsv")

if __name__ == "__main__":
    main()
