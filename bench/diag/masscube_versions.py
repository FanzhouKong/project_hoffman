# diagnostic: why our MassCube 1.2.22 run scores below the MassCube 1.0 results the paper reports, on the paper's own
# benchmark. For every MC run: the feature table the benchmark scores (aligned_feature_table.txt of the full workflow),
# our per-file detection table (single_files/<file>.txt, the stage the paper scored) and the authors' published per-file
# table (MassCube 1.0, Zenodo 14159704): feature counts, scan-length distribution, and accuracy under the paper's rules.
# usage: masscube_versions.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.score import MC_TRUTH, mc_rows, window_counts  # noqa: E402
from bench.runs import MC_EXP, MCSI  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 30)
lab = pd.read_csv(MC_TRUTH / "eic_labels.tsv", sep="\t")
pub = pd.read_csv(MC_TRUTH / "published_features.tsv", sep="\t")
pub = pub[pub.tool == "masscube"]
PUBDIR = MCSI / "8_experimental_data/masscube"


def tables(f):
    d = ROOT / f"results/masscube/MC_{f}"
    al = pd.read_csv(d / "aligned_feature_table.txt", sep="\t", low_memory=False)
    sf = pd.read_csv(d / f"single_files/{f}.txt", sep="\t", low_memory=False)
    pb = pd.read_csv(PUBDIR / f"{f}.txt", sep="\t", low_memory=False)
    return {"aligned (scored)": pd.DataFrame({"mz": al["m/z"], "rt": al["RT"]}),
            "single_file 1.2.22": pd.DataFrame({"mz": sf["m/z"], "rt": sf["RT"], "length": sf.get("length")}),
            "published 1.0": pd.DataFrame({"mz": pb["m/z"], "rt": pb["RT"], "length": pb.get("length")})}, sf, pb


rows, cnt = [], []
n_peaks = lab["n_peaks"].values
for f in MC_EXP:
    T, sf, pb = tables(f)
    m = (lab.file == f).values
    tol = np.where(n_peaks[m] == 1, 0.002, 0.005)
    for name, d in T.items():
        c = window_counts(d, lab.mz.values[m], tol, lab.rt.values[m], 0.2)
        rows.append(dict(mc_rows(name, c, n_peaks[m]), file=f, n_features=len(d)))
        if "length" in d and d["length"] is not None and d["length"].notna().any():
            cnt.append(dict(file=f, table=name, n=len(d), length_min=int(d["length"].min()),
                            length_lt5=float((d["length"] < 5).mean())))
R = pd.DataFrame(rows)
print("== EIC accuracy per file (paper rule) ==")
print(R.pivot_table(index="file", columns="tool", values="single_acc").round(3).to_string())
print(R.pivot_table(index="file", columns="tool", values="n_features").to_string())
print("\n== pooled over the 8 files ==")
tot = []
for name in R.tool.unique():
    c_all, k_all = [], []
    for f in MC_EXP:
        T, _, _ = tables(f)
        m = (lab.file == f).values
        tol = np.where(n_peaks[m] == 1, 0.002, 0.005)
        c_all.append(window_counts(T[name], lab.mz.values[m], tol, lab.rt.values[m], 0.2)); k_all.append(n_peaks[m])
    tot.append(mc_rows(name, np.concatenate(c_all), np.concatenate(k_all)))
print(pd.DataFrame(tot)[["tool", "single_acc", "single_missed", "single_extra", "double_acc", "double_missed", "double_merged",
                         "double_extra", "avg_acc"]].round(3).to_string(index=False))
print("\n== scan length of the per-file tables ==")
print(pd.DataFrame(cnt).round(3).to_string(index=False))
print("\ncolumns, our single-file table:", list(sf.columns)[:20])
print("columns, published table:     ", list(pb.columns)[:20])

# synthetic: the same three stages
t = pd.read_csv(MC_TRUTH / "synth_peaks.tsv", sep="\t")
d = ROOT / "results/masscube/MC_SYNTH"
al = pd.read_csv(d / "aligned_feature_table.txt", sep="\t", low_memory=False)
sf = pd.read_csv(d / "single_files/simulated_file.txt", sep="\t", low_memory=False)
syn = {"aligned (scored)": al, "single_file 1.2.22": sf}
p = MCSI / "Synthetic_data_benchmarking/Synthetic_data_MassCube.txt"
if p.exists():
    syn["published 1.0 table"] = pd.read_csv(p, sep="\t", low_memory=False)
print("\n== synthetic (paper rule: count within 0.01 Da) ==")
for name, df in syn.items():
    c = window_counts(pd.DataFrame({"mz": df["m/z"], "rt": df["RT"]}), t["mz"].values, 0.01)
    r = mc_rows(name, c, t["n_peaks"].values)
    print(f"{name:22s} n={len(df):6d} single {r['single_acc']:.3f} (missed {r['single_missed']:.3f}, extra {r['single_extra']:.3f}) "
          f"double {r['double_acc']:.3f} (missed {r['double_missed']:.3f}, merged {r['double_merged']:.3f}, extra {r['double_extra']:.3f})")
c = t["pub_MassCube 1.0"].values
r = mc_rows("paper", c, t["n_peaks"].values)
print(f"{'paper (roi_mz_seq)':22s}          single {r['single_acc']:.3f} double {r['double_acc']:.3f}")
