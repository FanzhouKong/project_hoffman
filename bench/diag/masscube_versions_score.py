# diagnostic: score the per-file MassCube detections of masscube_version_runs.py (A = 1.0.22, B = 1.2.22 with the
# 1.0.22 segmentation settings, C = 1.2.22 as installed) next to the authors' published 1.0 results and the full
# 1.2.22 workflow's aligned table that the benchmark scores. Paper rules (bench/score.py); synthetic by noise level.
# usage: masscube_versions_score.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.runs import MC_EXP  # noqa: E402
from bench.score import MC_TRUTH, load, mc_rows, window_counts  # noqa: E402

V = ROOT / "results/diag_double/masscube_versions"
pd.set_option("display.width", 250, "display.max_columns", 30)
t = pd.read_csv(MC_TRUTH / "synth_peaks.tsv", sep="\t")
lab = pd.read_csv(MC_TRUTH / "eic_labels.tsv", sep="\t")
pub = pd.read_csv(MC_TRUTH / "published_features.tsv", sep="\t")
pub = pub[pub.tool == "masscube"]
n_peaks = lab["n_peaks"].values
tol = np.where(n_peaks == 1, 0.002, 0.005)


def eic_counts(get):
    c = np.full(len(lab), -1)
    for f in MC_EXP:
        d = get(f)
        if d is None:
            continue
        m = (lab.file == f).values
        c[m] = window_counts(d, lab.mz.values[m], tol[m], lab.rt.values[m], 0.2)
    return c


def mode_table(mode, stem):
    p = V / mode / f"{stem}.tsv"
    return pd.read_csv(p, sep="\t") if p.exists() else None


sources = {
    "published 1.0 (paper)": (lambda: t["pub_MassCube 1.0"].values,
                              lambda f: pub[pub.file == f][["mz", "rt"]]),
    "A: 1.0.22 per-file (run here)": (lambda: window_counts(mode_table("A", "simulated_file"), t.mz.values, 0.01),
                                      lambda f: mode_table("A", f)),
    "B: 1.2.22 per-file, 1.0 segmentation": (lambda: window_counts(mode_table("B", "simulated_file"), t.mz.values, 0.01),
                                             lambda f: mode_table("B", f)),
    "C: 1.2.22 per-file": (lambda: window_counts(mode_table("C", "simulated_file"), t.mz.values, 0.01),
                           lambda f: mode_table("C", f)),
    "1.2.22 workflow, aligned (benchmark)": (lambda: window_counts(load("masscube", "MC_SYNTH")[0], t.mz.values, 0.01),
                                             lambda f: load("masscube", f"MC_{f}")[0]),
}
rows, by_noise = [], []
for name, (syn, eic) in sources.items():
    try:
        cs = syn()
    except Exception as e:  # a mode that has not finished
        print(f"{name}: synthetic not available ({e})")
        continue
    ce = eic_counts(eic)
    done = ce >= 0
    r = {"source": name}
    s = mc_rows(name, cs, t.n_peaks.values)
    noise = t.noise.values
    lv = [mc_rows(name, cs[noise == z], t.n_peaks.values[noise == z], noise=z) for z in sorted(set(noise))]
    by_noise += lv
    r.update(syn_single=np.mean([x["single_acc"] for x in lv]), syn_double=np.mean([x["double_acc"] for x in lv]))
    r["syn_avg"] = (r["syn_single"] + r["syn_double"]) / 2
    e = mc_rows(name, ce[done], n_peaks[done])
    r.update(eic_files=int(sum(eic(f) is not None for f in MC_EXP)), eic_single=e["single_acc"], eic_double=e["double_acc"],
             eic_avg=e["avg_acc"], eic_single_extra=e["single_extra"], eic_single_missed=e["single_missed"])
    rows.append(r)
R = pd.DataFrame(rows).set_index("source")
print((R * 100).round(1).assign(eic_files=R.eic_files).to_string())
N = pd.DataFrame(by_noise)
print("\nsynthetic by noise level (single / double accuracy, %):")
print((N.pivot_table(index="noise", columns="tool", values="single_acc") * 100).round(1).to_string())
print((N.pivot_table(index="noise", columns="tool", values="double_acc") * 100).round(1).to_string())
R.to_csv(ROOT / "results/diag_double/masscube_versions_score.tsv", sep="\t")
