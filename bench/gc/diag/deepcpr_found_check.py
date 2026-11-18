"""The 'found' column of the paper-style table under other readings: Mosa's component against DeepCPR's spectrum
of the same compound in the top-level file (instead of DeepCPR's spectrum in the file itself, which at low signal
is mostly noise masses), and how often one Mosa component is the match of two targets.
    python bench/gc/diag/deepcpr_found_check.py [mosa_dsk]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from bench.gc.deepcpr_score import load_deepcpr, load_mosa, observed, root_spectra, cosine

M = Path("/quobyte/metabolomicsgrp/fanzhou/hoffmann"); R = M / "results/gc_deepcpr"
src = sys.argv[1] if len(sys.argv) > 1 else "mosa_dsk"
for st, folder in (("aa", "amino acids"), ("fa", "fatty acids")):
    T = pd.read_csv(R / "score" / st / f"paper_table_{src}.tsv", sep="\t")
    stems = sorted(T.file.unique(), key=lambda f: T.loc[T.file == f, "level"].iloc[0])
    obs = {s: observed(M / "data/raw/DEEPCPR_GC/data" / folder / f"{s}.CDF") for s in stems}
    lo, hi = min(o[1][0] for o in obs.values()), max(o[1][-1] for o in obs.values())
    masses = np.arange(lo, hi + 1)
    D = {s: load_deepcpr(R / "deepcpr" / st, s, masses, obs[s][0]) for s in stems}
    A = {s: load_mosa(R / src / st, s, masses, obs[s][0]) for s in stems}
    top = stems[-1]
    rows = []
    for r in T.itertuples():
        ref = D[top]["spec"][int(np.argmin(np.abs(D[top]["rt"] - r.rt)))]
        jd = int(np.argmin(np.abs(D[r.file]["rt"] - r.rt)))
        own = D[r.file]["spec"][jd]
        near = np.flatnonzero(np.abs(A[r.file]["rt"] - D[r.file]["rt"][jd]) <= 1.5)
        ra = root_spectra(A[r.file]["spec"][near])
        c_ref = np.array([cosine(a, root_spectra(ref[None])[0]) for a in ra]) if len(near) else np.zeros(1)
        c_own = np.array([cosine(a, root_spectra(own[None])[0]) for a in ra]) if len(near) else np.zeros(1)
        rows.append(dict(rt=r.rt, file=r.file, level=r.level, snr=r.snr, cos_file=c_own.max(), cos_top=c_ref.max(),
                         best=int(near[int(np.argmax(c_ref))]) if len(near) else -1,
                         d_s=float(A[r.file]["rt"][near[int(np.argmax(c_ref))]] - D[r.file]["rt"][jd]) if len(near) else np.nan,
                         deepcpr_self=cosine(root_spectra(own[None])[0], root_spectra(ref[None])[0])))
    X = pd.DataFrame(rows)
    print(f"\n== {st} ({src}): {len(X)} cases; found against DeepCPR's spectrum in the file {int((X.cos_file >= 0.7).sum())}, "
          f"against its top-level spectrum {int((X.cos_top >= 0.7).sum())}; DeepCPR's own spectrum in the file matches its "
          f"top-level one at >= 0.7 in {int((X.deepcpr_self >= 0.7).sum())}, >= 0.9 in {int((X.deepcpr_self >= 0.9).sum())}")
    ok = X[X.cos_top >= 0.7]
    dup = ok[ok.duplicated(["file", "best"], keep=False)].sort_values(["file", "best", "rt"])
    print(f"cases whose matching Mosa component is also the match of another target: {len(dup)}")
    if len(dup):
        print(dup[["rt", "file", "cos_top", "best", "d_s"]].round(3).to_string(index=False))
    print("cases not found against the top-level spectrum, or only there:")
    print(X[(X.cos_top < 0.7) | (X.cos_file < 0.7)][["rt", "file", "snr", "cos_file", "cos_top", "deepcpr_self", "d_s"]].round(3).to_string(index=False))
