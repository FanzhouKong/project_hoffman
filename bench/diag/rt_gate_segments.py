# per-file, per-segment gate factors and the validation-loss reduction they achieve (diagnostic)
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench/diag"))
from rt_gate_eval import load
import peak3d.align as AL
rows = []
for run in sys.argv[1:]:
    stems, feats, grids, nat, cor, dt = load(run)
    res = AL.align_run({s: f.copy() for s, f in feats.items()}, grids, AL.AlignParams())
    for s, fa in res.files.items():
        if fa.gate:
            for k in range(len(fa.gate["lam"])):
                rows.append(dict(run=run, file=s, node=fa.gate["nodes"][k], lam=fa.gate["lam"][k], gain=fa.gate["gain"][k]))
R = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(R.groupby("run").gain.describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())
for run, g in R.groupby("run"):
    h = np.histogram(g.gain, [-1, 0.0001, 0.02, 0.05, 0.1, 0.2, 0.4, 1.01])[0]
    print(f"{run:12s} gain bins [0,.02,.05,.1,.2,.4,1]: {h}  lam==0: {(g.lam == 0).mean():.0%}")
R.to_csv(ROOT / "results/rt_harm/gate_segments.tsv", sep="\t", index=False, float_format="%.4f")
