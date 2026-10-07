# where is the remaining harm after the gate? per file and RT window: mean |deviation from the other
# files' median| of truth features, native / full anchor fit / gated, with the gate factor there
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench/diag"))
from rt_gate_eval import load
import peak3d.align as AL
run = sys.argv[1]; edges = [float(v) for v in sys.argv[2].split(",")]
stems, feats, grids, nat, cor, dt = load(run)
res0 = AL.align_run({s: f.copy() for s, f in feats.items()}, grids, AL.AlignParams(gate=False))
res = AL.align_run({s: f.copy() for s, f in feats.items()}, grids, AL.AlignParams())
C0 = np.column_stack([res0.files[s].warp.forward(nat[:, j]) for j, s in enumerate(stems)])
C1 = np.column_stack([res.files[s].warp.forward(nat[:, j]) for j, s in enumerate(stems)])
N = len(stems); rows = []
for j, s in enumerate(stems):
    oth = np.delete(np.arange(N), j)
    dev = lambda M: 60 * np.abs(M[:, j] - np.nanmedian(M[:, oth], 1))
    dn, d0, d1 = dev(nat), dev(C0), dev(C1)
    g = res.files[s].gate
    for a, b in zip(edges[:-1], edges[1:]):
        m = np.isfinite(dn) & (nat[:, j] >= a) & (nat[:, j] < b)
        if m.sum() < 3: continue
        lam = np.interp(0.5 * (a + b), g["nodes"], g["lam"]) if g else np.nan
        rows.append(dict(file=s[-8:], win=f"{a:g}-{b:g}", n=int(m.sum()), nat=dn[m].mean(), fit=d0[m].mean(), gated=d1[m].mean(), lam_mid=lam,
                         worse_scan=np.mean(d1[m] - dn[m] > 60 * dt), better_scan=np.mean(d1[m] - dn[m] < -60 * dt)))
R = pd.DataFrame(rows); pd.set_option("display.width", 250)
print(R.round(2).to_string(index=False))
print(R.groupby("win")[["n", "nat", "fit", "gated", "worse_scan", "better_scan"]].mean().round(2).to_string())
