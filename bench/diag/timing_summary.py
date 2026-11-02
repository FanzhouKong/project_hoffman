# diagnostic: summarise scripts/43_paired_timing.sbatch: per run and variant, the mean wall, user and system time of
# the alternating repeats, and each variant's cost relative to the first variant of the pair.
# usage: timing_summary.py [TIMING_DIR]   (default results/diag_double/timing)
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
d = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/diag_double/timing"
rows = []
for f in sorted(d.glob("*/*.time")):
    txt = f.read_text()
    g = lambda k: re.search(rf"{k}: (.+)", txt).group(1).strip()  # noqa: E731
    wall = sum(float(x) * 60 ** i for i, x in enumerate(reversed(g(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\)").split(":"))))
    i, v = f.stem.split("_", 1)
    rows.append(dict(run=f.parent.name, rep=int(i), variant=v, wall_s=wall, user_s=float(g(r"User time \(seconds\)")),
                     sys_s=float(g(r"System time \(seconds\)"))))
T = pd.DataFrame(rows)
print(T.sort_values(["run", "rep"]).to_string(index=False))
M = T.groupby(["run", "variant"])[["wall_s", "user_s", "sys_s"]].mean().round(1)
M["cpu_s"] = M["user_s"] + M["sys_s"]
print("\n" + M.to_string())
order = list(dict.fromkeys(T.sort_values("rep")["variant"]))
ref = M.xs(order[0], level="variant")
for v in order[1:]:
    r = M.xs(v, level="variant") / ref
    print(f"\n{v} / {order[0]}:\n" + r[["wall_s", "cpu_s"]].round(3).to_string())
tot = M.groupby("variant")[["wall_s", "cpu_s"]].sum()
print("\nall runs:\n" + (tot / tot.loc[order[0]]).round(3).to_string())
