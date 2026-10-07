# diagnostic: run one of bench/score.py's scorers on an earlier peak3d snapshot (other tools as they are)
# usage: score_snapshot.py SCORER PEAK3D_DIR LABEL   e.g. score_snapshot.py credtruth results/peak3d_2026-10-02 oct02
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import bench.score as S  # noqa: E402

scorer, snap, label = sys.argv[1], Path(sys.argv[2]).resolve(), sys.argv[3]
with tempfile.TemporaryDirectory(dir=ROOT / "results/diag_excess") as tmp:
    tmp = Path(tmp)
    for tool in S.TOOLS:
        (tmp / tool).symlink_to(snap if tool == "peak3d" else S.RES / tool)
    (tmp / "inputs").symlink_to(S.RES / "inputs")
    S.RES = tmp
    df = getattr(S, f"score_{scorer}")()
df = df[df["tool"] == "peak3d"].assign(tool=f"peak3d ({label})")
out = ROOT / f"results/diag_excess/{scorer}_{label}.tsv"
df.to_csv(out, sep="\t", index=False)
print(out, len(df))
