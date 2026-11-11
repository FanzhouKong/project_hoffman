"""Leaving flagged files out of the downloads (the aligned tables and the report), on the user's choice in the app:
the files whose RT correction carries a warning in rt_correction/summary.tsv (few_anchors_matched, rt_residual_high).
The detection counts are recomputed over the remaining files and the presence rule is applied again over them; a
feature none of the remaining files detected (it exists only through the left-out files) is dropped. Group m/z and RT
stay the run's consensus over all files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

KINDS = ("few_anchors_matched", "rt_residual_high")
COUNTS = ("n_detected", "n_confirmed", "n_filled")


def file_flags(output) -> dict:
    """stem -> its alignment flag (rt_correction/summary.tsv; empty without it)"""
    p = Path(output) / "rt_correction" / "summary.tsv"
    if not p.exists():
        return {}
    s = pd.read_csv(p, sep="\t", dtype={"file": str, "flag": str}, usecols=lambda c: c in ("file", "flag"))
    if "file" not in s or "flag" not in s:
        return {}
    return dict(zip(s["file"], s["flag"].fillna("")))


def flagged(output) -> dict:
    """kind -> the stems whose alignment flag carries it, for every kind of KINDS"""
    out = {k: [] for k in KINDS}
    for stem, flag in file_flags(output).items():
        parts = set(flag.split(";"))
        for k in KINDS:
            if k in parts:
                out[k].append(stem)
    return out


def parse_kinds(value) -> list:
    """the known kinds in a list or a comma-separated string (others ignored)"""
    items = value.split(",") if isinstance(value, str) else (value or [])
    return [k for k in KINDS if k in {str(x).strip() for x in items}]


def to_drop(output, kinds) -> list:
    """the stems carrying any of kinds, sorted"""
    fl = flagged(output)
    return sorted({s for k in parse_kinds(kinds) for s in fl[k]})


def min_presence(output) -> int:
    """the run's presence rule (params.json options.min_presence; 2, the default, when it is not there)"""
    p = Path(output) / "params.json"
    try:
        v = json.loads(p.read_text()).get("options", {}).get("min_presence") if p.exists() else None
    except (ValueError, AttributeError):
        v = None
    return int(v) if v is not None else 2


def _mask(path: Path, ids, drop):
    """the drop columns of a group x file mask (presence_mask.tsv, filled_mask.tsv) in the order of ids, or None"""
    if not path.exists():
        return None
    head = pd.read_csv(path, sep="\t", nrows=0).columns
    if "group_id" not in head or not all(s in head for s in drop):
        return None
    m = pd.read_csv(path, sep="\t", usecols=["group_id", *drop], dtype={s: np.float64 for s in drop})
    m = m.drop_duplicates("group_id").set_index("group_id").reindex(ids)
    return m[drop].fillna(0).to_numpy(np.int8)


def recount(output, table_path, stems, drop) -> pd.DataFrame:
    """for every row of an aligned table (file columns stems): n_detected, n_confirmed and n_filled over the files not
    in drop, and keep: detected in >= 1 of them, and detected or confirmed in >= the run's presence rule (capped at
    their number). Indexed by group_id. The dropped files' parts come from the table's and the masks' columns of those
    files (presence_mask.tsv: 2 detected, 1 confirmed; filled_mask.tsv: 1 not detected); without presence_mask.tsv
    (external kernels) a value that is not gap-filled counts as detected."""
    output = Path(output)
    stems = [str(s) for s in stems]
    drop = [s for s in stems if s in {str(x) for x in drop}]
    head = pd.read_csv(table_path, sep="\t", nrows=0).columns
    t = pd.read_csv(table_path, sep="\t", usecols=["group_id", *[c for c in COUNTS if c in head], *drop],
                    dtype={s: np.float64 for s in drop})
    ids = t["group_id"].to_numpy()
    V = t[drop].to_numpy(np.float64) if drop else np.zeros((len(t), 0))
    pres = _mask(output / "presence_mask.tsv", ids, drop)
    filled = _mask(output / "filled_mask.tsv", ids, drop)
    if pres is not None:
        det, conf = pres == 2, pres == 1
    else:
        det = (V > 0) & ~(filled == 1) if filled is not None else V > 0
        conf = np.zeros_like(det)
    fil = (filled == 1) & (V > 0) if filled is not None else np.zeros_like(det)

    def col(c):
        return t[c].to_numpy(np.int64) if c in t else np.zeros(len(t), np.int64)

    out = pd.DataFrame({"n_detected": np.maximum(col("n_detected") - det.sum(axis=1), 0),
                        "n_confirmed": np.maximum(col("n_confirmed") - conf.sum(axis=1), 0),
                        "n_filled": np.maximum(col("n_filled") - fil.sum(axis=1), 0)}, index=ids)
    need = max(1, min(min_presence(output), len(stems) - len(drop)))
    if "n_detected" in t:
        out["keep"] = (out["n_detected"] >= 1) & (out["n_detected"] + out["n_confirmed"] >= need)
    else:
        out["keep"] = True
    return out


def subset_frame(output, table: pd.DataFrame, table_path, stems, drop) -> pd.DataFrame:
    """an aligned table read into a DataFrame, without the file columns drop: counts recomputed, rows dropped (recount)"""
    drop = [s for s in stems if s in set(drop)]
    if not drop:
        return table
    rc = recount(output, table_path, stems, drop).reindex(table["group_id"].to_numpy())
    keep = rc["keep"].fillna(False).to_numpy(bool)
    out = table.loc[keep].drop(columns=drop).reset_index(drop=True)
    for c in COUNTS:
        if c in out:
            out[c] = rc[c].to_numpy()[keep].astype(np.int64)
    return out
