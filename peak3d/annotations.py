"""Feature annotations: the library search's result and the user's own choice, kept apart.

identifications.tsv lists every library hit of both MS2 copies of each feature (library.identify). A feature's
default annotation is its best hit on the consensus spectrum (on the single spectrum for outputs made before there
was a consensus) when that scores at least the search's min_score. A user can replace it with any other listed hit
or mark the feature unknown; those choices live in user_annotations.json, keyed by group id, with the hit's library
fields copied in, so they outlast a new search. feature_annotations.tsv holds the annotation in force for every
feature with MS2 and where it came from (library, user, or unknown set by the user); the report and the exported
tables use it."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

USER_FILE = "user_annotations.json"
TABLE_FILE = "feature_annotations.tsv"
COLUMNS = ["group_id", "annotation", "source", "score", "spectrum", "adduct", "formula", "inchikey", "smiles",
           "best_hit", "best_score"]
TEXT = ("name", "precursor_type", "formula", "inchikey", "smiles", "spectrum")


def load_user(output) -> dict[int, dict]:
    """the user's choices: {group_id: {"status": "user" | "unknown", ...hit fields}}"""
    p = Path(output) / USER_FILE
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {int(k): v for k, v in (d.get("annotations") or {}).items()}


def _save_user(output, ann: dict) -> None:
    p = Path(output) / USER_FILE
    if not ann:
        p.unlink(missing_ok=True)
        return
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps({"version": 1, "annotations": {str(k): ann[k] for k in sorted(ann)}}, indent=1),
                   encoding="utf-8")
    tmp.replace(p)                      # never a half-written file, even if the app is closed mid-save


_HITS: dict = {}          # the last identifications.tsv read, by (path, size, mtime): one entry, a few MB


def read_hits(output) -> pd.DataFrame | None:
    """identifications.tsv with text columns as text and a spectrum column (single for older outputs); None if absent.
    Kept until the file changes (a user annotating feature after feature reads it every time); do not modify"""
    p = Path(output) / "identifications.tsv"
    if not p.exists():
        return None
    st = p.stat()
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    if key in _HITS:
        return _HITS[key]
    h = pd.read_csv(p, sep="\t", dtype={c: str for c in TEXT})
    if "spectrum" not in h:
        h["spectrum"] = "single"
    h = h.fillna({c: "" for c in TEXT if c in h})
    _HITS.clear()
    _HITS[key] = h
    return h


def _search_info(output) -> dict:
    p = Path(output) / "identifications.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except ValueError:
        return {}


def default_copy(output) -> str:
    return "consensus" if (Path(output) / "ms2_consensus.msp").exists() else "single"


def set_user(output, group_id: int, choice) -> dict | None:
    """record the user's annotation of one feature. choice: {"spectrum": ..., "rank": ...} = that listed hit;
    "unknown" = no compound; None = back to the library result. Rewrites feature_annotations.tsv; returns the entry
    stored (None after a reset). KeyError when the hit is not in identifications.tsv."""
    output = Path(output)
    ann = load_user(output)
    g = int(group_id)
    if choice is None:
        ann.pop(g, None)
        entry = None
    elif choice == "unknown":
        entry = ann[g] = {"status": "unknown", "set": time.strftime("%Y-%m-%d %H:%M:%S")}
    else:
        h = read_hits(output)
        spec, rank = str(choice.get("spectrum", "")), int(choice.get("rank", 0))
        row = None if h is None else h[(h["group_id"] == g) & (h["spectrum"] == spec) & (h["rank"] == rank)]
        if row is None or not len(row):
            raise KeyError(f"feature {g} has no hit {rank} on the {spec} spectrum")
        r = row.iloc[0]
        info = _search_info(output)
        entry = ann[g] = {"status": "user", **{c: str(r[c]) for c in TEXT}, "rank": rank,
                          "score": float(r["score"]), "library_index": int(r["library_index"]),
                          "library": info.get("library", ""), "method": info.get("method", ""),
                          "set": time.strftime("%Y-%m-%d %H:%M:%S")}
    _save_user(output, ann)
    write_table(output)
    return entry


def effective(output) -> pd.DataFrame:
    """the annotation in force for every feature with MS2 (and any the user annotated): COLUMNS"""
    output = Path(output)
    mp = output / "ms2_map.tsv"
    groups = set(pd.read_csv(mp, sep="\t", usecols=["group_id"])["group_id"].astype(int)) if mp.exists() else set()
    user = load_user(output)
    groups |= set(user)
    best: dict[int, pd.Series] = {}
    h = read_hits(output)
    min_score = float(_search_info(output).get("min_score", 0.7))
    if h is not None and len(h):
        top = h[(h["rank"] == 1) & (h["spectrum"] == default_copy(output))]
        best = {int(r.group_id): r for r in top.itertuples(index=False)}
    rows = []
    for g in sorted(groups):
        b = best.get(g)
        row = {"group_id": g, "annotation": "", "source": "", "score": np.nan, "spectrum": "", "adduct": "",
               "formula": "", "inchikey": "", "smiles": "", "best_hit": b.name if b is not None else "",
               "best_score": float(b.score) if b is not None else np.nan}
        u = user.get(g)
        if u and u.get("status") == "unknown":
            row.update(annotation="unknown", source="user")
        elif u:
            row.update(annotation=u.get("name", ""), source="user", score=u.get("score", np.nan),
                       spectrum=u.get("spectrum", ""), adduct=u.get("precursor_type", ""), formula=u.get("formula", ""),
                       inchikey=u.get("inchikey", ""), smiles=u.get("smiles", ""))
        elif b is not None and b.score >= min_score:
            row.update(annotation=b.name, source="library", score=float(b.score), spectrum=b.spectrum,
                       adduct=b.precursor_type, formula=b.formula, inchikey=b.inchikey, smiles=b.smiles)
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def write_table(output) -> Path:
    p = Path(output) / TABLE_FILE
    effective(output).to_csv(p, sep="\t", index=False, float_format="%.4f")
    return p


SEARCH_FILES = ("identifications.tsv", "identifications.msp", "identifications.json", TABLE_FILE)


def retire(output, keep_search: bool) -> None:
    """after the groups were rebuilt: move user_annotations.json aside (user_annotations.<time>.json, the ids it
    names now point at other features) and remove the earlier search's files (keep_search: it is about to be
    repeated, so say nothing)"""
    from .pipeline import log
    output = Path(output)
    p = output / USER_FILE
    if p.exists():
        old = output / f"user_annotations.{time.strftime('%Y%m%d-%H%M%S')}.json"
        p.replace(old)
        log(f"groups rebuilt: your annotations were moved to {old.name} (their group ids no longer apply)")
    gone = [n for n in SEARCH_FILES if (output / n).exists()]
    for n in gone:
        (output / n).unlink()
    if gone and not keep_search:
        log("groups rebuilt: the earlier identity search was removed (run it again with --library)")
