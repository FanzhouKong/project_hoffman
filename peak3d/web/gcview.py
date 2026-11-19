"""The GC-MS part of the results for the viewer (the Compounds tab): what `peak3d gc` wrote to <output>/gc as one
packed block (bundle.pack), and the chromatograms of one compound's ions in one file, read on request.

Block (header["kind"] == "peak3d-gc"), n compounds in retention order, N files:
  header   stems, name / formula / inchikey / ref_file per compound, summary (gc/params.json in short), digits
           (decimals worth showing in an m/z: 0 for nominal masses, 4 for accurate ones), hits (per-hit text columns)
  c_*      per compound: id, rt (minutes, aligned), ri, qmz (quantifier m/z), bmz (base peak), nions, nshared,
           nfiles, height (the quantifier's, reference file), ref (file index), refc (component there), match,
           reverse, lri, dri (the annotation's scores and library index; nan without one), riex (index extrapolated)
  s_*      the compounds' spectra (reference file): off (n + 1), mz, ab, sh (1: the ion also needed another
           component's model)
  q        n x N, row-major: the quantifier's abundance in each file (nan: compound not found there)
  h_*      library hits, best first per compound: off (n + 1), rank, match, reverse, nmatch, lri, dri, spec (index
           of the hit's library spectrum, -1: not stored); hs_off / hs_mz / hs_in: those spectra"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import __version__
from . import bundle

FORMAT = 1
TRACE_IONS = 8             # ions per chromatogram plot: the quantifier and the most abundant others
TEXT = ("name", "formula", "inchikey")
# files of <output>/gc the app offers for download
GC_FILES = ("compounds.tsv", "compound_table.tsv", "spectra.msp", "identifications.tsv", "ladder.tsv", "params.json")


def gc_dir(output) -> Path:
    return Path(output) / "gc"


def has_gc(output) -> bool:
    return (gc_dir(output) / "compounds.tsv").exists() and (gc_dir(output) / "compound_table.tsv").exists()


def stamp(output) -> tuple:
    """changes when the GC results do (the server's and the report's cache key)"""
    d = gc_dir(output)
    return tuple((d / n).stat().st_mtime_ns if (d / n).exists() else 0
                 for n in ("compounds.tsv", "compound_table.tsv", "identifications.tsv", "hit_spectra.npz"))


def _table(path: Path, text=()) -> pd.DataFrame:
    """a table of the GC stages: empty cells are missing numbers, text columns stay text (a file or compound named
    NA, a numeric file name)"""
    t = pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""], dtype={c: str for c in text})
    for c in text:
        if c in t:
            t[c] = t[c].fillna("").astype(str)
    return t


def _num(t: pd.DataFrame, name: str, dtype, default=np.nan) -> np.ndarray:
    if name not in t:
        return np.full(len(t), default, dtype)
    v = pd.to_numeric(t[name], errors="coerce")
    return (v.fillna(default) if np.issubdtype(np.dtype(dtype), np.integer) else v).to_numpy(dtype)


def _stems(path: Path) -> list[str]:
    """the file columns of compound_table.tsv (after compound_id, rt, ri, quant_mz, name), as written"""
    with open(path, encoding="utf-8") as fh:
        return fh.readline().rstrip("\r\n").split("\t")[5:]


def build(output) -> bytes:
    """the packed block for one output folder (raw bytes; callers gzip)"""
    output = Path(output)
    d = gc_dir(output)
    if not has_gc(output):
        raise FileNotFoundError(f"no GC results in {output} (peak3d gc, or peak3d process --mode gc, writes gc/)")
    comp = _table(d / "compounds.tsv", TEXT + ("reference_file",))
    n = len(comp)
    stems = _stems(d / "compound_table.tsv")
    N = len(stems)
    q = pd.read_csv(d / "compound_table.tsv", sep="\t", header=0).iloc[:, 5:].to_numpy(np.float32) if n and N \
        else np.zeros((n, N), np.float32)
    if q.shape != (n, N):
        raise ValueError("gc/compound_table.tsv does not match gc/compounds.tsv")
    file_at = {s: i for i, s in enumerate(stems)}
    ref = np.array([file_at.get(s, -1) for s in comp["reference_file"]], np.int32) if n else np.zeros(0, np.int32)
    refc = _num(comp, "reference_component", np.int64, -1)

    # spectra: each compound's reference component, from that file's components
    mzs, abs_, shs, counts = [None] * n, [None] * n, [None] * n, np.zeros(n, np.int64)
    for f in np.unique(ref[ref >= 0]):
        with np.load(d / "components" / f"{stems[f]}.npz") as z:
            off, mz, ab, sh = z["off"], z["mz"], z["abundance"], z["shared"]
        for k in np.flatnonzero(ref == f):
            c = int(refc[k])
            if 0 <= c < len(off) - 1:
                a, b = int(off[c]), int(off[c + 1])
                mzs[k], abs_[k], shs[k], counts[k] = mz[a:b], ab[a:b], sh[a:b], b - a
    have = [k for k in range(n) if counts[k]]
    s_mz = np.concatenate([mzs[k] for k in have]).astype(np.float64) if have else np.zeros(0)
    s_ab = np.concatenate([abs_[k] for k in have]).astype(np.float32) if have else np.zeros(0, np.float32)
    s_sh = np.concatenate([shs[k] for k in have]).astype(np.uint8) if have else np.zeros(0, np.uint8)
    # decimals worth showing in an m/z: none for nominal masses (unit-resolution runs), one for masses on a
    # 0.1 grid, else four
    digits = 4
    if len(s_mz) and np.all(np.abs(s_mz - np.round(s_mz)) < 1e-6):
        digits = 0
    elif len(s_mz) and np.all(np.abs(s_mz * 10 - np.round(s_mz * 10)) < 1e-3):
        digits = 1

    arrays = {"c_id": _num(comp, "compound_id", np.int32, 0), "c_rt": _num(comp, "rt", np.float64),
              "c_ri": _num(comp, "ri", np.float32), "c_qmz": _num(comp, "quant_mz", np.float64),
              "c_bmz": _num(comp, "base_mz", np.float64), "c_nions": _num(comp, "n_ions", np.int32, 0),
              "c_nshared": _num(comp, "n_shared_ions", np.int32, 0), "c_nfiles": _num(comp, "n_files", np.int32, 0),
              "c_height": _num(comp, "height", np.float32), "c_ref": ref, "c_refc": refc.astype(np.int32),
              "c_match": _num(comp, "match", np.float32), "c_reverse": _num(comp, "reverse", np.float32),
              "c_lri": _num(comp, "library_ri", np.float32), "c_dri": _num(comp, "delta_ri", np.float32),
              "c_riex": (comp["ri_extrapolated"].astype(str).str.lower() == "true").to_numpy(np.uint8)
              if "ri_extrapolated" in comp else np.zeros(n, np.uint8),
              "s_off": np.concatenate([[0], np.cumsum(counts)]).astype(np.int32), "s_mz": s_mz, "s_ab": s_ab, "s_sh": s_sh,
              "q": q.reshape(-1)}

    # library hits, and the library spectra kept for the best of them
    hits_head = None
    hp = d / "identifications.tsv"
    hits = _table(hp, TEXT + ("library",)) if hp.exists() else pd.DataFrame()
    if len(hits) and n:
        row_of = {int(c): k for k, c in enumerate(arrays["c_id"])}
        hits = hits.assign(_row=[row_of.get(int(c), -1) for c in hits["compound_id"]])
        hits = hits[hits["_row"] >= 0].sort_values(["_row", "rank"], kind="stable").reset_index(drop=True)
        rows = hits["_row"].to_numpy(np.int64)
        spec = np.full(len(hits), -1, np.int32)
        hs_off, hs_mz, hs_in = np.zeros(1, np.int32), np.zeros(0, np.float64), np.zeros(0, np.float32)
        sp = d / "hit_spectra.npz"
        if sp.exists():
            with np.load(sp) as z:
                at = {(int(c), int(r)): i for i, (c, r) in enumerate(zip(z["compound_id"], z["rank"]))}
                hs_off, hs_mz, hs_in = z["off"].astype(np.int32), z["mz"].astype(np.float64), z["inten"].astype(np.float32)
            spec = np.array([at.get((int(c), int(r)), -1) for c, r in zip(hits["compound_id"], hits["rank"])], np.int32)
        arrays.update({"h_off": np.searchsorted(rows, np.arange(n + 1)).astype(np.int32),
                       "h_rank": _num(hits, "rank", np.int32, 0), "h_match": _num(hits, "match", np.float32),
                       "h_reverse": _num(hits, "reverse", np.float32), "h_nmatch": _num(hits, "n_matched", np.int32, 0),
                       "h_lri": _num(hits, "library_ri", np.float32), "h_dri": _num(hits, "delta_ri", np.float32),
                       "h_spec": spec, "hs_off": hs_off, "hs_mz": hs_mz, "hs_in": hs_in})
        hits_head = {c: hits[c].tolist() if c in hits else [""] * len(hits) for c in TEXT + ("library",)}

    p = json.loads((d / "params.json").read_text()) if (d / "params.json").exists() else {}
    run = json.loads((output / "params.json").read_text()) if (output / "params.json").exists() else {}
    summary = {k: p.get(k) for k in ("peak3d_version", "n_files", "n_components", "n_compounds", "rt_tol_min", "min_presence",
                                     "ladder", "ladder_table", "n_ladder", "min_similarity", "ri_tol", "stage_times_s",
                                     "unit_mass")}
    summary["libraries"] = [Path(x).name for x in p.get("libraries") or []]
    summary["picked_in"] = run.get("mode")
    header = {"format": FORMAT, "kind": "peak3d-gc", "title": output.resolve().name, "viewer_version": __version__,
              "n": n, "stems": stems, "digits": digits, "summary": summary,
              "ref_file": comp["reference_file"].tolist() if n else [],
              **{c: (comp[c].tolist() if c in comp else [""] * n) for c in TEXT}, "hits": hits_head}
    return bundle.pack(bundle._clean(header), arrays)


def blob(output, level: int = 6) -> bytes:
    """build, gzip'd"""
    return gzip.compress(build(output), compresslevel=level, mtime=0)


def trace(output, cache_dir, compound_id: int, stem: str | None = None, n_ions: int = TRACE_IONS) -> dict:
    """one compound in one file (default: its reference file): the chromatograms of the quantifier and its most
    abundant other ions around the component (each ion's own m/z window, as the deconvolution read it), with the
    component's model profile. Times in seconds on the file's own (native) axis."""
    from ..gc.deconv import ion_tol, params_from_dict
    from ..io import eic_kernel
    output = Path(output)
    d = gc_dir(output)
    if not has_gc(output):
        raise FileNotFoundError(f"no GC results in {output}")
    comp = _table(d / "compounds.tsv", TEXT + ("reference_file",))
    row = comp[comp["compound_id"] == int(compound_id)]
    if not len(row):
        raise KeyError(f"no compound #{int(compound_id)}")
    row = row.iloc[0]
    stems = _stems(d / "compound_table.tsv")
    vals = pd.read_csv(d / "compound_table.tsv", sep="\t", header=0)
    vals = vals[vals.iloc[:, 0] == int(compound_id)].iloc[:, 5:].to_numpy(np.float64)
    found = [s for s, v in zip(stems, vals[0] if len(vals) else []) if v == v]
    stem = str(stem) if stem not in (None, "") else row["reference_file"]
    if stem not in stems:
        raise KeyError(f"not a file of this output: {stem!r}")
    ct = _table(d / "components" / f"{stem}.tsv")
    c = ct[ct["compound_id"] == int(compound_id)] if "compound_id" in ct else ct.iloc[:0]
    if not len(c):
        return {"file": stem, "files": found, "error": f"compound #{int(compound_id)} was not found in {stem}"}
    c = c.iloc[0]
    k = int(c["component_id"])
    with np.load(d / "components" / f"{stem}.npz") as z:
        off = z["off"]
        mz, ab, sh = (z[x][int(off[k]):int(off[k + 1])] for x in ("mz", "abundance", "shared"))
        model = z["model"][int(z["model_off"][k]):int(z["model_off"][k + 1])] if "model" in z.files else None
    pj = json.loads((output / "features" / f"{stem}.params.json").read_text())
    P = params_from_dict(pj)
    scale = float(pj.get("mass_scale") or 1.0)      # a unit-resolution run: the cloud holds nominal mass x scale
    cloud, source = bundle._source_cloud(output, Path(cache_dir), stem)
    lo, hi = int(c["scan_min"]), int(c["scan_max"])
    pad = max(10, hi - lo)
    s0, s1 = max(0, lo - pad), min(cloud.n_scans, hi + pad + 1)
    qmz = float(row["quant_mz"])
    quant = int(np.argmin(np.abs(mz - qmz))) if len(mz) else -1
    # each file reports the compound's quantifier; the others: this component's own most abundant ions
    order = [quant] + [int(i) for i in np.argsort(-ab, kind="stable") if int(i) != quant] if quant >= 0 else []
    tol = ion_tol(mz * scale, ab, P)
    top = float(ab.max()) if len(ab) else 1.0
    ions = []
    for i in order[:max(1, int(n_ions))]:
        y = eic_kernel(cloud.off, cloud.mz, cloud.inten, float(mz[i]) * scale, float(tol[i]), s0, s1)
        ions.append({"mz": float(mz[i]), "abundance": float(ab[i]), "rel": float(100.0 * ab[i] / top) if top > 0 else 0.0,
                     "shared": bool(sh[i]), "quant": i == quant, "y": [float(f"{v:.5g}") for v in y.tolist()]})
    rt = 60.0 * np.asarray(cloud.rt, np.float64)
    out = {"file": stem, "files": found, "source": source, "component": k, "rt_s": np.round(rt[s0:s1], 3).tolist(),
           "apex_s": float(60.0 * c["rt"]), "rt_min_s": float(rt[lo]), "rt_max_s": float(rt[hi]), "ions": ions,
           "n_ions": int(len(mz)), "model": None}
    if model is not None and len(model) == hi - lo + 1:
        out["model"] = {"rt_s": np.round(rt[lo:hi + 1], 3).tolist(), "y": [float(f"{v:.4g}") for v in model.tolist()]}
    return out
