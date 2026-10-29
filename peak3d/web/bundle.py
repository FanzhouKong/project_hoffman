"""Results bundle for the HTML viewer: one binary blob built from a peak3d output folder.

Layout (little-endian): uint32 header length | header JSON (utf-8) | zero padding to 8 bytes | arrays.
The header lists every array by name with its dtype, byte offset (from the start of the array block, a
multiple of 8) and length, so the viewer maps them as typed arrays without parsing text.

Per-file values are stored log-quantised (uint16, 0 = zero; 0.04 % relative precision) and chromatograms
as uint16 fractions of each group's maximum on a common corrected-RT grid, so a report stays small and
opens instantly. The exact numbers stay in the TSV tables next to it."""
from __future__ import annotations

import gzip
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .. import __version__
from ..io import Cloud, _lower_bound, _upper_bound

FORMAT = 1
BASE_COLS = ("group_id", "mz", "rt", "rt_min", "rt_max", "n_detected", "n_confirmed", "n_filled", "mz_ppm_spread",
             "rt_sd", "iso_offset", "charge")
# defaults keep a report of a few hundred files under ~15 MB
EIC_BUDGET = 3_000_000      # chromatogram values (uint16) across all groups
EIC_FILES = 16              # traces per group; larger studies show an evenly spaced subset of files
EIC_POINTS = 48             # grid points per trace
MAX_CELLS = 4_000_000       # group x file cells stored for the per-file view


# ------------------------------------------------------------------ packing ----
def pack(header: dict, arrays: dict) -> bytes:
    """header + named numpy arrays -> one blob (see module docstring)"""
    meta, chunks, pos = {}, [], 0
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        if a.dtype.byteorder == ">":
            a = a.astype(a.dtype.newbyteorder("<"))
        meta[name] = {"dtype": a.dtype.str.lstrip("<|="), "offset": pos, "length": int(a.size)}
        b = a.tobytes()
        chunks.append(b)
        pad = (-len(b)) % 8
        chunks.append(b"\0" * pad)
        pos += len(b) + pad
    h = json.dumps({**header, "arrays": meta}, separators=(",", ":"), allow_nan=False).encode()
    lead = 4 + len(h)
    return struct.pack("<I", len(h)) + h + b"\0" * ((-lead) % 8) + b"".join(chunks)


def unpack(blob: bytes):
    """blob -> (header, {name: array}); the inverse of pack"""
    (n,) = struct.unpack_from("<I", blob, 0)
    header = json.loads(blob[4:4 + n])
    start = 4 + n + ((-(4 + n)) % 8)
    arrays = {k: np.frombuffer(blob, dtype=np.dtype(m["dtype"]).newbyteorder("<"), count=m["length"],
                               offset=start + m["offset"]) for k, m in header.pop("arrays").items()}
    return header, arrays


def _clean(x):
    """JSON-safe copy: NaN/inf -> None, numpy scalars -> Python"""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _records(df: pd.DataFrame) -> list[dict]:
    return [_clean(r) for r in df.to_dict("records")]


# ------------------------------------------------------------------ chromatograms ----
@njit(cache=True)
def eic_on_grid(off, mz, inten, rt, targets, tols, x0, dx, n_pts):
    """per group: the max-intensity EIC within targets +/- tols, on the grid x0 + j*dx (j < n_pts) of the
    scan times ``rt``. Scans falling in a grid cell are max-pooled; empty cells interpolate linearly
    between the neighbouring scans, so a grid finer than the scan spacing does not leave holes."""
    n = len(targets)
    S = len(rt)
    out = np.zeros((n, n_pts), np.float32)
    seen = np.zeros(n_pts, np.bool_)
    for g in range(n):
        lo_t = x0[g] - 0.5 * dx[g]
        hi_t = x0[g] + (n_pts - 0.5) * dx[g]
        a = max(np.searchsorted(rt, lo_t) - 1, 0)
        b = min(np.searchsorted(rt, hi_t) + 1, S)
        m = b - a
        if m <= 0:
            continue
        vals = np.zeros(m, np.float32)
        for s in range(a, b):
            lo = _lower_bound(mz, off[s], off[s + 1], targets[g] - tols[g])
            hi = _upper_bound(mz, lo, off[s + 1], targets[g] + tols[g])
            v = np.float32(0.0)
            for p in range(lo, hi):
                if inten[p] > v:
                    v = inten[p]
            vals[s - a] = v
        seen[:] = False
        for s in range(a, b):
            j = int(math.floor((rt[s] - lo_t) / dx[g]))
            if 0 <= j < n_pts:
                if (not seen[j]) or vals[s - a] > out[g, j]:
                    out[g, j] = vals[s - a]
                seen[j] = True
        for j in range(n_pts):
            if seen[j]:
                continue
            t = x0[g] + j * dx[g]
            k = np.searchsorted(rt[a:b], t)
            if k <= 0 or k >= m:
                continue
            t0 = rt[a + k - 1]
            t1 = rt[a + k]
            w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            out[g, j] = (1.0 - w) * vals[k - 1] + w * vals[k]
    return out


def _eic_files(n_files: int, limit: int) -> list[int]:
    if n_files <= limit:
        return list(range(n_files))
    return sorted(set(np.linspace(0, n_files - 1, limit).round().astype(int).tolist()))


def _chromatograms(output: Path, cache_dir: Path, stems, table, files_info, ppm, budget, max_files, n_pts):
    """corrected-RT chromatograms of the most intense groups in up to max_files files, or (None, reason)"""
    if not cache_dir.is_dir():
        return None, "the point-cloud cache was removed after the run"
    shown = _eic_files(len(stems), max_files)
    missing = [stems[k] for k in shown if not (cache_dir / f"{stems[k]}.npz").exists()]
    if missing:
        return None, f"cache files missing for {len(missing)} files"
    G = len(table)
    if G == 0:
        return None, "no groups"
    F = len(shown)
    n_groups = int(min(G, max(1, budget // (F * n_pts))))
    order = np.argsort(-table["_max"].to_numpy(), kind="stable")[:n_groups]
    rows = np.sort(order)
    t = table.iloc[rows]
    rt_lo = t["rt_min"].to_numpy(float)
    rt_hi = t["rt_max"].to_numpy(float)
    pad = np.maximum(0.75 * (rt_hi - rt_lo), 0.1)
    x0 = rt_lo - pad
    dx = (rt_hi + pad - x0) / (n_pts - 1)
    mz_c = t["mz"].to_numpy(float)
    traces = np.zeros((n_groups, F, n_pts), np.float32)
    offsets = {r["stem"]: (r.get("mz_offset_ppm") or 0.0) for r in files_info}
    for j, k in enumerate(shown):
        s = stems[k]
        cloud = Cloud.load_npz(cache_dir / f"{s}.npz")
        warp = output / "rt_correction" / f"{s}.tsv"
        if warp.exists():
            corr = pd.read_csv(warp, sep="\t")["rt_corr"].to_numpy(float)
            if len(corr) != cloud.n_scans:
                corr = cloud.rt
        else:
            corr = cloud.rt
        target = mz_c / (1.0 - offsets.get(s, 0.0) * 1e-6)
        traces[:, j, :] = eic_on_grid(cloud.off, cloud.mz, cloud.inten, np.asarray(corr, np.float64),
                                      target, target * ppm * 1e-6, x0, dx, n_pts)
    scale = traces.reshape(n_groups, -1).max(axis=1)
    q = np.zeros_like(traces, dtype=np.uint16)
    ok = scale > 0
    q[ok] = np.round(traces[ok] / scale[ok, None, None] * 65535).astype(np.uint16)
    info = {"files": shown, "points": n_pts, "n": n_groups, "ppm": ppm, "of": G,
            "selection": "most intense groups" if n_groups < G else "all groups"}
    arrays = {"eic_row": rows.astype(np.int32), "eic_x0": x0.astype(np.float32), "eic_dx": dx.astype(np.float32),
              "eic_scale": scale.astype(np.float32), "eic_q": q.reshape(-1)}
    return (info, arrays), None


# ------------------------------------------------------------------ MS2 and identifications ----
def _take(off, arrays, idx):
    """rows idx of a ragged table (off: n + 1 offsets into every array in arrays) -> (offsets, arrays)"""
    idx = np.asarray(idx, np.int64)
    lens = (off[1:] - off[:-1])[idx]
    new = np.zeros(len(idx) + 1, np.int64)
    np.cumsum(lens, out=new[1:])
    sel = np.repeat(off[idx] - new[:-1], lens) + np.arange(new[-1])
    return new.astype(np.int32), [np.asarray(a)[sel] for a in arrays]


ID_TEXT = ("name", "precursor_type", "formula", "inchikey", "smiles")


def ms2_ids(output, group_ids) -> tuple[dict, dict]:
    """the representative MS2 spectrum of every group that has one (ms2.read_representatives) and the hits of the last
    identity search (identifications.tsv, .msp, .json), as header parts and arrays keyed by row of the aligned table
    (group_ids in table order). Header keys ms2 / ids are None when the output has no MS2 / no search"""
    output = Path(output)
    row_of = {int(g): i for i, g in enumerate(np.asarray(group_ids))}
    head, arrays = {"ms2": None, "ids": None}, {}
    from ..ms2 import read_representatives
    reps = read_representatives(output)
    if reps is not None and len(reps["group_id"]):
        k = np.array([i for i, g in enumerate(reps["group_id"]) if int(g) in row_of], np.int64)
        off, (mz, inten) = _take(np.asarray(reps["off"]), (reps["mz"], reps["inten"]), k)
        arrays.update(ms2_row=np.array([row_of[int(reps["group_id"][i])] for i in k], np.int32),
                      ms2_n=np.clip(np.asarray(reps["n_ms2"])[k], 0, 65535).astype(np.uint16),
                      ms2_prec=np.asarray(reps["prec_mz"], np.float64)[k],
                      ms2_rt=np.asarray(reps["rt_s"], np.float32)[k], ms2_off=off,
                      ms2_mz=mz.astype(np.float32), ms2_int=inten.astype(np.float32))
        head["ms2"] = {"n": int(len(k)), "file": [str(reps["file"][i]) for i in k],
                       "scan": [str(reps["scan"][i]) for i in k]}
    tsv = output / "identifications.tsv"
    if head["ms2"] is not None and tsv.exists():
        hits = pd.read_csv(tsv, sep="\t", dtype={c: str for c in ID_TEXT})
        meta = json.loads((output / "identifications.json").read_text()) if (output / "identifications.json").exists() else {}
        lib = None
        if (output / "identifications.msp").exists():
            from ..library import read_msp
            lib = read_msp(output / "identifications.msp")
            if len(lib.names) != len(hits):        # written together; a mismatch means a damaged pair
                lib = None
        hits["_row"] = [row_of.get(int(g), -1) for g in hits["group_id"]]
        hits["_hit"] = np.arange(len(hits))
        hits = hits[hits["_row"] >= 0].sort_values(["_row", "rank"], kind="stable")
        num = lambda c, d=np.nan: hits[c].to_numpy(np.float64) if c in hits else np.full(len(hits), d)
        arrays.update(id_row=hits["_row"].to_numpy(np.int32), id_rank=num("rank", 1).astype(np.uint8),
                      id_score=num("score").astype(np.float32),
                      id_nmatch=np.clip(np.nan_to_num(num("n_matched", 0)), 0, 65535).astype(np.uint16),
                      id_ppm=num("ppm_error").astype(np.float32), id_prec=num("library_precursor_mz"))
        if lib is not None:
            off, (mz, inten) = _take(np.asarray(lib.off), (lib.mz, lib.inten), hits["_hit"].to_numpy())
            arrays.update(id_off=off, id_mz=mz.astype(np.float32), id_int=inten.astype(np.float32))
        head["ids"] = {**meta, "n": int(len(hits)), "spectra": lib is not None,
                       **{c: hits[c].fillna("").astype(str).tolist() if c in hits else [""] * len(hits)
                          for c in ID_TEXT}}
    return head, arrays


def ms2_ids_blob(output) -> bytes:
    """gzip'd bundle holding only the MS2 and identification parts, for a viewer that already has the rest"""
    ids = pd.read_csv(Path(output) / "aligned_feature_table.tsv", sep="\t", usecols=["group_id"])["group_id"].to_numpy()
    head, arrays = ms2_ids(output, ids)
    return gzip.compress(pack(_clean({"format": FORMAT, "kind": "peak3d-ms2", **head}), arrays), compresslevel=6, mtime=0)


# ------------------------------------------------------------------ build ----
def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, sep="\t") if path.exists() else None


def _hist(values, lim, n_bins=61):
    bins = np.linspace(-lim, lim, n_bins + 1)
    return np.histogram(np.clip(values, -lim, lim), bins)[0].tolist()


def _mad(v):
    v = np.asarray(v, float)
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else None


def build(output, cache_dir=None, *, eic_budget: int = EIC_BUDGET, eic_files: int = EIC_FILES,
          eic_points: int = EIC_POINTS, max_cells: int = MAX_CELLS) -> bytes:
    """bundle for one output folder (raw bytes; callers gzip). cache_dir defaults to <output>/.cache"""
    output = Path(output)
    cache_dir = Path(cache_dir) if cache_dir else output / ".cache"
    table_path = output / "aligned_feature_table.tsv"
    if not table_path.exists():
        raise FileNotFoundError(f"not a peak3d output folder (no aligned_feature_table.tsv): {output}")
    params = json.loads((output / "params.json").read_text()) if (output / "params.json").exists() else {}
    options = params.get("options", {})
    table = pd.read_csv(table_path, sep="\t")
    man = _read(output / "features" / "manifest.tsv")
    stems = [c for c in table.columns if c not in BASE_COLS]
    if man is not None and "stem" in man and set(man["stem"].astype(str)) <= set(stems):
        stems = [s for s in stems if s in set(man["stem"].astype(str))]   # extra columns of newer versions
    G, N = len(table), len(stems)
    V = table[stems].to_numpy(np.float32) if N else np.zeros((G, 0), np.float32)
    table["_max"] = V.max(axis=1) if N else 0.0

    # files: picking manifest + alignment summary, in table order
    summ = _read(output / "rt_correction" / "summary.tsv")
    files = pd.DataFrame({"stem": stems})
    if man is not None and "stem" in man:
        files = files.merge(man.drop(columns=[c for c in ("path",) if c in man]), on="stem", how="left")
    if summ is not None and "file" in summ:
        files = files.merge(summ.rename(columns={"file": "stem"}), on="stem", how="left")
    files_info = _records(files)

    # groups
    def col(name, dtype, default=0):
        return table[name].to_numpy(dtype) if name in table else np.full(G, default, dtype)

    nz = V > 0
    cnt = nz.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(nz, V, 0).sum(axis=1) / cnt
        var = np.where(nz, (V - mean[:, None]) ** 2, 0).sum(axis=1) / np.maximum(cnt - 1, 1)
        cv = np.where(cnt >= 2, 100 * np.sqrt(var) / mean, np.nan)
    arrays = {"g_id": col("group_id", np.int32), "g_mz": col("mz", np.float64), "g_rt": col("rt", np.float32),
              "g_rt_min": col("rt_min", np.float32), "g_rt_max": col("rt_max", np.float32),
              "g_ndet": col("n_detected", np.uint16), "g_nconf": col("n_confirmed", np.uint16),
              "g_nfill": col("n_filled", np.uint16), "g_ppm": col("mz_ppm_spread", np.float32, np.nan),
              "g_rtsd": col("rt_sd", np.float32, np.nan), "g_iso": col("iso_offset", np.int8),
              "g_z": col("charge", np.int8), "g_max": table["_max"].to_numpy(np.float32),
              "g_cv": cv.astype(np.float32)}

    # per-file values and how each cell was obtained: 3 detected, 2 gap-filled with a confirmed peak,
    # 1 gap-filled (no confirmed peak), 0 nothing
    matrix = None
    if G * N and G * N <= max_cells:
        pos = V[V > 0]
        lo = math.floor(math.log10(float(pos.min()))) if len(pos) else 0
        hi = max(math.ceil(math.log10(float(pos.max()))) if len(pos) else 1, lo + 1)
        with np.errstate(divide="ignore"):
            qv = np.where(V > 0, 1 + np.round((np.log10(np.maximum(V, 1e-30)) - lo) / (hi - lo) * 65534), 0)
        status = np.where(V > 0, 1, 0).astype(np.uint8)
        pres = _read(output / "presence_mask.tsv")
        if pres is not None and all(s in pres for s in stems):
            p = pres.set_index("group_id").reindex(arrays["g_id"])[stems].fillna(0).to_numpy(np.int8)
            status = np.where(p == 2, 3, np.where(p == 1, 2, status)).astype(np.uint8)
        arrays["m_q"] = qv.astype(np.uint16).reshape(-1)
        arrays["m_status"] = status.reshape(-1)
        matrix = {"log10_lo": lo, "log10_hi": hi}
    matrix_note = None if matrix or not G * N else f"{G} groups x {N} files is above the {max_cells:,} cell limit"

    # RT correction: per-file shift curves (downsampled) and anchor residuals
    dx_, dy_, doff = [], [], [0]
    for s in stems:
        w = _read(output / "rt_correction" / f"{s}.tsv")
        if w is None or not len(w):
            doff.append(doff[-1])
            continue
        idx = np.unique(np.linspace(0, len(w) - 1, min(len(w), 400)).round().astype(int))
        x = w["rt_native"].to_numpy(float)[idx]
        dx_.append(x.astype(np.float32))
        dy_.append((60 * (w["rt_corr"].to_numpy(float)[idx] - x)).astype(np.float32))
        doff.append(doff[-1] + len(idx))
    arrays["drift_x"] = np.concatenate(dx_) if dx_ else np.zeros(0, np.float32)
    arrays["drift_y"] = np.concatenate(dy_) if dy_ else np.zeros(0, np.float32)
    arrays["drift_off"] = np.asarray(doff, np.int32)
    residuals = None
    anc = _read(output / "rt_correction" / "anchors.tsv")
    if anc is not None and len(anc):
        b = anc["residual_before_s"].to_numpy(float)
        a = anc.loc[anc["is_inlier"].astype(bool), "residual_after_s"].to_numpy(float)
        both = np.abs(np.concatenate([b, a]))
        lim = float(np.percentile(both[np.isfinite(both)], 99)) if len(both) else 1.0
        lim = lim if lim > 0 else 1.0
        residuals = {"lim": lim, "before": _hist(b, lim), "after": _hist(a, lim), "n_before": int(len(b)),
                     "n_after": int(len(a)), "mad_before": _mad(b), "mad_after": _mad(a)}

    eic, eic_note = None, None
    ppm = float(options.get("ppm") or 5.0)
    if G and N and eic_budget > 0:
        got, eic_note = _chromatograms(output, cache_dir, stems, table, files_info, ppm, eic_budget, eic_files,
                                       eic_points)
        if got:
            eic, extra = got
            arrays.update(extra)

    try:
        ms2_head, extra = ms2_ids(output, arrays["g_id"])
        arrays.update(extra)
    except Exception as e:   # a damaged MS2 or identification file never costs the rest of the results
        ms2_head = {"ms2": None, "ids": None, "ms2_note": f"{type(e).__name__}: {e}"}

    header = {"format": FORMAT, "kind": "peak3d-results", "title": output.resolve().name,
              "kernel": params.get("kernel", "peak3d"), "kernel_label": params.get("kernel_label") or params.get("kernel", "peak3d"),
              "kernel_version": params.get("kernel_version") or params.get("peak3d_version"), **ms2_head,
              "created": time.strftime("%Y-%m-%d %H:%M"), "viewer_version": __version__,
              "quant": options.get("quant", "height"), "n_groups": G, "stems": stems, "files": files_info,
              "params": _clean(params), "matrix": matrix, "matrix_note": matrix_note, "eic": eic,
              "eic_note": eic_note, "residuals": residuals}
    return pack(_clean(header), arrays)
