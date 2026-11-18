"""`peak3d gc OUTPUT`: the GC stages on an existing peak3d output folder (features/ and .cache/ of `peak3d process`
or `peak3d pick`). Everything is written to OUTPUT/gc/; no file of the LC output is touched.

  components/<file>.tsv           the file's components (deconv.Deconv.components, plus rt_corr and ri)
  components/<file>.npz           their spectra (off, mz, abundance, shared)
  components/<file>.features.tsv  per picked ion: component, correlation with its model, shared flag
  compounds.tsv                   components matched across files: one row per compound, with its best library hit
  compound_table.tsv              compound x file: abundance of the compound's quantifier ion
  spectra.msp                     one spectrum per compound (from the file where it is strongest)
  ladder.tsv                      the retention-index ladder found in each file
  identifications.tsv             the listed library hits of every compound
  hit_spectra.npz                 the library spectra of each compound's best hits (for the viewer)
  params.json                     settings, per-file calibration and timings

Compounds across files: the strongest unmatched component is a reference; in every other file the component within
the alignment's RT tolerance (on the corrected time axis) whose spectrum is most similar to it joins, when the
similarity (accurate-mass cosine, weights intensity^0.6 x m/z) reaches 0.6. The quantifier is the strongest ion of
the reference spectrum that needed no other component's model; a file's value is that ion's abundance in its own
deconvolved spectrum."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .. import __version__
from ..io import Cloud, _lower_bound, _upper_bound
from . import library as L
from . import ri as RI
from .deconv import DeconvParams, deconvolve, params_from_dict

MATCH_MIN = 0.6            # spectrum similarity for the same compound in two files
QUANT_TOL, QUANT_PPM = 0.003, 10.0
HIT_SPECTRA = 3            # library spectra kept per compound, best hits first (the viewer's head-to-tail plot)


def log(msg):
    print(f"[peak3d gc] {msg}", flush=True)


MZ_COLS = ("quant_mz", "base_mz")


def _write(table: pd.DataFrame, path) -> None:
    """a table of the GC stages: six significant digits, the m/z columns to 1e-5 (six digits would round an
    accurate mass at m/z 300 to 1 mDa)"""
    t = table.copy()
    for c in MZ_COLS:
        if c in t:
            t[c] = ["" if v != v else f"{v:.5f}" for v in t[c].astype(float)]
    t.to_csv(path, sep="\t", index=False, float_format="%.6g")


def _deconv_one(job):
    stem, features_dir, cache_dir, out_dir, D, threads = job
    import numba
    numba.set_num_threads(max(1, int(threads)))
    tic = time.perf_counter()
    cloud = Cloud.load_npz(Path(cache_dir) / f"{stem}.npz")
    feats = pd.read_csv(Path(features_dir) / f"{stem}.tsv", sep="\t")
    P = params_from_dict(json.loads((Path(features_dir) / f"{stem}.params.json").read_text()))
    d = deconvolve(cloud, feats, P, D)
    out = Path(out_dir)
    np.savez(out / f"{stem}.npz", off=d.off, mz=d.mz, abundance=d.abundance, shared=d.shared,
             model_off=d.model_off, model=d.model.astype(np.float32))
    pd.DataFrame({"feature_id": feats["feature_id"].values, "component_id": d.feature_component,
                  "model_r": np.round(d.feature_r, 4), "shared": d.feature_shared.astype(np.int8)}) \
        .to_csv(out / f"{stem}.features.tsv", sep="\t", index=False)
    _write(d.components, out / f"{stem}.tsv")
    info = dict(d.info)
    info.pop("params", None)
    info.update(file=stem, seconds=round(time.perf_counter() - tic, 3), fwhm_med=float(P.fwhm_med), dt=float(P.dt),
                mz_lo=float(cloud.mz.min()) if cloud.n_points else 0.0,
                mz_hi=float(cloud.mz.max()) if cloud.n_points else 0.0)
    return info


@njit(cache=True)
def _cosine(ma, wa, mb, wb, tol, ppm):
    num = 0.0
    na = 0.0
    nb = 0.0
    for x in wa:
        na += x * x
    for x in wb:
        nb += x * x
    if na <= 0 or nb <= 0:
        return 0.0
    a = 0
    for p in range(len(mb)):
        d = max(tol, ppm * 1e-6 * mb[p])
        while a < len(ma) and ma[a] < mb[p] - d:
            a += 1
        if a < len(ma) and ma[a] <= mb[p] + d:
            num += wa[a] * wb[p]
            a += 1
    return num / np.sqrt(na * nb)


@njit(cache=True)
def _group(rt, file, order, off, mz, w, rt_tol, min_sim, n_files, tol, ppm):
    """components sorted by corrected RT; order: strongest first. Returns the compound of every component."""
    n = len(rt)
    gid = np.full(n, -1, np.int64)
    best_s = np.zeros(n_files)
    best_i = np.full(n_files, -1, np.int64)
    G = 0
    for q in range(n):
        r = order[q]
        if gid[r] >= 0:
            continue
        gid[r] = G
        lo = _lower_bound(rt, 0, n, rt[r] - rt_tol)
        hi = _upper_bound(rt, lo, n, rt[r] + rt_tol)
        for c in range(lo, hi):
            f = file[c]
            if gid[c] >= 0 or f == file[r]:
                continue
            s = _cosine(mz[off[r]:off[r + 1]], w[off[r]:off[r + 1]], mz[off[c]:off[c + 1]], w[off[c]:off[c + 1]], tol, ppm)
            if s >= min_sim and s > best_s[f]:
                best_s[f] = s
                best_i[f] = c
        for c in range(lo, hi):
            f = file[c]
            if best_i[f] >= 0:
                gid[best_i[f]] = G
                best_i[f] = -1
                best_s[f] = 0.0
        G += 1
    return gid, G


def _corrected(output: Path, stem: str, t):
    f = output / "rt_correction" / f"{stem}.tsv"
    if not f.exists():
        return np.asarray(t, np.float64).copy()
    w = pd.read_csv(f, sep="\t", usecols=["rt_native", "rt_corr"])
    return np.interp(t, w["rt_native"].values, w["rt_corr"].values)


def _ladders(stems, comps, spectra, kind, carbons, table):
    """per-file ladder tables (rt native, rt_corr) and notes; table: a user ladder (applies to every file's
    corrected time axis)"""
    if table is not None:
        t = table.assign(file="", rt_corr=table["rt"], component=-1, series=np.nan, molecular_ion=False)
        return t, ["ladder read from the table given"]
    rows, notes = [], []
    for s in stems:
        c = comps[s]
        off, mz, ab = spectra[s]
        t, note = RI.find_ladder(c["rt"].values, off, mz, ab, kind, carbons)
        if len(t):
            t = t.assign(file=s, rt_corr=np.interp(t["rt"].values, c["rt"].values, c["rt_corr"].values)
                         if len(c) > 1 else t["rt"].values)
            rows.append(t)
        notes.append(f"{s}: {note}")
    cols = ["file", "carbon", "ri", "rt", "rt_corr", "component", "series", "molecular_ion"]
    return (pd.concat(rows, ignore_index=True)[cols] if rows else pd.DataFrame(columns=cols)), notes


def _study_map(ladder: pd.DataFrame):
    """one RT -> RI map on the corrected time axis: the median corrected time of every ladder member"""
    if len(ladder) == 0:
        return None
    g = ladder.groupby("ri")["rt_corr"].median().sort_values()
    g = g[np.concatenate([[True], np.diff(g.values) > 0])]
    if len(g) < 2 or np.any(np.diff(g.index.values) <= 0):
        return None
    return RI.RIMap(g.values, g.index.values)


def _hit_spectra(path: Path, hits: pd.DataFrame, libraries) -> None:
    """the library spectra of these hits (their records in libraries[hit _lib]) as one file: compound_id and rank
    per spectrum, then off / mz / inten"""
    mzs, its, counts = [], [], []
    for k, i in zip(hits["_lib"].to_numpy(np.int64), hits["library_index"].to_numpy(np.int64)):
        lib = libraries[k]
        a, b = int(lib.off[i]), int(lib.off[i + 1])
        mzs.append(np.asarray(lib.mz[a:b], np.float64))
        its.append(np.asarray(lib.inten[a:b], np.float32))
        counts.append(b - a)
    np.savez(path, compound_id=hits["compound_id"].to_numpy(np.int64), rank=hits["rank"].to_numpy(np.int64),
             off=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
             mz=np.concatenate(mzs) if mzs else np.zeros(0), inten=np.concatenate(its) if its else np.zeros(0, np.float32))


def run(output, cores: int = 1, cache_dir=None, ladder: str | None = None, carbons=None, ladder_table=None,
        library=None, min_similarity: float = 0.7, ri_tol: float | None = None, ri_type: str = "semi",
        min_presence: int = 2, deconv: DeconvParams | None = None, search: L.SearchParams | None = None) -> dict:
    """Run the GC stages. ladder: "alkane" or "fame" (find it in the runs), with carbons the expected carbon
    numbers; ladder_table: a ladder file instead (ri.read_ladder). library: EI .msp file(s). ri_tol: index window
    of the search (default: 20 for alkane, 3000 for fame indices; no window without a ladder)."""
    output = Path(output)
    features_dir = output / "features"
    cache_dir = Path(cache_dir or output / ".cache")
    stems = sorted(p.name[:-4] for p in features_dir.glob("*.tsv")
                   if p.name != "manifest.tsv" and not p.name.endswith(".rejected.tsv"))
    if not stems:
        raise FileNotFoundError(f"no feature tables in {features_dir}")
    missing = [s for s in stems if not (cache_dir / f"{s}.npz").exists()]
    if missing:
        raise FileNotFoundError(f"no point-cloud cache for {missing[0]} in {cache_dir} (run peak3d process without "
                                "--no-keep-cache)")
    D = deconv or DeconvParams()
    gc_dir = output / "gc"
    comp_dir = gc_dir / "components"
    comp_dir.mkdir(parents=True, exist_ok=True)
    times = {}

    # 1. deconvolution, file by file
    tic = time.perf_counter()
    n_workers = max(1, min(len(stems), cores))
    jobs = [(s, str(features_dir), str(cache_dir), str(comp_dir), D, max(1, cores // n_workers)) for s in stems]
    if n_workers == 1:
        infos = [_deconv_one(j) for j in jobs]
    else:
        with get_context("spawn").Pool(n_workers) as pool:
            infos = pool.map(_deconv_one, jobs, chunksize=1)
    times["deconvolution_s"] = round(time.perf_counter() - tic, 3)
    log(f"deconvolution: {sum(i['n_components'] for i in infos)} components in {len(stems)} files "
        f"({times['deconvolution_s']} s)")

    comps, spectra = {}, {}
    for s in stems:
        c = pd.read_csv(comp_dir / f"{s}.tsv", sep="\t")
        c["rt_corr"] = _corrected(output, s, c["rt"].values)
        with np.load(comp_dir / f"{s}.npz") as z:
            spectra[s] = (z["off"], z["mz"], z["abundance"])
        comps[s] = c

    # 2. the same compound across files
    tic = time.perf_counter()
    file_of = np.concatenate([np.full(len(comps[s]), k, np.int64) for k, s in enumerate(stems)])
    local = np.concatenate([np.arange(len(comps[s]), dtype=np.int64) for s in stems])
    rt_all = np.concatenate([comps[s]["rt_corr"].values for s in stems])
    tic_all = np.concatenate([comps[s]["tic"].values for s in stems])
    lens = np.concatenate([np.diff(spectra[s][0]) for s in stems])
    mz_all = np.concatenate([spectra[s][1] for s in stems])
    ab_all = np.concatenate([spectra[s][2] for s in stems])
    off_all = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    o = np.argsort(rt_all, kind="stable")
    take = np.concatenate([np.arange(off_all[i], off_all[i + 1]) for i in o]) if len(o) else np.zeros(0, np.int64)
    s_off = np.concatenate([[0], np.cumsum(lens[o])]).astype(np.int64)
    s_mz, s_ab = mz_all[take], ab_all[take]
    params_file = output / "params.json"
    rt_tol = None
    if params_file.exists():
        rt_tol = json.loads(params_file.read_text()).get("rt_tol_min")
    fw = float(np.median([i["fwhm_med"] for i in infos]))
    rt_tol = float(max(rt_tol or 0.0, 0.5 * fw))
    rank = np.argsort(-tic_all[o], kind="stable").astype(np.int64)
    gid_sorted, G = _group(rt_all[o], file_of[o], rank, s_off, s_mz, s_ab ** 0.6 * (s_mz / 100.0), rt_tol, MATCH_MIN,
                           len(stems), QUANT_TOL, QUANT_PPM)
    gid = np.empty(len(o), np.int64)
    gid[o] = gid_sorted
    times["matching_s"] = round(time.perf_counter() - tic, 3)

    # 3. compound rows: reference = the member with the largest summed abundance
    need = max(1, min(int(min_presence), len(stems)))
    n_in = np.bincount(gid, minlength=G)
    ref = np.full(G, -1, np.int64)
    for i in np.argsort(-tic_all, kind="stable"):
        if ref[gid[i]] < 0:
            ref[gid[i]] = i
    keep = np.flatnonzero(n_in >= need)
    keep = keep[np.argsort(rt_all[ref[keep]], kind="stable")]
    new = np.full(G, -1, np.int64)
    new[keep] = np.arange(len(keep))
    cid = new[gid]
    rows, sp_mz, sp_ab, counts = [], [], [], []
    quant = np.full((len(keep), len(stems)), np.nan)
    members = [np.flatnonzero(cid == k) for k in range(len(keep))]
    for k, g in enumerate(keep):
        r = ref[g]
        row = comps[stems[file_of[r]]].iloc[local[r]]
        m, a = mz_all[off_all[r]:off_all[r + 1]], ab_all[off_all[r]:off_all[r + 1]]
        sp_mz.append(m); sp_ab.append(a); counts.append(len(m))
        qmz = float(row["quant_mz"])
        tol = max(QUANT_TOL, QUANT_PPM * 1e-6 * qmz)
        for i in members[k]:
            mm, aa = mz_all[off_all[i]:off_all[i + 1]], ab_all[off_all[i]:off_all[i + 1]]
            lo, hi = np.searchsorted(mm, qmz - tol), np.searchsorted(mm, qmz + tol, side="right")
            if hi > lo:
                quant[k, file_of[i]] = aa[lo:hi].max()
        rows.append(dict(compound_id=k, rt=float(np.median(rt_all[members[k]])), n_files=int(n_in[g]),
                         quant_mz=qmz, base_mz=float(row["base_mz"]), n_ions=int(row["n_ions"]),
                         n_shared_ions=int(row["n_shared_ions"]), height=float(row["quant_height"]),
                         reference_file=stems[file_of[r]], reference_component=int(row["component_id"])))
    compounds = pd.DataFrame(rows, columns=["compound_id", "rt", "n_files", "quant_mz", "base_mz", "n_ions",
                                            "n_shared_ions", "height", "reference_file", "reference_component"])
    c_off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    c_mz = np.concatenate(sp_mz) if sp_mz else np.zeros(0)
    c_ab = np.concatenate(sp_ab) if sp_ab else np.zeros(0)
    pos = 0
    for s in stems:
        comps[s]["compound_id"] = cid[pos:pos + len(comps[s])]
        pos += len(comps[s])
    log(f"{len(compounds)} compounds in at least {need} of {len(stems)} files (RT tolerance {60 * rt_tol:.2f} s, "
        f"{G - len(keep)} below that)")

    # 4. retention index
    notes = []
    kind = ladder
    table = None
    if ladder_table is not None:
        table = RI.read_ladder(ladder_table, kind)
    lad = pd.DataFrame()
    rimap = None
    compounds["ri"] = np.nan
    compounds["ri_extrapolated"] = False
    for s in stems:
        comps[s]["ri"] = np.nan
    if kind or table is not None:
        lad, notes = _ladders(stems, comps, spectra, kind, RI.parse_carbons(carbons) if isinstance(carbons, str) else carbons,
                              table)
        rimap = _study_map(lad)
        if rimap is None:
            log("retention index: no ladder found, no index assigned")
        else:
            n_files_l = lad["file"].nunique() if table is None else 0
            log(f"retention index: {len(rimap.rt)} ladder compounds" +
                (f", found in {n_files_l} of {len(stems)} files" if table is None else " from the table"))
            r, ex = rimap.ri_at(compounds["rt"].values)
            compounds["ri"], compounds["ri_extrapolated"] = np.round(r, 2), ex
            for s in stems:
                comps[s]["ri"] = np.round(rimap.ri_at(comps[s]["rt_corr"].values)[0], 2)
        lad.to_csv(gc_dir / "ladder.tsv", sep="\t", index=False, float_format="%.6g")
        for n in notes[:20]:
            log(n)

    # 5. library search
    hits = pd.DataFrame(columns=["compound_id"] + L.HIT_COLUMNS[1:])
    for c in ("name", "formula", "inchikey"):
        compounds[c] = ""
    for c in ("match", "reverse", "library_ri", "delta_ri"):
        compounds[c] = np.nan
    libs = [Path(p) for p in ([library] if isinstance(library, (str, Path)) else (library or []))]
    (gc_dir / "hit_spectra.npz").unlink(missing_ok=True)          # an earlier search's
    if libs and len(compounds):
        tic = time.perf_counter()
        SP = search or L.SearchParams()
        if rimap is not None and not np.isfinite(SP.ri_tol):
            SP.ri_tol = float(ri_tol) if ri_tol is not None else RI.RI_TOL.get(kind or "", float("nan"))
        elif ri_tol is not None:
            SP.ri_tol = float(ri_tol)
        mz_range = (min(i["mz_lo"] for i in infos), max(i["mz_hi"] for i in infos))
        parts, loaded = [], []
        for p in libs:
            lib = L.load(p, gc_dir / ".library_cache", ri_type)
            h = L.search(c_off, c_mz, c_ab, lib, compounds["ri"].values, SP, mz_range, p.name)
            h["_lib"] = len(loaded)
            loaded.append(lib)
            parts.append(h)
            log(f"library {p.name}: {len(lib)} spectra ({'unit' if lib.nominal else 'accurate'} mass, "
                f"{int(np.isfinite(lib.ri).sum())} with a retention index), {h['query'].nunique()} compounds with a hit")
        hits = pd.concat(parts, ignore_index=True).sort_values(["query", "match"], ascending=[True, False], kind="stable")
        hits["rank"] = hits.groupby("query").cumcount() + 1
        hits = hits.rename(columns={"query": "compound_id"})
        _hit_spectra(gc_dir / "hit_spectra.npz", hits[hits["rank"] <= HIT_SPECTRA], loaded)
        hits = hits.drop(columns="_lib")
        top = hits[(hits["rank"] == 1) & (hits["match"] >= min_similarity)].set_index("compound_id")
        for c in ("name", "formula", "inchikey", "match", "reverse", "library_ri", "delta_ri"):
            compounds.loc[top.index, c] = top[c].values
        times["search_s"] = round(time.perf_counter() - tic, 3)
        log(f"{len(top)} of {len(compounds)} compounds annotated at match >= {min_similarity:g} ({times['search_s']} s)")

    # 6. write
    for s in stems:
        _write(comps[s], comp_dir / f"{s}.tsv")
    _write(compounds, gc_dir / "compounds.tsv")
    _write(pd.concat([compounds[["compound_id", "rt", "ri", "quant_mz", "name"]], pd.DataFrame(quant, columns=stems)],
                     axis=1), gc_dir / "compound_table.tsv")
    hits.to_csv(gc_dir / "identifications.tsv", sep="\t", index=False, float_format="%.6g")
    L.write_msp(gc_dir / "spectra.msp",
                [f"compound {k}" + (f" | {n}" if n else "") for k, n in zip(compounds["compound_id"], compounds["name"])],
                c_off, c_mz, c_ab, compounds["ri"].values, compounds["rt"].values)
    summary = dict(peak3d_version=__version__, n_files=len(stems), n_components=int(len(gid)), n_compounds=int(len(compounds)),
                   rt_tol_min=rt_tol, min_presence=need, ladder=kind, ladder_table=str(ladder_table) if ladder_table else None,
                   n_ladder=int(len(rimap.rt)) if rimap is not None else 0, libraries=[str(p) for p in libs],
                   min_similarity=min_similarity, ri_tol=SP.ri_tol if libs and np.isfinite(SP.ri_tol) else None,
                   deconvolution=asdict(D), files=infos, ladder_notes=notes, stage_times_s=times)
    (gc_dir / "params.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary
