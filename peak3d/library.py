"""Spectral libraries (.msp, .pkl) and identity search of the features' MS2 spectra.

read_library reads .msp files (read_msp: the common dialects of NIST, MoNA, MassBank, MS-DIAL and GNPS exports, with
a compiled scanner, msp.py, that reads in blocks and parses on all cores) and MassCube / ms_entropy .pkl libraries
(flashdb.py) into one CSR peak list; identify scores every feature's representative MS2 spectrum
(ms2.read_representatives) against the library records whose precursor m/z lies within precursor_ppm of the
feature's m/z and whose ion mode agrees, with one of the scores in similarity.py, and writes the hits next to the
tables. Importing this module stays light (numpy, pandas): the compiled readers and scores load on first use."""
from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = {"entropy": "Unweighted entropy similarity", "weighted_entropy": "Weighted entropy similarity",
           "dot": "Unweighted dot product", "weighted_dot": "Weighted dot product",
           "reverse_entropy": "Reverse entropy score"}
ID_COLUMNS = ["group_id", "spectrum", "rank", "score", "n_matched", "name", "precursor_type", "formula", "inchikey",
              "smiles", "library_precursor_mz", "ppm_error", "library_index"]


class Strings(Sequence):
    """one text field of every record, decoded on access from the reader's shared byte buffer (a 2-million-record
    library holds one bytes object per field set instead of millions of str)"""
    __slots__ = ("_text", "_off", "_k", "_stride", "_n")

    def __init__(self, text: bytes, off: np.ndarray, k: int, stride: int, n: int):
        self._text, self._off, self._k, self._stride, self._n = text, off, k, stride, n

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(self._n))]
        i = int(i)
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError("record index out of range")
        j = i * self._stride + self._k
        # the scanner strips ASCII blanks; strip() also takes Unicode ones (a trailing U+3000 in some MoNA names)
        return self._text[self._off[j]:self._off[j + 1]].decode("utf-8", "replace").strip()

    def __eq__(self, other):
        if isinstance(other, (list, tuple, Sequence)) and not isinstance(other, str):
            return list(self) == list(other)
        return NotImplemented

    __hash__ = None

    def __repr__(self):
        return f"Strings({self[:3]}{'...' if self._n > 3 else ''}, n={self._n})"


class Modes(Sequence):
    """ion mode of every record as "pos" / "neg" / "" over the int8 codes (1, 2, 0)"""
    NAMES = ("", "pos", "neg")

    def __init__(self, codes: np.ndarray):
        self.codes = codes

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self.NAMES[c] for c in self.codes[i].tolist()]
        return self.NAMES[int(self.codes[i])]

    def __eq__(self, other):
        if isinstance(other, (list, tuple, Sequence)) and not isinstance(other, str):
            return list(self) == list(other)
        return NotImplemented

    __hash__ = None


@dataclass
class Library:
    path: str
    names: Sequence
    precursor_mz: np.ndarray      # f8, nan when the record gives none
    precursor_type: Sequence
    mode: np.ndarray              # int8: 0 unknown, 1 positive, 2 negative
    formula: Sequence
    inchikey: Sequence
    smiles: Sequence
    comments: Sequence            # empty strings when read without comments (load)
    off: np.ndarray               # int64, len n + 1
    mz: np.ndarray                # f8, ascending within a record
    inten: np.ndarray             # f4
    n_skipped: int = 0            # records without peaks, MS1 records
    _clean: dict = field(default_factory=dict, repr=False)   # fragment tol -> cleaned CSR (similarity.clean_batch)
    _index: tuple = field(default=None, repr=False)

    def __len__(self):
        return len(self.precursor_mz)

    @property
    def ion_mode(self) -> Modes:
        return Modes(self.mode)

    def precursor_index(self):
        """(record order by precursor m/z, the sorted precursor m/z) over records that have one; built once"""
        if self._index is None:
            valid = np.flatnonzero(np.isfinite(self.precursor_mz))
            order = valid[np.argsort(self.precursor_mz[valid], kind="stable")]
            self._index = (order, self.precursor_mz[order])
        return self._index


# ------------------------------------------------------------------ parsing ----
LIBRARY_SUFFIXES = (".msp", ".pkl")


def _library(path, parsed) -> Library:
    from . import msp
    text, toff, prec, mode, off, mz, inten, skipped = parsed
    text = text.tobytes()
    nt, n = len(msp.TEXT), len(prec)
    names, ptype, formula, inchikey, smiles, comm = (Strings(text, toff, k, nt, n) for k in range(nt))
    return Library(str(path), names, prec, ptype, mode, formula, inchikey, smiles, comm, off, mz, inten, int(skipped))


def read_library(path, comments: bool = True) -> Library:
    """a spectral library: .msp (read_msp), or .pkl, a MassCube / ms_entropy FlashEntropySearch pickle read
    without running code from the file (flashdb.read)"""
    path = Path(path)
    if path.suffix.lower() in (".pkl", ".pickle"):
        from . import flashdb
        return _library(path, flashdb.read(path, comments))
    return read_msp(path, comments)


def read_msp(path, comments: bool = True) -> Library:
    """every MS2 record of an .msp file (rules in msp.py): keys are case- and punctuation-insensitive (PRECURSORMZ,
    PrecursorMZ, precursor_mz); a record ends at a blank line after its peaks or at the next NAME. Records without
    peaks and Spectrum_type MS1 records are skipped (n_skipped). comments=False leaves the Comment text out (a
    search never needs it, and in some libraries it is most of the file)."""
    from . import msp
    return _library(path, msp.read(path, comments))


_CACHE: OrderedDict = OrderedDict()


def load(path) -> Library:
    """read_library without comments, kept in memory per (path, size, mtime): a GUI re-search with another score reuses
    the parse; only the last library is held (a large one takes gigabytes)"""
    p = Path(path).expanduser().resolve()
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns)
    if key in _CACHE:
        return _CACHE[key]
    _CACHE.clear()                     # free the previous library before reading the next: never two in memory
    lib = read_library(p, comments=False)
    _CACHE[key] = lib
    return lib


# ------------------------------------------------------------------ writing ----
def _fmt(v) -> str:
    return "" if v is None or (isinstance(v, float) and not math.isfinite(v)) else str(v)


def write_msp_record(fh, fields: list[tuple[str, object]], mz, inten) -> None:
    """one .msp record: KEY: value lines (empty values left out), Num Peaks, "mz<TAB>intensity" lines, blank line"""
    for k, v in fields:
        s = _fmt(v)
        if s != "":
            fh.write(f"{k}: {s}\n")
    fh.write(f"Num Peaks: {len(mz)}\n")
    fh.write("".join(f"{m:.5f}\t{i:.6g}\n" for m, i in zip(mz, inten)))
    fh.write("\n")


# ------------------------------------------------------------------ search ----
_MODE_CODE = {"": 0, "pos": 1, "neg": 2}


def candidate_pairs(q_mz, q_mode, lib_prec, lib_mode, ppm, da=None, index=None):
    """(query index, library index) of every library record within ppm (or da, in Da, when given) of each query
    m/z with a compatible ion mode (either unknown, or equal); through a sorted precursor index (Library.
    precursor_index, else built here), so the cost follows the pair count"""
    if index is None:
        valid = np.flatnonzero(np.isfinite(lib_prec))
        order = valid[np.argsort(lib_prec[valid], kind="stable")]
        index = (order, lib_prec[order])
    order, sp = index
    tol = np.full(len(q_mz), float(da)) if da is not None else np.asarray(q_mz, float) * ppm * 1e-6
    lo = np.searchsorted(sp, q_mz - tol, side="left")
    hi = np.searchsorted(sp, q_mz + tol, side="right")
    n = hi - lo
    pq = np.repeat(np.arange(len(q_mz)), n)
    start = np.repeat(lo - np.concatenate([[0], np.cumsum(n)[:-1]]), n)
    pl = order[start + np.arange(int(n.sum()))] if n.sum() else np.zeros(0, np.int64)
    qm, lm = np.asarray(q_mode)[pq], np.asarray(lib_mode)[pl]
    ok = (qm == 0) | (lm == 0) | (qm == lm)
    return pq[ok].astype(np.int64), pl[ok].astype(np.int64)


CLEAN_CACHE = 400_000


def _cleaned_library(lib: Library, tol: float, idx: np.ndarray):
    """cleaned CSR of the library records idx (cached for the last tolerance used, up to CLEAN_CACHE records: a
    re-search only cleans records it has not seen); returns (off, mz, inten) over idx in that order"""
    from .similarity import clean_batch
    key = round(tol, 9)
    if key not in lib._clean:
        lib._clean.clear()
    store = lib._clean.setdefault(key, {})
    todo = np.array([i for i in idx.tolist() if i not in store], np.int64)
    if len(store) + len(todo) > CLEAN_CACHE:          # bounded: start over rather than grow with every search
        store.clear()
        todo = np.asarray(idx, np.int64)
    if len(todo):
        lens = lib.off[todo + 1] - lib.off[todo]
        off = np.zeros(len(todo) + 1, np.int64)
        np.cumsum(lens, out=off[1:])
        take = np.concatenate([np.arange(lib.off[i], lib.off[i + 1]) for i in todo])
        c_off, c_mz, c_it = clean_batch(off, lib.mz[take], lib.inten[take].astype(np.float64),
                                        lib.precursor_mz[todo], tol)
        for k, i in enumerate(todo.tolist()):
            store[i] = (c_mz[c_off[k]:c_off[k + 1]].copy(), c_it[c_off[k]:c_off[k + 1]].copy())
    parts = [store[i] for i in idx.tolist()]
    off = np.zeros(len(parts) + 1, np.int64)
    np.cumsum([len(p[0]) for p in parts], out=off[1:])
    mz = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0)
    it = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0)
    return off, mz, it


LIST_MIN = 0.1        # every hit scoring above this is listed; min_score only decides the default annotation


def identify(output, library_path, method: str = "weighted_entropy", precursor_ppm: float = 10.0,
             fragment_tol: float = 0.02, min_score: float = 0.7, top_k: int | None = None,
             precursor_da: float | None = None) -> pd.DataFrame:
    """search both MS2 copies of every feature (ms2.SPECTRA: the consensus of all its spectra and the single
    highest-TIC spectrum) in the library and list every hit scoring above LIST_MIN (or at least min_score, when that is
    lower), top_k per feature and copy when given. min_score is the annotation threshold: a feature's best consensus
    hit at or above it is its default annotation (annotations.py). precursor_da (Da) replaces precursor_ppm as the
    precursor window when given. Writes identifications.tsv (one row per hit, spectrum = consensus or single),
    identifications.msp (the library spectrum of every hit), identifications.json (settings and counts) and
    feature_annotations.tsv, replacing earlier ones."""
    from . import ms2
    from .annotations import default_copy, write_table
    from .pipeline import log
    from .similarity import METHOD_CODE, clean_batch, score_pairs
    if method not in METHODS:
        raise ValueError(f"unknown similarity {method!r}; one of {', '.join(METHODS)}")
    output = Path(output)
    tic = time.perf_counter()
    lib = load(library_path)
    t_load = time.perf_counter() - tic
    parts, nq = [], 0
    for which in ms2.SPECTRA:
        q = ms2.read_representatives(output, which)
        if q is None:
            continue
        if which == "single":
            nq = len(q["group_id"])
        if not len(lib):
            continue
        q_mode = np.array([_MODE_CODE.get(m, 0) for m in q["ion_mode"]], np.int8)
        pq, pl = candidate_pairs(q["prec_mz"], q_mode, lib.precursor_mz, lib.mode, precursor_ppm, precursor_da,
                                 lib.precursor_index())
        if not len(pq):
            continue
        qo, qm, qi = clean_batch(q["off"], q["mz"], q["inten"].astype(np.float64), q["prec_mz"], fragment_tol)
        uniq, inv = np.unique(pl, return_inverse=True)
        lo, lm, li = _cleaned_library(lib, fragment_tol, uniq)
        sc, nm = score_pairs(qo, qm, qi, lo, lm, li, pq, inv.astype(np.int64), METHOD_CODE[method],
                             float(fragment_tol))
        hits = pd.DataFrame({"q": pq, "lib": pl, "score": sc, "n_matched": nm})
        hits = hits[((hits["score"] > LIST_MIN) | (hits["score"] >= min_score)) & (hits["n_matched"] > 0)]
        hits = hits.sort_values(["q", "score", "n_matched"], ascending=[True, False, False], kind="stable")
        if top_k:
            hits = hits.groupby("q", sort=False).head(top_k)
        hits = hits.copy()
        hits["rank"] = hits.groupby("q", sort=False).cumcount() + 1
        L = hits["lib"].to_numpy()
        lp = lib.precursor_mz[L]
        parts.append(pd.DataFrame({
            "group_id": q["group_id"][hits["q"].to_numpy()], "spectrum": which, "rank": hits["rank"].to_numpy(),
            "score": hits["score"].round(4).to_numpy(), "n_matched": hits["n_matched"].to_numpy(),
            "name": [lib.names[i] for i in L], "precursor_type": [lib.precursor_type[i] for i in L],
            "formula": [lib.formula[i] for i in L], "inchikey": [lib.inchikey[i] for i in L],
            "smiles": [lib.smiles[i] for i in L], "library_precursor_mz": lp,
            "ppm_error": np.round((q["prec_mz"][hits["q"].to_numpy()] - lp) / lp * 1e6, 2), "library_index": L}))
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=ID_COLUMNS)
    rows = rows[ID_COLUMNS]
    rows.to_csv(output / "identifications.tsv", sep="\t", index=False, float_format="%.6g")
    with open(output / "identifications.msp", "w", encoding="utf-8") as fh:
        for r in rows.itertuples(index=False):
            i = int(r.library_index)
            write_msp_record(fh, [("NAME", lib.names[i]), ("PRECURSORMZ", lib.precursor_mz[i]),
                                  ("PRECURSORTYPE", lib.precursor_type[i]), ("FORMULA", lib.formula[i]),
                                  ("INCHIKEY", lib.inchikey[i]), ("SMILES", lib.smiles[i]),
                                  ("IONMODE", {1: "Positive", 2: "Negative"}.get(int(lib.mode[i]), "")),
                                  ("Comment", f"group_id={int(r.group_id)} spectrum={r.spectrum} rank={int(r.rank)} "
                                              f"score={r.score:.4f}")],
                             lib.mz[lib.off[i]:lib.off[i + 1]], lib.inten[lib.off[i]:lib.off[i + 1]])
    best = rows[rows["rank"] == 1]
    copy = best[best["spectrum"] == default_copy(output)].set_index("group_id")["score"]
    n_id = int((copy >= min_score).sum())
    secs = time.perf_counter() - tic
    info = dict(method=method, method_label=METHODS[method], library=str(Path(library_path).expanduser().resolve()),
                library_n=len(lib), precursor_ppm=float(precursor_ppm),
                precursor_da=None if precursor_da is None else float(precursor_da), fragment_tol=float(fragment_tol),
                min_score=float(min_score), list_min=LIST_MIN, top_k=top_k, n_queries=int(nq), n_identified=n_id,
                n_hits=int(len(rows)), seconds=round(secs, 3), created=time.strftime("%Y-%m-%d %H:%M:%S"))
    (output / "identifications.json").write_text(json.dumps(info, indent=1))
    write_table(output)
    log(f"identify: {n_id} of {nq} features with MS2 annotated from {Path(library_path).name} ({len(lib)} spectra, "
        f"{METHODS[method].lower()} >= {min_score:g} on the consensus spectrum; {len(rows)} hits above {LIST_MIN:g} "
        f"listed; {secs:.1f} s, library read {t_load:.1f} s)")
    return rows
