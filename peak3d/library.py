"""Spectral libraries (.msp, .pkl) and identity search of the features' MS2 spectra.

read_library reads .msp files (read_msp: the common dialects of NIST, MoNA, MassBank, MS-DIAL and GNPS exports, with
a compiled scanner, msp.py, that reads in blocks and parses on all cores) and MassCube / ms_entropy .pkl libraries
(flashdb.py) into one CSR peak list; identify scores every feature's representative MS2 spectrum
(ms2.read_representatives) against the library records whose precursor m/z lies within precursor_ppm of the
feature's m/z and whose ion mode agrees, with one of the scores in similarity.py, and writes the hits next to the
tables. Importing this module stays light (numpy, pandas): the compiled readers and scores load on first use."""
from __future__ import annotations

import csv
import io
import json
import math
import pickle
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
              "smiles", "library_precursor_mz", "ppm_error", "library_index", "library", "denoise_ion",
              "denoise_key"]
DENOISE_FILE = "identifications_denoise.npz"    # search: the keep mask of every (spectrum, ion formula) a listed hit used
DENOISE_MODES = {"search": "T1 per candidate formula (denoising search)", "t0": "T0 top-5", "off": None}
# fragment peaks match within the larger of FRAGMENT_TOL (Da) and FRAGMENT_PPM x m/z: the ion tolerance of the denoising
# (specdenoise.Params tol_abs, tol_ppm), so the raw and the denoised score of a hit pair the same peaks. A wider window
# (0.02 Da before 1.2.6) lets wrong ions pair up, which lifts raw scores above the denoised ones
FRAGMENT_TOL = 0.003
FRAGMENT_PPM = 10.0


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
    _ions: object = field(default=None, repr=False)          # specdenoise.search.IonTable, filled by the searches

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


class LibraryError(ValueError):
    """a file that cannot be read as a spectral library; path is the file"""

    def __init__(self, path, cause):
        self.path = Path(path)
        super().__init__(f"could not read the library {self.path.name}: {cause}")


def library_paths(library) -> list[Path]:
    """one spectral library or several (a path, or a list of paths) as resolved paths in the order given, each once"""
    items = [library] if isinstance(library, (str, Path)) else list(library)
    out = []
    for x in items:
        p = Path(x).expanduser().resolve()
        if p not in out:
            out.append(p)
    if not out:
        raise ValueError("no spectral library given")
    return out


def library_names(paths) -> list[str]:
    """the name of each library in a search (the hits' library column): its file name, with its folder where two
    libraries share a file name, the whole path where that does not tell them apart either"""
    paths = [Path(p) for p in paths]
    names = [p.name for p in paths]
    names = [f"{p.parent.name}/{p.name}" if names.count(p.name) > 1 else p.name for p in paths]
    return [str(p) if names.count(n) > 1 else n for p, n in zip(paths, names)]


def load_all(paths) -> list[Library]:
    """read_library without comments for each path, kept in memory per (path, size, mtime): a GUI re-search with
    another score, or with one more library, reuses the parses made; only the libraries of the last call are held
    (a large one takes gigabytes), the others are let go before anything new is read"""
    keys = []
    for path in paths:
        p = Path(path).expanduser().resolve()
        st = p.stat()
        keys.append((p, (str(p), st.st_size, st.st_mtime_ns)))
    want = {k for _, k in keys}
    for k in [k for k in _CACHE if k not in want]:
        del _CACHE[k]
    for p, k in keys:
        if k not in _CACHE:
            try:
                _CACHE[k] = read_library(p, comments=False)
            except (pickle.UnpicklingError, ValueError, EOFError) as e:     # e.g. a .pkl that is not a library
                raise LibraryError(p, e) from e
    return [_CACHE[k] for _, k in keys]


def load(path) -> Library:
    """one library through load_all: only this one is held afterwards"""
    return load_all([path])[0]


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
    n = min(len(mz), len(inten))
    if n:      # one %-format over the record ("%.5f" and "%.6g" format a float exactly as f"{m:.5f}", f"{i:.6g}")
        fh.write(("%.5f\t%.6g\n" * n) % tuple(np.column_stack((np.asarray(mz)[:n], np.asarray(inten)[:n])).ravel()
                                              .tolist()))
    fh.write("\n")


MSP_CHUNK = 20_000           # records per block of write_msp_records


def write_msp_records(fh, fields: list, off, mz, inten) -> None:
    """write_msp_record for many records, the same text: record k has the fields fields[k] and the peaks
    off[k]:off[k + 1] of mz / inten; a block of records is formatted and written at once"""
    off = np.asarray(off, np.int64)
    for a in range(0, len(fields), MSP_CHUNK):
        b = min(a + MSP_CHUNK, len(fields))
        p0, p1 = int(off[a]), int(off[b])
        both = np.empty(2 * (p1 - p0))
        both[0::2] = mz[p0:p1]
        both[1::2] = inten[p0:p1]
        vals = both.tolist()
        parts = []
        for k in range(a, b):
            for key, v in fields[k]:
                t = _fmt(v)
                if t != "":
                    parts.append(f"{key}: {t}\n")
            n = int(off[k + 1] - off[k])
            parts.append(f"Num Peaks: {n}\n")
            if n:
                s0 = 2 * (int(off[k]) - p0)
                parts.append(("%.5f\t%.6g\n" * n) % tuple(vals[s0:s0 + 2 * n]))
            parts.append("\n")
        fh.write("".join(parts))


def format_floats(values, fmt: str = "%.6g", na: str = "") -> np.ndarray:
    """fmt % v for every value of a float array (na where NaN) as an object array, each distinct value formatted once
    (keyed by its bits, so -0.0 and 0.0 stay apart): what DataFrame.to_csv(float_format=fmt, na_rep=na) writes"""
    v = np.ascontiguousarray(values, np.float64)
    codes, uniq = pd.factorize(v.view(np.int64), sort=False, use_na_sentinel=False)
    text = np.array([na if x != x else fmt % x for x in uniq.view(np.float64).tolist()], dtype=object)
    return text[codes]


def _csv_cells(values, term: str) -> np.ndarray:
    """the text DataFrame.to_csv(sep="\\t") writes for each cell of an object column: NaN / None empty, a value with
    a tab, quote or line break quoted by the csv module itself (QUOTE_MINIMAL); each distinct value once"""
    codes, uniq = pd.factorize(np.asarray(values, dtype=object), sort=False)
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t", lineterminator=term, quoting=csv.QUOTE_MINIMAL, quotechar='"',
                   doublequote=True, escapechar=None)
    out = []
    for v in uniq.tolist():
        if isinstance(v, str) and not ("\t" in v or '"' in v or "\r" in v or "\n" in v):
            out.append(v)
        else:                         # [v, ""]: never the one-empty-field row the csv module writes as ""
            buf.seek(0)
            buf.truncate()
            w.writerow([v, ""])
            out.append(buf.getvalue()[:-(1 + len(term))])
    return np.array(out + [""], dtype=object)[codes]          # code -1 (NaN, None): the "" appended last


TSV_ROWS = 200_000           # lines per block of write_tsv


def write_tsv(df: pd.DataFrame, path, float_format: str = "%.6g") -> None:
    """df.to_csv(path, sep="\\t", index=False, float_format=float_format), byte for byte, written faster: every column
    turned into its cell text first (floats and text per distinct value: format_floats, _csv_cells) and the lines
    joined; a frame with one column, repeated names or other column types (nullable numbers, dates, ...) goes
    through to_csv itself"""
    plain = (df.shape[1] >= 2 and df.columns.is_unique and not isinstance(df.columns, pd.MultiIndex)
             and all((isinstance(t, np.dtype) and t.kind in "fiubO") or isinstance(t, pd.StringDtype) for t in df.dtypes))
    if not plain:
        df.to_csv(path, sep="\t", index=False, float_format=float_format)
        return
    head = df.iloc[:0].to_csv(sep="\t", index=False)          # the header line, with pandas' line terminator
    term = "\r\n" if head.endswith("\r\n") else "\n"
    cols = []
    for c in df.columns:
        v = df[c].to_numpy()                                    # text (object or pandas str): an object array
        if v.dtype.kind == "f":
            cols.append(format_floats(v, float_format))
        elif v.dtype.kind == "O":
            cols.append(_csv_cells(v, term))
        else:                                                   # int, unsigned, bool: str(), as the csv module
            cols.append(np.array(list(map(str, v.tolist())), dtype=object))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(head)
        for a in range(0, len(df), TSV_ROWS):
            fh.write(term.join(map("\t".join, zip(*(c[a:a + TSV_ROWS] for c in cols)))) + term)


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


CLEAN_CACHE = 100_000_000      # cleaned peaks kept per library (16 bytes each); past it the cache starts over


def _csr_take(start, lens):
    """(offsets, flat peak positions) of the CSR records that start at start with lens peaks, in that order"""
    lens = np.asarray(lens, np.int64)
    off = np.zeros(len(lens) + 1, np.int64)
    np.cumsum(lens, out=off[1:])
    return off, np.repeat(np.asarray(start, np.int64) - off[:-1], lens) + np.arange(off[-1], dtype=np.int64)


def _cleaned_library(lib: Library, tol: float, idx: np.ndarray):
    """cleaned CSR of the library records idx (cached for the last tolerance used as one flat peak list, up to
    CLEAN_CACHE peaks: a re-search only cleans records it has not seen); returns (off, mz, inten) over idx in that
    order"""
    from .similarity import clean_batch
    key = round(tol, 9)
    idx = np.asarray(idx, np.int64)
    st = lib._clean.get(key)
    if st is None:
        lib._clean.clear()
    todo = np.unique(idx) if st is None else np.unique(idx[st["start"][idx] < 0])
    if len(todo):
        lens = lib.off[todo + 1] - lib.off[todo]
        off, take = _csr_take(lib.off[todo], lens)
        c_off, c_mz, c_it = clean_batch(off, lib.mz[take], lib.inten[take].astype(np.float64),
                                        lib.precursor_mz[todo], tol)
        if st is not None and len(st["mz"]) + len(c_mz) > CLEAN_CACHE:   # bounded: start over rather than grow
            st = None
            todo = np.unique(idx)
            lens = lib.off[todo + 1] - lib.off[todo]
            off, take = _csr_take(lib.off[todo], lens)
            c_off, c_mz, c_it = clean_batch(off, lib.mz[take], lib.inten[take].astype(np.float64),
                                            lib.precursor_mz[todo], tol)
        if st is None:
            st = {"start": np.full(len(lib), -1, np.int64), "len": np.zeros(len(lib), np.int64),
                  "mz": np.zeros(0), "it": np.zeros(0)}
        st["start"][todo] = len(st["mz"]) + c_off[:-1]
        st["len"][todo] = np.diff(c_off)
        st["mz"] = np.concatenate([st["mz"], c_mz])
        st["it"] = np.concatenate([st["it"], c_it])
        lib._clean[key] = st
    if st is None:                                      # nothing asked, nothing held
        return np.zeros(1, np.int64), np.zeros(0), np.zeros(0)
    off, take = _csr_take(st["start"][idx], st["len"][idx])
    return off, st["mz"][take], st["it"][take]


def _record_text(libs, lib_n, rec, field: str) -> np.ndarray:
    """one text field (Library attribute: names, formula, ...) of the records (lib_n, rec) as an object array,
    each distinct record decoded once"""
    key = np.asarray(lib_n, np.int64) * (1 << 40) + np.asarray(rec, np.int64)
    codes, uniq = pd.factorize(key, sort=False)
    vals = np.array([getattr(libs[k >> 40], field)[k & ((1 << 40) - 1)] for k in uniq.tolist()] or [""], dtype=object)
    return vals[codes] if len(codes) else np.zeros(0, dtype=object)


LIST_MIN = 0.1        # every hit scoring above this is listed; min_score only decides the default annotation


def _in_order(gid, other):
    """the spectra of other (a read_representatives dict) in the order of the group ids gid, as CSR (off, mz,
    intensity); empty where other has no record of that group"""
    at = {int(g): i for i, g in enumerate(other["group_id"])}
    src = np.array([at.get(int(g), -1) for g in gid], np.int64)
    o = np.asarray(other["off"], np.int64)
    lens = np.where(src >= 0, (o[1:] - o[:-1])[np.maximum(src, 0)], 0)
    off = np.zeros(len(gid) + 1, np.int64)
    np.cumsum(lens, out=off[1:])
    sel = np.repeat(o[np.maximum(src, 0)] - off[:-1], lens) + np.arange(off[-1])
    return off, np.asarray(other["mz"])[sel], np.asarray(other["inten"])[sel]


def _ion_ids(lib: Library, records: np.ndarray) -> np.ndarray:
    """the precursor ion formula id of each library record (specdenoise.search.IonTable kept with the library, -1
    where a record has none; a record without a formula is read from its SMILES)"""
    from specdenoise.search import IonTable
    if lib._ions is None:
        lib._ions = IonTable(len(lib))
    return lib._ions.ids(records, lib.formula, lib.precursor_type, lib.mode, lib.precursor_mz, lib.smiles)


def _calibrated(off, mz, shift_ppm) -> np.ndarray:
    """the m/z of a CSR batch of query spectra on the table's m/z scale: each spectrum's file offset (shift_ppm, its
    mz_offset_ppm) taken out, as the denoising and the MS2 mapping do"""
    off = np.asarray(off, np.int64)
    return np.asarray(mz, np.float64) * (1.0 - np.repeat(np.asarray(shift_ppm, np.float64), np.diff(off)) * 1e-6)


def _search_denoised(q, mz, lib: Library, pq, pl, inv, lo, lm, li, raw_sc, raw_nm, method, fragment_tol, fragment_ppm,
                     stats):
    """scores of the denoising search: each pair's query (mz: its calibrated m/z) denoised by T1 against its library
    record's precursor ion formula, once per (query, ion formula), then scored against that record; pairs whose
    record has no usable ion formula keep their raw score. Returns (scores, matched, ion formula id per pair (-1 =
    none), the keys: dict of pair_key, key_q, key_ion, key_off, keep)."""
    from specdenoise import Params
    from specdenoise.search import denoise_keys, masked_spectra, unique_keys
    from .similarity import METHOD_CODE, clean_batch, score_pairs
    t = time.perf_counter()
    u = np.unique(pl)
    pair_ion = _ion_ids(lib, u)[np.searchsorted(u, pl)].astype(np.int64)
    tab = lib._ions
    key_q, key_ion, pair_key = unique_keys(pq, pair_ion, q["prec_mz"], tab.mz, tab.Z)
    t1 = time.perf_counter()
    off = np.asarray(q["off"], np.int64)
    key_off, keep = denoise_keys(off, mz, q["inten"], q["prec_mz"], key_q, tab.P[key_ion], tab.Z[key_ion], Params())
    t2 = time.perf_counter()
    d_off, d_mz, d_it = masked_spectra(off, mz, np.asarray(q["inten"], np.float64), key_q, key_off, keep)
    ko, km, ki = clean_batch(d_off, d_mz, d_it, np.asarray(q["prec_mz"], np.float64)[key_q], fragment_tol)
    sc, nm = raw_sc.copy(), raw_nm.copy()
    has = np.flatnonzero(pair_key >= 0)
    if len(has):
        sc[has], nm[has] = score_pairs(ko, km, ki, lo, lm, li, pair_key[has], inv[has].astype(np.int64),
                                       METHOD_CODE[method], float(fragment_tol), float(fragment_ppm))
    t3 = time.perf_counter()
    for k, v in (("pairs", len(pl)), ("pairs_denoised", len(has)), ("keys", len(key_q)),
                 ("peaks_judged", int(key_off[-1])), ("seconds_ions", t1 - t), ("seconds_t1", t2 - t1),
                 ("seconds_score", t3 - t2)):
        stats[k] = stats.get(k, 0) + v
    stats["ion_formulas"] = len(tab)
    keys = dict(pair_key=pair_key, key_q=key_q, key_ion=key_ion, key_off=key_off, keep=keep)
    return sc, nm, np.where(pair_key >= 0, pair_ion, -1), keys


def identify(output, library_path, method: str = "weighted_entropy", precursor_ppm: float = 10.0,
             fragment_tol: float = FRAGMENT_TOL, min_score: float = 0.7, top_k: int | None = None,
             precursor_da: float | None = None, workers: int = 1, denoise: str = "search",
             fragment_ppm: float = FRAGMENT_PPM) -> pd.DataFrame:
    """search the MS2 copies of every feature (ms2.SPECTRA: the consensus of all its spectra and the single
    highest-TIC spectrum, each raw and denoised) in a library, or in several as one: library_path is a .msp / .pkl
    path or a list of them, a feature's hits from all of them ranked together; each hit's library column names its
    file (library_names) and its library_index is its record's position in the libraries taken in order (for the
    first library, its position in that file). denoise picks the denoised copies: "search" (the
    denoising search, specdenoise.search: per candidate, the query denoised by tier T1 against that library record's
    precursor ion formula, its formula and adduct; once per query and ion formula; a record without a usable formula
    keeps the raw score, its denoise_ion empty), "t0" (ms2.DENOISE: one T0 copy per spectrum, judged from the m/z
    alone, written by the MS2 mapping; an output made before it is denoised first, on workers processes) or "off". A library record is listed on a raw copy and on its denoised copy alike when it scores
    above LIST_MIN (or at least min_score, when that is lower) on either, so every listed hit has both scores; ranks
    are per copy, and top_k (when given) keeps the records within the top_k of either. min_score is the annotation
    threshold: a feature's default hit is the better of its best hits on the default copies (ms2.default_copies: the
    denoised consensus and the denoised highest-TIC spectrum; the raw ones without denoising), and it is the
    feature's default annotation when it scores at least min_score (annotations.py). Fragment peaks match within the
    larger of fragment_tol (Da) and fragment_ppm x m/z (FRAGMENT_TOL, FRAGMENT_PPM: the denoising's tolerance), on
    the query's m/z with its file's offset (mz_offset_ppm) taken out. precursor_da (Da) replaces precursor_ppm as the
    precursor window when given. Writes identifications.tsv (one row per hit, spectrum
    = the copy), identifications.msp (the library spectrum of every listed record once, Comment library_index=...
    library=...), identifications.json (settings, the libraries, default copy and counts) and
    feature_annotations.tsv, replacing earlier ones; the
    search also writes identifications_denoise.npz, the keep mask behind every denoised hit (denoise_key: its index,
    -1 when the record gave no ion formula; masks run over the raw copy's peaks in file order)."""
    from . import ms2
    from .annotations import default_hits, write_table
    from .pipeline import log
    from .similarity import METHOD_CODE, clean_batch, score_pairs
    if method not in METHODS:
        raise ValueError(f"unknown similarity {method!r}; one of {', '.join(METHODS)}")
    if denoise not in DENOISE_MODES:
        raise ValueError(f"unknown denoising {denoise!r}; one of {', '.join(DENOISE_MODES)}")
    output = Path(output)
    paths = library_paths(library_path)
    names = library_names(paths)
    tic = time.perf_counter()
    denoised = False
    if denoise == "t0":
        try:
            denoised = ms2.ensure_denoised(output, workers)
        except Exception as e:        # the raw copies are searched regardless
            log(f"ms2 denoising skipped: {e!r}")
    t_dn = time.perf_counter() - tic
    stats: dict = {}
    dk = dict(group_id=[], copy=[], ion=[], masks=[])     # the masks the listed denoised hits use (DENOISE_FILE)
    libs = load_all(paths)
    first = np.zeros(len(libs) + 1, np.int64)             # library_index of each library's first record
    np.cumsum([len(x) for x in libs], out=first[1:])
    t_load = time.perf_counter() - tic - t_dn
    parts, nq = [], 0
    for raw in ms2.RAW:
        q = ms2.read_representatives(output, raw)
        if q is None:
            continue
        if raw == "single":
            nq = len(q["group_id"])
        if not first[-1]:
            continue
        q_mode = np.array([_MODE_CODE.get(m, 0) for m in q["ion_mode"]], np.int8)
        q_shift = np.asarray(q.get("mz_offset_ppm", np.zeros(len(q["group_id"]))), np.float64)
        q_mz = _calibrated(q["off"], q["mz"], q_shift)          # every copy is scored on the table's m/z scale
        qd = ms2.read_representatives(output, ms2.DENOISED[raw]) if denoised else None
        cleaned = []        # the query copies, cleaned once for all the libraries
        found = []          # the listed pairs of each library with any, and their scores on every copy
        for n, lib in enumerate(libs):
            if not len(lib):
                continue
            pq, pl = candidate_pairs(q["prec_mz"], q_mode, lib.precursor_mz, lib.mode, precursor_ppm, precursor_da,
                                     lib.precursor_index())
            if not len(pq):
                continue
            if not cleaned:
                copies = [(raw, q["off"], q_mz, q["inten"])]
                if qd is not None:
                    d_off, d_mz, d_it = _in_order(q["group_id"], qd)
                    copies.append((ms2.DENOISED[raw], d_off, _calibrated(d_off, d_mz, q_shift), d_it))
                cleaned = [(name, clean_batch(np.asarray(off, np.int64), np.asarray(mz, np.float64),
                                              np.asarray(inten, np.float64), q["prec_mz"], fragment_tol))
                           for name, off, mz, inten in copies]
            uniq, inv = np.unique(pl, return_inverse=True)
            lo, lm, li = _cleaned_library(lib, fragment_tol, uniq)
            scored = []
            for name, (qo, qm, qi) in cleaned:
                sc, nm = score_pairs(qo, qm, qi, lo, lm, li, pq, inv.astype(np.int64), METHOD_CODE[method],
                                     float(fragment_tol), float(fragment_ppm))
                scored.append((name, sc, nm))
            pair_ion = keys = None
            if denoise == "search":
                sc, nm, pair_ion, keys = _search_denoised(q, q_mz, lib, pq, pl, inv, lo, lm, li, scored[0][1],
                                                          scored[0][2], method, fragment_tol, fragment_ppm, stats)
                scored.append((ms2.DENOISED[raw], sc, nm))
            listed = np.zeros(len(pq), bool)
            for _, sc, nm in scored:
                listed |= ((sc > LIST_MIN) | (sc >= min_score)) & (nm > 0)
            sel = np.flatnonzero(listed)
            if len(sel):
                found.append(dict(n=n, q=pq[sel], rec=pl[sel], sc=[sc[sel] for _, sc, _ in scored],
                                  nm=[nm[sel] for _, _, nm in scored], ion=None if pair_ion is None else pair_ion[sel],
                                  key=None if keys is None else keys["pair_key"][sel], keys=keys))
        if not found:
            continue
        # every library's pairs in one list: ranks, top_k and the masks run over all of them
        copy_names = [name for name, _ in cleaned] + ([ms2.DENOISED[raw]] if denoise == "search" else [])
        at = np.zeros(len(found) + 1, np.int64)            # where each library's pairs start in the list
        np.cumsum([len(f["q"]) for f in found], out=at[1:])
        Q = np.concatenate([f["q"] for f in found])
        N = np.concatenate([np.full(len(f["q"]), f["n"], np.int64) for f in found])
        R = np.concatenate([f["rec"] for f in found])
        SC = [np.concatenate([f["sc"][c] for f in found]) for c in range(len(copy_names))]
        NM = [np.concatenate([f["nm"][c] for f in found]) for c in range(len(copy_names))]
        sel = np.arange(len(Q))
        if top_k:
            near = np.zeros(len(Q), bool)
            for sc, nm in zip(SC, NM):
                h = pd.DataFrame({"q": Q, "score": sc, "n": nm, "i": sel})
                h = h.sort_values(["q", "score", "n"], ascending=[True, False, False], kind="stable")
                near[h.groupby("q", sort=False).head(top_k)["i"].to_numpy()] = True
            sel = np.flatnonzero(near)
        ion = np.full(len(Q), "", object)                  # the search's denoised copy: ion formula and mask per pair
        dkey = np.full(len(Q), -1, np.int64)
        if denoise == "search":
            for k, f in enumerate(found):
                lib = libs[f["n"]]
                u, back = np.unique(f["ion"], return_inverse=True)
                ion[at[k]:at[k + 1]] = np.array([lib._ions.label(int(v)) for v in u], object)[back]
                mine = sel[(sel >= at[k]) & (sel < at[k + 1])]
                pk = f["key"][mine - at[k]]
                used, back = np.unique(pk[pk >= 0], return_inverse=True)
                dkey[mine[pk >= 0]] = len(dk["group_id"]) + back
                keys = f["keys"]
                for key in used.tolist():
                    dk["group_id"].append(int(q["group_id"][keys["key_q"][key]]))
                    dk["copy"].append(0 if raw == "consensus" else 1)
                    dk["ion"].append(lib._ions.label(int(keys["key_ion"][key])))
                    dk["masks"].append(keys["keep"][keys["key_off"][key]:keys["key_off"][key + 1]])
        Q, N, R, ion, dkey = Q[sel], N[sel], R[sel], ion[sel], dkey[sel]
        txt = {f: _record_text(libs, N, R, f) for f in ("names", "precursor_type", "formula", "inchikey", "smiles")}
        for c, name in enumerate(copy_names):
            hits = pd.DataFrame({"q": Q, "lib": first[N] + R, "score": SC[c][sel], "n_matched": NM[c][sel]})
            hits = hits.sort_values(["q", "score", "n_matched"], ascending=[True, False, False], kind="stable")
            hits["rank"] = hits.groupby("q", sort=False).cumcount() + 1
            o = hits.index.to_numpy()
            n_, r_ = N[o], R[o]
            lp = np.empty(len(o))
            for a in np.unique(n_).tolist():
                lp[n_ == a] = libs[a].precursor_mz[r_[n_ == a]]
            search_copy = denoise == "search" and name != raw
            parts.append(pd.DataFrame({
                "group_id": q["group_id"][hits["q"].to_numpy()], "spectrum": name, "rank": hits["rank"].to_numpy(),
                "score": hits["score"].round(4).to_numpy(), "n_matched": hits["n_matched"].to_numpy(),
                "name": txt["names"][o], "precursor_type": txt["precursor_type"][o], "formula": txt["formula"][o],
                "inchikey": txt["inchikey"][o], "smiles": txt["smiles"][o], "library_precursor_mz": lp,
                "ppm_error": np.round((q["prec_mz"][hits["q"].to_numpy()] - lp) / lp * 1e6, 2),
                "library_index": hits["lib"].to_numpy(), "library": np.array(names, object)[n_],
                "denoise_ion": ion[o] if search_copy else [""] * len(o),
                "denoise_key": dkey[o] if search_copy else np.full(len(o), -1, np.int64)}))
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=ID_COLUMNS)
    rows = rows[ID_COLUMNS]
    write_tsv(rows, output / "identifications.tsv")
    (output / DENOISE_FILE).unlink(missing_ok=True)
    if denoise == "search":
        lens = np.array([len(m) for m in dk["masks"]], np.int64)
        off = np.zeros(len(lens) + 1, np.int64)
        np.cumsum(lens, out=off[1:])
        with open(output / DENOISE_FILE, "wb") as fh:
            np.savez(fh, group_id=np.array(dk["group_id"], np.int64), copy=np.array(dk["copy"], np.uint8),
                     ion=np.array(dk["ion"], dtype=str), off=off,
                     keep=np.concatenate(dk["masks"]).astype(bool) if dk["masks"] else np.zeros(0, bool))
    # the library spectrum of every listed record once, in order of first listing
    recs = pd.unique(rows["library_index"].to_numpy(np.int64))
    rn = np.searchsorted(first, recs, side="right") - 1
    ri = recs - first[rn]
    lens = np.zeros(len(recs), np.int64)
    for a in np.unique(rn).tolist():
        k = rn == a
        lens[k] = libs[a].off[ri[k] + 1] - libs[a].off[ri[k]]
    roff = np.zeros(len(recs) + 1, np.int64)
    np.cumsum(lens, out=roff[1:])
    used = np.unique(rn).tolist()
    pk_mz = np.zeros(int(roff[-1]), np.result_type(np.float64, *[libs[a].mz.dtype for a in used]))
    pk_it = np.zeros(int(roff[-1]), np.result_type(np.float32, *[libs[a].inten.dtype for a in used]))
    for a in np.unique(rn).tolist():
        k = np.flatnonzero(rn == a)
        _, take = _csr_take(libs[a].off[ri[k]], lens[k])
        _, put = _csr_take(roff[k], lens[k])
        pk_mz[put], pk_it[put] = libs[a].mz[take], libs[a].inten[take]
    fields = []
    for g, n, i in zip(recs.tolist(), rn.tolist(), ri.tolist()):
        lib = libs[n]
        fields.append([("NAME", lib.names[i]), ("PRECURSORMZ", lib.precursor_mz[i]),
                       ("PRECURSORTYPE", lib.precursor_type[i]), ("FORMULA", lib.formula[i]),
                       ("INCHIKEY", lib.inchikey[i]), ("SMILES", lib.smiles[i]),
                       ("IONMODE", {1: "Positive", 2: "Negative"}.get(int(lib.mode[i]), "")),
                       ("Comment", f'library_index={g} library="{names[n].replace(chr(34), chr(39))}"')])
    with open(output / "identifications.msp", "w", encoding="utf-8") as fh:
        write_msp_records(fh, fields, roff, pk_mz, pk_it)
    if denoise == "t0":
        default = ms2.default_copies(output)
    else:                         # search: a denoised copy exists wherever its raw one does; off: raw copies only
        have = [c for c in ms2.RAW if (output / ms2.SPECTRA[c]).exists() and (output / ms2.SPECTRA[c]).stat().st_size]
        default = ([ms2.DENOISED[c] for c in have] if denoise == "search" else have) or ["single"]
        denoised = denoise == "search" and bool(have)
    if stats:
        stats["ion_formulas"] = sum(len(x._ions) for x in libs if x._ions is not None)
    n_id = int((default_hits(rows, default)["score"] >= min_score).sum())
    label = ms2.default_label(default)
    secs = time.perf_counter() - tic
    info = dict(method=method, method_label=METHODS[method], library="; ".join(map(str, paths)),
                libraries=[{"path": str(p), "name": nm, "n": len(x)} for p, nm, x in zip(paths, names, libs)],
                library_n=int(first[-1]), precursor_ppm=float(precursor_ppm),
                precursor_da=None if precursor_da is None else float(precursor_da), fragment_tol=float(fragment_tol),
                fragment_ppm=float(fragment_ppm), mz_calibrated=True,
                min_score=float(min_score), list_min=LIST_MIN, top_k=top_k, n_queries=int(nq), n_identified=n_id,
                n_hits=int(len(rows)), default_spectra=default, default_spectrum=default[0], default_label=label,
                denoise=(DENOISE_MODES[denoise] if denoise == "search" else ms2.DENOISE_LABEL) if denoised else None,
                denoise_mode=denoise, denoise_stats={k: round(v, 3) if isinstance(v, float) else v
                                                     for k, v in stats.items()} or None,
                seconds=round(secs, 3),
                created=time.strftime("%Y-%m-%d %H:%M:%S"))
    (output / "identifications.json").write_text(json.dumps(info, indent=1))
    write_table(output)
    ds = (f"; denoising search: {stats['keys']} query x ion formula keys for {stats['pairs']} pairs, "
          f"T1 {stats['seconds_t1']:.1f} s" if stats.get("pairs") else "")
    log(f"identify: {n_id} of {nq} features with MS2 annotated from {', '.join(names)} ({int(first[-1])} spectra, "
        f"{METHODS[method].lower()} >= {min_score:g} on the {label}; {len(rows)} hits above "
        f"{LIST_MIN:g} listed; {secs:.1f} s, librar{'ies' if len(libs) > 1 else 'y'} read {t_load:.1f} s"
        f"{f', denoising {t_dn:.1f} s' if t_dn >= 0.05 else ''}{ds})")
    return rows
