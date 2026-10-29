"""Spectral libraries stored as ms_entropy FlashEntropySearch pickles (.pkl), the format of the MassCube MS/MS
databases (e.g. https://zenodo.org/records/19121645), read without ms_entropy and without running code from the file.

A pickle names the Python callables that rebuild its objects, so pickle.load on a downloaded file can run anything.
_Restricted resolves only numpy array, dtype and scalar reconstruction, plain containers, and the ms_entropy
classes, which it replaces by empty stand-ins that just take the stored attributes; any other name stops the read.

FlashEntropySearch.build_index keeps every library spectrum as its own pickled dict (precursor_mz, peaks, name,
precursor_type, ion_mode, formula, smiles, inchikey, ...) back to back in the byte array `metadata`, cut by
`metadata_loc`; those dicts are read here with the same restricted unpickler, a block of records at a time. The
result has the layout of msp.scan, so library.Library is built the same way for both formats."""
from __future__ import annotations

import builtins
import collections
import copyreg
import io
import pickle

import numpy as np

from .msp import COMMENT, FORMULA, INCHIKEY, KEYS, MODE, MODE_ALT, NAME, PREC, PTYPE, SMILES, TEXT

_BLOCK = 64 << 20              # metadata bytes decoded per unpickler (one copy of this size at a time)


class _Stub:
    """stand-in for an ms_entropy class: keeps whatever state the pickle gives it"""

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2:          # (dict, slots)
            for part in state:
                if isinstance(part, dict):
                    self.__dict__.update(part)
        elif isinstance(state, dict):
            self.__dict__.update(state)


_STUBS: dict = {}


def _path(*parts):
    return "/".join(str(p) for p in parts)


def _numpy_globals():
    out = {("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype, ("numpy", "memmap"): np.ndarray}
    rec = np.empty(0).__reduce__()[0]                              # numpy's array reconstructor
    buf = np.empty(0).__reduce_ex__(5)[0]                          # its pickle protocol 5 counterpart
    sca = np.float64(0).__reduce__()[0]                            # numpy scalars
    for core in ("numpy.core", "numpy._core"):
        out[(f"{core}.multiarray", "_reconstruct")] = rec
        out[(f"{core}.numeric", "_frombuffer")] = buf
        out[(f"{core}.multiarray", "scalar")] = sca
    for name in ("float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
                 "uint64", "bool_"):
        out[("numpy", name)] = getattr(np, name)
    return out


_ALLOWED = {**_numpy_globals(),
            **{("builtins", n): getattr(builtins, n) for n in ("dict", "list", "tuple", "set", "frozenset", "int",
                                                               "float", "complex", "str", "bytes", "bytearray",
                                                               "bool", "slice")},
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("copyreg", "_reconstructor"): copyreg._reconstructor,
            **{("pathlib", n): _path for n in ("Path", "PosixPath", "WindowsPath", "PurePath", "PurePosixPath",
                                                "PureWindowsPath")}}


class _Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        f = _ALLOWED.get((module, name))
        if f is not None:
            return f
        if module == "ms_entropy" or module.startswith("ms_entropy."):
            if (module, name) not in _STUBS:
                _STUBS[(module, name)] = type(name, (_Stub,), {"__module__": __name__})
            return _STUBS[(module, name)]
        raise pickle.UnpicklingError(f"this library file asks for {module}.{name}, which a spectral library never "
                                     "needs; it was not loaded")


def _mode_code(v) -> int:
    """1 positive, 2 negative, 0 unknown (the .msp reader's rule)"""
    v = str(v).strip().lower()
    if v.startswith("p") or v == "+" or "positive" in v:
        return 1
    if v.startswith("n") or v == "-" or "negative" in v:
        return 2
    return 0


def _charge_code(t) -> int:
    """1 / 2 from the charge sign after the closing bracket of a precursor type, or its last character"""
    t = str(t).strip()
    k = t.rfind("]")
    for ch in (t[k + 1:] if k >= 0 else t[-1:]):
        if ch == "+":
            return 1
        if ch == "-":
            return 2
    return 0


def _norm(key: str) -> str:
    return "".join(ch for ch in key.lower() if ch.isalnum())


def _text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v != v:
        return ""
    return str(v).strip()


def is_flash_pickle(path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(1) == b"\x80"


def read(path, keep_comment: bool = False):
    """msp.scan's tuple (text, text_off, precursor_mz, mode, off, mz, intensity, n_skipped) of a FlashEntropySearch
    pickle; keep_comment keeps a "comment" field when the records have one"""
    with open(path, "rb") as fh:
        obj = _Restricted(fh).load()
    meta, loc = getattr(obj, "metadata", None), getattr(obj, "metadata_loc", None)
    if meta is None or loc is None:
        raise ValueError(f"{path}: not a FlashEntropySearch library with stored spectra (no metadata found)")
    prec_arr = np.asarray(getattr(obj, "precursor_mz_array", np.zeros(0)), np.float64)
    meta = np.asarray(meta, np.uint8)
    loc = np.asarray(loc, np.int64)
    del obj                                    # drops the search index; only the stored spectra are kept
    n = len(loc) - 1
    nt = len(TEXT)
    keymaps: dict = {}
    mode_cache: dict = {}
    charge_cache: dict = {}
    prec = np.full(n, np.nan)
    mode = np.zeros(n, np.int8)
    off = np.zeros(n + 1, np.int64)
    toff = np.zeros(n * nt + 1, np.int64)
    text = bytearray()
    pmz, pit = np.empty(1 << 20), np.empty(1 << 20, np.float32)
    nrec = npk = skipped = 0
    i = 0
    while i < n:
        j = min(max(int(np.searchsorted(loc, loc[i] + _BLOCK, side="right")) - 1, i + 1), n)
        stream = io.BytesIO(meta[loc[i]:loc[j]].tobytes())
        for r in range(i, j):
            # a fresh unpickler per record: each was pickled on its own, so its memo must start empty (resetting
            # Unpickler.memo between loads crashes CPython's C unpickler)
            rec = _Restricted(stream).load()
            pk = rec.get("peaks") if isinstance(rec, dict) else None
            if pk is None or not len(pk):
                skipped += 1
                continue
            pk = np.asarray(pk, np.float64).reshape(-1, 2)
            m, v = pk[:, 0], pk[:, 1]
            ok = np.isfinite(m) & np.isfinite(v) & (v > 0)
            if not ok.all():
                m, v = m[ok], v[ok]
            if not len(m):
                skipped += 1
                continue
            if len(m) > 1 and (m[1:] < m[:-1]).any():
                o = np.argsort(m, kind="stable")
                m, v = m[o], v[o]
            ks = tuple(rec)
            km = keymaps.get(ks)
            if km is None:
                km = keymaps[ks] = {}
                for k in ks:
                    f = KEYS.get(_norm(str(k)))
                    if f is not None and f not in km:
                        km[f] = k
            if npk + len(m) > len(pmz):
                cap = max(2 * len(pmz), npk + len(m))
                pmz = np.concatenate([pmz[:npk], np.empty(cap - npk)])
                pit = np.concatenate([pit[:npk], np.empty(cap - npk, np.float32)])
            pmz[npk:npk + len(m)] = m
            pit[npk:npk + len(m)] = v
            npk += len(m)
            off[nrec + 1] = npk
            pv = np.nan
            if PREC in km:
                try:
                    pv = float(rec[km[PREC]])
                except (TypeError, ValueError):
                    pass
            if not np.isfinite(pv) and r < len(prec_arr):
                pv = prec_arr[r]
            prec[nrec] = pv
            md = 0
            for f in (MODE, MODE_ALT):
                raw = rec.get(km[f]) if f in km else None
                if raw is not None and str(raw).strip():
                    md = mode_cache.get(raw)
                    if md is None:
                        md = mode_cache[raw] = _mode_code(raw)
                    break
            pt = _text(rec.get(km[PTYPE])) if PTYPE in km else ""
            if md == 0 and pt:
                md = charge_cache.get(pt)
                if md is None:
                    md = charge_cache[pt] = _charge_code(pt)
            mode[nrec] = md
            w = nrec * nt
            for k, f in enumerate(TEXT):
                if f == PTYPE:
                    text += pt.encode("utf-8", "replace")
                elif f in km and (f != COMMENT or keep_comment):
                    text += _text(rec.get(km[f])).encode("utf-8", "replace")
                toff[w + k + 1] = len(text)
            nrec += 1
        i = j
    return (np.frombuffer(bytes(text), np.uint8), toff[:nrec * nt + 1].copy(), prec[:nrec].copy(), mode[:nrec].copy(),
            off[:nrec + 1].copy(), pmz[:npk].copy(), pit[:npk].copy(), skipped)
