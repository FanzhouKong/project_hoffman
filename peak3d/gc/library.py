"""EI spectral libraries (.msp) and the search of deconvolved spectra, with a retention-index window.

An EI record has no precursor; what narrows the search is the retention index. The reader takes the common .msp
dialects (NIST, MoNA, MassBank, Fiehn/MS-DIAL exports): Name, Formula, InChIKey, MW / ExactMass, the retention
index under its usual keys (Retention_index, RI, Kovats ..., NIST's "SemiStdNP=1128/3/15 StdNP=... StdPolar=..."
line, or "retention index=" inside the comment), and peak lines as "m/z intensity" pairs, one or several per line.
Records marked MS1 are EI spectra in MoNA exports and are kept.

Most EI libraries hold unit masses. A library is unit-mass when nearly all its peaks sit on integers; a measured
accurate-mass spectrum is then binned to nominal masses, a peak at m belonging to the integer n with
n - 0.3 <= m < n + 0.7 (the usual GC-MS rule: fragments are mass-sufficient by up to a few tenths, never
deficient by more than 0.3). An accurate-mass library is matched peak by peak within max(3 mDa, 10 ppm).

Scores, on peak weights intensity^0.6 x m/z^3 (Stein & Scott 1994), over the m/z range the run acquired:
  match    cosine of the two weighted spectra (identity search: every peak of both counts)
  reverse  the same with the measured peaks that have no library partner left out (a spectrum that still holds a
           co-eluting compound's ions keeps its score)
Hits are ranked by match. With a retention index on both sides a record outside +/- ri_tol is not a candidate;
records without an index stay candidates (many library entries have none)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

CACHE_VERSION = 1
NOMINAL_SHARE = 0.98       # a library with this share of its peaks within 0.02 of an integer holds unit masses
LOW, HIGH = 0.3, 0.7       # nominal mass n holds n - LOW <= m < n + HIGH
HIT_COLUMNS = ["query", "rank", "match", "reverse", "n_matched", "name", "formula", "inchikey", "library_ri",
               "delta_ri", "library_index", "library"]
_RI_KEYS = {"retentionindex", "ri", "kovats", "kovatsri", "kovatsindex", "retentionindices", "semistdnp", "stdnp",
            "lri", "linearretentionindex"}
_RI_TYPES = {"semi": ("semistdnp", "stdnp", "stdpolar"), "std": ("stdnp", "semistdnp", "stdpolar"),
             "polar": ("stdpolar",)}
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_COMMENT_RI = re.compile(r"(?:retention[ _]?index|kovats(?:[ _]ri)?|\bri)\s*[=:]\s*([0-9.]+)", re.I)


@dataclass
class EiLibrary:
    path: str
    name: np.ndarray
    formula: np.ndarray
    inchikey: np.ndarray
    ri: np.ndarray             # nan where the record has none
    mw: np.ndarray
    off: np.ndarray
    mz: np.ndarray
    inten: np.ndarray
    nominal: bool = False
    n_skipped: int = 0

    def __len__(self):
        return len(self.name)

    def spectrum(self, i: int):
        return self.mz[self.off[i]:self.off[i + 1]], self.inten[self.off[i]:self.off[i + 1]]


def _ri_value(key: str, value: str, ri_type: str) -> float:
    if "=" in value:                                   # NIST: SemiStdNP=1128/3/15 StdNP=1100/5/8 StdPolar=1650/10/3
        found = {k.lower(): v for k, v in re.findall(r"([A-Za-z]+)\s*=\s*([0-9.]+)", value)}
        for t in _RI_TYPES.get(ri_type, _RI_TYPES["semi"]):
            if t in found:
                return float(found[t])
        return float("nan")
    m = _NUM.search(value)
    return float(m.group()) if m else float("nan")


def _peaks(lines):
    mz, it = [], []
    for ln in lines:
        q = ln.find('"')
        if q >= 0:
            ln = ln[:q]
        pieces = ln.split(";") if ";" in ln else [ln]
        for piece in pieces:
            tok = piece.replace(",", " ").replace(":", " ").replace("(", " ").replace(")", " ").split()
            if len(tok) < 2:
                continue
            if ";" not in ln and len(tok) > 3 and len(tok) % 2 == 0:      # "m i m i m i" on one line
                pairs = zip(tok[0::2], tok[1::2])
            else:
                pairs = [(tok[0], tok[1])]
            for a, b in pairs:
                try:
                    x, y = float(a), float(b)
                except ValueError:
                    continue
                if x > 0 and y > 0 and np.isfinite(x) and np.isfinite(y):
                    mz.append(x)
                    it.append(y)
    return mz, it


def read_msp(path, ri_type: str = "semi") -> EiLibrary:
    """Read an EI .msp library. ri_type picks the value from NIST's three-column index line: semi (semi-standard
    non-polar, e.g. DB-5; the default), std (standard non-polar) or polar."""
    names, formulas, keys, ris, mws, offs, mzs, its = [], [], [], [], [], [0], [], []
    skipped = 0
    rec = None
    peak_lines = []

    def close():
        nonlocal rec, peak_lines, skipped
        if rec is None:
            return
        m, i = _peaks(peak_lines)
        if m:
            o = np.argsort(m, kind="stable")
            ri = rec.get("ri", float("nan"))
            if not np.isfinite(ri) and rec.get("comment"):
                hit = _COMMENT_RI.search(rec["comment"])
                ri = float(hit.group(1)) if hit else float("nan")
            names.append(rec.get("name", ""))
            formulas.append(rec.get("formula", ""))
            keys.append(rec.get("inchikey", ""))
            ris.append(ri)
            mws.append(rec.get("mw", float("nan")))
            mzs.append(np.asarray(m, np.float64)[o])
            its.append(np.asarray(i, np.float32)[o])
            offs.append(offs[-1] + len(m))
        else:
            skipped += 1
        rec, peak_lines = None, []

    with open(path, "r", encoding="utf-8", errors="replace", newline=None) as fh:
        for raw in fh:
            ln = raw.strip()
            if not ln:
                if peak_lines:
                    close()
                continue
            c = ln[0]
            if c.isdigit() or (c == "." and len(ln) > 1 and ln[1].isdigit()):
                if rec is not None:
                    peak_lines.append(ln)
                continue
            k, sep, v = ln.partition(":")
            if not sep:
                continue
            key = re.sub(r"[^a-z0-9]", "", k.lower())
            v = v.strip()
            if key == "name":
                if rec is not None and (peak_lines or "name" in rec):
                    close()
                rec = {"name": v}
                continue
            if rec is None:
                rec = {}
            if key in ("formula", "inchikey") and key not in rec:
                rec[key] = v
            elif key in ("comment", "comments") and "comment" not in rec:
                rec["comment"] = v
            elif key in _RI_KEYS and not np.isfinite(rec.get("ri", float("nan"))):
                rec["ri"] = _ri_value(key, v, ri_type)
            elif key in ("mw", "exactmass", "molecularweight") and "mw" not in rec:
                m = _NUM.search(v)
                rec["mw"] = float(m.group()) if m else float("nan")
        close()
    mz = np.concatenate(mzs) if mzs else np.zeros(0)
    nominal = bool(len(mz)) and float(np.mean(np.abs(mz - np.round(mz)) <= 0.02)) >= NOMINAL_SHARE
    return EiLibrary(str(path), np.array(names, dtype=str), np.array(formulas, dtype=str), np.array(keys, dtype=str),
                     np.asarray(ris, np.float64), np.asarray(mws, np.float64), np.asarray(offs, np.int64), mz,
                     np.concatenate(its) if its else np.zeros(0, np.float32), nominal, skipped)


def load(path, cache_dir=None, ri_type: str = "semi") -> EiLibrary:
    """read_msp with the parsed library kept in cache_dir (keyed on the file's path, size and time): a large
    library is read as text once"""
    path = Path(path).expanduser()
    if cache_dir is None:
        return read_msp(path, ri_type)
    st = path.stat()
    tag = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{ri_type}|{CACHE_VERSION}".encode()).hexdigest()[:16]
    f = Path(cache_dir) / f"ei_library_{tag}.npz"
    if f.exists():
        try:
            with np.load(f, allow_pickle=False) as z:
                return EiLibrary(str(path), z["name"], z["formula"], z["inchikey"], z["ri"], z["mw"], z["off"], z["mz"],
                                 z["inten"], bool(z["nominal"]), int(z["n_skipped"]))
        except Exception:
            f.unlink(missing_ok=True)
    lib = read_msp(path, ri_type)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + ".part.npz")
    np.savez(tmp, name=lib.name, formula=lib.formula, inchikey=lib.inchikey, ri=lib.ri, mw=lib.mw, off=lib.off, mz=lib.mz,
             inten=lib.inten, nominal=np.array(lib.nominal), n_skipped=np.array(lib.n_skipped))
    tmp.replace(f)
    return lib


def write_msp(path, names, off, mz, inten, ri=None, rt=None, extra=None) -> None:
    """Spectra as an EI .msp (accurate m/z, intensities scaled to a base peak of 999). extra: per-record lists of
    (key, value) lines."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for k, name in enumerate(names):
            m, i = mz[off[k]:off[k + 1]], inten[off[k]:off[k + 1]]
            fh.write(f"Name: {name}\n")
            if ri is not None and np.isfinite(ri[k]):
                fh.write(f"Retention_index: {ri[k]:.1f}\n")
            if rt is not None and np.isfinite(rt[k]):
                fh.write(f"RetentionTime: {rt[k]:.4f}\n")
            for key, val in (extra[k] if extra is not None else ()):
                fh.write(f"{key}: {val}\n")
            fh.write(f"Num Peaks: {len(m)}\n")
            top = float(i.max()) if len(i) else 1.0
            for a, b in zip(m, i):
                fh.write(f"{a:.5f} {999.0 * b / top:.2f}\n")
            fh.write("\n")


# ------------------------------------------------------------------ scores ---
def nominal_mass(mz) -> np.ndarray:
    return np.floor(np.asarray(mz, np.float64) + LOW).astype(np.int64)


@njit(cache=True)
def _insert(top_s, top_i, top_r, top_n, s, i, r, n):
    """keep the best scores in descending order"""
    K = len(top_s)
    if s <= top_s[K - 1]:
        return
    k = K - 1
    while k > 0 and top_s[k - 1] < s:
        top_s[k] = top_s[k - 1]; top_i[k] = top_i[k - 1]; top_r[k] = top_r[k - 1]; top_n[k] = top_n[k - 1]
        k -= 1
    top_s[k] = s; top_i[k] = i; top_r[k] = r; top_n[k] = n


@njit(cache=True, parallel=True)
def _search_nominal(q_dense, q_lo, q_hi, l_off, l_bin, l_w, lo, hi, no_ri0, n_lib, top_k):
    """q_dense: (queries, bins) peak weights on nominal masses; library CSR of nominal masses and weights, sorted
    by retention index with the index-less records from no_ri0 on. Query q is scored against records
    lo[q]..hi[q]-1 and no_ri0..n_lib-1, on the bins q_lo[q]..q_hi[q]."""
    nq = q_dense.shape[0]
    nb = q_dense.shape[1]
    S = np.full((nq, top_k), -1.0)
    I = np.full((nq, top_k), -1, np.int64)
    R = np.zeros((nq, top_k))
    N = np.zeros((nq, top_k), np.int64)
    for q in prange(nq):
        qd = q_dense[q]
        qn = 0.0
        for b in range(q_lo[q], min(q_hi[q] + 1, nb)):
            qn += qd[b] * qd[b]
        if qn <= 0:
            continue
        for part in range(2):
            a0 = lo[q] if part == 0 else max(no_ri0, hi[q])
            a1 = hi[q] if part == 0 else n_lib
            for l in range(a0, a1):
                num = 0.0
                ln = 0.0
                qm = 0.0
                nm = 0
                for p in range(l_off[l], l_off[l + 1]):
                    b = l_bin[p]
                    if b < q_lo[q] or b > q_hi[q]:
                        continue
                    w = l_w[p]
                    ln += w * w
                    if b < nb and qd[b] > 0:
                        num += w * qd[b]
                        qm += qd[b] * qd[b]
                        nm += 1
                if nm == 0 or ln <= 0:
                    continue
                s = min(num / np.sqrt(qn * ln), 1.0)
                r = min(num / np.sqrt(qm * ln), 1.0)
                _insert(S[q], I[q], R[q], N[q], s, l, r, nm)
    return S, I, R, N


@njit(cache=True, parallel=True)
def _search_exact(q_off, q_mz, q_w, q_lo, q_hi, l_off, l_mz, l_w, lo, hi, no_ri0, n_lib, top_k, tol, ppm):
    """as _search_nominal for an accurate-mass library: peaks pair up within max(tol, ppm x m/z), each once, in
    ascending m/z; q_lo, q_hi are m/z limits"""
    nq = len(q_off) - 1
    S = np.full((nq, top_k), -1.0)
    I = np.full((nq, top_k), -1, np.int64)
    R = np.zeros((nq, top_k))
    N = np.zeros((nq, top_k), np.int64)
    for q in prange(nq):
        a_q, b_q = q_off[q], q_off[q + 1]
        qn = 0.0
        for p in range(a_q, b_q):
            qn += q_w[p] * q_w[p]
        if qn <= 0:
            continue
        for part in range(2):
            a0 = lo[q] if part == 0 else max(no_ri0, hi[q])
            a1 = hi[q] if part == 0 else n_lib
            for l in range(a0, a1):
                num = 0.0
                ln = 0.0
                qm = 0.0
                nm = 0
                a = a_q
                for p in range(l_off[l], l_off[l + 1]):
                    m = l_mz[p]
                    if m < q_lo[q] or m > q_hi[q]:
                        continue
                    w = l_w[p]
                    ln += w * w
                    d = max(tol, ppm * 1e-6 * m)
                    while a < b_q and q_mz[a] < m - d:
                        a += 1
                    if a < b_q and q_mz[a] <= m + d:
                        num += w * q_w[a]
                        qm += q_w[a] * q_w[a]
                        nm += 1
                        a += 1
                if nm == 0 or ln <= 0:
                    continue
                s = min(num / np.sqrt(qn * ln), 1.0)
                r = min(num / np.sqrt(qm * ln), 1.0)
                _insert(S[q], I[q], R[q], N[q], s, l, r, nm)
    return S, I, R, N


def _weights(mz, inten, intensity_power, mass_power):
    return np.asarray(inten, np.float64) ** intensity_power * (np.asarray(mz, np.float64) / 100.0) ** mass_power


@dataclass
class SearchParams:
    intensity_power: float = 0.6
    mass_power: float = 3.0
    ri_tol: float = float("nan")       # index window; nan: no window
    keep_no_ri: bool = True            # records without an index stay candidates when a window is set
    fragment_tol: float = 0.003        # accurate-mass library: Da
    fragment_ppm: float = 10.0
    min_rel: float = 0.005             # measured peaks below this fraction of the base peak are left out
    top_k: int = 10
    min_match: float = 0.3             # hits below it are not listed


@dataclass
class Prepared:
    """a library sorted by retention index, with its peak weights"""
    lib: EiLibrary
    order: np.ndarray
    ri: np.ndarray
    no_ri0: int
    off: np.ndarray
    mz: np.ndarray
    w: np.ndarray
    key: tuple = field(default_factory=tuple)


def prepare(lib: EiLibrary, P: SearchParams) -> Prepared:
    order = np.argsort(np.where(np.isfinite(lib.ri), lib.ri, np.inf), kind="stable")
    ri = lib.ri[order]
    no_ri0 = int(np.isfinite(ri).sum())
    lens = np.diff(lib.off)[order]
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    take = np.concatenate([np.arange(lib.off[i], lib.off[i + 1]) for i in order]) if len(order) else np.zeros(0, np.int64)
    mz = lib.mz[take]
    w = _weights(nominal_mass(mz) if lib.nominal else mz, lib.inten[take], P.intensity_power, P.mass_power)
    return Prepared(lib, order, ri, no_ri0, off, mz, w, (P.intensity_power, P.mass_power))


def search(q_off, q_mz, q_int, lib: EiLibrary | Prepared, q_ri=None, P: SearchParams | None = None,
           mz_range=None, library_name: str = "") -> pd.DataFrame:
    """Hits of every query spectrum (CSR; m/z ascending) in the library: one row per hit, HIT_COLUMNS, best first.
    q_ri: the queries' retention indices (nan: none). mz_range: (low, high) m/z the run acquired; library peaks
    outside it are left out of the comparison (default: each query's own lowest and highest m/z, widened by 1)."""
    P = P or SearchParams()
    prep = lib if isinstance(lib, Prepared) else prepare(lib, P)
    L = prep.lib
    nq = len(q_off) - 1
    q_ri = np.full(nq, np.nan) if q_ri is None else np.asarray(q_ri, np.float64)
    if nq == 0 or len(L) == 0:
        return pd.DataFrame(columns=HIT_COLUMNS)
    # measured spectra: drop the weakest peaks
    keep = np.zeros(len(q_mz), bool)
    lo_m, hi_m = np.zeros(nq), np.zeros(nq)
    for q in range(nq):
        a, b = q_off[q], q_off[q + 1]
        if b > a:
            keep[a:b] = q_int[a:b] >= P.min_rel * q_int[a:b].max()
            lo_m[q], hi_m[q] = q_mz[a] - 1.0, q_mz[b - 1] + 1.0
    if mz_range is not None:
        lo_m[:], hi_m[:] = float(mz_range[0]), float(mz_range[1])
    counts = np.array([int(keep[q_off[q]:q_off[q + 1]].sum()) for q in range(nq)], np.int64)
    off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    m, i = np.asarray(q_mz, np.float64)[keep], np.asarray(q_int, np.float64)[keep]
    # candidate range on the index axis
    n_ri = prep.no_ri0
    lo = np.zeros(nq, np.int64)
    hi = np.full(nq, n_ri, np.int64)
    no_ri0 = n_ri
    if np.isfinite(P.ri_tol):
        has = np.isfinite(q_ri)
        lo[has] = np.searchsorted(prep.ri[:n_ri], q_ri[has] - P.ri_tol, side="left")
        hi[has] = np.searchsorted(prep.ri[:n_ri], q_ri[has] + P.ri_tol, side="right")
        if not P.keep_no_ri:
            no_ri0 = len(L)
    if L.nominal:
        nb = int(max(nominal_mass(m).max() if len(m) else 0, nominal_mass(prep.mz).max() if len(prep.mz) else 0)) + 2
        dense = np.zeros((nq, nb))
        row = np.repeat(np.arange(nq), counts)
        np.add.at(dense, (row, nominal_mass(m)), i)                 # peaks of one nominal mass add up
        dense = dense ** P.intensity_power * (np.arange(nb) / 100.0) ** P.mass_power
        S, I, R, N = _search_nominal(dense, nominal_mass(lo_m + LOW + 1e-9).astype(np.int64), nominal_mass(hi_m).astype(np.int64),
                                     prep.off, nominal_mass(prep.mz), prep.w, lo, hi, no_ri0, len(L), P.top_k)
    else:
        S, I, R, N = _search_exact(off, m, _weights(m, i, P.intensity_power, P.mass_power), lo_m, hi_m, prep.off, prep.mz,
                                   prep.w, lo, hi, no_ri0, len(L), P.top_k, P.fragment_tol, P.fragment_ppm)
    rows = []
    for q in range(nq):
        for k in range(P.top_k):
            if I[q, k] < 0 or S[q, k] < P.min_match:
                break
            rec = int(prep.order[I[q, k]])
            ri = float(L.ri[rec])
            rows.append((q, k + 1, round(float(S[q, k]), 4), round(float(R[q, k]), 4), int(N[q, k]), str(L.name[rec]),
                         str(L.formula[rec]), str(L.inchikey[rec]), ri,
                         q_ri[q] - ri if np.isfinite(ri) and np.isfinite(q_ri[q]) else np.nan, rec, library_name))
    return pd.DataFrame(rows, columns=HIT_COLUMNS)
