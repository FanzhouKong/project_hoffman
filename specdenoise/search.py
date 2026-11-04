"""Denoising search (after Kong et al., Nat Methods 2025): a query spectrum is denoised against each library candidate.

Every library record in a query's precursor window names a structure: its formula and adduct give the precursor ion
formula p, and tier T1 (denoise.py, explain_t1) keeps only the query ions whose neutral loss is a plausible
sub-formula of p. The query is scored against that candidate after this candidate-specific denoising, so a wrong
candidate cannot borrow the explanation of a right one.

Candidates in one window share few precursor ion formulas (many spectra per compound: collision energies,
instruments, replicate submissions; isomers), so the work is keyed by (query, ion formula), never by candidate:
  ion_formula     a record's (formula, adduct, ion mode, precursor m/z) -> (p, charge), the adduct inferred from the
                  precursor m/z when it is missing or disagrees with it; without a formula, the SMILES's (smiles.py)
  IonTable        library records -> ids of unique ion formulas, each (formula, adduct, mode) text parsed once
  unique_keys     candidate pairs -> unique (query, ion formula) keys, pair -> key
  denoise_keys    every key judged by T1 in one compiled pass, parallel over keys; electronic noise (identical
                  intensities) is found once per query
The keep masks equal denoise(mz, inten, precursor_mz, formula, adduct) (tier T1) key by key."""
from __future__ import annotations

import re

import numba
import numpy as np
from numba import njit, prange

from . import formula as fm
from .denoise import Params
from .smiles import smiles_formula
from .formula import BR, CL, NE, S
from .losses import explain_t1, iso_rescue

MZ_AGREE = 0.02     # Da: a precursor ion formula is used only when its m/z lies this close to the query's (as denoise())

# adducts tried, in this order, when a record names none (or one that does not fit its precursor m/z)
GUESS = {1: ("[M+H]+", "[M]+", "[M+Na]+", "[M+NH4]+", "[M+K]+", "[M+H-H2O]+", "[M+H-2H2O]+", "[M+2Na-H]+"),
         -1: ("[M-H]-", "[M]-", "[M+Cl]-", "[M+HCOO]-", "[M+CH3COO]-", "[M-H-H2O]-", "[M+Br]-")}
_GUESS_PARSED = {z: [(a, fm.parse_adduct(a)) for a in names] for z, names in GUESS.items()}
_CHARGED = re.compile(r"^\[?([A-Za-z0-9]+)\]?(\d*)([+-])$")


def _neutral(text) -> np.ndarray | None:
    """count vector of a library formula; a trailing charge mark ('C5H12NO+', '[C5H12NO]+') is dropped"""
    t = str(text or "").strip().replace(" ", "")
    if not t:
        return None
    f = fm.parse(t)
    if f is None:
        m = _CHARGED.match(t)
        if m and not m.group(2):
            f = fm.parse(m.group(1))
    return f


def _sign(adduct: str, mode: int) -> list[int]:
    a = str(adduct or "").strip()
    if a.endswith("+"):
        return [1]
    if a.endswith("-"):
        return [-1]
    return [1] if mode == 1 else [-1] if mode == 2 else [1, -1]


def neutral_forms(formula, smiles=None) -> list:
    """the (count vector M, net charge or None) a record may mean: its formula when that parses (charge unknown);
    otherwise from its SMILES the whole structure and its largest component (a salt's organic ion), each as written
    and, when charged, neutralised by protons (acetate -> acetic acid, a betaine cation -> the zwitterion, thiamine
    -> its free base); empty when neither gives one"""
    f = _neutral(formula)
    if f is not None:
        return [(f, None)]
    out = []
    if smiles:
        h = np.eye(NE, dtype=np.int64)[fm.H]
        for comp in ("all", "largest"):
            r = smiles_formula(smiles, comp)
            if r is None:
                continue
            forms = [r]
            if r[1] != 0 and r[0][fm.H] - r[1] >= 0:
                forms.append((r[0] - r[1] * h, 0))
            for g in forms:
                if not any((g[0] == x[0]).all() and g[1] == x[1] for x in out):
                    out.append(g)
    return out


_GUESS_ARR = {z: (np.array([a[0] for _, a in items], np.int64), np.array([a[1] for _, a in items], np.int64),
                   np.array([a[2] for _, a in items], np.int64), np.array([n in ("[M]+", "[M]-") for n, _ in items]))
              for z, items in _GUESS_PARSED.items()}
_CANDS: dict = {}
_FORMULA_OK: dict = {}


def _formula_ok(text) -> bool:
    ok = _FORMULA_OK.get(text)
    if ok is None:
        if len(_FORMULA_OK) > 500_000:
            _FORMULA_OK.clear()
        ok = _FORMULA_OK[text] = _neutral(text) is not None
    return ok


def _ion_candidates(formula, adduct, mode, smiles):
    """[M from a formula?, the named adduct's ions [(p, z, m/z)], the forms M, the polarities, the guessed ions (made
    on first use)] of one (formula, adduct, mode, SMILES) text: computed once, so records sharing them only compare
    m/z"""
    key = (formula, adduct, mode, smiles)
    c = _CANDS.get(key)
    if c is not None:
        return c
    forms = neutral_forms(formula, smiles)
    named = []
    if adduct:
        for f, _ in forms:
            pa = fm.precursor_formula(f, str(adduct).strip())
            if pa is not None:
                named.append((pa[0], pa[1], fm.ion_mz(*pa)))
    c = [bool(forms) and forms[0][1] is None, named, forms, _sign(adduct, mode), None]
    if len(_CANDS) > 500_000:
        _CANDS.clear()
    _CANDS[key] = c
    return c


def _guesses(c):
    """(m/z, ion count vectors, charges) of the GUESS adducts on every form of a candidate entry, in the order
    forms x polarities x GUESS; a guessed [M]+ / [M]- only for a structure with that net charge"""
    if c[4] is None:
        mzs, ps, zs = [], [], []
        for f, fz in c[2]:
            for z in c[3]:
                delta, nmol, charge, is_m = _GUESS_ARR[z]
                P = nmol[:, None] * f[None, :] + delta
                keep = (P >= 0).all(axis=1)
                if fz is not None and fz != z:
                    keep &= ~is_m
                P, ch = P[keep], charge[keep]
                mzs.append((P @ fm.MASSES - ch * fm.ELECTRON) / np.abs(ch))
                ps.append(P)
                zs.append(ch)
        c[4] = ((np.concatenate(mzs), np.concatenate(ps), np.concatenate(zs)) if mzs else
                (np.zeros(0), np.zeros((0, NE), np.int64), np.zeros(0, np.int64)))
    return c[4]


def _pick(c, precursor_mz):
    from_formula, named = c[0], c[1]
    if precursor_mz is None or not np.isfinite(precursor_mz):
        return (named[0][0].copy(), named[0][1]) if from_formula and named else None
    for p, z, mz in named:
        if abs(mz - precursor_mz) <= MZ_AGREE:
            return p.copy(), z
    gmz, gp, gz = _guesses(c)
    if len(gmz):
        e = np.abs(gmz - precursor_mz)
        i = int(np.argmin(e))
        if e[i] < MZ_AGREE:
            return gp[i].copy(), int(gz[i])
    return None


def ion_formula(formula, adduct, mode: int = 0, precursor_mz: float = float("nan"), smiles=None):
    """(precursor ion count vector, charge) of a library record, or None. mode: 1 positive, 2 negative, 0 unknown.
    M is the record's formula, or without one the forms its SMILES gives (neutral_forms, tried in that order; a
    guessed [M]+ / [M]- only for a structure with that net charge); a formula that fits no adduct at the precursor
    m/z (a typo such as C7H710O7) gives way to the SMILES. The named adduct is used when it parses and (with a
    precursor m/z) its m/z agrees within MZ_AGREE; otherwise the GUESS adduct of the record's polarity whose m/z is
    closest to the precursor m/z, within MZ_AGREE. Without a precursor m/z only a named adduct on a formula (never
    a SMILES) is taken unchecked."""
    formula, adduct, mode = formula or "", adduct or "", int(mode)
    ok = _formula_ok(formula)
    got = _pick(_ion_candidates(formula, adduct, mode, "" if ok else (smiles or "")), precursor_mz)
    if got is None and ok and smiles and precursor_mz is not None and np.isfinite(precursor_mz):
        got = _pick(_ion_candidates("", adduct, mode, smiles), precursor_mz)
    return got


class IonTable:
    """unique precursor ion formulas of library records, filled on demand and kept with the library: ids(records)
    looks only at records it has not seen; each distinct (formula, adduct, mode, SMILES) text is parsed once
    (_ion_candidates)"""

    def __init__(self, n_records: int):
        self.rec = np.full(n_records, -2, np.int32)     # -2 not looked at, -1 no usable ion formula, else the id
        self._id: dict = {}                              # (p bytes, charge) -> id
        self._p: list = []
        self._z: list = []
        self.P = np.zeros((0, NE), np.int64)
        self.Z = np.zeros(0, np.int64)
        self.mz = np.zeros(0)

    def __len__(self):
        return len(self._p)

    def ids(self, records, formula, adduct, mode, precursor_mz, smiles=None) -> np.ndarray:
        """ion formula id of each record index (formula, adduct, smiles: sequences of text by record; mode: int codes;
        precursor_mz: floats), -1 where there is none. The SMILES is read only for records without a usable
        formula."""
        records = np.asarray(records, np.int64)
        todo = records[self.rec[records] == -2]
        if len(todo):
            for r in np.unique(todo).tolist():
                pa = ion_formula(formula[r], adduct[r], int(mode[r]), float(precursor_mz[r]),
                                 smiles[r] if smiles is not None else None)
                if pa is None:
                    self.rec[r] = -1
                    continue
                p, z = pa
                k = (np.asarray(p, np.int64).tobytes(), int(z))
                i = self._id.get(k)
                if i is None:
                    i = self._id[k] = len(self._p)
                    self._p.append(np.asarray(p, np.int64))
                    self._z.append(int(z))
                self.rec[r] = i
            if len(self._p) != len(self.P):
                self.P = np.array(self._p, np.int64).reshape(-1, NE)
                self.Z = np.array(self._z, np.int64)
                self.mz = np.array([fm.ion_mz(p, z) for p, z in zip(self._p, self._z)])
        return self.rec[records]

    def label(self, i: int) -> str:
        """'C6H14NO2+' for an id ('' for -1)"""
        if i < 0:
            return ""
        z = int(self.Z[i])
        return fm.format(self.P[i]) + ("" if abs(z) == 1 else str(abs(z))) + ("+" if z > 0 else "-")


def unique_keys(pair_q, pair_ion, q_mz, ion_mz, ion_z):
    """(key_q, key_ion, pair_key): the unique (query, ion formula) combinations of candidate pairs and each pair's
    key (-1 where the pair has no usable ion formula: none, |charge| != 1, or its m/z more than MZ_AGREE from the
    query's)"""
    pair_q = np.asarray(pair_q, np.int64)
    pair_ion = np.asarray(pair_ion, np.int64)
    n_ion = max(1, len(ion_mz))
    ok = pair_ion >= 0
    if ok.any():
        ion = np.where(ok, pair_ion, 0)
        ok &= (np.abs(np.asarray(ion_z)[ion]) == 1) & (np.abs(np.asarray(ion_mz)[ion] - np.asarray(q_mz)[pair_q]) <= MZ_AGREE)
    code = pair_q[ok] * n_ion + pair_ion[ok]
    u, inv = np.unique(code, return_inverse=True)
    pair_key = np.full(len(pair_q), -1, np.int64)
    pair_key[ok] = inv
    return u // n_ion, u % n_ion, pair_key


@njit(cache=True)
def _electronic_one(inten, n_min, rel, out):
    """denoise.electronic_mask on one spectrum, into out (True = removed)"""
    order = np.argsort(inten)
    s = inten[order]
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] <= s[i] * (1 + rel):
            j += 1
        if j - i + 1 >= n_min:
            for t in range(i, j + 1):
                out[order[t]] = True
        i = j + 1


@njit(cache=True, parallel=True)
def _live(off, inten, use, n_min, rel):
    """per peak: not electronic noise (all True when use is False)"""
    dead = np.zeros(inten.shape[0], np.bool_)
    if use:
        for q in prange(off.shape[0] - 1):
            _electronic_one(inten[off[q]:off[q + 1]], n_min, rel, dead[off[q]:off[q + 1]])
    return ~dead


@njit(cache=True, parallel=True)
def _t1_keys(off, mz, inten, live, key_q, key_p, key_z, key_ref, key_off, tol_abs, tol_ppm, exchange, isotopes,
             max_losses, n_threads):
    nk = key_q.shape[0]
    keep = np.zeros(key_off[nk], np.bool_)
    n_chunk = min(n_threads, nk)
    for c in prange(n_chunk):
        work = np.zeros((max_losses, NE), np.int64)          # one decomposition buffer per thread, not per key
        for k in range(c, nk, n_chunk):                       # interleaved: a run of heavy keys spreads over threads
            q = key_q[k]
            a = off[q]
            n = off[q + 1] - a
            if n == 0:
                continue
            err = np.empty(n)
            err_x = np.empty(n)
            nexp = np.empty(n, np.int64)
            flag = np.empty(n, np.int64)
            p = key_p[k]
            explain_t1(mz[a:a + n], key_ref[k], key_z[k], p, tol_abs, tol_ppm, work, err, nexp, flag, err_x)
            o = key_off[k]
            reason = np.zeros(n, np.int8)
            for i in range(n):
                if not live[a + i]:
                    reason[i] = 5
                elif flag[i] == 1:
                    keep[o + i] = True
                    reason[i] = 1
                elif flag[i] == 2:
                    reason[i] = 4
                elif np.isfinite(err[i]) or (exchange and np.isfinite(err_x[i])):
                    keep[o + i] = True
                else:
                    reason[i] = 3
            if isotopes:
                allow = np.array([True, p[S] > 0, p[CL] > 0, p[BR] > 0])
                iso_rescue(mz[a:a + n], inten[a:a + n], keep[o:o + n], reason, tol_abs, tol_ppm, allow)
    return keep


def denoise_keys(off, mz, inten, precursor_mz, key_q, key_p, key_z, params: Params = Params()):
    """T1 keep masks of the (query, precursor ion formula) keys over a CSR batch of query spectra (off, mz ascending
    within each spectrum, intensity; precursor_mz per query): key k judges query key_q[k] with ion formula key_p[k]
    (count vector) and charge key_z[k] (+-1). Returns (key_off, keep): keep[key_off[k]:key_off[k+1]] is query
    key_q[k]'s mask, peak by peak. As denoise(..., formula, adduct) with params: losses measured from the query's
    precursor m/z (params.recalibrate) or the ion formula's; electronic noise, halide exchange, isotopologues."""
    off = np.ascontiguousarray(off, np.int64)
    mz = np.ascontiguousarray(mz, np.float64)
    inten = np.ascontiguousarray(inten, np.float64)
    key_q = np.ascontiguousarray(key_q, np.int64)
    key_p = np.ascontiguousarray(np.asarray(key_p, np.int64).reshape(-1, NE))
    key_z = np.ascontiguousarray(key_z, np.int64)
    lens = (off[1:] - off[:-1])[key_q]
    key_off = np.zeros(len(key_q) + 1, np.int64)
    np.cumsum(lens, out=key_off[1:])
    if not len(key_q):
        return key_off, np.zeros(0, bool)
    if len(mz) > 1:
        d = np.diff(mz)
        inner = np.ones(len(d), bool)
        inner[off[1:-1][(off[1:-1] > 0) & (off[1:-1] < len(mz))] - 1] = False
        if (d[inner] < 0).any():
            raise ValueError("denoise_keys needs m/z ascending within each spectrum")
    if params.recalibrate:
        ref = np.asarray(precursor_mz, np.float64)[key_q]
    else:
        ref = np.array([fm.ion_mz(p, z) for p, z in zip(key_p, key_z)])
    live = _live(off, inten, bool(params.electronic), int(params.electronic_n), 0.001)
    keep = _t1_keys(off, mz, inten, live, key_q, key_p, key_z, np.ascontiguousarray(ref, np.float64), key_off,
                    float(params.tol_abs), float(params.tol_ppm), bool(params.exchange), bool(params.isotopes),
                    int(params.max_losses), int(numba.get_num_threads()))
    return key_off, keep


def masked_spectra(off, mz, inten, key_q, key_off, keep):
    """the denoised spectra of the keys as one CSR batch (off, mz, intensity) over the query arrays given (their
    measured m/z, whatever m/z the masks were judged on)"""
    off = np.asarray(off, np.int64)
    key_q = np.asarray(key_q, np.int64)
    lens = key_off[1:] - key_off[:-1]
    src = np.repeat(off[key_q] - key_off[:-1], lens) + np.arange(key_off[-1])
    src = src[keep]
    owner = np.repeat(np.arange(len(key_q)), lens)[keep]
    d_off = np.zeros(len(key_q) + 1, np.int64)
    np.cumsum(np.bincount(owner, minlength=len(key_q)), out=d_off[1:])
    return d_off, np.asarray(mz)[src], np.asarray(inten)[src]
