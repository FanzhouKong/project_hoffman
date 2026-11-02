"""Elemental formulas as int64 vectors, adducts, and the plausibility rules.

Element order is by decreasing monoisotopic mass with hydrogen last: that is the order the mass decomposer wants
(heavy elements first, hydrogen solved directly). Rules take a count vector and the formula mass and return a
bool; they are numba functions so the decomposer can call them in its inner loop."""
from __future__ import annotations

import re

import numpy as np
from numba import njit

ELEMENTS = ("I", "Br", "K", "Cl", "S", "P", "Si", "Na", "F", "O", "N", "C", "H")
I, BR, K, CL, S, P, SI, NA, F, O, N, C, H = range(13)
NE = 13
MASSES = np.array([126.904473, 78.9183371, 38.96370668, 34.96885268, 31.97207100, 30.97376163, 27.9769265325,
                   22.9897692809, 18.99840322, 15.99491461956, 14.0030740048, 12.0, 1.00782503207])
ELECTRON = 0.00054857990946
PROTON = MASSES[H] - ELECTRON
C13 = 1.0033548378          # 13C - 12C
IDX = {e: i for i, e in enumerate(ELEMENTS)}

# Seven Golden Rules, rule 1: element count limits, 99.7 % coverage, by mass window (upper edge, Da)
_CAP_WINDOWS = np.array([500.0, 1000.0, 2000.0, 3000.0])
_CAPS = np.array([
    # I  Br  K  Cl  S   P  Si  Na  F   O   N   C    H     (Na, K raised from the Golden Rules' 1 to 4: multiply
    #                                                        cationised precursors, [M-2H+3Na]+, keep 2-3 Na in fragments)
    [4, 4, 4, 10, 10, 9, 8, 4, 16, 20, 20, 39, 72],
    [8, 8, 4, 12, 14, 9, 14, 4, 34, 27, 20, 78, 126],
    [10, 10, 4, 12, 14, 9, 15, 4, 48, 63, 20, 156, 236],
    [8, 8, 4, 11, 9, 6, 15, 4, 16, 78, 48, 162, 208],
], dtype=np.int64)

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse(text: str) -> np.ndarray | None:
    """'C4H9N3O2' -> count vector; None when the text is not a plain formula over the known elements"""
    text = str(text).strip()
    if not text or not re.fullmatch(r"([A-Z][a-z]?\d*)+", text):
        return None
    f = np.zeros(NE, np.int64)
    for sym, num in _TOKEN.findall(text):
        if sym not in IDX:
            return None
        f[IDX[sym]] += int(num) if num else 1
    return f


def format(f) -> str:
    """Hill-order text of a count vector ('' for the empty formula)"""
    f = np.asarray(f)
    parts = []
    order = [C, H] + sorted((i for i in range(NE) if i not in (C, H)), key=lambda i: ELEMENTS[i])
    for i in order:
        if f[i] > 0:
            parts.append(ELEMENTS[i] + (str(int(f[i])) if f[i] > 1 else ""))
    return "".join(parts)


def mass(f) -> float:
    return float(np.dot(np.asarray(f, np.float64), MASSES))


def ion_mz(f, charge: int) -> float:
    """m/z of an ion with count vector f and charge (sign carries the polarity)"""
    return (mass(f) - charge * ELECTRON) / abs(charge)


def caps(m: float) -> np.ndarray:
    k = int(np.searchsorted(_CAP_WINDOWS, m, side="right"))
    return _CAPS[min(k, len(_CAPS) - 1)].copy()


# ----------------------------------------------------------------------------------------------- adducts
# neutral pieces that adduct strings name
_PIECES = {"H": "H", "Na": "Na", "K": "K", "NH4": "NH4", "NH3": "NH3", "H2O": "H2O", "Cl": "Cl", "Br": "Br",
           "HCOO": "CHO2", "HCOOH": "CH2O2", "FA": "CH2O2", "CH3COO": "C2H3O2", "CH3COOH": "C2H4O2", "Hac": "C2H4O2",
           "OAc": "C2H3O2", "ACN": "C2H3N", "CH3CN": "C2H3N", "CH3OH": "CH4O", "MeOH": "CH4O", "IsoProp": "C3H8O",
           "DMSO": "C2H6OS", "CO2": "CO2", "CO": "CO", "HF": "HF", "HCl": "HCl", "i": None}
_ADDUCT = re.compile(r"^\[(\d*)M((?:[+-][A-Za-z0-9]+)*)\](\d*)([+-])$")
_TERM = re.compile(r"([+-])(\d*)([A-Za-z][A-Za-z0-9]*)")


def parse_adduct(text: str):
    """'[M+H-H2O]+' -> (delta count vector added to n*M, n, charge); None if unknown. '[M+H+2i]+' (isotopologue
    precursors) are refused: their precursor m/z is not the monoisotopic one."""
    m = _ADDUCT.match(str(text).replace(" ", ""))
    if not m:
        return None
    nmol = int(m.group(1) or 1)
    charge = int(m.group(3) or 1) * (1 if m.group(4) == "+" else -1)
    delta = np.zeros(NE, np.int64)
    for sign, num, piece in _TERM.findall(m.group(2)):
        n = int(num or 1)
        if piece in _PIECES:
            if _PIECES[piece] is None:
                return None
            vec = parse(_PIECES[piece])
        else:
            vec = parse(piece)
            if vec is None:
                return None
        delta += (n if sign == "+" else -n) * vec
    return delta, nmol, charge


def precursor_formula(neutral, adduct: str):
    """(ion count vector, charge) of adduct applied to the neutral formula; None when unknown or impossible"""
    a = parse_adduct(adduct)
    if a is None:
        return None
    delta, nmol, charge = a
    p = nmol * np.asarray(neutral, np.int64) + delta
    if (p < 0).any():
        return None
    return p, charge


# ------------------------------------------------------------------------------------------------- rules
@njit(cache=True)
def _caps_row(m):
    if m < 500.0:
        return 0
    if m < 1000.0:
        return 1
    if m < 2000.0:
        return 2
    return 3


@njit(cache=True)
def dbe2(f):
    """twice the ring-and-double-bond equivalent (monovalent: H, halogens, Na, K; trivalent: N, P)"""
    return 2 + 2 * (f[C] + f[SI]) - (f[H] + f[F] + f[CL] + f[BR] + f[I] + f[NA] + f[K]) + (f[N] + f[P])


@njit(cache=True)
def _ratios_ok(f, caps_tab, row):
    c = f[C]
    for e in range(NE):
        if f[e] > caps_tab[row, e]:
            return False
    if f[N] > 4 * c or f[O] > 3 * c or f[P] > 2 * c or f[S] > 3 * c or f[F] > 6 * c or f[CL] > 2 * c \
            or f[BR] > 2 * c or f[I] > 2 * c or f[SI] > c:
        return False
    return True


@njit(cache=True)
def _combos_ok(f):
    """rule 6, element probability"""
    n, o, p, s = f[N], f[O], f[P], f[S]
    if n > 0 and o > 0 and p > 0 and s > 0 and (n >= 10 or o >= 20 or p >= 4 or s >= 3):
        return False
    if n > 3 and o > 3 and p > 3 and (n >= 11 or o >= 22 or p >= 6):
        return False
    if o > 1 and p > 1 and s > 1 and (o >= 14 or p >= 3 or s >= 3):
        return False
    if p > 1 and s > 1 and n > 1 and (p >= 3 or s >= 3 or n >= 4):
        return False
    if n > 6 and o > 6 and s > 6 and (n >= 19 or o >= 14 or s >= 8):
        return False
    return True


@njit(cache=True)
def hmax(f, extra):
    """most hydrogens a formula can carry when it is a sum of saturated pieces: 2 per C, O, S; 3 per N, P;
    one per monovalent atom; plus extra (2 for a loss made of several neutral molecules or a cation)"""
    return 2 * (f[C] + f[O] + f[S]) + 3 * (f[N] + f[P]) + (f[F] + f[CL] + f[BR] + f[I] + f[NA] + f[K]) + 4 * f[SI] + extra


@njit(cache=True)
def _small_inorganic(f, extra_h):
    """carbon-free formula allowed: at most 6 heavy atoms of at most 2 kinds, hydrogens within hmax
    (H2O, 3 H2O, NH3, NH3+H2O, HCl, H2S, SO3, H3PO4, PO3-, HSO4-, NH4+, Na+, NaCl ...)"""
    heavy = 0
    kinds = 0
    for e in range(NE - 1):
        if f[e] > 0:
            heavy += f[e]
            kinds += 1
    return heavy >= 1 and heavy <= 6 and kinds <= 2 and f[H] <= hmax(f, extra_h)


@njit(cache=True)
def loss_ok(f, m, caps_tab):
    """a neutral (or radical) loss, possibly the sum of several molecules (NH3 + CH4 = CH7N): no electron-count
    rule, hydrogens within hmax + 2, heteroatom ratios and element caps of the Golden Rules; carbon-free losses
    are H..H4 or small inorganics"""
    if f[C] == 0:
        heavy = 0
        for e in range(NE - 1):
            heavy += f[e]
        if heavy == 0:
            return f[H] >= 1 and f[H] <= 4
        return _small_inorganic(f, 0)
    if f[H] > hmax(f, 2):
        return False
    return _ratios_ok(f, caps_tab, _caps_row(m))


@njit(cache=True)
def ion_ok(f, m, caps_tab):
    """a fragment ion: odd- or even-electron, hydrogens within hmax + 2 (CH3NH3+ = CH6N), Golden Rules ratios,
    caps and element combinations; carbon-free only as a small inorganic ion"""
    if f[C] == 0:
        return _small_inorganic(f, 1)
    if f[H] > hmax(f, 2):
        return False
    if not _ratios_ok(f, caps_tab, _caps_row(m)):
        return False
    return _combos_ok(f)


@njit(cache=True)
def precursor_ok(f, m, caps_tab, even_electron):
    """a precursor ion formula inferred from its mass: an organic ion within the Golden Rules' common ranges
    (99.7 % of known compounds: H/C 0.2-3.1, N/C <= 1.3, O/C <= 1.2, P/C <= 0.3, S/C <= 0.8, F/C <= 1.5, Cl/C <= 0.8,
    Br/C <= 0.8, Si/C <= 0.5); even-electron (the usual ESI case: [M+H]+, [M+Na]+, [M-H]-, [M+Cl]- ...) means a
    half-integer DBE"""
    c = f[C]
    if c == 0 or 5 * f[H] < c or 10 * f[H] > 31 * c:
        return False
    if 10 * f[N] > 13 * c or 10 * f[O] > 12 * c or 10 * f[P] > 3 * c or 10 * f[S] > 8 * c or 10 * f[F] > 15 * c \
            or 10 * f[CL] > 8 * c or 10 * f[BR] > 8 * c or 10 * f[I] > 8 * c or 10 * f[SI] > 5 * c:
        return False
    if not _ratios_ok(f, caps_tab, _caps_row(m)):
        return False
    if not _combos_ok(f):
        return False
    d = dbe2(f)
    if d < -1:
        return False
    if even_electron and d % 2 == 0:
        return False
    return True
