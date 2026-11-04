"""Molecular formula of a SMILES string, without a cheminformatics toolkit.

Library records that give a structure but no formula (most of MoNA) still need their precursor ion formula for the
denoising search. smiles_formula reads the atoms, bonds, branches and ring closures of an (Open)SMILES string and
adds the implicit hydrogens:
  bracket atoms         [13CH3+], [nH], [O-], [C@@H]: exactly the hydrogens written, the charge written
  aliphatic organic     B C N O P S F Cl Br I: the smallest normal valence (B 3, C 4, N 3/5, O 2, P 3/5, S 2/4/6,
                        halogens 1) at or above the bond order sum, minus that sum
  aromatic organic      b c n o p s: the lowest normal valence minus the bond count minus one (the shared pi bond),
                        never below zero: c1ccccc1 C6H6, c1ccsc1 C4H4S, n1ccccc1 C5H5N; a pyrrole-type nitrogen must
                        be written [nH], as SMILES requires
It refuses rather than guesses: any syntax it does not know, an unclosed ring or branch, a wildcard (*), an isotope
label, or an element outside formula.ELEMENTS gives None. A ChemAxon CXSMILES extension (" |&1:4,r|") is dropped;
any other text after a blank is refused. Callers check the result against the record's precursor
m/z (search.ion_formula), so a wrong hydrogen count cannot pass silently: one H is 1.008 Da."""
from __future__ import annotations

import re

import numpy as np

from . import formula as fm

_TOKEN = re.compile(r"(Cl|Br|[BCNOPSFI])|(%\d\d|\d)|([()])|([-=#$:/\\.])|([bcnops])|(\[[^\[\]]*\])|(.)")
_BRACKET = re.compile(r"(\d+)?([A-Z][a-z]?|se|as|te|[bcnops])"
                      r"(@(?:@|TH[12]|AL[12]|SP[123]|TB(?:[1-9]|1\d|20)|OH(?:[1-9]|[12]\d|30))?)?"
                      r"(H\d*)?(\+\+?|--?|\+\d+|-\d+)?(:\d+)?")
_ORDER = {"-": 1, "/": 1, "\\": 1, ":": 1, "=": 2, "#": 3, "$": 4}
_VALENCE = {"B": (3,), "C": (4,), "N": (3, 5), "O": (2,), "P": (3, 5), "S": (2, 4, 6), "F": (1,), "Cl": (1,),
            "Br": (1,), "I": (1,)}
_AROMATIC = {"b": "B", "c": "C", "n": "N", "o": "O", "p": "P", "s": "S", "se": "Se", "as": "As", "te": "Te"}
_ARO_VALENCE = {"B": 3, "C": 4, "N": 3, "O": 2, "P": 3, "S": 2}
_CACHE: dict = {}


def _charge(text: str) -> int:
    if not text:
        return 0
    sign = 1 if text[0] == "+" else -1
    if len(text) > 1 and text[1:].isdigit():
        return sign * int(text[1:])
    return sign * len(text)


def _clean(smiles) -> str | None:
    s = str(smiles or "").strip()
    head, _, tail = s.partition(" ") if " " in s else s.partition("\t")
    if tail:
        if not (tail.strip().startswith("|") and tail.strip().endswith("|")):
            return None
        s = head
    if not s or "\t" in s:
        return None
    return s


def parse(smiles: str):
    """the atoms of one SMILES string, as parallel lists (element, kind: 0 bracket / 1 aliphatic organic / 2 aromatic
    organic, written H, charge, bond order sum, bond count, component); None when the string is not understood"""
    s = _clean(smiles)
    if s is None:
        return None
    el, kind, hw, chg, osum, nb, comp = [], [], [], [], [], [], []
    prev = -1
    bond = ""                  # pending bond symbol
    stack = []
    rings = {}                 # ring number -> (atom, bond symbol)
    bonded = set()             # atom pairs already joined (a ring closure may not repeat a bond)
    c = 0
    order = _ORDER
    for ali, ring, par, bsym, aro, br, bad in _TOKEN.findall(s):
        if ali or aro or br:
            if ali:
                el.append(ali); kind.append(1); hw.append(0); chg.append(0)
            elif aro:
                el.append(_AROMATIC[aro]); kind.append(2); hw.append(0); chg.append(0)
            else:
                b = _BRACKET.fullmatch(br[1:-1])
                if b is None or b.group(1) is not None:          # unknown syntax, or an isotope label
                    return None
                hs = b.group(4)
                el.append(_AROMATIC.get(b.group(2), b.group(2))); kind.append(0)
                hw.append(0 if not hs else (int(hs[1:]) if len(hs) > 1 else 1)); chg.append(_charge(b.group(5) or ""))
            k = len(el) - 1
            comp.append(c)
            if prev >= 0:
                o = order[bond] if bond else 1
                osum.append(o); nb.append(1)
                osum[prev] += o; nb[prev] += 1
                bonded.add((prev, k))
            else:
                if bond:
                    return None    # a bond with nothing before it
                osum.append(0); nb.append(0)
            prev, bond = k, ""
        elif ring:
            if prev < 0:
                return None
            n = int(ring[1:]) if ring[0] == "%" else int(ring)
            if n in rings:
                a, b0 = rings.pop(n)
                pair = (a, prev) if a < prev else (prev, a)
                if a == prev or pair in bonded or (b0 and bond and order[b0] != order[bond]):
                    return None
                bonded.add(pair)
                sym = bond or b0
                o = order[sym] if sym else 1
                osum[a] += o; nb[a] += 1
                osum[prev] += o; nb[prev] += 1
            else:
                rings[n] = (prev, bond)
            bond = ""
        elif par:
            if bond:
                return None
            if par == "(":
                if prev < 0:
                    return None
                stack.append(prev)
            else:
                if not stack:
                    return None
                prev = stack.pop()
        elif bsym:
            if bsym == ".":
                if bond or stack:
                    return None
                prev = -1
                c += 1
            else:
                if bond or prev < 0:
                    return None
                bond = bsym
        else:
            return None        # a character no token matches
    if stack or rings or bond or not el:
        return None
    return el, kind, hw, chg, osum, nb, comp


def _formulas(smiles):
    """((counts, charge) of the whole structure, (counts, charge) of its largest component or None), or None"""
    if smiles in _CACHE:
        return _CACHE[smiles]
    out = None
    atoms = parse(smiles)
    if atoms is not None:
        el, kind, hw, chg, osum, nb, comp = atoms
        idx = fm.IDX
        per = {}
        ok = True
        for i in range(len(el)):
            e = el[i]
            j = idx.get(e)
            if j is None:
                ok = False
                break
            k = kind[i]
            if k == 0:
                h = hw[i]
            elif k == 1:
                h = 0
                o = osum[i]
                for v in _VALENCE[e]:
                    if v >= o:
                        h = v - o
                        break
            else:
                h = _ARO_VALENCE[e] - nb[i] - 1
                if h < 0:
                    h = 0
            cnt = per.get(comp[i])
            if cnt is None:
                cnt = per[comp[i]] = [[0] * fm.NE, 0]
            cnt[0][j] += 1
            cnt[0][fm.H] += h
            cnt[1] += chg[i]
        if ok:
            whole = (np.array([sum(v[0][j] for v in per.values()) for j in range(fm.NE)], np.int64),
                     sum(v[1] for v in per.values()))
            if len(per) == 1:
                largest = whole
            else:
                heavy = {c: sum(v[0]) - v[0][fm.H] for c, v in per.items()}
                top = max(heavy.values())
                tied = [per[c] for c in per if heavy[c] == top]
                largest = ((np.array(tied[0][0], np.int64), tied[0][1])
                           if all(t[0] == tied[0][0] and t[1] == tied[0][1] for t in tied) else None)
            out = (whole, largest)
    if len(_CACHE) > 200_000:
        _CACHE.clear()
    _CACHE[smiles] = out
    return out


def smiles_formula(smiles: str, component: str = "all"):
    """(count vector over formula.ELEMENTS, net formal charge) of a SMILES string, or None (see the module docstring).
    component "all": every component of a dotted SMILES together; "largest": only the one with the most heavy atoms
    (a salt's organic ion); when several tie, their common formula if they share one (two tautomers or isomers of
    one compound), else None"""
    r = _formulas(smiles)
    if r is None:
        return None
    got = r[0] if component == "all" else r[1]
    return None if got is None else (got[0].copy(), got[1])
