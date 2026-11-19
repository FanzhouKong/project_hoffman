"""Retention index: the ladder (n-alkanes or FAMEs) and the map from retention time to index.

Kovats' index for temperature-programmed runs (van den Dool & Kratz 1963) interpolates linearly between the two
ladder compounds that bracket a peak; before the first and after the last one the nearest segment is extended and
the value is marked extrapolated. Two ladders are built in:
  alkane  n-alkane CnH2n+2, RI = 100 n
  fame    methyl esters of the saturated C8-C30 fatty acids with the FiehnLib index values (Kind et al. 2009)
Any other ladder is a table (read_ladder).

find_ladder picks the ladder out of a deconvolved run. A candidate is a component whose spectrum is dominated by
the series ions (alkane: C4H9+, C5H11+, C6H13+ at m/z 57.0699, 71.0855, 85.1012; FAME: 74.0362 and 87.0441). With
accurate mass the molecular ion names the carbon number of a candidate outright (CnH2n+2 at 14.01565 n + 2.01565);
candidates are then assigned to the expected carbon numbers in elution order, the labelled ones fixed, by the
order-preserving assignment with the largest summed log intensity. Without labels and with ladder members missing
the assignment is ambiguous, which the result says."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ELECTRON = 0.00054858
H, C12, O16 = 1.00782503207, 12.0, 15.99491461956
ALKANE_IONS = (4 * C12 + 9 * H - ELECTRON, 5 * C12 + 11 * H - ELECTRON, 6 * C12 + 13 * H - ELECTRON)
FAME_IONS = (3 * C12 + 6 * H + 2 * O16 - ELECTRON, 4 * C12 + 7 * H + 2 * O16 - ELECTRON)
FAME_RI = {8: 262320, 9: 323120, 10: 381020, 12: 487220, 14: 582620, 16: 668720, 18: 747420, 20: 819620,
           22: 886620, 24: 948820, 26: 1006900, 28: 1061700, 30: 1113100}
KINDS = ("alkane", "fame")
RI_TOL = {"alkane": 20.0, "fame": 3000.0}     # default search window per index scale
SERIES_SHARE = {"alkane": 0.25, "fame": 0.2}  # the series ions hold at least this share of a ladder compound's spectrum
MIN_MARKERS = 3


def molecular_mz(kind: str, n_c: int) -> float:
    if kind == "alkane":
        return C12 * n_c + H * (2 * n_c + 2) - ELECTRON
    return C12 * (n_c + 1) + H * (2 * n_c + 2) + 2 * O16 - ELECTRON


def ladder_ri(kind: str, n_c: int) -> float:
    if kind == "alkane":
        return 100.0 * n_c
    return float(FAME_RI.get(n_c, np.nan))


def default_carbons(kind: str) -> list[int]:
    return list(FAME_RI) if kind == "fame" else []


def parse_carbons(text: str) -> list[int]:
    """"10-40" or "8,9,10,12-30:2" (a:b:step) -> carbon numbers"""
    out = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        rng, _, step = part.partition(":")
        a, _, b = rng.partition("-")
        out += list(range(int(a), int(b or a) + 1, int(step or 1)))
    if not out or min(out) < 1:
        raise ValueError(f"not a list of carbon numbers: {text!r}")
    return sorted(set(out))


@dataclass
class RIMap:
    """retention time (minutes, ascending) -> retention index, piecewise linear"""
    rt: np.ndarray
    ri: np.ndarray

    def __post_init__(self):
        rt, ri = np.asarray(self.rt, np.float64), np.asarray(self.ri, np.float64)
        o = np.argsort(rt, kind="stable")
        self.rt, self.ri = rt[o], ri[o]
        if len(self.rt) < 2:
            raise ValueError("a retention index ladder needs at least two compounds")
        if np.any(np.diff(self.rt) <= 0) or np.any(np.diff(self.ri) <= 0):
            raise ValueError("ladder retention times and index values must both increase")

    def ri_at(self, t):
        """(index, extrapolated) for times t"""
        t = np.asarray(t, np.float64)
        k = np.clip(np.searchsorted(self.rt, t, side="right") - 1, 0, len(self.rt) - 2)
        slope = (self.ri[k + 1] - self.ri[k]) / (self.rt[k + 1] - self.rt[k])
        return self.ri[k] + slope * (t - self.rt[k]), (t < self.rt[0]) | (t > self.rt[-1])


def read_ladder(path, kind: str | None = None) -> pd.DataFrame:
    """A ladder table (tab- or comma-separated, header row): rt (minutes) or rt_s (seconds), and ri or carbon
    (carbon needs the kind). Returns columns carbon, ri, rt."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "rt" not in df and "rt_s" in df:
        df["rt"] = df["rt_s"] / 60.0
    if "rt" not in df:
        raise ValueError(f"{path}: no rt (minutes) or rt_s (seconds) column")
    if "ri" not in df:
        if "carbon" not in df or kind not in KINDS:
            raise ValueError(f"{path}: needs an ri column, or a carbon column and the ladder kind")
        df["ri"] = [ladder_ri(kind, int(c)) for c in df["carbon"]]
    if "carbon" not in df:
        df["carbon"] = -1
    df = df[["carbon", "ri", "rt"]].dropna(subset=["ri", "rt"]).sort_values("rt").reset_index(drop=True)
    RIMap(df["rt"].values, df["ri"].values)        # validates
    return df


def _abundance_at(mz, ab, target, ppm, da):
    tol = max(da, target * ppm * 1e-6)
    lo, hi = np.searchsorted(mz, target - tol), np.searchsorted(mz, target + tol, side="right")
    return float(ab[lo:hi].max()) if hi > lo else 0.0


def _unit(target: float) -> float:
    """the nominal mass an ion of this m/z is measured at on a unit-resolution instrument (n - 0.3 <= m < n + 0.7,
    the rule of library.nominal_mass)"""
    return float(np.floor(target + 0.3))


def candidates(rt, off, mz, ab, kind: str, carbons=None, ppm: float = 10.0, da: float = 0.003,
               unit: bool = False) -> pd.DataFrame:
    """Components that look like a ladder compound: component, rt, series (summed series-ion abundance), share (of
    the spectrum), carbon (from the molecular ion, -1 when none is found among `carbons` or 4..60). unit: the
    spectra hold nominal masses (a unit-resolution run): the ladder's ions are looked up at theirs, and a
    molecular ion counts only when it is stronger than the ion one mass below it (at unit resolution the 13C peak
    of the fragment C(n)H(2n+1) sits on the molecular mass of the alkane with n carbons)."""
    if kind not in KINDS:
        raise ValueError(f"ladder kind must be one of {KINDS}")
    ions = ALKANE_IONS if kind == "alkane" else FAME_IONS
    at = (lambda t: t) if not unit else _unit
    if unit:
        ions, ppm, da = [_unit(t) for t in ions], 0.0, 0.25
    search = list(carbons) if carbons else list(range(4, 61))
    rows = []
    for c in range(len(off) - 1):
        m, a = mz[off[c]:off[c + 1]], ab[off[c]:off[c + 1]]
        if len(m) == 0:
            continue
        s = [_abundance_at(m, a, t, ppm, da) for t in ions]
        if min(s) <= 0 or sum(s) < SERIES_SHARE[kind] * a.sum() or s[0] < s[-1]:
            continue
        best_n, best_a = -1, 0.0
        for n in search:
            x = _abundance_at(m, a, at(molecular_mz(kind, n)), ppm, da)
            if unit and x <= _abundance_at(m, a, at(molecular_mz(kind, n)) - 1.0, ppm, da):
                continue
            if x > best_a:
                best_n, best_a = n, x
        rows.append(dict(component=c, rt=float(rt[c]), series=float(sum(s)), share=float(sum(s) / a.sum()),
                         carbon=best_n, molecular=best_a))
    return pd.DataFrame(rows, columns=["component", "rt", "series", "share", "carbon", "molecular"])


def assign(cand: pd.DataFrame, carbons) -> np.ndarray:
    """Index into `carbons` for every candidate (-1: not in the ladder): candidates in elution order take ladder
    members in order, a candidate labelled by its molecular ion only its own carbon number, maximising the summed
    log10 series abundance (so with more candidates than ladder members the strongest stay)."""
    carbons = list(carbons)
    N, K = len(cand), len(carbons)
    out = np.full(N, -1, np.int64)
    if N == 0 or K == 0:
        return out
    o = np.argsort(cand["rt"].values, kind="stable")
    w = np.log10(np.maximum(cand["series"].values[o], 1.0)) + 1.0
    lab = cand["carbon"].values[o]
    best = np.zeros((N + 1, K + 1))
    take = np.zeros((N + 1, K + 1), np.int8)                    # 1 skip candidate, 2 skip ladder member, 3 match
    for i in range(1, N + 1):
        for j in range(1, K + 1):
            b, t = best[i - 1, j], 1
            if best[i, j - 1] > b:
                b, t = best[i, j - 1], 2
            if lab[i - 1] in (-1, carbons[j - 1]) and best[i - 1, j - 1] + w[i - 1] > b:
                b, t = best[i - 1, j - 1] + w[i - 1], 3
            best[i, j], take[i, j] = b, t
    i, j = N, K
    while i > 0 and j > 0:
        if take[i, j] == 3:
            out[o[i - 1]] = j - 1
            i, j = i - 1, j - 1
        elif take[i, j] == 2:
            j -= 1
        else:
            i -= 1
    return out


def find_ladder(rt, off, mz, ab, kind: str, carbons=None, ppm: float = 10.0, da: float = 0.003, unit: bool = False):
    """The ladder of one deconvolved run: (table, note). Table columns: carbon, ri, rt, component, series,
    molecular_ion (True where the molecular ion named the carbon number); empty when fewer than three members
    are found. carbons: the ladder's carbon numbers (default: the FiehnLib set for fame; for alkanes none, and
    then only candidates named by their molecular ion are used)."""
    carbons = list(carbons) if carbons else default_carbons(kind)
    cols = ["carbon", "ri", "rt", "component", "series", "molecular_ion"]
    cand = candidates(rt, off, mz, ab, kind, carbons or None, ppm, da, unit)
    if len(cand) == 0:
        return pd.DataFrame(columns=cols), f"no {kind} ladder compound found"
    if carbons:
        cand.loc[~cand["carbon"].isin(carbons), "carbon"] = -1
        idx = assign(cand, carbons)
        got = cand[idx >= 0].assign(carbon=[carbons[k] for k in idx[idx >= 0]], labelled=cand["carbon"].values[idx >= 0] >= 0)
    else:
        got = cand[cand["carbon"] >= 0].sort_values("series", ascending=False).drop_duplicates("carbon").assign(labelled=True)
    got = got.sort_values("rt")
    # the carbon numbers must rise with retention time; drop the weaker of two that do not
    while len(got) > 1 and np.any(np.diff(got["carbon"].values) <= 0):
        k = int(np.flatnonzero(np.diff(got["carbon"].values) <= 0)[0])
        drop = got.index[k] if got["series"].iloc[k] < got["series"].iloc[k + 1] else got.index[k + 1]
        got = got.drop(index=drop)
    table = pd.DataFrame(dict(carbon=got["carbon"].values.astype(np.int64),
                              ri=[ladder_ri(kind, int(n)) for n in got["carbon"]], rt=got["rt"].values,
                              component=got["component"].values.astype(np.int64), series=got["series"].values,
                              molecular_ion=got["labelled"].values.astype(bool)), columns=cols)
    table = table[np.isfinite(table["ri"].values)].reset_index(drop=True)
    if len(table) < MIN_MARKERS:
        return table.iloc[:0], f"only {len(table)} {kind} ladder compound(s) found"
    n_lab = int(table["molecular_ion"].sum())
    note = f"{len(table)} {kind} ladder compounds, {n_lab} named by their molecular ion"
    if carbons and len(table) < len(carbons) and n_lab < len(table):
        note += f"; {len(carbons) - len(table)} of the expected {len(carbons)} missing, so the carbon numbers of the " \
                "unnamed ones are not certain"
    return table, note
