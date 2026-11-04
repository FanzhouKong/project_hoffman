"""Tab-separated tables, written from compiled code.

write_tsv(df, path, float_format, lineterminator) writes the bytes of df.to_csv(path, sep="\\t", index=False,
float_format=float_format, lineterminator=lineterminator): to_csv formats every cell in Python, about a minute for
the 268 feature lists of a 268-file study. Floats are formatted like Python's '%.<p>g' / '%.<p>f' (p <= 8): the
digits come from float arithmetic, and a value that lies too close to a rounding tie for that to be certain is
formatted by Python. text_formats ({column: '%.<p>f'}) first turns a column into that text, NaN included, as
features.precise_mz does. A table with anything else (dates, categories, text that needs quoting, another float
format) is written by to_csv."""
from __future__ import annotations

import csv
import io
import math
import os
import re

import numpy as np
import pandas as pd
from numba import njit, prange

G, F, INT, BOOL, TEXT = 0, 1, 2, 3, 4          # column kinds
UNSURE = -2                                     # a float cell Python formats
NA_EMPTY, NA_TEXT = 0, 1                        # NaN as "" (to_csv's na_rep) or as "nan" (text_formats)
BLOCK = 256                                     # rows per parallel task
_POW10 = np.array([10.0 ** k for k in range(23)])   # exact
_FMT = re.compile(r"%\.([0-8])([gf])$")


@njit(cache=True)
def _pow10(k):
    if 0 <= k <= 22:
        return _POW10[k]
    return 10.0 ** k


@njit(cache=True)
def _scaled(ax, k):
    """ax x 10^k; the power is exact up to 10^22, and dividing by it is exact-er than multiplying by 10^-k"""
    if k >= 0:
        return ax * _pow10(k)
    return ax / _pow10(-k)


@njit(cache=True)
def _int_digits(m, buf, n):
    """the decimal digits of m >= 0 at buf[n:], returns the new end"""
    k = 1
    t = m
    while t >= 10:
        t //= 10
        k += 1
    for j in range(n + k - 1, n - 1, -1):
        buf[j] = 48 + m % 10
        m //= 10
    return n + k


@njit(cache=True)
def _special(x, na, buf):
    """sign, NaN and inf: (end, done); done < 0 for a finite non-zero x left to format"""
    if x != x:
        if na == NA_TEXT:
            buf[0], buf[1], buf[2] = 110, 97, 110         # nan
            return 3, 1
        return 0, 1
    n = 0
    if x < 0.0 or (x == 0.0 and math.copysign(1.0, x) < 0.0):
        buf[0] = 45                                        # -
        n = 1
    if math.isinf(x):
        buf[n], buf[n + 1], buf[n + 2] = 105, 110, 102     # inf
        return n + 3, 1
    return n, -1


@njit(cache=True)
def fmt_g(x, p, na, buf):
    """'%.<p>g' % x (p in 1..8) into buf, returns its length; -1 when x is too near a rounding tie (or a range end)"""
    n, done = _special(x, na, buf)
    if done > 0:
        return n
    ax = abs(x)
    if ax == 0.0:
        buf[n] = 48
        return n + 1
    if ax < 1e-290 or ax > 1e290:
        return -1
    lo = _POW10[p - 1]
    hi = _POW10[p]
    e = int(math.floor(math.log10(ax)))
    s = _scaled(ax, p - 1 - e)
    if s < lo:
        e -= 1
        s = _scaled(ax, p - 1 - e)
    elif s >= hi:
        e += 1
        s = _scaled(ax, p - 1 - e)
    if s < lo or s >= hi:
        return -1
    m = math.floor(s)
    fr = s - m
    if abs(fr - 0.5) <= 1e-6:
        return -1
    mi = int(m) + (1 if fr > 0.5 else 0)
    if mi >= int(hi):                                      # rounded up to the next power of ten
        mi = int(lo)
        e += 1
    nd = p
    while nd > 1 and mi % 10 == 0:                         # %g drops trailing zeros
        mi //= 10
        nd -= 1
    if -4 <= e < p:
        if e < 0:                                          # 0.000ddd
            buf[n], buf[n + 1] = 48, 46
            n += 2
            for _ in range(-e - 1):
                buf[n] = 48
                n += 1
            return _int_digits(mi, buf, n)
        if nd <= e + 1:                                    # an integer: the digits, then zeros
            n = _int_digits(mi, buf, n)
            for _ in range(e + 1 - nd):
                buf[n] = 48
                n += 1
            return n
        end = _int_digits(mi, buf, n)                      # ddd.ddd: open a '.' after e + 1 digits
        cut = n + e + 1
        for j in range(end, cut, -1):
            buf[j] = buf[j - 1]
        buf[cut] = 46
        return end + 1
    end = _int_digits(mi, buf, n)                          # d.ddde+XX
    if nd > 1:
        for j in range(end, n + 1, -1):
            buf[j] = buf[j - 1]
        buf[n + 1] = 46
        end += 1
    buf[end] = 101
    buf[end + 1] = 45 if e < 0 else 43
    end += 2
    ae = abs(e)
    if ae < 10:
        buf[end] = 48
        end += 1
    return _int_digits(ae, buf, end)


@njit(cache=True)
def fmt_f(x, p, na, buf):
    """'%.<p>f' % x (p in 0..8) into buf, returns its length; -1 when x is too near a rounding tie (or too large)"""
    n, done = _special(x, na, buf)
    if done > 0:
        return n
    s = abs(x) * _POW10[p]
    if s >= 1e15:
        return -1
    m = math.floor(s)
    fr = s - m
    if abs(fr - 0.5) <= max(1e-6, s * 1e-15):
        return -1
    mi = int(m) + (1 if fr > 0.5 else 0)
    q = int(_POW10[p])
    n = _int_digits(mi // q, buf, n)
    if p > 0:
        buf[n] = 46
        r = mi % q
        for j in range(n + p, n, -1):
            buf[j] = 48 + r % 10
            r //= 10
        n += p + 1
    return n


@njit(cache=True)
def _fmt_int(v, buf):
    if v < 0:
        buf[0] = 45
        return _int_digits(-v, buf, 1)
    return _int_digits(v, buf, 0)


@njit(cache=True)
def _cell(kind, prec, na, j, c, i, Fl, In, so, ov, buf):
    """length of cell (i, j) formatted into buf; text cells are not copied (their bytes are in the text buffer)"""
    k = kind[j]
    if k == G or k == F:
        if ov[i, c] >= 0:
            return -3                                      # Python wrote it
        if k == G:
            return fmt_g(Fl[i, c], prec[j], na[j], buf)
        return fmt_f(Fl[i, c], prec[j], na[j], buf)
    if k == INT:
        return _fmt_int(In[i, c], buf)
    if k == BOOL:
        if In[i, c]:
            buf[0], buf[1], buf[2], buf[3] = 84, 114, 117, 101
            return 4
        buf[0], buf[1], buf[2], buf[3], buf[4] = 70, 97, 108, 115, 101
        return 5
    return so[c, i + 1] - so[c, i]


@njit(cache=True, parallel=True)
def _row_lengths(n, kind, col, prec, na, Fl, In, so, ov, n_eol):
    """bytes per row (tabs and line end included); float cells float arithmetic cannot settle are marked UNSURE in
    ov and left out of the length"""
    out = np.empty(n, np.int64)
    nc = len(kind)
    for b in prange((n + BLOCK - 1) // BLOCK):
        buf = np.empty(48, np.uint8)
        for i in range(b * BLOCK, min(n, (b + 1) * BLOCK)):
            t = nc - 1 + n_eol
            for j in range(nc):
                L = _cell(kind, prec, na, j, col[j], i, Fl, In, so, ov, buf)
                if L == -1:
                    ov[i, col[j]] = UNSURE
                else:
                    t += L
            out[i] = t
    return out


@njit(cache=True, parallel=True)
def _write_rows(n, kind, col, prec, na, Fl, In, so, sb, ov, oo, ob, start, eol, out):
    nc = len(kind)
    for b in prange((n + BLOCK - 1) // BLOCK):
        buf = np.empty(48, np.uint8)
        for i in range(b * BLOCK, min(n, (b + 1) * BLOCK)):
            p = start[i]
            for j in range(nc):
                if j:
                    out[p] = 9
                    p += 1
                c = col[j]
                L = _cell(kind, prec, na, j, c, i, Fl, In, so, ov, buf)
                if kind[j] == TEXT:
                    for q in range(so[c, i], so[c, i + 1]):
                        out[p] = sb[q]
                        p += 1
                elif L == -3:
                    o = ov[i, c]
                    for q in range(oo[o], oo[o + 1]):
                        out[p] = ob[q]
                        p += 1
                else:
                    for q in range(L):
                        out[p] = buf[q]
                        p += 1
            for q in range(len(eol)):
                out[p] = eol[q]
                p += 1


def _csr(strings: list[str]):
    """utf-8 bytes of a list of strings, and where each starts (n + 1 offsets)"""
    joined = "".join(strings)
    data = joined.encode("utf-8")
    if len(data) == len(joined):
        lens = np.fromiter(map(len, strings), np.int64, len(strings))
    else:
        lens = np.fromiter((len(s.encode("utf-8")) for s in strings), np.int64, len(strings))
    off = np.zeros(len(strings) + 1, np.int64)
    np.cumsum(lens, out=off[1:])
    return np.frombuffer(data, np.uint8), off


_QUOTE = re.compile(r'[\t"\r\n]')


def _text(s: pd.Series, float_format) -> list[str] | None:
    """the cells to_csv writes for a column it does not float-format, or None when one needs quoting"""
    v = s.to_numpy()
    if v.dtype.kind == "f":                                # to_csv with float_format None: numpy's str
        out = v.astype(str).astype(object)
        out[np.isnan(v)] = ""
        out = out.tolist()
    else:
        v = s.to_numpy(object)
        miss = pd.isna(v)
        out = ["" if m else (x if isinstance(x, str) else str(x)) for x, m in zip(v.tolist(), miss.tolist())]
    if _QUOTE.search("".join(out)):
        return None
    return out


def _encode(df: pd.DataFrame, float_format, lineterminator, text_formats):
    """(header bytes, row bytes) of the table, or None when to_csv must write it"""
    eol = lineterminator or os.linesep
    if df.shape[1] < 2 or eol not in ("\n", "\r\n") or not all(isinstance(c, str) for c in df.columns):
        return None
    ff = _FMT.match(float_format) if float_format else None
    if float_format and ff is None:
        return None
    text_formats = {c: f for c, f in (text_formats or {}).items() if c in df.columns}
    n, nc = df.shape
    kind = np.empty(nc, np.int8)
    col = np.empty(nc, np.int64)
    prec = np.zeros(nc, np.int64)
    na = np.zeros(nc, np.int8)
    floats, ints, texts = [], [], []
    for j in range(nc):
        s = df.iloc[:, j]
        name = df.columns[j]
        dt = s.dtype
        if name in text_formats:                           # features.precise_mz: '%.6f' text, NaN as "nan"
            tf = _FMT.match(text_formats[name])
            if tf is None or tf.group(2) != "f" or not (isinstance(dt, np.dtype) and dt.kind in "iuf"):
                return None
            kind[j], prec[j], na[j] = F, int(tf.group(1)), NA_TEXT
            col[j] = len(floats)
            floats.append(s.to_numpy(np.float64))
        elif isinstance(dt, np.dtype) and dt.kind == "f" and ff is not None:
            kind[j], prec[j], na[j] = (G if ff.group(2) == "g" else F), int(ff.group(1)), NA_EMPTY
            if kind[j] == G and prec[j] == 0:
                prec[j] = 1                                # %.0g is %.1g
            col[j] = len(floats)
            floats.append(s.to_numpy(np.float64))
        elif isinstance(dt, np.dtype) and dt.kind in "iub":
            v = s.to_numpy()
            if dt.kind == "u" and len(v) and v.max() > np.iinfo(np.int64).max:
                return None
            v = v.astype(np.int64)
            if len(v) and v.min() == np.iinfo(np.int64).min:
                return None
            kind[j] = BOOL if dt.kind == "b" else INT
            col[j] = len(ints)
            ints.append(v)
        elif (isinstance(dt, np.dtype) and dt.kind in "fO") or isinstance(dt, pd.StringDtype):
            t = _text(s, float_format)
            if t is None:
                return None
            kind[j] = TEXT
            col[j] = len(texts)
            texts.append(t)
        else:
            return None
    Fl = np.empty((n, len(floats)))
    for c, v in enumerate(floats):
        Fl[:, c] = v
    In = np.empty((n, len(ints)), np.int64)
    for c, v in enumerate(ints):
        In[:, c] = v
    so = np.zeros((len(texts), n + 1), np.int64)
    parts, base = [], 0
    for c, t in enumerate(texts):
        data, off = _csr(t)
        so[c] = off + base
        base += len(data)
        parts.append(data)
    sb = np.concatenate(parts) if parts else np.zeros(0, np.uint8)
    ov = np.full((n, len(floats)), -1, np.int32)
    eol_b = np.frombuffer(eol.encode(), np.uint8)
    lens = _row_lengths(n, kind, col, prec, na, Fl, In, so, ov, len(eol_b))
    # the cells float arithmetic could not settle: Python's own formatting
    ii, cc = np.nonzero(ov == UNSURE)
    fcol = np.zeros(len(floats), np.int64)
    for j in range(nc):
        if kind[j] in (G, F):
            fcol[col[j]] = j
    pieces = []
    for o, (i, c) in enumerate(zip(ii.tolist(), cc.tolist())):
        j = fcol[c]
        x = float(Fl[i, c])
        t = "" if (x != x and na[j] == NA_EMPTY) else (f"%.{prec[j]}{'g' if kind[j] == G else 'f'}" % x)
        pieces.append(t)
        ov[i, c] = o
        lens[i] += len(t)
    ob, oo = _csr(pieces)
    start = np.zeros(n + 1, np.int64)
    np.cumsum(lens, out=start[1:])
    out = np.empty(int(start[-1]), np.uint8)
    _write_rows(n, kind, col, prec, na, Fl, In, so, sb, ov, oo, ob, start, eol_b, out)
    head = io.StringIO()
    csv.writer(head, delimiter="\t", lineterminator=eol, quoting=csv.QUOTE_MINIMAL).writerow(list(df.columns))
    return head.getvalue().encode("utf-8"), out


def to_bytes(df: pd.DataFrame, float_format: str | None = None, lineterminator: str | None = None,
             text_formats: dict | None = None) -> bytes | None:
    """the bytes write_tsv writes, or None when to_csv must write this table"""
    enc = _encode(df, float_format, lineterminator, text_formats)
    return None if enc is None else enc[0] + enc[1].tobytes()


def write_tsv(df: pd.DataFrame, path, float_format: str | None = None, lineterminator: str | None = None,
              text_formats: dict | None = None) -> None:
    """df.to_csv(path, sep="\\t", index=False, float_format=..., lineterminator=...) after text_formats' columns are
    turned into text (fmt % value, for every value), byte for byte"""
    enc = _encode(df, float_format, lineterminator, text_formats)
    if enc is None:
        if text_formats:
            df = df.copy()
            for c, f in text_formats.items():
                if c in df.columns:
                    df[c] = np.char.mod(f, df[c].to_numpy().astype(np.float64))
        df.to_csv(path, sep="\t", index=False, float_format=float_format, lineterminator=lineterminator)
        return
    with open(path, "wb") as fh:
        fh.write(enc[0])
        fh.write(memoryview(enc[1]))


def warm() -> None:
    """compile (or load) the kernels, before worker processes start"""
    to_bytes(pd.DataFrame({"a": [1.5, np.nan], "b": [1, 2], "c": [True, False], "d": ["x", None], "e": [0.1, 2.0]}),
             "%.6g", "\n", {"e": "%.6f"})
