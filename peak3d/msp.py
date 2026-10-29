"""Compiled .msp reader: one pass over the file's bytes; library.read_msp builds the Library from its output.

Record rules: lines are stripped (LF, CRLF or CR endings); a line starting with a digit (or "." and a digit) is a
peak line, any other line holding ":" is "key: value" (keys case- and punctuation-insensitive, first occurrence
wins, unknown keys ignored). A record ends at a blank line after peak lines, at a NAME line once the record has
peaks or a name, and at the end of the file. A peak line is cut at its first double quote (annotations) and split at
";"; each piece gives one (m/z, intensity) pair from its first two numbers (separated by blanks or commas), or, when
it holds ":", pairs from its numbers in order ("m/z:intensity m/z:intensity", MassBank style). Non-finite and
non-positive peaks are dropped, a record's peaks are sorted by m/z, and records left without peaks or marked MS1 are
skipped and counted. The ion mode comes from the ion mode field (else Ionization), and when that says nothing, from
the charge sign of the precursor type ("[M+H]+" positive, "[M-H]-" negative)."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

BLOCK = 256 << 20          # bytes read at a time: memory stays at a block plus the parsed output, whatever the file size

FIELDS = ("name", "prec", "ptype", "mode", "mode_alt", "formula", "inchikey", "smiles", "comment", "stype", "npeaks")
NAME, PREC, PTYPE, MODE, MODE_ALT, FORMULA, INCHIKEY, SMILES, COMMENT, STYPE, NPEAKS = range(len(FIELDS))
N_FIELDS = len(FIELDS)
TEXT = (NAME, PTYPE, FORMULA, INCHIKEY, SMILES, COMMENT)      # fields kept as text, in this order
KEYS = {"name": NAME, "precursormz": PREC, "precursor": PREC, "precursortype": PTYPE, "adduct": PTYPE,
        "ionmode": MODE, "polarity": MODE, "ionization": MODE_ALT, "formula": FORMULA, "inchikey": INCHIKEY,
        "smiles": SMILES, "comment": COMMENT, "comments": COMMENT, "spectrumtype": STYPE, "mslevel": STYPE,
        "numpeaks": NPEAKS}
_KEY_BYTES = np.frombuffer("".join(KEYS).encode(), np.uint8).copy()
_KEY_OFF = np.cumsum([0] + [len(k) for k in KEYS]).astype(np.int64)
_KEY_FIELD = np.array(list(KEYS.values()), np.int64)
_TEXT = np.array(TEXT, np.int64)
_POW10 = np.array([float(10 ** k) for k in range(23)])        # exact: a short mantissa / 10^k rounds once
_POSITIVE = np.frombuffer(b"positive", np.uint8).copy()
_NEGATIVE = np.frombuffer(b"negative", np.uint8).copy()


# the per-line helpers are inlined: a call that passes arrays costs reference counting on every line
@njit(cache=True, inline="always")
def _blank(c):
    return c == 32 or (c >= 9 and c <= 13)


@njit(cache=True, inline="always")
def _sep(c):
    return c == 32 or (c >= 9 and c <= 13) or c == 44


@njit(cache=True, inline="always")
def _lower(c):
    return np.int64(c) + 32 if c >= 65 and c <= 90 else np.int64(c)


@njit(cache=True, inline="always")
def _number(buf, a, b, pow10):
    """(value, ok) of the decimal number buf[a:b]: [sign] digits [. digits] [e [sign] digits] and nothing else"""
    if a >= b:
        return np.nan, False
    neg = buf[a] == 45
    if buf[a] == 45 or buf[a] == 43:
        a += 1
    mant = np.int64(0)
    scale = 0
    digits = False
    i = a
    while i < b and buf[i] >= 48 and buf[i] <= 57:
        if mant < 100000000000000000:
            mant = mant * 10 + (np.int64(buf[i]) - 48)
        else:
            scale += 1
        digits = True
        i += 1
    if i < b and buf[i] == 46:
        i += 1
        while i < b and buf[i] >= 48 and buf[i] <= 57:
            if mant < 100000000000000000:
                mant = mant * 10 + (np.int64(buf[i]) - 48)
                scale -= 1
            digits = True
            i += 1
    if not digits:
        return np.nan, False
    if i < b and (buf[i] == 101 or buf[i] == 69):
        i += 1
        eneg = False
        if i < b and (buf[i] == 45 or buf[i] == 43):
            eneg = buf[i] == 45
            i += 1
        if i >= b or buf[i] < 48 or buf[i] > 57:
            return np.nan, False
        e = 0
        while i < b and buf[i] >= 48 and buf[i] <= 57:
            if e < 100000:
                e = e * 10 + (np.int64(buf[i]) - 48)
            i += 1
        scale += -e if eneg else e
    if i != b:
        return np.nan, False
    v = float(mant)
    if scale < 0 and scale >= -22:
        v = v / pow10[-scale]
    elif scale > 0 and scale <= 22:
        v = v * pow10[scale]
    elif scale != 0:
        v = v * 10.0 ** scale
    return (-v if neg else v), True


@njit(cache=True, inline="always")
def _key(buf, a, b, scratch, kb, ko, kf):
    """field slot of the key buf[a:b] compared on its letters and digits in lower case; -1 when unknown"""
    n = 0
    for i in range(a, b):
        c = _lower(buf[i])
        if not ((c >= 97 and c <= 122) or (c >= 48 and c <= 57)):
            continue
        if n == len(scratch):
            return -1
        scratch[n] = c
        n += 1
    for k in range(len(kf)):
        if ko[k + 1] - ko[k] != n:
            continue
        same = True
        for j in range(n):
            if kb[ko[k] + j] != scratch[j]:
                same = False
                break
        if same:
            return kf[k]
    return -1


@njit(cache=True, inline="always")
def _contains(buf, a, b, word):
    n = len(word)
    for i in range(a, b - n + 1):
        same = True
        for j in range(n):
            if _lower(buf[i + j]) != word[j]:
                same = False
                break
        if same:
            return True
    return False


@njit(cache=True, inline="always")
def _mode(buf, a, b, pos_word, neg_word):
    """1 positive, 2 negative, 0 unknown: starts with p / is "+" / says positive, likewise for negative"""
    if a >= b:
        return 0
    c = _lower(buf[a])
    if c == 112 or (b - a == 1 and c == 43) or _contains(buf, a, b, pos_word):
        return 1
    if c == 110 or (b - a == 1 and c == 45) or _contains(buf, a, b, neg_word):
        return 2
    return 0


@njit(cache=True, inline="always")
def _charge_sign(buf, a, b):
    """1 / 2 from the sign after the closing bracket of a precursor type ("[M+H]+", "[M+2H]2+", "[M-H]-"), or from
    its last character when it has no brackets; 0 when there is none"""
    k = b - 1
    while k >= a and buf[k] != 93:
        k -= 1
    if k >= a:
        for i in range(k + 1, b):
            if buf[i] == 43:
                return 1
            if buf[i] == 45:
                return 2
        return 0
    if b > a and buf[b - 1] == 43:
        return 1
    if b > a and buf[b - 1] == 45:
        return 2
    return 0


# scanner state between calls (an int64 array): the loop below writes only into arrays it is given and returns when
# one of them could overflow on the next line; scan() enlarges that one and calls again from the same line. Arrays
# replaced inside a compiled loop cost reference counting on every pass (measured: ~2x slower per line).
_POS, _NREC, _NBLOB, _NPK, _REC_P0, _N_SET, _N_PEAK_LINES, _SKIPPED, _REC_BYTE0, _NEED, _NEED_PEAKS, _NEED_BLOB = range(12)
_N_STATE = 12


@njit(cache=True, nogil=True)
def _scan_some(buf, st, cur_set, cur_a, cur_b, scratch, blob, soff, prec, mode, poff, pmz, pit, keep_comment,
               kb, ko, kf, text, pow10, pos_word, neg_word):
    """parse from line st[_POS] on; returns 0 at the end of buf, 1 when an output array must grow first
    (st[_NEED] bits: 1 peaks, 2 records, 4 text)"""
    n = len(buf)
    nt = len(text)
    pos, nrec, nblob, npk = st[_POS], st[_NREC], st[_NBLOB], st[_NPK]
    rec_p0, n_set, n_peak_lines, skipped, rec_byte0 = st[_REC_P0], st[_N_SET], st[_N_PEAK_LINES], st[_SKIPPED], \
        st[_REC_BYTE0]
    status = 0
    while True:
        at_end = pos >= n
        line0 = pos
        a = b = nxt = n
        if not at_end:
            j = pos
            while j < n and buf[j] != 10 and buf[j] != 13:
                j += 1
            a, b = pos, j
            nxt = j + 1
            if j < n and buf[j] == 13 and j + 1 < n and buf[j + 1] == 10:
                nxt = j + 2
        # room for whatever this line can add: a flush of the record so far, and the line's own peaks
        need = 0
        if nrec + 2 > len(prec) or (nrec + 1) * nt + 1 > len(soff):
            need |= 2
        if nblob + (line0 - rec_byte0) > len(blob):
            need |= 4
        if npk + (b - a) // 2 + 1 > len(pmz):
            need |= 1
        if need:
            st[_NEED], st[_NEED_PEAKS], st[_NEED_BLOB] = need, npk + (b - a) // 2 + 1, nblob + (line0 - rec_byte0)
            status = 1
            break
        kind = 0                     # 0 blank, 1 peaks, 2 known key, 3 other
        fid = -1
        if not at_end:
            pos = nxt
            while a < b and _blank(buf[a]):
                a += 1
            while b > a and _blank(buf[b - 1]):
                b -= 1
            if a == b:
                kind = 0
            elif (buf[a] >= 48 and buf[a] <= 57) or (buf[a] == 46 and b - a > 1 and buf[a + 1] >= 48
                                                      and buf[a + 1] <= 57):
                kind = 1
            else:
                c = a
                while c < b and buf[c] != 58:
                    c += 1
                kind = 3
                if c < b:
                    fid = _key(buf, a, c, scratch, kb, ko, kf)
                    if fid >= 0:
                        kind = 2
                        a = c + 1                         # the value, stripped
                        while a < b and _blank(buf[a]):
                            a += 1
        flush = at_end or (kind == 0 and n_peak_lines > 0) or \
            (kind == 2 and fid == NAME and (n_peak_lines > 0 or cur_set[NAME]))
        if flush and (n_set > 0 or n_peak_lines > 0):
            ms1 = False
            if cur_set[STYPE]:
                sa, sb = cur_a[STYPE], cur_b[STYPE]
                if sb - sa == 1 and buf[sa] == 49:
                    ms1 = True
                elif sb - sa == 3 and _lower(buf[sa]) == 109 and _lower(buf[sa + 1]) == 115 and buf[sa + 2] == 49:
                    ms1 = True
            if npk == rec_p0 or ms1:
                skipped += 1
                npk = rec_p0
            else:
                for q in range(rec_p0 + 1, npk):
                    if pmz[q] < pmz[q - 1]:
                        o = np.argsort(pmz[rec_p0:npk], kind="mergesort")
                        sm = pmz[rec_p0:npk][o]
                        si = pit[rec_p0:npk][o]
                        pmz[rec_p0:npk] = sm
                        pit[rec_p0:npk] = si
                        break
                v = np.nan
                if cur_set[PREC]:
                    pa, pb = cur_a[PREC], cur_b[PREC]
                    q = pa
                    while q < pb and not _blank(buf[q]):
                        q += 1
                    x, ok = _number(buf, pa, q, pow10)
                    if ok:
                        v = x
                prec[nrec] = v
                md = 0
                if cur_set[MODE] and cur_b[MODE] > cur_a[MODE]:
                    md = _mode(buf, cur_a[MODE], cur_b[MODE], pos_word, neg_word)
                elif cur_set[MODE_ALT]:
                    md = _mode(buf, cur_a[MODE_ALT], cur_b[MODE_ALT], pos_word, neg_word)
                if md == 0 and cur_set[PTYPE]:
                    md = _charge_sign(buf, cur_a[PTYPE], cur_b[PTYPE])
                mode[nrec] = md
                for k in range(nt):
                    f = text[k]
                    if cur_set[f] and (f != COMMENT or keep_comment):
                        for q in range(cur_a[f], cur_b[f]):
                            blob[nblob] = buf[q]
                            nblob += 1
                    soff[nrec * nt + k + 1] = nblob
                poff[nrec + 1] = npk
                nrec += 1
            rec_p0 = npk
            rec_byte0 = line0
            cur_set[:] = False
            n_set = 0
            n_peak_lines = 0
        if at_end:
            break
        if kind == 1:
            n_peak_lines += 1
            e = a
            while e < b and buf[e] != 34:
                e += 1
            s = a
            while s < e:
                t = s
                colon = False
                while t < e and buf[t] != 59:
                    if buf[t] == 58:
                        colon = True
                    t += 1
                p = s
                got = 0
                x0 = 0.0
                while p < t and (colon or got < 2):
                    while p < t and (_sep(buf[p]) or buf[p] == 58):
                        p += 1
                    if p >= t:
                        break
                    q = p
                    while q < t and not (_sep(buf[q]) or buf[q] == 58):
                        q += 1
                    x, ok = _number(buf, p, q, pow10)
                    if not ok:
                        break
                    p = q
                    if got % 2 == 0:
                        x0 = x
                    elif np.isfinite(x0) and np.isfinite(x) and x > 0:
                        pmz[npk] = x0
                        pit[npk] = x
                        npk += 1
                    got += 1
                s = t + 1
        elif kind == 2:
            if not cur_set[fid]:
                cur_set[fid] = True
                cur_a[fid] = a
                cur_b[fid] = b
                n_set += 1
    st[_POS], st[_NREC], st[_NBLOB], st[_NPK] = pos, nrec, nblob, npk
    st[_REC_P0], st[_N_SET], st[_N_PEAK_LINES], st[_SKIPPED], st[_REC_BYTE0] = rec_p0, n_set, n_peak_lines, skipped, \
        rec_byte0
    return status


def _bigger(a, need):
    out = np.empty(max(need, 2 * len(a)), a.dtype)
    out[:len(a)] = a
    return out


def scan(buf, keep_comment: bool = True):
    """parse .msp bytes (a uint8 array): (text, text_off, precursor_mz, mode, off, mz, intensity, n_skipped) where
    record i's text field k (TEXT order) is text[text_off[i * len(TEXT) + k]:text_off[i * len(TEXT) + k + 1]] and its
    peaks are mz/intensity[off[i]:off[i + 1]]; mode 0 unknown, 1 positive, 2 negative"""
    n, nt = len(buf), len(TEXT)
    recs = n // 2048 + 1024                      # first guesses from the file size; grown when short
    pmz, pit = np.empty(n // 128 + 4096), np.empty(n // 128 + 4096, np.float32)
    prec, mode, poff = np.empty(recs), np.empty(recs, np.int8), np.zeros(recs, np.int64)
    soff = np.zeros(recs * nt + 1, np.int64)
    blob = np.empty(n // 64 + (1 << 16), np.uint8)
    st = np.zeros(_N_STATE, np.int64)
    cur_set, cur_a, cur_b = np.zeros(N_FIELDS, np.bool_), np.zeros(N_FIELDS, np.int64), np.zeros(N_FIELDS, np.int64)
    scratch = np.empty(max(len(k) for k in KEYS), np.int64)
    while _scan_some(buf, st, cur_set, cur_a, cur_b, scratch, blob, soff, prec, mode, poff, pmz, pit,
                     bool(keep_comment), _KEY_BYTES, _KEY_OFF, _KEY_FIELD, _TEXT, _POW10, _POSITIVE, _NEGATIVE):
        need = st[_NEED]
        if need & 1:
            pmz, pit = _bigger(pmz, st[_NEED_PEAKS]), _bigger(pit, st[_NEED_PEAKS])
        if need & 2:
            prec, mode, poff = _bigger(prec, 0), _bigger(mode, 0), _bigger(poff, 0)
            soff = _bigger(soff, len(prec) * nt + 1)
        if need & 4:
            blob = _bigger(blob, st[_NEED_BLOB])
    nrec, npk = st[_NREC], st[_NPK]
    return (blob[:st[_NBLOB]].copy(), soff[:nrec * nt + 1].copy(), prec[:nrec].copy(), mode[:nrec].copy(),
            poff[:nrec + 1].copy(), pmz[:npk].copy(), pit[:npk].copy(), int(st[_SKIPPED]))


# ------------------------------------------------------------------ files: blocks, pieces in parallel ----
@njit(cache=True, nogil=True)
def _boundary_from(buf, start):
    """the first place at or after start where the reader is between records with nothing carried over: just after
    a blank line that follows a peak line (the reader always ends the record there); -1 when there is none"""
    n = len(buf)
    i = start
    while i < n and buf[i] != 10 and buf[i] != 13:
        i += 1
    if i < n and buf[i] == 13 and i + 1 < n and buf[i + 1] == 10:
        i += 2
    elif i < n:
        i += 1
    prev_peak = False
    while i < n:
        j = i
        while j < n and buf[j] != 10 and buf[j] != 13:
            j += 1
        nxt = j + 1
        if j < n and buf[j] == 13 and j + 1 < n and buf[j + 1] == 10:
            nxt = j + 2
        a, b = i, j
        while a < b and _blank(buf[a]):
            a += 1
        if a == b:
            if prev_peak:
                return min(nxt, n)
        else:
            prev_peak = (buf[a] >= 48 and buf[a] <= 57) or (buf[a] == 46 and b - a > 1 and buf[a + 1] >= 48
                                                             and buf[a + 1] <= 57)
        i = nxt
    return -1


def _pieces(buf, k: int):
    """buf cut at record boundaries into about k pieces of similar size"""
    cuts = [0]
    for t in range(1, k):
        c = _boundary_from(buf, max(len(buf) * t // k, cuts[-1]))
        if c < 0:
            break
        if c > cuts[-1]:
            cuts.append(c)
    cuts.append(len(buf))
    return [buf[a:b] for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def _last_boundary(buf) -> int:
    for start in (len(buf) - (1 << 20), len(buf) - (32 << 20), len(buf) // 2, 0):
        if start >= 0:
            c = _boundary_from(buf, start)
            if c > 0:
                return c
    return -1


def _merge(parts):
    nt = len(TEXT)
    blobs, soffs, precs, modes, poffs, mzs, its = [], [np.zeros(1, np.int64)], [], [], [np.zeros(1, np.int64)], [], []
    nb = npk = skipped = 0
    for blob, soff, prec, mode, poff, mz, it, sk in parts:
        blobs.append(blob)
        soffs.append(soff[1:] + nb)
        precs.append(prec)
        modes.append(mode)
        poffs.append(poff[1:] + npk)
        mzs.append(mz)
        its.append(it)
        nb += len(blob)
        npk += len(mz)
        skipped += sk
    cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt)   # noqa: E731
    return (cat(blobs, np.uint8), np.concatenate(soffs), cat(precs, np.float64), cat(modes, np.int8),
            np.concatenate(poffs), cat(mzs, np.float64), cat(its, np.float32), skipped)


def threads() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, min(n, 16))


def read(path, keep_comment: bool = True, n_threads: int | None = None, block: int = BLOCK):
    """scan() of a whole file, read in blocks cut at record boundaries, each block parsed in pieces on n_threads
    threads (the compiled loop releases the GIL); the same result as one scan of the whole file"""
    k = n_threads or threads()
    parts = []
    carry = np.zeros(0, np.uint8)
    with open(path, "rb") as fh, ThreadPoolExecutor(k) as pool:
        while True:
            buf = np.empty(len(carry) + block, np.uint8)
            buf[:len(carry)] = carry
            got = len(carry)
            while got < len(buf):
                r = fh.readinto(memoryview(buf)[got:])
                if not r:
                    break
                got += r
            final = got < len(buf)
            buf = buf[:got]
            cut = len(buf) if final else _last_boundary(buf)
            if cut <= 0 and not final:           # no record boundary in the whole block yet: read more first
                carry = buf
                continue
            parts.extend(pool.map(lambda piece: scan(piece, keep_comment), _pieces(buf[:cut], k)))
            carry = buf[cut:].copy()
            if final:
                break
    return _merge(parts)
