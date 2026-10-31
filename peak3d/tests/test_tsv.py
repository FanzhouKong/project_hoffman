"""tsv.write_tsv against DataFrame.to_csv: the same bytes"""
import numpy as np
import pandas as pd
import pytest

from peak3d import tsv
from peak3d.features import COLUMNS, MZ_COLUMNS, precise_mz, write_features

SPECIAL = [0.0, -0.0, np.nan, np.inf, -np.inf, 0.5, 2.5, 1.5, 0.125, 1e5, 1e6, 999999.5, 9999995.0, 1234565.0,
           99999.95, 0.0001, 0.00001, 0.000123456789, 123456789.0, 5e-324, 2.2250738585072014e-308,
           1.7976931348623157e308, 1e-300, 1e300, 1e22, 1e23, 0.1, 0.2, 0.3, 1 / 3, 2 / 3, 100.0, 12345.6785,
           -12345.6785, 0.0000005, -0.0000005, -1e-9, 1e-7, 4.35, 2.675, 1.005, 0.045, 1e15, 9.9999995, 0.99999949]


def _floats(rng, n):
    x = np.sign(rng.uniform(-1, 1, n)) * 10.0 ** rng.uniform(-320, 308, n)
    p = 10.0 ** rng.integers(-12, 16, len(SPECIAL) * 20)
    near = np.concatenate([p, np.nextafter(p, 0), np.nextafter(p, np.inf), p * (1 - 5e-7), p * (1 + 5e-7)])
    ties = rng.integers(1, 10 ** 7, 4000) * 10.0 ** rng.integers(-10, 4, 4000) + 5 * 10.0 ** rng.integers(-11, 3, 4000)
    halves = rng.integers(0, 10 ** 6, 4000) / 2.0 ** rng.integers(1, 12, 4000)
    v = np.concatenate([SPECIAL, x, near, ties, halves, rng.normal(0, 1e4, n), rng.uniform(0, 2000, n)])
    return v[rng.permutation(len(v))]


def _reference(df, float_format, lineterminator, text_formats, path):
    df = df.copy()
    for c, f in (text_formats or {}).items():
        if c in df.columns:
            df[c] = np.char.mod(f, df[c].to_numpy().astype(np.float64))
    df.to_csv(path, sep="\t", index=False, float_format=float_format, lineterminator=lineterminator)
    return path.read_bytes()


def _same(df, tmp_path, float_format=None, lineterminator=None, text_formats=None, compiled=True):
    ref = _reference(df, float_format, lineterminator, text_formats, tmp_path / "ref.tsv")
    if compiled:
        assert tsv.to_bytes(df, float_format, lineterminator, text_formats) is not None
    tsv.write_tsv(df, tmp_path / "new.tsv", float_format, lineterminator, text_formats)
    new = (tmp_path / "new.tsv").read_bytes()
    if new != ref:
        a, b = ref.decode().splitlines(), new.decode().splitlines()
        bad = next(i for i in range(min(len(a), len(b))) if a[i] != b[i]) if a != b else None
        pytest.fail(f"differs at line {bad}: {a[bad]!r} vs {b[bad]!r}" if bad is not None else "differs")


@pytest.mark.parametrize("fmt", ["%.6g", "%.4g", "%.8g", "%.1g", "%.0g", "%.6f", "%.2f", "%.0f", None])
def test_floats_match_to_csv(tmp_path, fmt):
    rng = np.random.default_rng(1)
    v = _floats(rng, 60000)
    with np.errstate(over="ignore"):                     # beyond float32: inf
        v32 = v.astype(np.float32)
    df = pd.DataFrame({"a": v, "b": v[::-1].copy(), "c32": v32, "i": np.arange(len(v)) - 7})
    _same(df, tmp_path, fmt, "\n")


def test_mixed_columns_and_line_ends(tmp_path):
    rng = np.random.default_rng(2)
    n = 5000
    words = np.array(["", "x", "flag;iso", "µ-ion", "é", "a b", "1.5"], dtype=object)
    obj = words[rng.integers(0, len(words), n)]
    obj[rng.random(n) < 0.1] = None
    obj[rng.random(n) < 0.1] = np.nan
    df = pd.DataFrame({"group_id": np.arange(n), "mz": rng.uniform(50, 1500, n), "rt": rng.uniform(0, 20, n),
                       "i8": rng.integers(-128, 127, n).astype(np.int8), "u16": rng.integers(0, 60000, n).astype(np.uint16),
                       "big": rng.integers(-2 ** 62, 2 ** 62, n), "ok": rng.random(n) < 0.5, "flags": obj,
                       "s": pd.array([None if (x is None or x != x) else x for x in obj], dtype="string"),
                       "h": rng.lognormal(10, 3, n).astype(np.float32)})
    df.loc[3, "mz"] = np.nan
    df.loc[4, "mz"] = -1e-9
    for eol in ("\n", "\r\n", None):
        _same(df, tmp_path, "%.6g", eol, {c: "%.6f" for c in MZ_COLUMNS})
        _same(df, tmp_path, None, eol, {"mz": "%.6f"})


def test_tables_to_csv_must_write(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x\ty", "z"]})
    assert tsv.to_bytes(df, "%.6g") is None              # text that needs quoting
    _same(df, tmp_path, "%.6g", compiled=False)
    one = pd.DataFrame({"a": [1.0, np.nan]})
    assert tsv.to_bytes(one, "%.6g") is None
    _same(one, tmp_path, "%.6g", compiled=False)
    other = pd.DataFrame({"a": [1.0, 2.0], "b": [3, 4]})
    assert tsv.to_bytes(other, "%.3e") is None           # a float format it does not write
    _same(other, tmp_path, "%.3e", compiled=False)
    dates = pd.DataFrame({"a": pd.to_datetime(["2026-10-30", "2026-10-31"]), "b": [1, 2]})
    _same(dates, tmp_path, "%.6g", compiled=False)


def test_empty_table(tmp_path):
    _same(pd.DataFrame({"a": np.zeros(0), "b": np.zeros(0, np.int64)}), tmp_path, "%.6g", "\n")


def test_write_features_as_before(tmp_path):
    rng = np.random.default_rng(3)
    n = 3000
    df = pd.DataFrame({c: rng.lognormal(5, 4, n) for c in COLUMNS})
    for c in ("feature_id", "scan_apex", "n_scans", "charge"):
        df[c] = rng.integers(0, 5000, n)
    df["flags"] = np.where(rng.random(n) < 0.3, "sat", "")
    df["group_id"] = rng.integers(-1, 900, n)
    df["group_kept"] = rng.random(n) < 0.8
    write_features(df, tmp_path / "new.tsv")
    precise_mz(df).to_csv(tmp_path / "old.tsv", sep="\t", index=False, float_format="%.6g", lineterminator="\n")
    assert (tmp_path / "new.tsv").read_bytes() == (tmp_path / "old.tsv").read_bytes()
