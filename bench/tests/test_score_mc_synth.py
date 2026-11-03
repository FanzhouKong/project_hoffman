"""Double-peak hit rule of the MassCube synthetic benchmark (bench/score.py synth_found_both): a double signal is a hit
when both labelled peaks have a feature of their own; extra features at the m/z do not count against it, a merged
or missed pair does."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench import score as S  # noqa: E402

# one double signal per case (apexes 0.18 min apart, as in the truth table), plus a single signal (no apex2_rt)
MZS = [2000.0, 2000.1, 2000.2, 2000.3, 2000.4, 2000.5]
A1, A2 = 10.15, 10.33
T = pd.DataFrame({"mz": MZS + [3000.0], "apex1_rt": A1, "apex2_rt": [A2] * 6 + [np.nan], "n_peaks": [2] * 6 + [1]})


def feats(rows):
    return pd.DataFrame(rows, columns=["mz", "rt"])


def test_found_both_rule():
    f = feats([
        (2000.0, A1), (2000.0, A2),                       # both found -> hit
        (2000.1, A1), (2000.1, A2), (2000.1, A2 + 0.03),  # both found plus an extra feature -> still a hit
        (2000.2, (A1 + A2) / 2 - 0.01),                   # merged into one feature -> miss
        (2000.3, A1),                                     # only the larger peak -> miss
        (2000.4, A1 - 0.002), (2000.4, A1 + 0.004),       # two features, both on the first peak -> miss
        (2000.5 + 0.02, A1), (2000.5 + 0.02, A2),         # right RTs, wrong m/z (0.02 Da off) -> miss
        (3000.0, A1),                                     # single signal: the rule does not apply -> False
    ])
    assert S.synth_found_both(f, T).tolist() == [True, True, False, False, False, False, False]


def test_mc_rows_reports_found_next_to_exact_count():
    cnt = np.array([1, 2, 3, 1])
    n_peaks = np.array([1, 2, 2, 2])
    found = np.array([False, True, True, False])
    row = S.mc_rows("x", cnt, n_peaks, found)
    assert np.isclose(row["double_acc"], 1 / 3)       # exact count: only the 2-feature signal
    assert np.isclose(row["double_found"], 2 / 3)     # found rule: the 3-feature signal counts too
    assert np.isclose(row["avg_found"], (1 + 2 / 3) / 2)
    assert "double_found" not in S.mc_rows("x", cnt, n_peaks)
