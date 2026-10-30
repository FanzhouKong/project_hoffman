"""Annotations: the library search's default, the user's choice, and what happens when the groups are rebuilt."""
import json

import numpy as np
import pandas as pd
import pytest

from peak3d import annotations as A
from peak3d.library import identify, write_msp_record

FRAG = np.array([60.04, 72.08, 86.10, 104.11])
FRAG_I = np.array([30.0, 100.0, 45.0, 10.0])


def _output(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame({"group_id": [1, 2], "mz": [132.1019, 250.0], "rt": [1.0, 2.0]}).to_csv(
        out / "aligned_feature_table.tsv", sep="\t", index=False)
    pd.DataFrame({"group_id": [1, 1, 2], "file": ["S0", "S1", "S0"], "scan": ["5", "6", "9"],
                  "representative": [1, 0, 1], "in_consensus": [1, 1, 1]}).to_csv(out / "ms2_map.tsv", sep="\t", index=False)
    for name, scale in (("ms2_spectra.msp", 1.0), ("ms2_consensus.msp", 1.0)):
        with open(out / name, "w") as fh:
            write_msp_record(fh, [("NAME", "feature_1"), ("PRECURSORMZ", "132.101900"), ("IONMODE", "Positive"),
                                  ("Comment", 'group_id=1 file="S0" scan="5" rt_s=60.000 n_ms2=2 n_merged=2')],
                             FRAG, FRAG_I * scale)
            write_msp_record(fh, [("NAME", "feature_2"), ("PRECURSORMZ", "250.000000"), ("IONMODE", "Positive"),
                                  ("Comment", 'group_id=2 file="S0" scan="9" rt_s=120.000 n_ms2=1')],
                             np.array([80.0, 120.0]), np.array([1.0, 1.0]))
    lib = tmp_path / "lib.msp"
    with open(lib, "w") as fh:
        write_msp_record(fh, [("NAME", "Leucine"), ("PRECURSORMZ", 132.1019), ("PRECURSORTYPE", "[M+H]+"),
                              ("IONMODE", "Positive")], FRAG, FRAG_I)
        write_msp_record(fh, [("NAME", "Isoleucine-ish"), ("PRECURSORMZ", 132.1020), ("IONMODE", "Positive")],
                         FRAG, np.array([100.0, 20.0, 10.0, 5.0]))
        write_msp_record(fh, [("NAME", "Weak"), ("PRECURSORMZ", 250.0), ("IONMODE", "Positive")],
                         np.array([80.0, 95.0, 130.0]), np.array([60.0, 100.0, 100.0]))
    return out, lib


def test_default_is_the_best_consensus_hit_above_min_score(tmp_path):
    out, lib = _output(tmp_path)
    hits = identify(out, lib, "entropy", min_score=0.7)
    assert set(hits["spectrum"]) == {"consensus", "single"} and (hits["score"] > 0.1).all()
    e = pd.read_csv(out / A.TABLE_FILE, sep="\t").set_index("group_id")
    assert e.loc[1, "annotation"] == "Leucine" and e.loc[1, "source"] == "library" and e.loc[1, "spectrum"] == "consensus"
    w = hits[(hits["group_id"] == 2) & (hits["spectrum"] == "consensus")]
    assert len(w) == 1 and w["score"].iloc[0] < 0.7                      # listed, but too weak to annotate
    assert pd.isna(e.loc[2, "annotation"]) and e.loc[2, "best_hit"] == "Weak"


def test_user_choice_unknown_and_reset(tmp_path):
    out, lib = _output(tmp_path)
    identify(out, lib, "entropy", min_score=0.7)
    with pytest.raises(KeyError):
        A.set_user(out, 1, {"spectrum": "consensus", "rank": 7})
    entry = A.set_user(out, 1, {"spectrum": "single", "rank": 2})
    assert entry["name"] == "Isoleucine-ish" and entry["status"] == "user" and entry["library"].endswith("lib.msp")
    e = A.effective(out).set_index("group_id")
    assert e.loc[1, "annotation"] == "Isoleucine-ish" and e.loc[1, "source"] == "user" and e.loc[1, "best_hit"] == "Leucine"
    A.set_user(out, 2, {"spectrum": "consensus", "rank": 1})             # a hit below min_score is a valid choice
    assert A.effective(out).set_index("group_id").loc[2, "annotation"] == "Weak"
    identify(out, lib, "dot", min_score=0.7)                               # a new search keeps the user's choices
    e = A.effective(out).set_index("group_id")
    assert e.loc[1, "annotation"] == "Isoleucine-ish" and e.loc[2, "source"] == "user"
    A.set_user(out, 1, "unknown")
    assert A.effective(out).set_index("group_id").loc[1, "annotation"] == "unknown"
    A.set_user(out, 1, None)
    A.set_user(out, 2, None)
    assert not (out / A.USER_FILE).exists()
    assert A.effective(out).set_index("group_id").loc[1, "source"] == "library"


def test_rebuilt_groups_retire_annotations_and_the_old_search(tmp_path):
    out, lib = _output(tmp_path)
    identify(out, lib, "entropy", min_score=0.7)
    A.set_user(out, 1, "unknown")
    A.retire(out, keep_search=False)
    assert not (out / A.USER_FILE).exists() and len(list(out.glob("user_annotations.*.json"))) == 1
    assert not any((out / n).exists() for n in A.SEARCH_FILES)
    saved = json.loads(next(out.glob("user_annotations.*.json")).read_text())
    assert saved["annotations"]["1"]["status"] == "unknown"
