"""Injections whose m/z calibration is off (data/truth/mz_test_injections.tsv; written by
results/mz_file_check_2026-10-30/write_registry.py from two tool-independent checks: internal standards and truth
peaks). They are used only to test m/z calibration: bench/runs.py leaves them out of the standard benchmark runs
(and lists them in MZTEST_RUNS) and bench/spiketruth.study_files leaves them out of the truth builders.
Standard library only: bench/runs.py is imported by scripts run with the system python."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/truth/mz_test_injections.tsv"


def registry() -> list[dict]:
    if not REGISTRY.exists():
        return []
    with REGISTRY.open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def mz_test_files(study: str | None = None) -> set[str]:
    """file stems in the registry, of one PR002464 study (e.g. "ST004502") or all"""
    return {r["file"] for r in registry() if study is None or r["study"] == study}
