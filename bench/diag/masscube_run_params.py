# diagnostic: the parameters MassCube actually used in the benchmark runs (results/masscube/<run>/project_files/
# data_processing_metadata_*.pkl, written by untargeted_metabolomics_workflow), so a report to the authors can quote
# them instead of the source defaults. Needs masscube importable (unpickling its Params object).
# Writes results/diag_double/masscube_1.2.22_run_params.json and prints the instrument-dependent values per run.
# usage: masscube_run_params.py
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def plain(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) for x in v):
        return list(v)
    return None


out = {}
for d in sorted((ROOT / "results/masscube").glob("MC_*")):
    pk = sorted((d / "project_files").glob("data_processing_metadata_*.pkl"))
    if not pk:
        continue
    with open(pk[-1], "rb") as fh:
        meta = pickle.load(fh)            # a list of step records; "overview" holds versions, "parameters" the Params
    steps = {s.get("name"): s for s in meta if isinstance(s, dict)}
    keep = {k: plain(v) for k, v in steps["parameters"].items()
            if plain(v) is not None and not k.endswith("_dir") and k not in ("file_path", "name", "layer")}
    keep["_versions"] = steps.get("overview", {}).get("dependencies")
    out[d.name] = keep
    print(f"{d.name:42s} ms_type={keep.get('ms_type')} ion_mode={keep.get('ion_mode')} ms1_abs_int_tol={keep.get('ms1_abs_int_tol')} "
          f"ms2_abs_int_tol={keep.get('ms2_abs_int_tol')}")
(ROOT / "results/diag_double/masscube_1.2.22_run_params.json").write_text(json.dumps(out, indent=1, sort_keys=True))
first = next(iter(out.values()))
same = [k for k in first if all(v.get(k) == first[k] for v in out.values())]
print("\nidentical in every run:", json.dumps({k: first[k] for k in sorted(same)}, indent=0)[:4000])
