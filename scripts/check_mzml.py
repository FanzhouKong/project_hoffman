#!/usr/bin/env python3
"""Report spectrum count and centroid/profile status per MS level for mzML files.

usage: check_mzml.py FILE.mzML [...]   -> TSV on stdout
Streams the file, so it is cheap on memory; reads the whole file once.
"""
import sys
import xml.etree.ElementTree as ET

CENTROID, PROFILE, MSLEVEL = "MS:1000127", "MS:1000128", "MS:1000511"


def check(path):
    counts = {}  # (ms_level, mode) -> n
    level = mode = None
    in_spec = False
    for ev, el in ET.iterparse(path, events=("start", "end")):
        tag = el.tag.rsplit("}", 1)[-1]
        if ev == "start" and tag == "spectrum":
            in_spec, level, mode = True, None, None
        elif ev == "end" and tag == "cvParam" and in_spec:
            acc = el.get("accession")
            if acc == MSLEVEL:
                level = el.get("value")
            elif acc == CENTROID:
                mode = "centroid"
            elif acc == PROFILE:
                mode = "profile"
        elif ev == "end" and tag == "spectrum":
            in_spec = False
            k = (level or "?", mode or "unknown")
            counts[k] = counts.get(k, 0) + 1
            el.clear()
        elif ev == "end" and tag in ("run", "spectrumList"):
            el.clear()
    return counts


if __name__ == "__main__":
    print("file\tn_spectra\tstatus\tdetail")
    for f in sys.argv[1:]:
        try:
            c = check(f)
        except ET.ParseError as e:
            print(f"{f}\t0\tERROR\t{e}")
            continue
        n = sum(c.values())
        modes = {m for (_, m) in c}
        status = ("centroid" if modes == {"centroid"} else
                  "profile" if modes == {"profile"} else "mixed/unknown")
        detail = ";".join(f"ms{l}:{m}={v}" for (l, m), v in sorted(c.items()))
        print(f"{f}\t{n}\t{status}\t{detail}")
