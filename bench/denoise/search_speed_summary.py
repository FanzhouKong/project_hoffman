"""Tables of search_speed.py's JSON lines.   python search_speed_summary.py speed.jsonl > speed.md"""
import json
import sys

import pandas as pd


def md(df):
    cols = list(df.columns)
    fmt = lambda v: "" if v is None or (isinstance(v, float) and v != v) else (f"{v:,}" if isinstance(v, int) else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.astype(object).itertuples(index=False)]
    return "\n".join(lines)


def main(path):
    rows = [json.loads(x) for x in open(path) if x.strip()]
    start = next(r for r in rows if r["step"] == "start")
    out = [f"# Denoising search speed\n\nnode {start['node']}, {start['threads']} threads; times in seconds.\n"]
    t0 = pd.DataFrame([r for r in rows if r["step"] == "t0_copies"])
    if len(t0):
        t0["spectra"] = t0["spectra"].map(lambda d: sum(d.values()))
        t0["ms_per_spectrum"] = (t0["seconds"] / t0["spectra"] * 1e3).round(1)
        out.append("## T0 top-5 copies (today's processing step, 8 worker processes)\n")
        out.append(md(t0[["run", "spectra", "seconds", "ms_per_spectrum"]]))
        out.append(f"\ntotal {t0['spectra'].sum()} spectra, {t0['seconds'].sum():.1f} s\n")
    s = pd.DataFrame([r for r in rows if r["step"] == "search"])
    for lib, g in s.groupby("library", sort=False):
        load = next(r for r in rows if r["step"] == "load" and r["library"] == lib)
        out.append(f"## {lib} ({load['n_spectra']:,} spectra, read {load['seconds']} s)\n")
        tab = []
        for run, h in g.groupby("run", sort=False):
            def get(mode, tag="", thr=start["threads"], col="seconds"):
                x = h[(h["mode"] == mode) & (h["tag"] == tag) & (h["threads"] == thr)]
                return x[col].iloc[0] if len(x) else None
            st = h[(h["mode"] == "search") & (h["tag"] == "warm") & (h["threads"] == start["threads"])]["stats"].iloc[0] or {}
            tab.append(dict(run=run, queries=get("off", col="n_queries"), off=get("off"), t0=get("t0"),
                            search_cold=get("search", "cold"), search_warm=get("search", "warm"),
                            search_1thread=get("search", "warm", 1), off_1thread=get("off", "", 1),
                            pairs=st.get("pairs"), keys=st.get("keys"),
                            pairs_per_key=round(st["pairs"] / st["keys"], 1) if st.get("keys") else None,
                            t1=st.get("seconds_t1"), ids_off=get("off", col="n_identified"),
                            ids_t0=get("t0", col="n_identified"), ids_search=get("search", "warm", col="n_identified")))
        t = pd.DataFrame(tab)
        out.append(md(t))
        tot = t[["off", "t0", "search_cold", "search_warm", "search_1thread", "off_1thread", "pairs", "keys", "t1"]].sum()
        out.append(f"\nsum over runs: off {tot['off']:.2f}, t0 {tot['t0']:.2f}, search cold {tot['search_cold']:.2f}, "
                   f"warm {tot['search_warm']:.2f}, one thread {tot['search_1thread']:.2f} (off one thread "
                   f"{tot['off_1thread']:.2f}); {int(tot['pairs']):,} pairs -> {int(tot['keys']):,} keys; T1 "
                   f"{tot['t1']:.2f} s\n")
    o = pd.DataFrame([r for r in rows if r["step"] == "t1_only"])
    if len(o):
        out.append("## T1 step alone: deduplicated keys vs one T1 per pair vs denoise() per pair in Python\n")
        g = o.groupby("library", sort=False)[["pairs_t1", "keys", "seconds_dedup", "seconds_per_pair_kernel",
                                              "naive_extrapolated"]].sum().reset_index()
        out.append(md(g))
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
