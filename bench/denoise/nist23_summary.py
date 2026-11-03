"""Summarise a tier-1 NIST23 pass: ion and intensity retention at several tolerances, by instrument type,
collision energy, precursor mass, and the mass-error distribution of explained ions.

usage: python bench/denoise/nist23_summary.py IN.npz RESULT.npz OUT.md [--iso]"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from specdenoise import formula as fm                 # noqa: E402
from specdenoise.losses import iso_rescue_batch      # noqa: E402

TOLS = [("2 mDa", 0.002, 0.0), ("3 mDa", 0.003, 0.0), ("5 mDa", 0.005, 0.0), ("10 mDa", 0.010, 0.0),
        ("20 mDa", 0.020, 0.0), ("5 ppm/2 mDa", 0.002, 5.0), ("10 ppm/2 mDa", 0.002, 10.0),
        ("10 ppm/3 mDa", 0.003, 10.0), ("20 ppm/3 mDa", 0.003, 20.0)]


def keep_mask(err, flag, mz, tol_abs, tol_ppm, slack=None):
    tol = np.maximum(tol_abs, tol_ppm * 1e-6 * mz)
    if slack is not None:
        tol = tol + slack
    return (flag == 1) | ((flag == 0) & (err <= tol))


def iso_pass(keep, reason, off, mz, it, tol_abs, tol_ppm, slack_spec, elem):
    allow = np.ones((len(off) - 1, 4), np.bool_)
    allow[:, 1:] = elem
    iso_rescue_batch(off.astype(np.int64), mz.astype(np.float64), it, keep, reason, tol_abs, tol_ppm, slack_spec, allow)


def spec_of(off, n):
    return np.repeat(np.arange(n), np.diff(off[: n + 1]))


def nce_bin(s):
    m = re.search(r"(\d+(?:\.\d+)?)", str(s))
    if not m:
        return "n/a"
    v = float(m.group(1))
    return "<=20" if v <= 20 else "21-40" if v <= 40 else "41-60" if v <= 60 else ">60"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("res"); ap.add_argument("out")
    ap.add_argument("--iso", action="store_true", help="apply the isotopologue rescue (slow-ish)")
    ap.add_argument("--exchange", action="store_true", help="count halide -> OH exchange products as explained")
    a = ap.parse_args()
    z = np.load(a.inp, allow_pickle=True); r = np.load(a.res)
    formulas = z["formula"]; names = z["name"]          # npz members are re-read on every access: load once
    n = int(r["n"]); off = z["peak_offsets"][: n + 1]; nion = int(off[-1])
    mz = z["peak_mz"][:nion]; it = z["peak_intensity"][:nion].astype(np.float64)
    err = r["err"]; err_x = r["err_x"]; flag = r["flag"]; status = r["status"]; elec = r["elec"]
    dec = r["decimals"]; slack = np.where(dec[spec_of(off, n)] <= 3, 0.5 * 10.0 ** -dec[spec_of(off, n)].astype(float), 0.0)
    if a.exchange:
        err = np.minimum(err, err_x)
    spec = np.repeat(np.arange(n), np.diff(off))
    ok_spec = status == 0
    ion_ok = ok_spec[spec]
    inst = z["instrument_type"][:n]; ce = np.array([nce_bin(s) for s in z["collision_energy"][:n]])
    pm = r["pmz_theo"]; mbin = np.array(["<200", "200-400", "400-700", ">=700"])[np.digitize(pm, [200, 400, 700])]
    # spectrum-level intensity totals (fragments + precursor; above-precursor ions count as removable)
    tot_i = np.bincount(spec, it, n)
    out = []
    out.append(f"# NIST23 tier-1 pass: {os.path.basename(a.inp)}\n")
    out.append(f"spectra {n}, usable (formula + adduct + precursor m/z agree) {ok_spec.sum()} "
               f"(status counts {np.bincount(status, minlength=5).tolist()}: ok, bad formula, unknown adduct, "
               f"m/z disagree, |z|>1); ions in usable spectra {ion_ok.sum()}\n")
    out.append(f"flags among usable ions: precursor {np.sum(ion_ok & (flag == 1))}, above precursor "
               f"{np.sum(ion_ok & (flag == 2))} ({100 * np.sum(it[ion_ok & (flag == 2)]) / it[ion_ok].sum():.3f} % of intensity), "
               f"electronic-rule members {np.sum(ion_ok & elec)} ({100 * np.sum(it[ion_ok & elec]) / it[ion_ok].sum():.2f} % of intensity)\n")
    out.append("## Retention vs tolerance (usable spectra; ions above the precursor count as removed)\n")
    out.append("| tolerance | ions kept | intensity kept | spectra with 100 % ions kept | spectra >= 99 % intensity | median spectrum intensity kept |")
    out.append("|---|---|---|---|---|---|")
    rows = {}
    elem = np.zeros((n, 3), bool)
    if a.iso:
        for i in np.flatnonzero(ok_spec):
            f = fm.parse(formulas[i])
            if f is not None:
                elem[i] = (f[fm.S] > 0, f[fm.CL] > 0, f[fm.BR] > 0)
    slack_spec = np.where(dec <= 3, 0.5 * 10.0 ** -dec.astype(float), 0.0)
    out.append(f"spectra on a 2-decimal m/z grid: {np.sum(ok_spec & (dec == 2))}, 3-decimal: {np.sum(ok_spec & (dec == 3))} "
               f"(tolerance rows marked +grid add half a grid step for those)\n")
    tols = TOLS + [(n_ + " +grid", ta, tp, True) for n_, ta, tp in TOLS if n_ in ("3 mDa", "5 mDa", "10 ppm/3 mDa")]
    for row_ in tols:
        name, ta, tp = row_[:3]
        keep = keep_mask(err, flag, mz, ta, tp, slack if len(row_) > 3 else None) & ion_ok
        if a.iso:
            reason = np.where(keep, 0, np.where(flag == 2, 4, 3)).astype(np.int8)
            iso_pass(keep, reason, off, mz, it, ta, tp, slack_spec if len(row_) > 3 else np.zeros(n), elem)
            keep &= ion_ok
        ki = np.bincount(spec, keep & ion_ok, n)[ok_spec]; ni = np.bincount(spec, ion_ok, n)[ok_spec]
        kint = np.bincount(spec, it * keep, n)[ok_spec]; tint = tot_i[ok_spec]
        frac = np.where(tint > 0, kint / np.maximum(tint, 1e-12), 1)
        rows[name] = keep
        out.append(f"| {name} | {100 * keep.sum() / ion_ok.sum():.3f} % | {100 * kint.sum() / tint.sum():.3f} % | "
                   f"{100 * np.mean(ki == ni):.2f} % | {100 * np.mean(frac >= 0.99):.2f} % | {100 * np.median(frac):.3f} % |")
    out.append("\n## By instrument type, collision energy and precursor mass (10 ppm / 3 mDa +grid)\n")
    keep = rows["10 ppm/3 mDa +grid"]
    for label, cat in (("instrument type", inst), ("NCE / CE", ce), ("precursor m/z", mbin)):
        out.append(f"| {label} | spectra | ions kept | intensity kept |")
        out.append("|---|---|---|---|")
        for v in pd.unique(cat[ok_spec]):
            m = ion_ok & (cat[spec] == v)
            out.append(f"| {v} | {np.sum(ok_spec & (cat == v))} | {100 * np.sum(keep & m) / m.sum():.3f} % | "
                       f"{100 * np.sum(it[keep & m]) / np.sum(it[m]):.3f} % |")
        out.append("")
    # mass-error distribution of explained ions (loose pass) by instrument type
    out.append("## Mass error of the best explanation (Da), explained fragment ions, by instrument type\n")
    out.append("| instrument type | n | p50 | p90 | p99 | p99.5 | p99.9 |")
    out.append("|---|---|---|---|---|---|---|")
    expl = ion_ok & (flag == 0) & np.isfinite(err)
    for v in pd.unique(inst[ok_spec]):
        e = err[expl & (inst[spec] == v)]
        q = np.quantile(e, [0.5, 0.9, 0.99, 0.995, 0.999]) if len(e) else [np.nan] * 5
        out.append(f"| {v} | {len(e)} | " + " | ".join(f"{x * 1000:.2f} mDa" for x in q) + " |")
    e_ppm = 1e6 * err[expl] / mz[expl]
    out.append(f"\nppm error of explained ions: p50 {np.median(e_ppm):.2f}, p90 {np.quantile(e_ppm, .9):.2f}, "
               f"p99 {np.quantile(e_ppm, .99):.2f}, p99.9 {np.quantile(e_ppm, .999):.2f}\n")
    # removed ions at 10 ppm / 3 mDa: how intense are they?
    rem = ion_ok & ~keep & (flag != 2)
    base = np.bincount(spec, it, n).max() if False else None
    relint = it / np.repeat(np.bincount(spec, it, n) / 1.0, np.diff(off)).clip(1e-12)
    out.append("## Why ions are removed at 10 ppm / 3 mDa +grid\n")
    tol = np.maximum(0.003, 10e-6 * mz) + slack
    strict = r["err"]
    cat = {"kept": keep & ion_ok,
           "removed: precursor isotopologue region (above precursor, not rescued)": ion_ok & ~keep & (flag == 2),
           "removed: explained only beyond tolerance, <= 10 mDa": ion_ok & ~keep & (flag == 0) & (err > tol) & (err <= 0.010),
           "removed: explained only beyond tolerance, 10-25 mDa": ion_ok & ~keep & (flag == 0) & (err > 0.010) & np.isfinite(err),
           "removed: no permitted loss within 25 mDa": ion_ok & ~keep & (flag == 0) & ~np.isfinite(err)}
    out.append("| category | ions | % ions | % intensity |")
    out.append("|---|---|---|---|")
    for k_, m_ in cat.items():
        out.append(f"| {k_} | {m_.sum()} | {100 * m_.sum() / ion_ok.sum():.3f} | {100 * it[m_].sum() / it[ion_ok].sum():.3f} |")
    xr = ion_ok & keep & (flag == 0) & ~(strict <= tol) & (err <= tol)
    out.append(f"\nkept through the halide -> OH exchange rule: {xr.sum()} ions ({100 * it[xr].sum() / it[ion_ok].sum():.3f} % of intensity)\n")
    out.append("## Removed fragment ions at 10 ppm / 3 mDa +grid\n")
    out.append(f"removed {rem.sum()} of {ion_ok.sum()} ions ({100 * rem.sum() / ion_ok.sum():.3f} %), "
               f"{100 * it[rem].sum() / it[ion_ok].sum():.3f} % of intensity; of these, ions >= 10 % of base peak: "
               f"{np.sum(rem & (it >= 99.9))}, >= 1 %: {np.sum(rem & (it >= 9.99))} (NIST base peak = 999)\n")
    out.append(f"spectra losing any ion: {100 * np.mean(np.bincount(spec, rem, n)[ok_spec] > 0):.2f} %; losing >= 5 % intensity: "
               f"{100 * np.mean((np.bincount(spec, it * rem, n) / np.maximum(tot_i, 1e-12))[ok_spec] >= 0.05):.3f} %\n")
    # top removed ions by intensity, for a gallery
    idx = np.flatnonzero(rem)
    idx = idx[np.argsort(-it[idx])][:40]
    out.append("| name | formula | precursor m/z | ion m/z | intensity | loss (Da) | best err (mDa) |")
    out.append("|---|---|---|---|---|---|---|")
    for j in idx:
        i = spec[j]
        e = err[j] * 1000 if np.isfinite(err[j]) else float("nan")
        out.append(f"| {str(names[i])[:40]} | {formulas[i]} | {pm[i]:.4f} | {mz[j]:.4f} | {it[j]:.0f} | "
                   f"{pm[i] - mz[j]:.4f} | {e:.1f} |")
    open(a.out, "w").write("\n".join(out) + "\n")
    print("\n".join(out[:16]))
    np.savez(a.out.replace(".md", "_keep.npz"), keep=rows["10 ppm/3 mDa +grid"], rem=rem)


if __name__ == "__main__":
    main()
