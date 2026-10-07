# LI2018: residual after correction for the two compound populations (arm = SB earlier than SA by > 2.5 s; core = same RT in both)
import numpy as np, pandas as pd, sys
sys.path.insert(0, "bench/diag")
from rt_external_validation import collect
pd.set_option("display.width", 250)
tr, stems, dt, nat, cor = collect("LI2018")
ok = np.isfinite(nat).sum(1) >= 5; tr = tr[ok].reset_index(drop=True); nat, cor = nat[ok], cor[ok]
ref = np.nanmedian(nat, axis=1); refc = np.nanmedian(cor, axis=1)
sh = 60 * (ref[:, None] - nat); res = 60 * (refc[:, None] - cor)
sa = [i for i, s in enumerate(stems) if s.startswith("SA")]; sb = [i for i, s in enumerate(stems) if s.startswith("SB")]
sb_shift = np.nanmedian(sh[:, sb], axis=1); sa_shift = np.nanmedian(sh[:, sa], axis=1); gap = sb_shift - sa_shift
maf = pd.read_csv("data/raw/LI2018_QE/m_MTBLS733_mass_spectrometry_v2_maf.tsv", sep="\t")
ab = maf["Compound concentration ratio"].str.split("\\\\", expand=True).astype(float); maf["exp_log2"] = np.log2(ab[0] / ab[1])
key = {(round(m, 4), round(t, 3)): e for m, t, e in zip(maf.mass_to_charge, maf.retention_time, maf.exp_log2)}
tr["exp_log2"] = [key.get((round(m, 4), round(t, 3)), np.nan) for m, t in zip(tr.mz, tr.rt)]
tr["gap_s"] = gap; tr["ref_rt"] = ref
print("SB-SA gap (s) of truth features by RT bin: median, p25, p75, n")
b = (ref // 5 * 5).astype(int)
print(pd.DataFrame(dict(gap=gap, bin=b)).groupby("bin").gap.describe()[["count", "25%", "50%", "75%"]].round(2).to_string())
print("\ngap vs differential/constant (whole run):")
tr["cls"] = np.where(tr.exp_log2 == 0, "constant", np.where(tr.exp_log2.isna(), "unknown", "differential"))
print(tr.groupby("cls").gap_s.describe()[["count", "25%", "50%", "75%"]].round(2).to_string())
print("\ngap vs m/z (4-15 min window):")
w = (ref >= 4) & (ref <= 15)
print(pd.DataFrame(dict(gap=gap[w], mzbin=(tr.mz.values[w] // 100 * 100).astype(int))).groupby("mzbin").gap.describe()[["count", "25%", "50%", "75%"]].round(2).to_string())
print("\n=== residual after correction (s), per file and RT bin, arm vs core ===")
arm = gap > 2.5; core = np.abs(gap) < 1.0
rows = []
for j, s in enumerate(stems):
    for lo in (0, 5, 10, 15):
        inb = (ref >= lo) & (ref < lo + 5)
        r = {}
        for name, m in (("arm", arm), ("core", core)):
            k = inb & m & np.isfinite(res[:, j])
            r[f"{name}_n"] = int(k.sum()); r[f"{name}_native_s"] = np.median(sh[k, j]) if k.sum() >= 5 else np.nan; r[f"{name}_corr_s"] = np.median(res[k, j]) if k.sum() >= 5 else np.nan
        rows.append(dict(file=s, bin=f"{lo}-{lo + 5}", **r))
print(pd.DataFrame(rows).round(2).to_string(index=False))
print("\nmedian |residual| over all (truth feature, file) in 4-15 min, native -> corrected:",
      round(float(np.nanmedian(np.abs(sh[w]))), 2), "->", round(float(np.nanmedian(np.abs(res[w]))), 2), "s;  outside the window:",
      round(float(np.nanmedian(np.abs(sh[~w]))), 2), "->", round(float(np.nanmedian(np.abs(res[~w]))), 2), "s")
