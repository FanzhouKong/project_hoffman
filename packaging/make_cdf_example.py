"""Two small synthetic unit-resolution GC-MS runs as ANDI-MS netCDF files (.CDF), as a quadrupole export writes
them (one value per nominal mass, on a 0.1 Da grid), for the app build's check that the app reads .cdf files:

    python packaging/make_cdf_example.py OUT_DIR"""
import sys
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # the checkout, when peak3d is not installed
from peak3d.gc.synth import Compound, make_run, random_spectrum     # noqa: E402


def write_cdf(path, rt_s, scans):
    mz = np.concatenate([m for m, _ in scans])
    it = np.concatenate([i for _, i in scans])
    idx = np.concatenate([[0], np.cumsum([len(m) for m, _ in scans])[:-1]])
    with netcdf_file(str(path), "w") as nc:
        nc.test_ionization_polarity = b"Positive Polarity"
        nc.createDimension("scan_number", len(rt_s))
        nc.createDimension("point_number", len(mz))
        for name, dim, typ, val in (("scan_acquisition_time", "scan_number", "d", rt_s), ("scan_index", "scan_number", "i", idx),
                                    ("mass_values", "point_number", "d", mz), ("intensity_values", "point_number", "d", it)):
            v = nc.createVariable(name, typ, (dim,))
            v[:] = val


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    comps = [Compound(f"compound {k}", 0.5 + 0.3 * k, 10 ** rng.uniform(4.5, 5.2), *random_spectrum(rng, 15)) for k in range(8)]
    for i, (stem, scale) in enumerate((("run_a", 1.0), ("run_b", 0.6))):
        run = [Compound(c.name, c.rt, c.height * scale, c.mz, c.rel, c.fwhm) for c in comps]
        cloud, _ = make_run(run, n_scans=900, dt=1.0 / 300.0, seed=7 + i, name=stem)
        scans = []
        for s in range(cloud.n_scans):
            m, v = cloud.mz[cloud.off[s]:cloud.off[s + 1]], cloud.inten[cloud.off[s]:cloud.off[s + 1]].astype(np.float64)
            u, inv = np.unique(np.floor(m + 0.3).astype(np.int64), return_inverse=True)
            w = np.zeros(len(u))
            np.add.at(w, inv, v)
            scans.append((u + rng.integers(-2, 4, len(u)) / 10.0, w))
        write_cdf(out / f"{stem}.CDF", 60.0 * cloud.rt, scans)
    print(f"wrote 2 .CDF runs of {len(comps)} compounds to {out}")


if __name__ == "__main__":
    main(sys.argv[1])
