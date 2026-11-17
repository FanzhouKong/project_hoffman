import numpy as np

from bench.gc.centroid_profile import centroid_scan


def test_gaussian_profile_peaks_become_exact_centroids():
    # two Orbitrap-like profile peaks on an uneven m/z grid, zeros between them
    grid = np.concatenate([np.linspace(99.99, 100.01, 23), np.linspace(250.48, 250.52, 31)])
    y = np.zeros(len(grid))
    for mz0, h, sd in ((100.0007, 5e5, 0.0018), (250.5011, 2e4, 0.004)):
        y += h * np.exp(-0.5 * ((grid - mz0) / sd) ** 2)
    y[y < 1.0] = 0.0
    m, i = centroid_scan(grid, y, 0.0)
    assert len(m) == 2
    assert abs(m[0] - 100.0007) < 2e-6 and abs(m[1] - 250.5011) < 2e-6
    assert abs(i[0] / 5e5 - 1) < 1e-3 and abs(i[1] / 2e4 - 1) < 1e-3


def test_isolated_sample_and_threshold():
    mz = np.array([50.0, 50.001, 50.002, 60.0, 60.001])
    it = np.array([0.0, 7.0, 0.0, 3.0, 2.0])
    m, i = centroid_scan(mz, it, 0.0)
    assert m.tolist() == [50.001, 60.0] and i.tolist() == [7.0, 3.0]        # zero neighbours: the sample itself
    m, i = centroid_scan(mz, it, 5.0)
    assert m.tolist() == [50.001]


def test_profile_mzml_file_is_read_and_centroided(tmp_path):
    from bench.gc.centroid_profile import centroid_file
    from peak3d.io import from_scans, load_cloud
    from peak3d.synth import write_mzml
    import pytest

    grid = np.linspace(149.98, 150.02, 41)
    scans_m, scans_i = [], []
    for s in range(6):
        y = (1e4 * (s + 1)) * np.exp(-0.5 * ((grid - 150.0013) / 0.003) ** 2)
        y[y < 1.0] = 0.0
        scans_m.append(grid[y > 0])
        scans_i.append(y[y > 0])
    prof = from_scans("prof", 0.01 * np.arange(6), scans_m, scans_i, "pos")
    f = tmp_path / "prof.mzML"
    write_mzml(prof, f)
    f.write_text(f.read_text().replace('accession="MS:1000127" name="centroid spectrum"',
                                       'accession="MS:1000128" name="profile spectrum"'))
    with pytest.raises(ValueError, match="profile"):
        load_cloud(f)                                   # the picker itself refuses profile data
    cloud, n_prof = centroid_file(f)
    assert cloud.n_scans == 6 and cloud.n_points == 6 and n_prof == prof.n_points
    assert np.allclose(cloud.mz, 150.0013, atol=3e-6)
    assert np.allclose(cloud.inten, 1e4 * np.arange(1, 7), rtol=1e-3)
    write_mzml(cloud, tmp_path / "cent.mzML")
    back = load_cloud(tmp_path / "cent.mzML")
    assert back.n_points == 6 and np.allclose(back.rt, cloud.rt)
