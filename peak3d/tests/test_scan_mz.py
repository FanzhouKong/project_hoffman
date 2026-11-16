"""scan_mz: the per-scan m/z shift measured on the file's long-lived ions and taken out before the estimation"""
import numpy as np

from peak3d.estimate import Params, correct_scan_mz, estimate, prepare, scan_mz_shifts
from peak3d.features import build_features
from peak3d.pick import pick_cloud
from peak3d.synth import Peak, make_cloud

FWHM = 0.02
N_SCANS = 600


def _study(seed=0, n_bg=120, shifts=True, frac=0.25, sig_a=1.0, sig_b=80.0):
    """n_bg background ions present in every scan (2e3-1e5 counts), 250 peaks, one known strong peak at m/z 205.0972
    whose apex lies in a shifted scan; with shifts, a quarter of the scans (and the three around the known apex) get
    their own mass-axis shift a + b / sqrt(m/z) ppm: about -30 ppm at m/z 100 and -5 at m/z 900 for the large ones"""
    rng = np.random.default_rng(seed)
    peaks = [Peak(205.0972, 1.5, 6.0e5, FWHM)]
    peaks += [Peak(float(rng.uniform(60, 900)), 1.5, float(10 ** rng.uniform(3.3, 5.0)), 1.0e4) for _ in range(n_bg)]
    peaks += [Peak(float(rng.uniform(100, 900)), float(rng.uniform(0.2, 2.8)), float(10 ** rng.uniform(3.0, 5.8)), FWHM)
              for _ in range(250)]
    cloud, truth = make_cloud(peaks, n_scans=N_SCANS, dt=0.005, sig_a=sig_a, sig_b=sig_b, thresh=150.0, floor=200.0,
                              n_noise=20000, seed=seed)
    a, b = np.zeros(cloud.n_scans), np.zeros(cloud.n_scans)
    if shifts:
        hit = rng.random(cloud.n_scans) < frac
        apex = int(np.argmin(np.abs(cloud.rt - 1.5)))
        hit[apex - 1:apex + 2] = True
        big = hit & (rng.random(cloud.n_scans) < 0.3)
        b[hit] = rng.normal(0.0, 60.0, hit.sum())
        b[big] = rng.choice([-1.0, 1.0], big.sum()) * rng.uniform(250.0, 450.0, big.sum())
        b[apex] = -400.0
        a[hit] = -b[hit] / 30.0 + rng.normal(0.0, 1.0, hit.sum())       # the top of the m/z range barely moves
        sc = cloud.scan_index()
        cloud.mz *= 1.0 + (a[sc] + b[sc] / np.sqrt(cloud.mz)) * 1e-6
    return cloud, truth, a, b


def _shift_at(a, b, m):
    return a + b / np.sqrt(m)


JUMP = 12.0          # the linking cap of the shifted study (4 x the m/z sd at its intensity floor), ppm


def test_on_by_default_and_off_leaves_the_file_alone():
    assert Params().scan_mz is True
    cloud, _, _, _ = _study()
    mz0 = cloud.mz.copy()
    P = prepare(cloud, {"scan_mz": False})
    assert np.array_equal(cloud.mz, mz0) and P.mz_scan_n == 0 and P.n_mz_ref == 0
    assert estimate(_study()[0]).mz_scan_n > 0                 # the default corrects it


def test_shifts_are_recovered():
    cloud, _, a, b = _study()
    sh = scan_mz_shifts(cloud)
    assert sh["n_ref"] >= 80
    done = sh["n"] > 0
    assert done.mean() > 0.95
    for m in (100.0, 300.0, 800.0):
        err = _shift_at(sh["a"], sh["b"], m)[done] - _shift_at(a, b, m)[done]
        assert np.abs(err).max() < 2.5 and 1.4826 * np.median(np.abs(err)) < 0.5, (m, np.abs(err).max())
    # significance: the shifted scans, hardly any of the others
    shifted = (np.abs(_shift_at(a, b, 150.0)) > 3.0) & done
    assert (sh["chi2"][shifted] > 22.0).mean() > 0.95            # 22: p = 0.01 / 600 scans
    assert (sh["chi2"][done & (b == 0)] > 22.0).mean() < 0.005


def test_correction_restores_the_mass_axis():
    cloud, truth, a, b = _study()
    clean, _, _, _ = _study(shifts=False)
    raw = cloud.mz.copy()
    info = correct_scan_mz(cloud, JUMP)
    assert info["mz_scan_n"] > 0.95 * cloud.n_scans and info["mz_scan_jumps"] > 20 and info["mz_scan_max"] > 15
    err_before = (raw - clean.mz) / clean.mz * 1e6
    err_after = (cloud.mz - clean.mz) / clean.mz * 1e6
    assert np.abs(err_before).max() > 25
    assert np.abs(err_after).max() < 3.0 and np.sqrt(np.mean(err_after ** 2)) < 0.4
    for s in range(cloud.n_scans):                                  # every scan still sorted
        assert np.all(np.diff(cloud.mz[cloud.off[s]:cloud.off[s + 1]]) >= 0)


def test_known_peak_recovers_and_the_error_model_stays():
    """with the correction the known peak, whose apex scans are shifted by 20-30 ppm, is one feature at its m/z over
    its whole width; the error model stays the one fitted on the file as recorded (no tolerance gets tighter)"""
    c_off, _, _, _ = _study()
    c_on, _, _, _ = _study()
    P_off, P_on = prepare(c_off, {"scan_mz": False}), prepare(c_on)
    assert P_on.scan_mz and P_on.mz_scan_n > 0 and P_on.mz_scan_jumps >= 3 and 0 < P_on.mz_scan_gain < 0.3
    assert (P_on.sig_a, P_on.sig_b, P_on.tol_max_ppm) == (P_off.sig_a, P_off.sig_b, P_off.tol_max_ppm)
    got = {}
    for name, c, P in (("off", c_off, P_off), ("on", c_on, P_on)):
        fe, _ = build_features(c, pick_cloud(c, P), None, False)
        k = fe[(np.abs(fe["mz"] - 205.0972) / 205.0972 * 1e6 < 40) & (np.abs(fe["rt"] - 1.5) < 0.05)]
        got[name] = k
    on = got["on"]
    assert len(on) == 1
    assert abs(on["mz"].iloc[0] - 205.0972) / 205.0972 * 1e6 < 1.5
    assert abs(on["height"].iloc[0] - 6.0e5) < 0.1 * 6.0e5
    off = got["off"]
    assert len(off) != 1 or abs(off["mz"].iloc[0] - 205.0972) / 205.0972 * 1e6 > 1.5 or off["n_scans"].iloc[0] < on["n_scans"].iloc[0]


def test_steady_file_is_left_alone():
    cloud, _, _, _ = _study(shifts=False)
    mz0 = cloud.mz.copy()
    info = correct_scan_mz(cloud, JUMP)
    assert np.array_equal(cloud.mz, mz0)
    assert info["mz_scan_n"] == 0 and info["mz_scan_fit"] > 500 and info["mz_scan_jumps"] == 0 and info["n_mz_ref"] >= 80
    assert info["mz_scan_max"] < 3.0
    P = prepare(cloud, {"scan_mz": True})
    assert P.mz_scan_n == 0 and np.array_equal(cloud.mz, mz0)
    P0 = estimate(cloud)
    assert (P.sig_a, P.sig_b, P.tol_max_ppm) == (P0.sig_a, P0.sig_b, P0.tol_max_ppm)


def test_too_few_reference_ions():
    cloud, _, _, _ = _study(n_bg=3)
    mz0 = cloud.mz.copy()
    assert scan_mz_shifts(cloud) is None
    info = correct_scan_mz(cloud, JUMP)
    assert info["mz_scan_n"] == 0 and np.array_equal(cloud.mz, mz0)


def test_strong_centroids_are_not_references():
    """a reference ion's centroids within a decade of the file's strongest are not used: their m/z may be distorted"""
    cloud, _, _, _ = _study(shifts=False)
    top = float(cloud.inten.max())
    sh = scan_mz_shifts(cloud)
    assert sh is not None
    # make every background centroid strong: nothing is left to measure with
    cloud.inten[:] = np.maximum(cloud.inten, np.float32(0.5 * top))
    assert scan_mz_shifts(cloud) is None


def test_two_jumps_are_not_enough_three_are():
    """the file is corrected from three jumped scans on (one saturating compound: its apex scans)"""
    for n_jump, want in ((2, False), (3, True)):
        cloud, _, _, _ = _study(shifts=False)
        sc = cloud.scan_index()
        for s in (300, 301, 302)[:n_jump]:           # around the known peak's apex (scan 300): its trace is cut there
            m = sc == s
            cloud.mz[m] *= 1.0 + (10.0 - 350.0 / np.sqrt(cloud.mz[m])) * 1e-6
        mz0 = cloud.mz.copy()
        P = prepare(cloud, {"scan_mz": True})
        assert P.mz_scan_jumps == n_jump
        assert (P.mz_scan_n > 0) == want and np.array_equal(cloud.mz, mz0) != want
        assert (P.mz_scan_gain < 0.3) == want                 # the known peak's centroids in the three scans: 20 ppm -> 1


def test_shift_concentrated_ions_do_not_hide_the_slope():
    """most reference ions in one narrow m/z region: the fit still follows the few ions elsewhere (the start is the
    line through the medians of slices of the m/z range, not the median of all ions)"""
    rng = np.random.default_rng(3)
    peaks = [Peak(205.0972, 1.5, 6.0e6, FWHM)]      # the file's strongest centroid: the background ions lie far below
    peaks += [Peak(float(rng.uniform(700, 850)), 1.5, float(10 ** rng.uniform(3.5, 5.0)), 1.0e4) for _ in range(90)]
    peaks += [Peak(float(m), 1.5, float(10 ** rng.uniform(3.8, 5.0)), 1.0e4) for m in np.linspace(60, 600, 25)]
    cloud, _ = make_cloud(peaks, n_scans=N_SCANS, dt=0.005, sig_a=1.0, sig_b=80.0, thresh=150.0, floor=200.0,
                          n_noise=20000, seed=3)
    sc = cloud.scan_index()
    for s in (150, 250, 350, 450):
        m = sc == s
        cloud.mz[m] *= 1.0 + (-16.0 + 550.0 / np.sqrt(cloud.mz[m])) * 1e-6      # +55 ppm at m/z 60, +3 at m/z 800
    sh = scan_mz_shifts(cloud)
    for s in (150, 250, 350, 450):
        assert sh["m_lo"][s] < 100 and sh["m_hi"][s] > 700
        for m in (80.0, 300.0, 800.0):
            assert abs(sh["a"][s] + sh["b"][s] / np.sqrt(m) - (-16.0 + 550.0 / np.sqrt(m))) < 3.0


def test_correction_is_undone_when_the_eluting_ions_do_not_share_it():
    """background ions that jitter from scan to scan under peaks that do not (a recorded background with simulated
    peaks laid over it): moving the scans back would move the peaks off, so the cloud is left as it was"""
    rng = np.random.default_rng(5)
    bg = [Peak(float(rng.uniform(60, 900)), 1.5, float(10 ** rng.uniform(3.3, 4.6)), 1.0e4) for _ in range(120)]
    cloud_bg, _ = make_cloud(bg, n_scans=N_SCANS, dt=0.005, sig_a=1.0, sig_b=80.0, thresh=150.0, floor=200.0, n_noise=20000,
                             seed=5)
    sc = cloud_bg.scan_index()
    hit = rng.random(cloud_bg.n_scans) < 0.3
    b = np.where(hit, rng.choice([-1.0, 1.0], cloud_bg.n_scans) * rng.uniform(200.0, 400.0, cloud_bg.n_scans), 0.0)
    cloud_bg.mz *= 1.0 + (b[sc] / np.sqrt(cloud_bg.mz)) * 1e-6
    pk = [Peak(float(rng.uniform(100, 900)), float(rng.uniform(0.2, 2.8)), float(10 ** rng.uniform(5.0, 6.8)), FWHM)
          for _ in range(300)]
    cloud_pk, _ = make_cloud(pk, n_scans=N_SCANS, dt=0.005, sig_a=0.3, sig_b=20.0, thresh=150.0, seed=6)
    from peak3d.io import from_scans
    rts, mzs, ins = [], [], []
    for s in range(cloud_bg.n_scans):
        m = np.r_[cloud_bg.mz[cloud_bg.off[s]:cloud_bg.off[s + 1]], cloud_pk.mz[cloud_pk.off[s]:cloud_pk.off[s + 1]]]
        i = np.r_[cloud_bg.inten[cloud_bg.off[s]:cloud_bg.off[s + 1]], cloud_pk.inten[cloud_pk.off[s]:cloud_pk.off[s + 1]]]
        o = np.argsort(m)
        rts.append(cloud_bg.rt[s]); mzs.append(m[o]); ins.append(i[o])
    cloud = from_scans("mixed", rts, mzs, ins, "pos")
    mz0 = cloud.mz.copy()
    P0 = estimate(cloud, overrides={"scan_mz": False})
    P = estimate(cloud)
    assert P.mz_scan_jumps > 50 and P.n_mz_ref >= 80          # the background ions do show the jumps
    assert P.mz_scan_n == 0 and P.mz_scan_gain > 1.0          # ... but the peaks sit further from their traces with it
    assert np.array_equal(cloud.mz, mz0)
    assert (P.sig_a, P.sig_b, P.tol_max_ppm) == (P0.sig_a, P0.sig_b, P0.tol_max_ppm)


def test_jitter_within_the_linking_cap_is_left_alone():
    """scan-to-scan shifts of a few ppm, all inside the widest step the picker follows: part of the file's error model,
    not jumps; the cloud is not touched"""
    cloud, _, _, _ = _study(shifts=False)
    rng = np.random.default_rng(9)
    sc = cloud.scan_index()
    a = rng.normal(0.0, 1.5, cloud.n_scans)
    cloud.mz *= 1.0 + a[sc] * 1e-6
    mz0 = cloud.mz.copy()
    P = estimate(cloud, overrides={"scan_mz": True})
    assert P.mz_scan_fit > 500 and 1.0 < P.mz_scan_sd < 2.0 and P.mz_scan_max < P.tol_max_ppm
    assert P.mz_scan_jumps == 0 and P.mz_scan_n == 0 and np.array_equal(cloud.mz, mz0)


def test_short_run_or_non_positive_mz_is_not_measured():
    """fewer scans than one reference window, or an m/z of 0 or below (not a recorded value): nothing is measured and
    the estimation goes on as without the option"""
    cloud, _, _, _ = _study()
    cloud.mz[0] = -1.0
    mz0 = cloud.mz.copy()
    assert scan_mz_shifts(cloud) is None
    P = estimate(cloud)
    assert P.scan_mz and P.mz_scan_n == 0 and P.n_mz_ref == 0 and np.array_equal(cloud.mz, mz0)
    short, _ = make_cloud([Peak(300.0, 0.2, 1e5, 0.05)] + [Peak(100.0 + 50 * i, 0.2, 1e4, 1e4) for i in range(10)],
                          n_scans=80, dt=0.005, floor=50.0, n_noise=500, thresh=20.0)
    assert scan_mz_shifts(short) is None and estimate(short).mz_scan_n == 0
