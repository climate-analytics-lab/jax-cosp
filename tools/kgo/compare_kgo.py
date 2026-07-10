"""Compare jcosp's radar path against the compiled COSP Fortran (KGO).

Generates a physically plausible multi-subcolumn case, runs the reference
binary built by build.sh, and checks z_vol / kr_vol / g_vol / dBZe / Ze_non
and prec_scops output against the JAX implementation.

Run directly (prints a report) or through jcosp/kgo_test.py (pytest).
"""

import pathlib
import subprocess

import numpy as np

HERE = pathlib.Path(__file__).parent
BINARY = HERE / "drv_radar"


def make_case(nprof=20, ngate=30, seed=7, snap_temperature=False):
    """Idealized raining/snowing profiles exercising 6 hydrometeor classes.

    With ``snap_temperature`` every level's temperature sits exactly on the
    LUT's 5-K bin grid (X.15 K values shared by the liquid and ice grids).
    There the Fortran's lazy scale-table fill (which evaluates the Mie/DSD
    integrals at the *first encountered* temperature in a bin) coincides with
    jcosp's offline bin-center table, making the comparison exact; with
    realistic temperatures the agreement is bounded by that temperature
    quantization instead (see docs/translation_notes.md).
    """
    import jcosp.config as config

    rng = np.random.default_rng(seed)
    z = np.linspace(19500.0, 250.0, ngate)  # mid-point heights (m), TOA-first
    p = 101325.0 * np.exp(-z / 7500.0)
    t = np.clip(288.0 - 6.5e-3 * z, 210.0, None)
    if snap_temperature:
        t = np.round((t - 273.15) / 5.0) * 5.0 + 273.15
    sh = 0.012 * (p / 101325.0) ** 2

    hm = np.zeros((nprof, ngate, config.N_HYDRO))
    re = np.zeros((nprof, ngate, config.N_HYDRO))
    frac_out = np.zeros((nprof, ngate), dtype=int)

    liq = (z > 1000) & (z < 3200)
    ice = (z > 5500) & (z < 9000)
    below = z < 3200
    for i in range(nprof):
        if i % 4 != 3:  # 3/4 of subcolumns are cloudy
            frac_out[i, liq | ice] = 1 if i % 3 else 2
            hm[i, liq, config.I_LSCLIQ] = rng.uniform(0.05, 0.6)
            hm[i, ice, config.I_LSCICE] = rng.uniform(0.01, 0.15)
            hm[i, below, config.I_LSRAIN] = rng.uniform(0.01, 0.8)
            hm[i, ice, config.I_LSSNOW] = rng.uniform(0.01, 0.3)
            if i % 5 == 0:
                hm[i, below, config.I_CVRAIN] = rng.uniform(0.05, 0.5)
                hm[i, ice, config.I_LSGRPL] = rng.uniform(0.01, 0.2)
            if i % 2 == 0:  # half the columns prescribe cloud Reff (um)
                re[i, liq, config.I_LSCLIQ] = rng.uniform(6.0, 14.0)
                re[i, ice, config.I_LSCICE] = rng.uniform(20.0, 60.0)
    ls_p = np.where(z < 3500, 1.0, 0.0)
    cv_p = np.where(z < 2500, 1.0, 0.0)
    return dict(p=p, t=t, sh=sh, hgt_km=z / 1000.0, hm=hm, re=re,
                frac_out=frac_out, ls_p=ls_p, cv_p=cv_p)


def run_fortran(case, workdir=HERE):
    nprof, ngate, nh = case["hm"].shape
    lines = [f"{nprof} {ngate}"]
    for k in range(ngate):
        lines.append(f'{case["p"][k]:.17e} {case["t"][k]:.17e} '
                     f'{case["sh"][k]:.17e} {case["hgt_km"][k]:.17e}')
    for i in range(nprof):
        for k in range(ngate):
            vals = list(case["hm"][i, k]) + list(case["re"][i, k])
            lines.append(" ".join(f"{v:.17e}" for v in vals))
    for i in range(nprof):
        lines.append(" ".join(str(v) for v in case["frac_out"][i]))
    lines.append(" ".join(f"{v:.1f}" for v in case["ls_p"]))
    lines.append(" ".join(f"{v:.1f}" for v in case["cv_p"]))
    (workdir / "kgo_input.txt").write_text("\n".join(lines) + "\n")

    subprocess.run([str(BINARY)], cwd=workdir, check=True, capture_output=True)

    raw = (workdir / "kgo_output.txt").read_text().split("\n")
    flat = np.array([[float(v) for v in ln.split()]
                     for ln in raw[: nprof * ngate]])
    out = {name: flat[:, j].reshape(nprof, ngate)
           for j, name in enumerate(["z_vol", "kr_vol", "g_vol",
                                     "dbze", "ze_non"])}
    prec = np.array([[int(v) for v in ln.split()]
                     for ln in raw[nprof * ngate: nprof * ngate + nprof]])
    out["prec_frac"] = prec
    return out


def run_jcosp(case):
    import importlib.resources

    import jax
    from jcosp import config
    from jcosp.radar.gases import gases
    from jcosp.radar.optics import load_lut, quickbeam_optics
    from jcosp.radar.quickbeam import quickbeam_subcolumn
    from jcosp.subcol import prec_scops

    jax.config.update("jax_enable_x64", True)
    lut = load_lut(importlib.resources.files("jcosp") / "data"
                   / "quickbeam_lut_94GHz_1M.npz")
    # jcosp layout: (nlev, ncol, ...); the case arrays are (ncol, nlev, ...).
    hm = np.moveaxis(case["hm"], 0, 1)
    re = np.moveaxis(case["re"], 0, 1)
    p = case["p"][:, None]
    t = case["t"][:, None]
    z_vol, _, kr_vol = quickbeam_optics(
        lut, config.HYDRO_CLASSES_1M, hm, re, None, p, t)
    # Legacy modes reproduce the reference Fortran's O2 interference term
    # and gate-2 attenuation exactly; the library defaults fix both (see
    # docs/translation_notes.md "Fixed upstream bugs").
    g_vol = gases(p, t, case["sh"][:, None], 94.0, legacy_o2_interference=True)
    g_vol = np.broadcast_to(np.asarray(g_vol), z_vol.shape)
    dbze, ze_non = quickbeam_subcolumn(
        z_vol, kr_vol, g_vol, case["hgt_km"][:, None], legacy_gate2=True)
    prec = prec_scops(case["frac_out"].T, case["ls_p"], case["cv_p"])
    return {"z_vol": np.asarray(z_vol).T, "kr_vol": np.asarray(kr_vol).T,
            "g_vol": np.asarray(g_vol).T, "dbze": np.asarray(dbze).T,
            "ze_non": np.asarray(ze_non).T,
            "prec_frac": np.asarray(prec).T}


def compare(fort, ours, exact=True, verbose=True):
    """Return dict of max relative/absolute errors; raise on mismatch.

    ``exact`` selects the snapped-temperature tolerances; otherwise the
    comparison is bounded by the Fortran-vs-jcosp temperature-bin evaluation
    difference (~3% in Ze, ~0.15 dB in dBZe).
    """
    report = {}
    hydro = fort["z_vol"] > 0
    if exact:
        # 1e-6 relative is the numerical floor: the ports use math.gamma and
        # float64 Liebe coefficients where the Fortran uses its own gamma
        # series and single-precision literals.
        tols = [("z_vol", 1e-6, "rel"), ("kr_vol", 1e-6, "rel"),
                ("g_vol", 1e-5, "rel"), ("ze_non", 1e-5, "abs"),
                ("dbze", 1e-4, "abs")]
    else:
        tols = [("z_vol", 5e-2, "rel"), ("kr_vol", 5e-2, "rel"),
                ("g_vol", 1e-5, "rel"), ("ze_non", 0.25, "abs"),
                ("dbze", 0.25, "abs")]
    for name, tol, kind in tols:
        f, o = fort[name], ours[name]
        if kind == "rel":
            err = np.abs(o - f) / np.maximum(np.abs(f), 1e-30)
            err = np.where(hydro | (name == "g_vol"), err, 0.0)
        else:
            err = np.where(hydro, np.abs(o - f), 0.0)  # sentinel cells equal
        report[name] = float(err.max())
        assert err.max() < tol, f"{name}: max {kind} err {err.max():.3e} > {tol}"
        if verbose:
            print(f"  {name:8s} max {kind} err {err.max():.3e}  (tol {tol})")
    same = (fort["prec_frac"] == ours["prec_frac"]).all()
    report["prec_frac_identical"] = bool(same)
    assert same, "prec_scops mismatch"
    if verbose:
        print("  prec_frac identical")
    return report


def main():
    print("KGO comparison vs compiled COSP Fortran")
    print("1) temperatures on the LUT bin grid (exact regime):")
    case = make_case(snap_temperature=True)
    compare(run_fortran(case), run_jcosp(case), exact=True)
    print("2) realistic temperatures (bounded by the Fortran's "
          "first-encounter-T lazy table fill vs jcosp's bin-center LUT):")
    case = make_case(snap_temperature=False)
    fort = run_fortran(case)
    compare(fort, run_jcosp(case), exact=False)
    n = (fort["z_vol"] > 0).sum()
    print(f"  ({n} hydrometeor-bearing volumes compared; "
          f"dBZe range {fort['dbze'][fort['z_vol'] > 0].min():.1f} to "
          f"{fort['dbze'][fort['z_vol'] > 0].max():.1f})")


if __name__ == "__main__":
    main()
