"""Tests for the ISCCP (ICARUS) port.

The vectorized JAX implementation is checked against a direct, loop-for-loop
NumPy transliteration of icarus.F90 (ICARUS_SUBCOLUMN and ICARUS_COLUMN,
Fortran (npoints, ncol, nlev) layout, 1-based level indices), plus physical
sanity checks of the radiance stack, the IR cloud-top adjustment, the night
handling and the joint histogram, and gradient-finiteness checks.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcosp import config
from jcosp.constants import R_UNDEF
from jcosp.isccp import simulate_isccp
from jcosp.isccp.simulator import (isccp_column, isccp_subcolumn,
                                   water_vapour_emissivity)

TAUCHK = config.ISCCP_TAUCHK
PL = config.ISCCP_PLANCK_K


# ---------------------------------------------------------------------------
# Loop transliteration of icarus.F90 (Fortran array layout and 1-based ilev).
# ---------------------------------------------------------------------------
def icarus_subcolumn_ref(sunlit, dtau, dem, skt, emsfc_lw, qv, at, pfull,
                         phalf, frac_out, top_height, top_height_direction):
    """ICARUS_SUBCOLUMN transliteration; arrays (npoints[, ncol][, nlev])."""
    npoints, ncol, nlev = dtau.shape
    tau = dtau.sum(axis=2).astype(float)

    ptrop = np.full(npoints, 5000.0)
    attropmin = np.full(npoints, 400.0)
    atmax = np.zeros(npoints)
    attrop = np.full(npoints, 120.0)
    itrop = np.ones(npoints, int)
    meantbclr = np.full(npoints, R_UNDEF)
    tb = np.zeros((npoints, ncol))
    boxttop = np.zeros((npoints, ncol))
    fluxtop = np.zeros((npoints, ncol))
    fluxtop_clrsky = np.zeros(npoints)
    dem_wv = np.zeros((npoints, nlev))
    ptop = np.zeros((npoints, ncol))
    levmatch = np.zeros((npoints, ncol), int)

    if top_height in (1, 3):
        for ilev in range(1, nlev + 1):
            for j in range(npoints):
                if (pfull[j, ilev - 1] < 40000.0
                        and pfull[j, ilev - 1] > 5000.0
                        and at[j, ilev - 1] < attropmin[j]):
                    ptrop[j] = pfull[j, ilev - 1]
                    attropmin[j] = at[j, ilev - 1]
                    attrop[j] = attropmin[j]
                    itrop[j] = ilev
        for ilev in range(1, nlev + 1):
            for j in range(npoints):
                if at[j, ilev - 1] > atmax[j] and ilev >= itrop[j]:
                    atmax[j] = at[j, ilev - 1]

        # Water-vapour continuum emissivity (Schwarzkopf & Ramaswamy 1999).
        for ilev in range(nlev):
            press = pfull[:, ilev] * 10.0
            dpress = (phalf[:, ilev + 1] - phalf[:, ilev]) * 10.0
            atmden = dpress / (config.ISCCP_GRAV * 100.0)
            rvh20 = qv[:, ilev] * config.ISCCP_AMD / config.ISCCP_AMW
            wk = rvh20 * config.ISCCP_AVO * atmden / config.ISCCP_AMD
            rhoave = (press / config.ISCCP_PSTD) * (config.ISCCP_T0
                                                    / at[:, ilev])
            rh20s = rvh20 * rhoave
            rfrgn = rhoave - rh20s
            tmpexp = np.exp(-0.02 * (at[:, ilev] - config.ISCCP_T0))
            tauwv = wk * 1e-20 * ((0.0224697 * rh20s * tmpexp)
                                  + (3.41817e-7 * rfrgn)) * 0.98
            dem_wv[:, ilev] = 1.0 - np.exp(-tauwv)

        # Clear-sky radiance.
        trans = np.ones(npoints)
        fluxtop_clrsky[:] = 0.0
        for ilev in range(nlev):
            bb = 1.0 / (np.exp(PL / at[:, ilev]) - 1.0)
            fluxtop_clrsky += dem_wv[:, ilev] * bb * trans
            trans = trans * (1.0 - dem_wv[:, ilev])
        bb = 1.0 / (np.exp(PL / skt) - 1.0)
        fluxtop_clrsky += emsfc_lw * bb * trans
        meantbclr = PL / np.log(1.0 + 1.0 / fluxtop_clrsky)

        # All-sky radiance.
        trans_a = np.ones((npoints, ncol))
        for ilev in range(nlev):
            bb = 1.0 / (np.exp(PL / at[:, ilev]) - 1.0)
            for ibox in range(ncol):
                d = np.where(dem[:, ibox, ilev] == 0.0, dem_wv[:, ilev],
                             1.0 - (1.0 - dem[:, ibox, ilev])
                             * (1.0 - dem_wv[:, ilev]))
                fluxtop[:, ibox] += d * bb * trans_a[:, ibox]
                trans_a[:, ibox] *= (1.0 - d)
        bb = 1.0 / (np.exp(PL / skt) - 1.0)
        for ibox in range(ncol):
            fluxtop[:, ibox] += emsfc_lw * bb * trans_a[:, ibox]
        boxttop = PL / np.log(1.0 + 1.0 / fluxtop)

        # Cloud-top temperature with the ISCCP partial-emissivity adjustment.
        btcmin = 1.0 / (np.exp(PL / (attrop - 5.0)) - 1.0)
        fluxtopinit = np.zeros(npoints)
        for ibox in range(ncol):
            transmax = (fluxtop[:, ibox] - btcmin) / (fluxtop_clrsky - btcmin)
            tauir = tau[:, ibox] / 2.13
            taumin = -np.log(np.clip(transmax, 0.001, 0.9999999))
            if top_height == 1:
                for j in range(npoints):
                    if 0.001 < transmax[j] <= 0.9999999:
                        fluxtopinit[j] = fluxtop[j, ibox]
                        tauir[j] = tau[j, ibox] / 2.13
                for _ in range(2):
                    for j in range(npoints):
                        if tau[j, ibox] > TAUCHK and \
                                0.001 < transmax[j] <= 0.9999999:
                            emcld = 1.0 - np.exp(-tauir[j])
                            ft = fluxtopinit[j] \
                                - (1.0 - emcld) * fluxtop_clrsky[j]
                            ft = max(1e-6, ft / emcld)
                            tb[j, ibox] = PL / np.log(1.0 + 1.0 / ft)
                            fluxtop[j, ibox] = ft
                            if tb[j, ibox] > 260.0:
                                tauir[j] = tau[j, ibox] / 2.56
            for j in range(npoints):
                if tau[j, ibox] > TAUCHK:
                    tb[j, ibox] = PL / np.log(1.0 + 1.0 / fluxtop[j, ibox])
                    if top_height == 1 and tauir[j] < taumin[j]:
                        tb[j, ibox] = attrop[j] - 5.0
                        tau[j, ibox] = 2.13 * taumin[j]
                else:
                    tb[j, ibox] = meantbclr[j]

    # Cloud-top pressure.
    for ibox in range(ncol):
        if top_height in (1, 3):
            nmatch = np.zeros(npoints, int)
            match = np.zeros((npoints, nlev - 1), int)
            for k1 in range(1, nlev):
                ilev = nlev - k1 if top_height_direction == 2 else k1
                for j in range(npoints):
                    if (ilev >= itrop[j]
                        and ((at[j, ilev - 1] >= tb[j, ibox]
                              and at[j, ilev] <= tb[j, ibox])
                             or (at[j, ilev - 1] <= tb[j, ibox]
                                 and at[j, ilev] >= tb[j, ibox]))):
                        nmatch[j] += 1
                        match[j, nmatch[j] - 1] = ilev
            for j in range(npoints):
                if nmatch[j] >= 1:
                    k1v = match[j, nmatch[j] - 1]
                    k2v = k1v + 1
                    logp1 = np.log(pfull[j, k1v - 1])
                    logp2 = np.log(pfull[j, k2v - 1])
                    atd = max(TAUCHK, abs(at[j, k2v - 1] - at[j, k1v - 1]))
                    logp = logp1 + (logp2 - logp1) \
                        * abs(tb[j, ibox] - at[j, k1v - 1]) / atd
                    ptop[j, ibox] = np.exp(logp)
                    levmatch[j, ibox] = (
                        k1v if abs(pfull[j, k1v - 1] - ptop[j, ibox])
                        < abs(pfull[j, k2v - 1] - ptop[j, ibox]) else k2v)
                else:
                    if tb[j, ibox] <= attrop[j]:
                        ptop[j, ibox] = ptrop[j]
                        levmatch[j, ibox] = itrop[j]
                    if tb[j, ibox] >= atmax[j]:
                        ptop[j, ibox] = pfull[j, nlev - 1]
                        levmatch[j, ibox] = nlev
        else:
            for ilev in range(1, nlev + 1):
                for j in range(npoints):
                    if (ptop[j, ibox] == 0.0
                            and frac_out[j, ibox, ilev - 1] > 0.0):
                        ptop[j, ibox] = phalf[j, ilev - 1]
                        levmatch[j, ibox] = ilev
        for j in range(npoints):
            if tau[j, ibox] <= TAUCHK:
                ptop[j, ibox] = 0.0
                levmatch[j, ibox] = 0

    boxtau = np.full((npoints, ncol), R_UNDEF)
    boxptop = np.full((npoints, ncol), R_UNDEF)
    for ibox in range(ncol):
        for j in range(npoints):
            if tau[j, ibox] > TAUCHK and ptop[j, ibox] > 0.0:
                if sunlit[j] == 1 or top_height == 3:
                    boxtau[j, ibox] = tau[j, ibox]
                    boxptop[j, ibox] = ptop[j, ibox]

    return dict(boxtau=boxtau, boxptop=boxptop, boxttop=boxttop,
                meantbclr=meantbclr, levmatch=levmatch)


def icarus_column_ref(boxtau, boxptop, sunlit, boxttop, top_height):
    """ICARUS_COLUMN transliteration (boxptop in hPa, as at the call sites)."""
    npoints, ncol = boxtau.shape
    ntau, npres = config.NUM_ISCCP_TAU_BINS, config.NUM_ISCCP_PRES_BINS
    tau_bins = config.ISCCP_HIST_TAU
    pres_bins = config.ISCCP_HIST_PRES

    if top_height in (1, 3):
        meantb = boxttop.sum(axis=1) / ncol
    else:
        meantb = np.full(npoints, R_UNDEF)

    active = (sunlit == 1) | (top_height == 3)
    fq = np.where(active[:, None, None], 0.0, R_UNDEF) \
        * np.ones((npoints, ntau, npres))
    totalcldarea = np.where(active, 0.0, R_UNDEF)
    meanalbedocld = np.where(active, 0.0, R_UNDEF)
    meanptop = np.where(active, 0.0, R_UNDEF)
    meantaucld = np.where(active, 0.0, R_UNDEF)

    for j in range(npoints):
        cloudy = (boxtau[j] > TAUCHK) & (boxptop[j] > 0.0)
        if not active[j]:
            continue
        for it in range(ntau):
            for ip in range(npres):
                sel = ((boxtau[j] >= tau_bins[it])
                       & (boxtau[j] < tau_bins[it + 1])
                       & (boxptop[j] >= pres_bins[ip])
                       & (boxptop[j] < pres_bins[ip + 1]))
                fq[j, it, ip] = sel.sum() / ncol
        thick = cloudy & (boxtau[j] > config.ISCCP_TAUMIN)
        totalcldarea[j] = thick.sum() / ncol
        tt = np.where(thick, boxtau[j], 1.0)  # keep R_UNDEF out of the power
        alb = np.where(thick, tt ** 0.895 / (tt ** 0.895 + 6.82), 0.0)
        meanalbedocld[j] = alb.sum() / ncol
        meanptop[j] = boxptop[j][thick].sum() / ncol

    for j in range(npoints):
        if totalcldarea[j] > 0.0:
            meanptop[j] = 100.0 * meanptop[j] / totalcldarea[j]
            meanalbedocld[j] = meanalbedocld[j] / totalcldarea[j]
            meantaucld[j] = (6.82 / ((1.0 / meanalbedocld[j]) - 1.0)) \
                ** (1.0 / 0.895)
        else:
            meanptop[j] = R_UNDEF
            meanalbedocld[j] = R_UNDEF
            meantaucld[j] = R_UNDEF

    totalcldarea = np.where(totalcldarea != R_UNDEF, totalcldarea * 100.0,
                            totalcldarea)
    fq = np.where(fq != R_UNDEF, fq * 100.0, fq)
    return dict(fq_isccp=fq, totalcldarea=totalcldarea, meanptop=meanptop,
                meantaucld=meantaucld, meanalbedocld=meanalbedocld,
                meantb=meantb)


# ---------------------------------------------------------------------------
# Test-input construction.
# ---------------------------------------------------------------------------
def make_profiles(npoints=4, nlev=25, seed=0):
    """Fortran-layout atmosphere + subcolumn optics with varied cloud types."""
    rng = np.random.default_rng(seed)
    # Edge pressures 40 Pa .. 101325 Pa; midpoints between them.
    phalf_1d = np.geomspace(40.0, 101325.0, nlev + 1)
    pfull_1d = 0.5 * (phalf_1d[:-1] + phalf_1d[1:])
    pfull = np.tile(pfull_1d, (npoints, 1))
    phalf = np.tile(phalf_1d, (npoints, 1))
    # Troposphere lapse to 210 K, mild stratospheric warming above 100 hPa.
    at = np.clip(288.0 * (pfull / 101325.0) ** 0.19, 210.0, None)
    at = at + np.where(pfull < 10000.0,
                       12.0 * (1.0 - pfull / 10000.0), 0.0)
    at += rng.uniform(-0.5, 0.5, at.shape)
    qv = 0.008 * (pfull / 101325.0) ** 2

    ncol = 10
    dtau = np.zeros((npoints, ncol, nlev))
    # Random broken cloud layers.
    mask = rng.random((npoints, ncol, nlev)) < 0.15
    dtau[mask] = rng.uniform(0.02, 6.0, mask.sum())
    dtau[:, 0, :] = 0.0                      # a guaranteed clear subcolumn
    dtau[:, 1, :] = 0.0                      # thin cirrus near 250 hPa
    dtau[:, 1, np.argmin(np.abs(pfull_1d - 25000.0))] = 0.4
    dtau[:, 2, :] = 0.0                      # thick low cloud near 900 hPa
    dtau[:, 2, np.argmin(np.abs(pfull_1d - 90000.0))] = 18.0
    # Mostly consistent 10.5-um emissivities, with some scatter.
    dem = np.where(dtau > 0.0,
                   np.clip((1.0 - np.exp(-dtau / 2.13))
                           * rng.uniform(0.8, 1.2, dtau.shape), 0.0, 0.99),
                   0.0)
    frac_out = np.where(dtau > 0.0,
                        rng.integers(1, 3, dtau.shape), 0).astype(float)
    skt = at[:, -1] + rng.uniform(-1.0, 3.0, npoints)
    sunlit = np.array(([1, 0] * npoints)[:npoints])
    sunlit[0] = 1
    return dict(sunlit=sunlit, dtau=dtau, dem=dem, skt=skt, emsfc_lw=0.98,
                qv=qv, at=at, pfull=pfull, phalf=phalf, frac_out=frac_out)


def to_port(f):
    """Fortran (npoints, ncol, nlev) / (npoints, nlev) -> port layout."""
    return np.ascontiguousarray(
        f.transpose(2, 1, 0) if f.ndim == 3 else f.T)


def run_port_subcolumn(fin, top_height, top_height_direction):
    return isccp_subcolumn(
        to_port(fin["dtau"]), to_port(fin["dem"]), to_port(fin["frac_out"]),
        fin["skt"], fin["emsfc_lw"], to_port(fin["qv"]), to_port(fin["at"]),
        to_port(fin["pfull"]), to_port(fin["phalf"]), sunlit=fin["sunlit"],
        top_height=top_height, top_height_direction=top_height_direction)


class IsccpSubcolumnTest(unittest.TestCase):
    def assert_matches_ref(self, top_height, top_height_direction, seed=0):
        fin = make_profiles(seed=seed)
        ref = icarus_subcolumn_ref(**fin, top_height=top_height,
                                   top_height_direction=top_height_direction)
        out = run_port_subcolumn(fin, top_height, top_height_direction)
        np.testing.assert_allclose(np.asarray(out.boxtau).T, ref["boxtau"],
                                   rtol=1e-12, atol=0)
        np.testing.assert_allclose(np.asarray(out.boxptop).T, ref["boxptop"],
                                   rtol=1e-10, atol=0)
        # levmatch: port is 0-based with -1 for unset; Fortran 1-based/0.
        np.testing.assert_array_equal(np.asarray(out.levmatch).T,
                                      ref["levmatch"] - 1)
        if top_height in (1, 3):
            np.testing.assert_allclose(np.asarray(out.boxttop).T,
                                       ref["boxttop"], rtol=1e-12)
            np.testing.assert_allclose(np.asarray(out.meantbclr),
                                       ref["meantbclr"], rtol=1e-12)
        else:
            # Documented deviation: the Fortran leaves boxttop/meantbclr
            # uninitialized for top_height=2; the port fills R_UNDEF.
            self.assertTrue((np.asarray(out.boxttop) == R_UNDEF).all())
            self.assertTrue((np.asarray(out.meantbclr) == R_UNDEF).all())

    def test_matches_transliteration_th1_dir2(self):
        for seed in range(3):
            self.assert_matches_ref(1, 2, seed=seed)

    def test_matches_transliteration_th1_dir1(self):
        self.assert_matches_ref(1, 1)

    def test_matches_transliteration_th3(self):
        self.assert_matches_ref(3, 2)

    def test_matches_transliteration_th2(self):
        self.assert_matches_ref(2, 2)

    def test_invalid_options_raise(self):
        fin = make_profiles()
        with self.assertRaises(ValueError):
            run_port_subcolumn(fin, 4, 2)
        with self.assertRaises(ValueError):
            run_port_subcolumn(fin, 1, 3)


class IsccpColumnTest(unittest.TestCase):
    def test_matches_transliteration(self):
        for top_height in (1, 3, 2):
            fin = make_profiles(seed=1)
            sub_ref = icarus_subcolumn_ref(**fin, top_height=top_height,
                                           top_height_direction=2)
            boxttop = (sub_ref["boxttop"] if top_height in (1, 3)
                       else np.full_like(sub_ref["boxtau"], R_UNDEF))
            ref = icarus_column_ref(sub_ref["boxtau"],
                                    sub_ref["boxptop"] / 100.0,
                                    fin["sunlit"], boxttop, top_height)
            col = isccp_column(jnp.asarray(sub_ref["boxtau"].T),
                               jnp.asarray(sub_ref["boxptop"].T / 100.0),
                               fin["sunlit"], jnp.asarray(boxttop.T),
                               top_height=top_height)
            np.testing.assert_allclose(
                np.asarray(col.fq_isccp).transpose(2, 0, 1),
                ref["fq_isccp"], rtol=1e-12, atol=0)
            for name in ("totalcldarea", "meanptop", "meantaucld",
                         "meanalbedocld", "meantb"):
                np.testing.assert_allclose(np.asarray(getattr(col, name)),
                                           ref[name], rtol=1e-12, atol=0,
                                           err_msg=f"{name} th={top_height}")


class IsccpPhysicsTest(unittest.TestCase):
    """Physical sanity of the radiances, pc adjustment and statistics."""

    def _single_cloud(self, p_cloud, dtau_val, dem_val=None, nlev=25,
                      ncol=4, cloudy_boxes=(1, 2, 3)):
        """One gridpoint; a single-layer cloud in the given subcolumns."""
        fin = make_profiles(npoints=1, nlev=nlev, seed=3)
        lev = int(np.argmin(np.abs(fin["pfull"][0] - p_cloud)))
        dtau = np.zeros_like(fin["dtau"][:, :ncol, :])
        dem = np.zeros_like(dtau)
        frac = np.zeros_like(dtau)
        for b in cloudy_boxes:
            dtau[0, b, lev] = dtau_val
            dem[0, b, lev] = (1.0 - np.exp(-dtau_val / 2.13)
                              if dem_val is None else dem_val)
            frac[0, b, lev] = 1.0
        fin.update(dtau=dtau, dem=dem, frac_out=frac,
                   sunlit=np.ones(1, int))
        return fin, lev

    def test_opaque_low_cloud_ptop_and_category(self):
        # An opaque warm cloud near 900 hPa: ptop lands near the cloud
        # pressure and the histogram mass in the thick/low cell.
        fin, lev = self._single_cloud(90000.0, 18.0)
        out = simulate_isccp(
            to_port(fin["frac_out"]), to_port(fin["dtau"]),
            to_port(fin["dem"]), fin["skt"], fin["emsfc_lw"],
            to_port(fin["at"]), to_port(fin["pfull"]), to_port(fin["phalf"]),
            to_port(fin["qv"]), sunlit=fin["sunlit"])
        ptop = np.asarray(out.subcolumn.boxptop)[1:, 0]
        self.assertTrue((np.abs(ptop - fin["pfull"][0, lev]) < 7000.0).all(),
                        msg=f"ptop={ptop}, cloud p={fin['pfull'][0, lev]}")
        fq = np.asarray(out.column.fq_isccp)[..., 0]
        # tau = 18 -> bin [9.4, 23); p ~ 900 hPa -> (800, 10000] hPa, which
        # is index 0 after the surface-first flip of the pressure axis.
        self.assertEqual(fq[4, 0], 75.0)
        self.assertAlmostEqual(fq.sum(), 75.0)
        self.assertAlmostEqual(float(np.asarray(out.column.totalcldarea)[0]),
                               75.0)
        # Mean-tau round trip through the albedo fit is exact for a
        # homogeneous cloudy area.
        self.assertAlmostEqual(float(np.asarray(out.column.meantaucld)[0]),
                               18.0, places=8)
        self.assertAlmostEqual(float(np.asarray(out.column.meanptop)[0]),
                               float(ptop[0]), delta=1e-3)

    def test_thin_cirrus_ir_adjustment(self):
        # A thin cirrus with consistent 11-um emissivity: with the ISCCP
        # adjustment (top_height=1) the retrieved top recovers the cloud
        # level; without it (top_height=3) the blended radiance places the
        # cloud too low (warmer Tb -> higher pressure).
        fin, lev = self._single_cloud(25000.0, 0.8)
        args = (to_port(fin["frac_out"]), to_port(fin["dtau"]),
                to_port(fin["dem"]), fin["skt"], fin["emsfc_lw"],
                to_port(fin["at"]), to_port(fin["pfull"]),
                to_port(fin["phalf"]), to_port(fin["qv"]))
        p_cloud = fin["pfull"][0, lev]
        out1 = simulate_isccp(*args, sunlit=fin["sunlit"], top_height=1)
        out3 = simulate_isccp(*args, sunlit=fin["sunlit"], top_height=3)
        ptop1 = float(np.asarray(out1.subcolumn.boxptop)[1, 0])
        ptop3 = float(np.asarray(out3.subcolumn.boxptop)[1, 0])
        self.assertGreater(ptop3, ptop1 + 5000.0)
        self.assertLess(abs(ptop1 - p_cloud), 6000.0)
        # And the physical top (top_height=2) sits at the cloud layer edge.
        out2 = simulate_isccp(*args, sunlit=fin["sunlit"], top_height=2)
        ptop2 = float(np.asarray(out2.subcolumn.boxptop)[1, 0])
        self.assertAlmostEqual(ptop2, fin["phalf"][0, lev], places=6)

    def test_clear_sky_tb_warmer_than_cold_cloud(self):
        fin, _ = self._single_cloud(30000.0, 15.0)
        out = simulate_isccp(
            to_port(fin["frac_out"]), to_port(fin["dtau"]),
            to_port(fin["dem"]), fin["skt"], fin["emsfc_lw"],
            to_port(fin["at"]), to_port(fin["pfull"]), to_port(fin["phalf"]),
            to_port(fin["qv"]), sunlit=fin["sunlit"])
        tbclr = float(np.asarray(out.subcolumn.meantbclr)[0])
        ttop = np.asarray(out.subcolumn.boxttop)[1:, 0]
        self.assertTrue((tbclr > ttop + 20.0).all(),
                        msg=f"tbclr={tbclr}, cloudy Tb={ttop}")
        # Clear subcolumn Tb equals the clear-sky Tb.
        self.assertAlmostEqual(float(np.asarray(out.subcolumn.boxttop)[0, 0]),
                               tbclr, places=9)

    def test_histogram_sums_to_totalcldarea(self):
        # Bins with tau > 0.3 hold exactly the totalcldarea mass; the full
        # histogram additionally counts the detected tau <= 0.3 boxes.
        fin = make_profiles(seed=5)
        out = simulate_isccp(
            to_port(fin["frac_out"]), to_port(fin["dtau"]),
            to_port(fin["dem"]), fin["skt"], fin["emsfc_lw"],
            to_port(fin["at"]), to_port(fin["pfull"]), to_port(fin["phalf"]),
            to_port(fin["qv"]), sunlit=np.ones_like(fin["sunlit"]))
        fq = np.asarray(out.column.fq_isccp)
        tca = np.asarray(out.column.totalcldarea)
        np.testing.assert_allclose(fq[1:].sum(axis=(0, 1)), tca, rtol=1e-12)
        self.assertTrue((fq.sum(axis=(0, 1)) >= tca - 1e-12).all())

    def test_full_cover_is_exact_hundred(self):
        # cosp.F90 carries a stale snap-to-1 clamp for near-total cover that
        # predates the percent conversion (see translation notes, "Fixed
        # upstream bugs"); the port needs no clamp — full cover is exact.
        fin, _ = self._single_cloud(90000.0, 18.0, ncol=4,
                                    cloudy_boxes=(0, 1, 2, 3))
        out = simulate_isccp(
            to_port(fin["frac_out"]), to_port(fin["dtau"]),
            to_port(fin["dem"]), fin["skt"], fin["emsfc_lw"],
            to_port(fin["at"]), to_port(fin["pfull"]), to_port(fin["phalf"]),
            to_port(fin["qv"]), sunlit=fin["sunlit"])
        self.assertEqual(float(np.asarray(out.column.totalcldarea)[0]), 100.0)

    def test_night_points(self):
        fin = make_profiles(seed=2)
        night = np.zeros_like(fin["sunlit"])
        out = simulate_isccp(
            to_port(fin["frac_out"]), to_port(fin["dtau"]),
            to_port(fin["dem"]), fin["skt"], fin["emsfc_lw"],
            to_port(fin["at"]), to_port(fin["pfull"]), to_port(fin["phalf"]),
            to_port(fin["qv"]), sunlit=night, top_height=1)
        # Box values and daytime-only column stats are undefined at night...
        self.assertTrue((np.asarray(out.subcolumn.boxtau) == R_UNDEF).all())
        self.assertTrue((np.asarray(out.subcolumn.boxptop) == R_UNDEF).all())
        for name in ("fq_isccp", "totalcldarea", "meanptop", "meantaucld",
                     "meanalbedocld"):
            self.assertTrue(
                (np.asarray(getattr(out.column, name)) == R_UNDEF).all(),
                msg=name)
        # ...but the 10.5-um brightness temperatures remain valid.
        self.assertTrue((np.asarray(out.column.meantb) > 150.0).all())
        self.assertTrue((np.asarray(out.subcolumn.meantbclr) > 150.0).all())
        # top_height=3 reports night points like day points.
        out3 = simulate_isccp(
            to_port(fin["frac_out"]), to_port(fin["dtau"]),
            to_port(fin["dem"]), fin["skt"], fin["emsfc_lw"],
            to_port(fin["at"]), to_port(fin["pfull"]), to_port(fin["phalf"]),
            to_port(fin["qv"]), sunlit=night, top_height=3)
        self.assertTrue((np.asarray(out3.column.totalcldarea) >= 0.0).all())

    def test_top_height_direction(self):
        # A temperature profile that crosses the cloud brightness
        # temperature three times (a cold layer under an inversion around
        # 700-850 hPa): direction 2 keeps the highest-altitude match,
        # direction 1 the lowest.
        pfull = np.array([[5000.0, 10000.0, 20000.0, 30000.0, 40000.0,
                           50000.0, 60000.0, 70000.0, 75000.0, 85000.0,
                           95000.0, 101000.0]])
        phalf = np.array([[2500.0, 7500.0, 15000.0, 25000.0, 35000.0,
                           45000.0, 55000.0, 65000.0, 72500.0, 80000.0,
                           90000.0, 98000.0, 101325.0]])
        atm = np.array([[230.0, 220.0, 225.0, 240.0, 255.0, 265.0, 275.0,
                         268.0, 262.0, 275.0, 283.0, 288.0]])
        qv = np.full_like(pfull, 1e-6)  # negligible continuum
        nlev = pfull.shape[1]
        # Opaque cloud at 500 hPa (T = 265 K): Tb ~ 265 K also brackets in
        # the 268->262 dip and the 262->275 recovery below it.
        dtau = np.zeros((1, 2, nlev))
        dem = np.zeros_like(dtau)
        dtau[0, 1, 5] = 20.0
        dem[0, 1, 5] = 0.99
        frac = (dtau > 0).astype(float)
        args = (to_port(frac), to_port(dtau), to_port(dem),
                np.array([288.0]), 0.98, to_port(atm), to_port(pfull),
                to_port(phalf), to_port(qv))
        p2 = float(np.asarray(simulate_isccp(
            *args, sunlit=np.ones(1), top_height_direction=2
        ).subcolumn.boxptop)[1, 0])
        p1 = float(np.asarray(simulate_isccp(
            *args, sunlit=np.ones(1), top_height_direction=1
        ).subcolumn.boxptop)[1, 0])
        self.assertLess(p2, 60000.0,
                        msg=f"direction 2 ({p2}) should match near the cloud")
        self.assertGreater(p1, 70000.0,
                        msg=f"direction 1 ({p1}) should match in the dip")


class IsccpGradientTest(unittest.TestCase):
    """The radiance/Tb/ptop stack must carry finite gradients.

    Differentiable: radiances, Tb, the IR adjustment, the log-p
    interpolation of ptop and the albedo/mean-tau fits. Step-like (zero
    gradient a.e., jumps at thresholds): the level-match selection, the
    tau > tauchk / taumin overrides, histogram counts and cloud areas.
    """

    def test_grad_finite_through_radiances_and_ptop(self):
        fin = make_profiles(npoints=2, seed=7)
        dtau0 = jnp.asarray(to_port(fin["dtau"]))

        def objective(dtau):
            sub = isccp_subcolumn(
                dtau, to_port(fin["dem"]), to_port(fin["frac_out"]),
                fin["skt"], fin["emsfc_lw"], to_port(fin["qv"]),
                to_port(fin["at"]), to_port(fin["pfull"]),
                to_port(fin["phalf"]), sunlit=fin["sunlit"])
            col = isccp_column(sub.boxtau, sub.boxptop / 100.0,
                               fin["sunlit"], sub.boxttop)
            ptop_sum = jnp.sum(jnp.where(sub.boxptop > 0.0,
                                         sub.boxptop, 0.0))
            tau_sum = jnp.sum(jnp.where(sub.boxtau > 0.0, sub.boxtau, 0.0))
            taucld = jnp.sum(jnp.where(col.meantaucld > 0.0,
                                       col.meantaucld, 0.0))
            return (jnp.sum(sub.boxttop) + jnp.sum(sub.meantbclr)
                    + 1e-4 * ptop_sum + tau_sum + taucld)

        val, grad = jax.value_and_grad(objective)(dtau0)
        self.assertTrue(np.isfinite(float(val)))
        g = np.asarray(grad)
        self.assertTrue(np.isfinite(g).all(), "gradient contains NaN/inf")
        self.assertGreater(np.abs(g).max(), 0.0, "gradient identically zero")

    def test_grad_finite_wrt_atmosphere(self):
        fin = make_profiles(npoints=2, seed=8)

        def tbclr(qv, at):
            sub = isccp_subcolumn(
                to_port(fin["dtau"]), to_port(fin["dem"]),
                to_port(fin["frac_out"]), fin["skt"], fin["emsfc_lw"],
                qv, at, to_port(fin["pfull"]), to_port(fin["phalf"]),
                sunlit=fin["sunlit"])
            return jnp.sum(sub.meantbclr) + jnp.sum(sub.boxttop)

        gq, ga = jax.grad(tbclr, argnums=(0, 1))(
            jnp.asarray(to_port(fin["qv"])), jnp.asarray(to_port(fin["at"])))
        for g in (np.asarray(gq), np.asarray(ga)):
            self.assertTrue(np.isfinite(g).all())
            self.assertGreater(np.abs(g).max(), 0.0)

    def test_jit(self):
        fin = make_profiles(npoints=2, seed=9)
        f = jax.jit(lambda dtau: simulate_isccp(
            to_port(fin["frac_out"]), dtau, to_port(fin["dem"]),
            fin["skt"], fin["emsfc_lw"], to_port(fin["at"]),
            to_port(fin["pfull"]), to_port(fin["phalf"]),
            to_port(fin["qv"]), sunlit=fin["sunlit"]).column.totalcldarea)
        out = np.asarray(f(jnp.asarray(to_port(fin["dtau"]))))
        self.assertEqual(out.shape, (2,))


class WaterVapourEmissivityTest(unittest.TestCase):
    def test_monotone_in_humidity_and_bounded(self):
        fin = make_profiles(npoints=1, seed=4)
        args = (to_port(fin["at"]), to_port(fin["pfull"]),
                to_port(fin["phalf"]))
        e1 = np.asarray(water_vapour_emissivity(to_port(fin["qv"]), *args))
        e2 = np.asarray(water_vapour_emissivity(2 * to_port(fin["qv"]),
                                                *args))
        self.assertTrue((e1 >= 0.0).all() and (e1 < 1.0).all())
        self.assertTrue((e2 >= e1).all())


if __name__ == "__main__":
    unittest.main()
