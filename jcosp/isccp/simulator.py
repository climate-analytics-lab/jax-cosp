"""ISCCP simulator (ICARUS): 10.5-um radiances, IR-adjusted cloud tops, stats.

Port of COSP v2.0 ``src/simulator/icarus/icarus.F90`` (``ICARUS_SUBCOLUMN``
and ``ICARUS_COLUMN``; the ISCCP simulator of Klein and Jakob (1999) and
Webb et al. (2001), modified for COSPv2.0 by D. Swales, May 2015). The
original is Copyright (c) 2009, Lawrence Livermore National Security LLC
(source-file header) and (c) British Crown Copyright 2009, the Met Office
(the ``icarus/license`` file), both BSD-3-Clause; see LICENSES/ and NOTICE.

Subcolumn stage (:func:`isccp_subcolumn`): per-subcolumn 0.67-um optical
depth, the clear and all-sky 10.5-um radiance stacks (with the Schwarzkopf
and Ramaswamy (1999) water-vapour continuum emissivity at 955 cm-1),
brightness temperatures, and the cloud-top pressure — either radiatively
(``top_height`` 1 with the ISCCP D-series partial-emissivity adjustment of
Rossow et al. (1996, pp. 85-87), or 3 without it) by matching the cloud
brightness temperature to the sounding, or physically (``top_height`` 2)
from the highest subcolumn cloud. Column stage (:func:`isccp_column`):
total cloud area, cloudy-mean cloud-top pressure / albedo / optical
thickness, mean all-sky and clear-sky brightness temperatures and the
7 x 7 tau/CTP joint histogram of the 49 ISCCP weather states.

Conventions and reproduced quirks (see docs/translation_notes.md):

* ``top_height_direction`` selects which temperature match wins when the
  sounding crosses the cloud brightness temperature more than once:
  2 (ISCCP-recommended default) keeps the highest-altitude match, 1 the
  lowest.
* ``boxptop`` from the subcolumn stage is in Pa (the Fortran comment says
  mb, but cosp.F90 divides by 100 before ICARUS_COLUMN and stores the Pa
  values); ``meanptop`` is likewise returned in Pa (the ``100 *`` in the
  Fortran restores hPa means to Pa). Cloud areas and the joint histogram
  are in percent.
* ``levmatch`` is returned as a 0-based level index with -1 for "no
  match", replacing the Fortran's 1-based index with 0 sentinel.
* The interpolation floor ``atd = max(tauchk, |dT|)`` reuses the optical
  -depth constant as a temperature difference floor, as in the Fortran.
* For ``top_height`` 2 the Fortran never assigns ``boxttop``/``meantbclr``
  (intent(inout), read back uninitialized memory upstream); the port
  returns R_UNDEF for both.

Array convention: level axis 0 TOA-first, subcolumn axis 1, batch trailing.
The Fortran's (npoints, ncol, nlev) becomes (nlev, ncol, *batch).
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from jcosp import config, stats, surrogate
from jcosp.constants import R_UNDEF


def _planck_rad(t):
    """Return the normalized 10.5-um Planck radiance at temperature ``t`` (K)."""
    return 1.0 / (jnp.exp(config.ISCCP_PLANCK_K / t) - 1.0)


def _planck_tb(rad):
    """Brightness temperature (K) of a normalized 10.5-um radiance."""
    return config.ISCCP_PLANCK_K / jnp.log(1.0 + 1.0 / rad)


def water_vapour_emissivity(qv, at, pfull, phalf):
    """Layer 10.5-um water-vapour continuum emissivity (icarus.F90 'dem_wv').

    Follows Schwarzkopf and Ramaswamy (JGR 1999), evaluated at 955 cm-1
    (10.47 um), exactly as in ICARUS_SUBCOLUMN.

    Parameters
    ----------
    qv : (nlev, *batch)
        Water-vapour specific humidity (kg/kg).
    at : (nlev, *batch)
        Layer temperature (K).
    pfull : (nlev, *batch)
        Layer pressure (Pa).
    phalf : (nlev+1, *batch)
        Layer-edge pressure (Pa), TOA-first.

    Returns
    -------
    (nlev, *batch) layer emissivity.
    """
    press = pfull * 10.0                            # dyn cm-2
    dpress = (phalf[1:] - phalf[:-1]) * 10.0
    atmden = dpress / (config.ISCCP_GRAV * 100.0)   # column mass (g cm-2)
    rvh20 = qv * config.ISCCP_AMD / config.ISCCP_AMW  # volume mixing ratio
    wk = rvh20 * config.ISCCP_AVO * atmden / config.ISCCP_AMD  # molec cm-2
    rhoave = (press / config.ISCCP_PSTD) * (config.ISCCP_T0 / at)
    rh20s = rvh20 * rhoave
    rfrgn = rhoave - rh20s
    tmpexp = jnp.exp(-0.02 * (at - config.ISCCP_T0))
    tauwv = wk * 1.0e-20 * ((0.0224697 * rh20s * tmpexp)
                            + (3.41817e-7 * rfrgn)) * 0.98
    return 1.0 - jnp.exp(-tauwv)


def _radiance_stack(dem, bb, skt_bb, emsfc_lw):
    """TOA radiance of an emitting/absorbing layer stack over a surface.

    Accumulates top-down exactly as the Fortran loop: each layer adds
    ``dem * bb`` attenuated by the transmittance of all layers above, and
    the surface adds ``emsfc_lw * skt_bb`` through the whole column.
    ``dem`` is (nlev, ...); ``bb`` broadcasts against it level by level.
    """
    def step(carry, x):
        trans_above, flux = carry
        d, b = x
        return (trans_above * (1.0 - d), flux + d * b * trans_above), None

    shape = jnp.broadcast_shapes(dem.shape, bb.shape)[1:]
    init = (jnp.ones(shape, dem.dtype), jnp.zeros(shape, dem.dtype))
    (trans_sfc, flux), _ = lax.scan(step, init, (dem, bb))
    return flux + emsfc_lw * skt_bb * trans_sfc


def _take_level(x, idx):
    """Gather per-(subcolumn, batch) level indices from an (nlev, *batch) field."""
    xb = jnp.broadcast_to(x[:, None], (x.shape[0],) + idx.shape)
    return jnp.take_along_axis(xb, idx[None], axis=0)[0]


class IsccpSubcolumn(NamedTuple):
    """Per-subcolumn ICARUS outputs (ICARUS_SUBCOLUMN), (ncol, *batch)."""

    boxtau: jnp.ndarray    # 0.67-um optical thickness (R_UNDEF: clear/night)
    boxptop: jnp.ndarray   # Cloud-top pressure (Pa; R_UNDEF: clear/night)
    boxttop: jnp.ndarray   # All-sky 10.5-um brightness temperature (K)
    meantbclr: jnp.ndarray  # (*batch) clear-sky brightness temperature (K)
    levmatch: jnp.ndarray  # int32 0-based matched level (-1: none); testing aid


def isccp_subcolumn(dtau, dem, frac_out, skt, emsfc_lw, qv, at, pfull, phalf,
                    sunlit=None, top_height: int = 1,
                    top_height_direction: int = 2) -> IsccpSubcolumn:
    """Per-subcolumn ISCCP tau, brightness temperature and cloud-top pressure.

    Port of icarus.F90 ``ICARUS_SUBCOLUMN``.

    Parameters
    ----------
    dtau, dem : (nlev, ncol, *batch)
        Subcolumn 0.67-um layer optical thickness and 10.5-um layer cloud
        emissivity (``cospIN%tau_067`` / ``cospIN%emiss_11``, e.g. from
        ``jcosp.modis.optics.simulator_optics``).
    frac_out : (nlev, ncol, *batch)
        SCOPS subcolumn cloud-type map (read only for ``top_height=2``).
    skt : (*batch)
        Skin temperature (K).
    emsfc_lw : scalar
        10.5-um surface emissivity.
    qv, at, pfull : (nlev, *batch)
        Specific humidity (kg/kg), temperature (K), layer pressure (Pa).
    phalf : (nlev+1, *batch)
        Layer-edge pressure (Pa), TOA-first.
    sunlit : (*batch), optional
        Day mask (> 0 sunlit; None means sunlit everywhere). Night points
        report R_UNDEF ``boxtau``/``boxptop`` unless ``top_height=3``.
    top_height : {1, 2, 3}
        1: IR-adjusted radiative cloud top (ISCCP-like, default);
        2: physical cloud top (no radiances); 3: radiative without the
        partial-emissivity adjustment (also reports night points).
    top_height_direction : {1, 2}
        Temperature-match selection, see module docstring. Default 2.
    """
    if top_height not in (1, 2, 3):
        raise ValueError(f"top_height must be 1, 2 or 3, got {top_height}")
    if top_height_direction not in (1, 2):
        raise ValueError("top_height_direction must be 1 or 2, "
                         f"got {top_height_direction}")

    dtau = jnp.asarray(dtau)
    dtype = dtau.dtype
    nlev = dtau.shape[0]
    at = jnp.asarray(at, dtype)
    pfull = jnp.asarray(pfull, dtype)
    skt = jnp.asarray(skt, dtype)
    tauchk = config.ISCCP_TAUCHK

    # Column optical depth from the subcolumn layers.
    tau = jnp.sum(dtau, axis=0)                              # (ncol, *batch)

    batch_shape = tau.shape[1:]
    radiative = top_height in (1, 3)

    if radiative:
        # --- Tropopause: coldest level with 50 < p < 400 hPa (defaults
        # ptrop = 50 hPa, attrop = 120 K, itrop = top level when the window
        # is empty). The Fortran's sequential strict-< update keeps the
        # first occurrence of the minimum, as argmin does.
        window = (pfull < 40000.0) & (pfull > 5000.0)
        at_win = jnp.where(window, at, jnp.inf)
        has_win = jnp.any(window, axis=0)
        itrop = jnp.where(has_win, jnp.argmin(at_win, axis=0), 0)
        attrop = jnp.where(has_win, jnp.min(at_win, axis=0), 120.0)
        ptrop = jnp.where(
            has_win, jnp.take_along_axis(pfull, itrop[None], axis=0)[0],
            5000.0)
        # Warmest temperature at or below the tropopause (0 K floor).
        levels = jnp.arange(nlev).reshape((-1,) + (1,) * itrop.ndim)
        atmax = jnp.max(jnp.where(levels >= itrop[None], at, 0.0), axis=0)

        # --- Clear-sky 10.5-um radiance through the water-vapour continuum.
        dem_wv = water_vapour_emissivity(qv, at, pfull, phalf)
        bb = _planck_rad(at)
        skt_bb = _planck_rad(skt)
        fluxtop_clrsky = _radiance_stack(dem_wv, bb, skt_bb, emsfc_lw)
        meantbclr = _planck_tb(fluxtop_clrsky)

        # --- All-sky radiance: cloud emissivity combined with the vapour
        # continuum (independent absorbers).
        demc = jnp.where(dem == 0.0, dem_wv[:, None],
                         1.0 - (1.0 - dem) * (1.0 - dem_wv[:, None]))
        fluxtop = _radiance_stack(demc, bb[:, None], skt_bb[None], emsfc_lw)
        boxttop = _planck_tb(fluxtop)

        # --- ISCCP cloud-top temperature: adjust for partially transmitting
        # cloud, recomputing the flux ISCCP would see for a single-layer
        # cloud. tau_vis/tau_IR = 2.13 (ice); 2.56 for warm (Tb > 260 K,
        # liquid) clouds. Rossow et al. (1996), pp. 85-87.
        btcmin = _planck_rad(attrop - 5.0)
        transmax = (fluxtop - btcmin[None]) / (fluxtop_clrsky - btcmin)[None]
        tauir = tau / 2.13
        taumin = -jnp.log(jnp.clip(transmax, 0.001, 0.9999999))

        if top_height == 1:
            inrange = (transmax > 0.001) & (transmax <= 0.9999999)
            cond = (tau > tauchk) & inrange
            fluxtopinit = fluxtop
            for _ in range(2):
                emcld = 1.0 - jnp.exp(-tauir)
                ft = fluxtopinit - (1.0 - emcld) * fluxtop_clrsky[None]
                # The clamps only harden the gradients of the masked-out
                # side: under `cond`, tau > tauchk implies emcld > 0 and
                # ft >= 1e-6 by the Fortran's own max().
                ft = jnp.maximum(1.0e-6, ft / jnp.maximum(emcld, 1e-30))
                tb_cycle = _planck_tb(ft)
                fluxtop = jnp.where(cond, ft, fluxtop)
                tauir = jnp.where(cond & (tb_cycle > 260.0),
                                  tau / 2.56, tauir)

        # Cloud-top temperature of detectable boxes; clear boxes carry the
        # clear-sky brightness temperature into the level match below.
        tb = jnp.where(tau > tauchk, _planck_tb(fluxtop), meantbclr[None])
        if top_height == 1:
            # Optically-thin override: radiance implies more transmission
            # than the IR tau allows -> tropopause-adjacent cold top, and
            # the reported tau becomes the radiance-implied visible value.
            over = (tau > tauchk) & (tauir < taumin)
            tb = jnp.where(over, (attrop - 5.0)[None], tb)
            tau = jnp.where(over, 2.13 * taumin, tau)

        # --- Cloud-top pressure: level whose temperature brackets tb,
        # searched below the tropopause; direction picks among multiple
        # matches (2: highest altitude, 1: lowest).
        at_up = at[:-1][:, None]                    # (nlev-1, 1, *batch)
        at_dn = at[1:][:, None]
        bracket = (((at_up >= tb[None]) & (at_dn <= tb[None]))
                   | ((at_up <= tb[None]) & (at_dn >= tb[None])))
        lev = jnp.arange(nlev - 1).reshape((-1,) + (1,) * tb.ndim)
        bracket = bracket & (lev >= itrop[None, None])
        any_match = jnp.any(bracket, axis=0)
        if top_height_direction == 2:
            k1 = jnp.argmax(bracket, axis=0)
        else:
            k1 = (nlev - 2) - jnp.argmax(bracket[::-1], axis=0)
        k2 = k1 + 1
        at_k1 = _take_level(at, k1)
        at_k2 = _take_level(at, k2)
        p_k1 = _take_level(pfull, k1)
        p_k2 = _take_level(pfull, k2)
        logp1 = jnp.log(p_k1)
        logp2 = jnp.log(p_k2)
        # tauchk doubles as the temperature-difference floor (Fortran quirk).
        atd = jnp.maximum(tauchk, jnp.abs(at_k2 - at_k1))
        logp = logp1 + (logp2 - logp1) * jnp.abs(tb - at_k1) / atd
        ptop_match = jnp.exp(logp)
        lev_match = jnp.where(
            jnp.abs(p_k1 - ptop_match) < jnp.abs(p_k2 - ptop_match), k1, k2)

        # No-match fallbacks (sequential, the second overriding the first).
        ptop_fb = jnp.zeros_like(tb)
        lev_fb = jnp.full(tb.shape, -1, jnp.int32)
        colder = tb <= attrop[None]
        ptop_fb = jnp.where(colder, ptrop[None], ptop_fb)
        lev_fb = jnp.where(colder, itrop[None].astype(jnp.int32), lev_fb)
        warmer = tb >= atmax[None]
        ptop_fb = jnp.where(warmer, pfull[-1][None], ptop_fb)
        lev_fb = jnp.where(warmer, nlev - 1, lev_fb)

        ptop = jnp.where(any_match, ptop_match, ptop_fb)
        levmatch = jnp.where(any_match, lev_match, lev_fb).astype(jnp.int32)
    else:
        # top_height == 2: physical cloud top — upper interface of the
        # highest cloudy layer of the subcolumn. No radiances are computed;
        # the Fortran leaves boxttop/meantbclr uninitialized (R_UNDEF here).
        meantbclr = jnp.full(batch_shape, R_UNDEF, dtype)
        boxttop = jnp.full(tau.shape, R_UNDEF, dtype)
        cloudy = jnp.asarray(frac_out) > 0.0
        has_cloud = jnp.any(cloudy, axis=0)
        first = jnp.argmax(cloudy, axis=0)
        ptop = jnp.where(has_cloud,
                         _take_level(jnp.asarray(phalf, dtype), first), 0.0)
        levmatch = jnp.where(has_cloud, first, -1).astype(jnp.int32)

    clear = tau <= tauchk
    ptop = jnp.where(clear, 0.0, ptop)
    levmatch = jnp.where(clear, -1, levmatch)

    # Report only detected cloud tops; night points only for top_height 3.
    detected = (tau > tauchk) & (ptop > 0.0)
    if top_height != 3:
        day = (jnp.ones(batch_shape, bool) if sunlit is None
               else jnp.asarray(sunlit) > 0)
        detected = detected & day[None]
    boxtau = jnp.where(detected, tau, R_UNDEF)
    boxptop = jnp.where(detected, ptop, R_UNDEF)

    return IsccpSubcolumn(boxtau=boxtau, boxptop=boxptop, boxttop=boxttop,
                          meantbclr=meantbclr, levmatch=levmatch)


class IsccpColumn(NamedTuple):
    """Column-stage ISCCP diagnostics (ICARUS_COLUMN outputs).

    Cloud areas and the joint histogram are in percent of subcolumns;
    ``meanptop`` is in Pa; means are over the cloudy (tau > 0.3) area only.
    The joint-histogram CTP axis is in ascending-pressure (TOA-first) bin
    order here; the ``simulate_isccp`` driver flips it to surface-first as
    cosp.F90 does on output.
    """

    fq_isccp: jnp.ndarray       # (ntau, npres, *batch) cloud-type freq. (%)
    totalcldarea: jnp.ndarray   # (*batch) cloud area, tau > 0.3 (%)
    meanptop: jnp.ndarray       # (*batch) cloudy-mean cloud-top pressure (Pa)
    meantaucld: jnp.ndarray     # (*batch) cloudy-mean optical thickness
    meanalbedocld: jnp.ndarray  # (*batch) cloudy-mean visible albedo
    meantb: jnp.ndarray         # (*batch) all-sky brightness temperature (K)


def isccp_column(boxtau, boxptop, sunlit, boxttop,
                 top_height: int = 1, weights=None,
                 smoothing=surrogate.DEFAULT_SMOOTHING) -> IsccpColumn:
    """Aggregate subcolumn ISCCP retrievals to column statistics.

    Port of icarus.F90 ``ICARUS_COLUMN``. As at the Fortran call sites,
    ``boxptop`` is in **hPa** here (the drivers pass ``boxptop / 100``);
    ``meanptop`` comes back in Pa (the Fortran's ``100 *``). ``sunlit`` may
    be None (all points sunlit). ``meantb`` is reported for night points
    too (the 10.5-um stack does not need daylight); everything else is
    R_UNDEF at night unless ``top_height = 3``.

    ``weights`` is the (ncol, *batch) SCOPS subcolumn weight
    (``jcosp.subcol.scops.subcolumn_weights``), through which the
    statistics respond to the cloud fraction. ``smoothing`` gives the
    cloud-area limit (tau = 0.3) and the histogram bin edges the
    derivatives of smooth steps; no value depends on either.
    """
    dtype = boxtau.dtype
    ncol = boxtau.shape[0]
    batch_shape = boxtau.shape[1:]
    tauchk = config.ISCCP_TAUCHK

    one = jnp.ones((), dtype) if weights is None else weights
    radiative = top_height in (1, 3)
    if radiative:
        meantb = jnp.sum(one * boxttop, axis=0) / ncol
    else:
        meantb = jnp.full(batch_shape, R_UNDEF, dtype)

    if top_height == 3:
        active = jnp.ones(batch_shape, bool)
    else:
        active = (jnp.ones(batch_shape, bool) if sunlit is None
                  else jnp.asarray(sunlit) > 0)

    # Undetected boxes hold R_UNDEF, far below the limit: their step is 0
    # with no derivative. Detected ones cross the limit smoothly.
    box_cloudy = (boxtau > tauchk) & (boxptop > 0.0)
    thick = box_cloudy * surrogate.gt(boxtau, config.ISCCP_TAUMIN,
                                      smoothing.optical_depth)

    fq = stats.hist2d(boxtau, boxptop,
                      jnp.asarray(config.ISCCP_HIST_TAU, dtype),
                      jnp.asarray(config.ISCCP_HIST_PRES, dtype),
                      weights=weights, width=smoothing.histogram) / ncol

    totalcldarea = jnp.sum(one * thick, axis=0).astype(dtype) / ncol
    # Visible albedo fit alpha(tau) = tau^0.895 / (tau^0.895 + 6.82); the
    # clamp keeps the masked-out cells (boxtau = R_UNDEF) out of the
    # fractional-power gradient graph.
    tau_safe = jnp.maximum(boxtau, config.ISCCP_TAUMIN) ** 0.895
    albedocld = surrogate.gated(thick, tau_safe / (tau_safe + 6.82))
    meanalbedocld = jnp.sum(one * albedocld, axis=0) / ncol
    # R_UNDEF tops belong to undetected boxes, which the step leaves out
    # with no derivative; zero them so the fill cannot scale one.
    ptop_known = jnp.where(box_cloudy, boxptop, 0.0)
    meanptop = jnp.sum(one * surrogate.gated(thick, ptop_known), axis=0) / ncol

    # Cloudy-area means; totalcldarea is still the 0-1 fraction here. The
    # mean albedo converts back to a mean optical thickness through the
    # inverse of the fit above.
    have_cloud = totalcldarea > 0.0
    area_safe = jnp.maximum(totalcldarea, 1.0 / (2 * ncol))
    alb = meanalbedocld / area_safe
    # Cloud-free points (alb = 0, masked below) take a mid-range value: the
    # derivative of the inverse fit at the 1e-30 clamp overflows float32
    # and would turn the masked branch's zero derivative into NaN.
    alb_safe = jnp.clip(jnp.where(have_cloud, alb, 0.5), 1e-30, 1.0 - 1e-12)
    meanptop = jnp.where(have_cloud, 100.0 * meanptop / area_safe, R_UNDEF)
    meanalbedocld = jnp.where(have_cloud, alb, R_UNDEF)
    meantaucld = jnp.where(
        have_cloud, (6.82 / ((1.0 / alb_safe) - 1.0)) ** (1.0 / 0.895),
        R_UNDEF)

    # Percent, and night masking (the Fortran initializes night points to
    # missing and never enters the sunlit-only block).
    fq = jnp.where(active[None, None], 100.0 * fq, R_UNDEF)
    totalcldarea = jnp.where(active, 100.0 * totalcldarea, R_UNDEF)
    meanptop = jnp.where(active, meanptop, R_UNDEF)
    meanalbedocld = jnp.where(active, meanalbedocld, R_UNDEF)
    meantaucld = jnp.where(active, meantaucld, R_UNDEF)

    return IsccpColumn(fq_isccp=fq, totalcldarea=totalcldarea,
                       meanptop=meanptop, meantaucld=meantaucld,
                       meanalbedocld=meanalbedocld, meantb=meantb)
