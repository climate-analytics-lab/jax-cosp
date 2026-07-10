"""MODIS simulator: subcolumn retrievals and column-mean/histogram stage.

Port of COSP v2.0 ``src/simulator/MODIS_simulator/modis_simulator.F90``
(R. Pincus, S. Platnick 2009; D. Swales 2015 for COSPv2.0; Copyright (c)
2015, Regents of the University of Colorado, BSD-3-Clause; see LICENSES/).

Subcolumn stage (``modis_subcolumn``): per-subcolumn cloud detection
(tau >= 0.3), CO2-slicing cloud-top pressure (extinction-weighted pressure
down to tau = 1), phase from the extinction-weighted liquid fraction, and
particle size by matching the observed 3.7-um TOA reflectance (delta-
Eddington two-stream + adding) against reflectances predicted for 15 trial
radii. Column stage (``modis_column``): phase cloud fractions, in-cloud
means, and the tau/CTP, tau/Reff and LWP-IWP/Reff joint histograms.

Deviations (documented in docs/translation_notes.md):

* The Fortran replaces the CO2-slicing CTP of clouds below 700 hPa with the
  ISCCP simulator's cloud-top pressure. ISCCP is not ported yet, so
  ``isccp_ctp`` is an *optional* input; when absent the CO2-slicing value is
  kept everywhere.
* ``compute_toa_reflectance`` runs the adding recursion over all layers with
  clear layers forced to the identity (R, T) = (0, 1) instead of packing the
  cloudy ones — algebraically identical.
* ``beta`` of the two-stream solution is computed as -(rk - g1)/(rk + g1);
  the Fortran's -r5/r4 carries a common (1 - (rk xmu)^2) factor that cancels
  algebraically (and would be 0/0 at rk = 1/xmu).

Array convention: level axis 0 TOA-first, subcolumn axis next, batch axes
trailing. Retrieval outputs are (ncol, *batch).
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from jcosp import config, stats
from jcosp.constants import R_UNDEF
from jcosp.modis import optics

# Two-stream parameters (modis_simulator.F90; beam = 2 is a compile-time
# constant there, so only the diffuse-beam branch is ported).
_XMU = 0.866
_MIN_CONSERVATIVE_W0 = 0.9999999


def trial_tables(dtype=jnp.float64):
    """Trial radii and their 3.7-um optical properties (COSP_MODIS_INIT).

    Returns ``(trial_re_w, trial_re_i, g_w, w0_w, g_i, w0_i)``, each of
    shape (MODIS_NUM_TRIAL_RES,), radii in microns.
    """
    i = jnp.arange(config.MODIS_NUM_TRIAL_RES, dtype=dtype)
    n1 = config.MODIS_NUM_TRIAL_RES - 1
    trial_re_w = (config.MODIS_RE_WATER_MIN
                  + (config.MODIS_RE_WATER_MAX - config.MODIS_RE_WATER_MIN)
                  / n1 * i)
    trial_re_i = (config.MODIS_RE_ICE_MIN
                  + (config.MODIS_RE_ICE_MAX - config.MODIS_RE_ICE_MIN)
                  / n1 * i)
    g_w = optics.get_g_nir(config.MODIS_PHASE_LIQUID, trial_re_w)
    w0_w = optics.get_ssa_nir(config.MODIS_PHASE_LIQUID, trial_re_w)
    g_i = optics.get_g_nir(config.MODIS_PHASE_ICE, trial_re_i)
    w0_i = optics.get_ssa_nir(config.MODIS_PHASE_ICE, trial_re_i)
    return trial_re_w, trial_re_i, g_w, w0_w, g_i, w0_i


def _delta_eddington(tau_in, g_in, w0_in):
    """Delta-Eddington-scaled (tau, w0, g) and the two-stream gammas/rk/beta."""
    f = g_in**2
    tau = (1.0 - w0_in * f) * tau_in
    w0 = (1.0 - f) * w0_in / (1.0 - w0_in * f)
    g = (g_in - f) / (1.0 - f)
    gamma1 = (7.0 - w0 * (4.0 + 3.0 * g)) / 4.0
    gamma2 = -(1.0 - w0 * (4.0 - 3.0 * g)) / 4.0
    # Clamp under the sqrt: at conservative scattering gamma1 == -gamma2 and
    # the product vanishes, which is fine for the value (the conservative
    # branch is selected by `where`) but would poison the gradient.
    rk = jnp.sqrt(jnp.maximum(gamma1**2 - gamma2**2, 1e-30))
    # -r5/r4 in the Fortran; the common (1 - (rk xmu)^2) factor cancels.
    beta = -(rk - gamma1) / (rk + gamma1)
    return tau, gamma1, gamma2, rk, beta


def two_stream(tau_in, g_in, w0_in):
    """Single-layer reflectance and transmittance (two_stream, beam = 2)."""
    tau, gamma1, gamma2, rk, beta = _delta_eddington(tau_in, g_in, w0_in)
    # Conservative scattering.
    ref_c = gamma1 * tau / (1.0 + gamma1 * tau)
    tra_c = 1.0 - ref_c
    # Non-conservative scattering.
    e1 = jnp.minimum(rk * tau, 500.0)
    ef1 = jnp.exp(-e1)
    ef2 = jnp.exp(-2.0 * e1)
    den = (rk + gamma1) * (1.0 - beta * ef2)
    ref_nc = gamma2 * (1.0 - ef2) / den
    tra_nc = 2.0 * rk * ef1 / den
    conservative = w0_in > _MIN_CONSERVATIVE_W0
    return (jnp.where(conservative, ref_c, ref_nc),
            jnp.where(conservative, tra_c, tra_nc))


def two_stream_reflectance(tau_in, g_in, w0_in):
    """Single-layer reflectance only (two_stream_reflectance, beam = 2)."""
    return two_stream(tau_in, g_in, w0_in)[0]


def adding_doubling(refl, tran):
    """Combine layer (R, T) pairs from the top down (adding_doubling).

    ``refl``/``tran`` are (nlev, ...) stacks; returns the total (R, T).
    """
    def step(carry, x):
        r_cum, t_cum = carry
        r, t = x
        den = 1.0 - r_cum * r
        r_new = r_cum + r * t_cum**2 / den
        t_new = t_cum * t / den
        return (r_new, t_new), None

    init = (jnp.zeros_like(refl[0]), jnp.ones_like(tran[0]))
    (r_tot, t_tot), _ = lax.scan(step, init, (refl, tran))
    return r_tot, t_tot


def compute_toa_reflectance(tau, g, w0):
    """TOA reflectance of the layer stack (compute_toa_reflectace).

    ``tau``/``g``/``w0`` are (nlev, ...) subcolumn profiles. The Fortran
    packs the tau > 0 layers and combines them with adding/doubling; here
    clear layers are forced to the identity (R, T) = (0, 1), which leaves
    the recursion unchanged.
    """
    refl, tran = two_stream(tau, g, w0)
    cloudy = tau > 0.0
    refl = jnp.where(cloudy, refl, 0.0)
    tran = jnp.where(cloudy, tran, 1.0)
    return adding_doubling(refl, tran)[0]


def cloud_top_pressure(tau, pressure_half, tau_limit):
    """Extinction-weighted CO2-slicing cloud-top pressure (cloud_top_pressure).

    ``tau`` is the (nlev, ncol, *batch) layer optical thickness and
    ``pressure_half`` the (nlev+1, *batch) layer-edge pressures, TOA-first.
    Trapezoidal integration of pressure against optical depth down to
    ``tau_limit``, assuming pressure varies linearly across each layer.

    FIXED UPSTREAM BUG (docs/translation_notes.md): the reference call site
    passes an (nLevels+1)-element tau array (0 prepended) into an nLevels
    dummy, so its *bottom layer never contributes* to the integral and a
    cloud confined entirely to the bottom layer yields CTP = 0 (which then
    also escapes the ISCCP low-cloud substitution). Here every layer is
    integrated between its own edges.
    """
    nlev = tau.shape[0]
    p = pressure_half[:, None] if pressure_half.ndim < tau.ndim else pressure_half
    xs = (tau,                                          # tauIncrement(L)
          jnp.broadcast_to(p[:nlev], tau.shape),        # edge L
          jnp.broadcast_to(p[1:nlev + 1], tau.shape))   # edge L+1

    def step(carry, x):
        tot_tau, tot_prod = carry
        inc, p_lo, p_hi = x
        active = tot_tau < tau_limit           # Fortran exits at >= tau_limit
        over = tot_tau + inc > tau_limit
        dx = tau_limit - tot_tau
        # `over & active` implies inc > dx >= 0; the clamp only hardens the
        # gradient of the masked-out side.
        inc_safe = jnp.maximum(inc, 1e-30)
        prod_partial = p_lo * dx + (p_hi - p_lo) * dx**2 / (2.0 * inc_safe)
        prod_full = inc * (p_hi + p_lo) / 2.0
        new_tau = jnp.where(active,
                            jnp.where(over, tot_tau + dx, tot_tau + inc),
                            tot_tau)
        new_prod = jnp.where(active,
                             jnp.where(over, tot_prod + prod_partial,
                                       tot_prod + prod_full),
                             tot_prod)
        return (new_tau, new_prod), None

    shape = tau.shape[1:]
    init = (jnp.zeros(shape, tau.dtype), jnp.zeros(shape, tau.dtype))
    (tot_tau, tot_prod), _ = lax.scan(step, init, xs)
    return jnp.where(tot_tau > 0.0,
                     tot_prod / jnp.maximum(tot_tau, 1e-30), 0.0)


def weight_by_extinction(tau, f, tau_limit):
    """Extinction-weighted mean of ``f`` down to ``tau_limit`` (weight_by_extinction).

    ``tau``/``f`` are (nlev, ...) profiles (f constant within each layer).
    """
    def step(carry, x):
        tot_tau, tot_prod = carry
        inc, f_k = x
        active = tot_tau < tau_limit
        over = tot_tau + inc > tau_limit
        dx = tau_limit - tot_tau
        add_tau = jnp.where(over, dx, inc)
        new_tau = jnp.where(active, tot_tau + add_tau, tot_tau)
        new_prod = jnp.where(active, tot_prod + add_tau * f_k, tot_prod)
        return (new_tau, new_prod), None

    shape = tau.shape[1:]
    init = (jnp.zeros(shape, tau.dtype), jnp.zeros(shape, tau.dtype))
    (tot_tau, tot_prod), _ = lax.scan(step, init, (tau, f))
    return jnp.where(tot_tau > 0.0,
                     tot_prod / jnp.maximum(tot_tau, 1e-30), 0.0)


def interpolate_to_min(x, y, yobs):
    """Invert monotone y(x) for the x where y = yobs (interpolate_to_min).

    ``x`` is (ntrial,), ``y`` (ntrial, ...) and ``yobs`` (...). Finds the
    first minimum of abs(y - yobs) (as Fortran ``minloc``), brackets the root
    with the neighbouring trial point, and interpolates linearly; returns
    MODIS_RE_FILL where the root is not bracketed.
    """
    n = x.shape[0]
    x = jnp.asarray(x)
    diff = y - yobs[None]

    def at(idx):
        return jnp.take_along_axis(diff, idx[None], axis=0)[0]

    minloc = jnp.argmin(jnp.abs(diff), axis=0)
    d_min = at(minloc)
    d_prev = at(jnp.maximum(minloc - 1, 0))
    lower = jnp.where(minloc == 0, 0,
                      jnp.where(minloc == n - 1, n - 2,
                                jnp.where(d_prev * d_min < 0.0,
                                          minloc - 1, minloc)))
    upper = lower + 1
    d_lo = at(lower)
    d_hi = at(upper)
    # FIXED UPSTREAM BUG (docs/translation_notes.md): the Fortran requires a
    # strictly negative bracket product, so a trial point hitting the
    # observed reflectance *exactly* returns the fill value instead of that
    # root. Accept the boundary case (<=); with d_hi = 0 the interpolation
    # returns x_hi exactly, with d_lo = 0 it returns x_lo.
    bracket = d_lo * d_hi <= 0.0
    x_lo = jnp.take(x, lower)
    x_hi = jnp.take(x, upper)
    den = jnp.where(d_hi != d_lo, d_hi - d_lo, 1.0)
    val = x_hi - d_hi * (x_hi - x_lo) / den
    return jnp.where(bracket, val, config.MODIS_RE_FILL)


class ModisRetrievals(NamedTuple):
    """Per-subcolumn MODIS retrievals (modis_subcolumn outputs), (ncol, *batch)."""

    phase: jnp.ndarray  # int32 phase code (0 none / 1 liquid / 2 ice / 3 undet.)
    cloud_top_pressure: jnp.ndarray  # Pa (R_UNDEF where no cloud detected)
    tau: jnp.ndarray                 # 0.67-um optical thickness (R_UNDEF clear)
    size: jnp.ndarray                # particle radius (m); R_UNDEF clear,
    #                                  1e-6 * MODIS_RE_FILL failed retrieval


def modis_subcolumn(pressure_half, tau, liq_frac, g, w0,
                    isccp_ctp=None) -> ModisRetrievals:
    """Per-subcolumn MODIS retrievals (modis_subcolumn).

    Parameters
    ----------
    pressure_half : (nlev+1, *batch)
        Gridbox-mean layer-edge pressures (Pa), TOA-first.
    tau, liq_frac, g, w0 : (nlev, ncol, *batch)
        Subcolumn 0.67-um optical thickness, liquid fraction of the
        extinction, 3.7-um asymmetry parameter and single-scattering albedo
        (see ``jcosp.modis.optics.build_modis_optics``).
    isccp_ctp : (ncol, *batch), optional
        ISCCP-simulator cloud-top pressure (Pa) used for clouds below the
        CO2-slicing limit (700 hPa). ISCCP is not ported yet; when None the
        CO2-slicing CTP is kept everywhere (documented deviation).
    """
    retrieved_tau = jnp.sum(tau, axis=0)
    cloud_mask = retrieved_tau >= config.MODIS_MIN_OPTICAL_THICKNESS

    ctp = cloud_top_pressure(tau, pressure_half,
                             config.MODIS_CO2SLICING_TAU_LIMIT)

    liq_fraction = weight_by_extinction(tau, liq_frac,
                                        config.MODIS_PHASE_TAU_LIMIT)
    thresh = config.MODIS_PHASE_DISCRIMINATION_THRESHOLD
    phase = jnp.where(
        liq_fraction >= thresh, config.MODIS_PHASE_LIQUID,
        jnp.where(liq_fraction <= 1.0 - thresh, config.MODIS_PHASE_ICE,
                  config.MODIS_PHASE_UNDETERMINED)).astype(jnp.int32)

    obs_refl_nir = compute_toa_reflectance(tau, g, w0)

    trial_re_w, trial_re_i, g_w, w0_w, g_i, w0_i = trial_tables(tau.dtype)
    tshape = (-1,) + (1,) * retrieved_tau.ndim
    pred_w = two_stream_reflectance(retrieved_tau[None], g_w.reshape(tshape),
                                    w0_w.reshape(tshape))
    pred_i = two_stream_reflectance(retrieved_tau[None], g_i.reshape(tshape),
                                    w0_i.reshape(tshape))
    size_w = interpolate_to_min(trial_re_w, pred_w, obs_refl_nir)
    size_i = interpolate_to_min(trial_re_i, pred_i, obs_refl_nir)
    # Liquid and undetermined phases use the water table, ice the ice table
    # (the Fortran's re_fill else-branch is unreachable: phase is always one
    # of the three codes here). Retrieved size is in metres.
    size = 1.0e-6 * jnp.where(phase == config.MODIS_PHASE_ICE, size_i, size_w)

    # Fill the non-cloudy subcolumns.
    ctp = jnp.where(cloud_mask, ctp, R_UNDEF)
    phase = jnp.where(cloud_mask, phase, config.MODIS_PHASE_NONE)
    size = jnp.where(cloud_mask, size, R_UNDEF)
    retrieved_tau = jnp.where(cloud_mask, retrieved_tau, R_UNDEF)
    # Failed size retrievals (re_fill = -999 through the 1e-6 scaling).
    size = jnp.where((size < 0.0) & (size != R_UNDEF),
                     1.0e-6 * config.MODIS_RE_FILL, size)

    if isccp_ctp is not None:
        ctp = jnp.where(
            cloud_mask & (ctp > config.MODIS_CO2SLICING_PRESSURE_LIMIT),
            isccp_ctp, ctp)

    return ModisRetrievals(phase=phase, cloud_top_pressure=ctp,
                           tau=retrieved_tau, size=size)


class ModisColumn(NamedTuple):
    """Column-stage MODIS diagnostics (modis_column outputs).

    Cloud fractions and joint histograms are in percent (of subcolumns), as
    in COSP; means are in-cloud means in natural units (tau unitless, sizes
    m, water paths kg m-2, CTP Pa). The joint-histogram CTP axis is in
    ascending-pressure (TOA-first) bin order here; the ``simulate_modis``
    driver flips it to surface-first as cosp.F90 does on output.
    """

    cf_total: jnp.ndarray       # (*batch) total cloud fraction (%)
    cf_liquid: jnp.ndarray      # (*batch) liquid cloud fraction (%)
    cf_ice: jnp.ndarray         # (*batch) ice cloud fraction (%)
    cf_high: jnp.ndarray        # (*batch) CTP <= 440 hPa (%)
    cf_mid: jnp.ndarray         # (*batch) (%)
    cf_low: jnp.ndarray         # (*batch) CTP > 680 hPa (%)
    tau_total: jnp.ndarray      # (*batch) mean tau, all clouds
    tau_liquid: jnp.ndarray     # (*batch) mean tau, liquid clouds
    tau_ice: jnp.ndarray        # (*batch) mean tau, ice clouds
    tau_total_log: jnp.ndarray  # (*batch) mean log10(tau)
    tau_liquid_log: jnp.ndarray
    tau_ice_log: jnp.ndarray
    size_liquid: jnp.ndarray    # (*batch) mean liquid Reff (m)
    size_ice: jnp.ndarray       # (*batch) mean ice Reff (m)
    ctp_mean: jnp.ndarray       # (*batch) mean CTP (Pa; 0 where cloud-free)
    lwp: jnp.ndarray            # (*batch) liquid water path (kg m-2)
    iwp: jnp.ndarray            # (*batch) ice water path (kg m-2)
    tau_vs_ctp: jnp.ndarray     # (ntau, npres, *batch) joint histogram (%)
    tau_vs_ctp_liq: jnp.ndarray
    tau_vs_ctp_ice: jnp.ndarray
    tau_vs_reff_ice: jnp.ndarray  # (ntau, nreffice, *batch) (%)
    tau_vs_reff_liq: jnp.ndarray  # (ntau, nreffliq, *batch) (%)
    lwp_vs_reff_liq: jnp.ndarray  # (nlwp, nreffliq, *batch) (%)
    iwp_vs_reff_ice: jnp.ndarray  # (niwp, nreffice, *batch) (%)


def modis_column(phase, ctp, tau, size) -> ModisColumn:
    """Aggregate subcolumn retrievals to column statistics (modis_column).

    All inputs are (ncol, *batch) retrieval fields from
    :func:`modis_subcolumn` (``size`` in metres).
    """
    dtype = tau.dtype
    ncol = phase.shape[0]

    valid = size > 0.0
    cloud = (phase != config.MODIS_PHASE_NONE) & valid
    water = (phase == config.MODIS_PHASE_LIQUID) & valid
    ice = (phase == config.MODIS_PHASE_ICE) & valid

    n_tot = jnp.sum(cloud, axis=0).astype(dtype)
    n_wat = jnp.sum(water, axis=0).astype(dtype)
    n_ice = jnp.sum(ice, axis=0).astype(dtype)
    n_high = jnp.sum(
        cloud & (ctp <= config.MODIS_HIGH_CLOUD_PRESSURE_LIMIT),
        axis=0).astype(dtype)
    n_low = jnp.sum(
        cloud & (ctp > config.MODIS_LOW_CLOUD_PRESSURE_LIMIT),
        axis=0).astype(dtype)
    n_mid = n_tot - n_high - n_low

    def msum(x, mask):
        return jnp.sum(jnp.where(mask, x, 0.0), axis=0)

    def mean_or_undef(total, n):
        return jnp.where(n > 0.0, total / jnp.maximum(n, 1.0), R_UNDEF)

    # log10(|tau|): only masked-in cells (tau >= 0.3) contribute; the clamp
    # keeps the masked-out cells (tau = R_UNDEF, |.| huge but finite; or 0)
    # out of the gradient graph's singular range.
    logtau = jnp.log10(jnp.maximum(jnp.abs(tau), 1e-30))
    lwp_cells = config.MODIS_LWP_CONVERSION * size * tau
    iwp_cells = (config.MODIS_LWP_CONVERSION * config.MODIS_ICE_DENSITY
                 * size * tau)

    tau_total = mean_or_undef(msum(tau, cloud), n_tot)
    tau_total_log = mean_or_undef(msum(logtau, cloud), n_tot)
    tau_liquid = mean_or_undef(msum(tau, water), n_wat)
    tau_liquid_log = mean_or_undef(msum(logtau, water), n_wat)
    size_liquid = mean_or_undef(msum(size, water), n_wat)
    lwp = mean_or_undef(msum(lwp_cells, water), n_wat)
    tau_ice = mean_or_undef(msum(tau, ice), n_ice)
    tau_ice_log = mean_or_undef(msum(logtau, ice), n_ice)
    size_ice = mean_or_undef(msum(size, ice), n_ice)
    iwp = mean_or_undef(msum(iwp_cells, ice), n_ice)
    # FIXED UPSTREAM BUG (docs/translation_notes.md): the Fortran divides by
    # max(1, count), so a cloud-free sunlit point reports mean CTP = 0 Pa
    # rather than "undefined". Report R_UNDEF like every other mean here.
    ctp_mean = mean_or_undef(msum(ctp, cloud), n_tot)

    # Histogram working copies: -999 in undetected subcolumns; the Reff
    # copies keep R_UNDEF there (the Fortran's where-block does not touch
    # them), which the bins ignore either way.
    tau_w = jnp.where(cloud, tau, -999.0)
    ctp_w = jnp.where(cloud, ctp, -999.0)
    tau_liq_w = jnp.where(cloud, jnp.where(water, tau, R_UNDEF), -999.0)
    tau_ice_w = jnp.where(cloud, jnp.where(ice, tau, R_UNDEF), -999.0)
    lwp_w = jnp.where(cloud, jnp.where(water, lwp_cells, R_UNDEF), -999.0)
    iwp_w = jnp.where(cloud, jnp.where(ice, iwp_cells, R_UNDEF), -999.0)
    reff_liq_w = jnp.where(water, size, R_UNDEF)
    reff_ice_w = jnp.where(ice, size, R_UNDEF)

    scale = 100.0 / ncol
    tau_bins = jnp.asarray(config.MODIS_HIST_TAU, dtype)
    pres_bins = jnp.asarray(config.MODIS_HIST_PRES, dtype)
    reff_liq_bins = jnp.asarray(config.MODIS_HIST_REFF_LIQ, dtype)
    reff_ice_bins = jnp.asarray(config.MODIS_HIST_REFF_ICE, dtype)
    tau_vs_ctp = scale * stats.hist2d(tau_w, ctp_w, tau_bins, pres_bins)
    tau_vs_ctp_liq = scale * stats.hist2d(tau_liq_w, ctp_w, tau_bins, pres_bins)
    tau_vs_ctp_ice = scale * stats.hist2d(tau_ice_w, ctp_w, tau_bins, pres_bins)
    lwp_vs_reff_liq = scale * stats.hist2d(
        lwp_w, reff_liq_w, jnp.asarray(config.MODIS_HIST_LWP, dtype),
        reff_liq_bins)
    iwp_vs_reff_ice = scale * stats.hist2d(
        iwp_w, reff_ice_w, jnp.asarray(config.MODIS_HIST_IWP, dtype),
        reff_ice_bins)
    tau_vs_reff_ice = scale * stats.hist2d(tau_w, reff_ice_w, tau_bins,
                                           reff_ice_bins)
    tau_vs_reff_liq = scale * stats.hist2d(tau_w, reff_liq_w, tau_bins,
                                           reff_liq_bins)

    pct = 100.0 / ncol
    return ModisColumn(
        cf_total=n_tot * pct, cf_liquid=n_wat * pct, cf_ice=n_ice * pct,
        cf_high=n_high * pct, cf_mid=n_mid * pct, cf_low=n_low * pct,
        tau_total=tau_total, tau_liquid=tau_liquid, tau_ice=tau_ice,
        tau_total_log=tau_total_log, tau_liquid_log=tau_liquid_log,
        tau_ice_log=tau_ice_log, size_liquid=size_liquid, size_ice=size_ice,
        ctp_mean=ctp_mean, lwp=lwp, iwp=iwp,
        tau_vs_ctp=tau_vs_ctp, tau_vs_ctp_liq=tau_vs_ctp_liq,
        tau_vs_ctp_ice=tau_vs_ctp_ice, tau_vs_reff_ice=tau_vs_reff_ice,
        tau_vs_reff_liq=tau_vs_reff_liq, lwp_vs_reff_liq=lwp_vs_reff_liq,
        iwp_vs_reff_ice=iwp_vs_reff_ice)
