"""Host-model inputs -> subcolumn hydrometeor fields.

Ports of COSP v2.0 ``optics/cosp_utils.F90:COSP_PRECIP_MXRATIO`` and the
``subsample_and_optics`` input pipeline of the reference driver
(``driver/src/cosp2_test.F90``); Copyright (c) 2015, Regents of the
University of Colorado, BSD-3-Clause; see LICENSES/.

Pipeline (given SCOPS/PREC_SCOPS subcolumn maps):

1. gridbox-mean cloud condensate -> in-cloud values in cloudy subcolumns
   (divide by the sampled cloud fraction of the matching type);
2. gridbox-mean precipitation fluxes -> in-precip fluxes (divide by the
   sampled precipitation fraction);
3. in-precip fluxes -> mixing ratios and effective radii through each class's
   assumed PSD (cosp_precip_mxratio).

Array convention: level axis 0 TOA-first, subcolumn axis 1, batch trailing;
hydrometeor class is the last axis of stacked outputs.
"""

from typing import NamedTuple

import jax.numpy as jnp

from jcosp import config
from jcosp.constants import RD_PRECIP


def precip_mxratio(pressure, temperature, prec_frac, prec_type,
                   flux, reff_in, *, n_ax, n_bx, alpha_x, c_x, d_x, g_x,
                   a_x, b_x, gamma1, gamma2, gamma3, gamma4):
    """Precipitation flux -> in-precip mixing ratio and effective radius.

    Port of COSP_PRECIP_MXRATIO for one hydrometeor class. ``pressure`` /
    ``temperature`` / ``flux`` are (nlev, *batch); ``prec_frac`` is the
    PREC_SCOPS subcolumn map (nlev, ncol, *batch); scalars are the class's
    PSD/fallspeed parameters (config.PRECIP_*). Returns (mxratio, reff) of
    shape (nlev, ncol, *batch): mixing ratio (kg/kg) and effective radius (m).

    Following the Fortran, ``reff_in`` (same shape, metres) is kept where it
    is positive; the PSD-derived radius fills the rest (where the flux is
    nonzero).
    """
    if n_ax < 0:  # class without flux input (cloud water/ice): nothing to do
        zeros = jnp.zeros(prec_frac.shape, dtype=pressure.dtype)
        return zeros, reff_in if reff_in is not None else zeros

    xi = d_x / (alpha_x + b_x - n_bx + 1.0)
    rho0 = 1.29
    sigma = (gamma2 / (gamma1 * c_x)) * (n_ax * a_x * gamma2) ** xi
    one_over_xip1 = 1.0 / (xi + 1.0)
    gamma_4_3_2 = 0.5 * gamma4 / gamma3
    delta = alpha_x + b_x + d_x - n_bx + 1.0

    in_precip = (prec_frac == prec_type) | (prec_frac == config.PREC_MIXED)
    rho = (pressure / (RD_PRECIP * temperature))[:, None]
    flux_b = flux[:, None]

    # Clamp the flux inside the masked branch: hydrometeor-free cells are
    # excluded by `in_precip` but a hard zero would put 0**x and 1/0 in the
    # gradient graph.
    flux_safe = jnp.maximum(flux_b, 1e-30)
    mxratio = (flux_safe * (rho / rho0) ** g_x * sigma) ** one_over_xip1 / rho
    mxratio = jnp.where(in_precip & (flux_b > 0.0), mxratio, 0.0)

    lambda_x = (a_x * c_x * (rho0 / rho) ** g_x * n_ax * gamma1
                / flux_safe) ** (1.0 / delta)
    reff = jnp.where(in_precip & (flux_b != 0.0), gamma_4_3_2 / lambda_x, 0.0)
    if reff_in is not None:
        reff = jnp.where(reff_in > 0.0, reff_in, reff)
    return mxratio, reff


class SubcolumnHydrometeors(NamedTuple):
    """Subcolumn hydrometeor fields feeding the radar optics."""

    mr_hydro: jnp.ndarray   # (nlev, ncol, *batch, N_HYDRO) mixing ratio (kg/kg)
    reff: jnp.ndarray       # (nlev, ncol, *batch, N_HYDRO) effective radius (m)
    frac_precip_ice: jnp.ndarray  # (nlev, ncol, *batch) frozen precip fraction


def distribute_hydrometeors(frac_out, prec_frac, pressure, temperature,
                            mr_lsliq, mr_lsice, mr_ccliq, mr_ccice,
                            fl_lsrain, fl_lssnow, fl_lsgrpl,
                            fl_ccrain, fl_ccsnow, reff_in=None):
    """Distribute gridbox-mean condensate and precip fluxes onto subcolumns.

    Port of the ``use_precipitation_fluxes=.true.`` path of
    ``subsample_and_optics`` (cosp2_test.F90). All gridbox-mean inputs are
    (nlev, *batch): condensate mixing ratios in kg/kg, fluxes in kg m-2 s-1.
    ``frac_out``/``prec_frac`` are the SCOPS/PREC_SCOPS maps
    (nlev, ncol, *batch). ``reff_in``, if given, is (nlev, *batch, N_HYDRO)
    hydrometeor effective radii (m); 0 selects the PSD defaults downstream.

    Returns
    -------
    SubcolumnHydrometeors
        In-cloud / in-precip subcolumn values (zero outside cloud/precip).
    """
    ncol = frac_out.shape[1]

    # Sampled cloud and precip fractions per level (the subcolumn realization
    # of the input fractions; dividing by these conserves gridbox means).
    frac_ls = jnp.mean(frac_out == config.SGCLD_ST, axis=1)
    frac_cv = jnp.mean(frac_out == config.SGCLD_CUM, axis=1)
    prec_ls = jnp.mean((prec_frac == config.PREC_LS)
                       | (prec_frac == config.PREC_MIXED), axis=1)
    prec_cv = jnp.mean((prec_frac == config.PREC_CV)
                       | (prec_frac == config.PREC_MIXED), axis=1)

    def in_cloud(mr, frac):
        # Gridbox mean -> in-cloud value; guard the empty-fraction case that
        # the Fortran skips with an explicit `if frac /= 0`.
        return jnp.where(frac > 0, mr / jnp.maximum(frac, 1.0 / (2 * ncol)), 0.0)

    ls_mask = frac_out == config.SGCLD_ST
    cv_mask = frac_out == config.SGCLD_CUM

    zeros = jnp.zeros(frac_out.shape, dtype=pressure.dtype)
    mr = {i: zeros for i in range(config.N_HYDRO)}
    mr[config.I_LSCLIQ] = jnp.where(ls_mask, in_cloud(mr_lsliq, frac_ls)[:, None], 0.0)
    mr[config.I_LSCICE] = jnp.where(ls_mask, in_cloud(mr_lsice, frac_ls)[:, None], 0.0)
    mr[config.I_CVCLIQ] = jnp.where(cv_mask, in_cloud(mr_ccliq, frac_cv)[:, None], 0.0)
    mr[config.I_CVCICE] = jnp.where(cv_mask, in_cloud(mr_ccice, frac_cv)[:, None], 0.0)

    # Cloud effective radii ride along on the cloud masks (driver lines
    # assigning Reff for I_LS*/I_CV* under frac_out). These are gridbox-mean
    # (nlev, *batch) fields, broadcast onto subcolumns via [:, None] below.
    zeros_grid = jnp.zeros(pressure.shape, dtype=pressure.dtype)

    def reff_class(i):
        if reff_in is None:
            return zeros_grid
        return reff_in[..., i]

    reff = {i: zeros for i in range(config.N_HYDRO)}
    for i in (config.I_LSCLIQ, config.I_LSCICE):
        reff[i] = jnp.where(ls_mask, reff_class(i)[:, None], 0.0)
    for i in (config.I_CVCLIQ, config.I_CVCICE):
        reff[i] = jnp.where(cv_mask, reff_class(i)[:, None], 0.0)

    # In-precip fluxes (gridbox mean / sampled precip fraction).
    fl_lsrain_ip = in_cloud(fl_lsrain, prec_ls)
    fl_lssnow_ip = in_cloud(fl_lssnow, prec_ls)
    fl_lsgrpl_ip = in_cloud(fl_lsgrpl, prec_ls)
    fl_ccrain_ip = in_cloud(fl_ccrain, prec_cv)
    fl_ccsnow_ip = in_cloud(fl_ccsnow, prec_cv)

    ls_precip_mask = (prec_frac == config.PREC_LS) | (prec_frac == config.PREC_MIXED)
    cv_precip_mask = (prec_frac == config.PREC_CV) | (prec_frac == config.PREC_MIXED)
    precip_reff_in = {
        i: (jnp.where(ls_precip_mask if i in (config.I_LSRAIN, config.I_LSSNOW,
                                              config.I_LSGRPL)
                      else cv_precip_mask, reff_class(i)[:, None], 0.0))
        for i in (config.I_LSRAIN, config.I_LSSNOW, config.I_LSGRPL,
                  config.I_CVRAIN, config.I_CVSNOW)
    }

    for i, flux in ((config.I_LSRAIN, fl_lsrain_ip),
                    (config.I_LSSNOW, fl_lssnow_ip),
                    (config.I_CVRAIN, fl_ccrain_ip),
                    (config.I_CVSNOW, fl_ccsnow_ip),
                    (config.I_LSGRPL, fl_lsgrpl_ip)):
        mr[i], reff[i] = precip_mxratio(
            pressure, temperature, prec_frac, float(config.PRECIP_TYPE[i]),
            flux, precip_reff_in[i],
            n_ax=float(config.PRECIP_N_AX[i]), n_bx=float(config.PRECIP_N_BX[i]),
            alpha_x=float(config.PRECIP_ALPHA_X[i]), c_x=float(config.PRECIP_C_X[i]),
            d_x=float(config.PRECIP_D_X[i]), g_x=float(config.PRECIP_G_X[i]),
            a_x=float(config.PRECIP_A_X[i]), b_x=float(config.PRECIP_B_X[i]),
            gamma1=float(config.PRECIP_GAMMA_1[i]), gamma2=float(config.PRECIP_GAMMA_2[i]),
            gamma3=float(config.PRECIP_GAMMA_3[i]), gamma4=float(config.PRECIP_GAMMA_4[i]))

    mr_hydro = jnp.stack([mr[i] for i in range(config.N_HYDRO)], axis=-1)
    reff_out = jnp.stack([reff[i] for i in range(config.N_HYDRO)], axis=-1)

    # Frozen fraction of precipitation in each volume (driver, cloudsat block).
    frozen = (mr[config.I_LSSNOW] + mr[config.I_CVSNOW] + mr[config.I_LSGRPL])
    total = frozen + mr[config.I_LSRAIN] + mr[config.I_CVRAIN]
    frac_precip_ice = jnp.where(total > 0.0,
                                frozen / jnp.maximum(total, 1e-30), 0.0)

    return SubcolumnHydrometeors(mr_hydro=mr_hydro, reff=reff_out,
                                 frac_precip_ice=frac_precip_ice)
