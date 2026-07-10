"""MODIS input optics: subcolumn merges, phase partition, 3.7-um properties.

Ports of COSP v2.0 ``subsample_and_optics_example/optics/cosp_optics.F90``
(``cosp_simulator_optics``, ``MODIS_OPTICS_PARTITION``, ``modis_optics``;
D. Swales 2015) and the near-IR polynomial fits of
``src/simulator/MODIS_simulator/modis_simulator.F90`` (``get_g_nir``,
``get_ssa_nir``; fits by S. Platnick, MODIS Collection 6, for the 3.7-um
channel size retrievals). Copyright (c) 2015, Regents of the University of
Colorado, BSD-3-Clause; see LICENSES/. The Collection-5 ``*_old`` fits in the
Fortran are dead code and are not ported.

The build mirrors the reference driver's MODIS block
(``driver/src/cosp2_test.F90:subsample_and_optics``): only the *cloud*
liquid/ice classes (LS + CV) count as cloud for MODIS — rain, snow and
graupel are excluded — and the host-supplied 0.67-um layer optical depths
(stratiform / convective) are painted onto subcolumns through the SCOPS map.

Array convention: level axis 0 TOA-first, subcolumn axis 1, batch trailing.
"""

from typing import NamedTuple

import jax.numpy as jnp

from jcosp import config


def simulator_optics(frac_out, var_strat, var_conv):
    """Merge stratiform/convective fields onto subcolumns (cosp_simulator_optics).

    ``frac_out`` is the SCOPS subcolumn cloud-type map (nlev, ncol, *batch);
    ``var_strat``/``var_conv`` are either gridbox-mean (nlev, *batch) fields
    (the Fortran 2D variant) or subcolumn-resolved (nlev, ncol, *batch)
    fields (the 3D variant). Returns the merged (nlev, ncol, *batch) field:
    the stratiform value in stratiform cells, the convective value in
    convective cells, zero in clear cells.
    """
    vs = var_strat if var_strat.ndim == frac_out.ndim else var_strat[:, None]
    vc = var_conv if var_conv.ndim == frac_out.ndim else var_conv[:, None]
    return jnp.where(frac_out == config.SGCLD_ST, vs,
                     jnp.where(frac_out == config.SGCLD_CUM, vc, 0.0))


def optics_partition(cloud_water, cloud_ice, water_size, ice_size, tau):
    """Partition 0.67-um optical thickness by phase (MODIS_OPTICS_PARTITION).

    All inputs are subcolumn fields (nlev, ncol, *batch): cloud water/ice
    content (any consistent unit), particle sizes (any consistent unit) and
    total optical thickness. Returns ``(tau_liq, tau_ice)``.

    In the geometric-optics limit tau is proportional to (water path)/re, so
    the liquid fraction is (W_liq/re_liq) / (W_liq/re_liq + W_ice/(0.93 re_ice)).
    """
    # Clamp the sizes and the denominator inside the mixed-phase branch:
    # cells without both phases are masked out by the where-tree, but hard
    # zeros would put 1/0 and 0/0 into the gradient graph.
    ws = jnp.maximum(water_size, 1e-30)
    ice_s = jnp.maximum(ice_size, 1e-30)
    wl = cloud_water / ws
    wi = cloud_ice / (config.MODIS_ICE_DENSITY * ice_s)
    frac_liq_mixed = wl / jnp.maximum(wl + wi, 1e-30)
    frac_liq = jnp.where(cloud_ice <= 0.0, 1.0,
                         jnp.where(cloud_water <= 0.0, 0.0, frac_liq_mixed))
    tau_liq = frac_liq * tau
    return tau_liq, tau - tau_liq


def _fit_to_quadratic(x, c):
    return c[0] + x * (c[1] + x * c[2])


def _fit_to_cubic(x, c):
    return c[0] + x * (c[1] + x * (c[2] + x * c[3]))


def get_g_nir(phase: int, re):
    """3.7-um asymmetry parameter vs size (get_g_nir, MODIS Collection 6).

    ``phase`` is a static phase code (MODIS_PHASE_LIQUID or MODIS_PHASE_ICE);
    ``re`` is the effective radius in *microns*. Out-of-range radii evaluate
    the fit at the corresponding bound, exactly as the Fortran.
    """
    if phase == config.MODIS_PHASE_LIQUID:
        small = (1.0364, -8.8800e-2, 7.0000e-3)
        big = (0.6035, 2.8993e-2, -1.1051e-3, 1.5134e-5)
        g_small = _fit_to_quadratic(
            jnp.maximum(re, config.MODIS_RE_WATER_MIN), small)
        g_big = _fit_to_cubic(jnp.minimum(re, config.MODIS_RE_WATER_MAX), big)
        return jnp.where(re < 7.0, g_small, g_big)
    ice = (0.7490, 6.5153e-3, -5.4136e-5)
    return _fit_to_quadratic(
        jnp.clip(re, config.MODIS_RE_ICE_MIN, config.MODIS_RE_ICE_MAX), ice)


def get_ssa_nir(phase: int, re):
    """3.7-um single-scattering albedo vs size (get_ssa_nir, Collection 6).

    Same conventions as :func:`get_g_nir`.
    """
    if phase == config.MODIS_PHASE_LIQUID:
        water = (1.0044, -1.1397e-2, 1.3300e-4)
        return _fit_to_quadratic(
            jnp.clip(re, config.MODIS_RE_WATER_MIN, config.MODIS_RE_WATER_MAX),
            water)
    ice = (0.9625, -1.8069e-2, 3.3281e-4, -2.2865e-6)
    return _fit_to_cubic(
        jnp.clip(re, config.MODIS_RE_ICE_MIN, config.MODIS_RE_ICE_MAX), ice)


def modis_optics(tau_liq, size_liq, tau_ice, size_ice):
    """Combine 3.7-um asymmetry/SSA and the liquid fraction (modis_optics).

    ``tau_liq``/``tau_ice`` are the phase-partitioned 0.67-um optical
    thicknesses and ``size_liq``/``size_ice`` the particle sizes in *microns*
    (the driver passes Reff * 1e6). Returns ``(frac_liq, g, w0)``, each zero
    where the cell has no cloud optical thickness.
    """
    water_g = get_g_nir(config.MODIS_PHASE_LIQUID, size_liq)
    water_w0 = get_ssa_nir(config.MODIS_PHASE_LIQUID, size_liq)
    ice_g = get_g_nir(config.MODIS_PHASE_ICE, size_ice)
    ice_w0 = get_ssa_nir(config.MODIS_PHASE_ICE, size_ice)

    tau = tau_liq + tau_ice
    cloudy = tau > 0.0
    # Clamp the denominators inside the masked branch (tau == 0 cells are
    # zeroed by the where); w0 > 0 whenever tau > 0 because both fits are
    # positive over their clamped ranges.
    tau_safe = jnp.maximum(tau, 1e-30)
    w0 = jnp.where(cloudy,
                   (tau_liq * water_w0 + tau_ice * ice_w0) / tau_safe, 0.0)
    g = jnp.where(cloudy,
                  (tau_liq * water_g * water_w0 + tau_ice * ice_g * ice_w0)
                  / jnp.maximum(w0 * tau, 1e-30), 0.0)
    frac_liq = jnp.where(cloudy, tau_liq / tau_safe, 0.0)
    return frac_liq, g, w0


class ModisOptics(NamedTuple):
    """Subcolumn optical inputs to the MODIS simulator (all (nlev, ncol, *batch))."""

    tau_067: jnp.ndarray      # 0.67-um layer optical thickness
    frac_liq: jnp.ndarray     # Liquid fraction of the extinction
    g: jnp.ndarray            # 3.7-um asymmetry parameter
    w0: jnp.ndarray           # 3.7-um single-scattering albedo
    tau_liq: jnp.ndarray      # Liquid part of tau_067
    tau_ice: jnp.ndarray      # Ice part of tau_067
    cloud_water: jnp.ndarray  # Cloud liquid mixing ratio (kg/kg, in-cloud)
    cloud_ice: jnp.ndarray    # Cloud ice mixing ratio (kg/kg, in-cloud)
    water_size: jnp.ndarray   # Cloud droplet effective radius (m)
    ice_size: jnp.ndarray     # Cloud ice effective radius (m)


def build_modis_optics(frac_out, mr_hydro, reff, dtau_s, dtau_c) -> ModisOptics:
    """Assemble the MODIS optical inputs (driver MODIS block, cosp2_test.F90).

    Parameters
    ----------
    frac_out : (nlev, ncol, *batch)
        SCOPS subcolumn cloud-type map.
    mr_hydro, reff : (nlev, ncol, *batch, N_HYDRO)
        In-cloud subcolumn mixing ratios (kg/kg) and effective radii (m),
        e.g. from ``jcosp.inputs.distribute_hydrometeors``. Only the cloud
        liquid/ice classes are read.
    dtau_s, dtau_c : (nlev, *batch)
        Gridbox-mean 0.67-um optical depth of the stratiform / convective
        cloud in each layer (the host model's ``dtau_s``/``dtau_c``).
    """
    tau_067 = simulator_optics(frac_out, dtau_s, dtau_c)
    cloud_water = simulator_optics(frac_out, mr_hydro[..., config.I_LSCLIQ],
                                   mr_hydro[..., config.I_CVCLIQ])
    cloud_ice = simulator_optics(frac_out, mr_hydro[..., config.I_LSCICE],
                                 mr_hydro[..., config.I_CVCICE])
    water_size = simulator_optics(frac_out, reff[..., config.I_LSCLIQ],
                                  reff[..., config.I_CVCLIQ])
    ice_size = simulator_optics(frac_out, reff[..., config.I_LSCICE],
                                reff[..., config.I_CVCICE])

    tau_liq, tau_ice = optics_partition(cloud_water, cloud_ice,
                                        water_size, ice_size, tau_067)
    frac_liq, g, w0 = modis_optics(tau_liq, water_size * 1.0e6,
                                   tau_ice, ice_size * 1.0e6)
    return ModisOptics(tau_067=tau_067, frac_liq=frac_liq, g=g, w0=w0,
                       tau_liq=tau_liq, tau_ice=tau_ice,
                       cloud_water=cloud_water, cloud_ice=cloud_ice,
                       water_size=water_size, ice_size=ice_size)
