"""CALIPSO 532-nm lidar optics: cloud-particle and molecular beta/tau profiles.

Port of COSP v2.0 ``subsample_and_optics_example/optics/cosp_optics.F90``
subroutine ``lidar_optics`` (Copyright (c) 2015, Regents of the University of
Colorado, BSD-3-Clause; see LICENSES/). The scheme originates in the ActSim
lidar simulator of M. Chiriaco and H. Chepfer (see lidar_simulator.F90
history).

Given in-cloud subcolumn mixing ratios of the four CALIPSO hydrometeor
classes (large-scale / convective x liquid / ice) and their effective radii,
this computes the total (molecular + particle) backscatter coefficient and
TOA-integrated optical depth per subcolumn, plus the separate liquid-only and
ice-only fields the perpendicular-channel (phase) diagnostics need.

Only the spaceborne integration direction is implemented (the ground-based
variant integrates surface-up); the 355-nm (ATLID) molecular constants are
plumbed through ``wavelength`` but only 532 nm is validated.

Array convention: level axis 0, TOA-first; subcolumn axis 1 of the ``q_*``
fields; batch axes trailing (see README).
"""

from typing import NamedTuple

import jax.numpy as jnp

from jcosp import config


class LidarOptics(NamedTuple):
    """Optical inputs to the lidar simulator (Fortran ``lidar_optics`` outputs).

    Profile fields are (nlev, *batch); subcolumn fields (nlev, ncol, *batch).
    """

    beta_mol: jnp.ndarray     # Molecular backscatter coefficient (m-1 sr-1)
    tau_mol: jnp.ndarray      # Molecular optical depth, TOA-integrated
    betatot: jnp.ndarray      # Total backscatter coefficient (m-1 sr-1)
    tautot: jnp.ndarray       # Total optical depth, TOA-integrated
    betatot_ice: jnp.ndarray  # Molecular + ice-particle backscatter
    tautot_ice: jnp.ndarray   # Molecular + ice-particle optical depth
    betatot_liq: jnp.ndarray  # Molecular + liquid-particle backscatter
    tautot_liq: jnp.ndarray   # Molecular + liquid-particle optical depth
    tau_sfc_liq: jnp.ndarray  # (ncol, *batch) column liquid particle OD (PARASOL)
    tau_sfc_ice: jnp.ndarray  # (ncol, *batch) column ice particle OD (PARASOL)


def lidar_optics(q_lsliq, q_lsice, q_cvliq, q_cvice,
                 ls_radliq, ls_radice, cv_radliq, cv_radice,
                 pressure, pressure_half, temperature,
                 ice_type: int = 0, wavelength: int = 532) -> LidarOptics:
    """Cloud and molecular lidar optics (cosp_optics.F90:lidar_optics).

    Parameters
    ----------
    q_lsliq, q_lsice, q_cvliq, q_cvice : (nlev, ncol, *batch)
        In-cloud subcolumn mixing ratios (kg/kg) of large-scale / convective
        liquid and ice (e.g. from ``jcosp.inputs.distribute_hydrometeors``).
    ls_radliq, ls_radice, cv_radliq, cv_radice : (nlev, *batch)
        Effective radii (m) of the four classes; values are clipped to
        [0, 70 um] and radius <= 0 disables the class in that volume.
    pressure : (nlev, *batch)
        Pressure at full (layer-midpoint) levels (Pa), TOA-first.
    pressure_half : (nlev+1, *batch)
        Pressure at half levels (Pa), TOA-first (index 0 = model top).
    temperature : (nlev, *batch)
        Temperature (K).
    ice_type : int
        0 for ice spheres, 1 for non-spherical ice (Fortran ``ice_type``).
    wavelength : int
        532 (CALIPSO; validated) or 355 nm (ATLID; plumbing only).

    Notes
    -----
    Layer geometric thicknesses are derived hydrostatically from the pressure
    profile with COSP's own constants (rd = 287.04, grav = 9.80665), exactly
    as the Fortran does — the host's geometric heights are *not* used here
    (they enter later, in the statistical-grid regridding).
    """
    if wavelength not in config.LIDAR_CMOL:
        raise ValueError(f"wavelength must be one of {list(config.LIDAR_CMOL)}")
    if ice_type not in (0, 1):
        raise ValueError("ice_type must be 0 (spheres) or 1 (non-spherical)")
    cmol = config.LIDAR_CMOL[wavelength]
    rdiffm = config.LIDAR_RDIFFM[wavelength]
    pol_ice = (config.LIDAR_POLPART_ICE_SPHERE if ice_type == 0
               else config.LIDAR_POLPART_ICE_NONSPHERE)

    # Particle classes in COSP's lidar_optics order: LSLIQ, LSICE, CVLIQ, CVICE.
    rad = jnp.stack([ls_radliq, ls_radice, cv_radliq, cv_radice])  # (4, nlev, *b)
    rad = jnp.clip(rad, 0.0, config.LIDAR_MAX_RADIUS)
    q = jnp.stack([q_lsliq, q_lsice, q_cvliq, q_cvice])  # (4, nlev, ncol, *b)
    rho_part = jnp.array([config.LIDAR_RHO_LIQ, config.LIDAR_RHO_ICE,
                          config.LIDAR_RHO_LIQ, config.LIDAR_RHO_ICE])
    polpart = jnp.stack([jnp.asarray(config.LIDAR_POLPART_LIQ), jnp.asarray(pol_ice),
                         jnp.asarray(config.LIDAR_POLPART_LIQ), jnp.asarray(pol_ice)])

    # Clear-sky air density and hydrostatic layer thickness (m). The Fortran
    # builds zheight bottom-up from 0; only the differences enter, so the
    # layer thickness is used directly here.
    rhoair = pressure / (config.LIDAR_RD * temperature)          # (nlev, *b)
    dz = ((pressure_half[1:] - pressure_half[:-1])
          / (rhoair * config.LIDAR_GRAV))                        # (nlev, *b)

    # Molecular backscatter, extinction and TOA-integrated optical depth.
    beta_mol = pressure / config.LIDAR_BOLTZMANN / temperature * cmol
    alpha_mol = 8.0 * jnp.pi / 3.0 * beta_mol
    tau_mol = jnp.cumsum(alpha_mol * dz, axis=0)

    # Backscatter-to-extinction polynomial kp(re) per class, re in microns.
    # Gradient hardening: the radius is clamped away from zero inside the
    # masked branch so hydrometeor-free volumes (rad <= 0, masked out) don't
    # poison reverse-mode gradients through the division.
    x_um = rad * 1.0e6
    kp = jnp.where(
        rad > 0.0,
        polpart[:, 0].reshape((-1,) + (1,) * (rad.ndim - 1)) * x_um**4
        + polpart[:, 1].reshape((-1,) + (1,) * (rad.ndim - 1)) * x_um**3
        + polpart[:, 2].reshape((-1,) + (1,) * (rad.ndim - 1)) * x_um**2
        + polpart[:, 3].reshape((-1,) + (1,) * (rad.ndim - 1)) * x_um
        + polpart[:, 4].reshape((-1,) + (1,) * (rad.ndim - 1)),
        0.0)                                                    # (4, nlev, *b)

    # Particle extinction alpha = (3/4) Qscat rho_air q / (rho_part re) and
    # TOA-integrated particle optical depth (with the multiple-scattering
    # factor rdiffm), per class. Insert the subcolumn axis after the level.
    rad_b = rad[:, :, None]                                     # (4, nlev, 1, *b)
    rad_safe = jnp.maximum(rad_b, 1e-30)
    rho_part_b = rho_part.reshape((-1,) + (1,) * (q.ndim - 1))
    alpha_part = jnp.where(
        rad_b > 0.0,
        0.75 * config.LIDAR_QSCAT * rhoair[None, :, None] * q
        / (rho_part_b * rad_safe),
        0.0)                                                    # (4, nlev, ncol, *b)
    tau_part = jnp.cumsum(rdiffm * alpha_part * dz[None, :, None], axis=1)

    # Totals: molecular plus particles (and the ice-only / liquid-only splits,
    # which also include the molecular contribution, as in the Fortran).
    kp_b = kp[:, :, None]
    beta_part = kp_b * alpha_part
    betatot = beta_mol[:, None] + jnp.sum(beta_part, axis=0)
    tautot = tau_mol[:, None] + jnp.sum(tau_part, axis=0)
    betatot_ice = beta_mol[:, None] + beta_part[1] + beta_part[3]
    tautot_ice = tau_mol[:, None] + tau_part[1] + tau_part[3]
    betatot_liq = beta_mol[:, None] + beta_part[0] + beta_part[2]
    tautot_liq = tau_mol[:, None] + tau_part[0] + tau_part[2]

    # Column particle-only optical depths (PARASOL simulator inputs).
    tau_sfc_liq = tau_part[0, -1] + tau_part[2, -1]
    tau_sfc_ice = tau_part[1, -1] + tau_part[3, -1]

    return LidarOptics(beta_mol=beta_mol, tau_mol=tau_mol,
                       betatot=betatot, tautot=tautot,
                       betatot_ice=betatot_ice, tautot_ice=tautot_ice,
                       betatot_liq=betatot_liq, tautot_liq=tautot_liq,
                       tau_sfc_liq=tau_sfc_liq, tau_sfc_ice=tau_sfc_ice)
