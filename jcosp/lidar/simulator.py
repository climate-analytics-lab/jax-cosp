"""CALIPSO lidar subcolumn stage: attenuated backscatter and channels.

Port of COSP v2.0 ``src/simulator/actsim/lidar_simulator.F90`` subroutines
``lidar_subcolumn``, ``cmp_backsignal`` and ``cmp_beta`` (Copyright (c) 2009,
Centre National de la Recherche Scientifique, BSD-3-Clause; see LICENSES/).
Original ActSim code by M. Chiriaco and H. Chepfer, rewritten by S. Bony;
perpendicular channel by G. Cesana and H. Chepfer (2013, JGR,
doi:10.1002/jgrd.50376).

The attenuated backscatter uses the analytic within-layer integration of the
lidar equation (eq. 1 of doi:10.1029/2008GL034207):

    ATB_k = beta_k * exp(-2 tau_{k-1}) * (1 - exp(-2 dtau_k)) / (2 dtau_k)

where tau_k is the optical depth integrated from the lidar (TOA) through
layer k and dtau_k = tau_k - tau_{k-1}.

Only the spaceborne direction is implemented (COSP's ``lground=.false.``).
Array convention: level axis 0 TOA-first, subcolumn axis next, batch axes
trailing.
"""

from typing import NamedTuple

import jax.numpy as jnp

from jcosp import config


def _layer_tau(tau):
    """Split TOA-integrated optical depth into (tau above layer, layer dtau)."""
    tau_above = jnp.concatenate([jnp.zeros_like(tau[:1]), tau[:-1]], axis=0)
    return tau_above, tau - tau_above


def attenuated_backscatter(beta, tau):
    """Attenuated backscatter from beta and TOA-integrated tau (cmp_backsignal).

    ``beta`` and ``tau`` are (nlev, ...) TOA-first; returns the same shape.

    The Fortran computes the top layer unconditionally (dividing by 2*tau_1);
    since the molecular contribution makes tau_1 > 0 in every COSP call, the
    uniform ``dtau > 0`` guard used here (top layer included) is equivalent —
    and keeps the dtau <= 0 fallback finite for gradients.
    """
    tau_above, dtau = _layer_tau(tau)
    trans = jnp.exp(-2.0 * tau_above)
    dtau_safe = jnp.where(dtau > 0.0, dtau, 1.0)
    return jnp.where(
        dtau > 0.0,
        beta * trans / (2.0 * dtau_safe) * (1.0 - jnp.exp(-2.0 * dtau_safe)),
        beta * trans)


def backscatter_coefficient(pnorm, tau):
    """Invert the lidar equation: beta from attenuated backscatter (cmp_beta).

    ``pnorm`` and ``tau`` are (nlev, ...) TOA-first. Where the two-way
    transmission to the layer top underflows (exp(-2 tau) <= machine eps) the
    Fortran returns pnorm / eps; reproduced here.
    """
    eps = jnp.finfo(jnp.asarray(pnorm).dtype).eps
    tau_above, dtau = _layer_tau(tau)
    trans = jnp.exp(-2.0 * tau_above)
    trans_safe = jnp.where(trans > eps, trans, 1.0)
    dtau_safe = jnp.where(dtau > 0.0, dtau, 1.0)
    expm = 1.0 - jnp.exp(-2.0 * dtau_safe)
    expm_safe = jnp.where(dtau > 0.0, expm, 1.0)
    inner = jnp.where(dtau > 0.0,
                      pnorm / trans_safe * (2.0 * dtau_safe) / expm_safe,
                      pnorm / trans_safe)
    return jnp.where(trans > eps, inner, pnorm / eps)


class LidarSubcolumn(NamedTuple):
    """Subcolumn-stage lidar signals."""

    pmol: jnp.ndarray            # (nlev, *batch) molecular ATB (m-1 sr-1)
    pnorm: jnp.ndarray           # (nlev, ncol, *batch) total ATB
    pnorm_perp: jnp.ndarray | None  # (nlev, ncol, *batch) perpendicular ATB


def lidar_subcolumn(beta_mol, tau_mol, betatot, tautot,
                    betatot_ice=None, tautot_ice=None,
                    betatot_liq=None, tautot_liq=None) -> LidarSubcolumn:
    """Attenuated backscatter signals per subcolumn (lidar_subcolumn).

    Parameters
    ----------
    beta_mol, tau_mol : (nlev, *batch)
        Molecular backscatter coefficient and TOA-integrated optical depth.
    betatot, tautot : (nlev, ncol, *batch)
        Total (molecular + particle) backscatter and optical depth.
    betatot_ice, tautot_ice, betatot_liq, tautot_liq : optional
        Phase-split optics (each including the molecular part, as produced by
        ``jcosp.lidar.optics.lidar_optics``). If all four are given, the
        perpendicular-channel ATB is computed (Cesana and Chepfer 2013);
        otherwise ``pnorm_perp`` is None (the ATLID/ground-lidar mode).
    """
    pmol = attenuated_backscatter(beta_mol, tau_mol)
    pnorm = attenuated_backscatter(betatot, tautot)

    phase = (betatot_ice is not None and tautot_ice is not None
             and betatot_liq is not None and tautot_liq is not None)
    if not phase:
        return LidarSubcolumn(pmol=pmol, pnorm=pnorm, pnorm_perp=None)

    # Per-phase ATB, then the observed perpendicular ATB parameterization
    # (GOCCP 120-m fits): ice is linear in ATB_ice, liquid quadratic.
    pnorm_ice = attenuated_backscatter(betatot_ice, tautot_ice)
    pnorm_liq = attenuated_backscatter(betatot_liq, tautot_liq)
    pnorm_perp_ice = config.CALIPSO_ATBPERP_ALPHA * pnorm_ice
    pnorm_perp_liq = (1000.0 * config.CALIPSO_ATBPERP_BETA * pnorm_liq**2
                      + config.CALIPSO_ATBPERP_GAMMA * pnorm_liq)

    # Un-attenuate each channel with its own phase optical depth to get the
    # perpendicular backscatter coefficients.
    beta_perp_ice = backscatter_coefficient(pnorm_perp_ice, tautot_ice)
    beta_perp_liq = backscatter_coefficient(pnorm_perp_liq, tautot_liq)

    # Total perpendicular ATB, re-attenuated with the *total* optical depth.
    # The molecular perpendicular contribution enters twice (once per phase
    # channel), so one copy beta_mol/(1 + 1/0.0284) is removed (Bodhaine et
    # al. 1999 molecular depolarization).
    s = (beta_perp_ice + beta_perp_liq
         - config.CALIPSO_MOL_DEPOL * beta_mol[:, None])
    tau_above, dtau = _layer_tau(tautot)
    trans = jnp.exp(-2.0 * tau_above)
    dtau_safe = jnp.where(dtau > 0.0, dtau, 1.0)
    base = jnp.where(
        dtau > 0.0,
        s * trans / (2.0 * dtau_safe) * (1.0 - jnp.exp(-2.0 * dtau_safe)),
        s * trans)
    # Fortran masks: the top layer is zeroed where tautot(1) <= 0; deeper
    # layers are zeroed where the parallel ATB is exactly zero.
    top = jnp.where(tautot[:1] > 0.0, base[:1], 0.0)
    rest = jnp.where(pnorm[1:] == 0.0, 0.0, base[1:])
    pnorm_perp = jnp.concatenate([top, rest], axis=0)

    return LidarSubcolumn(pmol=pmol, pnorm=pnorm, pnorm_perp=pnorm_perp)
