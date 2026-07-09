"""End-to-end CALIPSO lidar simulator: model state in, GOCCP diagnostics out.

The lean JAX driver replacing COSP's ``cosp.F90`` orchestration for the
CALIPSO path (translation-plan Phase 4 scope): lidar optics (532 nm) ->
subcolumn attenuated backscatter (total + perpendicular channels) ->
statistical-grid scattering ratio -> CFAD, GOCCP cloud fractions, low/mid/
high/total cover and phase diagnostics; optionally the joint CALIPSO/CloudSat
lidar-only cloud product when radar reflectivities are supplied.

Inputs are the *subcolumn* cloud condensate fields (e.g. the LS/CV liquid/ice
classes of ``jcosp.inputs.distribute_hydrometeors``): CALIPSO uses the same
subgrid cloud realization as the radar path, so hosts should feed both
simulators from one SCOPS draw (or their own McICA generator).

Everything is pure and differentiable. Array convention: level axis 0
TOA-first, subcolumn axis 1, batch axes trailing; see README.
"""

from typing import NamedTuple

import jax.numpy as jnp

from jcosp import config
from jcosp.lidar.diagnostics import (CalipsoColumn, CloudFractions,
                                     LidarOnlyCloud, cfad_sr, cldfrac,
                                     lidar_column, lidar_only_cloud,
                                     scattering_ratio)
from jcosp.lidar.optics import LidarOptics, lidar_optics
from jcosp.lidar.simulator import (LidarSubcolumn, attenuated_backscatter,
                                   backscatter_coefficient, lidar_subcolumn)
from jcosp.stats import change_vertical_grid

__all__ = [
    "CalipsoInputs", "CalipsoOutputs", "simulate_calipso",
    "LidarOptics", "lidar_optics",
    "LidarSubcolumn", "lidar_subcolumn",
    "attenuated_backscatter", "backscatter_coefficient",
    "CalipsoColumn", "CloudFractions", "LidarOnlyCloud",
    "lidar_column", "cldfrac", "cfad_sr", "scattering_ratio",
    "lidar_only_cloud",
]


class CalipsoInputs(NamedTuple):
    """Host-model state feeding the CALIPSO simulator.

    Profile fields are (nlev, *batch) TOA-first; subcolumn condensate fields
    are (nlev, ncol, *batch) *in-cloud* values (kg/kg), e.g. the LSCLIQ /
    LSCICE / CVCLIQ / CVCICE classes of
    ``jcosp.inputs.distribute_hydrometeors(...).mr_hydro``.
    """

    pressure: jnp.ndarray        # Full-level pressure (Pa)
    pressure_half: jnp.ndarray   # (nlev+1, *batch) half-level pressure (Pa)
    temperature: jnp.ndarray     # Temperature (K)
    zfull: jnp.ndarray           # Layer mid-point height (m)
    zhalf: jnp.ndarray           # Layer bottom-interface height (m)
    q_lsliq: jnp.ndarray         # Large-scale cloud liquid (kg/kg, in-cloud)
    q_lsice: jnp.ndarray         # Large-scale cloud ice
    q_cvliq: jnp.ndarray         # Convective cloud liquid
    q_cvice: jnp.ndarray         # Convective cloud ice
    reff_lsliq: jnp.ndarray      # (nlev, *batch) effective radii (m)
    reff_lsice: jnp.ndarray
    reff_cvliq: jnp.ndarray
    reff_cvice: jnp.ndarray


class CalipsoOutputs(NamedTuple):
    """CALIPSO simulator diagnostics (COSP output names in parentheses)."""

    beta_mol: jnp.ndarray       # (nlev, *batch) molecular ATB (calipso_beta_mol)
    beta_tot: jnp.ndarray       # (nlev, ncol, *batch) total ATB (calipso_beta_tot)
    betaperp_tot: jnp.ndarray   # (nlev, ncol, *batch) perp. ATB (calipso_betaperp_tot)
    cfad_sr: jnp.ndarray        # (SR_BINS, nvgrid, *batch) (calipso_cfad_sr)
    srbval: jnp.ndarray         # (SR_BINS+1,) SR bin edges (calipso_srbval)
    lidarcld: jnp.ndarray       # (nvgrid, *batch) 3D cloud fraction, %
    cldlayer: jnp.ndarray       # (4, *batch) low/mid/high/total cover, %
    lidarcldphase: jnp.ndarray  # (nvgrid, 6, *batch) phase fractions, %
    cldlayerphase: jnp.ndarray  # (4, 6, *batch) layered phase cover, %
    lidarcldtmp: jnp.ndarray    # (40, 5, *batch) phase vs temperature, %
    sr: jnp.ndarray             # (nvgrid, ncol, *batch) scattering ratio
    lidar_only: LidarOnlyCloud | None  # joint radar products (needs dbze)


def simulate_calipso(state: CalipsoInputs, ice_type: int = 0,
                     dbze=None) -> CalipsoOutputs:
    """Run the CALIPSO lidar path on subcolumn model state.

    Parameters
    ----------
    state : CalipsoInputs
    ice_type : int
        0 ice spheres, 1 non-spherical ice (COSP ``lidar_ice_type``).
    dbze : (nlev, ncol, *batch), optional
        Attenuated radar reflectivity on *model* levels (e.g.
        ``simulate_cloudsat(...).dbze``). When given, the joint
        CALIPSO/CloudSat lidar-only cloud product is also computed.
    """
    opt = lidar_optics(
        state.q_lsliq, state.q_lsice, state.q_cvliq, state.q_cvice,
        state.reff_lsliq, state.reff_lsice, state.reff_cvliq, state.reff_cvice,
        state.pressure, state.pressure_half, state.temperature,
        ice_type=ice_type)

    sub = lidar_subcolumn(opt.beta_mol, opt.tau_mol, opt.betatot, opt.tautot,
                          opt.betatot_ice, opt.tautot_ice,
                          opt.betatot_liq, opt.tautot_liq)

    # The category pressure COSP feeds the column stage is the layer's
    # lower-interface pressure (cosp.F90 passes phalf(:,2:nlev+1)).
    col = lidar_column(sub.pnorm, sub.pmol, sub.pnorm_perp,
                       state.pressure_half[1:], state.temperature,
                       state.zfull, state.zhalf)

    lonly = None
    if dbze is not None:
        # Joint product: attenuated lidar signals and (log-averaged) radar
        # reflectivity regridded to the statistical grid (cosp.F90, use_vgrid).
        betamol_r = change_vertical_grid(
            state.zfull, state.zhalf, sub.pmol[:, None],
            config.VGRID_ZL, config.VGRID_ZU)[:, 0]
        pnorm_r = change_vertical_grid(state.zfull, state.zhalf, sub.pnorm,
                                       config.VGRID_ZL, config.VGRID_ZU)
        dbze_r = change_vertical_grid(state.zfull, state.zhalf, dbze,
                                      config.VGRID_ZL, config.VGRID_ZU,
                                      log_units=True)
        lonly = lidar_only_cloud(pnorm_r, betamol_r, dbze_r)

    return CalipsoOutputs(
        beta_mol=sub.pmol, beta_tot=sub.pnorm, betaperp_tot=sub.pnorm_perp,
        cfad_sr=col.cfad_sr, srbval=jnp.asarray(config.CALIPSO_HIST_BSCT),
        lidarcld=col.lidarcld, cldlayer=col.cldlayer,
        lidarcldphase=col.lidarcldphase, cldlayerphase=col.cldlayerphase,
        lidarcldtmp=col.lidarcldtemp, sr=col.sr, lidar_only=lonly)
