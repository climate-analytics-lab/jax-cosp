"""Quickbeam radar simulator: attenuation, dBZe, CFADs and precip flags, in JAX.

Ported from COSP v2.0 ``src/simulator/quickbeam.F90`` (QuickBeam by John
Haynes, Roj Marchand et al.; CloudSat 2C-PRECIP-COLUMN flagging by Tristan
L'Ecuyer after Haynes et al. 2009; Copyright (c) 2015, Regents of the
University of Colorado, BSD-3-Clause; see LICENSES/).

Array convention: level axis 0 TOA-first, subcolumn axis next, batch axes
trailing.
"""

from typing import NamedTuple

import jax.numpy as jnp

from jcosp import config, stats
from jcosp.constants import R_UNDEF


def _attenuation_to_volume(kr_vol, hgt_km, weight: float, two_start: bool):
    """Cumulative attenuation from the index-0 layer down to each volume (dB).

    Port of the ``d_gate == 1`` gate loop of quickbeam_subcolumn: each step
    adds a trapezoid over the layer above and the current layer, using the
    mid-point heights. ``weight`` is 0.5 for the (already two-way) gas
    attenuation and 1 for the one-way hydrometeor kr (which thereby counts
    twice, giving the two-way path). The Fortran's two gate-direction
    branches are asymmetric: toward increasing index the *hydrometeor* sum
    leaves both of the first two gates as un-accumulated single-layer terms
    (``two_start``), while the gas sum and the decreasing-index branch leave
    only one.
    """
    dh = hgt_km[:-1] - hgt_km[1:]  # (nlev-1, ...) layer-midpoint separations
    trap = weight * (kr_vol[:-1] + kr_vol[1:]) * dh
    a0 = weight * kr_vol[:1] * dh[:1]
    if not two_start:
        return jnp.concatenate([a0, a0 + jnp.cumsum(trap, axis=0)], axis=0)
    # Legacy (reference-Fortran) behaviour only: gate 2's value ignores gate
    # 1's accumulation and uses the layer spacing BELOW it. See the
    # fixed-upstream-bug note in quickbeam_subcolumn.
    a1 = weight * kr_vol[1:2] * dh[1:2]
    return jnp.concatenate([a0, a1, a1 + jnp.cumsum(trap[1:], axis=0)], axis=0)


def quickbeam_subcolumn(z_vol, kr_vol, g_vol, hgt_km,
                        radar_at_layer_one: bool = True,
                        legacy_gate2: bool = False):
    """Attenuated and non-attenuated effective reflectivity (dBZe).

    Parameters
    ----------
    z_vol : (nlev, ...) reflectivity factor (mm^6/m^3).
    kr_vol : (nlev, ...) hydrometeor attenuation (dB/km, one-way).
    g_vol : (nlev, ...) gaseous attenuation (dB/km, two-way).
    hgt_km : (nlev, ...) layer mid-point heights (km), TOA-first.
    radar_at_layer_one : True for a radar at the index-0 end of the level
        axis (spaceborne for TOA-first data).
    legacy_gate2 : bool
        Reproduce a FIXED UPSTREAM BUG (docs/translation_notes.md): in the
        reference's increasing-gate branch, the hydrometeor attenuation of
        gate 2 ignores gate 1's contribution and uses the layer spacing
        below it (``a(2) = kr(2)*(h(2)-h(3))``), inconsistently with the
        gas sum, the decreasing-gate branch, and the trapezoid recurrence
        used everywhere else. The corrected default accumulates gate 2 as
        ``a(1) + (kr(1)+kr(2))*(h(1)-h(2))``. For a spaceborne radar the
        difference is confined to near-TOA gates and is usually negligible.

    Returns
    -------
    dbze, ze_non : (nlev, ...), R_UNDEF where the volume has no reflectivity.
    """
    if radar_at_layer_one:
        a_to_vol = _attenuation_to_volume(kr_vol, hgt_km, 1.0,
                                          two_start=legacy_gate2)
        g_to_vol = _attenuation_to_volume(g_vol, hgt_km, 0.5, two_start=False)
    else:
        # The decreasing-gate branch is the increasing-gate recurrence on
        # flipped arrays, with a single un-accumulated start gate for both
        # the hydrometeor and gas sums.
        flip = lambda a: jnp.flip(a, axis=0)
        a_to_vol = flip(_attenuation_to_volume(
            flip(kr_vol), flip(hgt_km), 1.0, two_start=False))
        g_to_vol = flip(_attenuation_to_volume(
            flip(g_vol), flip(hgt_km), 0.5, two_start=False))

    valid = z_vol > 0.0
    ze_non = jnp.where(valid,
                       10.0 * jnp.log10(jnp.maximum(z_vol, 1e-30)), R_UNDEF)
    dbze = jnp.where(valid, ze_non - a_to_vol - g_to_vol, R_UNDEF)
    return dbze, ze_non


class CloudsatColumnOutputs(NamedTuple):
    """Gridded CloudSat diagnostics (quickbeam_column)."""

    cfad_ze: jnp.ndarray            # (DBZE_BINS, nglev, *batch) normalized CFAD
    dbze_vgrid: jnp.ndarray         # (nglev, ncol, *batch) Ze on the 480-m grid
    ze_non_vgrid: jnp.ndarray       # (nglev, ncol, *batch) unattenuated
    precip_cover: jnp.ndarray       # (N_CLOUDSAT_PRECIP_CLASS, *batch)
    pia: jnp.ndarray                # (*batch) path-integrated attenuation


def cloudsat_precip_occurrence(ze, ze_non, land, surfelev, t2m,
                               frac_precip_ice, zstep=config.VGRID_ZSTEP):
    """CloudSat 2C-PRECIP-COLUMN-style near-surface precipitation flags.

    Port of quickbeam.F90:cloudsat_precipOccurence. Operates on the CloudSat
    statistical grid (TOA-first); ``ze``/``ze_non`` are (nglev, ncol, *batch),
    ``frac_precip_ice`` is the frozen fraction of surface precipitation
    (ncol, *batch), and ``land``/``surfelev``/``t2m`` are (*batch).

    Returns
    -------
    precip_cover : (N_CLOUDSAT_PRECIP_CLASS, *batch)
        Fraction of subcolumns per precipitation class.
    pia : (*batch)
        Mean path-integrated attenuation over subcolumns (dB).
    """
    nglev = ze.shape[0]
    # Range gate used for the flags: 480-960 m above the surface (fortran
    # cloudsat_preclvl, shifted by the surface elevation), one gate higher
    # over land. Gather per batch element.
    lvl_ocean = config.CLOUDSAT_PRECLVL - jnp.floor(surfelev / zstep).astype(jnp.int32)
    lvl = jnp.clip(jnp.where(land >= 0.5, lvl_ocean - 1, lvl_ocean), 0, nglev - 1)

    idx = lvl[None]  # broadcast over the subcolumn axis
    ze_lvl = jnp.take_along_axis(ze, idx[None], axis=0)[0]          # (ncol, *b)
    ze_non_lvl = jnp.take_along_axis(ze_non, idx[None], axis=0)[0]  # (ncol, *b)

    # Path-integrated attenuation where the gate contains hydrometeors.
    pia_valid = ((ze_non_lvl > -100) & (ze_lvl > -100)
                 & (ze_non_lvl < 100) & (ze_lvl < 100))
    pia_col = jnp.where(pia_valid, ze_non_lvl - ze_lvl, 0.0)

    zmax = jnp.max(ze, axis=0)  # column max of the attenuated profile (land)

    znb = config.ZE_NON_BINVAL
    zlb = config.Z_BINVAL_LND
    fpi = frac_precip_ice
    flags = jnp.full(ze_lvl.shape, config.PCLASS_DEFAULT, dtype=jnp.int32)

    # ---- Ocean points (thresholds on the unattenuated gate value). --------
    o = jnp.full_like(flags, config.PCLASS_DEFAULT)
    snow = fpi > 0.9
    o = jnp.where(snow & (ze_non_lvl > znb[1]), config.PCLASS_SNOW2, o)
    o = jnp.where(snow & (ze_non_lvl > znb[3]) & (ze_non_lvl <= znb[1]),
                  config.PCLASS_SNOW1, o)
    mixed = (fpi > 0.1) & (fpi <= 0.9)
    o = jnp.where(mixed & (ze_non_lvl > znb[1]), config.PCLASS_MIXED2, o)
    o = jnp.where(mixed & (ze_non_lvl > znb[3]) & (ze_non_lvl <= znb[1]),
                  config.PCLASS_MIXED1, o)
    rain = fpi <= 0.1
    o = jnp.where(rain & (ze_non_lvl > znb[0]), config.PCLASS_RAIN3, o)
    o = jnp.where(rain & (ze_non_lvl > znb[2]) & (ze_non_lvl <= znb[0]),
                  config.PCLASS_RAIN2, o)
    o = jnp.where(rain & (ze_non_lvl > znb[3]) & (ze_non_lvl <= znb[2]),
                  config.PCLASS_RAIN1, o)
    o = jnp.where(rain & (pia_col > 40), config.PCLASS_RAIN4, o)
    o = jnp.where(ze_non_lvl <= -15, config.PCLASS_NO_PRECIP, o)

    # ---- Land points (attenuated gate value + column max + 2-m T). --------
    ll = jnp.full_like(flags, config.PCLASS_DEFAULT)
    cold = t2m < 273.0
    ll = jnp.where(cold & (ze_lvl > zlb[4]), config.PCLASS_SNOW2, ll)
    ll = jnp.where(cold & (ze_lvl > zlb[5]) & (ze_lvl <= zlb[4]),
                   config.PCLASS_SNOW1, ll)
    mixed_t = (t2m >= 273.0) & (t2m <= 275.0)
    ll = jnp.where(mixed_t & (((zmax > zlb[0]) & (pia_col > 30))
                              | (ze_lvl > zlb[3])), config.PCLASS_MIXED2, ll)
    ll = jnp.where(mixed_t & (ze_lvl > zlb[5]) & (ze_lvl <= zlb[3])
                   & (zmax > zlb[4]), config.PCLASS_MIXED1, ll)
    warm = t2m > 275.0
    ll = jnp.where(warm & (((zmax > zlb[0]) & (pia_col > 30))
                           | (ze_lvl > zlb[1])), config.PCLASS_RAIN3, ll)
    ll = jnp.where(warm & (ze_lvl > zlb[5]) & (zmax > zlb[2]),
                   config.PCLASS_RAIN2, ll)
    ll = jnp.where(warm & (ze_lvl > zlb[5]) & (zmax < zlb[2]),
                   config.PCLASS_RAIN1, ll)
    ll = jnp.where(warm & (pia_col > 40), config.PCLASS_RAIN4, ll)
    ll = jnp.where(ze_lvl <= -15, config.PCLASS_NO_PRECIP, ll)

    flags = jnp.where(land >= 0.5, ll, o)

    # Aggregate over subcolumns.
    classes = jnp.arange(config.N_CLOUDSAT_PRECIP_CLASS).reshape(
        (-1,) + (1,) * flags.ndim)
    precip_cover = jnp.mean(flags[None] == classes, axis=1)
    pia = jnp.mean(pia_col, axis=0)
    return precip_cover, pia


def quickbeam_column(dbze, ze_non, zfull, zhalf, land, surfelev, t2m,
                     frac_precip_ice):
    """CloudSat gridded diagnostics: CFAD + near-surface precipitation flags.

    Port of quickbeam.F90:quickbeam_column with ``use_vgrid=True`` (the
    CloudSat 40x480 m statistical grid).

    Parameters
    ----------
    dbze, ze_non : (nlev, ncol, *batch) dBZe on model levels, TOA-first.
    zfull, zhalf : (nlev, *batch) layer mid/bottom heights (m).
    land, surfelev, t2m : (*batch) land mask (0/1), surface elevation (m),
        2-m temperature (K).
    frac_precip_ice : (ncol, *batch) frozen fraction of near-surface precip
        (regridded to the flag gate by the caller, as in the driver).
    """
    dbze_i = stats.change_vertical_grid(
        zfull, zhalf, dbze, config.VGRID_ZL, config.VGRID_ZU, log_units=True)
    ze_non_i = stats.change_vertical_grid(
        zfull, zhalf, ze_non, config.VGRID_ZL, config.VGRID_ZU, log_units=True)

    # CFAD: per new-grid level, histogram over subcolumns, / ncolumns.
    ncol = dbze.shape[1]
    cfad = stats.hist1d(jnp.moveaxis(dbze_i, 0, 1), config.CLOUDSAT_HIST_REF)
    cfad = jnp.where(cfad != R_UNDEF, cfad / ncol, cfad)

    precip_cover, pia = cloudsat_precip_occurrence(
        dbze_i, ze_non_i, land, surfelev, t2m, frac_precip_ice)
    return CloudsatColumnOutputs(
        cfad_ze=cfad, dbze_vgrid=dbze_i, ze_non_vgrid=ze_non_i,
        precip_cover=precip_cover, pia=pia)
