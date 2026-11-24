"""Warm-rain classification of radar subcolumns (Mülmenstädt et al. 2020).

Implements the model-side warm/cold rain classification of Mülmenstädt et al.
(2020, Sci. Adv. 6, eaaz6433, Methods), the diagnostic used to constrain
warm-rain processes against the CloudSat/CALIPSO climatology of Mülmenstädt
et al. (2015, GRL 42, doi:10.1002/2015GL064604):

  COSP splits each grid box into n subcolumns and uses the Quickbeam radar
  simulator to compute a radar reflectivity in each subcolumn. Model columns
  that produce liquid precipitation at the surface are classified as
  *drizzling* if their maximum reflectivity exceeds -15 dBZe and *raining* if
  it exceeds 0 dBZe. A column is *cold* if the highest cloud layer in which
  the reflectivity threshold is reached contains ice, and *warm* otherwise
  (a cloud layer = vertically contiguous levels with nonzero cloud
  condensate).

This module is an original implementation against that published definition
(not a Fortran port); the reflectivity thresholds match COSP's
``CFODD_BNDZE`` (cosp_config.F90).

Array convention: level axis 0 TOA-first, subcolumn axis 1, batch trailing.
"""

from typing import NamedTuple

import jax.numpy as jnp

from jcosp import config, surrogate


class WarmRainFlags(NamedTuple):
    """Per-gridbox warm-rain occurrence fractions (over subcolumns).

    The regimes here are *exclusive* (drizzle means -15 < max Ze <= 0 dBZe).
    Mülmenstädt et al. (2020) count raining columns as also drizzling
    (inclusive); recover that convention with ``drizzle + rain``.
    """

    warm_rain: jnp.ndarray      # fraction of subcolumns: warm & max Ze > 0 dBZe
    cold_rain: jnp.ndarray      # cold & max Ze > 0 dBZe
    warm_drizzle: jnp.ndarray   # warm & -15 < max Ze <= 0 dBZe
    cold_drizzle: jnp.ndarray   # cold & -15 < max Ze <= 0 dBZe


def _cloud_layer_ids(cloudy):
    """Label vertically contiguous cloudy runs, counting from the top.

    ``cloudy`` is boolean (nlev, ...); the returned int array holds a layer
    id (1, 2, ... top-down) on cloudy levels and 0 elsewhere.
    """
    starts = cloudy & ~jnp.concatenate(
        [jnp.zeros_like(cloudy[:1]), cloudy[:-1]], axis=0)
    return jnp.where(cloudy, jnp.cumsum(starts, axis=0), 0)


def _classify_at_threshold(dbze, cloudy, has_ice, layer_id, threshold, width):
    """Threshold exceedance + warm/cold split of each subcolumn.

    Returns (exceeds, cold) as 0/1 floats: ``exceeds`` where any cloudy
    level's reflectivity passes ``threshold``; ``cold`` where the *highest*
    such cloud layer contains ice anywhere in its contiguous extent.

    ``exceeds`` is differentiable in the reflectivity (``width``). ``cold``
    is a property of the cloud layer and has no derivative; for a subcolumn
    that does not reach the threshold it is evaluated at its strongest
    cloudy echo, the level that would cross first, so that the derivative
    of a near-miss is booked to the regime it would join. The value is
    unaffected: a subcolumn that does not exceed is in neither regime.
    """
    dtype = dbze.dtype
    hit = cloudy & (dbze > threshold)
    hit_gate = cloudy.astype(dtype) * surrogate.gt(dbze, threshold, width)
    exceeds = surrogate.any_of(hit_gate, axis=0)
    # First hit from the top (TOA-first); argmax returns the first True.
    k_top = jnp.argmax(hit, axis=0)
    k_near = jnp.argmax(jnp.where(cloudy, dbze, -jnp.inf), axis=0)
    k_top = jnp.where(jnp.any(hit, axis=0), k_top, k_near)
    top_layer = jnp.take_along_axis(layer_id, k_top[None], axis=0)[0]
    layer_ice = jnp.any(has_ice & (layer_id == top_layer[None]), axis=0)
    return exceeds, layer_ice.astype(dtype)


def warm_rain_classification(dbze, cloud_liq, cloud_ice, surf_liquid_precip,
                             condensate_threshold: float = 0.0,
                             ice_threshold: float = 0.0, weights=None,
                             smoothing=surrogate.DEFAULT_SMOOTHING,
                             ) -> WarmRainFlags:
    """Classify subcolumns into warm/cold rain and drizzle occurrence.

    Parameters
    ----------
    dbze : (nlev, ncol, *batch)
        Simulated subcolumn reflectivity (dBZe). Mülmenstädt et al. apply the
        thresholds to the simulated CloudSat reflectivity, i.e. the
        *attenuated* dBZe from quickbeam_subcolumn.
    cloud_liq, cloud_ice : (nlev, ncol, *batch)
        Subcolumn cloud liquid / ice condensate (kg/kg; large-scale plus
        convective, e.g. summed from inputs.distribute_hydrometeors).
    surf_liquid_precip : (ncol, *batch) bool
        Whether the subcolumn produces liquid precipitation at the surface.
    condensate_threshold, ice_threshold : float
        "Nonzero" cutoffs for cloud condensate and layer ice (kg/kg).
    weights : (ncol, *batch), optional
        SCOPS subcolumn weight (``jcosp.subcol.scops.subcolumn_weights``).
    smoothing : jcosp.surrogate.Smoothing
        ``reflectivity`` is the width of the derivative across the -15 and
        0 dBZe thresholds; no value depends on it.

    Returns
    -------
    WarmRainFlags of (*batch) occurrence fractions over subcolumns.
    """
    dtype = dbze.dtype
    cloudy = (cloud_liq + cloud_ice) > condensate_threshold
    layer_id = _cloud_layer_ids(cloudy)
    has_ice = cloudy & (cloud_ice > ice_threshold)

    driz_thr, rain_thr = float(config.CFODD_BNDZE[0]), float(config.CFODD_BNDZE[1])
    width = smoothing.reflectivity
    driz, driz_cold = _classify_at_threshold(dbze, cloudy, has_ice, layer_id,
                                             driz_thr, width)
    rain, rain_cold = _classify_at_threshold(dbze, cloudy, has_ice, layer_id,
                                             rain_thr, width)

    wet = jnp.asarray(surf_liquid_precip).astype(dtype)
    driz = driz * wet
    rain = rain * wet
    driz_only = driz * (1.0 - rain)  # drizzle regime: -15 < max Ze <= 0

    def ncol_mean(f):
        return surrogate.occurrence(f, weights, axis=0)

    return WarmRainFlags(
        warm_rain=ncol_mean(rain * (1.0 - rain_cold)),
        cold_rain=ncol_mean(rain * rain_cold),
        warm_drizzle=ncol_mean(driz_only * (1.0 - driz_cold)),
        cold_drizzle=ncol_mean(driz_only * driz_cold),
    )


def warm_rain_fraction(flags: WarmRainFlags, include_drizzle: bool = False):
    """f_warm = warm / (warm + cold) rain occurrence, per gridbox.

    With ``include_drizzle`` the drizzle occurrences count as rain events
    (some analyses pool both regimes). Gridboxes with no raining subcolumns
    return NaN; average the numerator and denominator over time *before*
    taking the ratio to build the Mülmenstädt et al. (2015)-comparable
    climatology.
    """
    warm = flags.warm_rain
    cold = flags.cold_rain
    if include_drizzle:
        warm = warm + flags.warm_drizzle
        cold = cold + flags.cold_drizzle
    total = warm + cold
    return jnp.where(total > 0, warm / jnp.maximum(total, 1e-30), jnp.nan)
