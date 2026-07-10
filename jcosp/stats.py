"""Statistical post-processing: histograms and vertical regridding, in JAX.

Ported from COSP v2.0 ``src/cosp_stats.F90`` (Copyright (c) 2015, Regents of
the University of Colorado, BSD-3-Clause; see LICENSES/).

Array convention: level axis 0, TOA-first, subcolumn axis next, batch axes
trailing (see jcosp.subcol). COSP_CHANGE_VERTICAL_GRID internally requires
surface-to-TOA ordering in the Fortran; the flip is handled here so callers
stay TOA-first throughout, exactly as quickbeam_column does in the Fortran.
"""

import jax.numpy as jnp

from jcosp.constants import R_GROUND, R_UNDEF


def hist1d(var, bins):
    """Per-bin counts over the subcolumn axis (cosp_stats.F90:hist1D).

    Parameters
    ----------
    var : (ncol, *batch)
        Samples (one histogram per batch element, counting over subcolumns).
    bins : (nbins+1,)
        Bin boundaries [lowest, tops...].

    Returns
    -------
    (nbins, *batch) counts; R_UNDEF wherever any sample equals R_GROUND
    (below-surface levels poison the whole histogram, as in the Fortran).
    """
    bins = jnp.asarray(bins)
    lo = bins[:-1].reshape((-1,) + (1,) * var.ndim)
    hi = bins[1:].reshape((-1,) + (1,) * var.ndim)
    counts = jnp.sum((var[None] >= lo) & (var[None] < hi), axis=1).astype(var.dtype)
    any_ground = jnp.any(var == R_GROUND, axis=0)
    return jnp.where(any_ground[None], R_UNDEF, counts)


def hist2d(var1, var2, bins1, bins2):
    """Joint histogram counts over the sample axis (cosp_stats.F90:hist2D).

    Parameters
    ----------
    var1, var2 : (n, *batch)
        Paired samples (one joint histogram per batch element, counting over
        axis 0 — e.g. subcolumns, or flattened subcolumn x level cells).
    bins1 : (nbin1+1,), bins2 : (nbin2+1,)
        Ascending bin boundaries for each variable.

    Returns
    -------
    (nbin1, nbin2, *batch) counts. A sample lands in bin (i, j) when
    ``bins1[i] <= var1 < bins1[i+1]`` and ``bins2[j] <= var2 < bins2[j+1]``;
    out-of-range samples (including R_UNDEF fills) are simply not counted.
    Unlike ``hist1d``, the Fortran hist2D applies no R_GROUND poisoning.
    """
    bins1 = jnp.asarray(bins1)
    bins2 = jnp.asarray(bins2)
    shape = (-1, 1) + (1,) * (var1.ndim - 1)
    m1 = ((var1[None] >= bins1[:-1].reshape(shape))
          & (var1[None] < bins1[1:].reshape(shape)))     # (nbin1, n, *batch)
    m2 = ((var2[None] >= bins2[:-1].reshape(shape))
          & (var2[None] < bins2[1:].reshape(shape)))     # (nbin2, n, *batch)
    # Contract over the sample axis without materializing the
    # (nbin1, nbin2, n, *batch) intermediate.
    return jnp.einsum("in...,jn...->ij...", m1.astype(var1.dtype),
                      m2.astype(var1.dtype))


def change_vertical_grid(zfull, zhalf, y, newgrid_bot, newgrid_top,
                         log_units: bool = False):
    """Layer-thickness-weighted regrid onto a fixed height grid.

    Port of COSP_CHANGE_VERTICAL_GRID. All height arguments in metres.

    Parameters
    ----------
    zfull : (nlev, *batch)
        Height of model layer midpoints, TOA-first.
    zhalf : (nlev, *batch)
        Height of the *bottom* interface of each layer, TOA-first.
    y : (nlev, ncol, *batch)
        Field to regrid.
    newgrid_bot, newgrid_top : (nglev,)
        Bottom/top of the new layers, TOA-first (e.g. config.VGRID_ZL/ZU).
    log_units : bool
        True for dBZ-like fields: average in linear units, return dBZ, with
        R_UNDEF where the average is not positive.

    Returns
    -------
    (nglev, ncol, *batch), with R_GROUND for new levels below the surface.
    """
    # Old-grid layer bounds (computed in TOA-first order): the bottom of layer
    # l is zhalf[l]; its top is the bottom of the layer above, except the top
    # layer which is symmetric about its midpoint (Fortran oldgrid_top(Nlevels)).
    old_bot = zhalf
    old_top = jnp.concatenate(
        [2.0 * zfull[:1] - zhalf[:1], zhalf[:-1]], axis=0)

    nglev = newgrid_bot.shape[0]
    new_bot = jnp.asarray(newgrid_bot).reshape((nglev,) + (1,) * zhalf.ndim)
    new_top = jnp.asarray(newgrid_top).reshape((nglev,) + (1,) * zhalf.ndim)

    # Overlap length between every (new k, old l) layer pair; the Fortran's
    # dbb/dtb/dbt/dtt case tree computes exactly this clipped intersection.
    w = jnp.clip(jnp.minimum(old_top[None], new_top)
                 - jnp.maximum(old_bot[None], new_bot), 0.0, None)  # (nglev, nlev, *b)

    if log_units:
        yp = jnp.where(y == R_UNDEF, 0.0, jnp.power(10.0, y / 10.0))
    else:
        yp = y
    # Weighted average over old levels. einsum keeps the contraction from
    # materializing a (nglev, nlev, ncol, *batch) intermediate; the weights
    # are shared across the subcolumn axis.
    wt = jnp.sum(w, axis=1)                                    # (nglev, *b)
    num = jnp.einsum("kl...,lc...->kc...", w, yp)              # (nglev, ncol, *b)
    r = jnp.where(wt[:, None] > 0.0, num / jnp.maximum(wt[:, None], 1e-30), 0.0)

    if log_units:
        r = jnp.where(r <= 0.0, R_UNDEF, 10.0 * jnp.log10(jnp.maximum(r, 1e-30)))

    # New levels whose top is at/below the surface (bottom interface of the
    # lowest model layer) are flagged as ground.
    surface = zhalf[-1]  # (*batch)
    above_ground = (jnp.asarray(newgrid_top).reshape(
        (nglev,) + (1,) * surface.ndim) > surface[None])  # (nglev, *batch)
    return jnp.where(above_ground[:, None], r, R_GROUND)
