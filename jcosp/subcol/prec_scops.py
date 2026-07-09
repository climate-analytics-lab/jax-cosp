"""PREC_SCOPS: distribute precipitation into SCOPS subcolumns, in JAX.

Ported from COSP v2.0 ``subsample_and_optics_example/subcol/prec_scops.F90``
(Copyright (c) 2008, Lawrence Livermore National Security Limited Liability
Corporation, BSD-3-Clause; see LICENSES/COSP-BSD-3-Clause-LLNS.txt).

Where a level has nonzero large-scale (convective) precipitation flux, the
level's subcolumns are flagged as precipitating by a cascade of five
possibilities, each tried only if the previous one selected no subcolumn at
that gridpoint:

1. subcolumns cloudy (of matching type) at this level, or — below the top
   level — already precipitating (of matching type) in the level above;
2. (folded into 1 in the Fortran loop structure);
3. subcolumns cloudy at the level *below* (precipitation falling out of a
   cloud base into the layer where the flux is diagnosed);
4. subcolumns with matching-type cloud *anywhere* in the column;
5. every subcolumn (large-scale) or the first ``scops_ccfrac*ncol``
   subcolumns (convective).

Output codes: 0 none, 1 large-scale, 2 convective, 3 both.

Array convention: level axis 0 (TOA-first), subcolumn axis 1, batch axes
trailing, matching :mod:`jcosp.subcol.scops`.
"""

import jax.numpy as jnp
from jax import lax

from jcosp.constants import SCOPS_CCFRAC


def _tiered_mask(tier1, tier3, tier4, tier5):
    """Select the first tier with any flagged subcolumn, per gridpoint.

    Each tier is a boolean ``(ncol, *batch)`` mask; ``any`` reductions are
    over the subcolumn axis, reproducing the Fortran flag_ls/flag_cv logic.
    """
    any1 = jnp.any(tier1, axis=0, keepdims=True)
    any3 = jnp.any(tier3, axis=0, keepdims=True)
    any4 = jnp.any(tier4, axis=0, keepdims=True)
    return jnp.where(any1, tier1,
                     jnp.where(any3, tier3,
                               jnp.where(any4, tier4, tier5)))


def prec_scops(frac_out, ls_p_rate, cv_p_rate):
    """Assign precipitation type to each subcolumn.

    Parameters
    ----------
    frac_out : (nlev, ncol, *batch)
        SCOPS subcolumn cloud types (0 clear / 1 stratiform / 2 convective).
    ls_p_rate, cv_p_rate : (nlev, *batch)
        Gridbox-mean large-scale and convective precipitation fluxes (any
        units; only ``> 0`` matters).

    Returns
    -------
    prec_frac : (nlev, ncol, *batch) int32
        0 none, 1 large-scale, 2 convective, 3 both.
    """
    nlev, ncol = frac_out.shape[:2]
    cv_col = int(SCOPS_CCFRAC * ncol)
    if cv_col == 0:
        cv_col = 1

    # Does each subcolumn contain LS / CV cloud anywhere in the column?
    col_has_ls = jnp.any(frac_out == 1, axis=0)
    col_has_cv = jnp.any(frac_out == 2, axis=0)

    # Tier-5 convective mask: the first cv_col subcolumns.
    cv_first = (jnp.arange(ncol) < cv_col).reshape(
        (ncol,) + (1,) * (frac_out.ndim - 2))
    cv_first = jnp.broadcast_to(cv_first, frac_out.shape[1:])
    all_cols = jnp.ones(frac_out.shape[1:], dtype=bool)

    # frac_out at the level below each level; clear below the bottom level,
    # which disables tier 3 there exactly as the Fortran's ilev<nlev guard.
    frac_below = jnp.concatenate(
        [frac_out[1:], jnp.zeros_like(frac_out[:1])], axis=0)

    def step(prec_above, inputs):
        frac_k, frac_bk, ls_k, cv_k = inputs

        # Large-scale precipitation.
        ls_tier1 = (frac_k == 1) | (prec_above == 1) | (prec_above == 3)
        ls_mask = _tiered_mask(ls_tier1, frac_bk == 1, col_has_ls, all_cols)
        ls_mask &= ls_k > 0.0

        # Convective precipitation (evaluated after LS at the same level, so
        # overlap with fresh LS flags produces code 3).
        cv_tier1 = (frac_k == 2) | (prec_above == 2) | (prec_above == 3)
        cv_mask = _tiered_mask(cv_tier1, frac_bk == 2, col_has_cv, cv_first)
        cv_mask &= cv_k > 0.0

        prec_k = jnp.where(ls_mask, 1, 0)
        prec_k = jnp.where(cv_mask, jnp.where(prec_k == 0, 2, 3), prec_k)
        return prec_k.astype(jnp.int32), prec_k.astype(jnp.int32)

    prec0 = jnp.zeros(frac_out.shape[1:], dtype=jnp.int32)
    _, prec_frac = lax.scan(
        step, prec0,
        (frac_out, frac_below, ls_p_rate[:, None], cv_p_rate[:, None]))
    return prec_frac
