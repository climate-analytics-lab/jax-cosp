"""SCOPS: Subgrid Cloud Overlap Profile Sampler, in JAX.

Ported from COSP v2.0 ``subsample_and_optics_example/subcol/scops.F90``
(Copyright (c) 2008, Lawrence Livermore National Security Limited Liability
Corporation, BSD-3-Clause; see LICENSES/COSP-BSD-3-Clause-LLNS.txt). SCOPS is
originally part of the ISCCP simulator (Klein & Jakob 1999; Webb et al. 2001).

Given per-level stratiform and convective cloud fractions, SCOPS assigns each
of ``ncol`` subcolumns a cloud type at every level (0 clear, 1 stratiform,
2 convective) such that the horizontal cloud fraction is reproduced in
expectation and the vertical arrangement follows the requested overlap
assumption (1 maximum, 2 random, 3 maximum/random). Convective cloud is always
maximally overlapped and pinned to the same subcolumns (the first
``conv*ncol`` box positions).

Array convention: vertical level on axis 0 (index 0 = top of atmosphere, as
in COSP), then the subcolumn axis, then any broadcast batch axes:
``cloud_frac`` is ``(nlev, *batch)`` and the output is ``(nlev, ncol, *batch)``.

Deviations from the Fortran (documented in docs/translation_notes.md):

* Random numbers come from ``jax.random`` (threefry) instead of the congruent
  generator seeded from surface pressure, so subcolumn realizations differ
  bit-wise but agree statistically.
* At the top level the Fortran initializes the threshold with
  ``ran(npoints)`` — the *last point's* random number applied to every point
  (an apparent indexing slip for the intended ``ran(1:npoints)``). We draw an
  independent value per point.
"""

import jax
import jax.numpy as jnp
from jax import lax


def generate_uniforms(key, nlev: int, ncol: int, batch_shape: tuple = ()):
    """Draw the uniform variates SCOPS consumes.

    Returns ``(u_init, u_levels)`` with shapes ``(ncol, *batch)`` and
    ``(nlev, ncol, *batch)``: one draw per subcolumn for the top-level
    threshold initialization and one per (level, subcolumn) for the
    threshold updates.
    """
    k1, k2 = jax.random.split(key)
    u_init = jax.random.uniform(k1, (ncol, *batch_shape))
    u_levels = jax.random.uniform(k2, (nlev, ncol, *batch_shape))
    return u_init, u_levels


def scops_from_uniforms(cloud_frac, conv_frac, u_init, u_levels, overlap: int = 3):
    """Deterministic SCOPS given externally supplied uniform variates.

    Parameters
    ----------
    cloud_frac : (nlev, *batch)
        Total cloud fraction per level (the Fortran ``cc``/``tca``); must
        already include the convective part.
    conv_frac : (nlev, *batch)
        Convective cloud fraction per level.
    u_init : (ncol, *batch)
        Uniforms for the top-level threshold (unused for maximum overlap).
    u_levels : (nlev, ncol, *batch)
        Uniforms for the per-level threshold resets.
    overlap : int
        1 maximum, 2 random, 3 maximum/random (default, as in COSP).

    Returns
    -------
    frac_out : (nlev, ncol, *batch) int32
        Subcolumn cloud type: 0 clear, 1 stratiform, 2 convective.
    """
    if overlap not in (1, 2, 3):
        raise ValueError(f"overlap must be 1, 2 or 3, got {overlap}")
    ncol = u_levels.shape[1]
    tca = cloud_frac

    # Ordered subcolumn positions (i - 0.5)/ncol, shaped to broadcast against
    # (ncol, *batch) threshold arrays.
    boxpos = (jnp.arange(ncol) + 0.5) / ncol
    boxpos = boxpos.reshape((ncol,) + (1,) * (u_levels.ndim - 2))

    # Threshold before the first level: evenly spread positions for maximum
    # overlap, otherwise random within the non-convective part of the box.
    if overlap == 1:
        thresh0 = jnp.broadcast_to(boxpos, u_levels.shape[1:])
    else:
        thresh0 = conv_frac[0] + (1.0 - conv_frac[0]) * u_init

    # tca at the level above; zero above the model top (the DS2014 fix for
    # the ilev=1 bounds in the Fortran is equivalent to this padding).
    tca_above = jnp.concatenate([jnp.zeros_like(tca[:1]), tca[:-1]], axis=0)

    def step(threshold, inputs):
        conv_k, tca_k, tca_km1, ran_k = inputs
        # Boxes pinned to convective cloud (maximally overlapped by position).
        maxocc = boxpos <= conv_k
        if overlap == 1:
            threshold_min = jnp.broadcast_to(conv_k, threshold.shape)
            maxosc = jnp.ones_like(threshold, dtype=bool)
        elif overlap == 2:
            threshold_min = jnp.broadcast_to(conv_k, threshold.shape)
            maxosc = jnp.zeros_like(threshold, dtype=bool)
        else:
            overlap_frac = jnp.minimum(tca_km1, tca_k)
            threshold_min = jnp.broadcast_to(
                jnp.maximum(conv_k, overlap_frac), threshold.shape)
            # Maximally overlap the part of the box that was cloudy above.
            maxosc = (threshold < overlap_frac) & (threshold > conv_k)

        threshold = jnp.where(
            maxocc,
            jnp.broadcast_to(boxpos, threshold.shape),
            jnp.where(maxosc, threshold,
                      threshold_min + (1.0 - threshold_min) * ran_k),
        )
        frac_k = jnp.where(tca_k > threshold, 1, 0)
        frac_k = jnp.where((threshold <= conv_k) & (conv_k > 0.0), 2, frac_k)
        return threshold, frac_k.astype(jnp.int32)

    # Broadcast per-level scalars against the subcolumn axis inside the scan.
    conv_b = conv_frac[:, None]
    tca_b = tca[:, None]
    tca_above_b = tca_above[:, None]
    _, frac_out = lax.scan(step, thresh0, (conv_b, tca_b, tca_above_b, u_levels))
    return frac_out


def scops(key, cloud_frac, conv_frac, ncol: int, overlap: int = 3):
    """Stochastic SCOPS: sample subcolumn cloud types (see scops_from_uniforms)."""
    nlev = cloud_frac.shape[0]
    u_init, u_levels = generate_uniforms(key, nlev, ncol, cloud_frac.shape[1:])
    return scops_from_uniforms(cloud_frac, conv_frac, u_init, u_levels, overlap)
