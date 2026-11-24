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

from jcosp import surrogate


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


def _weight_surrogate(cloud_frac, conv_frac, u_init, u_levels, overlap, floor):
    """Evaluate the function whose derivative ``subcolumn_weights`` carries.

    ``1 + sum over decisions of (p - stop_gradient(p)) / max(p, floor)``,
    where ``p`` is the probability SCOPS had of taking each decision it
    took for this subcolumn. Its value is one and its derivative is the
    derivative of the log-probability of the subcolumn's whole cloud
    profile: the likelihood-ratio (score-function) weight, for which

        d/d(theta) E[f] = E[f * d log p / d(theta)]

    holds for any statistic ``f`` of a subcolumn, however discontinuous.

    SCOPS is not a sequence of independent draws: under maximum-random
    overlap a subcolumn carries its threshold down through contiguous
    cloud, so the probability of what it does at one level depends on
    what it did above. Given the path so far, the carried threshold is
    uniform on an interval ``[lo, hi)`` that the path narrows level by
    level, and each decision's probability is the share of that interval
    on the side the subcolumn took. The convectively pinned boxes are
    treated the same way through the interval of their box position.

    One simplification: a box leaving convective cloud carries its box
    position as its threshold, and whether that is then kept or redrawn is
    not scored. Scoring it was tried and is worse: the box positions are a
    fixed stratified set, not draws, so with the handful of convective
    boxes an ensemble has, the two outcomes do not balance. The effect is
    confined to the response of the cloud just below a convective layer to
    the convective fraction.

    The mean over subcolumns is removed at the end, so the weights sum to
    ``ncol`` to first order (the number of subcolumns does not change) and
    a statistic that is the same for every subcolumn has exactly zero
    derivative. ``floor`` bounds the weight a rare state can receive.
    """
    ncol = u_levels.shape[1]
    dtype = u_levels.dtype
    sg = lax.stop_gradient
    tca = cloud_frac
    boxpos = (jnp.arange(ncol) + 0.5) / ncol
    boxpos = boxpos.reshape((ncol,) + (1,) * (u_levels.ndim - 2)).astype(dtype)
    shape = u_levels.shape[1:]
    if overlap == 1:
        thresh0 = jnp.broadcast_to(boxpos, shape)
    else:
        thresh0 = conv_frac[0] + (1.0 - conv_frac[0]) * u_init
    tca_above = jnp.concatenate([jnp.zeros_like(tca[:1]), tca[:-1]], axis=0)
    tiny = jnp.finfo(dtype).tiny ** 0.5

    def share(lo, hi, within_lo, within_hi):
        """Probability that a value uniform on [lo, hi) lies in [within)."""
        length = jnp.clip(jnp.minimum(hi, within_hi)
                          - jnp.maximum(lo, within_lo), 0.0, None)
        return jnp.clip(length / jnp.maximum(hi - lo, tiny), 0.0, 1.0)

    def term(prob, taken):
        """Score of one decision: d(prob) / max(prob, floor) where taken."""
        prob = jnp.where(taken, prob, 1.0 - prob)
        return (prob - sg(prob)) / sg(jnp.maximum(prob, floor))

    def step(carry, inputs):
        threshold, lo, hi, known, box_lo, box_hi, acc = carry
        conv_k, tca_k, tca_km1, ran_k = inputs
        conv_b = jnp.broadcast_to(conv_k, shape)
        tca_b = jnp.broadcast_to(tca_k, shape)
        pos = jnp.broadcast_to(boxpos, shape)
        zero = jnp.zeros(shape, dtype)
        one = jnp.ones(shape, dtype)

        # The decisions SCOPS took, exactly as scops_from_uniforms.
        maxocc = sg(pos <= conv_b)
        if overlap == 1:
            threshold_min = conv_b
            keep_hi = one
            maxosc = jnp.ones(shape, bool)
        elif overlap == 2:
            threshold_min = conv_b
            keep_hi = zero
            maxosc = jnp.zeros(shape, bool)
        else:
            keep_hi = jnp.broadcast_to(jnp.minimum(tca_km1, tca_k), shape)
            threshold_min = jnp.maximum(conv_b, keep_hi)
            maxosc = sg((threshold < keep_hi) & (threshold > conv_b))
        redrawn = threshold_min + (1.0 - threshold_min) * ran_k
        threshold = jnp.where(maxocc, pos,
                              jnp.where(maxosc, threshold, redrawn))
        cloudy = sg(tca_b > threshold)

        # Pinned to convective cloud, or not: the box position is uniform
        # on [box_lo, box_hi) given the levels above.
        acc = acc + term(share(box_lo, box_hi, zero, conv_b), maxocc)
        box_hi = jnp.where(maxocc, jnp.minimum(box_hi, conv_b), box_hi)
        box_lo = jnp.where(maxocc, box_lo, jnp.maximum(box_lo, conv_b))

        # A carried threshold is kept between conv and the overlap bound
        # (always, for maximum overlap). One that is a pinned box's position
        # is treated as known: what it does next is not scored.
        free = ~maxocc & ~known
        keep_lo = zero if overlap == 1 else conv_b
        if overlap != 1:
            acc = acc + jnp.where(
                free, term(share(lo, hi, keep_lo, keep_hi), maxosc), 0.0)
        kept_lo = jnp.maximum(lo, keep_lo)
        kept_hi = jnp.minimum(hi, keep_hi)

        # Cloudy or clear. A kept threshold is below the cloud fraction by
        # construction under maximum-random overlap; under maximum overlap
        # it is compared with it. A redrawn one is uniform above its floor.
        if overlap == 1:
            acc = acc + jnp.where(
                free, term(share(kept_lo, kept_hi, zero, tca_b), cloudy), 0.0)
            kept_lo, kept_hi = (jnp.where(cloudy, kept_lo,
                                          jnp.maximum(kept_lo, tca_b)),
                                jnp.where(cloudy,
                                          jnp.minimum(kept_hi, tca_b),
                                          kept_hi))
        fresh = ~maxocc & ~maxosc
        acc = acc + jnp.where(
            fresh, term(share(threshold_min, one, zero, tca_b), cloudy), 0.0)
        fresh_lo = jnp.where(cloudy, threshold_min,
                             jnp.maximum(threshold_min, tca_b))
        fresh_hi = jnp.where(cloudy, jnp.maximum(tca_b, threshold_min), one)

        lo = jnp.where(maxosc, kept_lo, fresh_lo)
        hi = jnp.where(maxosc, kept_hi, fresh_hi)
        known = jnp.where(maxocc, True, jnp.where(maxosc, known, False))
        return (threshold, lo, hi, known, box_lo, box_hi, acc), None

    zeros = jnp.zeros(shape, dtype)
    ones = jnp.ones(shape, dtype)
    init = (thresh0.astype(dtype), zeros, ones, jnp.zeros(shape, bool),
            zeros, ones, zeros)
    carry, _ = lax.scan(
        step, init,
        (conv_frac[:, None], tca[:, None], tca_above[:, None], u_levels))
    acc = carry[-1]
    return 1.0 + acc - jnp.mean(acc, axis=0, keepdims=True)


def subcolumn_weights(cloud_frac, conv_frac, u_init, u_levels,
                      overlap: int = 3,
                      floor: float = surrogate.DEFAULT_SMOOTHING.cloud_fraction):
    """Weight of each SCOPS subcolumn: exactly one, with a derivative.

    SCOPS turns cloud fractions into a cloud map by comparing them with a
    threshold per subcolumn, so the map, and every statistic counted over
    it, is piecewise constant in the cloud fractions: its derivative is
    zero. What does vary smoothly is how probable each subcolumn's cloud
    profile is. This weight carries that variation. Its value is 1.0, so
    ``mean(weight * f)`` over subcolumns is the plain mean bit for bit;
    its derivative is that of the log-probability of the subcolumn's
    profile (see ``_weight_surrogate``), which makes the derivative of
    that mean an unbiased estimate of how the expectation of ``f`` over
    SCOPS draws responds to the cloud and convective fractions.

    Every statistic over subcolumns in this package takes the weight
    (``weights=``). A statistic formed outside it from per-subcolumn
    output has to multiply by it too, or it sees no sensitivity to the
    cloud fraction through the sampling.

    Parameters are those of :func:`scops_from_uniforms`. ``floor`` is the
    smallest probability a decision is credited with in the derivative
    (a state rarer than this is weighted as if it had this probability);
    0 switches the derivative off.

    Returns
    -------
    weight : (ncol, *batch)
    """
    if overlap not in (1, 2, 3):
        raise ValueError(f"overlap must be 1, 2 or 3, got {overlap}")
    dtype = u_levels.dtype
    cloud_frac = jnp.asarray(cloud_frac, dtype)
    conv_frac = jnp.asarray(conv_frac, dtype)

    def exact(cloud_frac, conv_frac, u_init, u_levels):
        # Exactly one, formed from the variates rather than as a constant:
        # under vmap a constant value is unbatched while its derivative is
        # batched, which jit's linearization of a custom rule rejects.
        return u_levels[0] * 0.0 + 1.0

    if not floor:
        return exact(cloud_frac, conv_frac, u_init, u_levels)

    def smooth(cloud_frac, conv_frac, u_init, u_levels):
        return _weight_surrogate(cloud_frac, conv_frac, u_init, u_levels,
                                 overlap, floor)

    return surrogate.with_surrogate_gradient(exact, smooth)(
        cloud_frac, conv_frac, u_init, u_levels)


def scops_with_weights(key, cloud_frac, conv_frac, ncol: int,
                       overlap: int = 3,
                       floor: float = surrogate.DEFAULT_SMOOTHING.cloud_fraction):
    """Sample subcolumn cloud types and their weights from one draw.

    Returns ``(frac_out, weight)``: the map of :func:`scops` and the
    matching :func:`subcolumn_weights`.
    """
    nlev = cloud_frac.shape[0]
    u_init, u_levels = generate_uniforms(key, nlev, ncol, cloud_frac.shape[1:])
    frac_out = scops_from_uniforms(cloud_frac, conv_frac, u_init, u_levels,
                                   overlap)
    weight = subcolumn_weights(cloud_frac, conv_frac, u_init, u_levels,
                               overlap, floor)
    return frac_out, weight
