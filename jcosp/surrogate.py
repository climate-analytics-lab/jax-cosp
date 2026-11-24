"""Reference-exact values with the derivatives of a smooth surrogate.

COSP is full of hard decisions: a subcolumn is cloudy or clear, a retrieval
is detected or not, a sample falls in one histogram bin. Each is part of the
reference formulation, and the value this package returns stays exactly
that. What is useless for gradient-based calibration is the derivative: a
count of subcolumns is piecewise constant in everything it depends on, so
automatic differentiation returns exactly zero for a cloud fraction, a
histogram or an occurrence frequency.

The functions here separate the two, following the construction JAX-GCM
uses for its physics (``jcm.physics.surrogate_gradient``): the value is the
hard operation's, bit for bit, and the derivatives, in forward and reverse
mode, are those of a smooth function named next to it. Two kinds of
decision occur, and they need different treatment.

**A threshold on a continuous quantity** (optical depth of at least 0.3, a
reflectivity above -15 dBZe, a histogram bin edge). The model moves the
quantity by a small amount, and the fraction of subcolumns crossing the
threshold is the density of subcolumns at the threshold times that amount.
The derivative of a logistic step of half-width ``width`` is a kernel
estimate of that density: :func:`gt`, :func:`ge`, :func:`lt`, :func:`le`
and the histograms built on them (:func:`bin_membership`).

**The cloudy/clear state of a subcolumn** (SCOPS). Here the perturbation is
not small: a subcolumn that becomes cloudy gains the whole in-cloud
optical depth at once, so pushing "a little more cloud" through the
simulator as if it were a small change of optical depth gives a wrong
answer. What changes smoothly is the *probability* of each subcolumn's
cloud profile, so the derivative is carried by a weight on each subcolumn
instead (``jcosp.subcol.scops.subcolumn_weights``): its value is exactly
one and its derivative is that of the profile's log-probability. Every
statistic over subcolumns multiplies its samples by that weight.

The widths are configuration of the derivative, not physical parameters:
no value depends on them. They are static Python numbers held in a
:class:`Smoothing`; a width of zero selects the reference derivative for
that kind of decision, and :data:`NO_SMOOTHING` for all of them.

See ``docs/gradients.md`` for what this means for a calibration.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "Smoothing", "DEFAULT_SMOOTHING", "NO_SMOOTHING",
    "with_surrogate_gradient", "gt", "ge", "lt", "le", "any_of", "select",
    "gated", "occurrence", "edge_widths", "bin_membership",
    "hard_bin_membership",
]


class Smoothing(NamedTuple):
    """Half-widths of the smooth steps that define the derivatives.

    Each is in the units of the quantity being thresholded and sets the
    range over which the derivative sees the hard step as a ramp. Larger
    widths give smoother, less noisy gradients that are biased towards the
    behaviour of subcolumns further from the threshold; zero gives the
    reference derivative (identically zero across the step). No output
    value depends on any of them. ``cloud_fraction`` is the exception in
    kind: the SCOPS sampling derivative needs no smoothing, only a bound.
    """

    # SCOPS: smallest probability a subcolumn's sampled state is credited
    # with in the sampling derivative (0 switches that derivative off).
    cloud_fraction: float = 0.01
    # Detection limits on a column optical depth (MODIS/ISCCP tau = 0.3).
    optical_depth: float = 0.03
    # Radar reflectivity thresholds and path-integrated attenuation (dB).
    reflectivity: float = 1.0
    # Lidar scattering-ratio thresholds, as a fraction of the threshold.
    scattering_ratio: float = 0.1
    # Cloud-top-pressure categories (Pa).
    pressure: float = 1500.0
    # Liquid/ice shares: cloud-top phase, frozen fraction of precipitation.
    phase_fraction: float = 0.05
    # Temperature thresholds (K).
    temperature: float = 1.0
    # Histogram bin edges, as a fraction of the narrower adjacent bin.
    histogram: float = 0.1


DEFAULT_SMOOTHING = Smoothing()
NO_SMOOTHING = Smoothing(*([0.0] * len(Smoothing._fields)))


def with_surrogate_gradient(exact: Callable, surrogate: Callable) -> Callable:
    """Return ``f`` with ``exact``'s value and ``surrogate``'s derivatives.

    ``f(*args)`` evaluates ``exact(*args)`` and nothing else. Under
    ``jax.jvp``, ``jax.grad`` and every transformation built on them the
    tangent of ``f`` is the tangent of ``surrogate`` at the same arguments,
    obtained by differentiating ``surrogate`` itself, so forward and reverse
    mode are adjoint by construction.

    Both functions take the same positional arguments and return outputs of
    the same structure, shape and dtype. Everything a caller may
    differentiate with respect to must be an argument: a traced value
    reached through a closure is invisible to the derivative rule.
    """

    @jax.custom_jvp
    def f(*args):
        return exact(*args)

    @f.defjvp
    def f_jvp(primals, tangents):
        # The value comes from ``f`` rather than ``exact`` so that a second
        # differentiation meets this rule again instead of differentiating
        # the hard operation's kinks.
        primal_out = f(*primals)
        _, tangent_out = jax.jvp(surrogate, primals, tangents)
        tangent_out = jax.tree.map(
            lambda t, p: t.astype(p.dtype), tangent_out, primal_out)
        return primal_out, tangent_out

    return f


def _float_dtype(*xs):
    dtype = jnp.result_type(*xs)
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.result_type(float)
    return dtype


def _step(compare, sign, x, threshold, width):
    """Hard comparison as a 0/1 float with a logistic step's derivative."""
    dtype = _float_dtype(x, threshold)
    if isinstance(width, (int, float)) and not width:
        return compare(x, threshold).astype(dtype)
    x = jnp.asarray(x, dtype)
    threshold = jnp.asarray(threshold, dtype)
    width = jnp.asarray(width, dtype)
    shape = jnp.broadcast_shapes(x.shape, threshold.shape, width.shape)

    def exact(x, threshold, width):
        return jnp.broadcast_to(compare(x, threshold), shape).astype(dtype)

    def surrogate(x, threshold, width):
        # The width shapes the derivative only: it is held fixed, so no
        # derivative is taken with respect to it. It is an argument rather
        # than a closure because it may be traced (a width relative to a
        # computed threshold), and a derivative rule must not capture a
        # tracer. A zero entry means a hard edge there.
        width = jax.lax.stop_gradient(width)
        on = width > 0
        soft = jax.nn.sigmoid(
            sign * (x - threshold) / jnp.where(on, width, 1.0))
        return jnp.where(on, soft, exact(x, threshold, width)).astype(dtype)

    return with_surrogate_gradient(exact, surrogate)(x, threshold, width)


def gt(x, threshold, width):
    """``x > threshold`` as a 0/1 float, differentiated as a logistic step."""
    return _step(jnp.greater, 1.0, x, threshold, width)


def ge(x, threshold, width):
    """``x >= threshold`` as a 0/1 float, differentiated as a logistic step."""
    return _step(jnp.greater_equal, 1.0, x, threshold, width)


def lt(x, threshold, width):
    """``x < threshold`` as a 0/1 float, differentiated as a logistic step."""
    return _step(jnp.less, -1.0, x, threshold, width)


def le(x, threshold, width):
    """``x <= threshold`` as a 0/1 float, differentiated as a logistic step."""
    return _step(jnp.less_equal, -1.0, x, threshold, width)


def any_of(gate, axis):
    """Logical OR of 0/1 gates along ``axis``, as a 0/1 float.

    The derivative is that of ``1 - prod(1 - gate)``: where no gate is on
    every one of them can switch the result on, where exactly one is on it
    alone can switch it off, and where several are on none can.
    """
    def exact(gate):
        return jnp.max(gate, axis=axis)

    def surrogate(gate):
        return 1.0 - jnp.prod(1.0 - gate, axis=axis)

    return with_surrogate_gradient(exact, surrogate)(gate)


def select(gate, on_value, off_value):
    """``where(gate, on_value, off_value)`` that is differentiable in ``gate``.

    The derivative is that of ``gate * on_value + (1 - gate) * off_value``.
    Both values must be meaningful numbers wherever the gate has a
    derivative; a missing-value sentinel belongs in a plain ``jnp.where``.
    """
    def exact(gate, on_value, off_value):
        return jnp.where(gate > 0.5, on_value, off_value)

    def surrogate(gate, on_value, off_value):
        return gate * on_value + (1.0 - gate) * off_value

    dtype = _float_dtype(gate, on_value, off_value)
    args = [jnp.asarray(a, dtype) for a in (gate, on_value, off_value)]
    shape = jnp.broadcast_shapes(*[a.shape for a in args])
    args = [jnp.broadcast_to(a, shape) for a in args]
    return with_surrogate_gradient(exact, surrogate)(*args)


def gated(gate, value):
    """``where(gate, value, 0)`` with the derivative of ``gate * value``."""
    return select(gate, value, 0.0)


def occurrence(gate, weights=None, axis=0):
    """Fraction of samples along ``axis`` whose 0/1 ``gate`` is on.

    The value is ``jnp.mean`` of the boolean ``gate > 0.5``, the operation
    COSP's occurrence fractions are ported as (including its result dtype).
    The derivative is that of ``mean(weights * gate)``: it follows the
    gate's own derivative and, through ``weights`` (the SCOPS subcolumn
    weight), the sampling's.
    """
    gate = jnp.asarray(gate)
    if gate.dtype == jnp.bool_:
        gate = gate.astype(_float_dtype(gate) if weights is None
                           else jnp.asarray(weights).dtype)
    weights = jnp.ones((), gate.dtype) if weights is None else weights

    def exact(gate, weights):
        return jnp.mean(gate > 0.5, axis=axis)

    def surrogate(gate, weights):
        return jnp.mean(weights * gate, axis=axis)

    return with_surrogate_gradient(exact, surrogate)(gate, weights)


def edge_widths(bins, relative_width):
    """Step half-width of each histogram bin edge.

    ``relative_width`` times the width of the narrower bin adjacent to the
    edge, so that a sample is "near" an edge on the scale of the bins it
    separates whatever their units and however unevenly they are spaced.
    """
    xp = np if isinstance(bins, np.ndarray) else jnp
    gaps = xp.diff(bins)
    inf = xp.full((1,), xp.inf, gaps.dtype)
    return relative_width * xp.minimum(xp.concatenate([inf, gaps]),
                                       xp.concatenate([gaps, inf]))


def hard_bin_membership(x, bins):
    """Boolean ``bins[i] <= x < bins[i+1]``, shape ``(nbins, *x.shape)``."""
    bins = jnp.asarray(bins)
    lo = bins[:-1].reshape((-1,) + (1,) * x.ndim)
    hi = bins[1:].reshape((-1,) + (1,) * x.ndim)
    return (x[None] >= lo) & (x[None] < hi)


def bin_membership(x, bins, relative_width):
    """Float ``bins[i] <= x < bins[i+1]`` with a smooth derivative in ``x``.

    Returns ``(nbins, *x.shape)``. Each membership is the difference of the
    steps at its two edges, so a sample leaving one bin enters its
    neighbour: the derivative of a histogram's total is exactly that of
    the samples entering and leaving its outer edges.
    """
    edges = jnp.asarray(bins)
    dtype = _float_dtype(x, edges)
    if not relative_width:
        return hard_bin_membership(x, edges).astype(dtype)
    widths = edge_widths(bins if isinstance(bins, np.ndarray) else edges,
                         relative_width)
    shape = (-1,) + (1,) * x.ndim
    above = ge(x[None], edges.reshape(shape),
               jnp.asarray(widths, dtype).reshape(shape))
    return above[:-1] - above[1:]
