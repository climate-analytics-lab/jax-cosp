"""Tests of the surrogate-gradient primitives.

Each primitive has to satisfy three things: its value is the hard
operation's, bit for bit; its derivative is the named smooth function's, in
both modes; and that derivative means what the module says it means, which
for a step is "the density of samples at the threshold" and is checked
against a difference quotient of a large population.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcosp import stats, surrogate
from jcosp.constants import R_GROUND, R_UNDEF


def _adjoint_gap(f, x, seed=0):
    """Return |<J t, c> - <t, J^T c>| / scale for random t and c."""
    rng = np.random.default_rng(seed)
    t = jnp.asarray(rng.normal(size=x.shape))
    y, jt = jax.jvp(f, (x,), (t,))
    c = jnp.asarray(rng.normal(size=y.shape))
    (jc,) = jax.vjp(f, x)[1](c)
    lhs, rhs = float(jnp.vdot(jt, c)), float(jnp.vdot(t, jc))
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)


class StepTest(unittest.TestCase):
    STEPS = ((surrogate.gt, np.greater), (surrogate.ge, np.greater_equal),
             (surrogate.lt, np.less), (surrogate.le, np.less_equal))

    def test_value_is_the_hard_comparison(self):
        # Ties, both sides, and the missing-value sentinels.
        x = np.array([-3.0, 0.29999, 0.3, 0.30001, 7.0, R_UNDEF, R_GROUND])
        for step, hard in self.STEPS:
            for width in (0.0, 0.05, 2.0):
                got = np.asarray(step(jnp.asarray(x), 0.3, width))
                np.testing.assert_array_equal(got, hard(x, 0.3).astype(float))
                self.assertEqual(got.dtype, np.float64)

    def test_derivative_is_the_logistic_steps(self):
        x = jnp.asarray(np.linspace(-1.0, 1.5, 41))
        width = 0.2
        for step, sign in ((surrogate.gt, 1.0), (surrogate.ge, 1.0),
                           (surrogate.lt, -1.0), (surrogate.le, -1.0)):
            got = jax.vmap(jax.grad(lambda v: step(v, 0.3, width)))(x)
            want = jax.vmap(jax.grad(
                lambda v: jax.nn.sigmoid(sign * (v - 0.3) / width)))(x)
            np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                                       rtol=1e-12)
            # ... and with respect to the threshold, with the other sign.
            got_t = jax.vmap(jax.grad(
                lambda t, v: step(v, t, width)), (None, 0))(0.3, x)
            np.testing.assert_allclose(np.asarray(got_t), -np.asarray(want),
                                       rtol=1e-12)

    def test_zero_width_is_the_reference_derivative(self):
        x = jnp.asarray(np.linspace(-1.0, 1.5, 11))
        g = jax.grad(lambda v: jnp.sum(surrogate.gt(v, 0.3, 0.0)))(x)
        np.testing.assert_array_equal(np.asarray(g), 0.0)

    def test_sentinels_have_no_derivative(self):
        x = jnp.asarray([R_UNDEF, R_GROUND, 0.2])
        g = np.asarray(jax.grad(
            lambda v: jnp.sum(surrogate.ge(v, 0.3, 0.05)))(x))
        self.assertTrue(np.isfinite(g).all())
        np.testing.assert_array_equal(g[:2], 0.0)
        self.assertGreater(g[2], 0.0)

    def test_array_width_with_hard_entries(self):
        # A zero entry of an array width is a hard edge, not a division by 0.
        x = jnp.asarray([0.25, 0.25, 0.25])
        width = jnp.asarray([0.0, 0.1, 0.5])
        g = np.asarray(jax.grad(
            lambda v: jnp.sum(surrogate.gt(v, 0.3, width)))(x))
        self.assertTrue(np.isfinite(g).all())
        self.assertEqual(g[0], 0.0)
        self.assertTrue((g[1:] > 0.0).all())

    def test_width_carries_no_derivative(self):
        g = jax.grad(lambda w: surrogate.gt(0.25, 0.3, w))(jnp.asarray(0.1))
        self.assertEqual(float(g), 0.0)

    def test_traced_width_under_jit_and_grad(self):
        # A width computed from a traced value must not leak a tracer.
        @jax.jit
        def f(x, line):
            return jnp.sum(surrogate.ge(x, line, 0.05 * jnp.abs(line)))

        x = jnp.asarray([0.9, 1.0, 1.1])
        g = np.asarray(jax.grad(f)(x, jnp.asarray([1.0, 1.0, 1.0])))
        self.assertTrue(np.isfinite(g).all() and (g > 0.0).all())

    def test_modes_are_adjoint(self):
        x = jnp.asarray(np.linspace(-1.0, 1.5, 23))
        self.assertLess(
            _adjoint_gap(lambda v: surrogate.gt(v, 0.3, 0.2) * v, x), 1e-12)

    def test_derivative_is_the_density_at_the_threshold(self):
        # A population scaled by s: the fraction above the threshold is
        # smooth in s, and its slope is what the step's derivative estimates.
        rng = np.random.default_rng(0)
        x = jnp.asarray(rng.lognormal(mean=-1.0, sigma=0.8, size=200_000))
        thr, width = 0.3, 0.01

        def fraction(s):
            return jnp.mean(surrogate.ge(s * x, thr, width))

        eps = 0.02
        fd = float((fraction(1.0 + eps) - fraction(1.0 - eps)) / (2 * eps))
        ad = float(jax.grad(fraction)(1.0))
        self.assertGreater(fd, 0.1)
        self.assertAlmostEqual(ad / fd, 1.0, delta=0.03)


class GateCombinationTest(unittest.TestCase):
    def test_any_of(self):
        g = jnp.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        val = surrogate.any_of(g, axis=1)
        np.testing.assert_array_equal(np.asarray(val), [0.0, 1.0, 1.0])
        jac = np.asarray(jax.jacobian(
            lambda a: surrogate.any_of(a, axis=1))(g))
        # None on: each can switch it on. One on: it alone can switch it
        # off. Two on: neither can.
        np.testing.assert_array_equal(jac[0, 0], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(jac[1, 1], [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(jac[2, 2], [0.0, 0.0, 0.0])

    def test_select_and_gated(self):
        gate = jnp.asarray([0.0, 1.0])
        a, b = jnp.asarray([2.0, 3.0]), jnp.asarray([5.0, 7.0])
        np.testing.assert_array_equal(
            np.asarray(surrogate.select(gate, a, b)), [5.0, 3.0])
        dg = jax.jacobian(lambda g: surrogate.select(g, a, b))(gate)
        np.testing.assert_array_equal(np.diag(np.asarray(dg)), [-3.0, -4.0])
        np.testing.assert_array_equal(
            np.asarray(surrogate.gated(gate, a)), [0.0, 3.0])
        dv = jax.jacobian(lambda v: surrogate.gated(gate, v))(a)
        np.testing.assert_array_equal(np.diag(np.asarray(dv)), [0.0, 1.0])

    def test_occurrence_value_is_the_boolean_mean(self):
        rng = np.random.default_rng(1)
        mask = rng.uniform(size=(7, 30)) > 0.4
        weights = jnp.ones(30)
        got = surrogate.occurrence(jnp.asarray(mask), weights[None], axis=1)
        want = jnp.mean(jnp.asarray(mask), axis=1)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        self.assertEqual(got.dtype, want.dtype)
        # Same from a 0/1 float gate, and without weights.
        gate = jnp.asarray(mask, jnp.float64)
        for w in (weights[None], None):
            got = surrogate.occurrence(gate, w, axis=1)
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    def test_occurrence_follows_the_weights(self):
        mask = jnp.asarray([True, False, True, True])
        g = jax.grad(lambda w: surrogate.occurrence(mask, w))(jnp.ones(4))
        np.testing.assert_allclose(np.asarray(g), [0.25, 0.0, 0.25, 0.25])


class BinMembershipTest(unittest.TestCase):
    BINS = np.array([0.0, 0.3, 1.3, 3.6, 9.4, 23.0, 60.0, 10000.0])

    def test_edge_widths(self):
        w = surrogate.edge_widths(self.BINS, 0.1)
        np.testing.assert_allclose(w[:4], [0.03, 0.03, 0.1, 0.23])
        np.testing.assert_allclose(
            np.asarray(surrogate.edge_widths(jnp.asarray(self.BINS), 0.1)), w)

    def test_value_is_the_hard_membership(self):
        rng = np.random.default_rng(2)
        x = np.concatenate([rng.lognormal(0.5, 1.5, 500), self.BINS,
                            [R_UNDEF, R_GROUND, -999.0]])
        want = np.asarray(surrogate.hard_bin_membership(
            jnp.asarray(x), self.BINS)).astype(float)
        for width in (0.0, 0.1, 0.4):
            got = surrogate.bin_membership(jnp.asarray(x), self.BINS, width)
            np.testing.assert_array_equal(np.asarray(got), want)

    def test_a_sample_leaving_a_bin_enters_its_neighbour(self):
        x = jnp.asarray(np.linspace(0.6, 50.0, 200))
        total = jax.grad(lambda v: jnp.sum(
            surrogate.bin_membership(v, self.BINS, 0.1)))(x)
        each = jax.jacobian(lambda v: jnp.sum(
            surrogate.bin_membership(v, self.BINS, 0.1), axis=1))(x)
        # The bins exchange samples, and the total changes only through
        # the outermost edges (here far away).
        self.assertGreater(float(jnp.abs(each).max()), 1.0)
        self.assertLess(float(jnp.abs(total).max()), 1e-3)

    def test_histograms_keep_their_counts(self):
        rng = np.random.default_rng(3)
        v1 = jnp.asarray(rng.lognormal(0.5, 1.5, (40, 5)))
        v2 = jnp.asarray(rng.uniform(0.0, 1100.0, (40, 5)))
        pbins = np.array([0.0, 180.0, 310.0, 440.0, 560.0, 680.0, 800.0, 1100.0])
        keep = jnp.asarray(rng.uniform(size=(40, 5)) > 0.3)
        # The pre-existing convention: deselected samples moved out of range.
        want = stats.hist2d(jnp.where(keep, v1, -999.0), v2, self.BINS, pbins)
        got = stats.hist2d(v1, v2, self.BINS, pbins,
                           weights=keep * jnp.ones((40, 1)), width=0.1)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        self.assertEqual(float(jnp.sum(got)), float(jnp.sum(keep)))
        one = stats.hist1d(v1, self.BINS, width=0.1)
        np.testing.assert_array_equal(
            np.asarray(one), np.asarray(stats.hist1d(v1, self.BINS)))

    def test_histogram_derivatives_are_finite_and_nonzero(self):
        rng = np.random.default_rng(4)
        v1 = jnp.asarray(rng.lognormal(0.5, 1.5, (40, 5)))
        pick = jnp.asarray(rng.normal(size=(7, 5)))
        g = np.asarray(jax.grad(lambda v: jnp.sum(
            pick * stats.hist1d(v, self.BINS, width=0.1)))(v1))
        g0 = np.asarray(jax.grad(lambda v: jnp.sum(
            pick * stats.hist1d(v, self.BINS)))(v1))
        self.assertTrue(np.isfinite(g).all())
        self.assertGreater(np.abs(g).max(), 0.0)
        np.testing.assert_array_equal(g0, 0.0)


class SurrogateGradientTest(unittest.TestCase):
    def test_value_and_both_modes(self):
        exact = lambda x: jnp.clip(x, 0.0, 1.0)
        smooth = lambda x: jax.nn.sigmoid(4.0 * (x - 0.5))
        f = surrogate.with_surrogate_gradient(exact, smooth)
        x = jnp.asarray(np.linspace(-1.0, 2.0, 13))
        np.testing.assert_array_equal(np.asarray(f(x)), np.asarray(exact(x)))
        np.testing.assert_allclose(
            np.asarray(jax.vmap(jax.grad(f))(x)),
            np.asarray(jax.vmap(jax.grad(smooth))(x)), rtol=1e-12)
        self.assertLess(_adjoint_gap(f, x), 1e-12)
        np.testing.assert_array_equal(np.asarray(jax.jit(f)(x)),
                                      np.asarray(exact(x)))

    def test_no_smoothing_is_all_zero(self):
        self.assertTrue(all(w == 0.0 for w in surrogate.NO_SMOOTHING))
        self.assertTrue(all(w > 0.0 for w in surrogate.DEFAULT_SMOOTHING))


if __name__ == "__main__":
    unittest.main()
