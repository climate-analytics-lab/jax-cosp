"""Tests of the SCOPS subcolumn weights (the sampling derivative).

The weight's value is one. Its derivative is supposed to make
``mean(weight * f)`` differentiate like the expectation of ``f`` over SCOPS
draws, for any statistic ``f`` of a subcolumn. That is checked directly:
the expectation is estimated with a very large number of subcolumns, where
it is smooth enough to difference, and compared with the derivative the
weights give at the ensemble size a model run uses.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcosp.subcol.scops import (generate_uniforms, scops_from_uniforms,
                                scops_with_weights, subcolumn_weights)


def _profile(nlev=20, seed=3, convective=False):
    """Two cloud decks with distinct cloud fractions at every level."""
    rng = np.random.default_rng(seed)
    z = np.linspace(0.0, 1.0, nlev)
    tca = (0.55 * np.exp(-((z - 0.75) / 0.12) ** 2)
           + 0.35 * np.exp(-((z - 0.3) / 0.08) ** 2)
           + rng.uniform(0.0, 0.03, nlev))
    tca[:2] = 0.0
    tca[9:11] = 0.0
    conv = np.zeros(nlev)
    if convective:
        conv = np.where((z > 0.45) & (z < 0.9), 0.06, 0.0)
        tca = np.maximum(tca, 1.5 * conv)
    return jnp.asarray(tca), jnp.asarray(conv)


def _statistics(frac_out, tau):
    """Subcolumn statistics of very different kinds, each (ncol,)."""
    cloudy = frac_out > 0
    column_tau = jnp.sum(jnp.where(cloudy, tau[:, None], 0.0), axis=0)
    starts = cloudy & ~jnp.concatenate(
        [jnp.zeros_like(cloudy[:1]), cloudy[:-1]], axis=0)
    return {
        "total cover": jnp.any(cloudy, axis=0),
        "thick": column_tau >= 3.0,           # a threshold on a column sum
        "level 7": cloudy[7],
        "column tau": column_tau,
        "cloud layers": jnp.sum(starts, axis=0),
        "convective": jnp.any(frac_out == 2, axis=0),
    }


class SubcolumnWeightTest(unittest.TestCase):
    def test_value_is_exactly_one(self):
        tca, conv = _profile(convective=True)
        for overlap in (1, 2, 3):
            u0, ul = generate_uniforms(jax.random.PRNGKey(0), tca.shape[0], 50)
            w = subcolumn_weights(tca, conv, u0, ul, overlap)
            self.assertEqual(w.shape, (50,))
            np.testing.assert_array_equal(np.asarray(w), 1.0)

    def test_scops_with_weights_matches_scops(self):
        tca, conv = _profile(convective=True)
        key = jax.random.PRNGKey(4)
        frac_out, w = scops_with_weights(key, tca, conv, 30)
        u0, ul = generate_uniforms(key, tca.shape[0], 30)
        np.testing.assert_array_equal(
            np.asarray(frac_out),
            np.asarray(scops_from_uniforms(tca, conv, u0, ul)))
        np.testing.assert_array_equal(np.asarray(w), 1.0)

    def test_batched_shapes_and_jit(self):
        tca, conv = _profile()
        tca_b = jnp.stack([tca, 0.5 * tca, jnp.zeros_like(tca)], axis=1)
        conv_b = jnp.zeros_like(tca_b)
        key = jax.random.PRNGKey(1)

        @jax.jit
        def cover(t):
            frac_out, w = scops_with_weights(key, t, conv_b, 40)
            return jnp.mean(w * jnp.any(frac_out > 0, axis=0), axis=0)

        val, jac = cover(tca_b), jax.jacobian(cover)(tca_b)
        self.assertEqual(val.shape, (3,))
        jac = np.asarray(jac)
        self.assertTrue(np.isfinite(jac).all())
        # Each gridpoint's cover responds to its own cloud fractions only.
        self.assertGreater(np.abs(jac[0, :, 0]).max(), 0.0)
        np.testing.assert_array_equal(jac[0, :, 1], 0.0)
        # A cloud-free column has nothing to respond with.
        np.testing.assert_array_equal(jac[2], 0.0)

    def test_weights_sum_is_conserved(self):
        # The number of subcolumns does not depend on the cloud fraction.
        tca, conv = _profile(convective=True)
        u0, ul = generate_uniforms(jax.random.PRNGKey(2), tca.shape[0], 60)
        for overlap in (1, 2, 3):
            g_tca, g_conv = jax.grad(
                lambda t, c: jnp.sum(subcolumn_weights(t, c, u0, ul, overlap)),
                argnums=(0, 1))(tca, conv)
            np.testing.assert_allclose(np.asarray(g_tca), 0.0, atol=1e-9)
            np.testing.assert_allclose(np.asarray(g_conv), 0.0, atol=1e-9)

    def test_zero_floor_switches_the_derivative_off(self):
        tca, conv = _profile()
        u0, ul = generate_uniforms(jax.random.PRNGKey(2), tca.shape[0], 20)
        frac_out = scops_from_uniforms(tca, conv, u0, ul)
        g = jax.grad(lambda t: jnp.mean(
            subcolumn_weights(t, conv, u0, ul, floor=0.0)
            * jnp.any(frac_out > 0, axis=0)))(tca)
        np.testing.assert_array_equal(np.asarray(g), 0.0)

    def test_derivative_without_weights_is_zero(self):
        # What the weights are for: the sampled map itself cannot respond.
        tca, conv = _profile()
        u0, ul = generate_uniforms(jax.random.PRNGKey(2), tca.shape[0], 20)
        g = jax.grad(lambda t: jnp.mean(jnp.any(
            scops_from_uniforms(t, conv, u0, ul) > 0, axis=0).astype(float)))(
                tca)
        np.testing.assert_array_equal(np.asarray(g), 0.0)

    def _check_against_expectation(self, overlap, convective, ncol=100,
                                   draws=160, nbig=300_000, eps=0.002):
        tca, conv = _profile(convective=convective)
        nlev = tca.shape[0]
        tau = jnp.asarray(np.linspace(0.3, 2.0, nlev))
        directions = {
            # More cloud everywhere, as a parameter change would give.
            "all levels": (tca, jnp.zeros_like(conv)),
            # One level on its own.
            "one level": (jnp.zeros(nlev).at[7].set(1.0),
                          jnp.zeros_like(conv)),
        }
        if convective:
            directions["convective"] = (jnp.zeros_like(tca), conv)
        # What a box leaving convective cloud does next is not scored (see
        # _weight_surrogate), which the cloud below a convective layer
        # shows in its response to the convective fraction.
        skip = {("convective", "column tau")}

        u0, ul = generate_uniforms(jax.random.PRNGKey(0), nlev, nbig)

        def expectation(t, c):
            frac_out = scops_from_uniforms(t, c, u0, ul, overlap)
            return {k: jnp.mean(v.astype(float))
                    for k, v in _statistics(frac_out, tau).items()}

        @jax.jit
        def estimate(key, dt, dc):
            a0, al = generate_uniforms(key, nlev, ncol)
            stat = _statistics(
                scops_from_uniforms(tca, conv, a0, al, overlap), tau)

            def means(t, c):
                w = subcolumn_weights(t, c, a0, al, overlap)
                return {k: jnp.mean(w * v.astype(float))
                        for k, v in stat.items()}

            return jax.jvp(means, (tca, conv), (dt, dc))[1]

        keys = jax.random.split(jax.random.PRNGKey(5), draws)
        for name, (dt, dc) in directions.items():
            hi = expectation(tca + eps * dt, conv + eps * dc)
            lo = expectation(tca - eps * dt, conv - eps * dc)
            est = jax.vmap(lambda k: estimate(k, dt, dc))(keys)
            for stat in hi:
                if (name, stat) in skip:
                    continue
                truth = float((hi[stat] - lo[stat]) / (2 * eps))
                mean = float(jnp.mean(est[stat]))
                stderr = float(jnp.std(est[stat])) / np.sqrt(draws)
                # Within sampling error of the estimate, plus a little for
                # the difference quotient's own noise and the floor.
                tol = 4.0 * stderr + 0.03 * abs(truth) + 0.02
                self.assertLess(
                    abs(mean - truth), tol,
                    msg=f"overlap {overlap}, {name}, {stat}: "
                        f"estimate {mean:+.4f} +- {stderr:.4f}, "
                        f"expectation {truth:+.4f}")

    def test_derivative_of_expectation_maximum_random(self):
        self._check_against_expectation(overlap=3, convective=False)

    def test_derivative_of_expectation_random(self):
        self._check_against_expectation(overlap=2, convective=False)

    def test_derivative_of_expectation_with_convection(self):
        self._check_against_expectation(overlap=3, convective=True)

    def test_small_ensembles_are_unbiased(self):
        # One level of independent draws: the cover's derivative in the
        # cloud fraction is 1 whatever the ensemble size. A baseline that
        # includes the subcolumn's own score gives (ncol - 1) / ncol
        # instead: half with two subcolumns, nothing at all with one.
        tca, conv = jnp.asarray([0.5]), jnp.zeros(1)
        keys = jax.random.split(jax.random.PRNGKey(7), 40_000)
        for overlap in (2, 3):
            for ncol in (1, 2, 5):

                @jax.jit
                def estimate(key):
                    u0, ul = generate_uniforms(key, 1, ncol)
                    cloudy = scops_from_uniforms(
                        tca, conv, u0, ul, overlap)[0] > 0
                    return jax.grad(lambda t: jnp.mean(
                        subcolumn_weights(t, conv, u0, ul, overlap)
                        * cloudy))(tca)[0]

                est = np.asarray(jax.vmap(estimate)(keys))
                stderr = est.std() / np.sqrt(est.size)
                self.assertLess(
                    abs(est.mean() - 1.0), 4.0 * stderr,
                    msg=f"overlap {overlap}, {ncol} subcolumns: "
                        f"{est.mean():+.4f} +- {stderr:.4f}")

    def test_maximum_overlap_coherent_change(self):
        # Under maximum overlap the subcolumn thresholds are a fixed
        # stratified set, so a single ensemble is one deterministic estimate
        # with no draws to average; a coherent change of all levels is the
        # derivative it resolves well.
        tca, conv = _profile()
        nlev = tca.shape[0]
        u0, ul = generate_uniforms(jax.random.PRNGKey(0), nlev, 400)

        def cover(t):
            frac_out = scops_from_uniforms(t, conv, u0, ul, 1)
            return jnp.any(frac_out > 0, axis=0).astype(float)

        stat = cover(tca)
        ad = float(jax.jvp(
            lambda t: jnp.mean(subcolumn_weights(t, conv, u0, ul, 1) * stat),
            (tca,), (tca,))[1])
        # Maximum overlap: the cover is the largest cloud fraction.
        self.assertAlmostEqual(ad, float(jnp.max(tca)), delta=0.02)


if __name__ == "__main__":
    unittest.main()
