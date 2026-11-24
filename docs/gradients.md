# Gradients through the simulators

COSP's diagnostics are, almost all of them, counts of subcolumns that pass
hard tests: a cloud fraction counts the subcolumns whose optical depth is at
least 0.3, a histogram counts the ones in each bin, a warm-rain frequency
counts the ones whose reflectivity exceeds 0 dBZe. A count is piecewise
constant in everything it depends on. Its derivative is exactly zero, so a
calibration against a MODIS cloud fraction or a CFAD has nothing to follow.

This package keeps the reference formulation as the forward model and gives
it usable derivatives separately, following the construction JAX-GCM uses
for its own physics (`jcm.physics.surrogate_gradient`):

- **Every value is the hard operation's, bit for bit.** Nothing here changes
  an output. The outputs of all drivers were compared bitwise with the
  previous release in float64 and float32.
- **Every derivative is that of a named function**, taken by automatic
  differentiation of that function (`jcosp.surrogate.with_surrogate_gradient`),
  so forward and reverse mode are adjoint by construction.

The consequence a user has to keep in mind: **the gradient is not the
derivative of the value.** A finite-difference check of a simulator output
disagrees with `jax.grad` wherever a surrogate is active, by design. What
the gradient estimates is the derivative of the output's *expectation*, which
is the smooth quantity a calibration actually wants to move.

Two kinds of decision occur, and they are handled differently.

## Thresholds on continuous quantities

Detection limits, reflectivity and scattering-ratio thresholds, cloud-top
phase and pressure categories, histogram bin edges. A model change moves the
tested quantity by a small amount, and the number of subcolumns that cross
the threshold is the density of subcolumns at the threshold times that
amount. The derivative of a logistic step of half-width `w` is a kernel
estimate of that density, so each test `x > t` is evaluated as
`jcosp.surrogate.gt(x, t, w)`: the value of the comparison, the derivative
of `sigmoid((x - t) / w)`.

The widths are static configuration held in a `jcosp.surrogate.Smoothing`,
which every driver and column-stage function takes as `smoothing=`:

| Field | Default | Applies to |
|---|---|---|
| `optical_depth` | 0.03 | MODIS and ISCCP detection at tau = 0.3 |
| `reflectivity` | 1 dB | -30, -15, 0 dBZe; 2C-PRECIP-COLUMN class limits; PIA limits |
| `scattering_ratio` | 0.1 x threshold | CALIPSO SR > 5, SR > 0.01, SR > 30 |
| `pressure` | 1500 Pa | MODIS high/mid/low cloud-top-pressure limits |
| `phase_fraction` | 0.05 | MODIS / imager cloud-top phase; frozen fraction of precipitation; the CALIPSO phase line (as a share of the line) |
| `temperature` | 1 K | 2-m temperature classes; false-liquid / false-ice temperatures; the CFODD warm-top test |
| `histogram` | 0.1 x narrower adjacent bin | every CFAD and joint-histogram bin edge |
| `cloud_fraction` | 0.01 | the SCOPS sampling derivative (below); a bound, not a width |

A larger width averages over more subcolumns: a smoother, less noisy
gradient that reflects subcolumns further from the threshold. A zero width
gives the reference derivative for that kind of test, and
`jcosp.surrogate.NO_SMOOTHING` for all of them.

Histograms are differences of the steps at their edges, so a sample leaving
one bin enters its neighbour and a histogram's total changes only through
its outermost edges.

## The subcolumn sampler

SCOPS turns cloud fractions into a subcolumn cloud map by comparing them
with a threshold per subcolumn. This is a threshold too, but it cannot be
treated like the others. A subcolumn that becomes cloudy gains the whole
in-cloud optical depth at once: the change is not small, so pushing "a
little more cloud" through the simulator as if it were a small change of
optical depth is wrong, and badly so for anything nonlinear such as a
detection limit.

What changes smoothly is the probability of each subcolumn's cloud profile.
The derivative of an expectation with respect to a parameter of the
distribution is `E[f * d(log p)]`, for any `f` however discontinuous. So the
sampler returns, with the cloud map, a **weight per subcolumn**
(`jcosp.subcol.scops.subcolumn_weights`) whose value is exactly one and
whose derivative is that of the log-probability of the subcolumn's profile
under SCOPS. Every statistic over subcolumns multiplies its samples by the
weight: `mean(weight * f)` is the plain mean, and its derivative is an
unbiased estimate of how the expectation of `f` responds to the cloud and
convective fractions.

SCOPS is not a sequence of independent draws. Under maximum-random overlap a
subcolumn carries its threshold down through contiguous cloud, so the
log-probability is accumulated by tracking the interval that threshold is
uniform on given the levels above.

Measured against difference quotients of 3x10^5-subcolumn ensembles, for
100-subcolumn ensembles on a two-deck profile:

| Change | Statistic | Expectation | Estimate (mean of 300 draws +- std of one) |
|---|---|---|---|
| all cloud fractions scaled | total cover | +0.522 | +0.516 +- 0.05 |
| | fraction with column tau >= 3 | +0.526 | +0.529 +- 0.04 |
| | mean column tau | +4.56 | +4.52 +- 0.52 |
| one level | that level's cover | +0.962 | +0.999 +- 0.28 |
| | total cover | 0.000 | -0.002 +- 0.13 |

(maximum-random overlap; random overlap agrees equally well.) For MODIS
total cloud fraction against a scaling of the model cloud fraction, the
derivative from 50 000 subcolumns agrees with the difference quotient to
within its 1 % noise.

Things to know:

- **It is an estimate with sampling noise**, like the statistics themselves.
  A coherent change (a parameter that moves cloud fraction at many levels
  the same way) is resolved well by one 100-subcolumn ensemble; the
  response to a single level is noisy and needs averaging over gridpoints
  and time steps, which a calibration does anyway. The noise is unbiased.
- **`cloud_fraction` is a probability floor.** A subcolumn state rarer than
  this is weighted as if it had this probability, which bounds the weight a
  rare state can receive at the cost of under-weighting levels whose cloud
  fraction is below it. Zero switches the sampling derivative off.
- **Maximum overlap** (`overlap=1`) uses a fixed stratified set of
  thresholds rather than draws, so one ensemble gives one deterministic
  estimate: good for coherent changes, coarse for single levels.
- **A cloud fraction of exactly zero has no sampled cloud to weight**, so
  the derivative with respect to it is zero from below and missing from
  above.
- **The in-cloud water responds too.** `distribute_hydrometeors` divides
  gridbox means by the sampled cloud fraction; with `weights=` that fraction
  carries its derivative, so more cloud at the same gridbox-mean water is
  thinner in-cloud.
- **A caller's own subcolumn map carries no sampling derivative** unless it
  supplies matching weights.

## What a host has to do

The drivers thread everything themselves:

```python
out = simulate_cloudsat_modis(state, lut, dtau_s, dtau_c, p_half, key=key,
                              sunlit=sunlit)
loss = jnp.mean((out.modis.column.cf_total - observed) ** 2)   # has a gradient
```

Two cases need the weight passed by hand.

A simulator run on the same SCOPS draw outside the joint driver takes the
radar run's weight:

```python
cs = simulate_cloudsat(state, lut, key=key)
hyd = distribute_hydrometeors(cs.frac_out, cs.prec_frac, ...,
                              weights=cs.subcolumn_weight)
col = lidar_column(..., weights=cs.subcolumn_weight)
```

A statistic the host forms itself from per-subcolumn output multiplies by
it:

```python
cover = jnp.mean(cs.subcolumn_weight * (cs.dbze.max(axis=0) > -30.0), axis=0)
```

Without the weight such a statistic keeps its value and has no derivative
with respect to the cloud fraction through the sampling.

## What has no derivative

Discrete choices of *which* level or bin of a fixed grid: the cloud-top and
cloud-bottom levels of the CFODD scan, the level at which a lidar profile
saturates, the ISCCP level match, the pressure category and temperature bin
of a model level, the CFODD particle-size class and gridpoint screening, the
trial-radius bracket of the MODIS size retrieval, and the precipitation
cascade of PREC_SCOPS. The quantities computed on either side of these are
differentiated as usual.

## Float32

The derivatives are finite in float32, the precision a host model runs in,
and that is tested separately from the float64 suite. Three masked-out
branches needed hardening for it, none of which changes a value: a guard of
the form `max(x, 1e-30)` keeps a branch finite in float64 but can overflow
float32 one operation later, and a masked branch that evaluates to infinity
returns NaN to reverse mode (`0 * inf`) for *every* output, including ones
that never read it. They were the slope parameter of the precipitation size
distribution at flux-free cells (`inputs.precip_mxratio`), which made the
gradient of any MODIS output with respect to a precipitation flux NaN; the
inverse albedo fit at cloud-free points (`isccp_column`); and the CFODD
optical-depth profile above a single-level cloud.

## For optimisers

First-order methods are unaffected in practice: they need a direction that
reduces the cost, and the surrogate provides one where the reference
provides none. Methods that compare the predicted with the actual change of
the cost (a line search with a curvature condition, a trust-region ratio)
can reject a good step, because the two differ near a threshold and the
actual change of a count is a staircase.

## Tests

`jcosp/surrogate_test.py` checks each primitive: value, derivative, both
modes, and that a step's derivative is the density at the threshold.
`jcosp/subcol/weights_test.py` checks the sampling derivative against the
expectation as in the table above. `jcosp/gradients_test.py` checks, for
every column-stage output of every simulator, that the value does not
depend on the smoothing, that the derivatives are finite in both modes and
adjoint, and that the output responds to the cloud fraction and to the
in-cloud quantities its instrument sees; it repeats the finiteness check in
a float32 process, and checks the MODIS cloud fraction's derivative in the
model cloud fraction against a 50 000-subcolumn difference quotient.
