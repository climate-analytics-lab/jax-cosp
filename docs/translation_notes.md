# Translation notes: COSP v2 Fortran → jax-cosp

Documented, intentional differences between this port and the reference
Fortran (COSPv2.0, github.com/CFMIP/COSPv2.0). Everything not listed here is
meant to be algorithm- and value-faithful; the `*_test.py` suites check the
ported routines against loop-for-loop transliterations of the Fortran, and
`gases()` against values from the compiled Fortran itself.

## Global

- **Array convention.** Level axis 0, TOA-first (COSP's ordering), then the
  subcolumn axis, then broadcast batch axes. The Fortran's
  `(npoints, ncolumns, nlevels)` becomes `(nlevels, ncolumns, ...batch)`,
  following the JAX-GCM column-physics convention (vertical on axis 0).
- **Precision.** The tests run in float64 (`conftest.py`), where
  transliteration comparisons are exact. Under JAX's default float32 the
  physics is unchanged but LUT bin selection (temperature / effective-radius
  quantization) can shift by one slot at exact bin edges.
- **Sentinels.** `R_UNDEF = -1e30` and `R_GROUND = -1e20` are kept (not NaN)
  for output compatibility and because the CloudSat flag logic tests against
  them.

## SCOPS (`jcosp/subcol/scops.py`)

- **RNG.** `jax.random` (Threefry) replaces the congruential generator seeded
  from surface pressure. Realizations differ bit-wise; statistics agree
  (tested). Deterministic comparisons use `scops_from_uniforms`, which accepts
  externally supplied uniforms.
- **Top-level threshold seed.** The Fortran initializes the level-1 threshold
  with `ran(npoints)` — the *last* gridpoint's random number applied to every
  gridpoint (an apparent indexing slip for `ran(1:npoints)`). We draw an
  independent uniform per gridpoint.

## Quickbeam optics (`jcosp/radar/optics.py`, `jcosp/radar/lut/`)

- **Full LUT precomputation.** The Fortran fills its Z-scale table lazily at
  run time (first Mie/DSD integration per (class, T-bin, Re-bin) is cached;
  `Ze = Ze_scaled · ρ_air · q` afterwards). We precompute the entire table
  offline; the runtime is the pure lookup. Consequences:
  - Slots the Fortran never caches (`iRe_type == 1`, i.e. Re quantizing below
    the first slot, and `iRe_type >= nRe_types`, Re ≳ 2.5 mm) are computed
    ab initio *per volume* in the Fortran with the unquantized Re; we use the
    nearest table slot instead. Both regimes are physically marginal
    (sub-0.15-µm and multi-mm effective radii).
  - The 2-moment (`HCLASS2`, Morrison) hydrometeor tables are **not yet
    supported**: their liquid classes set `p3 = -2` (Martin-1994 shape
    parameter from number concentration), so a (class, T, Re)-keyed table is
    ill-posed — the Fortran itself caches whatever Np it saw first for a bin,
    which is order-dependent. Supporting 2M properly needs an extra
    shape-parameter axis in the LUT (planned).
- **Re-bin arithmetic.** The Fortran `base/step` mapping is reproduced
  exactly, including its single-slot overlaps at a few coarse-interval
  boundaries (Re = 90, 120, 150, 160, 170, 180 µm), where the mapping is
  non-monotone by one slot.
- **Gradient hardening.** Inputs to `calc_Re` are clamped away from zero for
  hydrometeor-free volumes (which are masked out of the result) so that
  reverse-mode gradients aren't poisoned by `0 · inf` through `where`.

## Gaseous attenuation (`jcosp/radar/gases.py`)

- **Pressure unit.** The Fortran argument is named `PRES_mb` but the driver
  passes Pa (the internal `/1000` factors produce kPa); we take Pa and say so.
- **Coefficient precision.** The Fortran's Liebe coefficient literals are
  single precision (no `_wp` suffix); our tables hold the decimal values in
  float64. Agreement with the compiled Fortran is ~2·10⁻⁸ relative.
- **Faithfully kept quirk.** In the O2 line sum, the delta-correction term
  uses `(aux3/x - aux4/x)` — both denominators the *near*-resonance one — as
  in the Fortran (Liebe's formulation would use the two respective
  denominators).

## Quickbeam attenuation (`jcosp/radar/quickbeam.py`)

- The Fortran's two gate-direction branches are asymmetric (increasing-gate
  hydrometeor attenuation leaves the first *two* gates un-accumulated, all
  other combinations one). Reproduced exactly; see
  `_attenuation_to_volume(two_start=...)`.

## Vertical regrid / stats (`jcosp/stats.py`)

- `COSP_CHANGE_VERTICAL_GRID`'s weight case-tree reduces to the clipped
  interval intersection; implemented that way (verified against the
  transliteration, including `log_units`, surface flagging and R_UNDEF
  handling).

## Warm-rain classification (`jcosp/classify.py`)

- Original implementation of the Mülmenstädt et al. (2020) Methods text (not
  a Fortran port). Assumptions made explicit:
  - The thresholds (−15 / 0 dBZe) are applied to the **attenuated** simulated
    reflectivity (what CloudSat measures).
  - "Produce liquid precipitation at the surface" = rain-water mixing ratio
    in the subcolumn's lowest level > 0.
  - "Nonzero cloud condensate" thresholds default to exactly 0 and are
    exposed as parameters.
- COSP's own `COSP_DIAG_WARMRAIN` (Michibata et al. 2019 CFODD/regime
  diagnostics) additionally requires MODIS-simulator retrievals and will be
  ported together with the MODIS simulator (Phase 5).
