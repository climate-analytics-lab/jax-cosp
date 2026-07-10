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

## Offline Mie/DSD library (`jcosp/radar/lut/`)

- **Deduplication.** The Fortran re-runs the Bohren–Huffman Mie integration
  per LUT entry; for a fixed (class, temperature) the efficiencies on the
  fixed size grid do not depend on Re, so the generator computes them once per
  (class, T) and reuses them across Re bins via zeff's Mie-table path.
  Verified exactly equal to the direct path.
- **m_ice short-wavelength branch** (λ < 167 µm, unreachable at 94 GHz): the
  Fortran's search loop `if (alam < wl(i)) continue` is a no-op that would
  read out of bounds; the intended bracketing (searchsorted) is implemented.
- **`cr = 10./log(10.)`** is evaluated in float32 as in the Fortran's
  single-precision literals (a ~1e-8 relative effect the COSP KGO notes).
- **Rayleigh column always generated** (Fortran with `do_ray=0` would store
  zeros in `Zr_scaled`); Ze/kr are unaffected.
- **Fatal `errorMessage` paths raise `ValueError`**, including the calc_Re
  lognormal missing-Np branch that the Fortran reports non-fatally and then
  uses an undefined variable.
- **`gamma` of non-positive integers**: the Fortran returns `1.+300` (301.0,
  a transcription slip of `1.0d+300`); unreachable for physical inputs, the
  port raises.

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

## CALIPSO lidar optics (`jcosp/lidar/optics.py`)

- Port of `cosp_optics.F90:lidar_optics`, **spaceborne branch only** (the
  ground-based lidar integrates surface-up and is not ported). 532 nm is
  validated; the 355-nm (ATLID) molecular constants are plumbed through
  `wavelength` but the ATLID variant is not wired into any driver.
- The Fortran builds `zheight` by accumulation from the surface and then
  differences adjacent entries; we use the hydrostatic layer thickness
  directly (identical up to one rounding of the accumulate-then-difference).
- The lidar path keeps COSP's *own* physical constants (`rd = 287.04`,
  `grav = 9.80665`, `km = 1.38e-23`, local `rhoice = 500`), which differ from
  the radar-path values (`RD = 287.0`, `RHO_ICE = 917`); they are stored as
  `LIDAR_*` in `jcosp.config` so each ported path stays faithful to its
  source.
- **Gradient hardening.** The effective radius is clamped away from zero
  inside the `rad > 0` masked branch of the extinction so hydrometeor-free
  volumes don't poison reverse-mode gradients.
- PARASOL's `tautot_S_liq/ice` column optical depths are computed (they are
  free by-products) and returned, ready for a future PARASOL port.

## CALIPSO subcolumn stage (`jcosp/lidar/simulator.py`)

- Port of `lidar_simulator.F90:lidar_subcolumn` / `cmp_backsignal` /
  `cmp_beta`, spaceborne branch. The Fortran computes the *top layer* of the
  ATB and beta recursions unconditionally (dividing by `2*tau(1)`, which is
  always > 0 because the molecular part is included); the port applies the
  same `dtau > 0` guard to every layer, which is equivalent for all real
  inputs and keeps the degenerate branch finite for gradients.
- `cmp_beta`'s transmission-underflow fallback (`pnorm/eps` when
  `exp(-2 tau) <= epsilon`) is reproduced with the input dtype's machine
  epsilon.

## CALIPSO column stage (`jcosp/lidar/diagnostics.py`)

- Ports of `lidar_simulator.F90:lidar_column` (CALIPSO branch) and
  `COSP_CLDFRAC`, **only for COSP's default `use_vgrid=.true.`** with the
  40 x 480 m statistical grid: the model-level (`use_vgrid=.false.`) branch is
  not ported, and the 23-level "above 8.16 km" split of the phase logic is
  hard-wired to that grid exactly as in the Fortran.
- As in the Fortran, ATB, the molecular ATB, temperature, the perpendicular
  ATB and the category pressure are each regridded **linearly** to the
  statistical grid *before* the scattering ratio is formed; the category
  pressure is the layer's **lower-interface** pressure (`phalf(2:nlev+1)` in
  cosp.F90).
- **Deviation (indexing slip):** `COSP_CLDFRAC` section 4.1 (levels above
  8.16 km) uses `pplay(1, nlev)` — gridpoint 1's pressure for every
  gridpoint — when assigning the high/mid/low category of phase-classified
  clouds (section 4.2 uses `pplay(i, nlev)`). The port uses the per-point
  pressure everywhere. The transliteration test carries a
  `point0_pressure_bug` flag; with a single gridpoint (where the slip is
  invisible) the port matches the verbatim Fortran logic exactly.
- The full-attenuation scan (undefined phase below the first volume with
  SR > 30 beneath 8.16 km) reproduces the Fortran's early-exit loop: the
  saturating volume itself is still phase-classified, everything cloudy below
  it becomes phase-undefined, and volumes above 8.16 km never trigger it.
- The phase-vs-temperature counts test `tmpi/tmpl/tmpu > 0` in the Fortran;
  the port bins the profile temperature under the (disjoint) phase masks,
  identical unless a phase-classified volume had T <= 0 K (impossible: such
  volumes are cloudy, hence above ground with physical temperatures).
- `COSP_LIDAR_ONLY_CLOUD` is ported and driven exactly as cosp.F90's
  `use_vgrid` path does: on TOA-first regridded fields. Note the Fortran's
  near-surface search for the `cloudsat_tcc2` exclusion index scans from
  index 1 and expects surface-first data; on the TOA-first input it always
  lands on index 2 (a COSP quirk, reproduced as-is). A zero molecular
  backscatter would be NaN in the Fortran (triggering no thresholds); the
  port substitutes the equally inert scattering ratio 1.
- OPAQ diagnostics (`COSP_OPAQ`: opaque/thin cloud covers, z_opaque,
  emissivity — Guzman et al. 2017) are **not ported yet**; `lidarcldtype`,
  `cldtype*` and `cldthinemis` outputs are absent.

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
  diagnostics) is ported in `jcosp/modis/cfodd.py` (see below); this module
  remains the independent Mülmenstädt-style classification.

## MODIS input optics (`jcosp/modis/optics.py`)

- Ports of `cosp_optics.F90` (`cosp_simulator_optics`,
  `MODIS_OPTICS_PARTITION`, `modis_optics`) and the `modis_simulator.F90`
  Collection-6 near-IR fits (`get_g_nir`/`get_ssa_nir`). The Collection-5
  `get_g_nir_old`/`get_ssa_nir_old` fits are dead code in COSP v2 and are
  not ported.
- `build_modis_optics` mirrors the reference driver's MODIS block
  (cosp2_test.F90): only the cloud liquid/ice classes (LS + CV) count as
  cloud for MODIS, and the host's gridbox-mean `dtau_s`/`dtau_c` 0.67-um
  layer optical depths are painted onto subcolumns through the SCOPS map.
- **Gradient hardening.** The mixed-phase partition clamps the particle
  sizes and the extinction-weight denominator away from zero inside the
  where-masked branch (cells with a missing phase are masked out); the
  Fortran would produce Inf/NaN there only for inconsistent inputs
  (condensate without a size).

## MODIS simulator (`jcosp/modis/simulator.py`)

- Port of `modis_simulator.F90` `modis_subcolumn` + `modis_column`,
  including the two-stream (delta-Eddington, diffuse beam: the Fortran's
  `beam` is a compile-time constant 2, so the collimated branch is not
  ported), the adding recursion, the CO2-slicing CTP integration and the
  15-trial-radius size pseudo-retrieval.
- **ISCCP CTP substitution.** The Fortran replaces the CO2-slicing CTP of
  clouds below 700 hPa with the ISCCP simulator's `boxptop`. ISCCP is not
  ported yet, so `isccp_ctp` is an *optional* argument of
  `modis_subcolumn`/`simulate_modis`; when absent the CO2-slicing value is
  kept everywhere (deviation; low-cloud CTP statistics will shift slightly
  relative to COSP until ISCCP lands).
- **Faithfully kept call-site quirk:** `modis_subcolumn` passes an
  (nLevels+1)-element `[0, tau]` array into an nLevels-sized dummy of
  `cloud_top_pressure`, so the *bottom layer never contributes* to the CTP
  integration; a cloud confined to the bottom layer retrieves CTP = 0 (and,
  being <= 700 hPa, is *not* replaced by the ISCCP value). Reproduced and
  tested.
- **Faithfully kept quirks:** an exact zero of `predicted - observed`
  reflectance makes `interpolate_to_min`'s bracketing product 0 (not < 0),
  so the retrieval reports the fill size; the mean CTP divides by
  `max(1, count)` so cloud-free (but sunlit) points report 0, not R_UNDEF;
  the delta-Eddington reflectance is genuinely negative for strongly
  absorbing inputs (observed and predicted values compare in the same
  frame, so the size retrieval is unaffected).
- `compute_toa_reflectance` runs the adding recursion over *all* layers
  with clear layers forced to the identity (R, T) = (0, 1) instead of
  packing the cloudy ones (algebraically identical; JAX needs static
  shapes). `beta` is computed as -(rk - g1)/(rk + g1); the Fortran's
  `-r5/r4` carries a common (1 - (rk xmu)^2) factor that cancels
  algebraically and would be 0/0 at rk = 1/xmu. `sqrt(g1^2 - g2^2)` is
  clamped inside the non-conservative branch (selected away by `where` at
  conservative w0) for gradient safety.
- Column stage: joint histograms via the `hist2d` port; cosp.F90's
  surface-first reordering of the CTP histogram axis is applied by the
  `simulate_modis` driver (the `modis_column` function itself returns the
  Fortran-internal ascending-pressure order). Night points: all column
  outputs R_UNDEF except the two tau-vs-Reff histograms, which cosp.F90
  explicitly zero-fills (the LWP/IWP-vs-Reff histograms are left
  *uninitialized* by the Fortran for night points; we use R_UNDEF).
- Only the COSP v2 default tau-bin axis (7 bins including [0, 0.3)) is
  ported; the v1.4 6-bin compatibility variant is not.
- **Differentiability.** Retrieved tau, CTP, reflectances and (piecewise)
  particle size are differentiable w.r.t. the optical inputs; the size
  retrieval's trial-point selection (argmin + bracket choice) makes size a
  piecewise-smooth function with jumps only where the selected bracket
  changes. Cloud detection and phase are hard thresholds (step functions),
  as in the CloudSat flags. Tested via `jax.grad` finiteness.

## CFODD / warm-rain occurrence (`jcosp/modis/cfodd.py`)

- Port of `cosp_stats.F90:COSP_DIAG_WARMRAIN` ((c) 2018 RIAM Kyushu
  University, T. Michibata; Michibata et al. 2019 GMD). Constants
  (`CFODD_*`, `WR_NREGIME`, `CWP_THRESHOLD`, `COT_THRESHOLD`) were already
  in `jcosp.config`.
- **Index mapping.** The Fortran scans `k = Nlevels, 1, -1` over TOA-first
  data (surface upward); on the axis-0 TOA-first arrays this is index
  nlev-1 down to 0, so cloud top has the *smaller* index (kctop <= kcbtm).
  The per-column level scans (bottom/top detection, SLWC check, ICOD
  profile) are vectorized with argmax-based first/last-True reductions and
  masked reductions over the level axis; proven against a loop
  transliteration (`cfodd_test.py`).
- The `lwp == R_UNDEF` guard keys the R_UNDEF outputs exactly as the
  Fortran initialization loop: it fires both for non-sunlit points and for
  sunlit points where MODIS retrieved no liquid cloud at all.
- **Faithfully kept quirk:** the CFODD histogram is accumulated over *all*
  (subcolumn, level) cells of a screened-in gridpoint, so cells with
  dBZe >= -30 in subcolumns that failed the SLWC checks still land in the
  first ICOD bin (their ICOD stays at the 0 initialization).
- The unused `liqcfrc`/`icecfrc` arguments of the Fortran signature are
  omitted.
- `ratio**(5/3)` in the ICOD profile is clip-guarded at 0 and the cloud
  geometric thickness clamped away from zero inside the masked branch
  (single-level clouds use the bottom-level branch, as in the Fortran).
- Counting diagnostics (histograms, occurrence counts) carry no useful
  gradients; treat CFODD outputs as forward diagnostics.
- Driven by `jcosp.simulator.simulate_cloudsat_modis` exactly as cosp.F90's
  default `use_vgrid=.true.` path: temperature and the SCOPS map regridded
  linearly and dBZe log-averaged onto the 40 x 480 m grid, with
  `zlev = vgrid_zu` (the gates' upper interfaces). The model-level
  (`use_vgrid=.false.`) branch is not wired into the driver but
  `diag_warmrain` accepts model-level inputs directly.

## hist2d (`jcosp/stats.py`)

- Port of `cosp_stats.F90:hist2D`. Unlike `hist1D`, the Fortran applies no
  R_GROUND poisoning here; out-of-range samples (including R_UNDEF/-999
  fills) are simply not counted. Reproduced as-is.
