# Fixed upstream bugs

As of 2026-07-10, jax-cosp's policy is to **fix outright bugs found in the
reference Fortran rather than reproduce them**, even where that creates
numerical mismatches with COSP v2. Every fix here (a) is demonstrably a
defect, not a modelling choice, (b) is covered by a test asserting the
corrected behaviour plus a transliteration test documenting the difference,
and (c) has been reported upstream. Modelling choices, tuned approximations
and vintage physics (e.g. the Liebe-85 water-vapour continuum) are still
ported faithfully; those are documented in `translation_notes.md`.

Where a fix can materially change values, a `legacy_*` switch reproduces the
reference behaviour so the compiled-Fortran KGO harness (`tools/kgo`) can
keep validating the port exactly.

| # | Component | Defect | Fix / switch | Impact | Upstream |
|---|---|---|---|---|---|
| 1 | `radar/gases.py` | O2 line-overlap interference term mis-transcribed from Liebe (1985): mirror resonance with wrong sign and the near-resonance denominator | Rosenkranz/Liebe form; `legacy_o2_interference=True` restores | W-band dry-air attenuation was ~4–5× high (0.23 vs literature ~0.05 dB/km one-way); moist columns ~15% high | [#159](https://github.com/CFMIP/COSPv2.0/issues/159) |
| 2 | `modis/simulator.py` `cloud_top_pressure` | (nLevels+1) actual argument truncated by nLevels dummy: bottom layer never enters the CO2-slicing integral; bottom-layer cloud → CTP = 0, escaping the ISCCP substitution | integrate every layer | low-cloud CTP bias; CTP=0 artefacts | [#160](https://github.com/CFMIP/COSPv2.0/issues/160) |
| 3 | `radar/lut/mie.py` `m_ice` | short-wavelength branch search loop is a no-op (`continue` ≠ C continue) → out-of-bounds interpolation | intended bracketing (searchsorted) | latent at 94 GHz (branch unreachable) | [#161](https://github.com/CFMIP/COSPv2.0/issues/161) |
| 4 | `subcol/scops.py` | top-level threshold uses `ran(npoints)` — one scalar for all gridpoints | independent uniform per gridpoint | domain-correlated top-level subcolumns; chunking-dependent | [#162](https://github.com/CFMIP/COSPv2.0/issues/162) |
| 5 | `radar/quickbeam.py` | gate-2 hydrometeor attenuation (d_gate=1) neither accumulates gate 1 nor uses its own layer spacing, inconsistent with the gas sum and the opposite direction | trapezoid accumulation; `legacy_gate2=True` restores | offset propagates to all gates below whenever the top two layers attenuate; negligible for typical spaceborne CloudSat use | [#163](https://github.com/CFMIP/COSPv2.0/issues/163) |
| 6 | `modis/cfodd.py` | CFODD histogram counts cells of subcolumns that failed the SLWC screening (their ICOD stays 0 → contaminates ICOD bin 0) | only in-cloud cells of SLWC-passing subcolumns are samples | cleaner low-ICOD column in mixed scenes | [#164](https://github.com/CFMIP/COSPv2.0/issues/164) |
| 7 | `modis/simulator.py` `interpolate_to_min` | exact reflectance match at a trial radius returns the fill value (strict `< 0` bracket test) | boundary case accepted (`<= 0`), returns the matching radius | rare (measure-zero) retrieval failures removed | noted in [#160](https://github.com/CFMIP/COSPv2.0/issues/160) |
| 8 | `modis/simulator.py` column stage | sunlit cloud-free mean CTP reported as 0 Pa (`max(1, count)` divisor) instead of undefined | `R_UNDEF` like every other mean | fill-value consistency | noted in [#160](https://github.com/CFMIP/COSPv2.0/issues/160) |
| 9 | `lidar/diagnostics.py` CLDFRAC | phase-category pressure test indexes point 1's pressure for all points (`pplay(1,·)`) | per-point pressure; `point0_pressure_bug` flag restores | high/mid/low phase split correct away from point 1 | (predates policy; see translation_notes lidar section) |
| 10 | `isccp` (cosp.F90 post-processing) | stale snap of totalcldarea "within 1e-5 of 1.0" to 1.0 — predates the percent conversion, so it can only mis-snap ~1% covers | not ported | unreachable below ~1e7 subcolumns | (documented in translation_notes ISCCP section) |

Additional defects fixed from the start of the port (misuse would have been
UB or crashes; no switch provided): the `calc_Re` lognormal missing-Np branch
that the Fortran reports non-fatally and then reads an undefined variable
(raises `ValueError` here), and `gamma()` of non-positive integers returning
`1.+300` = 301.0 (a transcription slip of `1.0d+300`; raises here).

## What this means for comparisons with Fortran COSP

- The `tools/kgo` harness runs the port in legacy modes and still matches the
  compiled reference to ~2e-7 (see `validation.md`).
- Simulated dBZe with default settings is slightly **higher** than Fortran
  COSP along dry paths (less spurious gas attenuation); CloudSat precip
  flags and warm-rain classification shift accordingly at the margins.
- MODIS CTP for bottom-layer cloud and mixed-scene CFODDs differ by design.
