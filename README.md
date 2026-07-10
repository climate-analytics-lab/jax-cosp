# jax-cosp

**Differentiable satellite simulators in JAX** — a Python/JAX translation of the
[CFMIP Observation Simulator Package (COSP v2)](https://github.com/CFMIP/COSPv2.0),
built to give differentiable climate models (in particular
[JAX-GCM](https://github.com/climate-analytics-lab/jax-gcm)) observation-space
diagnostics: quantities a satellite would actually measure, simulated from model
state, so model and observations can be compared like-for-like.

The first milestone is the **CloudSat radar path** — SCOPS/PREC_SCOPS subcolumn
sampling and the Quickbeam 94-GHz reflectivity simulator — which is what the
**warm-rain-fraction** diagnostic of Mülmenstädt et al. (2015, 2020) is defined
on. Phase 4 added the **CALIPSO lidar path** (`jcosp.lidar.simulate_calipso`):
532-nm optics, total + perpendicular attenuated backscatter, and the
GOCCP-style scattering-ratio CFAD, cloud-fraction and thermodynamic-phase
diagnostics. Phase 5 added the **MODIS simulator**
(`jcosp.modis.simulate_modis`: per-subcolumn cloud mask / CTP / optical
thickness / phase / particle-size retrievals, column means and joint
histograms) and the **CFODD + warm-rain-occurrence diagnostics** of
Michibata et al. (2019) that join MODIS retrievals with the CloudSat radar
(`jcosp.simulator.simulate_cloudsat_modis`). Later phases grow the package
toward the full COSP instrument suite (ISCCP, MISR, PARASOL).

## Status: alpha (Phases 0–5 of the translation plan)

| Component | COSP v2 origin | Status |
| --- | --- | --- |
| Config / hydrometeor classes | `cosp_config.F90`, `hydro_class_init` | ported |
| SCOPS cloud subcolumns | `subcol/scops.F90` | ported |
| PREC_SCOPS precip subcolumns | `subcol/prec_scops.F90` | ported |
| Precip flux → mixing ratio | `optics/cosp_utils.F90` | ported |
| Mie / DSD optics (offline LUT) | `optics_lib.F90`, `quickbeam_optics.F90` | ported |
| Quickbeam radar (runtime) | `quickbeam.F90`, `quickbeam_optics.F90` | ported |
| Gaseous attenuation (Liebe 85) | `quickbeam_optics.F90:gases` | ported |
| CFAD / vertical regrid / precip flags | `cosp_stats.F90`, `quickbeam.F90` | ported |
| Warm-rain classification (Mülmenstädt et al. 2020) | original implementation | done |
| CFODD / warm-rain regimes (`COSP_DIAG_WARMRAIN`) | `cosp_stats.F90` | ported (vgrid path) |
| MODIS input optics (tau merge, phase partition, 3.7-µm g/w0) | `cosp_optics.F90`, driver MODIS block | ported |
| MODIS subcolumn retrievals (mask, CTP, tau, phase, size) | `modis_simulator.F90` | ported (ISCCP CTP substitution pending ISCCP) |
| MODIS column means + joint histograms (tau/CTP, tau/Reff, LWP-IWP/Reff) | `modis_simulator.F90`, `cosp_stats.F90:hist2D` | ported |
| CALIPSO lidar optics (532 nm, sphere + non-sphere ice) | `cosp_optics.F90:lidar_optics` | ported |
| CALIPSO ATB + perpendicular channel | `actsim/lidar_simulator.F90` | ported |
| CALIPSO SR / CFAD / cloud + phase fractions (GOCCP) | `lidar_simulator.F90:COSP_CLDFRAC` | ported (vgrid path) |
| Lidar-only cloud / radar-lidar TCC | `cosp_stats.F90:COSP_LIDAR_ONLY_CLOUD` | ported |
| CALIPSO OPAQ diagnostics (`COSP_OPAQ`) | `lidar_simulator.F90` | planned |
| ATLID / ground lidar variants | `lidar_simulator.F90` | planned |
| ISCCP / MISR / PARASOL | — | planned |

## Design

- **Pure functional JAX.** Every runtime routine is jit/vmap/grad-compatible:
  static shapes, no Python branching on traced values.
- **Vertical axis 0, TOA-first.** Arrays are `(nlev, ...)` with level index 0 at
  the model top (COSP's native ordering); trailing axes (subcolumn, gridpoint)
  broadcast NumPy-style, following the JAX-GCM column-physics convention.
- **Radar optics as a precomputed lookup table.** COSP's Quickbeam already
  computes reflectivity as `Ze = Ze_scaled(class, T_bin, Re_bin) · ρ_air · q`,
  filling its scale table lazily with Mie/DSD integrations. We precompute that
  entire table offline (exact NumPy port of the Mie code, `jcosp.radar.lut`) so
  the runtime simulator is a differentiable table lookup — numerically the same
  quantization the Fortran applies, with none of the iterative Mie loops in the
  computation graph.
- **Stochastic subcolumns (option A of the translation plan).** SCOPS is ported
  faithfully with `jax.random`; the warm-rain fraction is a forward diagnostic.
  Hosts using a McICA generator (e.g. JAX-GCM's) can instead inject their own
  cloud masks so radiation and the satellite simulator see the same subgrid
  cloud field. A relaxed/differentiable sampler is a planned research spike.
- **Host hooks live in the host.** This package is model-agnostic; the JAX-GCM
  `PhysicsTerm` wrapper lives in the JAX-GCM repository.

## Install

```bash
pip install -e .[test]
JAX_PLATFORMS=cpu pytest -n 8
```

## License and attribution

jax-cosp is Apache-2.0 licensed (see `LICENSE`). It is a derived work of
COSP v2, which is BSD-3-Clause licensed by the Regents of the University of
Colorado, Lawrence Livermore National Security LLC, and RIAM Kyushu University;
the original license texts are preserved in `LICENSES/` and the attribution is
detailed in `NOTICE`. Each ported module cites its source file and original
authors. If you use this package, please cite Swales et al. (2018,
doi:10.5194/gmd-11-77-2018) alongside it.

## References

- Swales, D. J., Pincus, R., Bodas-Salcedo, A. (2018). COSP2. *GMD* 11, 77–81.
- Haynes, J. M., et al. (2007). A multipurpose radar simulation package:
  QuickBeam. *BAMS* 88, 1723–1727.
- Mülmenstädt, J., et al. (2015). *GRL* 42, 6502–6509. (warm-rain climatology)
- Mülmenstädt, J., et al. (2020). *Sci. Adv.* 6, eaaz6433. (COSP warm-rain constraint)
- Michibata, T., et al. (2019). *GMD* 12, 4297–4315. (CFODD / warm-rain regimes)
