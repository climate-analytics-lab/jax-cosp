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
(`jcosp.simulator.simulate_cloudsat_modis`). Phase 6 added the **ISCCP
simulator** (`jcosp.isccp.simulate_isccp`: ICARUS 10.5-µm brightness
temperatures, IR-adjusted cloud-top pressures, tau/CTP joint histogram),
whose `boxptop` completes the MODIS low-cloud CTP substitution in the
combined driver. Later phases grow the package toward the full COSP
instrument suite (MISR, PARASOL).

The package also carries a small number of diagnostics that are **not in
COSP** but are built on its simulators, listed under
[Beyond COSP](#beyond-cosp) below: the warm-rain classification and the
**imager water paths under another product's definition**
(`jcosp.modis.imager`, ESA Cloud_cci).

## Status: alpha (Phases 0–6 of the translation plan)

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
| MODIS subcolumn retrievals (mask, CTP, tau, phase, size) | `modis_simulator.F90` | ported (incl. ISCCP CTP substitution) |
| MODIS column means + joint histograms (tau/CTP, tau/Reff, LWP-IWP/Reff) | `modis_simulator.F90`, `cosp_stats.F90:hist2D` | ported |
| Imager water paths under another product's definition (ESA Cloud_cci CC4CL) | not in COSP: original implementation (McGarragh et al. 2018, Eq. 60) on the MODIS subcolumn retrievals | done (`jcosp.modis.imager`) |
| CALIPSO lidar optics (532 nm, sphere + non-sphere ice) | `cosp_optics.F90:lidar_optics` | ported |
| CALIPSO ATB + perpendicular channel | `actsim/lidar_simulator.F90` | ported |
| CALIPSO SR / CFAD / cloud + phase fractions (GOCCP) | `lidar_simulator.F90:COSP_CLDFRAC` | ported (vgrid path) |
| Lidar-only cloud / radar-lidar TCC | `cosp_stats.F90:COSP_LIDAR_ONLY_CLOUD` | ported |
| CALIPSO OPAQ diagnostics (`COSP_OPAQ`) | `lidar_simulator.F90` | planned |
| ATLID / ground lidar variants | `lidar_simulator.F90` | planned |
| ISCCP (ICARUS: Tb, IR-adjusted CTP, tau/CTP histogram) | `simulator/icarus/icarus.F90` | ported |
| MISR / PARASOL | — | planned |

## Beyond COSP

Everything in the table marked "ported" is a translation of COSP v2 and is
meant to reproduce it. Two diagnostics are additions with no COSP
counterpart; they reuse the ported simulators' subcolumn output and leave
the ported results unchanged.

- **Warm-rain classification** (`jcosp.classify`): the Mülmenstädt et al.
  (2020) warm/cold rain and drizzle flags on the CloudSat reflectivity.
- **Imager water paths under another product's definition**
  (`jcosp.modis.imager`). A visible/near-infrared imager does not see a
  column's liquid and ice water paths: its retrieval treats a cloudy pixel
  as one layer of one phase, converts column optical depth and cloud-top
  effective radius with `4/3 · tau · r_e · rho / Q_e`, and books the whole
  column under the cloud-top phase, so liquid beneath an ice top is reported
  as ice. COSP's MODIS column stage is the MODIS L3 form of this. Other
  products differ only in the conversion constants, the retrieval limits
  and the phase rule, which an `ImagerRetrieval` holds;
  `imager_water_paths` applies one to the subcolumn retrievals the MODIS
  simulator has already made, at no further radiative-transfer cost. The
  packaged definition, `CLOUD_CCI`, is ESA Cloud_cci's CC4CL retrieval and
  gives the all-sky means comparable with its `lwp_allsky` / `iwp_allsky`:

  | | MODIS L3 (`modis_column`, COSP) | Cloud_cci (`CLOUD_CCI`) |
  | --- | --- | --- |
  | Liquid conversion | `2/3 · rho_w · tau · r_e` | the same (`Q_e` = 2.0, `rho` = 1000 kg m⁻³) |
  | Ice conversion | `2/3 · 0.93 rho_w · tau · r_e` | `4/3 · tau · r_e · 916.7 / 2.1` (6 % lower) |
  | Optical-depth limit | none | `tau` ≤ 256 |
  | Cloud-top phase | liquid ≥ 0.7, ice ≤ 0.3 of the extinction in the top optical depth; between them "undetermined", in neither path | liquid above 0.5, otherwise ice: every detected subcolumn has a phase |
  | Reported mean | in-cloud, over that phase's subcolumns | all-sky, over every subcolumn |
  | Daylight | `sunlit` mask in the driver | solar zenith below 80° (`CLOUD_CCI_MIN_COS_ZENITH`), applied by the host |

  See [docs/usage.md](docs/usage.md) for the call and
  [docs/translation_notes.md](docs/translation_notes.md) for what is and is
  not modelled.

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
  faithfully with `jax.random`. Hosts using a McICA generator (e.g. JAX-GCM's)
  can instead inject their own cloud masks so radiation and the satellite
  simulator see the same subgrid cloud field.
- **Reference values, usable gradients.** COSP's diagnostics are counts of
  subcolumns passing hard tests, whose derivatives are exactly zero. Every
  output here keeps the reference value bit for bit and takes its derivative
  from a smooth surrogate instead: a logistic step for each threshold and bin
  edge, and a likelihood-ratio weight per subcolumn for the SCOPS sampling.
  Cloud fractions, histograms and occurrence frequencies can therefore be
  calibration targets of a gradient-based optimiser. See
  [docs/gradients.md](docs/gradients.md).
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
Colorado, Lawrence Livermore National Security LLC, the Met Office (British
Crown Copyright, ISCCP/ICARUS) and RIAM Kyushu University;
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
- Pincus, R., et al. (2012). *J. Climate* 25, 4699–4720. (MODIS simulator)
- McGarragh, G. R., et al. (2018). *AMT* 11, 3397–3431. (CC4CL retrieval; imager water path)
- Stengel, M., et al. (2020). *ESSD* 12, 41–60. (Cloud_cci v3 products and cloud-top phase)
