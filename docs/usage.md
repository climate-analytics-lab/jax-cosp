# Using jax-cosp

## Standalone: CloudSat radar path on model profiles

```python
import importlib.resources
import jax
import numpy as np

from jcosp.radar.optics import load_lut
from jcosp.simulator import CloudsatInputs, simulate_cloudsat

lut = load_lut(importlib.resources.files("jcosp") / "data"
               / "quickbeam_lut_94GHz_1M.npz")

# All profile arrays are (nlev, *batch), TOA-first (level 0 = model top);
# see CloudsatInputs for the full field list and units.
state = CloudsatInputs(
    pressure=p, temperature=t, specific_humidity=q,       # Pa, K, kg/kg
    zfull=zfull, zhalf=zhalf,                             # m (mid / bottom)
    cloud_frac=cf, conv_frac=cvf,                         # fractions
    mr_lsliq=qc, mr_lsice=qi,                             # kg/kg, gridbox mean
    mr_ccliq=0 * qc, mr_ccice=0 * qc,
    fl_lsrain=rain_flux, fl_lssnow=snow_flux,             # kg m-2 s-1 profiles
    fl_lsgrpl=0 * rain_flux, fl_ccrain=cv_rain, fl_ccsnow=cv_snow,
    land=land, surfelev=orog, t2m=t2m,                    # (*batch,)
    reff=None,                                            # None -> PSD defaults
)

out = simulate_cloudsat(state, lut, key=jax.random.PRNGKey(0), ncolumns=100)
out.dbze          # (nlev, ncolumns, *batch) attenuated reflectivity, dBZe
out.cfad_ze       # CloudSat CFAD on the 40x480 m grid
out.precip_cover  # 2C-PRECIP-COLUMN class cover fractions
out.warm_rain     # Mülmenstädt-2020 warm/cold rain & drizzle fractions
```

`simulate_cloudsat` accepts a precomputed `frac_out` subcolumn cloud map in
place of `key`, so hosts with a McICA generator can make radiation and the
satellite simulator sample the same subgrid cloud field.

### Warm-rain fraction climatology

Average the flag fields over time *before* taking the ratio:

```python
from jcosp.classify import WarmRainFlags, warm_rain_fraction

f_warm = warm_rain_fraction(WarmRainFlags(
    warm_rain=mean_warm, cold_rain=mean_cold,
    warm_drizzle=mean_wdriz, cold_drizzle=mean_cdriz))
```

This is the quantity comparable to the Mülmenstädt et al. (2015)
CloudSat/CALIPSO climatology and used as the aerosol–cloud-interaction
constraint in Mülmenstädt et al. (2020).

## MODIS retrievals and CFODD / warm-rain occurrence

The MODIS path needs two extra host fields beyond `CloudsatInputs`: the
gridbox-mean 0.67-µm layer optical depths of stratiform and convective
cloud (`dtau_s`, `dtau_c`, `(nlev, *batch)`), and the layer-edge pressures
(`phalf`, `(nlev+1, *batch)`, TOA-first). The combined driver runs the
radar, the MODIS retrievals and the Michibata et al. (2019) joint
diagnostics on one SCOPS realization:

```python
from jcosp.simulator import simulate_cloudsat_modis

out = simulate_cloudsat_modis(
    state, lut, dtau_s, dtau_c, phalf,
    key=jax.random.PRNGKey(0), ncolumns=100,
    sunlit=sunlit,       # (*batch) 1 day / 0 night; dark points -> R_UNDEF
)

col = out.modis.column
col.cf_total, col.cf_liquid, col.cf_ice   # cloud fractions (%)
col.tau_liquid, col.size_liquid, col.lwp  # in-cloud means (SI; sizes in m)
col.tau_vs_ctp                            # (7, 7, *batch) joint histogram (%)
col.lwp_vs_reff_liq, col.iwp_vs_reff_ice  # water-path/size histograms (%)

out.warmrain.wr_occfreq_ntotal  # (3, *batch) non-precip/drizzle/rain counts
out.warmrain.cfodd_ntotal       # (25, 30, 3, *batch) CFODD samples
                                # (dBZe x in-cloud optical depth x Reff class)
```

MODIS can also run standalone on any subcolumn realization
(`jcosp.modis.simulate_modis(frac_out, mr_hydro, reff, dtau_s, dtau_c,
phalf, ...)`, with `mr_hydro`/`reff` from
`jcosp.inputs.distribute_hydrometeors`). Pass `isccp_ctp` when an ISCCP
simulator is available: COSP replaces the CO2-slicing cloud-top pressure of
clouds below 700 hPa with the ISCCP value; without it the CO2-slicing CTP
is kept (see docs/translation_notes.md). Cloud effective radii
(`state.reff`) should be provided for meaningful MODIS optics — the size
retrieval reads the 3.7-µm optical properties computed from them.

## From JAX-GCM

The `jcm` hook is the `CloudsatCosp` physics term (jax-gcm repository,
`jcm/physics/diagnostics/cosp_cloudsat.py`):

```python
physics = echam_physics(enable_cosp=True, cosp_ncolumns=40)
```

which emits `cosp_warm_rain`, `cosp_cold_rain`, `cosp_warm_drizzle`,
`cosp_cold_drizzle`, `cosp_pia` and `cosp_precip_cover.*` in the model
output, time-averaged when `output_averages=True`. Install the dependency
with `pip install jcm[cosp]`. The term's module docstring documents the
input-mapping approximations (no separate convective cloud fraction in jcm;
convective precip profile is a labelled stopgap).

## Differentiability

Everything downstream of the subcolumn sampling is jit- and grad-compatible;
reflectivity is differentiable w.r.t. water contents, fluxes and (piecewise)
effective radii. The subcolumn sampling itself is stochastic forward
(translation-plan option A): treat SCOPS draws as fixed (pass `frac_out`)
when differentiating, or use ensemble/gradient-free calibration across keys.
The hard dBZe thresholds in the CloudSat/warm-rain classifications are step
functions; a smooth-threshold variant is a planned research spike (plan
§5.1-5.2). The MODIS retrievals are differentiable w.r.t. the optical
inputs (tau, CTP, reflectances; particle size piecewise, through the
trial-radius bracketing), while the cloud/phase masks and the CFODD /
occurrence *counts* are step functions — treat those as forward
diagnostics.

## Regenerating the radar LUT

```bash
python -m jcosp.radar.lut.generate   # ~90 s, writes jcosp/data/*.npz
```

Only needed when changing frequency, hydrometeor tables, or the Mie/DSD
code; the repository ships the 94-GHz single-moment table.
