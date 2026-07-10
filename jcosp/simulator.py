"""End-to-end CloudSat simulator: model state in, radar diagnostics out.

The lean JAX driver replacing COSP's ``cosp.F90`` orchestration for the
CloudSat/warm-rain path (translation-plan Phase 0-3 scope): SCOPS ->
PREC_SCOPS -> hydrometeor distribution -> gaseous attenuation -> Quickbeam
optics (Z-scale LUT) -> attenuation -> CloudSat statistical-grid diagnostics
and the Mülmenstädt et al. (2020) warm-rain classification.

Everything after the (optional) stochastic subcolumn generation is pure and
differentiable; hosts with their own McICA generator can pass ``frac_out``
directly so radiation and the satellite simulator share one subgrid cloud
field.

Array convention: level axis 0 TOA-first, batch axes trailing; see README.
"""

from typing import NamedTuple

import jax.numpy as jnp

from jcosp import classify, config, inputs, modis
from jcosp.radar import optics, quickbeam
from jcosp.radar.gases import gases
from jcosp.stats import change_vertical_grid
from jcosp.subcol import prec_scops, scops


class CloudsatInputs(NamedTuple):
    """Gridbox-mean host-model state feeding the CloudSat simulator.

    All profile fields are (nlev, *batch), TOA-first; surface fields are
    (*batch). Units follow COSP's ``cosp_column_inputs``.
    """

    pressure: jnp.ndarray            # Pa
    temperature: jnp.ndarray         # K
    specific_humidity: jnp.ndarray   # kg/kg
    zfull: jnp.ndarray               # layer mid-point height (m)
    zhalf: jnp.ndarray               # layer bottom-interface height (m)
    cloud_frac: jnp.ndarray          # total cloud fraction (incl. convective)
    conv_frac: jnp.ndarray           # convective cloud fraction
    mr_lsliq: jnp.ndarray            # large-scale cloud liquid (kg/kg)
    mr_lsice: jnp.ndarray            # large-scale cloud ice (kg/kg)
    mr_ccliq: jnp.ndarray            # convective cloud liquid (kg/kg)
    mr_ccice: jnp.ndarray            # convective cloud ice (kg/kg)
    fl_lsrain: jnp.ndarray           # large-scale rain flux (kg m-2 s-1)
    fl_lssnow: jnp.ndarray           # large-scale snow flux
    fl_lsgrpl: jnp.ndarray           # large-scale graupel flux
    fl_ccrain: jnp.ndarray           # convective rain flux
    fl_ccsnow: jnp.ndarray           # convective snow flux
    land: jnp.ndarray                # land mask (0 ocean / 1 land)
    surfelev: jnp.ndarray            # surface elevation (m)
    t2m: jnp.ndarray                 # near-surface temperature (K)
    reff: jnp.ndarray | None = None  # (nlev, *batch, N_HYDRO) eff. radii (m);
    #                                  None/0 selects the PSD defaults


class CloudsatOutputs(NamedTuple):
    """CloudSat simulator diagnostics."""

    dbze: jnp.ndarray               # (nlev, ncol, *batch) attenuated dBZe
    ze_non: jnp.ndarray             # (nlev, ncol, *batch) unattenuated dBZe
    cfad_ze: jnp.ndarray            # (DBZE_BINS, nglev, *batch)
    precip_cover: jnp.ndarray       # (N_CLOUDSAT_PRECIP_CLASS, *batch)
    pia: jnp.ndarray                # (*batch)
    warm_rain: classify.WarmRainFlags  # (*batch) occurrence fractions
    frac_out: jnp.ndarray           # (nlev, ncol, *batch) subcolumn cloud map
    prec_frac: jnp.ndarray          # (nlev, ncol, *batch) subcolumn precip map


def simulate_cloudsat(state: CloudsatInputs, lut: optics.ZScaleLUT,
                      key=None, ncolumns: int = 100, overlap: int = 3,
                      frac_out=None,
                      radar: config.RadarConfig = config.CLOUDSAT_RADAR,
                      ) -> CloudsatOutputs:
    """Run the CloudSat radar path on gridbox-mean model state.

    Either ``key`` (to sample subcolumns with SCOPS) or a precomputed
    ``frac_out`` subcolumn cloud map must be given.
    """
    if frac_out is None:
        if key is None:
            raise ValueError("provide either key or frac_out")
        frac_out = scops(key, state.cloud_frac, state.conv_frac,
                         ncolumns, overlap)

    ls_p_rate = state.fl_lsrain + state.fl_lssnow + state.fl_lsgrpl
    cv_p_rate = state.fl_ccrain + state.fl_ccsnow
    prec_frac = prec_scops(frac_out, ls_p_rate, cv_p_rate)

    hyd = inputs.distribute_hydrometeors(
        frac_out, prec_frac, state.pressure, state.temperature,
        state.mr_lsliq, state.mr_lsice, state.mr_ccliq, state.mr_ccice,
        state.fl_lsrain, state.fl_lssnow, state.fl_lsgrpl,
        state.fl_ccrain, state.fl_ccsnow, reff_in=state.reff)

    # Gaseous attenuation is per profile (identical across subcolumns).
    g_vol = gases(state.pressure, state.temperature,
                  state.specific_humidity, radar.freq)

    sd = config.HYDRO_CLASSES[radar.micro_scheme]
    # Optics take g/kg and microns (the driver's unit conversions).
    z_vol, _, kr_vol = optics.quickbeam_optics(
        lut, sd, hyd.mr_hydro * 1000.0, hyd.reff * 1e6, None,
        state.pressure[:, None], state.temperature[:, None])

    dbze, ze_non = quickbeam.quickbeam_subcolumn(
        z_vol, kr_vol, g_vol[:, None], (state.zfull / 1000.0)[:, None],
        radar_at_layer_one=radar.radar_at_layer_one)

    # Frozen fraction of precipitation at the flag gate: regrid to the
    # CloudSat statistical grid, then pick the near-surface gate (the driver
    # uses the ocean gate index for this, without the land shift).
    fpi_vgrid = change_vertical_grid(
        state.zfull, state.zhalf, hyd.frac_precip_ice,
        config.VGRID_ZL, config.VGRID_ZU)
    lvl = jnp.clip(
        config.CLOUDSAT_PRECLVL
        - jnp.floor(state.surfelev / config.VGRID_ZSTEP).astype(jnp.int32),
        0, fpi_vgrid.shape[0] - 1)
    fpi_gate = jnp.take_along_axis(fpi_vgrid, lvl[None, None], axis=0)[0]

    column = quickbeam.quickbeam_column(
        dbze, ze_non, state.zfull, state.zhalf, state.land, state.surfelev,
        state.t2m, fpi_gate)

    # Warm-rain classification (Mülmenstädt et al. 2020): liquid surface
    # precipitation = rain water reaching the lowest model level.
    rain_sfc = (hyd.mr_hydro[-1, ..., config.I_LSRAIN]
                + hyd.mr_hydro[-1, ..., config.I_CVRAIN])
    cloud_liq = (hyd.mr_hydro[..., config.I_LSCLIQ]
                 + hyd.mr_hydro[..., config.I_CVCLIQ])
    cloud_ice = (hyd.mr_hydro[..., config.I_LSCICE]
                 + hyd.mr_hydro[..., config.I_CVCICE])
    warm_rain = classify.warm_rain_classification(
        dbze, cloud_liq, cloud_ice, rain_sfc > 0.0)

    return CloudsatOutputs(
        dbze=dbze, ze_non=ze_non, cfad_ze=column.cfad_ze,
        precip_cover=column.precip_cover, pia=column.pia,
        warm_rain=warm_rain, frac_out=frac_out, prec_frac=prec_frac)


class CloudsatModisOutputs(NamedTuple):
    """Joint CloudSat + MODIS diagnostics (Phase-5 scope)."""

    cloudsat: CloudsatOutputs
    modis: modis.ModisOutputs
    warmrain: modis.WarmRainDiagnostics  # Michibata et al. (2019) CFODD/regimes


def simulate_cloudsat_modis(state: CloudsatInputs, lut: optics.ZScaleLUT,
                            dtau_s, dtau_c, pressure_half,
                            key=None, ncolumns: int = 100, overlap: int = 3,
                            frac_out=None, sunlit=None, isccp_ctp=None,
                            radar: config.RadarConfig = config.CLOUDSAT_RADAR,
                            ) -> CloudsatModisOutputs:
    """Run CloudSat + MODIS + CFODD/warm-rain on one SCOPS realization.

    ``simulate_cloudsat`` is unchanged; this composes it with the MODIS
    simulator (which needs the extra host fields ``dtau_s``/``dtau_c`` —
    gridbox-mean 0.67-um layer optical depths of stratiform / convective
    cloud — and ``pressure_half``, the (nlev+1, *batch) layer-edge
    pressures) and with ``jcosp.modis.cfodd.diag_warmrain`` driven exactly
    as cosp.F90's default ``use_vgrid`` path: temperature and the SCOPS map
    regridded linearly, dBZe log-averaged, onto the 40 x 480 m statistical
    grid whose upper-interface heights serve as ``zlev``.

    ``sunlit``/``isccp_ctp`` are passed through to the MODIS column stage
    (see ``jcosp.modis.simulate_modis``).
    """
    cs = simulate_cloudsat(state, lut, key=key, ncolumns=ncolumns,
                           overlap=overlap, frac_out=frac_out, radar=radar)

    # The MODIS input build consumes the same subcolumn hydrometeors the
    # radar optics saw (recomputed here; distribute_hydrometeors is cheap
    # and deterministic given the SCOPS/PREC_SCOPS maps).
    hyd = inputs.distribute_hydrometeors(
        cs.frac_out, cs.prec_frac, state.pressure, state.temperature,
        state.mr_lsliq, state.mr_lsice, state.mr_ccliq, state.mr_ccice,
        state.fl_lsrain, state.fl_lssnow, state.fl_lsgrpl,
        state.fl_ccrain, state.fl_ccsnow, reff_in=state.reff)

    mod = modis.simulate_modis(cs.frac_out, hyd.mr_hydro, hyd.reff,
                               dtau_s, dtau_c, pressure_half,
                               sunlit=sunlit, isccp_ctp=isccp_ctp)

    # CFODD / warm-rain inputs on the CloudSat statistical grid (cosp.F90,
    # Lcloudsat_modis_wr with use_vgrid=.true.).
    temp_r = change_vertical_grid(
        state.zfull, state.zhalf, state.temperature[:, None],
        config.VGRID_ZL, config.VGRID_ZU)[:, 0]
    dbze_r = change_vertical_grid(state.zfull, state.zhalf, cs.dbze,
                                  config.VGRID_ZL, config.VGRID_ZU,
                                  log_units=True)
    frac_r = change_vertical_grid(
        state.zfull, state.zhalf,
        cs.frac_out.astype(state.temperature.dtype),
        config.VGRID_ZL, config.VGRID_ZU)

    col = mod.column
    wr = modis.diag_warmrain(
        temp_r, jnp.asarray(config.VGRID_ZU, state.temperature.dtype),
        lwp=col.lwp, liqcot=col.tau_liquid, liqreff=col.size_liquid,
        iwp=col.iwp, icecot=col.tau_ice, icereff=col.size_ice,
        frac_out=frac_r, dbze=dbze_r)

    return CloudsatModisOutputs(cloudsat=cs, modis=mod, warmrain=wr)
