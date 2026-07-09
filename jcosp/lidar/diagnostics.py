"""CALIPSO lidar column stage: SR, CFAD, GOCCP cloud and phase fractions.

Ports of COSP v2.0 ``src/simulator/actsim/lidar_simulator.F90`` subroutines
``lidar_column`` (CALIPSO branch) and ``COSP_CLDFRAC`` (Copyright (c) 2009,
Centre National de la Recherche Scientifique, BSD-3-Clause; see LICENSES/;
phase diagnostics by G. Cesana and H. Chepfer, 2013, JGR,
doi:10.1002/jgrd.50376), plus ``src/cosp_stats.F90:COSP_LIDAR_ONLY_CLOUD``
((c) 2008, Lawrence Livermore National Security LLC).

Only COSP's default configuration is ported: ``use_vgrid=.true.`` with the
40 x 480 m statistical grid (``Nlvgrid=40``). The level split of the phase
diagnostics (levels 0..22 above 8.16 km never trigger full attenuation) is
hard-wired to that grid in the Fortran and kept as such.

Deviations (documented in docs/translation_notes.md):

* ``COSP_CLDFRAC`` section 4.1 reads ``pplay(1, nlev)`` — gridpoint 1's
  pressure for every gridpoint — when assigning high/mid/low categories to
  phase-classified clouds above 8.16 km (an apparent indexing slip for
  ``pplay(i, nlev)``; section 4.2 uses the per-point value). We use the
  per-point pressure everywhere.

Array convention: level axis 0 TOA-first, subcolumn axis next, batch axes
trailing; diagnostic category axes (phase, temperature bin) follow the level
axis, before the batch axes.
"""

from typing import NamedTuple

import jax.numpy as jnp

from jcosp import config, stats
from jcosp.constants import R_GROUND, R_UNDEF


def scattering_ratio(pnorm, betamol):
    """Lidar scattering ratio SR = ATB / ATBmol (lidar_column).

    ``pnorm`` is (nlev, ncol, *batch); ``betamol`` (nlev, *batch) is the
    attenuated molecular signal on the same grid. Values are R_UNDEF where
    either signal is not physically meaningful (>= max histogram bin, e.g.
    R_GROUND below the surface) or the molecular signal is not positive.
    """
    xmax = float(config.CALIPSO_HIST_BSCT.max())
    betamol_b = betamol[:, None]
    ok = (pnorm < xmax) & (betamol_b < xmax) & (betamol_b > 0.0)
    return jnp.where(ok, pnorm / jnp.where(ok, betamol_b, 1.0), R_UNDEF)


def cfad_sr(sr):
    """CFAD of the scattering ratio (lidar_column CFAD block).

    ``sr`` is (nlev, ncol, *batch); returns (SR_BINS, nlev, *batch) fractions
    of subcolumns per SR bin (counts / ncol). R_UNDEF samples fall below the
    first bin edge (-1) and are simply not counted, as in the Fortran.
    """
    ncol = sr.shape[1]
    counts = stats.hist1d(jnp.moveaxis(sr, 1, 0), config.CALIPSO_HIST_BSCT)
    return jnp.where(counts != R_UNDEF, counts / ncol, counts)


class CloudFractions(NamedTuple):
    """COSP_CLDFRAC outputs (fractions in [0, 1]; R_UNDEF where undefined)."""

    lidarcld: jnp.ndarray       # (nlev, *batch) 3D cloud fraction
    cldlayer: jnp.ndarray       # (4, *batch) low/mid/high/total cloud cover
    lidarcldphase: jnp.ndarray  # (nlev, 6, *batch) phase cloud fractions
    cldlayerphase: jnp.ndarray  # (4, 6, *batch) layered phase cloud cover
    lidarcldtemp: jnp.ndarray   # (40, 5, *batch) phase fractions vs temperature


def cldfrac(temperature, sr, atb, atb_perp, pressure) -> CloudFractions:
    """GOCCP cloud detection, cloud fractions and phase (COSP_CLDFRAC).

    All inputs live on the 40-level statistical grid, TOA-first:
    ``temperature``/``pressure`` are (nlev, *batch) profiles (COSP feeds the
    layer's *lower-interface* pressure here), ``sr``/``atb``/``atb_perp`` are
    (nlev, ncol, *batch) subcolumn fields. Outputs are occurrence fractions
    (not yet scaled to %, which ``lidar_column`` applies).

    Phase axis order (Fortran 1..6): [ice, liquid, undefined, false liquid
    (reclassified to ice), false ice (reclassified to liquid), ice/(ice+liq)].
    Layer-category axis order: [low, mid, high, total].
    """
    nlev = sr.shape[0]
    dtype = sr.dtype
    lev = jnp.arange(nlev).reshape((nlev,) + (1,) * (sr.ndim - 1))

    # --- 2) Cloud detection and usable-subcolumn masks --------------------
    cldy = (sr > config.CALIPSO_S_CLD) & (sr != R_UNDEF)
    srok = (sr > config.CALIPSO_S_ATT) & (sr != R_UNDEF)

    # --- 3) 3D cloud fraction and layered cloud cover ---------------------
    nsub = jnp.sum(srok, axis=1).astype(dtype)                 # (nlev, *b)
    ncld = jnp.sum(cldy, axis=1).astype(dtype)
    lidarcld = jnp.where(nsub > 0.0, ncld / jnp.maximum(nsub, 1.0), R_UNDEF)

    p = pressure
    high = (p > 0.0) & (p < config.CALIPSO_HIGH_CLOUD_PMAX)
    mid = (p >= config.CALIPSO_HIGH_CLOUD_PMAX) & (p < config.CALIPSO_MID_CLOUD_PMAX)
    low = ~(high | mid)
    cats = jnp.stack([low, mid, high, jnp.ones_like(low)])     # (4, nlev, *b)
    cats_b = cats[:, :, None]                                  # (4, nlev, 1, *b)

    def _layered(mask):
        # Fortran cldlay/nsublay: per (column, category) OR over levels,
        # then summed over columns.
        return jnp.sum(jnp.any(mask[None] & cats_b, axis=1),
                       axis=1).astype(dtype)                   # (4, *b)

    cldlayer_cnt = _layered(cldy)
    nsublayer = _layered(srok)
    cldlayer = jnp.where(nsublayer > 0.0,
                         cldlayer_cnt / jnp.maximum(nsublayer, 1.0), R_UNDEF)

    # --- 4) Phase classification (Cesana and Chepfer 2013) ----------------
    # Discrimination line: ATBperp above the polynomial in ATB means ice.
    c = config.CALIPSO_PHASE_DISCR_COEFFS
    atbp_line = (atb**5 * c[0] + atb**4 * c[1] + atb**3 * c[2]
                 + atb**2 * c[3] + atb * c[4] + c[5])
    temp_b = temperature[:, None]                              # (nlev, 1, *b)
    consider = cldy & (atb_perp > 0.0)
    ice_line = (atb_perp - atbp_line) >= 0.0

    # Full attenuation: below the highest volume with SR > S_cld_att in the
    # lower zone (levels >= 23, i.e. under 8.16 km) the phase is undefined.
    # The Fortran scans downward and exits at the first such volume, so the
    # saturating volume itself is still phase-classified.
    sat = consider & (sr > config.CALIPSO_S_CLD_ATT) & (lev >= config.CALIPSO_NLEV_HIGHALT)
    has_sat = jnp.any(sat, axis=0)                             # (ncol, *b)
    ksat = jnp.argmax(sat, axis=0)                             # first True
    in_undef = has_sat[None] & (lev > ksat[None])              # (nlev, ncol, *b)
    classify = consider & ~in_undef

    warm = temp_b > config.CALIPSO_T_FALSE_ICE       # "ice" here is really liquid
    cold = ~(temp_b > config.CALIPSO_T_FALSE_LIQ)    # "liquid" here is really ice
    false_ice = classify & ice_line & warm           # counted as liquid (+ slot 4)
    true_ice = classify & ice_line & ~warm
    false_liq = classify & ~ice_line & cold          # counted as ice (+ slot 3)
    true_liq = classify & ~ice_line & ~cold
    ice_all = true_ice | false_liq
    liq_all = true_liq | false_ice
    undef = cldy & in_undef
    phase_masks = (ice_all, liq_all, undef, false_liq, false_ice)

    # 3D phase fractions: counts over columns / nsub; slot 5 (index 5) is the
    # ice occurrence ratio ice/(ice+liq), formed from the raw counts.
    cnts = [jnp.sum(m, axis=1).astype(dtype) for m in phase_masks]  # (nlev, *b)
    icnt, lcnt = cnts[0], cnts[1]
    ratio = jnp.where(icnt + lcnt > 0.0,
                      icnt / jnp.maximum(icnt + lcnt, 1.0), R_UNDEF)
    fracs = [jnp.where(nsub > 0.0, cn / jnp.maximum(nsub, 1.0), R_UNDEF)
             for cn in cnts]
    lidarcldphase = jnp.stack(fracs + [ratio], axis=1)         # (nlev, 6, *b)

    # Layered phase cover: per-column presence flags per category and phase,
    # summed over columns; phases 1..3 (ice/liq/undef) define the norm.
    flags = jnp.stack([jnp.any(m[None] & cats_b, axis=1) for m in phase_masks])
    lay_cnt = jnp.sum(flags, axis=2).astype(dtype)             # (5, 4, *b)
    phasesum = jnp.sum(lay_cnt[:3], axis=0)                    # (4, *b)
    lay_ratio = jnp.where(lay_cnt[0] + lay_cnt[1] > 0.0,
                          lay_cnt[0] / jnp.maximum(lay_cnt[0] + lay_cnt[1], 1.0),
                          R_UNDEF)
    # Rescale the phase partition to the layered cloud cover. Where phasesum
    # is zero all five counts are zero (kept, as in the Fortran).
    scaled = jnp.where(phasesum[None] > 0.0,
                       lay_cnt / jnp.maximum(phasesum[None], 1.0)
                       * jnp.where(cldlayer == R_UNDEF, 0.0, cldlayer)[None],
                       lay_cnt)
    scaled = jnp.where(nsublayer[None] == 0.0, R_UNDEF, scaled)
    cldlayerphase = jnp.concatenate([scaled, lay_ratio[None]], axis=0)
    cldlayerphase = jnp.moveaxis(cldlayerphase, 0, 1)          # (4, 6, *b)

    # --- Phase fractions as a function of temperature ---------------------
    edges = jnp.asarray(config.LIDAR_PHASE_TEMP_EDGES)
    tlo = edges[:-1].reshape((-1,) + (1,) * sr.ndim)
    thi = edges[1:].reshape((-1,) + (1,) * sr.ndim)

    def _tcounts(mask):
        # Occurrences per temperature bin, summed over levels and columns.
        # The Fortran bins the cell temperature stored in tmpi/tmpl/tmpu
        # (profile temperature where the mask is set, requiring T > 0, which
        # bin edge 0 already enforces for masked cells).
        in_bin = (temp_b[None] >= tlo) & (temp_b[None] < thi)
        return jnp.sum(in_bin & mask[None], axis=(1, 2)).astype(dtype)

    t_ind = _tcounts(srok)                                     # (40, *b)
    t_cnts = [_tcounts(m) for m in (cldy, ice_all, liq_all, undef)]
    t_ratio = jnp.where(t_cnts[1] + t_cnts[2] > 0.0,
                        t_cnts[1] / jnp.maximum(t_cnts[1] + t_cnts[2], 1.0),
                        R_UNDEF)
    t_fracs = [jnp.where(t_ind > 0.0, cn / jnp.maximum(t_ind, 1.0), R_UNDEF)
               for cn in t_cnts]
    lidarcldtemp = jnp.stack(t_fracs + [t_ratio], axis=1)      # (40, 5, *b)

    return CloudFractions(lidarcld=lidarcld, cldlayer=cldlayer,
                          lidarcldphase=lidarcldphase,
                          cldlayerphase=cldlayerphase,
                          lidarcldtemp=lidarcldtemp)


class CalipsoColumn(NamedTuple):
    """CALIPSO column-stage diagnostics (percent where applicable)."""

    cfad_sr: jnp.ndarray        # (SR_BINS, nvgrid, *batch) SR histogram fractions
    lidarcld: jnp.ndarray       # (nvgrid, *batch) 3D cloud fraction (%)
    cldlayer: jnp.ndarray       # (4, *batch) low/mid/high/total cover (%)
    lidarcldphase: jnp.ndarray  # (nvgrid, 6, *batch) phase fractions (%)
    cldlayerphase: jnp.ndarray  # (4, 6, *batch) layered phase cover (%)
    lidarcldtemp: jnp.ndarray   # (40, 5, *batch) phase vs temperature (%)
    sr: jnp.ndarray             # (nvgrid, ncol, *batch) scattering ratio


def _regrid_profile(zfull, zhalf, y):
    """Regrid a (nlev, *batch) profile to the statistical grid."""
    return stats.change_vertical_grid(
        zfull, zhalf, y[:, None], config.VGRID_ZL, config.VGRID_ZU)[:, 0]


def lidar_column(pnorm, pmol, pnorm_perp, pressure_lower, temperature,
                 zfull, zhalf) -> CalipsoColumn:
    """CALIPSO column diagnostics on the 40x480 m grid (lidar_column).

    Parameters
    ----------
    pnorm : (nlev, ncol, *batch)
        Total attenuated backscatter on model levels, TOA-first.
    pmol : (nlev, *batch)
        Molecular attenuated backscatter.
    pnorm_perp : (nlev, ncol, *batch)
        Perpendicular-channel attenuated backscatter.
    pressure_lower : (nlev, *batch)
        Pressure at the *lower interface* of each model layer (Pa) — COSP
        passes ``phalf(2:nlev+1)`` as the category pressure.
    temperature : (nlev, *batch)
        Temperature on model levels (K).
    zfull, zhalf : (nlev, *batch)
        Layer mid-point and bottom-interface heights (m).

    All fields are regridded (linearly, thickness-weighted) onto the 480-m
    statistical grid *before* the scattering ratio is formed, exactly as the
    Fortran does with ``use_vgrid=.true.``.
    """
    pplay_r = _regrid_profile(zfull, zhalf, pressure_lower)
    betamol_r = _regrid_profile(zfull, zhalf, pmol)
    tmp_r = _regrid_profile(zfull, zhalf, temperature)
    pnorm_r = stats.change_vertical_grid(zfull, zhalf, pnorm,
                                         config.VGRID_ZL, config.VGRID_ZU)
    perp_r = stats.change_vertical_grid(zfull, zhalf, pnorm_perp,
                                        config.VGRID_ZL, config.VGRID_ZU)

    sr = scattering_ratio(pnorm_r, betamol_r)
    cfad = cfad_sr(sr)
    fr = cldfrac(tmp_r, sr, pnorm_r, perp_r, pplay_r)

    def _pct(a):
        return jnp.where(a != R_UNDEF, a * 100.0, a)

    return CalipsoColumn(cfad_sr=cfad, lidarcld=_pct(fr.lidarcld),
                         cldlayer=_pct(fr.cldlayer),
                         lidarcldphase=_pct(fr.lidarcldphase),
                         cldlayerphase=_pct(fr.cldlayerphase),
                         lidarcldtemp=_pct(fr.lidarcldtemp), sr=sr)


class LidarOnlyCloud(NamedTuple):
    """Joint CALIPSO/CloudSat products (COSP_LIDAR_ONLY_CLOUD), in percent."""

    lidar_only_freq_cloud: jnp.ndarray  # (nlev, *batch) lidar-only cloud freq.
    radar_lidar_tcc: jnp.ndarray        # (*batch) combined total cloud cover
    cloudsat_tcc: jnp.ndarray           # (*batch) radar total cloud cover
    cloudsat_tcc2: jnp.ndarray          # (*batch) as tcc, excluding lowest ~1 km


def lidar_only_cloud(beta_tot, beta_mol, dbze) -> LidarOnlyCloud:
    """Cloud seen by the lidar but missed by the radar (COSP_LIDAR_ONLY_CLOUD).

    ``beta_tot``/``dbze`` are (nlev, ncol, *batch) and ``beta_mol`` is
    (nlev, *batch); in COSP's default configuration all three are on the
    40-level statistical grid, TOA-first (dbze regridded with
    ``log_units=True``). A volume counts as lidar-only cloud when the radar
    sees nothing (dbze < -30) but the lidar detects cloud (SR > 5) or is at
    its first fully-attenuated level (SR <= 0.01).

    The Fortran's near-surface exclusion index for ``cloudsat_tcc2`` is found
    by skipping R_GROUND gates from index 0; on TOA-first input (as COSP
    passes in the vgrid path) index 0 is never R_GROUND, so the exclusion
    covers levels 2..end. Reproduced as-is.
    """
    dtype = beta_tot.dtype
    nlev = beta_tot.shape[0]
    lev = jnp.arange(nlev).reshape((nlev,) + (1,) * (beta_tot.ndim - 1))

    # The Fortran divides unconditionally (below ground R_GROUND/R_GROUND = 1,
    # which triggers no threshold). A zero molecular signal would be NaN in
    # the Fortran (no flags trigger); we use the equally inert value 1.
    denom_ok = beta_mol[:, None] != 0.0
    sc_ratio = jnp.where(denom_ok,
                         beta_tot / jnp.where(denom_ok, beta_mol[:, None], 1.0),
                         1.0)
    attenuated = sc_ratio <= 0.01
    first_att = jnp.argmax(attenuated, axis=0)                 # (ncol, *b)
    is_flag_sat = jnp.any(attenuated, axis=0)[None] & (lev == first_att[None])

    radar_sees = dbze >= -30.0
    lidar_only = ~radar_sees & ((sc_ratio > 5.0) | is_flag_sat)

    ncol = beta_tot.shape[1]
    freq = jnp.sum(lidar_only, axis=1).astype(dtype) / ncol

    any_cloud = jnp.any(lidar_only | radar_sees, axis=0)       # (ncol, *b)
    tcc = jnp.sum(any_cloud, axis=0).astype(dtype) / ncol
    radar_tcc = jnp.sum(jnp.any(radar_sees, axis=0), axis=0).astype(dtype) / ncol

    # j_1km: one past the first non-R_GROUND gate (Fortran do-while from j=1).
    j_first = jnp.argmax(dbze != R_GROUND, axis=0)             # (ncol, *b)
    no1km = radar_sees & (lev > j_first[None] + 1)
    radar_tcc2 = jnp.sum(jnp.any(no1km, axis=0), axis=0).astype(dtype) / ncol

    return LidarOnlyCloud(lidar_only_freq_cloud=freq * 100.0,
                          radar_lidar_tcc=tcc * 100.0,
                          cloudsat_tcc=radar_tcc * 100.0,
                          cloudsat_tcc2=radar_tcc2 * 100.0)
