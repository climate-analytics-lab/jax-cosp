"""CFODD and warm-rain occurrence diagnostics (CloudSat + MODIS joint product).

Port of COSP v2.0 ``src/cosp_stats.F90:COSP_DIAG_WARMRAIN`` ((c) 2018,
Research Institute for Applied Mechanics (RIAM), Kyushu University;
T. Michibata; BSD-3-Clause, see LICENSES/). References: Michibata et al.
(2019, GMD, doi:10.5194/gmd-12-4297-2019), Michibata et al. (2020, GRL),
Suzuki et al. (2010, 2015, JAS), Jing et al. (2019, J. Climate).

For gridpoints whose MODIS retrievals identify a single-layer warm liquid
cloud scene (LWP > 0, liquid COT > 0.3, 5 um <= Reff_liq <= 35 um, no ice
contamination), each subcolumn is scanned from the surface upward for a
contiguous supercooled-free liquid water cloud (SCOPS stratiform cells with
dBZe >= -30 between the detected cloud bottom and top, cloud-top warmer than
freezing). Qualifying subcolumns are classified by column-maximum dBZe into
non-precipitating / drizzling / raining regimes, and their in-cloud optical
depth profile (adiabatic-like ICOD(z) = COT (1 - ((z - z_btm)/H)^(5/3))) is
binned against dBZe into the CFODD joint histogram, stratified into three
Reff classes.

Index mapping: the Fortran scans ``k = Nlevels, 1, -1`` over TOA-first data,
i.e. from the surface-most level upward; on our axis-0 TOA-first arrays that
is index nlev-1 down to 0, so the cloud *top* has the smaller level index
(kctop <= kcbtm). Verified against a loop transliteration in cfodd_test.py.

FIXED UPSTREAM BUG (docs/translation_notes.md): the Fortran accumulates the
CFODD histogram over *all* (subcolumn, level) cells of a screened-in
gridpoint, so a cell with dBZe >= -30 in a subcolumn that failed the SLWC
checks lands in the first ICOD bin (its ICOD stays at the 0
initialization). Here only in-cloud cells of SLWC-passing subcolumns are
counted.

Faithfully kept Fortran quirks:

* ``liqcfrc``/``icecfrc`` (MODIS cloud fractions) are inputs in the Fortran
  signature but never used; they are omitted here.
"""

from typing import NamedTuple

import jax.numpy as jnp

from jcosp import config, stats, surrogate
from jcosp.constants import R_GROUND, R_UNDEF, TMELT


class WarmRainDiagnostics(NamedTuple):
    """COSP_DIAG_WARMRAIN outputs (sample counts; R_UNDEF for non-sunlit)."""

    cfodd_ntotal: jnp.ndarray       # (NDBZE, NICOD, NCLASS, *batch)
    wr_occfreq_ntotal: jnp.ndarray  # (WR_NREGIME, *batch)


def _level_shape(x, ndim):
    """Reshape a (nlev,) or (nlev, *batch) field to broadcast over subcolumns."""
    if x.ndim == 1:
        return x.reshape((x.shape[0],) + (1,) * (ndim - 1))
    return x[:, None]


def diag_warmrain(temperature, zlev, lwp, liqcot, liqreff,
                  iwp, icecot, icereff, frac_out, dbze, weights=None,
                  smoothing=surrogate.DEFAULT_SMOOTHING) -> WarmRainDiagnostics:
    """CFODD sample histograms and warm-rain occurrence counts.

    Parameters
    ----------
    temperature : (nlev, *batch)
        Gridbox-mean temperature (K) on the same vertical grid as ``dbze``.
    zlev : (nlev,) or (nlev, *batch)
        Level altitude (m), TOA-first. In COSP's default ``use_vgrid`` path
        this is the statistical grid's *upper* interface heights (vgrid_zu).
    lwp, liqcot, liqreff : (*batch)
        MODIS liquid water path (kg m-2), liquid optical thickness and mean
        liquid Reff (m) — e.g. ``ModisColumn.lwp / tau_liquid / size_liquid``.
        ``lwp == R_UNDEF`` (non-sunlit point, or no liquid cloud retrieved)
        makes the outputs R_UNDEF, exactly as in the Fortran.
    iwp, icecot, icereff : (*batch)
        The corresponding MODIS ice retrievals, used to exclude
        ice-contaminated scenes.
    frac_out : (nlev, ncol, *batch)
        SCOPS subcolumn cloud-type map (float or int; regridded values are
        re-quantized with the Fortran's 0.5/1.5 thresholds).
    dbze : (nlev, ncol, *batch)
        Attenuated radar reflectivity (dBZe) on the same grid.
    weights : (ncol, *batch), optional
        SCOPS subcolumn weight (``jcosp.subcol.scops.subcolumn_weights``).
    smoothing : jcosp.surrogate.Smoothing
        Widths of the derivatives across the -30 dBZe detection limit, the
        regime thresholds, the freezing test at cloud top and the
        histogram bin edges. No value depends on it. The cloud top and
        bottom levels, the Reff class and the gridpoint screening are
        discrete and have no derivative.
    """
    dtype = dbze.dtype
    nlev = dbze.shape[0]
    lev = jnp.arange(nlev).reshape((nlev,) + (1,) * (dbze.ndim - 1))

    # fracout can arrive linearly regridded; re-quantize as the Fortran does.
    f = jnp.asarray(frac_out, dtype)
    fracout_int = jnp.where((f >= 0.5) & (f < 1.5), config.SGCLD_ST,
                            jnp.where((f >= 1.5) & (f < 2.5),
                                      config.SGCLD_CUM, config.SGCLD_CLR))

    # ------------------------------------------------------------------
    # Gridpoint screening from the MODIS retrievals.
    # R_UNDEF retrievals fail the same comparisons as in the Fortran
    # (e.g. liqcot = R_UNDEF <= COT_THRESHOLD excludes the point).
    # ------------------------------------------------------------------
    excluded = ((lwp <= config.CWP_THRESHOLD) & (lwp != R_UNDEF)) \
        | (liqcot <= config.COT_THRESHOLD) \
        | (liqreff < config.CFODD_BNDRE[0]) \
        | (liqreff > config.CFODD_BNDRE[3]) \
        | (iwp > config.CWP_THRESHOLD) \
        | (icecot > config.COT_THRESHOLD) \
        | (icereff > config.CFODD_BNDRE[0])
    point_ok = ~excluded                                        # (*batch)

    # Reff class 1..3 (screening guarantees one holds when point_ok).
    icls = jnp.where(
        (liqreff >= config.CFODD_BNDRE[0]) & (liqreff < config.CFODD_BNDRE[1]), 1,
        jnp.where(
            (liqreff >= config.CFODD_BNDRE[1]) & (liqreff < config.CFODD_BNDRE[2]), 2,
            jnp.where(
                (liqreff >= config.CFODD_BNDRE[2]) & (liqreff <= config.CFODD_BNDRE[3]),
                3, 0)))

    # ------------------------------------------------------------------
    # Cloud bottom / top detection per subcolumn (Fortran k = Nlevels..1,
    # i.e. surface upward): bottom = deepest detectable in-cloud echo,
    # top = shallowest one (gaps allowed at this stage).
    # ------------------------------------------------------------------
    echo = surrogate.ge(dbze, config.CFODD_DBZE_MIN, smoothing.reflectivity)
    detectable = ((dbze != R_GROUND) & (dbze != R_UNDEF)
                  & (fracout_int != config.SGCLD_CLR)
                  & (dbze >= config.CFODD_DBZE_MIN))            # (nlev, ncol, *b)
    exists = jnp.any(detectable, axis=0)                        # (ncol, *b)
    kctop = jnp.argmax(detectable, axis=0)                      # first from TOA
    kcbtm = nlev - 1 - jnp.argmax(detectable[::-1], axis=0)     # last from TOA

    # SLWC checks: warm cloud top ...
    temp_b = jnp.broadcast_to(_level_shape(temperature, dbze.ndim), dbze.shape)
    t_top = jnp.take_along_axis(temp_b, kctop[None], axis=0)[0]
    warm_top = surrogate.ge(t_top, TMELT, smoothing.temperature)
    # ... and every level between bottom and top a stratiform cell with a
    # detectable echo (convective or clear cells, or dBZe < -30, disqualify).
    in_cloud = (lev >= kctop[None]) & (lev <= kcbtm[None])
    # 0/1 per cell; its derivative is the echo's across the detection limit.
    fails = jnp.where(in_cloud,
                      1.0 - (fracout_int == config.SGCLD_ST) * echo, 0.0)
    slwc_ok = 1.0 - surrogate.any_of(fails, axis=0)
    col_ok = (exists * point_ok[None]).astype(dtype) * warm_top * slwc_ok

    # ------------------------------------------------------------------
    # Warm-rain regime from the column-maximum in-cloud dBZe.
    # ------------------------------------------------------------------
    cmxdbz = jnp.maximum(
        jnp.max(jnp.where(in_cloud, dbze, config.CFODD_DBZE_MIN), axis=0),
        config.CFODD_DBZE_MIN)  # Fortran initializes cmxdbz at CFODD_DBZE_MIN
    one = jnp.ones((), dtype) if weights is None else weights
    below0 = surrogate.lt(cmxdbz, config.CFODD_BNDZE[0], smoothing.reflectivity)
    below1 = surrogate.lt(cmxdbz, config.CFODD_BNDZE[1], smoothing.reflectivity)
    regimes = (below0, (1.0 - below0) * below1, (1.0 - below0) * (1.0 - below1))
    wr = jnp.stack([jnp.sum(one * col_ok * regime, axis=0).astype(dtype)
                    for regime in regimes])                     # (3, *b)

    # ------------------------------------------------------------------
    # In-cloud optical depth profile: ICOD(z) = COT (1 - ((z-z_btm)/H)^(5/3))
    # between cloud top and bottom; exactly COT at the bottom level.
    # ------------------------------------------------------------------
    z_b = jnp.broadcast_to(_level_shape(jnp.asarray(zlev, dtype), dbze.ndim),
                           dbze.shape)
    z_top = jnp.take_along_axis(z_b, kctop[None], axis=0)[0]
    z_btm = jnp.take_along_axis(z_b, kcbtm[None], axis=0)[0]
    cgt = z_top - z_btm                                         # (ncol, *b)
    # Clamp inside the masked branch: the ratio is only used for levels
    # strictly above the bottom of a multi-level cloud (cgt > 0 there); the
    # clip keeps 0^(5/3)-type gradients and single-level clouds harmless.
    # It is in [0, 1] wherever it is used; bounding it above as well keeps
    # the levels outside the cloud finite (above a single-level cloud it is
    # ~1e33 and its 5/3 power overflows float32), which a derivative
    # through the unused cells needs.
    ratio = jnp.clip((z_b - z_btm[None]) / jnp.maximum(cgt[None], 1e-30),
                     0.0, 1.0)
    liqcot_b = jnp.asarray(liqcot, dtype)[None, None]
    prof = liqcot_b * (1.0 - ratio ** (5.0 / 3.0))
    diagicod = jnp.where(lev == kcbtm[None], liqcot_b, prof)
    icod = jnp.minimum(diagicod, config.CFODD_ICOD_MAX)

    # ------------------------------------------------------------------
    # CFODD joint histogram (dBZe x ICOD) over all subcolumn/level cells,
    # stratified by Reff class; zero for screened-out points (the Fortran
    # never calls hist2D for them).
    # ------------------------------------------------------------------
    # FIXED UPSTREAM BUG (docs/translation_notes.md): the Fortran histograms
    # every subcolumn/level cell, so detectable echoes (dBZe >= -30) in
    # subcolumns that FAILED the SLWC screening still land in the first ICOD
    # bin (their ICOD stays 0). Only in-cloud cells of SLWC-passing
    # subcolumns are CFODD samples; everything else is pushed out of range.
    # The selection is the weight of each sample (see ``stats.hist2d``).
    cell_ok = jnp.broadcast_to(in_cloud * (one * col_ok)[None], dbze.shape)
    flat = lambda a: a.reshape((-1,) + dbze.shape[2:])
    hist = stats.hist2d(flat(dbze), flat(icod),
                        jnp.asarray(config.CFODD_HIST_DBZE, dtype),
                        jnp.asarray(config.CFODD_HIST_ICOD, dtype),
                        weights=flat(cell_ok), width=smoothing.histogram)
    cls_sel = jnp.stack([(icls == c + 1) & point_ok
                         for c in range(config.CFODD_NCLASS)])  # (3, *b)
    cfodd = jnp.where(cls_sel[None, None], hist[:, :, None], 0.0)

    # Points without a MODIS LWP retrieval (non-sunlit, or no liquid cloud
    # anywhere in the gridbox) are undefined, keyed on lwp == R_UNDEF alone
    # as in the Fortran initialization loop.
    dark = lwp == R_UNDEF
    cfodd = jnp.where(dark[None, None, None], R_UNDEF, cfodd)
    wr = jnp.where(dark[None], R_UNDEF, wr)
    return WarmRainDiagnostics(cfodd_ntotal=cfodd, wr_occfreq_ntotal=wr)
