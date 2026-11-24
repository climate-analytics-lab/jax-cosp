"""Water paths as a visible/near-infrared imager retrieval reports them.

A passive imager does not see a column's liquid and ice water paths. Its
retrieval treats a cloudy pixel as ONE layer of ONE phase: it finds the
column optical depth ``tau`` and the particle effective radius ``r_e`` near
cloud top, converts their product to a water path,

    CWP = 4/3 * tau * r_e * rho / Q_e,

and books the whole column under the phase seen at cloud top. Liquid beneath
an ice top is therefore reported as ice, at the ice crystal size.

The MODIS simulator already produces everything this needs per subcolumn
(``jcosp.modis.simulator.modis_subcolumn``: detection, the liquid share of the
extinction in the top optical depth, and the particle size the 3.7-um
reflectance implies for either phase). COSP's own column stage
(``modis_column``) is one such product, MODIS L3: it leaves the subcolumns of
mixed cloud-top phase out of both paths and reports in-cloud means. Other
imager products differ only in the constants of the conversion, the limits of
the retrieval and the phase rule, which :class:`ImagerRetrieval` holds, so a
second product is a second set of constants over the same subcolumn
retrievals and costs no further radiative transfer.

:data:`CLOUD_CCI` is ESA Cloud_cci's CC4CL optimal-estimation retrieval
(McGarragh et al. 2018, AMT 11, 3397), whose all-sky monthly means
(``lwp_allsky``, ``iwp_allsky``) :func:`imager_water_paths` mirrors.

Array convention: subcolumn axis 0, batch trailing (the ``ModisRetrievals``
layout).
"""

from typing import NamedTuple

import jax.numpy as jnp

from jcosp import config, surrogate
from jcosp.modis.simulator import ModisRetrievals


class ImagerRetrieval(NamedTuple):
    """The constants that define one imager cloud-water-path product."""

    density_liquid: float      # rho of the conversion, liquid (kg m-3)
    density_ice: float         # ... ice
    extinction_liquid: float   # Q_e of the conversion, liquid
    extinction_ice: float      # ... ice
    tau_max: float             # upper retrieval limit of the optical depth
    size_liquid: tuple         # (min, max) retrievable liquid radius (m)
    size_ice: tuple            # (min, max) retrievable ice radius (m)
    # A subcolumn is liquid when liquid holds MORE than this share of the
    # extinction in the top optical depth of the cloud, and ice otherwise:
    # every detected pixel gets one of the two phases.
    liquid_fraction: float


# ESA Cloud_cci, CC4CL (McGarragh et al. 2018). Conversion: their Eq. 60
# (after Han et al. 1994). Limits: their Tables 3 and 4 (log10(tau) <= 2.408;
# 1-35 um liquid, 4-92 um ice). Phase: Cloud_cci v3 assigns one of two phases
# to every cloudy pixel with a network trained on CALIOP's cloud-top phase
# (Stengel et al. 2020, ESSD 12, 41), hence the majority rule. The size
# limits contain the MODIS simulator's trial radii (4-30 and 5-90 um), so the
# simulated size is never clipped by them; they are stated because they are
# part of the product.
CLOUD_CCI = ImagerRetrieval(
    density_liquid=1000.0, density_ice=916.7,
    extinction_liquid=2.0, extinction_ice=2.1,
    tau_max=256.0,
    size_liquid=(1.0e-6, 35.0e-6), size_ice=(4.0e-6, 92.0e-6),
    liquid_fraction=0.5,
)

# Solar zenith angle below which CC4CL retrieves optical depth and particle
# size ("day"), as its cosine: 80 degrees. The retrieval needs reflected
# sunlight; the host masks dark points (see ``imager_water_paths``).
CLOUD_CCI_MIN_COS_ZENITH = 0.17364818


def imager_water_paths(sub: ModisRetrievals,
                       retrieval: ImagerRetrieval = CLOUD_CCI, weights=None,
                       smoothing=surrogate.DEFAULT_SMOOTHING):
    """All-sky liquid and ice water paths of an imager retrieval.

    Each subcolumn the MODIS simulator detects (column optical depth of at
    least ``MODIS_MIN_OPTICAL_THICKNESS``) is given one phase by
    ``retrieval.liquid_fraction``, the particle size the simulator retrieves
    for that phase, and the water path ``4/3 * min(tau, tau_max) * r_e * rho
    / Q_e``. The result is the mean over ALL subcolumns, with clear ones,
    those of the other phase and those whose size retrieval failed counting
    as zero: the all-sky convention of the satellite products' gridded means.

    Nothing here knows about daylight. The retrieval needs it, so the host
    multiplies the result by its day mask and averages that mask alongside
    (``CLOUD_CCI_MIN_COS_ZENITH`` for Cloud_cci): the mean over the daylight
    hours is the time mean of a path divided by the time mean of the mask.

    The detection and phase tests are hard steps in the value. Their
    derivatives are those of smooth steps of the widths in ``smoothing``
    (``optical_depth``, ``phase_fraction``), so that a path responds to a
    subcolumn approaching the detection limit or the other phase, and
    through ``weights`` (the SCOPS subcolumn weight) to the cloud fraction.
    Values and derivatives are finite everywhere.

    Parameters
    ----------
    sub : ModisRetrievals
        Per-subcolumn retrievals, fields ``(ncol, *batch)``.
    retrieval : ImagerRetrieval
        The product's constants.
    weights : (ncol, *batch), optional
        SCOPS subcolumn weight (``jcosp.subcol.scops.subcolumn_weights``).
    smoothing : jcosp.surrogate.Smoothing
        Widths of the derivatives; no value depends on it.

    Returns
    -------
    (liquid_water_path, ice_water_path) : each ``(*batch)``, kg m-2.
    """
    dtype = sub.liquid_fraction.dtype
    if sub.tau_all is None:
        detected = (sub.phase != config.MODIS_PHASE_NONE).astype(dtype)
        tau = jnp.where(detected > 0.5, sub.tau, 0.0)
    else:
        tau = sub.tau_all
        detected = surrogate.ge(tau, config.MODIS_MIN_OPTICAL_THICKNESS,
                                smoothing.optical_depth)
    liquid = surrogate.gt(sub.liquid_fraction, retrieval.liquid_fraction,
                          smoothing.phase_fraction)
    tau = jnp.minimum(tau, retrieval.tau_max)

    def path(of_phase, size, limits, density, extinction):
        # A failed size retrieval is negative (MODIS_RE_FILL).
        use = detected * of_phase * (size > 0.0)
        r_e = jnp.clip(size, *limits)
        cells = surrogate.gated(
            use, (4.0 / 3.0) * tau * r_e * (density / extinction))
        return jnp.mean(cells if weights is None else weights * cells, axis=0)

    lwp = path(liquid, sub.size_if_liquid, retrieval.size_liquid,
               retrieval.density_liquid, retrieval.extinction_liquid)
    iwp = path(1.0 - liquid, sub.size_if_ice, retrieval.size_ice,
               retrieval.density_ice, retrieval.extinction_ice)
    return lwp, iwp
