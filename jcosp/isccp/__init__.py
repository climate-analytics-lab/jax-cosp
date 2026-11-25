"""End-to-end ISCCP simulator: subcolumn optics in, ISCCP diagnostics out.

The lean JAX driver replacing COSP's ``cosp.F90`` orchestration for the
ISCCP path (translation-plan Phase 6 scope): the ICARUS subcolumn stage
(10.5-um radiances, brightness temperatures, IR-adjusted cloud-top
pressures) followed by the column stage (cloud areas, cloudy means, the
7 x 7 tau/CTP joint histogram). ``boxptop`` feeds the MODIS simulator's
low-cloud CTP substitution (``jcosp.modis.simulate_modis(...,
isccp_ctp=...)``) exactly as cosp.F90 wires ``isccp_boxptop`` into
``modis_subcolumn``; ``jcosp.simulator.simulate_cloudsat_modis`` composes
the two when ``run_isccp=True``.

Array convention: level axis 0 TOA-first, subcolumn axis 1, batch trailing.
"""

from typing import NamedTuple

from jcosp import surrogate
from jcosp.isccp.simulator import (IsccpColumn, IsccpSubcolumn, isccp_column,
                                   isccp_subcolumn, water_vapour_emissivity)

__all__ = [
    "IsccpOutputs", "simulate_isccp",
    "IsccpSubcolumn", "isccp_subcolumn",
    "IsccpColumn", "isccp_column",
    "water_vapour_emissivity",
]


class IsccpOutputs(NamedTuple):
    """ISCCP simulator results."""

    subcolumn: IsccpSubcolumn  # Per-subcolumn tau/ptop/Tb, (ncol, *batch);
    #                            boxptop (Pa) feeds simulate_modis(isccp_ctp=)
    column: IsccpColumn        # Column stats; CTP histogram axis flipped
    #                            to surface-first, dark points R_UNDEF


def simulate_isccp(frac_out, tau_067, emiss_11, skt, emsfc_lw, temperature,
                   pressure_full, pressure_half, qv, sunlit=None,
                   top_height: int = 1,
                   top_height_direction: int = 2, weights=None,
                   smoothing=surrogate.DEFAULT_SMOOTHING) -> IsccpOutputs:
    """Run the ISCCP (ICARUS) path on subcolumn optical fields.

    Parameters
    ----------
    frac_out : (nlev, ncol, *batch)
        SCOPS subcolumn cloud-type map (0 clear / 1 stratiform / 2 conv.);
        read only for ``top_height=2`` (physical cloud top).
    tau_067, emiss_11 : (nlev, ncol, *batch)
        Subcolumn 0.67-um layer optical thickness and 10.5-um cloud
        emissivity — the host's ``dtau_s``/``dtau_c`` and ``dem_s``/``dem_c``
        merged through the SCOPS map with
        ``jcosp.modis.optics.simulator_optics``, as in the COSP driver.
    skt : (*batch)
        Skin temperature (K).
    emsfc_lw : scalar
        10.5-um surface emissivity (fraction).
    temperature, pressure_full, qv : (nlev, *batch)
        Layer temperature (K), pressure (Pa) and specific humidity (kg/kg).
    pressure_half : (nlev+1, *batch)
        Layer-edge pressures (Pa), TOA-first.
    sunlit : (*batch), optional
        Day mask (> 0 sunlit). Defaults to sunlit everywhere. Night points
        report R_UNDEF box values and column statistics (``meantb`` and
        ``meantbclr`` stay valid — the 10.5-um channel works at night)
        unless ``top_height=3``, which reports day and night alike.
    top_height, top_height_direction : int
        Cloud-top method and match direction (COSP ``isccp_init``); the
        defaults (1, 2) are the ISCCP-recommended configuration. See
        :func:`jcosp.isccp.simulator.isccp_subcolumn`.
    weights : (ncol, *batch), optional
        SCOPS subcolumn weight (``jcosp.subcol.scops.subcolumn_weights``),
        through which the column statistics respond to the cloud fraction.
    smoothing : jcosp.surrogate.Smoothing
        Widths of the derivatives across the cloud-area limit and the bin
        edges; no value depends on it.
    """
    sub = isccp_subcolumn(tau_067, emiss_11, frac_out, skt, emsfc_lw, qv,
                          temperature, pressure_full, pressure_half,
                          sunlit=sunlit, top_height=top_height,
                          top_height_direction=top_height_direction)
    # The Fortran drivers hand ICARUS_COLUMN boxptop in hPa (boxptop/100,
    # applied to the R_UNDEF fills too — they stay far out of every bin).
    col = isccp_column(sub.boxtau, sub.boxptop / 100.0, sunlit, sub.boxttop,
                       top_height=top_height, weights=weights,
                       smoothing=smoothing)
    # cosp.F90 stores fq_isccp with the pressure bins reordered
    # surface-first (isccp_fq(:,:,7:1:-1)), as for the MODIS histograms.
    col = col._replace(fq_isccp=col.fq_isccp[:, ::-1])
    return IsccpOutputs(subcolumn=sub, column=col)
