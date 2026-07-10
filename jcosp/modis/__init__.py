"""End-to-end MODIS simulator: subcolumn state in, MODIS L3 diagnostics out.

The lean JAX driver replacing COSP's ``cosp.F90`` orchestration for the MODIS
path (translation-plan Phase 5 scope): input optics (0.67-um tau painted on
subcolumns, phase partition, 3.7-um g/w0) -> per-subcolumn retrievals (cloud
mask, CTP, tau, phase, particle size) -> column means, fractions and joint
histograms. The CFODD / warm-rain-occurrence diagnostics that join these
retrievals with the CloudSat radar live in ``jcosp.modis.cfodd`` and are
driven by ``jcosp.simulator.simulate_cloudsat_modis``.

As in COSP, MODIS retrievals are only meaningful for sunlit points: pass
``sunlit`` (1 day / 0 night) to blank the column statistics of dark points
with R_UNDEF (the two tau-vs-Reff histograms become 0, as in cosp.F90).

Array convention: level axis 0 TOA-first, subcolumn axis 1, batch trailing.
"""

from typing import NamedTuple

import jax.numpy as jnp

from jcosp.constants import R_UNDEF
from jcosp.modis.cfodd import WarmRainDiagnostics, diag_warmrain
from jcosp.modis.optics import (ModisOptics, build_modis_optics, get_g_nir,
                                get_ssa_nir, modis_optics, optics_partition,
                                simulator_optics)
from jcosp.modis.simulator import (ModisColumn, ModisRetrievals,
                                   modis_column, modis_subcolumn)

__all__ = [
    "ModisOutputs", "simulate_modis",
    "ModisOptics", "build_modis_optics", "simulator_optics",
    "optics_partition", "modis_optics", "get_g_nir", "get_ssa_nir",
    "ModisRetrievals", "modis_subcolumn", "ModisColumn", "modis_column",
    "WarmRainDiagnostics", "diag_warmrain",
]


class ModisOutputs(NamedTuple):
    """MODIS simulator results."""

    optics: ModisOptics          # Subcolumn optical inputs (diagnostic)
    subcolumn: ModisRetrievals   # Per-subcolumn retrievals, (ncol, *batch)
    column: ModisColumn          # Column stats; CTP histogram axis flipped
    #                              to surface-first, dark points R_UNDEF


def simulate_modis(frac_out, mr_hydro, reff, dtau_s, dtau_c, pressure_half,
                   sunlit=None, isccp_ctp=None) -> ModisOutputs:
    """Run the MODIS path on subcolumn hydrometeor fields.

    Parameters
    ----------
    frac_out : (nlev, ncol, *batch)
        SCOPS subcolumn cloud-type map (0 clear / 1 stratiform / 2 conv.).
    mr_hydro, reff : (nlev, ncol, *batch, N_HYDRO)
        In-cloud subcolumn mixing ratios (kg/kg) and effective radii (m)
        from ``jcosp.inputs.distribute_hydrometeors``.
    dtau_s, dtau_c : (nlev, *batch)
        Gridbox-mean 0.67-um optical depth of stratiform / convective cloud
        per layer (host-model fields, as in the COSP driver).
    pressure_half : (nlev+1, *batch)
        Layer-edge pressures (Pa), TOA-first.
    sunlit : (*batch), optional
        Day mask (> 0 sunlit). Defaults to sunlit everywhere. Column
        statistics of dark points are R_UNDEF (tau-vs-Reff histograms 0).
    isccp_ctp : (ncol, *batch), optional
        ISCCP cloud-top pressure for low clouds (see ``modis_subcolumn``).
    """
    opt = build_modis_optics(frac_out, mr_hydro, reff, dtau_s, dtau_c)
    sub = modis_subcolumn(pressure_half, opt.tau_067, opt.frac_liq,
                          opt.g, opt.w0, isccp_ctp=isccp_ctp)
    col = modis_column(sub.phase, sub.cloud_top_pressure, sub.tau, sub.size)

    # cosp.F90 stores the tau/CTP joint histograms with the pressure bins
    # reordered surface-first.
    col = col._replace(tau_vs_ctp=col.tau_vs_ctp[:, ::-1],
                       tau_vs_ctp_liq=col.tau_vs_ctp_liq[:, ::-1],
                       tau_vs_ctp_ice=col.tau_vs_ctp_ice[:, ::-1])

    if sunlit is not None:
        day = jnp.asarray(sunlit) > 0

        def mask(x, ncat, fill=R_UNDEF):
            # ncat leading category axes precede the batch axes.
            return jnp.where(day[(None,) * ncat], x, fill)

        col = ModisColumn(*[
            # cosp.F90 zero-fills the two tau-vs-Reff histograms for dark
            # points (and leaves LWP/IWP-vs-Reff untouched — undefined in
            # Fortran; R_UNDEF here); everything else is R_UNDEF.
            mask(v, ncat=v.ndim - day.ndim,
                 fill=0.0 if name in ("tau_vs_reff_ice", "tau_vs_reff_liq")
                 else R_UNDEF)
            for name, v in zip(ModisColumn._fields, col)])

    return ModisOutputs(optics=opt, subcolumn=sub, column=col)
