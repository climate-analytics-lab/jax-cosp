"""Physical and numerical constants shared across jax-cosp.

Ported from COSP v2.0 ``src/cosp_constants.F90`` and ``src/cosp_config.F90``
(Copyright (c) 2015, Regents of the University of Colorado, BSD-3-Clause; see
LICENSES/). Values are kept bit-identical to the Fortran so that ported
routines validate against the reference implementation.
"""

# Missing / below-ground sentinel values (cosp_config.F90).
# COSP uses giant negative sentinels rather than NaN so that Fortran
# comparisons stay well-defined; we keep them for output compatibility, and
# because the CloudSat precipitation-flag logic tests against them explicitly.
R_UNDEF = -1.0e30
R_GROUND = -1.0e20

# Physical constants (cosp_constants.F90). COSP's radar code uses its own
# dry-air gas constant (287 J/kg/K) inline in quickbeam_optics; we name it
# here rather than borrow a host model's value so results match the Fortran.
TMELT = 273.15  # Melting temperature of ice/snow (K)
RHO_ICE = 917.0  # Density of solid ice (kg m-3)
RHO_LIQ = 1000.0  # Density of liquid water (kg m-3)
RD = 287.0  # Dry-air gas constant as used by quickbeam_optics (J kg-1 K-1)
RD_PRECIP = 287.05  # Dry-air gas constant as used by cosp_precip_mxratio

# Fraction of a gridbox covered by convective precipitation when PREC_SCOPS
# has no other information (cosp_config.F90: scops_ccfrac).
SCOPS_CCFRAC = 0.05
