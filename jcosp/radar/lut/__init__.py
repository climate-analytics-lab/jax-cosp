"""Offline Mie/DSD radar-optics core and Z-scale LUT generator for jcosp.

NumPy-only ports of the COSP v2.0 quickbeam_optics/optics_lib/math_lib
Fortran (Copyright (c) 2015, Regents of the University of Colorado,
BSD-3-Clause; see LICENSES/). These run offline to build and validate the
lookup tables consumed by the JAX radar simulator; they are not part of the
traced runtime.
"""

from jcosp.radar.lut.dsd import avint, calc_re, dsd, fgamma, infind, zeff
from jcosp.radar.lut.generate import (
    enumerate_re_bins,
    generate_lut,
    hydro_optics_entry,
    re_quantize,
)
from jcosp.radar.lut.mie import m_ice, m_wat, mie_efficiencies, mie_int

__all__ = [
    "avint",
    "calc_re",
    "dsd",
    "enumerate_re_bins",
    "fgamma",
    "generate_lut",
    "hydro_optics_entry",
    "infind",
    "m_ice",
    "m_wat",
    "mie_efficiencies",
    "mie_int",
    "re_quantize",
    "zeff",
]
