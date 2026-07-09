"""Static configuration for the CloudSat/Quickbeam radar path.

Ported from COSP v2.0 ``src/cosp_config.F90``, ``src/cosp_stats.F90`` (table
dimensions), ``src/simulator/cosp_cloudsat_interface.F90`` (radar setup and
Re-bin structure) and ``quickbeam_optics.F90:hydro_class_init`` (hydrometeor
size-distribution tables). Copyright (c) 2015, Regents of the University of
Colorado, BSD-3-Clause; see LICENSES/.

Everything in this module is *static* (Python/NumPy, resolved at trace time):
hydrometeor class definitions, histogram bins, the Re-bin quantization used by
the radar scale LUT, and the CloudSat statistical vertical grid. Runtime
(traced) state never lives here.
"""

from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Hydrometeor classes (cosp_config.F90: N_HYDRO and index parameters).
# Order is COSP's canonical one; all per-class tables below follow it.
# ---------------------------------------------------------------------------
N_HYDRO = 9
I_LSCLIQ = 0  # Large-scale (stratiform) cloud liquid
I_LSCICE = 1  # Large-scale cloud ice
I_LSRAIN = 2  # Large-scale rain
I_LSSNOW = 3  # Large-scale snow
I_CVCLIQ = 4  # Convective cloud liquid
I_CVCICE = 5  # Convective cloud ice
I_CVRAIN = 6  # Convective rain
I_CVSNOW = 7  # Convective snow
I_LSGRPL = 8  # Large-scale graupel

HYDRO_NAMES = (
    "LSCLIQ", "LSCICE", "LSRAIN", "LSSNOW",
    "CVCLIQ", "CVCICE", "CVRAIN", "CVSNOW", "LSGRPL",
)

# SCOPS subcolumn cloud-type codes (frac_out) and PREC_SCOPS precip codes.
SGCLD_CLR = 0  # clear
SGCLD_ST = 1   # stratiform / large-scale cloud
SGCLD_CUM = 2  # convective cloud
PREC_NONE = 0
PREC_LS = 1
PREC_CV = 2
PREC_MIXED = 3


# ---------------------------------------------------------------------------
# Hydrometeor size-distribution tables (quickbeam_optics.F90:hydro_class_init)
#
# dtype: 1 modified gamma, 2 exponential, 3 power law, 4 monodisperse,
#        5 lognormal.  phase: 0 liquid, 1 ice.
# Density is apm*D^bpm (kg m-3, D in m) OR the constant rho; exactly one of
# (apm, bpm) / rho is set per class (-1 marks "unset", as in the Fortran).
# p1, p2, p3 are the per-distribution default parameters (see the extensive
# comment block in hydro_class_init).
# ---------------------------------------------------------------------------
class SizeDistribution(NamedTuple):
    """Per-hydrometeor-class PSD parameters (Fortran type ``size_distribution``)."""

    dtype: np.ndarray
    phase: np.ndarray
    dmin: np.ndarray
    dmax: np.ndarray
    apm: np.ndarray
    bpm: np.ndarray
    rho: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    p3: np.ndarray


def _sd(dtype, phase, dmin, dmax, apm, bpm, rho, p1, p2, p3) -> SizeDistribution:
    return SizeDistribution(
        dtype=np.array(dtype, dtype=np.int64),
        phase=np.array(phase, dtype=np.int64),
        dmin=np.array(dmin, dtype=np.float64),
        dmax=np.array(dmax, dtype=np.float64),
        apm=np.array(apm, dtype=np.float64),
        bpm=np.array(bpm, dtype=np.float64),
        rho=np.array(rho, dtype=np.float64),
        p1=np.array(p1, dtype=np.float64),
        p2=np.array(p2, dtype=np.float64),
        p3=np.array(p3, dtype=np.float64),
    )


# Single-moment scheme (HCLASS1_*): the COSP default
# ('MMF_v3_single_moment'); also what ECHAM-HAM style hosts use.
#                    LSL    LSI    LSR    LSS   CVL    CVI    CVR    CVS   LSG
HYDRO_CLASSES_1M = _sd(
    dtype=[5,     1,     2,     2,    5,     1,     2,     2,    2],
    phase=[0,     1,     0,     1,    0,     1,     0,     1,    1],
    dmin=[-1.,   -1.,   -1.,   -1.,  -1.,   -1.,   -1.,   -1.,  -1.],
    dmax=[-1.,   -1.,   -1.,   -1.,  -1.,   -1.,   -1.,   -1.,  -1.],
    apm=[524., 110.8,  524.,   -1., 524., 110.8,  524.,   -1.,  -1.],
    bpm=[3.,    2.91,   3.,    -1.,   3.,   2.91,   3.,   -1.,  -1.],
    rho=[-1.,   -1.,   -1.,   100.,  -1.,  -1.,   -1.,   100., 400.],
    p1=[-1.,    -1.,    8.e6,  3.e6, -1.,  -1.,    8.e6,  3.e6, 4.e6],
    p2=[6.,     40.,   -1.,   -1.,    6.,  40.,   -1.,   -1.,  -1.],
    p3=[0.3,     2.,   -1.,   -1.,    0.3,  2.,   -1.,   -1.,  -1.],
)

# Two-moment scheme (HCLASS2_*, 'MMF_v3.5_two_moment', Morrison-style).
# LSL/CVL use p3 = -2 (Martin-1994 shape parameter from number concentration),
# so their PSD width depends on the runtime Np — see radar/lut/generate.py for
# why this scheme is not yet supported by the precomputed-LUT runtime.
HYDRO_CLASSES_2M = _sd(
    dtype=[1,   1,    1,    1,    1,   1,    1,    1,    1],
    phase=[0,   1,    0,    1,    0,   1,    0,    1,    1],
    dmin=[-1., -1.,  -1.,  -1.,  -1., -1.,  -1.,  -1.,  -1.],
    dmax=[-1., -1.,  -1.,  -1.,  -1., -1.,  -1.,  -1.,  -1.],
    apm=[524., -1., 524.,  -1., 524., -1., 524.,  -1.,  -1.],
    bpm=[3.,   -1.,   3.,  -1.,   3., -1.,   3.,  -1.,  -1.],
    rho=[-1., 500.,  -1., 100.,  -1., 500., -1., 100., 900.],
    p1=[-1.,  -1.,   -1.,  -1.,  -1., -1.,  -1.,  -1.,  -1.],
    p2=[-1.,  -1.,   -1.,  -1.,  -1., -1.,  -1.,  -1.,  -1.],
    p3=[-2.,   1.,    1.,   1.,  -2.,  1.,   1.,   1.,   1.],
)

HYDRO_CLASSES = {"1M": HYDRO_CLASSES_1M, "2M": HYDRO_CLASSES_2M}


# ---------------------------------------------------------------------------
# Precipitation-flux -> mixing-ratio conversion parameters
# (driver/src/cosp2_test.F90; used by cosp_precip_mxratio). -1 marks classes
# that are not precipitation (their fluxes are never converted).
# ---------------------------------------------------------------------------
#                 LSL   LSI    LSR       LSS      CVL   CVI    CVR       CVS       LSG
PRECIP_N_AX = np.array([-1., -1., 8.e6, 3.e6, -1., -1., 8.e6, 3.e6, 4.e6])
PRECIP_N_BX = np.array([-1., -1., 0.0, 0.0, -1., -1., 0.0, 0.0, 0.0])
PRECIP_ALPHA_X = np.array([-1., -1., 0.0, 0.0, -1., -1., 0.0, 0.0, 0.0])
PRECIP_C_X = np.array([-1., -1., 842.0, 4.84, -1., -1., 842.0, 4.84, 94.5])
PRECIP_D_X = np.array([-1., -1., 0.8, 0.25, -1., -1., 0.8, 0.25, 0.5])
PRECIP_G_X = np.array([-1., -1., 0.5, 0.5, -1., -1., 0.5, 0.5, 0.5])
PRECIP_A_X = np.array([-1., -1., 524.0, 52.36, -1., -1., 524.0, 52.36, 209.44])
PRECIP_B_X = np.array([-1., -1., 3.0, 3.0, -1., -1., 3.0, 3.0, 3.0])
PRECIP_GAMMA_1 = np.array([-1., -1., 17.83725, 8.284701, -1., -1., 17.83725, 8.284701, 11.63230])
PRECIP_GAMMA_2 = np.array([-1., -1., 6.0, 6.0, -1., -1., 6.0, 6.0, 6.0])
PRECIP_GAMMA_3 = np.array([-1., -1., 2.0, 2.0, -1., -1., 2.0, 2.0, 2.0])
PRECIP_GAMMA_4 = np.array([-1., -1., 6.0, 6.0, -1., -1., 6.0, 6.0, 6.0])
# Which prec_scops code feeds each class: 1 large-scale, 2 convective.
PRECIP_TYPE = np.array([-1., -1., 1., 1., -1., -1., 2., 2., 1.])


# ---------------------------------------------------------------------------
# Quickbeam scale-LUT dimensions and discrete-DSD grid (cosp_stats.F90).
# ---------------------------------------------------------------------------
MAXHCLASS = 20      # Max hydrometeor classes the tables allow
N_RE_TYPES = 550    # Number of Re bins in the Z-scale LUT
ND = 85             # Number of discrete particle sizes used to build DSDs
MT_NTT = 39         # Temperature slots in the LUT (liquid uses 19, ice 20)
RE_BIN_LENGTH = 10  # Minimum Re interval (um) in the scale LUT
RE_MAX_BIN = 250    # Number of coarse Re intervals
DMIN = 0.1          # Smallest discrete particle size (um)
DMAX = 10000.0      # Largest discrete particle size (um)

# Discrete particle-size grid (um): nd log-spaced points from DMIN to DMAX
# (quickbeam_optics_init). Identical for every class and LUT entry.
D_GRID = DMIN * np.exp(np.arange(ND) * (np.log(DMAX) - np.log(DMIN)) / (ND - 1))

# LUT temperature grids (K): 5-K steps, liquid -60..+30 C, ice -90..+5 C
# (quickbeam_optics_init: mt_ttl / mt_tti).
MT_TTL = np.arange(19) * 5.0 - 60.0 + 273.15
MT_TTI = np.arange(20) * 5.0 - 90.0 + 273.15


def re_bin_structure() -> tuple[np.ndarray, np.ndarray]:
    """Return (step_list, base_list) of the Re quantization (COSP_CLOUDSAT_INIT).

    The scale LUT resolves effective radius on a piecewise grid: within coarse
    interval ``n`` (each RE_BIN_LENGTH um wide) the resolution is
    ``step_list[n] = 0.1 + 0.1*n**1.5`` um, capped at RE_BIN_LENGTH, and
    ``base_list[n]`` is the running offset of that interval's first LUT slot.
    """
    step_list = np.empty(RE_MAX_BIN)
    base_list = np.zeros(RE_MAX_BIN)
    for j in range(RE_MAX_BIN):
        step_list[j] = min(0.1 + 0.1 * j**1.5, float(RE_BIN_LENGTH))
        if j > 0:
            base_list[j] = base_list[j - 1] + np.floor(RE_BIN_LENGTH / step_list[j - 1])
    return step_list, base_list


RE_STEP_LIST, RE_BASE_LIST = re_bin_structure()


# ---------------------------------------------------------------------------
# CloudSat radar configuration (COSP_CLOUDSAT_INIT defaults from the driver).
# ---------------------------------------------------------------------------
class RadarConfig(NamedTuple):
    """Static radar-simulator configuration (Fortran type ``radar_cfg``).

    The Z-scale tables themselves live in a separate LUT file (see
    ``jcosp.radar.lut``); this holds only the scalar setup.
    """

    freq: float = 94.0        # Radar frequency (GHz)
    k2: float = -1.0          # |K|^2; -1 selects the frequency-dependent default
    use_gas_abs: int = 1      # 1: per-profile gaseous attenuation, 0: none
    do_ray: int = 0           # 1: also compute pure-Rayleigh reflectivity
    nhclass: int = N_HYDRO
    radar_at_layer_one: bool = True  # True: spaceborne radar at index-0 layer (TOA-first)
    micro_scheme: str = "1M"  # Which HYDRO_CLASSES table the LUT was built for


CLOUDSAT_RADAR = RadarConfig()


def effective_k2(freq: float, k2: float) -> float:
    """Resolve the |K|^2 dielectric factor default (quickbeam_optics.F90:zeff)."""
    if k2 >= 0:
        return k2
    if abs(94.0 - freq) < 3.0:
        return 0.75
    if abs(35.0 - freq) < 3.0:
        return 0.88
    if abs(13.8 - freq) < 3.0:
        return 0.925
    return 0.933


# ---------------------------------------------------------------------------
# CloudSat reflectivity histogram (CFAD) and precipitation-class constants.
# ---------------------------------------------------------------------------
CLOUDSAT_DBZE_BINS = 15
CLOUDSAT_DBZE_MIN = -100.0
CLOUDSAT_DBZE_MAX = 80.0
CLOUDSAT_CFAD_ZE_MIN = -50.0
CLOUDSAT_CFAD_ZE_WIDTH = 5.0
# Bin boundaries: [-100, -45, -40, ..., 20, 80] (cosp_config.F90 cloudsat_histRef)
CLOUDSAT_HIST_REF = np.concatenate([
    [CLOUDSAT_DBZE_MIN],
    CLOUDSAT_CFAD_ZE_MIN + CLOUDSAT_CFAD_ZE_WIDTH * np.arange(1, CLOUDSAT_DBZE_BINS),
    [CLOUDSAT_DBZE_MAX],
])

# CloudSat 2C-PRECIP-COLUMN-style precipitation classes (cosp_config.F90).
N_CLOUDSAT_PRECIP_CLASS = 10
PCLASS_NO_PRECIP = 0
PCLASS_RAIN1 = 1   # Rain possible
PCLASS_RAIN2 = 2   # Rain probable
PCLASS_RAIN3 = 3   # Rain certain
PCLASS_SNOW1 = 4   # Snow possible
PCLASS_SNOW2 = 5   # Snow certain
PCLASS_MIXED1 = 6  # Mixed possible
PCLASS_MIXED2 = 7  # Mixed certain
PCLASS_RAIN4 = 8   # Heavy rain
PCLASS_DEFAULT = 9

# Reflectivity thresholds (dBZe) for the ocean (Ze_non) and land (Ze) paths.
ZE_NON_BINVAL = np.array([0.0, -5.0, -7.5, -15.0])
Z_BINVAL_LND = np.array([10.0, 5.0, 2.5, -2.5, -5.0, -15.0])

# Index (0-based) into the 40-level CloudSat statistical grid used for the
# near-surface precipitation flags: level 38 of 40 spans 480-960 m, i.e. the
# first gate clear of CloudSat's ~720-m surface clutter. (Fortran
# cloudsat_preclvl=39 is 1-based into a TOA-first 40-level grid.)
CLOUDSAT_PRECLVL = 38

# CloudSat statistical vertical grid: 40 levels x 480 m, TOA-first
# (cosp.F90 COSP_INIT with luseCSATvgrid=.true.).
N_VGRID = 40
VGRID_ZSTEP = 480.0
# TOA-first: vgrid_zl[0] is the bottom of the topmost (highest) layer.
VGRID_ZL = VGRID_ZSTEP * np.arange(N_VGRID - 1, -1, -1.0)
VGRID_ZU = VGRID_ZL + VGRID_ZSTEP
VGRID_Z = 0.5 * (VGRID_ZL + VGRID_ZU)


# ---------------------------------------------------------------------------
# Warm-rain / CFODD diagnostic constants (cosp_config.F90, Michibata et al.
# 2019 GMD). CFODD_BNDZE are also the Mülmenstädt et al. (2020) drizzle/rain
# reflectivity thresholds (-15 / 0 dBZe).
# ---------------------------------------------------------------------------
WR_NREGIME = 3  # non-precipitating / drizzling / raining
CFODD_NCLASS = 3
CFODD_BNDRE = np.array([5.0e-6, 12.0e-6, 18.0e-6, 35.0e-6])  # Reff bounds (m)
CFODD_BNDZE = np.array([-15.0, 0.0])  # dBZe bounds: cloud/drizzle/rain
CFODD_DBZE_MIN = -30.0
CFODD_DBZE_MAX = 20.0
CFODD_ICOD_MIN = 0.0
CFODD_ICOD_MAX = 60.0
CFODD_DBZE_WIDTH = 2.0
CFODD_ICOD_WIDTH = 2.0
CFODD_NDBZE = int((CFODD_DBZE_MAX - CFODD_DBZE_MIN) / CFODD_DBZE_WIDTH)
CFODD_NICOD = int((CFODD_ICOD_MAX - CFODD_ICOD_MIN) / CFODD_ICOD_WIDTH)
CFODD_HIST_DBZE = CFODD_DBZE_MIN + CFODD_DBZE_WIDTH * np.arange(CFODD_NDBZE + 1)
CFODD_HIST_ICOD = CFODD_ICOD_MIN + CFODD_ICOD_WIDTH * np.arange(CFODD_NICOD + 1)
CWP_THRESHOLD = 0.0  # LWP/IWP screening threshold (kg m-2)
COT_THRESHOLD = 0.3  # Cloud-optical-thickness screening threshold
