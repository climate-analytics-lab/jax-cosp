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


# ---------------------------------------------------------------------------
# CALIPSO lidar simulator configuration (appended for the Phase-4 lidar port).
# Sources: cosp_config.F90 (CALIPSO section), cosp_calipso_interface.F90
# (cosp_calipso_init), lidar_simulator.F90 (COSP_CLDFRAC local parameters) and
# cosp_optics.F90:lidar_optics (local data). Copyright (c) 2015, Regents of
# the University of Colorado / (c) 2009 CNRS, BSD-3-Clause; see LICENSES/.
# ---------------------------------------------------------------------------
SR_BINS = 15  # Scattering-ratio histogram bins (cosp_config.F90)
# GOCCP scattering-ratio thresholds (cosp_config.F90).
CALIPSO_S_CLD = 5.0       # SR above which a volume is "cloudy"
CALIPSO_S_ATT = 0.01      # SR below which the signal is fully attenuated
CALIPSO_S_CLD_ATT = 30.0  # SR above which everything below is phase-"undefined"
# SR histogram bin boundaries (cosp_config.F90: calipso_histBsct). The last
# edge (999) doubles as the "physically meaningful SR" cap when forming SR.
CALIPSO_HIST_BSCT = np.array([
    -1.0, 0.01, 1.2, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0,
    50.0, 60.0, 80.0, 999.0])
LIDAR_NCAT = 4    # Cloud-layer categories: low, mid, high, total
LIDAR_NTYPE = 3   # OPAQ categories (opaque/thin/z_opaque) — OPAQ not yet ported
LIDAR_NTEMP = 40  # Temperature bins of the phase-vs-temperature diagnostic
LIDAR_NPHASE = 6  # [ice, liquid, undefined, false ice, false liquid, ice frac]
# Temperature bin edges (K) of the phase-vs-temperature diagnostic
# (lidar_simulator.F90: COSP_CLDFRAC 'tempmod'): 0, 183.15..297.15 by 3, 473.15.
LIDAR_PHASE_TEMP_EDGES = np.concatenate(
    [[0.0], 183.15 + 3.0 * np.arange(39), [473.15]])
# ISCCP-style pressure bounds (Pa) of the low/mid/high categories (COSP_CLDFRAC).
CALIPSO_HIGH_CLOUD_PMAX = 440.0e2  # p < 440 hPa: high
CALIPSO_MID_CLOUD_PMAX = 680.0e2   # 440 <= p < 680 hPa: mid; p >= 680 hPa: low
# Number of statistical-grid levels above 8.16 km (COSP_CLDFRAC treats levels
# 1..23 of the 40x480m grid, 19.2->8.16 km, as never fully attenuating).
CALIPSO_NLEV_HIGHALT = 23
# ATBperp(ATB) parameterization from CALIPSO-GOCCP observations at 120 m
# (Cesana and Chepfer 2013, JGR; cosp_calipso_init):
#   ice:    ATBperp = ALPHA * ATB
#   liquid: ATBperp = 1000 * BETA * ATB^2 + GAMMA * ATB
CALIPSO_ATBPERP_ALPHA = 0.2904
CALIPSO_ATBPERP_BETA = 0.4099
CALIPSO_ATBPERP_GAMMA = 0.009
# Phase-discrimination polynomial (COSP_CLDFRAC alpha50..zeta50, Cesana and
# Chepfer 2013): ATBperp_line = c0*ATB^5 + c1*ATB^4 + ... + c4*ATB + c5.
CALIPSO_PHASE_DISCR_COEFFS = np.array([
    9.0322e+15, -2.1358e+12, 173.3963e+06, -3.9514e+03, 0.2559, -9.4776e-07])
# Temperature overrides of the polarization phase classification (K).
CALIPSO_T_FALSE_ICE = 273.15  # "ice" warmer than this is reclassified liquid
CALIPSO_T_FALSE_LIQ = 231.15  # "liquid" colder than this is reclassified ice
# Molecular depolarization: the perpendicular molecular backscatter is
# beta_mol / (1 + 1/0.0284) (Bodhaine et al. 1999; lidar_simulator.F90).
CALIPSO_MOL_DEPOL = 1.0 / (1.0 + 1.0 / 0.0284)

# ---------------------------------------------------------------------------
# MODIS simulator configuration (appended for the Phase-5 MODIS port).
# Sources: cosp_modis_interface.F90 (COSP_MODIS_INIT retrieval parameters),
# modis_simulator.F90 (module parameters; R. Pincus, S. Platnick 2009-2010,
# D. Swales 2015) and cosp_config.F90 (joint-histogram bin tables; MODIS
# LWP/IWP/Reff histograms by C. Wall 2024). Copyright (c) 2015, Regents of
# the University of Colorado, BSD-3-Clause; see LICENSES/.
# ---------------------------------------------------------------------------
# Retrieval parameters (COSP_MODIS_INIT). Sizes here are microns, as inside
# the Fortran module; the retrieved size *output* is metres.
MODIS_NUM_TRIAL_RES = 15          # Trial radii of the size pseudo-retrieval
MODIS_MIN_OPTICAL_THICKNESS = 0.3  # Minimum detectable optical thickness
MODIS_CO2SLICING_PRESSURE_LIMIT = 70000.0  # Pa; lower clouds use ISCCP CTP
MODIS_CO2SLICING_TAU_LIMIT = 1.0  # Optical depth the CO2-slicing CTP sees
MODIS_PHASE_TAU_LIMIT = 1.0       # Optical depth the phase detection sees
MODIS_SIZE_TAU_LIMIT = 2.0        # Declared in COSP_MODIS_INIT; unused by the
#                                   v2 size retrieval (kept for completeness)
MODIS_PHASE_DISCRIMINATION_THRESHOLD = 0.7  # Extinction fraction for phase
MODIS_RE_FILL = -999.0            # Failed-retrieval fill value (um frame)
MODIS_RE_WATER_MIN = 4.0          # Liquid effective-radius bounds (um)
MODIS_RE_WATER_MAX = 30.0
MODIS_RE_ICE_MIN = 5.0            # Ice effective-radius bounds (um)
MODIS_RE_ICE_MAX = 90.0
MODIS_HIGH_CLOUD_PRESSURE_LIMIT = 44000.0  # Pa: CTP <= this is "high" cloud
MODIS_LOW_CLOUD_PRESSURE_LIMIT = 68000.0   # Pa: CTP >  this is "low" cloud
# Retrieved-phase codes.
MODIS_PHASE_NONE = 0
MODIS_PHASE_LIQUID = 1
MODIS_PHASE_ICE = 2
MODIS_PHASE_UNDETERMINED = 3
# modis_simulator.F90 module parameters.
MODIS_ICE_DENSITY = 0.93          # Ice density relative to liquid (=1)
MODIS_LWP_CONVERSION = 2.0 / 3.0 * 1000.0  # W = (2/3) rho_liq re tau (MKS)

# Joint-histogram bins (cosp_config.F90). The tau axis is the COSP v2 default
# (numMODISTauBins = ntau, set in COSP_INIT); the v1.4 6-bin variant that
# drops the [0, 0.3) bin is not ported.
NUM_MODIS_TAU_BINS = 7
MODIS_HIST_TAU = np.array([0.0, 0.3, 1.3, 3.6, 9.4, 23.0, 60.0, 10000.0])
MODIS_HIST_TAU_CENTERS = np.array([0.15, 0.80, 2.45, 6.5, 16.2, 41.5, 100.0])
# Cloud-top-pressure bins (Pa; modis_histPres = 100 * pres_binBounds [hPa]).
# Ascending pressure = TOA-first; cosp.F90 flips the CTP axis of the joint
# histograms to surface-first when storing output (reproduced by the driver).
NUM_MODIS_PRES_BINS = 7
MODIS_HIST_PRES = 100.0 * np.array(
    [0.0, 180.0, 310.0, 440.0, 560.0, 680.0, 800.0, 10000.0])
MODIS_HIST_PRES_CENTERS = np.array(
    [90000.0, 74000.0, 62000.0, 50000.0, 37500.0, 24500.0, 9000.0])
# Particle-size bins (m), Pincus et al. 2023 (doi:10.5194/essd-15-2483-2023).
NUM_MODIS_REFF_LIQ_BINS = 8
MODIS_HIST_REFF_LIQ = np.array(
    [0.0, 4.0e-6, 8e-6, 1.0e-5, 1.25e-5, 1.5e-5, 2.0e-5, 3.0e-5, 1.0e-2])
NUM_MODIS_REFF_ICE_BINS = 8
MODIS_HIST_REFF_ICE = np.array(
    [0.0, 5.0e-6, 1.0e-5, 2.0e-5, 3.0e-5, 4.0e-5, 5.0e-5, 6.0e-5, 1.0e-2])
# Water-path bins (kg m-2).
NUM_MODIS_LWP_BINS = 7
MODIS_HIST_LWP = np.array([0.0, 0.01, 0.03, 0.06, 0.10, 0.15, 0.25, 20.0])
NUM_MODIS_IWP_BINS = 7
MODIS_HIST_IWP = np.array([0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.00, 20.0])


# --- lidar_optics (cosp_optics.F90) constants ---
# The lidar optics use COSP's own physical constants (cosp_constants.F90),
# which differ slightly from the radar-path values above (RD = 287.0); they
# are named LIDAR_* here so each ported path stays bit-faithful to its source.
LIDAR_GRAV = 9.80665      # m s-2 (cosp_constants.F90 'grav')
LIDAR_RD = 287.04         # J kg-1 K-1 (cosp_constants.F90 'rd')
LIDAR_BOLTZMANN = 1.38e-23  # J K-1 (cosp_constants.F90 'km')
LIDAR_RHO_LIQ = 1000.0    # kg m-3 (cosp_constants.F90 'rholiq')
LIDAR_RHO_ICE = 0.5e3     # kg m-3 (lidar_optics local 'rhoice'; NOT 917)
LIDAR_QSCAT = 2.0         # Particle scattering efficiency at 532 nm
LIDAR_MAX_RADIUS = 70.0e-6  # Effective-radius cap (m) in lidar_optics
# Wavelength-dependent molecular scattering / multiple-scattering parameters.
LIDAR_CMOL = {532: 6.2446e-32, 355: 3.2662e-31}
LIDAR_RDIFFM = {532: 0.7, 355: 0.6}
# Polarized-optics polynomials kp(re[um]) = p0*x^4 + p1*x^3 + p2*x^2 + p3*x + p4
# (Mie theory for spheres; ray-tracing/FDTD composite for non-spherical ice).
# COSP uses identical coefficients for the LS and CONV variants of each phase.
LIDAR_POLPART_LIQ = np.array(
    [2.6980e-8, -3.7701e-6, 1.6594e-4, -0.0024, 0.0626])
LIDAR_POLPART_ICE_SPHERE = np.array(
    [-1.0176e-8, 1.7615e-6, -1.0480e-4, 0.0019, 0.0460])
LIDAR_POLPART_ICE_NONSPHERE = np.array(
    [1.3615e-8, -2.04206e-6, 7.51799e-5, 0.00078213, 0.0182131])
