"""Drop-size distributions, effective radius and radar reflectivity (zeff).

Exact NumPy port of COSP v2.0
``subsample_and_optics_example/optics/quickbeam_optics/quickbeam_optics.F90``
(``calc_Re``/``dsd``/``zeff`` by John Haynes and Roj Marchand, Quickbeam
v1.03-v3; COSPv2 adaptation by Dustin Swales), plus ``avint`` from
``math_lib.F90`` (SLATEC, public domain) and ``infind`` from
``array_lib.F90``. Copyright (c) 2015, Regents of the University of
Colorado, BSD-3-Clause; license text preserved in LICENSES/.

This module is OFFLINE-ONLY (lookup-table generation and validation): plain
NumPy, no JAX. Exactness against the Fortran reference is the priority.
"""

import math

import numpy as np

from jcosp.config import effective_k2
from jcosp.constants import RHO_ICE
from jcosp.radar.lut.mie import m_ice, m_wat, mie_int

__all__ = ["avint", "fgamma", "infind", "calc_re", "dsd", "zeff"]

_PI = math.acos(-1.0)


def fgamma(x: float) -> float:
    """Gamma function as used by the Fortran ``math_lib.F90:gamma``.

    ``math.gamma`` agrees with the Fortran series/recurrence implementation
    to ~1e-15 for the positive arguments the DSD code uses, so we use it
    directly. (The Fortran's non-positive-integer branch returns a garbage
    sentinel — ``1._wp+300`` = 301.0, a transcription slip of ``1.0d+300`` —
    but that path is unreachable for physical distribution parameters;
    ``math.gamma`` raises instead.)
    """
    return math.gamma(x)


def infind(arr, val: float) -> int:
    """Index of the element of sorted ``arr`` closest to ``val`` (0-based).

    Exact port of ``array_lib.F90:infind`` (minus the Fortran's 1-based
    indexing): values beyond either end clamp to that end; interior values
    pick the nearer of the two bracketing elements, with ties going to the
    first element >= val (the Fortran comparison is strict ``<``).
    """
    arr = np.asarray(arr)
    n = arr.shape[0]
    if val >= arr[n - 1]:
        return n - 1
    if val <= arr[0]:
        return 0
    # First index with arr[k] >= val (Fortran: minloc of the <val mask).
    k = int(np.searchsorted(arr, val, side="left"))
    if abs(arr[k - 1] - val) < abs(arr[k] - val):
        return k - 1
    return k


def avint(ftab, xtab, a: float, b: float) -> float:
    """Integrate unevenly spaced data by overlapping parabolas (SLATEC AVINT).

    Exact port of ``math_lib.F90:avint`` including its bracketing of the
    integration limits and the running ``syl`` accumulation. ``xtab`` must be
    strictly ascending with at least 3 points.

    Parameters
    ----------
    ftab, xtab : array_like
        Function values and abscissas.
    a, b : float
        Integration limits (b < a integrates with reversed sign, as SLATEC).

    Returns
    -------
    float
        Approximate integral of f from a to b.
    """
    ftab = np.asarray(ftab, dtype=np.float64)
    xtab = np.asarray(xtab, dtype=np.float64)
    ntab = xtab.shape[0]
    if ntab < 3:
        raise ValueError("avint: ntab is less than 3 (Fortran FATAL ERROR)")
    if np.any(np.diff(xtab) <= 0.0):
        raise ValueError(
            "avint: xtab is not strictly increasing (Fortran FATAL ERROR)")
    if a == b:
        # Fortran prints a warning and returns result = 0.
        return 0.0

    # If b < a, temporarily switch a and b, and store sign.
    if b < a:
        a, b = b, a
        ind = -1
    else:
        ind = 1
    syl = a

    # Bracket a and b between xtab(ilo) and xtab(ihi), 1-based as in Fortran.
    # ilo: first 1-based index with a <= xtab(i) (ntab+1 if none).
    ilo = int(np.searchsorted(xtab, a, side="left")) + 1
    ilo = max(2, ilo)
    ilo = min(ilo, ntab - 1)
    # ihi: the Fortran loop exits at the first i with xtab(i) <= b having
    # decremented ihi (i-1) times; for ascending xtab that means ihi = ntab
    # whenever b >= xtab(1) and 0 otherwise.
    ihi = ntab if b >= xtab[0] else 0
    ihi = min(ihi, ntab - 1)
    ihi = max(ilo, ihi - 1)

    # Precompute the per-panel parabola coefficients (independent of the
    # sequential smoothing below), preserving the Fortran evaluation order in
    # the running sum.
    sum1 = 0.0
    ca = cb = cc = 0.0
    for i in range(ilo, ihi + 1):  # 1-based i
        x1 = xtab[i - 2]
        x2 = xtab[i - 1]
        x3 = xtab[i]
        term1 = ftab[i - 2] / ((x1 - x2) * (x1 - x3))
        term2 = ftab[i - 1] / ((x2 - x1) * (x2 - x3))
        term3 = ftab[i] / ((x3 - x1) * (x3 - x2))
        atemp = term1 + term2 + term3
        btemp = -(x2 + x3) * term1 - (x1 + x3) * term2 - (x1 + x2) * term3
        ctemp = x2 * x3 * term1 + x1 * x3 * term2 + x1 * x2 * term3
        if i <= ilo:
            ca, cb, cc = atemp, btemp, ctemp
        else:
            # Smooth: average this panel's parabola with the previous one.
            ca = 0.5 * (atemp + ca)
            cb = 0.5 * (btemp + cb)
            cc = 0.5 * (ctemp + cc)
        sum1 = sum1 + ca * (x2 ** 3 - syl ** 3) / 3.0 \
            + cb * 0.5 * (x2 ** 2 - syl ** 2) + cc * (x2 - syl)
        ca, cb, cc = atemp, btemp, ctemp
        syl = x2

    result = sum1 + ca * (b ** 3 - syl ** 3) / 3.0 \
        + cb * 0.5 * (b ** 2 - syl ** 2) + cc * (b - syl)
    if ind != 1:
        result = -result
    return result


def calc_re(q, np_, rho_a, dtype, apm, bpm, rho_c, p1, p2, p3):
    """Effective radius (um) of a hydrometeor distribution (port of calc_Re).

    Re is half the ratio of the distribution's 3rd to 2nd moment.

    Parameters
    ----------
    q : float
        Hydrometeor mixing ratio (g/kg).
    np_ : float
        Total number concentration (1/kg); 0 selects the class defaults.
    rho_a : float
        Ambient air density (kg m-3).
    dtype : int
        Distribution type (1 modified gamma, 2 exponential, 3 power law,
        4 monodisperse, 5 lognormal).
    apm, bpm : float
        Mass-diameter power law m = apm * D**bpm (kg, D in m); -1 = unset.
    rho_c : float
        Constant particle density (kg m-3); -1 = unset. When set and apm is
        unset, the Fortran substitutes apm = (pi/6) rho_c, bpm = 3 in place —
        the possibly-updated values are returned.
    p1, p2, p3 : float
        Per-distribution default parameters (see quickbeam docs).

    Returns
    -------
    (re, apm, bpm) : tuple of float
        Effective radius (um) and the possibly-updated mass parameters.

    Raises
    ------
    ValueError
        Where the Fortran raises FATAL errors (missing required parameters).
    """
    # If density is constant, set equivalent values for apm and bpm.
    if rho_c > 0 and apm < 0:
        apm = (_PI / 6) * rho_c
        bpm = 3.0

    # Exponential with Np specified is treated as modified gamma with vu=1.
    if dtype == 2 and np_ > 0:
        local_dtype = 1
        local_p3 = 1.0
    else:
        local_dtype = dtype
        local_p3 = p3

    if local_dtype == 1:
        # Modified gamma.
        if abs(local_p3 + 2) < 1e-8:
            if np_ > 1e-30:
                # Morrison scheme with Martin 1994 shape parameter (vu = pc+1);
                # Nt = Np*rho_a in #/cm^3.
                vu = (1 / (0.2714 + 0.00057145 * np_ * rho_a * 1e-6)) ** 2
            else:
                raise ValueError(
                    "calc_re: must specify Np with Morrison/Martin scheme")
        elif abs(local_p3 + 1) > 1e-8:
            vu = local_p3
        else:
            raise ValueError(
                "calc_re: must specify vu for modified gamma distribution")

        # NOTE: the Fortran tests `p2+1 > 1E-8` here (no abs), faithfully kept.
        if np_ == 0 and p2 + 1 > 1e-8:
            dm = p2  # default MEAN diameter (um)
            d0 = fgamma(vu) / fgamma(vu + 1) * dm
        else:
            if np_ == 0:
                if abs(p1 + 1) > 1e-8:
                    local_np = p1  # default number concentration (1/kg)
                else:
                    raise ValueError(
                        "calc_re: must specify Np or default (p1/p2) for "
                        "modified gamma distribution")
            else:
                local_np = np_
            d0 = 1e6 * (q * 1e-3 * fgamma(vu)
                        / (apm * local_np * fgamma(vu + bpm))) ** (1 / bpm)
        re = 0.5 * d0 * fgamma(vu + 3) / fgamma(vu + 2)

    elif local_dtype == 2:
        # Exponential (Np not specified — handled above).
        if abs(p1 + 1) > 1e-8:
            n0 = p1
            tmp1 = 1.0 / (1.0 + bpm)
            ld = ((apm * fgamma(1.0 + bpm) * n0) / (rho_a * q * 1e-3)) ** tmp1
            ld = ld / 1e6  # to um^-1
        elif abs(p2 + 1) > 1e-8:
            ld = p2  # default slope (um^-1)
        else:
            raise ValueError(
                "calc_re: must specify Np or default (p1=N0 or p2=lambda) "
                "for exponential distribution")
        re = 1.5 / ld

    elif local_dtype == 3:
        # Power law: LUT scaling not supported, Re stays 0.
        re = 0.0
        if np_ > 0:
            raise ValueError(
                "calc_re: variable Np not supported for power law")

    elif local_dtype == 4:
        re = p1
        if np_ > 0:
            raise ValueError(
                "calc_re: variable Np not supported for monodisperse")

    elif local_dtype == 5:
        # Lognormal.
        if abs(local_p3 + 1) > 1e-8:
            log_sigma_g = local_p3
        else:
            raise ValueError(
                "calc_re: must specify sigma_g for lognormal distribution")
        if np_ == 0 and abs(p2 + 1) > 1e-8:
            rg = p2  # default geometric mean radius (um)
        else:
            if np_ > 0:
                local_np = np_
            elif abs(p2 + 1) < 1e-8:
                local_np = p1
            else:
                # The Fortran only prints a (non-fatal) error here and then
                # proceeds with local_Np undefined; we raise instead.
                raise ValueError(
                    "calc_re: must specify Np or default (p2=rg or p1=Np) "
                    "for lognormal distribution")
            log_sigma_g = p3
            tmp1 = (q * 1e-3) / (2.0 ** bpm * apm * local_np)
            # Faithful transcription: the Fortran exponent uses log_sigma_g,
            # NOT log_sigma_g**2 (kept bit-for-bit even if suspicious).
            tmp2 = math.exp(0.5 * bpm * bpm * log_sigma_g) \
                * math.exp(0.5 * bpm * bpm * log_sigma_g)
            rg = ((tmp1 / tmp2) ** (1.0 / bpm)) * 1e6
        re = rg * math.exp(2.5 * (log_sigma_g * log_sigma_g))
    else:
        raise ValueError(f"calc_re: unknown distribution type {dtype}")

    return re, apm, bpm


def dsd(q, re, np_, d_grid, dtype, rho_a, tk, dmin, dmax,
        apm, bpm, rho_c, p1, p2, p3):
    """Discrete drop-size distribution on ``d_grid`` (port of ``dsd``).

    Parameters
    ----------
    q : float
        Hydrometeor mixing ratio (g/kg).
    re : float
        Effective radius (um); 0 = derive from ``np_`` or class defaults.
    np_ : float
        Total number concentration (1/kg); used only when ``re`` == 0.
    d_grid : array_like
        Discrete particle sizes D (um) at which n(D) is evaluated.
    dtype : int
        Distribution type (see ``calc_re``).
    rho_a, tk : float
        Air density (kg m-3) and temperature (K).
    dmin, dmax : float
        Size cutoffs (um), used only by the power-law type.
    apm, bpm, rho_c, p1, p2, p3 : float
        As in ``calc_re``.

    Returns
    -------
    (n, apm, bpm, re) : tuple
        ``n``: concentrations (cm-3 um-1; for monodisperse, element 0 holds a
        constant in cm-3), zero outside filled ranges as in the Fortran
        caller which zero-initialises. ``apm``/``bpm``/``re`` echo the
        possibly-updated in/out arguments.
    """
    d = np.asarray(d_grid, dtype=np.float64)
    n = np.zeros_like(d)
    tc = tk - 273.15

    if rho_c > 0 and apm < 0:
        apm = (_PI / 6) * rho_c
        bpm = 3.0

    # Prefer Re over Np: with only Np given, derive Re first.
    if re == 0 and np_ > 0:
        re, apm, bpm = calc_re(q, np_, rho_a, dtype, apm, bpm, rho_c,
                               p1, p2, p3)

    if dtype == 1:
        # Modified gamma.
        if abs(p3 + 2) < 1e-8:
            if np_ > 1e-30:
                vu = (1 / (0.2714 + 0.00057145 * np_ * rho_a * 1e-6)) ** 2.0
            else:
                raise ValueError(
                    "dsd: must specify Np with Morrison/Martin scheme")
        elif abs(p3 + 1) > 1e-8:
            vu = p3
        else:
            raise ValueError(
                "dsd: must specify vu for modified gamma distribution")

        if re > 0:
            d0 = 2.0 * re * fgamma(vu + 2) / fgamma(vu + 3)
            fc = (((d * 1e-6) ** (vu - 1) * np.exp(-1 * d / d0))
                  / (apm * ((d0 * 1e-6) ** (vu + bpm)) * fgamma(vu + bpm))) \
                * 1e-12
            n = fc * rho_a * (q * 1e-3)
        elif p2 + 1 > 1e-8:  # default MEAN diameter (Fortran: no abs here)
            dm = p2
            d0 = fgamma(vu) / fgamma(vu + 1) * dm
            fc = (((d * 1e-6) ** (vu - 1) * np.exp(-1 * d / d0))
                  / (apm * ((d0 * 1e-6) ** (vu + bpm)) * fgamma(vu + bpm))) \
                * 1e-12
            n = fc * rho_a * (q * 1e-3)
        elif abs(p3 + 1) > 1e-8:  # default number concentration
            local_np = p1
            tmp1 = (q * 1e-3) ** (1.0 / bpm)
            fc = (d * 1e-6
                  / (fgamma(vu) / (apm * local_np * fgamma(vu + bpm)))
                  ** (1.0 / bpm)) ** vu
            n = ((rho_a * local_np * fc * (d * 1e-6) ** (-1.0))
                 / (fgamma(vu) * tmp1 ** vu)
                 * np.exp(-1.0 * fc ** (1.0 / vu) / tmp1)) * 1e-12
        else:
            raise ValueError("dsd: no default value for Dm or Np provided")

    elif dtype == 2:
        # Exponential.
        if re > 0:
            ld = 1.5 / re  # slope, 1/um
            fc = (ld * 1e6) ** (1.0 + bpm) / (apm * fgamma(1.0 + bpm)) \
                * np.exp(-1.0 * (ld * 1e6) * (d * 1e-6)) * 1e-12
            n = fc * rho_a * (q * 1e-3)
        elif abs(p1 + 1) > 1e-8:
            # Default intercept N0 (m^-4).
            n0 = p1
            tmp1 = 1.0 / (1.0 + bpm)
            fc = ((apm * fgamma(1.0 + bpm) * n0) ** tmp1) * (d * 1e-6)
            n = (n0 * np.exp(-1.0 * fc * (1.0 / (rho_a * q * 1e-3)) ** tmp1)) \
                * 1e-12
        elif abs(p2 + 1) > 1e-8:
            # Default slope lambda (um^-1).
            ld = p2
            fc = (ld * 1e6) ** (1.0 + bpm) / (apm * fgamma(1.0 + bpm)) \
                * np.exp(-1.0 * (ld * 1e6) * (d * 1e-6)) * 1e-12
            n = fc * rho_a * (q * 1e-3)
        else:
            # lambda parameterized from temperature (original Quickbeam).
            ld = 1220.0 * 10.0 ** (-0.0245 * tc) * 1e-6
            n0 = ((ld * 1e6) ** (1.0 + bpm) * q * 1e-3 * rho_a) \
                / (apm * fgamma(1.0 + bpm))
            n = (n0 * np.exp(-ld * d)) * 1e-12

    elif dtype == 3:
        # Power law.
        if re > 0:
            raise ValueError("dsd: variable Re not supported for power law")
        if np_ > 0:
            raise ValueError("dsd: variable Np not supported for power law")
        # br parameter: Ryan (2000) temperature parameterizations for
        # p1 = -2 (cirrus) and p1 = -3 (frontal); otherwise p1 itself.
        if abs(p1 + 2) < 1e-8:
            if tc < -30:
                bhp = -1.75 + 0.09 * ((tc + 273.0) - 243.16)
            elif -30 <= tc < -9:
                bhp = -3.25 - 0.06 * ((tc + 273.0) - 265.66)
            else:
                bhp = -2.15
        elif abs(p1 + 3) < 1e-8:
            if tc < -35:
                bhp = -1.75 + 0.09 * ((tc + 273.0) - 243.16)
            elif -35 <= tc < -17.5:
                bhp = -2.65 + 0.09 * ((tc + 273.0) - 255.66)
            elif -17.5 <= tc < -9:
                bhp = -3.25 - 0.06 * ((tc + 273.0) - 265.66)
            else:
                bhp = -2.15
        else:
            bhp = p1
        # Ar parameter from mass closure over [dmin, dmax].
        dmin_mm = dmin * 1e-3
        dmax_mm = dmax * 1e-3
        tmp1 = rho_a * (q * 1e-3) * (bhp + bpm + 1)
        tmp2 = apm * (dmax_mm ** bhp * dmax ** (bpm + 1)
                      - dmin_mm ** bhp * dmin ** (bpm + 1))
        ahp = tmp1 / tmp2 * 1e24
        lidx = infind(d, dmin)
        uidx = infind(d, dmax)
        n[lidx:uidx + 1] = (ahp * (d[lidx:uidx + 1] * 1e-3) ** bhp) * 1e-12

    elif dtype == 4:
        # Monodisperse: a single size with constant concentration (cm^-3).
        d0 = re if re > 0 else p1
        rho_e = (6.0 / _PI) * apm * (d0 * 1e-6) ** (bpm - 3)
        fc0 = (6.0 / (_PI * d0 * d0 * d0 * rho_e)) * 1e12
        n[0] = fc0 * rho_a * (q * 1e-3)

    elif dtype == 5:
        # Lognormal.
        if abs(p1 + 1) < 1e-8 or re > 0:
            # rg (or Re) and log_sigma_g given.
            log_sigma_g = p3
            tmp2 = (bpm * log_sigma_g) * (bpm * log_sigma_g)
            if re <= 0:
                rg = p2
            else:
                rg = re * math.exp(-2.5 * (log_sigma_g ** 2))
            fc = 0.5 * ((1.0 / ((2.0 * rg * 1e-6) ** bpm * apm
                                * (2.0 * _PI) ** 0.5
                                * log_sigma_g * d * 0.5 * 1e-6))
                        * np.exp(-0.5 * ((np.log(0.5 * d / rg)
                                          / log_sigma_g) ** 2.0 + tmp2))) \
                * 1e-12
            n = fc * rho_a * (q * 1e-3)
        elif abs(p2 + 1) < 1e-8 or np_ > 0:
            # Np and log_sigma_g given.
            local_np = np_ if np_ > 0 else p1
            log_sigma_g = p3
            n0 = local_np * rho_a
            tmp1 = (rho_a * (q * 1e-3)) / (2.0 ** bpm * apm * n0)
            # Faithful: exponent uses log_sigma_g, not its square (see
            # calc_re note).
            tmp2 = math.exp(0.5 * bpm * bpm * log_sigma_g) \
                * math.exp(0.5 * bpm * bpm * log_sigma_g)
            rg = ((tmp1 / tmp2) ** (1 / bpm)) * 1e6
            n = 0.5 * (n0 / ((2.0 * _PI) ** 0.5 * log_sigma_g * d * 0.5
                             * 1e-6)
                       * np.exp(-0.5 * (np.log(0.5 * d / rg)
                                        / log_sigma_g) ** 2.0)) * 1e-12
        else:
            raise ValueError("dsd: must specify a value for sigma_g")
    else:
        raise ValueError(f"dsd: unknown distribution type {dtype}")

    return n, apm, bpm, re


# Fortran zeff computes `cr = 10./log(10.)` with default (single precision)
# real literals; the result is promoted to double at assignment. The COSP
# KGO notes this deliberate single-precision quirk (~1e-8 relative vs the
# double value), so we reproduce it with float32 arithmetic.
_CR_SINGLE = float(np.float32(10.0) / np.log(np.float32(10.0)))


def zeff(freq, d_eq, n, k2, tt, ice, xr, rho_e, qe=None, qs=None):
    """Radar reflectivity of a discrete DSD of spheres (port of ``zeff``).

    Parameters
    ----------
    freq : float
        Radar frequency (GHz).
    d_eq : array_like
        Discrete (equivalent-volume) drop sizes (um).
    n : array_like
        Discrete concentrations (cm-3 um-1).
    k2 : float
        |K|^2 dielectric factor; -1 selects the frequency-dependent default
        (resolved via ``jcosp.config.effective_k2``, same values as the
        Fortran's in-place update).
    tt : float
        Hydrometeor temperature (K).
    ice : int
        1 if the volume is ice, else 0.
    xr : int
        1 to also compute the Rayleigh-only reflectivity ``z_ray``.
    rho_e : array_like
        Effective particle densities (kg m-3), -1 for pure spheres. Ice with
        min(rho_e) >= 0 triggers the Maxwell-Garnett-style refractive-index
        correction (identity when rho_e == RHO_ICE).
    qe, qs : array_like, optional
        Precomputed extinction/backscatter efficiencies on ``d_eq``. When
        given, the per-size Mie computation is skipped — this mirrors the
        Fortran's Mie-table path (sentinel qe(1) > -9) and is what the LUT
        generator uses to dedupe Mie calls across Re bins.

    Returns
    -------
    (z_eff, z_ray, kr) : tuple of float
        Effective and Rayleigh reflectivity factors (mm^6 m-3) and the
        attenuation coefficient (dB km-1).
    """
    d_eq = np.asarray(d_eq, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    rho_e = np.asarray(rho_e, dtype=np.float64)
    nsizes = d_eq.shape[0]

    # Unit conversions: sizes to m, concentrations to m^-3 m^-1, radar
    # wavelength to m.
    d0 = d_eq * 1e-6
    n0 = n * 1e12
    wl = 0.299792458 / freq
    k2 = effective_k2(freq, k2)

    if qe is None:
        # Bulk refractive index; zeff's convention is m = n_r - i*n_i.
        if ice == 0:
            n_r, n_i = m_wat(freq, tt)
        else:
            n_r, n_i = m_ice(freq, tt)
        m = complex(n_r, -n_i)
        m0 = np.full(nsizes, m, dtype=np.complex128)
        # Maxwell-Garnett-style density correction for ice; with
        # rho_e = RHO_ICE the fraction f = 1 and m0 reduces to m exactly.
        if ice == 1 and np.min(rho_e) >= 0:
            f = rho_e / RHO_ICE
            m0 = np.sqrt((2 + (m0 * m0) + 2 * f * ((m0 * m0) - 1))
                         / (2 + (m0 * m0) + f * (1 - (m0 * m0))))
        sizep = (_PI * d0) / wl
        qext = np.empty(nsizes)
        qbsca = np.empty(nsizes)
        for i in range(nsizes):
            qext[i], _, qbsca[i] = mie_int(float(sizep[i]), complex(m0[i]))
    else:
        qext = np.asarray(qe, dtype=np.float64)
        qbsca = np.asarray(qs, dtype=np.float64)

    # eta_mie = 0.25*pi*sum[qbsca*D^2*N(D) dD];  z0_eff = wl^4/pi^5/k2 * eta_mie
    if nsizes == 1:
        # Single size (monodisperse): no integration; note the Fortran uses
        # n (cm^-3) * 1e6 here, not the n0 conversion.
        eta_sum = qbsca[0] * (n[0] * 1e6) * d0[0] * d0[0]
    else:
        eta_sum = avint(qbsca * n0 * d0 * d0, d0, d0[0], d0[-1])
    eta_mie = eta_sum * 0.25 * _PI
    const = ((wl * wl * wl * wl) / (_PI ** 5)) * (1.0 / k2)
    z0_eff = const * eta_mie

    # kr = 0.25*pi*(1000*cr)*sum[qext*D^2*N(D) dD]
    if nsizes == 1:
        k_sum = qext[0] * (n[0] * 1e6) * d0[0] * d0[0]
    else:
        k_sum = avint(qext * n0 * d0 * d0, d0, d0[0], d0[-1])
    kr = k_sum * 0.25 * _PI * (1000.0 * _CR_SINGLE)

    # z_ray = sum[D^6*N(D) dD] (only when requested).
    z0_ray = 0.0
    if xr == 1:
        if nsizes == 1:
            z0_ray = (n[0] * 1e6) * d0[0] ** 6
        else:
            z0_ray = avint(n0 * d0 ** 6, d0, d0[0], d0[-1])

    # Convert m^6/m^3 -> mm^6/m^3.
    return z0_eff * 1e18, z0_ray * 1e18, kr
