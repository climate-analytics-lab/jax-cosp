"""Runtime radar optics: hydrometeors -> reflectivity and attenuation, in JAX.

Ported from COSP v2.0 ``quickbeam_optics.F90`` (QuickBeam by John Haynes and
Roj Marchand; Copyright (c) 2015, Regents of the University of Colorado,
BSD-3-Clause; see LICENSES/).

The Fortran computes, for every volume containing hydrometeors, an effective
reflectivity ``ze``, Rayleigh reflectivity ``zr`` and attenuation ``kr`` per
hydrometeor class, using a Mie/DSD integration whose result it caches in a
"Z-scale" lookup table because the result is exactly linear in
``rho_air * mixing_ratio`` for fixed (class, temperature bin, quantized Re):

    ze = Ze_scaled(class, itt, iRe) * rho_a * Q

Here that table is **precomputed in full offline** (:mod:`jcosp.radar.lut`)
and the runtime is the pure lookup + multiply, which is differentiable in the
mixing ratios (and piecewise-constant in Re and temperature, exactly like the
Fortran's quantization).

Documented deviations from the Fortran (see docs/translation_notes.md):

* Slots the Fortran always recomputes ab initio (effective radii quantizing to
  the first Re slot, or beyond the last) use the table's nearest slot instead.
* ``calc_Re`` inputs are clamped away from zero where the Fortran would
  produce inf/NaN for hydrometeor-free volumes; those volumes are masked out
  of the result regardless, and the clamp keeps reverse-mode gradients clean
  (0*inf poisoning through ``where``).
"""

import math
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln

from jcosp import config
from jcosp.constants import RD


def _gamma(x):
    """Gamma function via gammaln (arguments here are always positive)."""
    return jnp.exp(gammaln(x))


class ZScaleLUT(NamedTuple):
    """Precomputed Z-scale tables (see jcosp.radar.lut.generate)."""

    ze_scaled: jnp.ndarray  # (N_HYDRO, MT_NTT, N_RE_TYPES) mm^6/m^3 per (kg/m^3 * g/kg)
    zr_scaled: jnp.ndarray  # same, Rayleigh-only
    kr_scaled: jnp.ndarray  # same, dB/km
    freq: float
    k2: float
    micro_scheme: str


def load_lut(path) -> ZScaleLUT:
    """Load a Z-scale LUT written by ``python -m jcosp.radar.lut.generate``."""
    with np.load(path) as f:
        # Unreachable slots are NaN in the file; zero them so masked-out
        # lookups can never poison gradients or sums.
        return ZScaleLUT(
            ze_scaled=jnp.asarray(np.nan_to_num(f["ze_scaled"], nan=0.0)),
            zr_scaled=jnp.asarray(np.nan_to_num(f["zr_scaled"], nan=0.0)),
            kr_scaled=jnp.asarray(np.nan_to_num(f["kr_scaled"], nan=0.0)),
            freq=float(f["freq"]),
            k2=float(f["k2"]),
            micro_scheme=str(f["micro_scheme"]) if "micro_scheme" in f else "1M",
        )


def temperature_index(t_kelvin, phase: int):
    """Nearest LUT temperature-bin index (array_lib.F90:infind on a 5-K grid).

    infind returns the nearest grid point, clamped to the ends, breaking exact
    ties toward the upper neighbour — for a regular grid that is floor(x+0.5).
    """
    grid = config.MT_TTI if phase == 1 else config.MT_TTL
    idx = jnp.floor((t_kelvin - grid[0]) / 5.0 + 0.5).astype(jnp.int32)
    return jnp.clip(idx, 0, len(grid) - 1)


def quantize_re(re_um):
    """Map effective radius (um) to its Z-scale LUT slot (0-based).

    Port of the index arithmetic in quickbeam_optics.F90: coarse interval
    ``n = floor(Re/10)`` selects a resolution ``step`` and running offset
    ``base`` (config.RE_STEP_LIST/RE_BASE_LIST); within the interval the slot
    is ``floor(Re/step)`` (floored at 1, so Re=0 lands in slot 0).

    Returns
    -------
    index : int32 array
        0-based LUT slot, clipped to the table size.
    re_quant : array
        The quantized Re actually represented by that slot, ``step*(iRe+0.5)``.
    """
    step_list = jnp.asarray(config.RE_STEP_LIST)
    base_list = jnp.asarray(config.RE_BASE_LIST)
    n = jnp.clip((re_um / config.RE_BIN_LENGTH).astype(jnp.int32),
                 0, config.RE_MAX_BIN - 1)
    step = step_list[n]
    base = base_list[n]
    i_re = jnp.maximum((re_um / step).astype(jnp.int32), 1)
    re_quant = step * (i_re + 0.5)
    # Fortran 1-based: iRe_type = iRe + base - int(n*RE_BIN_LENGTH/step),
    # capped at nRe_types; converted to 0-based here.
    idx = i_re + base.astype(jnp.int32) - (
        n * config.RE_BIN_LENGTH / step).astype(jnp.int32)
    idx = jnp.clip(idx, 1, config.N_RE_TYPES) - 1
    return idx, re_quant


def calc_re_class(q, np_, rho_a, *, dtype, apm, bpm, rho, p1, p2, p3):
    """Effective radius (um) from mixing ratio / number for one class.

    Port of quickbeam_optics.F90:calc_Re with the class parameters static
    (Python) and q (g/kg), np_ (1/kg), rho_a (kg/m3) traced. Branches on the
    static parameters are resolved at trace time; only the formulas involving
    q/np_/rho_a stay in the graph.
    """
    if rho > 0 and apm < 0:
        apm = (np.pi / 6.0) * rho
        bpm = 3.0

    # Guard against hydrometeor-free volumes: results there are masked out by
    # the caller, the clamps only keep inf/NaN out of the gradient graph.
    q = jnp.maximum(q, 1e-12)

    if dtype == 2 and np_ is not None:
        # Exponential with Np supplied is treated as modified gamma with vu=1.
        local_dtype, local_p3 = 1, 1.0
    else:
        local_dtype, local_p3 = dtype, p3

    if local_dtype == 1:  # Modified gamma
        if abs(local_p3 + 2) < 1e-8:
            if np_ is None:
                raise ValueError(
                    "Morrison/Martin modified-gamma PSD (p3=-2) requires Np")
            nt_cm3 = jnp.maximum(np_, 1e-6) * rho_a * 1e-6
            vu = (1.0 / (0.2714 + 0.00057145 * nt_cm3)) ** 2
        elif abs(local_p3 + 1) > 1e-8:
            vu = local_p3
        else:
            raise ValueError("Modified gamma PSD requires vu (p3)")

        if np_ is None and p2 + 1 > 1e-8:
            # All-static branch (e.g. the 1M ice classes): constant Re.
            d0 = math.gamma(vu) / math.gamma(vu + 1) * p2
            re = 0.5 * d0 * math.gamma(vu + 3) / math.gamma(vu + 2)
            return jnp.full_like(q, re)
        if np_ is None:
            if abs(p1 + 1) > 1e-8:
                local_np = p1
            else:
                raise ValueError("Modified gamma PSD requires Np, Dm or "
                                 "default number concentration")
        else:
            local_np = jnp.maximum(np_, 1e-6)
        d0 = 1e6 * (q * 1e-3 * _gamma(vu)
                    / (apm * local_np * _gamma(vu + bpm))) ** (1.0 / bpm)
        return 0.5 * d0 * _gamma(vu + 3) / _gamma(vu + 2)

    if local_dtype == 2:  # Exponential
        if abs(p1 + 1) > 1e-8:
            n0 = p1
            ld = ((apm * math.gamma(1.0 + bpm) * n0)
                  / (rho_a * q * 1e-3)) ** (1.0 / (1.0 + bpm)) / 1e6
            return 1.5 / ld
        if abs(p2 + 1) > 1e-8:
            return jnp.full_like(q, 1.5 / p2)
        raise ValueError("Exponential PSD requires N0 (p1) or lambda (p2)")

    if local_dtype == 3:  # Power law: LUT approach unsupported (as in Fortran)
        return jnp.zeros_like(q)

    if local_dtype == 4:  # Monodisperse
        return jnp.full_like(q, p1)

    if local_dtype == 5:  # Lognormal
        if abs(p3 + 1) < 1e-8:
            raise ValueError("Lognormal PSD requires sigma_g (p3)")
        log_sigma_g = p3
        if np_ is None and abs(p2 + 1) > 1e-8:
            # All-static branch (the 1M liquid-cloud classes): constant Re.
            return jnp.full_like(q, p2 * math.exp(2.5 * log_sigma_g**2))
        local_np = p1 if np_ is None else jnp.maximum(np_, 1e-6)
        tmp1 = (q * 1e-3) / (2.0**bpm * apm * local_np)
        tmp2 = math.exp(0.5 * bpm * bpm * log_sigma_g) ** 2
        rg = ((tmp1 / tmp2) ** (1.0 / bpm)) * 1e6
        return rg * math.exp(2.5 * log_sigma_g**2)

    raise ValueError(f"Unknown PSD type {dtype}")


def quickbeam_optics(lut: ZScaleLUT, sd: config.SizeDistribution,
                     hm_matrix, re_matrix, np_matrix, pressure, temperature):
    """Effective reflectivity and hydrometeor attenuation of every volume.

    Parameters
    ----------
    lut : ZScaleLUT
        Precomputed Z-scale tables for the same microphysics scheme as ``sd``.
    sd : config.SizeDistribution
        Static per-class PSD parameters (config.HYDRO_CLASSES[...]).
    hm_matrix : (nlev, ..., N_HYDRO)
        Hydrometeor mixing ratios (g/kg), TOA-first.
    re_matrix : (nlev, ..., N_HYDRO) or None
        Effective radii (um); 0 means "derive from the PSD defaults/Np".
    np_matrix : (nlev, ..., N_HYDRO) or None
        Number concentrations (1/kg); only used where re_matrix is 0.
    pressure, temperature : (nlev, ...)
        Pa and K.

    Returns
    -------
    z_vol, z_ray, kr_vol : (nlev, ...)
        Effective and Rayleigh reflectivity factors (mm^6/m^3) and hydrometeor
        attenuation (dB/km), summed over hydrometeor classes.
    """
    rho_a = pressure / (RD * temperature)
    z_vol = jnp.zeros_like(pressure)
    z_ray = jnp.zeros_like(pressure)
    kr_vol = jnp.zeros_like(pressure)

    for tp in range(int(sd.dtype.shape[0])):
        dtype = int(sd.dtype[tp])
        if dtype <= 0:
            continue
        phase = int(sd.phase[tp])
        q = hm_matrix[..., tp]

        np_tp = None if np_matrix is None else np_matrix[..., tp]
        re_default = calc_re_class(
            q, np_tp, rho_a, dtype=dtype, apm=float(sd.apm[tp]),
            bpm=float(sd.bpm[tp]), rho=float(sd.rho[tp]), p1=float(sd.p1[tp]),
            p2=float(sd.p2[tp]), p3=float(sd.p3[tp]))
        if re_matrix is None:
            re = re_default
        else:
            re_in = re_matrix[..., tp]
            re = jnp.where(re_in > 0, re_in, re_default)

        itt = temperature_index(temperature, phase)
        i_re, _ = quantize_re(re)

        scale = rho_a * q
        mask = q > 1e-12
        z_vol = z_vol + jnp.where(mask, lut.ze_scaled[tp, itt, i_re] * scale, 0.0)
        z_ray = z_ray + jnp.where(mask, lut.zr_scaled[tp, itt, i_re] * scale, 0.0)
        kr_vol = kr_vol + jnp.where(mask, lut.kr_scaled[tp, itt, i_re] * scale, 0.0)

    return z_vol, z_ray, kr_vol
