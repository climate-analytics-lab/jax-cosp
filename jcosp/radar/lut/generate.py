"""Offline generator for the Quickbeam Z-scale lookup tables.

Reproduces, per LUT slot, exactly what COSP v2.0
``quickbeam_optics.F90:quickbeam_optics`` computes on its
"compute from distribution parameters" branch (the ``Z_scale_flag`` false
path) and then caches as ``Ze_scaled``/``Zr_scaled``/``kr_scaled``:
Ze/(rho_a*Q) etc. for one (hydrometeor class, temperature bin, quantized
effective radius). The Re-bin quantization mirrors
``cosp_cloudsat_interface.F90:COSP_CLOUDSAT_INIT`` (base_list/step_list).
Copyright (c) 2015, Regents of the University of Colorado, BSD-3-Clause;
license text preserved in LICENSES/.

Because every reflectivity path with Re > 0 produces N(D) proportional to
rho_a*Q (see ``dsd``), the table is generated directly with the reference
Q = 1 g/kg and rho_a = 1 kg m-3, making the ``zeff`` outputs the scaled
values themselves (scale_factor == 1). Linearity is asserted in the tests.

Performance note: for a fixed (class, temperature) the Mie efficiencies on
the fixed Deq grid do not depend on Re — only the DSD weights do. They are
therefore computed once per (class, T) and passed to ``zeff`` via its
qe/qs Mie-table path, exactly mirroring the Fortran's table mechanism.

This module is OFFLINE-ONLY: plain NumPy, no JAX.
"""

import math
import time
from pathlib import Path

import numpy as np

from jcosp.config import (
    D_GRID,
    HYDRO_CLASSES,
    MT_NTT,
    MT_TTI,
    MT_TTL,
    N_HYDRO,
    N_RE_TYPES,
    RE_BASE_LIST,
    RE_BIN_LENGTH,
    RE_MAX_BIN,
    RE_STEP_LIST,
    SizeDistribution,
    effective_k2,
)
from jcosp.constants import RHO_ICE
from jcosp.radar.lut.dsd import dsd, zeff
from jcosp.radar.lut.mie import m_ice, m_wat, mie_efficiencies

__all__ = [
    "re_quantize",
    "enumerate_re_bins",
    "hydro_optics_entry",
    "generate_lut",
    "main",
]

_PI = math.acos(-1.0)
_LUT_PATH = Path("/data/dwatsonparris/jax-cosp/jcosp/data/quickbeam_lut_94GHz_1M.npz")


def re_quantize(re_um: float) -> tuple[int, float]:
    """Map an effective radius to its Z-scale LUT slot (quickbeam_optics).

    Exact port of the Re -> (iRe_type, quantized Re) logic in subroutine
    ``quickbeam_optics``. All integer conversions are Fortran truncations.

    Parameters
    ----------
    re_um : float
        Effective radius (um).

    Returns
    -------
    (ire_type, re_quant) : tuple of (int, float)
        The **1-based** Fortran LUT index (kept 1-based so the arithmetic
        matches the reference; subtract 1 only when indexing arrays) and the
        quantized Re (um) that slot represents. ``ire_type`` == 1 is the
        "no scaling" slot (Re == 0 or tiny); ``ire_type`` == N_RE_TYPES is
        the overflow slot the Fortran flags as never-scalable.

    Notes
    -----
    The mapping is monotone non-decreasing in Re except at exact
    coarse-interval boundaries (Re = 90, 120, 150, ... um), where the index
    steps DOWN by exactly one slot: interval n's truncated slot count can
    overrun into interval n+1's first slot. This is faithful to the Fortran
    arithmetic and merely means two nearby quantized Re values share a slot
    (see ``enumerate_re_bins``).
    """
    ire_type = 1
    re_q = re_um
    if re_um > 0:
        # Coarse interval of width RE_BIN_LENGTH (integer truncation).
        nbin = int(re_um / RE_BIN_LENGTH)
        if nbin >= RE_MAX_BIN:
            nbin = RE_MAX_BIN - 1
        step = RE_STEP_LIST[nbin]
        base = RE_BASE_LIST[nbin]
        ire = int(re_um / step)
        if ire < 1:
            ire = 1
        re_q = step * (ire + 0.5)  # closest Re value allowed in the LUT
        # Offset within the coarse interval; base is real in the Fortran but
        # always holds an integral value.
        ire_type = int(ire + base - int(nbin * RE_BIN_LENGTH / step))
        if ire_type >= N_RE_TYPES:
            ire_type = N_RE_TYPES
    return ire_type, re_q


def enumerate_re_bins(re_scan=None) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate every reachable, usable LUT slot as (index, re_quant) arrays.

    Scans a fine Re grid through ``re_quantize`` and collects the unique
    slot -> quantized-Re mapping. The overflow slot (index == N_RE_TYPES,
    which the Fortran permanently flags as non-scalable) is excluded.

    Edge case (inherited from the Fortran): the last slot of coarse interval
    n and the first slot of interval n+1 can be the same LUT index while
    quantizing Re to slightly different values (they differ by at most about
    half of each interval's step, < ~5% relative). The Fortran runtime
    caches whichever a profile hits first; here the *larger*-Re
    representative (belonging to interval n+1, whose first slot it is) is
    kept deterministically. That choice makes the common default radii that
    sit exactly on a coarse boundary (e.g. the 1M ice default Re = 40 um)
    match the reference bit-for-bit, and leaves the narrow lower-side sliver
    with the same <~1% indeterminacy the Fortran itself has. The
    discrepancy bound is asserted.

    Returns
    -------
    (indices, re_quants) : tuple of ndarray
        1-based slot indices (ascending) and the quantized Re (um) each
        slot represents.
    """
    if re_scan is None:
        # The largest usable slot (N_RE_TYPES-1 = 549) is reached at
        # Re ~ 3250-3260 um; beyond that the mapping caps at the overflow
        # slot. Scan past that so every usable slot is enumerated.
        re_scan = np.arange(0.05, 3300.0, 0.05)
    mapping: dict[int, float] = {}
    for re in re_scan:
        idx, re_q = re_quantize(float(re))
        if idx >= N_RE_TYPES:
            continue  # overflow slot: never usable for scaling
        if idx in mapping:
            # Consistency: collisions only occur across coarse-interval
            # boundaries and must agree within the boundary-rounding bound
            # (~half a step from each side, < ~6% relative; see docstring).
            if not np.isclose(mapping[idx], re_q, rtol=6e-2):
                raise AssertionError(
                    f"LUT slot {idx} maps to inconsistent Re "
                    f"{mapping[idx]} vs {re_q}")
            mapping[idx] = max(mapping[idx], re_q)  # keep upper-interval Re
        else:
            mapping[idx] = re_q
    indices = np.array(sorted(mapping), dtype=np.int64)
    re_quants = np.array([mapping[i] for i in indices])
    return indices, re_quants


def _class_deq_rhoe(sd: SizeDistribution, tp: int):
    """Equivalent-volume sphere sizes Deq (um) and effective densities.

    Mirrors the "Calculate particle density" block of ``quickbeam_optics``:
    ice particles are mapped to solid-ice (RHO_ICE) equivalent-volume
    spheres, liquid drops are used as-is. The apm/bpm used here include the
    same rho->apm substitution ``dsd`` applies internally.
    """
    phase = int(sd.phase[tp])
    apm, bpm, rho_c = float(sd.apm[tp]), float(sd.bpm[tp]), float(sd.rho[tp])
    if rho_c > 0 and apm < 0:
        apm = (_PI / 6) * rho_c
        bpm = 3.0
    if int(sd.dtype[tp]) == 4:
        di = np.array([float(sd.p1[tp])])
    else:
        di = D_GRID
    if phase == 1:
        if float(sd.rho[tp]) < 0:
            deq = ((6 / _PI * apm / RHO_ICE) ** (1.0 / 3.0)) \
                * ((di * 1e-6) ** (bpm / 3.0)) * 1e6
        else:
            deq = di * ((float(sd.rho[tp]) / RHO_ICE) ** (1.0 / 3.0))
        rho_e = np.full(di.shape, RHO_ICE)
    else:
        deq = di.copy()
        # Liquid: rho_e is never consulted by zeff (ice == 0); keep the
        # "pure sphere" sentinel.
        rho_e = np.full(di.shape, -1.0)
    return deq, rho_e


def hydro_optics_entry(sd: SizeDistribution, tp: int, t_kelvin: float,
                       re_um: float, freq: float, k2: float,
                       q_gkg: float = 1.0, rho_a: float = 1.0, xr: int = 1):
    """One LUT entry: scaled (Ze, Zr, kr) for a class/temperature/Re triple.

    Reproduces the full "compute from distribution parameters" branch of
    subroutine ``quickbeam_optics`` for hydrometeor class ``tp`` at
    temperature ``t_kelvin`` and (quantized) effective radius ``re_um``.
    With the reference ``q_gkg`` = 1 g/kg and ``rho_a`` = 1 kg m-3, the raw
    ``zeff`` outputs equal the Fortran's cached ``Ze_scaled`` etc.
    (``ze / (rho_a * hm)``), because every Re > 0 DSD branch is linear in
    rho_a*Q. Note the Fortran only caches ``Zr_scaled`` when the run has
    ``do_ray`` on; we always integrate the Rayleigh term (``xr`` = 1) so the
    offline table is complete — Ze and kr are unaffected.

    Returns
    -------
    (ze_scaled, zr_scaled, kr_scaled) : tuple of float
        mm^6 m-3 per (kg m-3 * g/kg) for Ze/Zr; dB km-1 per the same for kr.
    """
    dtype = int(sd.dtype[tp])
    phase = int(sd.phase[tp])
    di = np.array([float(sd.p1[tp])]) if dtype == 4 else D_GRID
    ni, _, _, _ = dsd(q_gkg, re_um, 0.0, di, dtype, rho_a, t_kelvin,
                      float(sd.dmin[tp]), float(sd.dmax[tp]),
                      float(sd.apm[tp]), float(sd.bpm[tp]),
                      float(sd.rho[tp]), float(sd.p1[tp]),
                      float(sd.p2[tp]), float(sd.p3[tp]))
    deq, rho_e = _class_deq_rhoe(sd, tp)
    ze, zr, kr = zeff(freq, deq, ni, k2, t_kelvin, phase, xr, rho_e)
    scale = rho_a * q_gkg
    return ze / scale, zr / scale, kr / scale


def _mie_for_class_temperature(deq, rho_e, phase: int, t_kelvin: float,
                               freq: float):
    """Precompute (qext, qbsca) on the class Deq grid for one temperature.

    Replicates zeff's refractive-index block (including the ice density
    correction, an exact identity for rho_e == RHO_ICE) so the result can be
    fed back through zeff's qe/qs Mie-table path. This is the Re-dedupe: the
    efficiencies depend on (class, T) only, never on Re.
    """
    if phase == 0:
        n_r, n_i = m_wat(freq, t_kelvin)
    else:
        n_r, n_i = m_ice(freq, t_kelvin)
    m = complex(n_r, -n_i)
    m0 = np.full(deq.shape, m, dtype=np.complex128)
    if phase == 1 and np.min(rho_e) >= 0:
        f = rho_e / RHO_ICE
        m0 = np.sqrt((2 + (m0 * m0) + 2 * f * ((m0 * m0) - 1))
                     / (2 + (m0 * m0) + f * (1 - (m0 * m0))))
    wl = 0.299792458 / freq
    sizep = (_PI * (deq * 1e-6)) / wl
    # All 1M classes reduce to a single m0 (the density correction is the
    # identity at rho_e = RHO_ICE), but honour per-size indices anyway.
    if np.all(m0 == m0[0]):
        return mie_efficiencies(sizep, complex(m0[0]))
    qext = np.empty(deq.shape)
    qbsca = np.empty(deq.shape)
    for i in range(deq.shape[0]):
        qe_i, qb_i = mie_efficiencies(sizep[i:i + 1], complex(m0[i]))
        qext[i], qbsca[i] = qe_i[0], qb_i[0]
    return qext, qbsca


def generate_lut(scheme: str = "1M", freq: float = 94.0,
                 k2: float = -1.0) -> dict:
    """Build the full (N_HYDRO, MT_NTT, N_RE_TYPES) Z-scale tables.

    Liquid classes populate temperature slots 0..18 (MT_TTL) and ice classes
    slots 0..19 (MT_TTI) of the shared MT_NTT axis — the same slots Fortran
    ``infind(mt_ttl/mt_tti, T)`` (1-based) selects at the exact grid
    temperatures. Unreachable slots hold NaN.

    Parameters
    ----------
    scheme : str
        Hydrometeor-class table; only '1M' is supported (see the
        NotImplementedError below for why 2M is ill-posed here).
    freq : float
        Radar frequency (GHz).
    k2 : float
        |K|^2; -1 resolves the frequency-dependent default.

    Returns
    -------
    dict
        Arrays ``ze_scaled``, ``zr_scaled``, ``kr_scaled`` plus metadata
        (``freq``, ``k2``, ``d_grid``, ``mt_ttl``, ``mt_tti``,
        ``re_quant``).
    """
    if scheme != "1M":
        raise NotImplementedError(
            "Only the 1M scheme LUT can be precomputed: the 2M Morrison "
            "LSCLIQ/CVCLIQ classes use p3 = -2, i.e. a Martin-1994 PSD width "
            "vu that depends on the runtime number concentration, so a "
            "(class, T, Re)-keyed table is ill-posed. Supporting 2M needs an "
            "extra vu (or Np) LUT axis — documented follow-up.")
    sd = HYDRO_CLASSES[scheme]
    k2_res = effective_k2(freq, k2)

    indices, re_quants = enumerate_re_bins()
    shape = (N_HYDRO, MT_NTT, N_RE_TYPES)
    ze_scaled = np.full(shape, np.nan)
    zr_scaled = np.full(shape, np.nan)
    kr_scaled = np.full(shape, np.nan)
    re_quant_slots = np.full(N_RE_TYPES, np.nan)
    re_quant_slots[indices - 1] = re_quants

    for tp in range(N_HYDRO):
        dtype = int(sd.dtype[tp])
        phase = int(sd.phase[tp])
        temps = MT_TTL if phase == 0 else MT_TTI
        deq, rho_e = _class_deq_rhoe(sd, tp)
        args = (float(sd.dmin[tp]), float(sd.dmax[tp]), float(sd.apm[tp]),
                float(sd.bpm[tp]), float(sd.rho[tp]), float(sd.p1[tp]),
                float(sd.p2[tp]), float(sd.p3[tp]))
        di = np.array([float(sd.p1[tp])]) if dtype == 4 else D_GRID
        for itt, t_kelvin in enumerate(temps):
            qext, qbsca = _mie_for_class_temperature(
                deq, rho_e, phase, float(t_kelvin), freq)
            for idx, re_q in zip(indices, re_quants):
                ni, _, _, _ = dsd(1.0, float(re_q), 0.0, di, dtype, 1.0,
                                  float(t_kelvin), *args)
                ze, zr, kr = zeff(freq, deq, ni, k2, float(t_kelvin), phase,
                                  1, rho_e, qe=qext, qs=qbsca)
                # Q = 1 g/kg, rho_a = 1 kg m-3 -> scale_factor = 1: these ARE
                # the scaled table values.
                ze_scaled[tp, itt, idx - 1] = ze
                zr_scaled[tp, itt, idx - 1] = zr
                kr_scaled[tp, itt, idx - 1] = kr

    return {
        "ze_scaled": ze_scaled,
        "zr_scaled": zr_scaled,
        "kr_scaled": kr_scaled,
        "freq": np.float64(freq),
        "k2": np.float64(k2_res),
        "d_grid": D_GRID,
        "mt_ttl": MT_TTL,
        "mt_tti": MT_TTI,
        "re_quant": re_quant_slots,
    }


def main():
    """Generate the 94-GHz 1M table and save it under jcosp/data/."""
    t0 = time.time()
    lut = generate_lut(scheme="1M", freq=94.0, k2=-1.0)
    elapsed = time.time() - t0

    _LUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_LUT_PATH, **lut)

    ze = lut["ze_scaled"]
    occ = np.isfinite(ze)
    print(f"Wrote {_LUT_PATH}")
    print(f"Generation time: {elapsed:.1f} s")
    print(f"Table shape {ze.shape}; occupied slots: {occ.sum()} "
          f"({100.0 * occ.mean():.1f}% of full grid)")
    n_re = np.isfinite(lut["re_quant"]).sum()
    print(f"Reachable Re bins: {n_re} of {N_RE_TYPES}")
    # Sanity: large-scale rain at 0 C (MT_TTL slot for 273.15 K).
    itt = int(np.argmin(np.abs(MT_TTL - 273.15)))
    lsr = ze[2, itt, :]
    lsr = lsr[np.isfinite(lsr) & (lsr > 0)]
    print(f"LSR @ 0C 10*log10(Ze_scaled): min {10 * np.log10(lsr.min()):.1f} "
          f"dBZ, max {10 * np.log10(lsr.max()):.1f} dBZ "
          f"(for 1 g/kg at rho_a = 1)")


if __name__ == "__main__":
    main()
