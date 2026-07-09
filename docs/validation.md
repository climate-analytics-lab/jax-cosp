# Validation against the reference COSP Fortran

Three layers of validation, strongest first:

## 1. Compiled-Fortran KGO (tools/kgo/)

`tools/kgo/build.sh` compiles the actual COSP v2.0 radar path
(quickbeam_optics, optics_lib Mie, quickbeam, prec_scops, gases) with
gfortran into a standalone driver, `tools/kgo/compare_kgo.py` feeds identical
multi-subcolumn raining/snowing cases to it and to jcosp, and
`jcosp/kgo_test.py` runs the comparison under pytest (skipped when the binary
is absent).

Results (2026-07-09, 165 hydrometeor-bearing volumes, 6 hydrometeor classes,
default and prescribed effective radii):

| regime | z_vol | kr_vol | g_vol | dBZe | prec_scops |
| --- | --- | --- | --- | --- | --- |
| temperatures on the 5-K LUT grid | 2.2e-7 rel | 2.1e-7 rel | 2.5e-8 rel | 1.2e-6 dB | identical |
| realistic temperatures | 2.6e-2 rel | 4.4e-3 rel | 2.5e-8 rel | 0.13 dB | identical |

The exact-regime floor (~2e-7) is the gamma-function implementation and
float64-vs-single-precision-literal differences documented in
`translation_notes.md`. The realistic-regime envelope is **not an error of
the port**: COSP fills its Z-scale table lazily, evaluating the Mie/DSD
integrals at the *first temperature encountered* in each 5-K bin and caching
the result — so the reference output is itself encounter-order-dependent
within that envelope. jcosp evaluates every bin at its center, which is
deterministic and corresponds to running Fortran COSP with a pre-generated
scale-LUT file.

## 2. Loop-transliteration tests (co-located `*_test.py`)

Every intricate ported routine is checked element-for-element against a
direct NumPy loop transliteration of its Fortran: SCOPS (all overlap modes),
PREC_SCOPS, COSP_CHANGE_VERTICAL_GRID, the quickbeam attenuation gate loops
(both radar orientations), the Re-quantization index arithmetic, and infind
temperature binning.

## 3. Physical/property tests

Mie Rayleigh limits, DSD mass closure, calc_Re/DSD moment round-trips,
Z-scale linearity in ρq, CFAD normalization, warm/cold rain classification
on constructed cases, jit/grad finiteness through the full pipeline, and
fixed-key reproducibility.
