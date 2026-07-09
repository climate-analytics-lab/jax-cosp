"""Test configuration: run the suite in float64.

The Fortran reference is double precision throughout; x64 makes the
transliteration comparisons exact. The library itself also runs in JAX's
default float32 (LUT bin selection may then differ by one slot at exact bin
edges — the physical effect of the quantization either way).
"""

import jax

jax.config.update("jax_enable_x64", True)
