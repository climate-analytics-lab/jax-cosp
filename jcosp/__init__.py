"""jax-cosp: differentiable satellite simulators in JAX (COSP v2 translation)."""

from jcosp.classify import (WarmRainFlags, warm_rain_classification,
                            warm_rain_fraction)
from jcosp.simulator import CloudsatInputs, CloudsatOutputs, simulate_cloudsat
from jcosp.surrogate import DEFAULT_SMOOTHING, NO_SMOOTHING, Smoothing

__version__ = "0.1.0"
__all__ = [
    "CloudsatInputs", "CloudsatOutputs", "simulate_cloudsat",
    "WarmRainFlags", "warm_rain_classification", "warm_rain_fraction",
    "Smoothing", "DEFAULT_SMOOTHING", "NO_SMOOTHING",
]
