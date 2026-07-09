"""jax-cosp: differentiable satellite simulators in JAX (COSP v2 translation)."""

from jcosp.classify import (WarmRainFlags, warm_rain_classification,
                            warm_rain_fraction)
from jcosp.simulator import CloudsatInputs, CloudsatOutputs, simulate_cloudsat

__version__ = "0.1.0"
__all__ = [
    "CloudsatInputs", "CloudsatOutputs", "simulate_cloudsat",
    "WarmRainFlags", "warm_rain_classification", "warm_rain_fraction",
]
