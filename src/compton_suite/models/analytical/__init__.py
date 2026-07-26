"""Analytical: Fast closed-form Compton-source physics estimates."""

from .analytical import estimate_yield, estimate_spectrum_width, angle_integrated_spectrum
from .analytical_adapter import AnalyticalAdapter

__all__ = ["estimate_yield", "estimate_spectrum_width", "angle_integrated_spectrum", "AnalyticalAdapter"]
