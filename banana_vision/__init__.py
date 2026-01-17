"""BananaVision - Analyze banana ripeness from photographs."""

from .analyzer import analyze_image
from .config import (
    GREEN_LOWER,
    GREEN_UPPER,
    YELLOW_LOWER,
    YELLOW_UPPER,
    SPOT_LOWER,
    SPOT_UPPER,
    SPOT_THRESHOLD,
    MIN_CONTOUR_AREA,
)

__version__ = "1.0.0"
__all__ = [
    "analyze_image",
    "GREEN_LOWER",
    "GREEN_UPPER",
    "YELLOW_LOWER",
    "YELLOW_UPPER",
    "SPOT_LOWER",
    "SPOT_UPPER",
    "SPOT_THRESHOLD",
    "MIN_CONTOUR_AREA",
]
