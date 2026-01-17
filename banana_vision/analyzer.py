"""Main analysis pipeline for banana ripeness detection."""

import cv2
import numpy as np

from .detector import detect_bananas, get_debug_masks
from .classifier import classify_all_regions, get_spot_mask, RipenessCategory


def analyze_image(image_path: str) -> dict:
    """
    Analyze an image and return banana ripeness percentages.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary containing:
        - green_percent: Percentage of banana area that is green
        - yellow_clean_percent: Percentage that is yellow without spots
        - yellow_spotted_percent: Percentage that is yellow with spots
        - total_banana_pixels: Total number of banana pixels detected

    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image cannot be read
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return analyze_image_array(image)


def analyze_image_array(image: np.ndarray) -> dict:
    """
    Analyze a BGR image array and return banana ripeness percentages.

    Args:
        image: BGR image as numpy array

    Returns:
        Dictionary with ripeness percentages and total banana pixels
    """
    mask, contours = detect_bananas(image)
    counts = classify_all_regions(image, contours)

    total_pixels = sum(counts.values())

    if total_pixels == 0:
        return {
            "green_percent": 0.0,
            "yellow_clean_percent": 0.0,
            "yellow_spotted_percent": 0.0,
            "total_banana_pixels": 0,
        }

    return {
        "green_percent": (counts[RipenessCategory.GREEN] / total_pixels) * 100,
        "yellow_clean_percent": (counts[RipenessCategory.YELLOW_CLEAN] / total_pixels) * 100,
        "yellow_spotted_percent": (counts[RipenessCategory.YELLOW_SPOTTED] / total_pixels) * 100,
        "total_banana_pixels": total_pixels,
    }


def create_debug_visualization(image_path: str) -> dict[str, np.ndarray]:
    """
    Create debug visualizations for threshold tuning.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary containing:
        - original: Original image with contours drawn
        - green_mask: Green detection mask overlay
        - yellow_mask: Yellow detection mask overlay
        - spot_mask: Spot detection mask overlay
        - combined_mask: Combined banana detection mask
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    mask, contours = detect_bananas(image)
    debug_masks = get_debug_masks(image)
    spot_mask = get_spot_mask(image, mask)

    # Create visualization with contours
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    # Create colored overlays
    green_overlay = create_color_overlay(image, debug_masks["green"], (0, 255, 0))
    yellow_overlay = create_color_overlay(image, debug_masks["yellow"], (0, 255, 255))
    spot_overlay = create_color_overlay(image, spot_mask, (0, 0, 255))

    return {
        "original": contour_image,
        "green_mask": green_overlay,
        "yellow_mask": yellow_overlay,
        "spot_mask": spot_overlay,
        "combined_mask": mask,
    }


def create_color_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create a colored overlay visualization.

    Args:
        image: Original BGR image
        mask: Binary mask
        color: BGR color tuple for the overlay
        alpha: Transparency (0-1)

    Returns:
        Image with colored overlay applied
    """
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
