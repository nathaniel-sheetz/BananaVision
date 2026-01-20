"""Main analysis pipeline for banana ripeness detection."""

import cv2
import numpy as np

from .detector import detect_bananas, get_debug_masks, segment_individual_bananas
from .classifier import (
    classify_all_regions,
    classify_all_bananas,
    get_spot_mask,
    RipenessCategory,
)


def analyze_image(image_path: str, mode: str = "banana") -> dict:
    """
    Analyze an image and return banana ripeness percentages.

    Args:
        image_path: Path to the image file
        mode: Analysis mode - "banana" for per-banana, "pixel" for per-pixel

    Returns:
        Dictionary containing ripeness percentages and counts.
        For "banana" mode: percentages are based on banana count
        For "pixel" mode: percentages are based on pixel area

    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image cannot be read
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    if mode == "pixel":
        return analyze_image_pixels(image)
    else:
        return analyze_image_bananas(image)


def analyze_image_pixels(image: np.ndarray) -> dict:
    """
    Analyze a BGR image array using per-pixel classification.

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
            "mode": "pixel",
        }

    return {
        "green_percent": (counts[RipenessCategory.GREEN] / total_pixels) * 100,
        "yellow_clean_percent": (counts[RipenessCategory.YELLOW_CLEAN] / total_pixels) * 100,
        "yellow_spotted_percent": (counts[RipenessCategory.YELLOW_SPOTTED] / total_pixels) * 100,
        "total_banana_pixels": total_pixels,
        "mode": "pixel",
    }


def analyze_image_bananas(image: np.ndarray) -> dict:
    """
    Analyze a BGR image array using per-banana classification.

    Each banana is classified as a unit, then percentages are calculated
    based on how many bananas fall into each category.

    Args:
        image: BGR image as numpy array

    Returns:
        Dictionary with:
        - green_percent: Percentage of bananas that are green
        - yellow_clean_percent: Percentage of bananas that are yellow (no spots)
        - yellow_spotted_percent: Percentage of bananas that are spotted
        - green_count: Number of green bananas
        - yellow_clean_count: Number of clean yellow bananas
        - yellow_spotted_count: Number of spotted bananas
        - total_bananas: Total number of bananas detected
        - mode: "banana"
    """
    mask, contours = detect_bananas(image)
    banana_segments = segment_individual_bananas(image, mask, contours)
    counts = classify_all_bananas(image, banana_segments)

    total_bananas = sum(counts.values())

    if total_bananas == 0:
        return {
            "green_percent": 0.0,
            "yellow_clean_percent": 0.0,
            "yellow_spotted_percent": 0.0,
            "green_count": 0,
            "yellow_clean_count": 0,
            "yellow_spotted_count": 0,
            "total_bananas": 0,
            "mode": "banana",
        }

    return {
        "green_percent": (counts[RipenessCategory.GREEN] / total_bananas) * 100,
        "yellow_clean_percent": (counts[RipenessCategory.YELLOW_CLEAN] / total_bananas) * 100,
        "yellow_spotted_percent": (counts[RipenessCategory.YELLOW_SPOTTED] / total_bananas) * 100,
        "green_count": counts[RipenessCategory.GREEN],
        "yellow_clean_count": counts[RipenessCategory.YELLOW_CLEAN],
        "yellow_spotted_count": counts[RipenessCategory.YELLOW_SPOTTED],
        "total_bananas": total_bananas,
        "mode": "banana",
    }


def create_debug_visualization(
    image_path: str,
    mode: str = "banana"
) -> dict[str, np.ndarray]:
    """
    Create debug visualizations for threshold tuning.

    Args:
        image_path: Path to the image file
        mode: Analysis mode - "banana" for per-banana, "pixel" for per-pixel

    Returns:
        Dictionary containing:
        - original: Original image with contours drawn
        - green_mask: Green detection mask overlay
        - yellow_mask: Yellow detection mask overlay
        - spot_mask: Spot detection mask overlay
        - combined_mask: Combined banana detection mask
        - segmented: (banana mode only) Segmented bananas with numbered labels
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

    result = {
        "original": contour_image,
        "green_mask": green_overlay,
        "yellow_mask": yellow_overlay,
        "spot_mask": spot_overlay,
        "combined_mask": mask,
    }

    # Add category views for banana mode
    if mode == "banana":
        banana_segments = segment_individual_bananas(image, mask, contours)
        counts = classify_all_bananas(image, banana_segments)  # Adds 'category' to segments

        category_visualizations = create_category_visualizations(image, banana_segments)
        result["green_bananas"] = category_visualizations["green"]
        result["yellow_clean"] = category_visualizations["yellow_clean"]
        result["yellow_spotted"] = category_visualizations["yellow_spotted"]

        # Include counts for window titles
        result["green_count"] = counts[RipenessCategory.GREEN]
        result["yellow_clean_count"] = counts[RipenessCategory.YELLOW_CLEAN]
        result["yellow_spotted_count"] = counts[RipenessCategory.YELLOW_SPOTTED]

    return result


def create_category_visualizations(
    image: np.ndarray,
    banana_segments: list[dict]
) -> dict[str, np.ndarray]:
    """
    Create separate visualizations for each banana category.

    Each visualization shows a dimmed background with only bananas of that
    category highlighted with color overlay.

    Args:
        image: BGR image as numpy array
        banana_segments: List of segment dictionaries with 'contour' and 'category'

    Returns:
        Dictionary with keys 'green', 'yellow_clean', 'yellow_spotted'
        Each value is an image showing only bananas of that category
    """
    # Category colors (BGR)
    category_colors = {
        RipenessCategory.GREEN: (0, 200, 0),
        RipenessCategory.YELLOW_CLEAN: (0, 230, 230),
        RipenessCategory.YELLOW_SPOTTED: (0, 100, 200),
    }

    # Map categories to output keys
    category_keys = {
        RipenessCategory.GREEN: 'green',
        RipenessCategory.YELLOW_CLEAN: 'yellow_clean',
        RipenessCategory.YELLOW_SPOTTED: 'yellow_spotted',
    }

    # Group segments by category
    segments_by_category = {
        RipenessCategory.GREEN: [],
        RipenessCategory.YELLOW_CLEAN: [],
        RipenessCategory.YELLOW_SPOTTED: [],
    }

    for segment in banana_segments:
        category = segment.get('category', RipenessCategory.YELLOW_CLEAN)
        if category in segments_by_category:
            segments_by_category[category].append(segment)

    result = {}

    for category, segments in segments_by_category.items():
        # Start with dimmed original image
        vis_image = (image * 0.3).astype(np.uint8)
        color = category_colors[category]

        for segment in segments:
            contour = segment['contour']

            # Draw filled contour with 60% opacity
            overlay = vis_image.copy()
            cv2.drawContours(overlay, [contour], 0, color, -1)
            cv2.addWeighted(overlay, 0.6, vis_image, 0.4, 0, vis_image)

            # Draw contour outline
            cv2.drawContours(vis_image, [contour], 0, color, 2)

        result[category_keys[category]] = vis_image

    return result


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
