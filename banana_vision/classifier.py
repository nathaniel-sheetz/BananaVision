"""Ripeness classification for detected banana regions."""

import cv2
import numpy as np

from .config import (
    GREEN_LOWER,
    GREEN_UPPER,
    YELLOW_LOWER,
    YELLOW_UPPER,
    SPOT_LOWER,
    SPOT_UPPER,
    SPOT_THRESHOLD,
    BANANA_INTERIOR_EROSION_SIZE,
    MIN_SPOT_PIXELS,
)


class RipenessCategory:
    """Enumeration of ripeness categories."""
    GREEN = "green"
    YELLOW_CLEAN = "yellow_clean"
    YELLOW_SPOTTED = "yellow_spotted"


def classify_region(
    image: np.ndarray,
    contour: np.ndarray
) -> tuple[str, int]:
    """
    Classify a single banana region by ripeness.

    Args:
        image: BGR image as numpy array
        contour: Contour array defining the region

    Returns:
        Tuple of (category, pixel_count) where:
        - category: One of RipenessCategory values
        - pixel_count: Number of pixels in this region
    """
    # Create mask for this contour
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Count green and yellow pixels within this region
    green_mask = cv2.inRange(hsv, np.array(GREEN_LOWER), np.array(GREEN_UPPER))
    yellow_mask = cv2.inRange(hsv, np.array(YELLOW_LOWER), np.array(YELLOW_UPPER))

    # Apply region mask
    green_in_region = cv2.bitwise_and(green_mask, mask)
    yellow_in_region = cv2.bitwise_and(yellow_mask, mask)

    green_pixels = cv2.countNonZero(green_in_region)
    yellow_pixels = cv2.countNonZero(yellow_in_region)

    total_pixels = green_pixels + yellow_pixels

    if total_pixels == 0:
        return RipenessCategory.YELLOW_CLEAN, 0

    # Determine primary color
    if green_pixels > yellow_pixels:
        return RipenessCategory.GREEN, total_pixels

    # For yellow bananas, check for spots
    if yellow_pixels > 0:
        spot_mask = cv2.inRange(hsv, np.array(SPOT_LOWER), np.array(SPOT_UPPER))
        spot_in_region = cv2.bitwise_and(spot_mask, yellow_in_region)
        spot_pixels = cv2.countNonZero(spot_in_region)

        spot_ratio = spot_pixels / yellow_pixels

        if spot_ratio > SPOT_THRESHOLD:
            return RipenessCategory.YELLOW_SPOTTED, total_pixels

    return RipenessCategory.YELLOW_CLEAN, total_pixels


def classify_all_regions(
    image: np.ndarray,
    contours: list[np.ndarray]
) -> dict[str, int]:
    """
    Classify all detected banana pixels by color.

    Uses per-pixel classification rather than per-contour majority vote,
    so green and yellow bananas in the same contour are counted separately.

    Args:
        image: BGR image as numpy array
        contours: List of contour arrays

    Returns:
        Dictionary mapping category names to total pixel counts
    """
    if not contours:
        return {
            RipenessCategory.GREEN: 0,
            RipenessCategory.YELLOW_CLEAN: 0,
            RipenessCategory.YELLOW_SPOTTED: 0,
        }

    # Create combined mask from all contours
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, -1)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Get color masks
    green_mask = cv2.inRange(hsv, np.array(GREEN_LOWER), np.array(GREEN_UPPER))
    yellow_mask = cv2.inRange(hsv, np.array(YELLOW_LOWER), np.array(YELLOW_UPPER))
    spot_mask = cv2.inRange(hsv, np.array(SPOT_LOWER), np.array(SPOT_UPPER))

    # Apply banana region mask
    green_in_banana = cv2.bitwise_and(green_mask, mask)
    yellow_in_banana = cv2.bitwise_and(yellow_mask, mask)

    green_pixels = cv2.countNonZero(green_in_banana)
    total_yellow = cv2.countNonZero(yellow_in_banana)

    # Erode yellow mask to get interior region (excludes tips/edges)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    yellow_interior = cv2.erode(yellow_in_banana, erode_kernel)

    # Only detect spots in the interior of yellow regions (not tips)
    spot_in_interior = cv2.bitwise_and(spot_mask, yellow_interior)

    # Dilate spot mask to find yellow pixels near interior spots
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    spots_dilated = cv2.dilate(spot_in_interior, dilate_kernel)

    # Yellow pixels near spots are "spotted"
    yellow_near_spots = cv2.bitwise_and(yellow_in_banana, spots_dilated)
    yellow_spotted_pixels = cv2.countNonZero(yellow_near_spots)

    # Remaining yellow pixels are "clean"
    yellow_clean_pixels = total_yellow - yellow_spotted_pixels

    return {
        RipenessCategory.GREEN: green_pixels,
        RipenessCategory.YELLOW_CLEAN: max(0, yellow_clean_pixels),
        RipenessCategory.YELLOW_SPOTTED: yellow_spotted_pixels,
    }


def get_spot_mask(image: np.ndarray, banana_mask: np.ndarray) -> np.ndarray:
    """
    Generate spot mask for debug visualization.

    Args:
        image: BGR image as numpy array
        banana_mask: Binary mask of detected banana regions

    Returns:
        Binary mask of detected spots within banana regions
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    spot_mask = cv2.inRange(hsv, np.array(SPOT_LOWER), np.array(SPOT_UPPER))
    return cv2.bitwise_and(spot_mask, banana_mask)


def classify_single_banana(image: np.ndarray, banana_mask: np.ndarray) -> str:
    """
    Classify a single segmented banana by ripeness.

    A banana is classified as:
    - GREEN: if it has more green pixels than yellow
    - YELLOW_SPOTTED: if it's yellow and has ANY interior spots
    - YELLOW_CLEAN: if it's yellow with no spots

    Args:
        image: BGR image as numpy array
        banana_mask: Binary mask for this individual banana

    Returns:
        One of RipenessCategory values
    """
    if cv2.countNonZero(banana_mask) == 0:
        return RipenessCategory.YELLOW_CLEAN

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Count green and yellow pixels within this banana
    green_mask = cv2.inRange(hsv, np.array(GREEN_LOWER), np.array(GREEN_UPPER))
    yellow_mask = cv2.inRange(hsv, np.array(YELLOW_LOWER), np.array(YELLOW_UPPER))

    green_in_banana = cv2.bitwise_and(green_mask, banana_mask)
    yellow_in_banana = cv2.bitwise_and(yellow_mask, banana_mask)

    green_pixels = cv2.countNonZero(green_in_banana)
    yellow_pixels = cv2.countNonZero(yellow_in_banana)

    # Determine base color
    if green_pixels > yellow_pixels:
        return RipenessCategory.GREEN

    # For yellow bananas, check for spots in the interior
    if yellow_pixels > 0:
        # Erode the banana mask to get interior region (exclude edges/tips)
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (BANANA_INTERIOR_EROSION_SIZE, BANANA_INTERIOR_EROSION_SIZE)
        )
        interior_mask = cv2.erode(banana_mask, erode_kernel)

        # Detect spots
        spot_mask = cv2.inRange(hsv, np.array(SPOT_LOWER), np.array(SPOT_UPPER))

        # Only count spots in the interior
        spots_in_interior = cv2.bitwise_and(spot_mask, interior_mask)
        spot_pixel_count = cv2.countNonZero(spots_in_interior)

        # If ANY spots exist (above threshold), classify as spotted
        if spot_pixel_count >= MIN_SPOT_PIXELS:
            return RipenessCategory.YELLOW_SPOTTED

    return RipenessCategory.YELLOW_CLEAN


def classify_all_bananas(
    image: np.ndarray,
    banana_segments: list[dict]
) -> dict[str, int]:
    """
    Classify all segmented bananas and return counts per category.

    Args:
        image: BGR image as numpy array
        banana_segments: List of segment dictionaries from segment_individual_bananas()

    Returns:
        Dictionary mapping category names to banana counts (not pixel counts)
    """
    counts = {
        RipenessCategory.GREEN: 0,
        RipenessCategory.YELLOW_CLEAN: 0,
        RipenessCategory.YELLOW_SPOTTED: 0,
    }

    for segment in banana_segments:
        category = classify_single_banana(image, segment['mask'])
        counts[category] += 1
        # Store classification in segment for debug visualization
        segment['category'] = category

    return counts
