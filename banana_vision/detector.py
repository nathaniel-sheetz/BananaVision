"""Banana mask detection using HSV color masking."""

import cv2
import numpy as np

from .config import (
    GREEN_LOWER,
    GREEN_UPPER,
    YELLOW_LOWER,
    YELLOW_UPPER,
    MIN_CONTOUR_AREA,
)


def detect_bananas(image: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Detect banana regions in an image using HSV color masking.

    Args:
        image: BGR image as numpy array

    Returns:
        Tuple of (combined_mask, contours) where:
        - combined_mask: Binary mask of all detected banana pixels
        - contours: List of contour arrays for each detected banana region
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create masks for green and yellow bananas
    green_mask = cv2.inRange(hsv, np.array(GREEN_LOWER), np.array(GREEN_UPPER))
    yellow_mask = cv2.inRange(hsv, np.array(YELLOW_LOWER), np.array(YELLOW_UPPER))

    # Combine masks
    combined_mask = cv2.bitwise_or(green_mask, yellow_mask)

    # Morphological opening to remove noise (erode then dilate)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # Additional closing to fill small gaps
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter contours by minimum area
    filtered_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA
    ]

    # Create cleaned mask from filtered contours only
    cleaned_mask = np.zeros_like(combined_mask)
    if filtered_contours:
        cv2.drawContours(cleaned_mask, filtered_contours, -1, 255, -1)

    return cleaned_mask, filtered_contours


def get_debug_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    """
    Generate individual masks for debug visualization.

    Args:
        image: BGR image as numpy array

    Returns:
        Dictionary with 'green', 'yellow', and 'combined' masks
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, np.array(GREEN_LOWER), np.array(GREEN_UPPER))
    yellow_mask = cv2.inRange(hsv, np.array(YELLOW_LOWER), np.array(YELLOW_UPPER))
    combined_mask = cv2.bitwise_or(green_mask, yellow_mask)

    return {
        "green": green_mask,
        "yellow": yellow_mask,
        "combined": combined_mask,
    }
