"""Banana mask detection using HSV color masking."""

import cv2
import numpy as np

from .config import (
    GREEN_LOWER,
    GREEN_UPPER,
    YELLOW_LOWER,
    YELLOW_UPPER,
    MIN_CONTOUR_AREA,
    LOCAL_MAXIMA_KERNEL_SIZE,
    MIN_DISTANCE_THRESHOLD,
    MIN_BANANA_AREA,
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


def segment_individual_bananas(
    image: np.ndarray,
    banana_mask: np.ndarray,
    contours: list[np.ndarray]
) -> list[dict]:
    """
    Segment touching bananas using edge detection and watershed algorithm.

    Uses internal edges within the banana region to find boundaries between
    individual bananas, then applies watershed for final segmentation.

    Args:
        image: BGR image as numpy array
        banana_mask: Binary mask of all detected banana pixels
        contours: List of contour arrays from detect_bananas()

    Returns:
        List of dictionaries, each containing:
        - 'mask': Binary mask for this individual banana
        - 'contour': Contour array for this banana
        - 'area': Area in pixels
        - 'label': Integer label from watershed
    """
    if banana_mask is None or cv2.countNonZero(banana_mask) == 0:
        return []

    # Step 1: Use edge detection to find boundaries between bananas
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 120, 240)

    # Mask edges to only banana region
    edges_in_banana = cv2.bitwise_and(edges, banana_mask)

    # Dilate edges to create boundary gaps
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundaries = cv2.dilate(edges_in_banana, edge_kernel, iterations=2)

    # Subtract boundaries from mask to separate touching bananas
    separated_mask = cv2.subtract(banana_mask, boundaries)

    # Clean up with morphological opening
    separated_mask = cv2.morphologyEx(separated_mask, cv2.MORPH_OPEN, edge_kernel)

    # Step 2: Apply distance transform to the separated mask
    dist_transform = cv2.distanceTransform(separated_mask, cv2.DIST_L2, 5)

    dist_max = dist_transform.max()
    if dist_max == 0:
        return []

    # Step 3: Find local maxima in distance transform
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (LOCAL_MAXIMA_KERNEL_SIZE, LOCAL_MAXIMA_KERNEL_SIZE)
    )
    dilated = cv2.dilate(dist_transform, kernel)

    # Local maxima with tolerance for floating point comparison
    local_max = (np.abs(dist_transform - dilated) < 0.01) & (dist_transform >= MIN_DISTANCE_THRESHOLD)
    local_max = local_max.astype(np.uint8) * 255

    # Dilate local maxima slightly to ensure connected seed regions
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    local_max = cv2.dilate(local_max, small_kernel, iterations=1)

    # Step 4: Setup watershed markers
    sure_bg = cv2.dilate(banana_mask, small_kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, local_max)

    num_labels, markers = cv2.connectedComponents(local_max)

    # Add 1 to all labels so background is 1 instead of 0
    markers = markers + 1

    # Mark unknown region as 0
    markers[unknown == 255] = 0

    # Step 5: Apply watershed
    image_bgr = image.copy()
    markers = cv2.watershed(image_bgr, markers)

    # Step 6: Extract individual banana masks
    banana_segments = []

    for label in range(2, num_labels + 1):
        label_mask = np.zeros(banana_mask.shape, dtype=np.uint8)
        label_mask[markers == label] = 255

        # Intersect with original banana mask to clean up
        label_mask = cv2.bitwise_and(label_mask, banana_mask)

        segment_contours, _ = cv2.findContours(
            label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not segment_contours:
            continue

        largest_contour = max(segment_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < MIN_BANANA_AREA:
            continue

        banana_segments.append({
            'mask': label_mask,
            'contour': largest_contour,
            'area': area,
            'label': label
        })

    return banana_segments
