"""Color constants and thresholds for banana detection and classification."""

# HSV ranges for OpenCV (H: 0-179, S: 0-255, V: 0-255)

# Green bananas: greenish hue (32+)
GREEN_LOWER = (32, 80, 80)
GREEN_UPPER = (65, 255, 255)

# Yellow bananas: yellow hue (up to 32)
YELLOW_LOWER = (15, 100, 100)
YELLOW_UPPER = (32, 255, 255)

# Brown spots: wider range for varied spot colors
SPOT_LOWER = (5, 30, 30)
SPOT_UPPER = (30, 255, 200)

# Threshold: if >5% of yellow region is brown, classify as spotted
SPOT_THRESHOLD = 0.05

# Minimum contour area to filter noise (pixels)
MIN_CONTOUR_AREA = 500

# === Banana Segmentation Parameters ===

# Local maxima kernel size for watershed segmentation
# This determines minimum separation between banana centers (pixels)
# Smaller = more sensitive to individual bananas
LOCAL_MAXIMA_KERNEL_SIZE = 15

# Minimum distance transform value to consider as a banana center
# Filters out noise and thin regions
MIN_DISTANCE_THRESHOLD = 5.0

# Minimum area for a segmented banana (pixels)
MIN_BANANA_AREA = 800

# Interior erosion for per-banana spot detection
BANANA_INTERIOR_EROSION_SIZE = 10

# Minimum spot pixels to classify as "spotted" (1 = any spot counts)
MIN_SPOT_PIXELS = 5
