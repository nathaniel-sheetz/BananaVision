# BananaVision

A Python tool that analyzes photographs of banana displays and reports the percentage of bananas in three ripeness categories: green, yellow (no spots), and yellow (with spots).

## How It Works

BananaVision uses computer vision techniques to detect and classify bananas:

1. **Detection**: Identifies banana pixels using HSV color masking for green and yellow hues
2. **Classification**: Categorizes each pixel as green, yellow-clean, or yellow-spotted
3. **Spot Detection**: Finds brown spots in the interior of yellow banana regions (excluding tips/edges)
4. **Reporting**: Calculates percentage breakdown by pixel area

## Installation

```bash
pip install -r requirements.txt
```

Requires:
- opencv-python >= 4.8.0
- numpy >= 1.24.0

## Usage

```bash
# Analyze a single image
python main.py image.jpg

# Analyze multiple images
python main.py image1.jpg image2.jpg

# Analyze all images in a directory
python main.py path/to/folder/

# Show debug visualization (masks and contours)
python main.py image.jpg --debug
```

### Sample Output

```
Analyzing: bananas_shelf.jpg
=============================================
Banana Ripeness Analysis
---------------------------------------------
Green:                29.3%
Yellow (no spots):    50.2%
Yellow (spotted):     20.5%
---------------------------------------------
Total banana area: 64,750 pixels
=============================================
```

## Tuning Parameters

All color thresholds are in `banana_vision/config.py`. HSV values use OpenCV's ranges: H (0-179), S (0-255), V (0-255).

### Green/Yellow Boundary (Hue)

```python
GREEN_LOWER = (32, 80, 80)    # Hue 32+ = green
YELLOW_UPPER = (32, 255, 255) # Hue up to 32 = yellow
```

**Effect of changing the hue boundary:**
- **Lower value (e.g., 28-30)**: More pixels classified as green, fewer as yellow
- **Higher value (e.g., 34-36)**: Fewer pixels classified as green, more as yellow

Adjust this if green bananas are being misclassified as yellow (lower the boundary) or yellow bananas are being misclassified as green (raise the boundary).

### Spot Detection Range

```python
SPOT_LOWER = (5, 30, 30)
SPOT_UPPER = (30, 255, 200)
```

**Parameters:**
- **Hue (5-30)**: Brown/tan color range. Wider range catches more spot colors.
- **Saturation (30-255)**: Lower minimum catches desaturated spots.
- **Value (30-200)**: Upper limit excludes bright yellow pixels; spots are darker.

**Effect of changes:**
- **Wider hue range**: Detects more varied spot colors but may catch non-spot pixels
- **Lower value ceiling**: More selective for truly dark spots
- **Higher saturation minimum**: Excludes grayish pixels that aren't spots

### Erosion Kernel (Interior Detection)

In `banana_vision/classifier.py`:

```python
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
```

This erodes the yellow mask before spot detection to exclude banana tips/edges (which are naturally darker).

**Effect of kernel size:**
- **Larger (e.g., 20, 20)**: Only detects spots deep in the banana interior; may miss spots near edges
- **Smaller (e.g., 10, 10)**: Detects spots closer to edges but may pick up dark tips as false positives

### Dilation Kernel (Spot Spread)

```python
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
```

After detecting spot pixels, dilation expands the "spotted" region to include surrounding yellow pixels as part of a spotted banana.

**Effect of kernel size:**
- **Larger (e.g., 20, 20)**: Each spot affects a larger area; higher spotted percentage
- **Smaller (e.g., 8, 8)**: Spots affect smaller area; lower spotted percentage, more conservative

### Minimum Contour Area

```python
MIN_CONTOUR_AREA = 500
```

Filters out small detected regions (noise).

**Effect:**
- **Higher value**: Ignores small banana fragments or noise
- **Lower value**: Includes smaller regions but may catch non-banana artifacts

## Debug Mode

Run with `--debug` to visualize detection:
- Original image with contours outlined
- Green mask overlay
- Yellow mask overlay
- Spot detection overlay

This helps identify which pixels are being detected and whether thresholds need adjustment for your specific images.

## Project Structure

```
BananaVision/
├── banana_vision/
│   ├── __init__.py      # Package exports
│   ├── config.py        # Color thresholds (tune these)
│   ├── detector.py      # Banana mask detection
│   ├── classifier.py    # Ripeness classification
│   └── analyzer.py      # Main pipeline
├── main.py              # CLI entry point
├── requirements.txt
└── README.md
```

## Limitations

- Optimized for well-lit retail/display settings
- Synthetic or unusually colored images may require threshold tuning
- Bananas that are touching may be detected as a single region (classification still works per-pixel)
- Very dark or overexposed images may need adjusted value (V) thresholds
