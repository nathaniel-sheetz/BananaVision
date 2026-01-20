# BananaVision

A Python tool that analyzes photographs of banana displays and reports the percentage of bananas in three ripeness categories: green, yellow (no spots), and yellow (with spots).

## How It Works

BananaVision uses computer vision techniques to detect and classify bananas:

1. **Detection**: Identifies banana pixels using HSV color masking for green and yellow hues
2. **Segmentation** (optional): Separates touching bananas using edge detection and watershed algorithm
3. **Classification**: Categorizes bananas by ripeness (green, yellow-clean, yellow-spotted)
4. **Spot Detection**: Finds brown spots in the interior of yellow banana regions (excluding tips/edges)
5. **Reporting**: Calculates percentage breakdown by banana count or pixel area

### Analysis Modes

BananaVision supports two analysis modes:

**Per-Banana Mode** (default, `--mode=banana`):
- Segments the image to identify individual bananas
- Classifies each banana as a unit (one banana = one vote)
- Reports percentages based on banana count
- **Pros**: More intuitive results ("3 out of 10 bananas are spotted")
- **Cons**: Segmentation can be imperfect—touching bananas may merge or single bananas may fragment

**Per-Pixel Mode** (`--mode=pixel`):
- Classifies each pixel independently
- Reports percentages based on pixel area
- **Pros**: No segmentation errors; works well when bananas overlap heavily
- **Cons**: Large bananas have more influence than small ones; results are less intuitive

## Installation

```bash
pip install -r requirements.txt
```

Requires:
- opencv-python >= 4.8.0
- numpy >= 1.24.0

## Usage

```bash
# Analyze a single image (per-banana mode, default)
python main.py image.jpg

# Use per-pixel mode instead
python main.py image.jpg --mode=pixel

# Analyze multiple images
python main.py image1.jpg image2.jpg

# Analyze all images in a directory
python main.py path/to/folder/

# Show debug visualization (masks, contours, and category views)
python main.py image.jpg --debug
```

### Sample Output

Per-banana mode (default):
```
Analyzing: bananas_shelf.jpg
==================================================
Banana Ripeness Analysis (Per-Banana)
--------------------------------------------------
Green:                 30.0%  (3 bananas)
Yellow (no spots):     50.0%  (5 bananas)
Yellow (spotted):      20.0%  (2 bananas)
--------------------------------------------------
Total bananas detected: 10
==================================================
```

Per-pixel mode:
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

### Banana Segmentation (Separating Touching Bananas)

BananaVision uses edge detection and watershed segmentation to separate touching bananas. If bananas are being incorrectly split into fragments, or not being separated at all, adjust these parameters:

#### Canny Edge Thresholds

In `banana_vision/detector.py` (~line 115):

```python
edges = cv2.Canny(gray, 120, 240)
```

These thresholds control which edges are detected between bananas. The two values are the low and high thresholds.

**Effect of changes:**
- **Higher values (e.g., 150, 300)**: Only strong edges detected → less fragmentation, but touching bananas may not separate
- **Lower values (e.g., 50, 150)**: More edges detected → better separation of touching bananas, but may fragment single bananas

**Symptoms and fixes:**
- Single bananas split into fragments → raise thresholds
- Touching bananas not separating → lower thresholds

#### Edge Dilation Iterations

In `banana_vision/detector.py` (~line 122):

```python
boundaries = cv2.dilate(edges_in_banana, edge_kernel, iterations=2)
```

Controls how much detected edges are expanded to create boundary gaps.

**Effect of changes:**
- **Higher iterations (e.g., 3)**: Wider gaps between regions → more aggressive separation
- **Lower iterations (e.g., 1)**: Narrower gaps → less separation, fewer fragments

#### Minimum Banana Area

In `banana_vision/config.py`:

```python
MIN_BANANA_AREA = 800
```

Minimum pixel area for a segmented region to be counted as a banana. Filters out small fragments.

**Effect of changes:**
- **Higher value (e.g., 1500)**: Filters out small fragments but may miss small bananas
- **Lower value (e.g., 300)**: Keeps smaller regions but may include fragments

#### Local Maxima Kernel Size

In `banana_vision/config.py`:

```python
LOCAL_MAXIMA_KERNEL_SIZE = 15
```

Controls minimum separation between detected banana centers in the watershed algorithm.

**Effect of changes:**
- **Larger (e.g., 25)**: Requires more distance between banana centers → fewer, larger segments
- **Smaller (e.g., 10)**: Allows closer centers → more segments, may over-segment

#### Minimum Distance Threshold

In `banana_vision/config.py`:

```python
MIN_DISTANCE_THRESHOLD = 5.0
```

Minimum distance transform value to qualify as a banana center. Filters thin regions.

**Effect of changes:**
- **Higher value**: Only thick banana regions qualify as centers
- **Lower value**: Thinner regions can be banana centers

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
- Touching bananas are separated using edge detection, but heavily overlapping bananas may still merge or fragment (see Segmentation tuning above)
- Very dark or overexposed images may need adjusted value (V) thresholds
