"""CLI entry point for BananaVision."""

import argparse
import os
import sys

import cv2

from banana_vision import analyze_image
from banana_vision.analyzer import create_debug_visualization


def format_results(image_path: str, results: dict) -> str:
    """Format analysis results for display."""
    mode = results.get("mode", "pixel")

    if mode == "banana":
        lines = [
            f"Analyzing: {os.path.basename(image_path)}",
            "=" * 50,
            "Banana Ripeness Analysis (Per-Banana)",
            "-" * 50,
            f"Green:                {results['green_percent']:>5.1f}%  ({results['green_count']} bananas)",
            f"Yellow (no spots):    {results['yellow_clean_percent']:>5.1f}%  ({results['yellow_clean_count']} bananas)",
            f"Yellow (spotted):     {results['yellow_spotted_percent']:>5.1f}%  ({results['yellow_spotted_count']} bananas)",
            "-" * 50,
            f"Total bananas detected: {results['total_bananas']}",
            "=" * 50,
        ]
    else:
        lines = [
            f"Analyzing: {os.path.basename(image_path)}",
            "=" * 45,
            "Banana Ripeness Analysis",
            "-" * 45,
            f"Green:              {results['green_percent']:>6.1f}%",
            f"Yellow (no spots):  {results['yellow_clean_percent']:>6.1f}%",
            f"Yellow (spotted):   {results['yellow_spotted_percent']:>6.1f}%",
            "-" * 45,
            f"Total banana area: {results['total_banana_pixels']:,} pixels",
            "=" * 45,
        ]

    return "\n".join(lines)


def show_debug_windows(image_path: str, mode: str = "banana") -> None:
    """Display debug visualization windows."""
    try:
        visualizations = create_debug_visualization(image_path, mode=mode)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return

    window_names = {
        "original": "Original with Contours",
        "green_mask": "Green Mask",
        "yellow_mask": "Yellow Mask",
        "spot_mask": "Spot Detection",
    }

    # Add category views for banana mode with counts in titles
    if mode == "banana" and "green_bananas" in visualizations:
        green_count = visualizations.get("green_count", 0)
        yellow_clean_count = visualizations.get("yellow_clean_count", 0)
        yellow_spotted_count = visualizations.get("yellow_spotted_count", 0)

        window_names["green_bananas"] = f"Green Bananas ({green_count})"
        window_names["yellow_clean"] = f"Yellow Clean ({yellow_clean_count})"
        window_names["yellow_spotted"] = f"Yellow Spotted ({yellow_spotted_count})"

    for key, title in window_names.items():
        if key in visualizations:
            cv2.imshow(title, visualizations[key])

    print("\nDebug windows opened. Press any key or close windows to continue.")

    # Poll for key press or window closure (X button)
    while True:
        key = cv2.waitKey(100)  # 100ms timeout
        if key != -1:  # Key was pressed
            break
        # Check if any window was closed
        all_closed = True
        for title in window_names.values():
            try:
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
                    all_closed = False
                    break
            except cv2.error:
                pass  # Window doesn't exist
        if all_closed:
            break

    cv2.destroyAllWindows()


def get_image_files(paths: list[str]) -> list[str]:
    """
    Expand paths to a list of image files.

    Handles individual files and directories.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = []

    for path in paths:
        if os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in image_extensions:
                image_files.append(path)
            else:
                print(f"Warning: Skipping non-image file: {path}", file=sys.stderr)
        elif os.path.isdir(path):
            for filename in os.listdir(path):
                ext = os.path.splitext(filename)[1].lower()
                if ext in image_extensions:
                    image_files.append(os.path.join(path, filename))
        else:
            print(f"Warning: Path not found: {path}", file=sys.stderr)

    return sorted(image_files)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze banana ripeness from photographs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py image.jpg                  Analyze a single image (banana mode)
  python main.py image.jpg --mode=banana    Per-banana classification
  python main.py image.jpg --mode=pixel     Legacy per-pixel classification
  python main.py img1.jpg img2.jpg          Analyze multiple images
  python main.py path/to/folder/            Analyze all images in a directory
  python main.py image.jpg --debug          Show debug visualization
        """,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Image file(s) or directory to analyze",
    )
    parser.add_argument(
        "--mode",
        choices=["banana", "pixel"],
        default="banana",
        help="Analysis mode: 'banana' for per-banana classification (default), "
             "'pixel' for legacy per-pixel classification",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug visualization with intermediate masks",
    )

    args = parser.parse_args()

    image_files = get_image_files(args.paths)

    if not image_files:
        print("Error: No image files found", file=sys.stderr)
        return 1

    for image_path in image_files:
        try:
            results = analyze_image(image_path, mode=args.mode)
            print(format_results(image_path, results))
            print()

            if args.debug:
                show_debug_windows(image_path, mode=args.mode)

        except ValueError as e:
            print(f"Error processing {image_path}: {e}", file=sys.stderr)
            continue

    return 0


if __name__ == "__main__":
    sys.exit(main())
