#!/usr/bin/env python3
"""Export YOLO model to NCNN format for faster inference on Raspberry Pi.

NCNN is optimized for ARM devices and provides ~8x faster inference
compared to PyTorch on Raspberry Pi 5.

Usage:
    python scripts/export_ncnn.py [model_name]

Examples:
    python scripts/export_ncnn.py              # Export yolo11n (default)
    python scripts/export_ncnn.py yolo11s      # Export yolo11s
    python scripts/export_ncnn.py yolov8n      # Export yolov8n
"""

import sys
from pathlib import Path


def export_to_ncnn(model_name: str = "yolo11n") -> bool:
    """Export YOLO model to NCNN format.

    Args:
        model_name: Name of the model (e.g., yolo11n, yolo11s, yolov8n)

    Returns:
        True if export succeeded
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Run: pip install ultralytics")
        return False

    pt_path = Path(f"{model_name}.pt")
    ncnn_path = Path(f"{model_name}_ncnn_model")

    # Check if already exported
    if ncnn_path.exists():
        print(f"NCNN model already exists: {ncnn_path}")
        response = input("Re-export? [y/N]: ").strip().lower()
        if response != 'y':
            return True

    print(f"Loading model: {model_name}.pt")
    model = YOLO(str(pt_path))

    print("Exporting to NCNN format...")
    print("This may take a few minutes on first run (downloads NCNN tools)")

    # Export to NCNN - this creates {model_name}_ncnn_model/ directory
    model.export(format="ncnn", imgsz=320)

    if ncnn_path.exists():
        print(f"\nSuccess! NCNN model exported to: {ncnn_path}")
        print(f"\nExpected performance on Raspberry Pi 5:")
        print(f"  - PyTorch: ~800ms inference")
        print(f"  - NCNN:    ~94ms inference (8x faster)")
        print(f"\nThe detector will automatically use the NCNN model.")
        return True
    else:
        print("Export may have failed - check for errors above")
        return False


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "yolo11n"

    print("=" * 50)
    print("  YOLO to NCNN Export for Raspberry Pi")
    print("=" * 50)
    print()

    success = export_to_ncnn(model_name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
