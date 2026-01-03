#!/usr/bin/env python3
"""Export YOLO model to NCNN format for Raspberry Pi inference.

Usage:
    python scripts/export_ncnn.py                    # Export all models/resolutions
    python scripts/export_ncnn.py yolov8n 640       # Export specific model/resolution
    python scripts/export_ncnn.py yolo11n 416       # Export yolo11n at 416px
"""

import sys
from pathlib import Path

MODELS = ["yolov8n", "yolo11n"]
RESOLUTIONS = [640, 416, 320]


def export_model(model_name: str, resolution: int) -> bool:
    """Export a YOLO model to NCNN format at specified resolution."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Run: pip install ultralytics")
        return False

    pt_path = Path(f"models/pt/{model_name}.pt")
    ncnn_dir = Path(f"models/ncnn/{model_name}_{resolution}_ncnn_model")

    if not pt_path.exists():
        print(f"Error: {pt_path} not found")
        return False

    if ncnn_dir.exists():
        print(f"Skipping {ncnn_dir} (already exists)")
        return True

    print(f"Exporting {model_name} @ {resolution}px...")
    model = YOLO(str(pt_path))
    model.export(format="ncnn", imgsz=resolution, half=True)

    # Move exported model to correct location
    temp_dir = Path(f"models/pt/{model_name}_ncnn_model")
    if temp_dir.exists():
        ncnn_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir.rename(ncnn_dir)
        print(f"  -> {ncnn_dir}")
        # Clean up torchscript
        ts_file = Path(f"models/pt/{model_name}.torchscript")
        if ts_file.exists():
            ts_file.unlink()
        return True

    print(f"  Export failed for {model_name} @ {resolution}")
    return False


def main():
    Path("models/ncnn").mkdir(parents=True, exist_ok=True)

    if len(sys.argv) == 3:
        # Export specific model/resolution
        model = sys.argv[1]
        resolution = int(sys.argv[2])
        if model not in MODELS:
            print(f"Error: model must be one of {MODELS}")
            sys.exit(1)
        if resolution not in RESOLUTIONS:
            print(f"Error: resolution must be one of {RESOLUTIONS}")
            sys.exit(1)
        success = export_model(model, resolution)
        sys.exit(0 if success else 1)
    else:
        # Export all combinations
        print("Exporting all model/resolution combinations...")
        for model in MODELS:
            for res in RESOLUTIONS:
                export_model(model, res)
        print("\nDone!")


if __name__ == "__main__":
    main()
