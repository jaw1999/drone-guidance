"""
YOLO object detection using NCNN backend.

Optimized for Raspberry Pi 5 with configurable model resolution.
Uses Ultralytics YOLO with NCNN export for efficient CPU inference.
"""

# Configure thread count before importing NumPy/NCNN
# 3 threads is optimal for Pi 5's quad-core Cortex-A76
import os
os.environ["OMP_NUM_THREADS"] = "3"
os.environ["OPENBLAS_NUM_THREADS"] = "3"
os.environ["MKL_NUM_THREADS"] = "3"

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Supported model configurations
AVAILABLE_MODELS = ["yolov8n", "yolo11n"]
AVAILABLE_RESOLUTIONS = ["640", "416", "320"]


@dataclass
class Detection:
    """Single object detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) pixel coordinates
    center: tuple  # (cx, cy) center point

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class DetectorConfig:
    """
    Detector configuration.

    Resolution performance on Pi 5 @ 3GHz (NCNN FP16):
      - 640px: ~110ms inference, best detection range
      - 416px: ~44ms inference, balanced
      - 320px: ~35ms inference, close-range only
    """
    model: str = "yolov8n"
    resolution: str = "640"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    target_classes: List[str] = field(default_factory=list)
    detection_interval: int = 3

    @property
    def input_size(self) -> int:
        """Model input size in pixels (square)."""
        return int(self.resolution) if self.resolution in AVAILABLE_RESOLUTIONS else 640

    @property
    def weights_path(self) -> str:
        """Absolute path to NCNN model directory."""
        model = self.model if self.model in AVAILABLE_MODELS else "yolov8n"
        resolution = self.resolution if self.resolution in AVAILABLE_RESOLUTIONS else "640"
        rel_path = f"models/ncnn/{model}_{resolution}_ncnn_model"
        return str(Path(rel_path).resolve())

    @classmethod
    def from_dict(cls, config: dict) -> "DetectorConfig":
        """Create config from YAML dictionary."""
        det = config.get("detector", {})
        return cls(
            model=det.get("model", "yolov8n"),
            resolution=str(det.get("resolution", "640")),
            confidence_threshold=det.get("confidence_threshold", 0.5),
            nms_threshold=det.get("nms_threshold", 0.45),
            target_classes=det.get("target_classes", []),
            detection_interval=det.get("detection_interval", 3),
        )


class ObjectDetector:
    """
    YOLO object detector with NCNN backend.

    Uses Ultralytics YOLO library with pre-exported NCNN models
    for efficient inference on ARM CPUs.
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        self._model = None
        self._class_names: List[str] = []
        self._target_class_ids: set = set()
        self._target_class_ids_array: Optional[np.ndarray] = None
        self._inference_time = 0.0
        self._initialized = False

        # Error tracking to avoid log spam
        self._consecutive_errors = 0
        self._max_consecutive_errors = 10

    @property
    def inference_time_ms(self) -> float:
        """Last inference time in milliseconds."""
        return self._inference_time * 1000

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> bool:
        """Load the YOLO model. Returns True on success."""
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            return False

        model_path = self.config.weights_path
        if not Path(model_path).exists():
            logger.error(f"Model not found: {model_path}")
            return False

        try:
            logger.info(f"Loading model: {model_path}")
            self._model = YOLO(model_path)

            # Extract class names (may be dict or list depending on version)
            self._class_names = self._model.names
            if isinstance(self._class_names, dict):
                self._class_names = list(self._class_names.values())

            # Build set of target class IDs for filtering
            self._setup_class_filter()

            # Warmup inference to initialize NCNN kernels
            self._warmup()

            self._initialized = True
            logger.info(f"Detector ready: {self.config.model} @ {self.config.resolution}px")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}", exc_info=True)
            return False

    def _setup_class_filter(self) -> None:
        """Build target class ID set for filtering detections."""
        if self.config.target_classes:
            for i, name in enumerate(self._class_names):
                if name in self.config.target_classes:
                    self._target_class_ids.add(i)
            logger.info(f"Filtering classes: {self.config.target_classes}")
        else:
            self._target_class_ids = set(range(len(self._class_names)))

        # Pre-convert to numpy for vectorized filtering
        self._target_class_ids_array = np.array(
            list(self._target_class_ids), dtype=np.int32
        )

    def _warmup(self) -> None:
        """Run dummy inference to initialize NCNN kernels."""
        logger.info("Warming up model...")
        dummy = np.zeros(
            (self.config.input_size, self.config.input_size, 3),
            dtype=np.uint8
        )
        self._model.predict(
            dummy,
            imgsz=self.config.input_size,
            conf=self.config.confidence_threshold,
            verbose=False,
            device="cpu",
        )

    def detect(self, frame: np.ndarray, roi: Optional[tuple] = None) -> List[Detection]:
        """
        Run object detection on a frame.

        Args:
            frame: BGR image (numpy array)
            roi: Optional region of interest (x1, y1, x2, y2) to crop before detection

        Returns:
            List of Detection objects
        """
        if not self._initialized:
            return []

        start = time.perf_counter()

        # Crop to ROI if specified
        roi_offset = (0, 0)
        if roi is not None:
            frame, roi_offset = self._crop_to_roi(frame, roi)

        try:
            results = self._model(
                frame,
                imgsz=self.config.input_size,
                conf=self.config.confidence_threshold,
                iou=self.config.nms_threshold,
                verbose=False,
            )
            self._inference_time = time.perf_counter() - start

            detections = self._parse_results(results, roi_offset)
            self._consecutive_errors = 0
            return detections

        except Exception as e:
            self._handle_error(e)
            self._inference_time = time.perf_counter() - start
            return []

    def _crop_to_roi(self, frame: np.ndarray, roi: tuple) -> tuple:
        """Crop frame to ROI bounds. Returns (cropped_frame, offset)."""
        x1, y1, x2, y2 = roi
        h, w = frame.shape[:2]

        # Clamp to frame bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(x1, min(x2, w))
        y2 = max(y1, min(y2, h))

        return frame[y1:y2, x1:x2], (x1, y1)

    def _parse_results(self, results, roi_offset: tuple) -> List[Detection]:
        """Convert YOLO results to Detection objects."""
        detections = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            # Extract arrays from result
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            # Filter by target classes
            if len(self._target_class_ids_array) > 0:
                mask = np.isin(class_ids, self._target_class_ids_array)
                boxes, confs, class_ids = boxes[mask], confs[mask], class_ids[mask]

            # Offset coordinates if ROI was used
            if roi_offset != (0, 0):
                boxes[:, [0, 2]] += roi_offset[0]
                boxes[:, [1, 3]] += roi_offset[1]

            # Build Detection objects
            for box, conf, cid in zip(boxes, confs, class_ids):
                x1, y1, x2, y2 = map(int, box)
                detections.append(Detection(
                    class_id=int(cid),
                    class_name=self._class_names[cid] if cid < len(self._class_names) else "unknown",
                    confidence=float(conf),
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                ))

        return detections

    def _handle_error(self, error: Exception) -> None:
        """Log detection errors with progressive severity."""
        self._consecutive_errors += 1

        if self._consecutive_errors == 1:
            logger.warning(f"Detection error: {error}")
        elif self._consecutive_errors <= self._max_consecutive_errors:
            logger.error(f"Detection error ({self._consecutive_errors}x): {error}")
        elif self._consecutive_errors == self._max_consecutive_errors + 1:
            logger.critical("Detection failing repeatedly. Suppressing further errors.")

    def get_class_name(self, class_id: int) -> str:
        """Get class name by ID."""
        if 0 <= class_id < len(self._class_names):
            return self._class_names[class_id]
        return "unknown"

    def shutdown(self) -> None:
        """Release model resources."""
        self._model = None
        self._initialized = False
        logger.info("Detector shutdown")
