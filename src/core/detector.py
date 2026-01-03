"""YOLO object detection with NCNN backend for Pi 5."""

# Set thread counts before importing NumPy/NCNN (2 threads optimal on Pi 5)
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


# Available models and resolutions
AVAILABLE_MODELS = ["yolov8n", "yolo11n"]
AVAILABLE_RESOLUTIONS = ["640", "416", "320"]


def get_model_path(model: str, resolution: str) -> str:
    """Get the NCNN model path for a given model and resolution."""
    # Ultralytics expects *_ncnn_model suffix for NCNN detection
    return f"models/ncnn/{model}_{resolution}_ncnn_model"


@dataclass
class Detection:
    """Represents a single detection."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    center: tuple  # (cx, cy)

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
    """Detector configuration parameters.

    Performance notes for Raspberry Pi 5 @ 3GHz:
    - 640: Best detection range (~128ms, ~7.8 FPS)
    - 416: Balanced speed/range (~44ms, ~22.7 FPS)
    - 320: Fastest, close-range only (~50ms, ~20 FPS)
    """
    model: str = "yolov8n"  # "yolov8n" or "yolo11n"
    resolution: str = "640"  # "640", "416", or "320"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    target_classes: List[str] = field(default_factory=list)
    detection_interval: int = 3

    @property
    def input_size(self) -> int:
        """Get input size for current resolution."""
        return int(self.resolution) if self.resolution in AVAILABLE_RESOLUTIONS else 640

    @property
    def weights_path(self) -> str:
        """Get absolute model path for current model and resolution."""
        model = self.model if self.model in AVAILABLE_MODELS else "yolov8n"
        resolution = self.resolution if self.resolution in AVAILABLE_RESOLUTIONS else "640"
        rel_path = get_model_path(model, resolution)
        # Return absolute path - Ultralytics needs this for NCNN detection
        return str(Path(rel_path).resolve())

    @classmethod
    def from_dict(cls, config: dict) -> "DetectorConfig":
        """Create config from dictionary."""
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
    """YOLOv8n/YOLO11n detector with NCNN backend."""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self._model = None
        self._class_names: List[str] = []
        self._target_class_ids: set = set()
        self._target_class_ids_array: Optional[np.ndarray] = None  # Pre-converted for vectorized ops
        self._inference_time = 0.0
        self._initialized = False
        self._consecutive_errors = 0
        self._max_consecutive_errors = 10

    @property
    def inference_time_ms(self) -> float:
        """Last inference time in milliseconds."""
        return self._inference_time * 1000

    @property
    def is_initialized(self) -> bool:
        """Check if model is loaded."""
        return self._initialized

    def initialize(self) -> bool:
        """Load and prepare the detection model."""
        try:
            from ultralytics import YOLO
        except ImportError as e:
            logger.error(f"Failed to import ultralytics: {e}")
            logger.error("Install with: pip install ultralytics")
            return False

        try:
            model_path = self.config.weights_path
            if not Path(model_path).exists():
                logger.error(f"Model not found: {model_path}")
                return False

            logger.info(f"Loading NCNN model: {model_path} (resolution: {self.config.resolution})")
            self._model = YOLO(model_path)

            # Get class names from model
            self._class_names = self._model.names
            if isinstance(self._class_names, dict):
                self._class_names = list(self._class_names.values())

            # Build target class ID set and pre-convert to numpy array
            if self.config.target_classes:
                for i, name in enumerate(self._class_names):
                    if name in self.config.target_classes:
                        self._target_class_ids.add(i)
                logger.info(
                    f"Filtering for classes: {self.config.target_classes} "
                    f"(IDs: {self._target_class_ids})"
                )
            else:
                # Detect all classes
                self._target_class_ids = set(range(len(self._class_names)))

            # Pre-convert to numpy array for vectorized filtering
            self._target_class_ids_array = np.array(list(self._target_class_ids), dtype=np.int32)

            # Warm up model with dummy inference
            logger.info("Warming up model...")
            dummy = np.zeros((self.config.input_size, self.config.input_size, 3), dtype=np.uint8)
            self._model.predict(
                dummy,
                imgsz=self.config.input_size,
                conf=self.config.confidence_threshold,
                verbose=False,
                device='cpu',
                max_det=30,
            )
            logger.info("Warmup complete")

            self._initialized = True
            logger.info(f"Detector initialized: {self.config.model} @ {self.config.resolution}px (NCNN)")
            return True

        except RuntimeError as e:
            logger.error(f"Runtime error loading model: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}", exc_info=True)
            return False

    def detect(self, frame: np.ndarray, roi: Optional[tuple] = None) -> List[Detection]:
        """
        Run detection on a frame.

        Args:
            frame: BGR image as numpy array
            roi: Optional (x1, y1, x2, y2) region of interest for faster detection

        Returns:
            List of Detection objects
        """
        if not self._initialized or self._model is None:
            logger.warning("Detector not initialized")
            return []

        start_time = time.perf_counter()

        # ROI optimization: if provided, only process cropped region
        roi_offset = (0, 0)
        if roi is not None:
            x1, y1, x2, y2 = roi
            # Ensure ROI is within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(x1, min(x2, w))
            y2 = max(y1, min(y2, h))

            frame = frame[y1:y2, x1:x2]
            roi_offset = (x1, y1)

        try:
            # Minimal predict call for maximum speed
            results = self._model(
                frame,
                imgsz=self.config.input_size,
                conf=self.config.confidence_threshold,
                iou=self.config.nms_threshold,
                verbose=False,
            )

            self._inference_time = time.perf_counter() - start_time

            detections = []
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                # Get all detections at once
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                # Vectorized filtering by target classes (using pre-converted array)
                if self._target_class_ids_array is not None and len(self._target_class_ids_array) > 0:
                    mask = np.isin(class_ids, self._target_class_ids_array)
                    boxes = boxes[mask]
                    confs = confs[mask]
                    class_ids = class_ids[mask]

                # Apply ROI offset if needed (vectorized)
                if roi_offset != (0, 0):
                    boxes[:, [0, 2]] += roi_offset[0]  # x coords
                    boxes[:, [1, 3]] += roi_offset[1]  # y coords

                # Build detection objects
                for box, conf, class_id in zip(boxes, confs, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Safe class name lookup with bounds check
                    class_id_int = int(class_id)
                    if 0 <= class_id_int < len(self._class_names):
                        class_name = self._class_names[class_id_int]
                    else:
                        class_name = "unknown"

                    detections.append(Detection(
                        class_id=class_id_int,
                        class_name=class_name,
                        confidence=float(conf),
                        bbox=(x1, y1, x2, y2),
                        center=(cx, cy),
                    ))

            self._consecutive_errors = 0  # Reset on success
            return detections

        except Exception as e:
            self._consecutive_errors += 1
            self._inference_time = time.perf_counter() - start_time

            # Log with appropriate severity based on error frequency
            if self._consecutive_errors == 1:
                logger.warning(f"Detection error: {e}")
            elif self._consecutive_errors <= self._max_consecutive_errors:
                logger.error(f"Detection error ({self._consecutive_errors} consecutive): {e}")
            elif self._consecutive_errors == self._max_consecutive_errors + 1:
                logger.critical(
                    f"Detection failing repeatedly ({self._consecutive_errors}x). "
                    "Check model/camera. Suppressing further errors."
                )
            # Suppress logging after max errors to avoid log spam

            return []

    def get_class_name(self, class_id: int) -> str:
        """Get class name by ID."""
        if 0 <= class_id < len(self._class_names):
            return self._class_names[class_id]
        return "unknown"

    def shutdown(self) -> None:
        """Clean up resources."""
        self._model = None
        self._initialized = False
        logger.info("Detector shutdown")
