"""Object detection module using YOLO models."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


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
    """Detector configuration parameters."""
    model: str = "yolov8n"
    weights_path: str = ""
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    target_classes: List[str] = field(default_factory=list)
    input_size: int = 640
    half_precision: bool = True

    @classmethod
    def from_dict(cls, config: dict) -> "DetectorConfig":
        """Create config from dictionary."""
        det = config.get("detector", {})
        return cls(
            model=det.get("model", "yolov8n"),
            weights_path=det.get("weights_path", ""),
            confidence_threshold=det.get("confidence_threshold", 0.5),
            nms_threshold=det.get("nms_threshold", 0.45),
            target_classes=det.get("target_classes", []),
            input_size=det.get("input_size", 640),
            half_precision=det.get("half_precision", True),
        )


class ObjectDetector:
    """
    YOLO-based object detector optimized for Raspberry Pi 5.

    Supports YOLOv8 and YOLOv5 models with configurable precision
    and target class filtering.
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        self._model = None
        self._class_names: List[str] = []
        self._target_class_ids: set = set()
        self._inference_time = 0.0
        self._initialized = False

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

            # Determine model path - prefer NCNN format for speed on ARM
            if self.config.weights_path and Path(self.config.weights_path).exists():
                model_path = self.config.weights_path
                logger.info(f"Loading custom weights: {model_path}")
            else:
                # Check for NCNN format first (much faster on Pi)
                ncnn_path = Path(f"{self.config.model}_ncnn_model")
                pt_path = Path(f"{self.config.model}.pt")

                if ncnn_path.exists():
                    model_path = str(ncnn_path)
                    logger.info(f"Loading NCNN model: {model_path} (optimized for ARM)")
                elif pt_path.exists():
                    model_path = str(pt_path)
                    logger.info(f"Loading PyTorch model: {model_path}")
                    logger.warning("For faster inference on Pi, export to NCNN: yolo export model=yolo11n.pt format=ncnn")
                else:
                    model_path = f"{self.config.model}.pt"
                    logger.info(f"Loading default model: {model_path}")

            # Load model
            self._model = YOLO(model_path)

            # Get class names from model
            self._class_names = self._model.names
            if isinstance(self._class_names, dict):
                self._class_names = list(self._class_names.values())

            # Build target class ID set
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

            # Warm up model with dummy inference
            dummy = np.zeros((self.config.input_size, self.config.input_size, 3), dtype=np.uint8)
            self._model.predict(
                dummy,
                imgsz=self.config.input_size,
                conf=self.config.confidence_threshold,
                half=self.config.half_precision,
                verbose=False,
            )

            self._initialized = True
            logger.info(f"Detector initialized: {self.config.model}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            return False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of Detection objects
        """
        if not self._initialized or self._model is None:
            logger.warning("Detector not initialized")
            return []

        start_time = time.perf_counter()

        try:
            results = self._model.predict(
                frame,
                imgsz=self.config.input_size,
                conf=self.config.confidence_threshold,
                iou=self.config.nms_threshold,
                half=self.config.half_precision,
                verbose=False,
            )

            self._inference_time = time.perf_counter() - start_time

            detections = []
            for result in results:
                if result.boxes is None:
                    continue

                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, conf, class_id in zip(boxes, confs, class_ids):
                    # Filter by target classes
                    if class_id not in self._target_class_ids:
                        continue

                    # Validate class_id bounds
                    if class_id < 0 or class_id >= len(self._class_names):
                        logger.warning(f"Invalid class_id {class_id}, skipping")
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    detection = Detection(
                        class_id=int(class_id),
                        class_name=self._class_names[class_id],
                        confidence=float(conf),
                        bbox=(x1, y1, x2, y2),
                        center=(cx, cy),
                    )
                    detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"Detection error: {e}")
            self._inference_time = time.perf_counter() - start_time
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
