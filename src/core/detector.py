"""Object detection module using YOLO models.

Optimized for Raspberry Pi 5 with support for:
- NCNN format (best for Pi 5 real-time, ~130ms at 640px)
- OpenVINO format (better throughput, worse latency due to batch mode)
- Configurable input resolution (320/416/640)
- ROI-based detection for locked targets
"""

# CRITICAL: Set OpenMP threads BEFORE any imports (NumPy, NCNN, etc.)
import os
os.environ["OMP_NUM_THREADS"] = "4"

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Environment variable to force model re-export
FORCE_EXPORT = os.environ.get("FORCE_EXPORT", "0") == "1"


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

    Performance notes for Raspberry Pi 5:
    - NCNN is best for real-time (~130ms at 640px with FP16)
    - OpenVINO has worse latency due to batch mode (~200ms)
    - input_size 640 needed for distance detection, 480 for balanced, 320 for close-range
    - YOLO11n is recommended (faster than YOLOv8n on ARM)
    - Use detection_interval in pipeline config to skip frames (tracker interpolates)
    """
    model: str = "yolo11n"
    weights_path: str = ""
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    target_classes: List[str] = field(default_factory=list)
    input_size: int = 640  # 640 for distance, 480 balanced, 320 close-range only
    half_precision: bool = False  # FP16 not supported on Pi CPU
    backend: str = "ncnn"  # "ncnn" (best for Pi 5), "openvino", or "pytorch"

    @classmethod
    def from_dict(cls, config: dict) -> "DetectorConfig":
        """Create config from dictionary."""
        det = config.get("detector", {})
        return cls(
            model=det.get("model", "yolo11n"),
            weights_path=det.get("weights_path", ""),
            confidence_threshold=det.get("confidence_threshold", 0.5),
            nms_threshold=det.get("nms_threshold", 0.45),
            target_classes=det.get("target_classes", []),
            input_size=det.get("input_size", 640),
            half_precision=det.get("half_precision", False),
            backend=det.get("backend", "ncnn"),
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

    def _export_model(self, pt_path: Path, backend: str) -> Optional[Path]:
        """Export PyTorch model to optimized format for faster inference.

        Args:
            pt_path: Path to the .pt model file
            backend: Target format ("openvino" or "ncnn")

        Returns:
            Path to exported model, or None if export failed
        """
        try:
            from ultralytics import YOLO
            logger.info(f"Exporting {pt_path} to {backend.upper()} format (this may take a minute)...")

            model = YOLO(str(pt_path))

            if backend == "openvino":
                # OpenVINO is fastest on Pi 5 (~80ms vs 130ms NCNN)
                export_path = model.export(format="openvino")
            else:
                # NCNN fallback, half precision for speed
                export_path = model.export(format="ncnn", half=True)

            if export_path and Path(export_path).exists():
                logger.info(f"{backend.upper()} export successful: {export_path}")
                return Path(export_path)
            else:
                logger.warning(f"{backend.upper()} export completed but path not found")
                return None

        except Exception as e:
            logger.warning(f"{backend.upper()} export failed: {e}")
            logger.info("Falling back to PyTorch model (slower inference)")
            return None

    def initialize(self) -> bool:
        """Load and prepare the detection model.

        Model loading priority:
        1. Custom weights_path if specified
        2. OpenVINO format (fastest on Pi 5, ~80ms)
        3. NCNN format (fallback, ~130ms)
        4. PyTorch format (auto-exports to preferred backend)
        """
        try:
            from ultralytics import YOLO
        except ImportError as e:
            logger.error(f"Failed to import ultralytics: {e}")
            logger.error("Install with: pip install ultralytics")
            return False

        try:
            # Determine model path - prefer optimized formats for speed
            if self.config.weights_path and Path(self.config.weights_path).exists():
                model_path = self.config.weights_path
                logger.info(f"Loading custom weights: {model_path}")
            else:
                backend = self.config.backend.lower()
                pt_path = Path(f"{self.config.model}.pt")

                # Check for pre-exported models
                openvino_path = Path(f"{self.config.model}_openvino_model")
                ncnn_path = Path(f"{self.config.model}_ncnn_model")

                if backend == "openvino" and openvino_path.exists() and not FORCE_EXPORT:
                    model_path = str(openvino_path)
                    logger.info(f"Loading OpenVINO model: {model_path} (fastest on Pi 5)")
                elif backend == "ncnn" and ncnn_path.exists() and not FORCE_EXPORT:
                    model_path = str(ncnn_path)
                    logger.info(f"Loading NCNN model: {model_path}")
                elif pt_path.exists():
                    # Auto-export to preferred backend
                    if backend in ("openvino", "ncnn"):
                        exported = self._export_model(pt_path, backend)
                        if exported:
                            model_path = str(exported)
                        else:
                            model_path = str(pt_path)
                            logger.info(f"Loading PyTorch model: {model_path}")
                    else:
                        model_path = str(pt_path)
                        logger.info(f"Loading PyTorch model: {model_path}")
                else:
                    # Model doesn't exist locally - let ultralytics download it
                    model_path = f"{self.config.model}.pt"
                    logger.info(f"Model not found locally, downloading: {model_path}")

            # Load model (ultralytics auto-downloads if model_path is a known model name)
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
            logger.info("Warming up model...")
            dummy = np.zeros((self.config.input_size, self.config.input_size, 3), dtype=np.uint8)
            self._model.predict(
                dummy,
                imgsz=self.config.input_size,
                conf=self.config.confidence_threshold,
                half=self.config.half_precision,
                verbose=False,
                device='cpu',
                max_det=30,
            )
            logger.info("Warmup complete")

            self._initialized = True
            logger.info(f"Detector initialized: {self.config.model}")
            return True

        except RuntimeError as e:
            logger.error(f"Runtime error loading model: {e}")
            if "CUDA" in str(e):
                logger.error("CUDA error - try setting half_precision=False or use CPU")
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

        # DEBUG: Log frame info first time
        if not hasattr(self, '_frame_logged'):
            logger.info(f"[DETECTOR] Frame: shape={frame.shape}, dtype={frame.dtype}, size={frame.nbytes/1024/1024:.1f}MB, contiguous={frame.flags['C_CONTIGUOUS']}")
            self._frame_logged = True

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

                # Vectorized filtering by target classes
                mask = np.isin(class_ids, list(self._target_class_ids))
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

                    detections.append(Detection(
                        class_id=int(class_id),
                        class_name=self._class_names[class_id],
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
