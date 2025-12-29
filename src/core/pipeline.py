"""Async processing pipeline for low-latency video processing.

This module provides an asynchronous detection pipeline that separates
frame capture from detection to maximize throughput. Features include:
- Background detection thread with queue-based communication
- Velocity-based interpolation for smooth tracking between detections
- ROI-based detection for improved performance when tracking locked targets

Classes:
    FrameData: Container for frame and metadata.
    PipelineConfig: Configuration for the pipeline.
    DetectionWorker: Background thread for detection.
    TrackingInterpolator: Smooths tracking between detection frames.
    Pipeline: Main async processing pipeline.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Queue, Empty, Full
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .detector import Detection, ObjectDetector
from .tracker import TargetTracker, TrackedObject, TrackingState

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Container for frame and associated metadata."""
    frame: np.ndarray
    timestamp: float
    frame_id: int
    detections: List[Detection] = field(default_factory=list)
    tracked_objects: Dict[int, TrackedObject] = field(default_factory=dict)
    locked_target: Optional[TrackedObject] = None
    tracking_state: TrackingState = TrackingState.SEARCHING
    inference_time_ms: float = 0.0


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    detection_interval: int = 3
    detection_width: int = 640
    detection_height: int = 640
    queue_size: int = 4

    @classmethod
    def from_dict(cls, config: dict) -> "PipelineConfig":
        det = config.get("detector", {})
        det_res = det.get("detection_resolution", {})
        return cls(
            detection_interval=det.get("detection_interval", 3),
            detection_width=det_res.get("width", 640),
            detection_height=det_res.get("height", 640),
        )


class DetectionWorker:
    """Background thread for running object detection."""

    def __init__(
        self,
        detector: ObjectDetector,
        config: PipelineConfig,
    ):
        self.detector = detector
        self.config = config

        self._input_queue: Queue[Tuple[FrameData, Tuple[int, int]]] = Queue(maxsize=4)
        self._output_queue: Queue[FrameData] = Queue(maxsize=4)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @property
    def output_queue(self) -> Queue[FrameData]:
        return self._output_queue

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Detection worker started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Detection worker stopped")

    def submit(self, frame_data: FrameData, orig_size: Tuple[int, int]) -> bool:
        """Submit frame for detection."""
        try:
            self._input_queue.put_nowait((frame_data, orig_size))
            return True
        except Full:
            return False

    def _run(self) -> None:
        """Main detection loop."""
        while self._running:
            try:
                item = self._input_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                frame_data, orig_size = item
                frame = frame_data.frame
                if frame is None or frame.size == 0:
                    logger.warning("Received empty frame, skipping detection")
                    continue

                det_h, det_w = frame.shape[:2]
                if det_h == 0 or det_w == 0:
                    logger.warning("Frame has zero dimensions, skipping")
                    continue

                orig_h, orig_w = orig_size if orig_size else (det_h, det_w)

                # Calculate scale factors
                scale_x = orig_w / max(1, det_w)
                scale_y = orig_h / max(1, det_h)

                # Run detection
                start = time.perf_counter()
                detections = self.detector.detect(frame)
                frame_data.inference_time_ms = (time.perf_counter() - start) * 1000

                # Scale detections back to original resolution
                scaled_detections = []
                for det in detections:
                    x1, y1, x2, y2 = det.bbox
                    sx1 = int(x1 * scale_x)
                    sy1 = int(y1 * scale_y)
                    sx2 = int(x2 * scale_x)
                    sy2 = int(y2 * scale_y)
                    cx = (sx1 + sx2) // 2
                    cy = (sy1 + sy2) // 2

                    scaled_detections.append(Detection(
                        class_id=det.class_id,
                        class_name=det.class_name,
                        confidence=det.confidence,
                        bbox=(sx1, sy1, sx2, sy2),
                        center=(cx, cy),
                    ))

                frame_data.detections = scaled_detections

                try:
                    self._output_queue.put_nowait(frame_data)
                except Full:
                    # Drop oldest result to make room
                    try:
                        self._output_queue.get_nowait()
                        self._output_queue.put_nowait(frame_data)
                    except Empty:
                        pass

            except Exception as e:
                logger.error(f"Detection worker error: {e}", exc_info=True)


class TrackingInterpolator:
    """Interpolates tracking between detection frames using velocity.

    Optimized for smooth tracking:
    - Always predicts to current time for consistent motion
    - Uses smoothed positions to reduce jitter
    - Reuses objects to minimize allocations
    """

    def __init__(self):
        self._last_objects: Dict[int, TrackedObject] = {}
        self._predicted: Dict[int, TrackedObject] = {}  # Reusable cache
        self._smoothed_pos: Dict[int, Tuple[float, float]] = {}  # Smoothed centers
        self._last_update_time: float = 0.0

    def update_from_detection(
        self,
        objects: Dict[int, TrackedObject],
        timestamp: float,
    ) -> None:
        """Update with fresh detection results."""
        # Smooth position updates to reduce snap-to behavior
        alpha = 0.6  # Higher = more responsive, lower = smoother
        for obj_id, obj in objects.items():
            if obj_id in self._smoothed_pos:
                old_x, old_y = self._smoothed_pos[obj_id]
                new_x = alpha * obj.center[0] + (1 - alpha) * old_x
                new_y = alpha * obj.center[1] + (1 - alpha) * old_y
                self._smoothed_pos[obj_id] = (new_x, new_y)
            else:
                self._smoothed_pos[obj_id] = (float(obj.center[0]), float(obj.center[1]))

        # Remove stale smoothed positions
        stale_smooth = [k for k in self._smoothed_pos if k not in objects]
        for k in stale_smooth:
            del self._smoothed_pos[k]

        self._last_objects = objects
        self._last_update_time = timestamp

    def interpolate(
        self,
        timestamp: float,
    ) -> Dict[int, TrackedObject]:
        """Predict current positions based on velocity.

        Always returns interpolated positions for smooth motion.
        """
        if not self._last_objects:
            return {}

        dt = timestamp - self._last_update_time
        # Clamp dt to reasonable range
        dt = max(0.0, min(dt, 0.5))

        for obj_id, obj in self._last_objects.items():
            # Start from smoothed position for stability
            if obj_id in self._smoothed_pos:
                base_x, base_y = self._smoothed_pos[obj_id]
            else:
                base_x, base_y = float(obj.center[0]), float(obj.center[1])

            # Predict new center
            vx, vy = obj.velocity
            new_cx = int(base_x + vx * dt)
            new_cy = int(base_y + vy * dt)

            # Predict new bbox
            w = obj.bbox[2] - obj.bbox[0]
            h = obj.bbox[3] - obj.bbox[1]
            half_w = w >> 1
            half_h = h >> 1
            new_x1 = new_cx - half_w
            new_y1 = new_cy - half_h

            # Reuse existing TrackedObject if possible
            if obj_id in self._predicted:
                pred = self._predicted[obj_id]
                pred.center = (new_cx, new_cy)
                pred.bbox = (new_x1, new_y1, new_x1 + w, new_y1 + h)
                pred.confidence = obj.confidence
            else:
                self._predicted[obj_id] = TrackedObject(
                    object_id=obj_id,
                    class_name=obj.class_name,
                    center=(new_cx, new_cy),
                    bbox=(new_x1, new_y1, new_x1 + w, new_y1 + h),
                    confidence=obj.confidence,
                    frames_visible=obj.frames_visible,
                    frames_missing=obj.frames_missing,
                    velocity=obj.velocity,
                    last_update=obj.last_update,
                )

        # Remove stale predictions
        stale = [k for k in self._predicted if k not in self._last_objects]
        for k in stale:
            del self._predicted[k]

        return self._predicted


class Pipeline:
    """
    Async video processing pipeline.

    Separates capture, detection, and output into concurrent stages
    for maximum throughput on limited hardware.
    """

    def __init__(
        self,
        detector: ObjectDetector,
        tracker: TargetTracker,
        config: PipelineConfig,
    ):
        self.detector = detector
        self.tracker = tracker
        self.config = config

        self._detection_worker = DetectionWorker(detector, config)
        self._interpolator = TrackingInterpolator()

        self._frame_count = 0
        self._last_detection_frame = 0
        self._last_detections: List[Detection] = []
        self._running = False

        # Performance stats
        self._fps = 0.0
        self._inference_ms = 0.0
        self._fps_counter = 0
        self._fps_time = time.time()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def inference_ms(self) -> float:
        return self._inference_ms

    def start(self) -> None:
        self._running = True
        self._detection_worker.start()
        logger.info("Pipeline started")

    def stop(self) -> None:
        self._running = False
        self._detection_worker.stop()
        logger.info("Pipeline stopped")

    def process_frame(self, frame: np.ndarray) -> FrameData:
        """
        Process a frame through the pipeline.

        Returns immediately with interpolated tracking if not a detection frame.
        Detection runs async in background.
        """
        now = time.time()
        self._frame_count += 1

        frame_data = FrameData(
            frame=frame,
            timestamp=now,
            frame_id=self._frame_count,
        )

        # Check for completed detection results
        self._collect_detection_results()

        # Decide if we should run detection this frame
        frames_since_detect = self._frame_count - self._last_detection_frame
        should_detect = frames_since_detect >= self.config.detection_interval

        if should_detect:
            # Submit for async detection - downscale now to reduce memory copy
            det_w, det_h = self.config.detection_width, self.config.detection_height
            detect_frame = cv2.resize(frame, (det_w, det_h))
            if self._detection_worker.submit(FrameData(
                frame=detect_frame,
                timestamp=now,
                frame_id=self._frame_count,
            ), frame.shape[:2]):
                self._last_detection_frame = self._frame_count

        # Always use interpolated positions for smooth motion
        # This prevents "snapping" when new detections arrive
        predicted = self._interpolator.interpolate(now)
        if predicted:
            frame_data.tracked_objects = predicted
            # Get interpolated locked target position
            locked = self.tracker.locked_target
            if locked and locked.object_id in predicted:
                frame_data.locked_target = predicted[locked.object_id]
            else:
                frame_data.locked_target = locked
        else:
            # No predictions yet, use raw tracker data
            frame_data.tracked_objects = self.tracker.all_targets
            frame_data.locked_target = self.tracker.locked_target

        frame_data.tracking_state = self.tracker.state
        frame_data.detections = self._last_detections
        frame_data.inference_time_ms = self._inference_ms

        # Update FPS
        self._fps_counter += 1
        elapsed = now - self._fps_time
        if elapsed >= 1.0:
            self._fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_time = now

        return frame_data

    def _collect_detection_results(self) -> None:
        """Check for and process completed detection results."""
        try:
            result = self._detection_worker.output_queue.get_nowait()
        except Empty:
            return

        # Update tracker with new detections
        self.tracker.update(result.detections)
        self._last_detections = result.detections
        self._inference_ms = result.inference_time_ms

        # Update interpolator with fresh tracking data
        self._interpolator.update_from_detection(
            self.tracker.all_targets,
            result.timestamp,
        )
