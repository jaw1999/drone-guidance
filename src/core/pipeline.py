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

        self._input_queue: Queue[Tuple[np.ndarray, float, int, Tuple[int, int]]] = Queue(maxsize=2)
        self._output_queue: Queue[Tuple[List[Detection], float, float, int]] = Queue(maxsize=2)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # (Removed _detect_buffer - no longer needed since we pass full frames)

    @property
    def output_queue(self) -> Queue[Tuple[List[Detection], float, float, int]]:
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
            if self._thread.is_alive():
                logger.warning("Detection worker thread did not terminate cleanly")
            else:
                logger.info("Detection worker stopped")

    def submit(self, frame: np.ndarray, timestamp: float, frame_id: int, orig_size: Tuple[int, int]) -> bool:
        """Submit frame for detection. Resize happens in worker thread."""
        try:
            self._input_queue.put_nowait((frame, timestamp, frame_id, orig_size))
            return True
        except Full:
            return False

    def _run(self) -> None:
        """Main detection loop."""
        det_w, det_h = self.config.detection_width, self.config.detection_height

        # (Buffer pre-allocation removed - we pass full frames now)

        while self._running:
            try:
                item = self._input_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                unpack_start = time.perf_counter()
                frame, timestamp, frame_id, orig_size = item

                if frame is None or frame.size == 0:
                    logger.warning("Received empty frame, skipping detection")
                    continue

                orig_w, orig_h = orig_size if orig_size else (frame.shape[1], frame.shape[0])
                if orig_h == 0 or orig_w == 0:
                    logger.warning("Frame has zero dimensions, skipping")
                    continue

                unpack_ms = (time.perf_counter() - unpack_start) * 1000

                # Run detection on FULL frame (YOLO handles resize internally)
                detect_start = time.perf_counter()
                detections = self.detector.detect(frame)
                detect_ms = (time.perf_counter() - detect_start) * 1000

                # Use actual model inference time (more accurate)
                inference_ms = self.detector._inference_time * 1000

                # Build output (detections already in correct coordinates)
                package_start = time.perf_counter()
                result = (detections, timestamp, inference_ms, frame_id)
                package_ms = (time.perf_counter() - package_start) * 1000

                total_worker_ms = (time.perf_counter() - unpack_start) * 1000

                # DEBUG: Log timing breakdown for first few frames
                if not hasattr(self, '_timing_logged'):
                    self._timing_logged = 0
                if self._timing_logged < 3:
                    logger.info(f"[TIMING] Total={total_worker_ms:.0f}ms: Detect={detect_ms:.0f}ms (model={inference_ms:.0f}ms), Unpack={unpack_ms:.1f}ms, Package={package_ms:.1f}ms")
                    self._timing_logged += 1
                try:
                    self._output_queue.put_nowait(result)
                except Full:
                    # Drop oldest result to make room
                    try:
                        self._output_queue.get_nowait()
                    except Empty:
                        pass
                    try:
                        self._output_queue.put_nowait(result)
                    except Full:
                        pass

            except Exception as e:
                logger.error(f"Detection worker error: {e}", exc_info=True)


class TrackingInterpolator:
    """Interpolates tracking between detection frames using velocity.

    Optimized for smooth tracking:
    - Always predicts to current time for consistent motion
    - Uses smoothed positions to reduce jitter
    - Stores lightweight tuples instead of deep copies for performance
    """

    def __init__(self):
        # Store essential data as tuples: (center, bbox, velocity, class_name, confidence, frames_visible, frames_missing, last_update)
        self._object_data: Dict[int, Tuple] = {}
        self._smoothed_pos: Dict[int, Tuple[float, float]] = {}
        self._last_update_time: float = 0.0
        self._lock = threading.Lock()

    def update_from_detection(
        self,
        objects: Dict[int, TrackedObject],
        timestamp: float,
    ) -> None:
        """Update with fresh detection results (thread-safe)."""
        with self._lock:
            # Smooth position updates to reduce snap-to behavior
            alpha = 0.6  # Higher = more responsive, lower = smoother
            new_data: Dict[int, Tuple] = {}

            for obj_id, obj in objects.items():
                if obj_id in self._smoothed_pos:
                    old_x, old_y = self._smoothed_pos[obj_id]
                    new_x = alpha * obj.center[0] + (1 - alpha) * old_x
                    new_y = alpha * obj.center[1] + (1 - alpha) * old_y
                    self._smoothed_pos[obj_id] = (new_x, new_y)
                else:
                    self._smoothed_pos[obj_id] = (float(obj.center[0]), float(obj.center[1]))

                # Store as lightweight tuple instead of deep copy
                new_data[obj_id] = (
                    obj.center,
                    obj.bbox,
                    obj.velocity,
                    obj.class_name,
                    obj.confidence,
                    obj.frames_visible,
                    obj.frames_missing,
                    obj.last_update,
                )

            # Remove stale smoothed positions
            stale_smooth = [k for k in self._smoothed_pos if k not in objects]
            for k in stale_smooth:
                del self._smoothed_pos[k]

            self._object_data = new_data
            self._last_update_time = timestamp

    def interpolate(
        self,
        timestamp: float,
    ) -> Dict[int, TrackedObject]:
        """Predict current positions based on velocity (thread-safe).

        Returns new TrackedObject instances to prevent race conditions.
        """
        with self._lock:
            if not self._object_data:
                return {}

            dt = timestamp - self._last_update_time
            # Clamp dt to reasonable range
            dt = max(0.0, min(dt, 0.5))

            result: Dict[int, TrackedObject] = {}

            for obj_id, data in self._object_data.items():
                # Unpack tuple: (center, bbox, velocity, class_name, confidence, frames_visible, frames_missing, last_update)
                center, bbox, velocity, class_name, confidence, frames_visible, frames_missing, last_update = data

                # Start from smoothed position for stability
                if obj_id in self._smoothed_pos:
                    base_x, base_y = self._smoothed_pos[obj_id]
                else:
                    base_x, base_y = float(center[0]), float(center[1])

                # Predict new center
                vx, vy = velocity
                new_cx = int(base_x + vx * dt)
                new_cy = int(base_y + vy * dt)

                # Predict new bbox
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                half_w = w >> 1
                half_h = h >> 1
                new_x1 = new_cx - half_w
                new_y1 = new_cy - half_h

                # Create new TrackedObject for each call (no shared state)
                result[obj_id] = TrackedObject(
                    object_id=obj_id,
                    class_name=class_name,
                    center=(new_cx, new_cy),
                    bbox=(new_x1, new_y1, new_x1 + w, new_y1 + h),
                    confidence=confidence,
                    frames_visible=frames_visible,
                    frames_missing=frames_missing,
                    velocity=velocity,
                    last_update=last_update,
                )

            return result


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
            # Submit full frame - resize happens in worker thread (off main thread)
            if self._detection_worker.submit(
                frame, now, self._frame_count, (frame.shape[1], frame.shape[0])
            ):
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

        # Unpack tuple: (detections, timestamp, inference_ms, frame_id)
        detections, timestamp, inference_ms, frame_id = result

        # Update tracker with new detections
        self.tracker.update(detections)
        self._last_detections = detections
        self._inference_ms = inference_ms

        # Update interpolator with fresh tracking data
        self._interpolator.update_from_detection(
            self.tracker.all_targets,
            timestamp,
        )
