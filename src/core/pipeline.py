"""
Async detection pipeline with tracking interpolation.

Runs object detection in a background thread while the main loop
continues processing frames. Uses velocity-based interpolation to
provide smooth tracking between detection frames.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Queue, Empty, Full
from typing import Dict, List, Optional, Tuple

import numpy as np

from .detector import Detection, ObjectDetector
from .tracker import TargetTracker, TrackedObject, TrackingState

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Frame processing result with tracking data."""
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
    detection_interval: int = 3  # Run detection every N frames
    detection_width: int = 640
    detection_height: int = 640
    queue_size: int = 4

    @classmethod
    def from_dict(cls, config: dict) -> "PipelineConfig":
        det = config.get("detector", {})
        return cls(
            detection_interval=det.get("detection_interval", 3),
        )


class DetectionWorker:
    """
    Background thread for object detection.

    Receives frames via queue, runs detection, and outputs results.
    Uses small queues (size 2) to minimize latency.
    """

    def __init__(self, detector: ObjectDetector, config: PipelineConfig):
        self.detector = detector
        self.config = config

        # Small queues to avoid processing stale frames
        self._input_queue: Queue = Queue(maxsize=2)
        self._output_queue: Queue = Queue(maxsize=2)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @property
    def output_queue(self) -> Queue:
        return self._output_queue

    def start(self) -> None:
        """Start the detection worker thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Detection worker started")

    def stop(self) -> None:
        """Stop the detection worker thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("Detection worker did not terminate cleanly")

    def submit(self, frame: np.ndarray, timestamp: float, frame_id: int,
               frame_size: Tuple[int, int]) -> bool:
        """
        Submit a frame for detection.

        Args:
            frame: BGR image to process
            timestamp: Frame timestamp
            frame_id: Frame sequence number
            frame_size: Original (width, height)

        Returns:
            True if submitted, False if queue is full
        """
        try:
            # Copy frame to avoid race conditions with camera buffer
            self._input_queue.put_nowait((frame.copy(), timestamp, frame_id, frame_size))
            return True
        except Full:
            return False

    def _run(self) -> None:
        """Detection loop running in background thread."""
        while self._running:
            try:
                item = self._input_queue.get(timeout=0.1)
            except Empty:
                continue

            frame, timestamp, frame_id, frame_size = item

            # Validate frame
            if frame is None or frame.size == 0:
                continue

            try:
                # Run detection
                detections = self.detector.detect(frame)
                inference_ms = self.detector._inference_time * 1000

                # Output result, dropping oldest if queue is full
                result = (detections, timestamp, inference_ms, frame_id)
                try:
                    self._output_queue.put_nowait(result)
                except Full:
                    try:
                        self._output_queue.get_nowait()  # Drop oldest
                        self._output_queue.put_nowait(result)
                    except (Empty, Full):
                        pass

            except Exception as e:
                logger.error(f"Detection worker error: {e}")


class TrackingInterpolator:
    """
    Interpolates object positions between detection frames.

    Uses velocity estimation to predict positions, providing
    smooth tracking even when detection runs at lower FPS.
    Thread-safe for concurrent access.
    """

    def __init__(self):
        self._object_data: Dict[int, Tuple] = {}
        self._smoothed_pos: Dict[int, Tuple[float, float]] = {}
        self._last_update_time: float = 0.0
        self._lock = threading.Lock()

        # Position smoothing factor (0.6 = responsive, lower = smoother)
        self._alpha = 0.6

    def update_from_detection(self, objects: Dict[int, TrackedObject],
                               timestamp: float) -> None:
        """Update with fresh detection results."""
        with self._lock:
            new_data: Dict[int, Tuple] = {}

            for obj_id, obj in objects.items():
                # Smooth position updates
                if obj_id in self._smoothed_pos:
                    old_x, old_y = self._smoothed_pos[obj_id]
                    new_x = self._alpha * obj.center[0] + (1 - self._alpha) * old_x
                    new_y = self._alpha * obj.center[1] + (1 - self._alpha) * old_y
                    self._smoothed_pos[obj_id] = (new_x, new_y)
                else:
                    self._smoothed_pos[obj_id] = (float(obj.center[0]), float(obj.center[1]))

                # Store object data as tuple for efficiency
                new_data[obj_id] = (
                    obj.center, obj.bbox, obj.velocity, obj.class_name,
                    obj.confidence, obj.frames_visible, obj.frames_missing, obj.last_update,
                )

            # Remove stale entries
            for k in list(self._smoothed_pos.keys()):
                if k not in objects:
                    del self._smoothed_pos[k]

            self._object_data = new_data
            self._last_update_time = timestamp

    def interpolate(self, timestamp: float) -> Dict[int, TrackedObject]:
        """
        Predict current positions using velocity.

        Returns new TrackedObject instances (no shared state).
        """
        with self._lock:
            if not self._object_data:
                return {}

            # Time since last detection (clamped to avoid huge jumps)
            dt = max(0.0, min(timestamp - self._last_update_time, 0.5))

            result: Dict[int, TrackedObject] = {}

            for obj_id, data in self._object_data.items():
                center, bbox, velocity, class_name, conf, visible, missing, last_update = data

                # Start from smoothed position
                if obj_id in self._smoothed_pos:
                    base_x, base_y = self._smoothed_pos[obj_id]
                else:
                    base_x, base_y = float(center[0]), float(center[1])

                # Predict new position
                vx, vy = velocity
                new_cx = int(base_x + vx * dt)
                new_cy = int(base_y + vy * dt)

                # Compute new bbox (preserve size)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                new_x1 = new_cx - w // 2
                new_y1 = new_cy - h // 2

                result[obj_id] = TrackedObject(
                    object_id=obj_id,
                    class_name=class_name,
                    center=(new_cx, new_cy),
                    bbox=(new_x1, new_y1, new_x1 + w, new_y1 + h),
                    confidence=conf,
                    frames_visible=visible,
                    frames_missing=missing,
                    velocity=velocity,
                    last_update=last_update,
                )

            return result


class Pipeline:
    """
    Async video processing pipeline.

    Coordinates detection (background thread) with tracking (main thread).
    Provides smooth interpolated positions between detection frames.
    """

    def __init__(self, detector: ObjectDetector, tracker: TargetTracker,
                 config: PipelineConfig):
        self.detector = detector
        self.tracker = tracker
        self.config = config

        self._detection_worker = DetectionWorker(detector, config)
        self._interpolator = TrackingInterpolator()

        self._frame_count = 0
        self._last_detection_frame = 0
        self._last_detections: List[Detection] = []
        self._running = False

        # Performance metrics
        self._fps = 0.0
        self._detection_fps = 0.0
        self._inference_ms = 0.0
        self._fps_counter = 0
        self._detection_counter = 0
        self._fps_time = time.time()

    @property
    def fps(self) -> float:
        """Pipeline frame rate (all frames)."""
        return self._fps

    @property
    def detection_fps(self) -> float:
        """Detection rate (detections per second)."""
        return self._detection_fps

    @property
    def inference_ms(self) -> float:
        """Last inference time in milliseconds."""
        return self._inference_ms

    def start(self) -> None:
        """Start the pipeline."""
        self._running = True
        self._detection_worker.start()
        logger.info("Pipeline started")

    def stop(self) -> None:
        """Stop the pipeline."""
        self._running = False
        self._detection_worker.stop()
        logger.info("Pipeline stopped")

    def process_frame(self, frame: np.ndarray) -> FrameData:
        """
        Process a frame through the pipeline.

        Submits frames for detection at configured interval.
        Returns immediately with interpolated tracking data.
        """
        now = time.time()
        self._frame_count += 1

        frame_data = FrameData(
            frame=frame,
            timestamp=now,
            frame_id=self._frame_count,
        )

        # Collect any completed detection results
        self._collect_detection_results()

        # Submit frame for detection if interval reached
        frames_since_detect = self._frame_count - self._last_detection_frame
        if frames_since_detect >= self.config.detection_interval:
            if self._detection_worker.submit(
                frame, now, self._frame_count, (frame.shape[1], frame.shape[0])
            ):
                self._last_detection_frame = self._frame_count

        # Get tracking data with interpolation
        self._populate_tracking_data(frame_data, now)

        # Update FPS counters
        self._update_fps(now)

        return frame_data

    def _collect_detection_results(self) -> bool:
        """
        Drain detection queue and update tracker with latest result.

        Processes all queued results but only uses the newest one.
        """
        latest_result = None

        # Drain queue to get newest result
        while True:
            try:
                result = self._detection_worker.output_queue.get_nowait()
                latest_result = result
            except Empty:
                break

        if latest_result is None:
            return False

        detections, timestamp, inference_ms, frame_id = latest_result

        # Update tracker and stats
        self.tracker.update(detections)
        self._last_detections = detections
        self._inference_ms = inference_ms
        self._detection_counter += 1

        # Update interpolator with new tracking data
        self._interpolator.update_from_detection(self.tracker.all_targets, timestamp)

        return True

    def _populate_tracking_data(self, frame_data: FrameData, now: float) -> None:
        """Fill frame_data with tracking info, using interpolation where available."""
        all_targets = self.tracker.all_targets
        locked = self.tracker.locked_target

        # Get interpolated positions
        predicted = self._interpolator.interpolate(now)

        if predicted:
            # Merge interpolated with tracker data
            merged = {}
            for obj_id, obj in all_targets.items():
                merged[obj_id] = predicted.get(obj_id, obj)
            frame_data.tracked_objects = merged

            # Use interpolated locked target if available
            if locked and locked.object_id in predicted:
                frame_data.locked_target = predicted[locked.object_id]
            else:
                frame_data.locked_target = locked
        else:
            frame_data.tracked_objects = all_targets
            frame_data.locked_target = locked

        frame_data.tracking_state = self.tracker.state
        frame_data.detections = self._last_detections
        frame_data.inference_time_ms = self._inference_ms

    def _update_fps(self, now: float) -> None:
        """Update FPS counters (called once per second)."""
        self._fps_counter += 1
        elapsed = now - self._fps_time

        if elapsed >= 1.0:
            self._fps = self._fps_counter / elapsed
            self._detection_fps = self._detection_counter / elapsed
            self._fps_counter = 0
            self._detection_counter = 0
            self._fps_time = now
