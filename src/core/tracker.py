"""Target tracking module with lock-on capability."""

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .detector import Detection

logger = logging.getLogger(__name__)

# Tracking constants
MIN_VELOCITY_DT = 0.01  # Minimum time delta for velocity calculation (seconds)
MAX_VELOCITY_DT = 1.0   # Maximum time delta before considering update stale
VELOCITY_SMOOTHING_ALPHA = 0.3  # Exponential smoothing factor for velocity
REACQUIRE_RADIUS_MULTIPLIER = 3  # Multiplier for max_distance when reacquiring
FRAME_TIME_ESTIMATE = 0.033  # Assumed frame time for prediction (~30fps)
MIN_MATCH_SCORE = 0.5  # Minimum score for target reacquisition match


class TrackingState(Enum):
    """Target tracking states."""
    SEARCHING = "searching"      # Looking for targets
    ACQUIRING = "acquiring"      # Target found, confirming lock
    LOCKED = "locked"           # Target locked, tracking active
    LOST = "lost"               # Target lost, searching


@dataclass
class TrackedObject:
    """Represents a tracked object."""
    object_id: int
    class_name: str
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    confidence: float
    frames_visible: int = 1
    frames_missing: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)
    last_update: float = 0.0

    def predict_position(self, dt: float = 1.0) -> Tuple[int, int]:
        """Predict next position based on velocity."""
        px = int(self.center[0] + self.velocity[0] * dt)
        py = int(self.center[1] + self.velocity[1] * dt)
        return (px, py)


@dataclass
class TrackerConfig:
    """Tracker configuration parameters."""
    algorithm: str = "centroid"
    max_disappeared: int = 30
    max_distance: int = 100
    min_confidence: float = 0.6
    frames_to_lock: int = 5
    frames_to_unlock: int = 15

    @classmethod
    def from_dict(cls, config: dict) -> "TrackerConfig":
        """Create config from dictionary."""
        tracker = config.get("tracker", {})
        lock_on = tracker.get("lock_on", {})
        return cls(
            algorithm=tracker.get("algorithm", "centroid"),
            max_disappeared=tracker.get("max_disappeared", 30),
            max_distance=tracker.get("max_distance", 100),
            min_confidence=lock_on.get("min_confidence", 0.6),
            frames_to_lock=lock_on.get("frames_to_lock", 5),
            frames_to_unlock=lock_on.get("frames_to_unlock", 15),
        )


class CentroidTracker:
    """Simple centroid-based object tracker."""

    def __init__(self, max_disappeared: int = 30, max_distance: int = 100):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self._next_object_id = 0
        self._objects: OrderedDict[int, TrackedObject] = OrderedDict()
        self._disappeared: Dict[int, int] = {}

    @property
    def objects(self) -> Dict[int, TrackedObject]:
        """Get all tracked objects."""
        return dict(self._objects)

    def update(self, detections: List[Detection]) -> Dict[int, TrackedObject]:
        """
        Update tracker with new detections.

        Args:
            detections: List of Detection objects from detector

        Returns:
            Dictionary of object_id -> TrackedObject
        """
        current_time = time.time()

        # No detections - mark all as disappeared
        if len(detections) == 0:
            # Collect IDs to deregister (can't modify dict during iteration)
            to_deregister = []
            for object_id in self._disappeared:
                self._disappeared[object_id] += 1
                if object_id in self._objects:
                    self._objects[object_id].frames_missing += 1
                if self._disappeared[object_id] > self.max_disappeared:
                    to_deregister.append(object_id)
            for object_id in to_deregister:
                self._deregister(object_id)
            return self.objects

        # No existing objects - register all new detections
        if len(self._objects) == 0:
            for detection in detections:
                self._register(detection, current_time)
        else:
            # Match existing objects to new detections
            object_ids = tuple(self._objects.keys())  # tuple is lighter than list

            # Fast distance calculation without scipy
            # Use squared distances to avoid sqrt
            n_obj = len(object_ids)
            n_det = len(detections)
            D = np.empty((n_obj, n_det), dtype=np.float32)

            for i, oid in enumerate(object_ids):
                ox, oy = self._objects[oid].center
                for j, det in enumerate(detections):
                    dx = det.center[0] - ox
                    dy = det.center[1] - oy
                    D[i, j] = dx * dx + dy * dy  # Squared distance

            max_dist_sq = self.max_distance * self.max_distance

            # Find minimum distance matches
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > max_dist_sq:
                    continue

                object_id = object_ids[row]
                detection = detections[col]

                # Update object with new detection
                old_obj = self._objects[object_id]
                old_center = old_obj.center
                new_center = detection.center

                # Calculate velocity with exponential smoothing for stability
                dt = current_time - old_obj.last_update
                # Clamp dt to avoid velocity spikes from very small intervals
                if MIN_VELOCITY_DT < dt < MAX_VELOCITY_DT:
                    raw_vx = (new_center[0] - old_center[0]) / dt
                    raw_vy = (new_center[1] - old_center[1]) / dt
                    # Smooth velocity for stability
                    alpha = VELOCITY_SMOOTHING_ALPHA
                    old_vx, old_vy = old_obj.velocity
                    vx = alpha * raw_vx + (1 - alpha) * old_vx
                    vy = alpha * raw_vy + (1 - alpha) * old_vy
                else:
                    vx, vy = old_obj.velocity  # Keep previous velocity

                self._objects[object_id] = TrackedObject(
                    object_id=object_id,
                    class_name=detection.class_name,
                    center=new_center,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    frames_visible=old_obj.frames_visible + 1,
                    frames_missing=0,
                    velocity=(vx, vy),
                    last_update=current_time,
                )
                self._disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Handle unmatched existing objects
            unused_rows = set(range(len(object_ids))) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self._disappeared[object_id] += 1
                self._objects[object_id].frames_missing += 1

                if self._disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)

            # Register new unmatched detections
            unused_cols = set(range(len(detections))) - used_cols
            for col in unused_cols:
                self._register(detections[col], current_time)

        return self.objects

    def _register(self, detection: Detection, timestamp: float) -> int:
        """Register a new object."""
        object_id = self._next_object_id
        self._objects[object_id] = TrackedObject(
            object_id=object_id,
            class_name=detection.class_name,
            center=detection.center,
            bbox=detection.bbox,
            confidence=detection.confidence,
            frames_visible=1,
            frames_missing=0,
            velocity=(0.0, 0.0),
            last_update=timestamp,
        )
        self._disappeared[object_id] = 0
        self._next_object_id += 1
        return object_id

    def _deregister(self, object_id: int) -> None:
        """Deregister an object."""
        del self._objects[object_id]
        del self._disappeared[object_id]

    def reset(self) -> None:
        """Reset tracker state."""
        self._objects.clear()
        self._disappeared.clear()
        self._next_object_id = 0


class TargetTracker:
    """
    High-level target tracker with lock-on capability.

    Manages target selection, lock-on state, and provides
    error signals for the PID controller.
    """

    def __init__(self, config: TrackerConfig, frame_size: Tuple[int, int]):
        self.config = config
        # Validate frame size to prevent division by zero
        self.frame_width = max(1, frame_size[0])
        self.frame_height = max(1, frame_size[1])
        self.frame_center = (self.frame_width // 2, self.frame_height // 2)

        # Initialize centroid tracker
        self._tracker = CentroidTracker(
            max_disappeared=config.max_disappeared,
            max_distance=config.max_distance,
        )

        # Lock-on state
        self._state = TrackingState.SEARCHING
        self._locked_target_id: Optional[int] = None
        self._lock_candidate_id: Optional[int] = None
        self._lock_frames = 0
        self._lost_frames = 0

        # Target re-identification state
        self._last_locked_class: Optional[str] = None
        self._last_locked_position: Optional[Tuple[int, int]] = None
        self._last_locked_bbox_size: Optional[Tuple[int, int]] = None
        self._last_locked_velocity: Tuple[float, float] = (0.0, 0.0)
        self._reacquire_radius = config.max_distance * REACQUIRE_RADIUS_MULTIPLIER

        # Target selection callback
        self._target_selector = self._default_target_selector

    @property
    def state(self) -> TrackingState:
        """Current tracking state."""
        return self._state

    @property
    def locked_target(self) -> Optional[TrackedObject]:
        """Get locked target if any."""
        if self._locked_target_id is not None:
            objects = self._tracker.objects
            return objects.get(self._locked_target_id)
        return None

    @property
    def all_targets(self) -> Dict[int, TrackedObject]:
        """Get all tracked objects."""
        return self._tracker.objects

    def update(self, detections: List[Detection]) -> Optional[TrackedObject]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detections from detector

        Returns:
            Locked target if tracking, None otherwise
        """
        # Update object tracker
        objects = self._tracker.update(detections)

        # State machine
        if self._state == TrackingState.SEARCHING:
            self._handle_searching(objects)

        elif self._state == TrackingState.ACQUIRING:
            self._handle_acquiring(objects)

        elif self._state == TrackingState.LOCKED:
            self._handle_locked(objects)

        elif self._state == TrackingState.LOST:
            self._handle_lost(objects)

        return self.locked_target

    def lock_target(self, target_id: int) -> bool:
        """
        Manually lock onto a specific target.

        Args:
            target_id: Object ID to lock onto

        Returns:
            True if lock successful
        """
        if target_id in self._tracker.objects:
            self._locked_target_id = target_id
            self._state = TrackingState.LOCKED
            self._lost_frames = 0
            self._save_target_info(self._tracker.objects[target_id])
            logger.info(f"Manual lock on target {target_id}")
            return True
        return False

    def _save_target_info(self, target: TrackedObject) -> None:
        """Save target info for re-identification."""
        self._last_locked_class = target.class_name
        self._last_locked_position = target.center
        bbox = target.bbox
        self._last_locked_bbox_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        self._last_locked_velocity = target.velocity

    def unlock(self) -> None:
        """Release lock and return to searching."""
        self._locked_target_id = None
        self._lock_candidate_id = None
        self._lock_frames = 0
        self._lost_frames = 0
        self._state = TrackingState.SEARCHING
        logger.info("Target unlocked, returning to search mode")

    def get_tracking_error(self) -> Optional[Tuple[float, float]]:
        """
        Get normalized tracking error for PID controller.

        Returns:
            (x_error, y_error) normalized to [-1, 1] where:
            - Negative X = target is left of center
            - Positive X = target is right of center
            - Negative Y = target is above center
            - Positive Y = target is below center
            Returns None if no locked target.
        """
        target = self.locked_target
        if target is None:
            return None

        # Calculate error from center
        error_x = (target.center[0] - self.frame_center[0]) / (self.frame_width / 2)
        error_y = (target.center[1] - self.frame_center[1]) / (self.frame_height / 2)

        # Clamp to [-1, 1]
        error_x = max(-1.0, min(1.0, error_x))
        error_y = max(-1.0, min(1.0, error_y))

        return (error_x, error_y)

    def reset(self) -> None:
        """Reset tracker to initial state."""
        self._tracker.reset()
        self.unlock()

    def _handle_searching(self, objects: Dict[int, TrackedObject]) -> None:
        """Handle SEARCHING state."""
        if not objects:
            return

        # Select best target candidate
        candidate = self._target_selector(objects)
        if candidate and candidate.confidence >= self.config.min_confidence:
            self._lock_candidate_id = candidate.object_id
            self._lock_frames = 1
            self._state = TrackingState.ACQUIRING
            logger.debug(f"Acquiring target {candidate.object_id}")

    def _handle_acquiring(self, objects: Dict[int, TrackedObject]) -> None:
        """Handle ACQUIRING state."""
        if self._lock_candidate_id not in objects:
            # Lost candidate, back to searching
            self._lock_candidate_id = None
            self._lock_frames = 0
            self._state = TrackingState.SEARCHING
            return

        candidate = objects[self._lock_candidate_id]
        if candidate.confidence >= self.config.min_confidence:
            self._lock_frames += 1
            if self._lock_frames >= self.config.frames_to_lock:
                # Lock confirmed
                self._locked_target_id = self._lock_candidate_id
                self._lock_candidate_id = None
                self._state = TrackingState.LOCKED
                logger.info(f"Target {self._locked_target_id} locked")
        else:
            # Confidence dropped, reset acquisition and return to searching
            self._lock_frames = 0
            self._lock_candidate_id = None
            self._state = TrackingState.SEARCHING

    def _handle_locked(self, objects: Dict[int, TrackedObject]) -> None:
        """Handle LOCKED state."""
        if self._locked_target_id not in objects:
            # Target lost
            self._lost_frames += 1
            if self._lost_frames >= self.config.frames_to_unlock:
                self._state = TrackingState.LOST
                logger.warning(f"Target {self._locked_target_id} lost, searching for reacquisition")
        else:
            # Update saved info while locked
            self._save_target_info(objects[self._locked_target_id])
            self._lost_frames = 0

    def _handle_lost(self, objects: Dict[int, TrackedObject]) -> None:
        """Handle LOST state - try to re-identify the target."""
        # Check if original target ID reappeared
        if self._locked_target_id in objects:
            self._lost_frames = 0
            self._state = TrackingState.LOCKED
            logger.info(f"Target {self._locked_target_id} reacquired (same ID)")
            return

        # Try to re-identify target among available objects
        best_match = self._find_matching_target(objects)
        if best_match is not None:
            old_id = self._locked_target_id
            self._locked_target_id = best_match.object_id
            self._lost_frames = 0
            self._state = TrackingState.LOCKED
            self._save_target_info(best_match)
            logger.info(f"Target reacquired: ID changed {old_id} -> {best_match.object_id}")
            return

        self._lost_frames += 1

        # Predict where target should be based on velocity
        if self._last_locked_position and self._last_locked_velocity:
            vx, vy = self._last_locked_velocity
            dt = self._lost_frames * FRAME_TIME_ESTIMATE
            px = int(self._last_locked_position[0] + vx * dt)
            py = int(self._last_locked_position[1] + vy * dt)
            # Keep predicted position in frame
            px = max(0, min(self.frame_width, px))
            py = max(0, min(self.frame_height, py))
            self._last_locked_position = (px, py)

        # After extended loss, return to searching
        max_lost_frames = self.config.frames_to_unlock * 3
        if self._lost_frames > max_lost_frames:
            logger.warning(f"Target lost for {self._lost_frames} frames, returning to search")
            self.unlock()

    def _find_matching_target(self, objects: Dict[int, TrackedObject]) -> Optional[TrackedObject]:
        """Find a target that matches the lost target's characteristics."""
        if not objects or not self._last_locked_class:
            return None

        # Pre-compute squared radius for fast comparison
        radius_sq = self._reacquire_radius * self._reacquire_radius

        best_obj = None
        best_score = MIN_MATCH_SCORE

        for obj in objects.values():
            # Must be same class
            if obj.class_name != self._last_locked_class:
                continue

            # Check distance from predicted position (squared, no sqrt)
            if self._last_locked_position:
                dx = obj.center[0] - self._last_locked_position[0]
                dy = obj.center[1] - self._last_locked_position[1]
                dist_sq = dx * dx + dy * dy
                if dist_sq > radius_sq:
                    continue
                # Normalize distance score (0-1, higher is closer)
                dist_score = 1.0 - (dist_sq / radius_sq)
            else:
                dist_score = 1.0

            # Check size similarity
            size_score = 1.0
            if self._last_locked_bbox_size:
                obj_w = max(1, obj.bbox[2] - obj.bbox[0])
                obj_h = max(1, obj.bbox[3] - obj.bbox[1])
                last_w, last_h = self._last_locked_bbox_size
                if last_w > 0 and last_h > 0:
                    # Use min/max ratio for size similarity (safe division)
                    w_ratio = min(obj_w, last_w) / max(obj_w, last_w)
                    h_ratio = min(obj_h, last_h) / max(obj_h, last_h)
                    size_score = (w_ratio + h_ratio) * 0.5

            # Score: closer is better, similar size is better, higher confidence is better
            score = dist_score * 0.4 + size_score * 0.3 + obj.confidence * 0.3

            if score > best_score:
                best_score = score
                best_obj = obj

        return best_obj

    def _default_target_selector(
        self, objects: Dict[int, TrackedObject]
    ) -> Optional[TrackedObject]:
        """
        Default target selection: largest, highest confidence object closest to center.
        """
        if not objects:
            return None

        def score_target(obj: TrackedObject) -> float:
            # Distance from center (normalized, lower is better)
            # frame_width/height guaranteed >= 1 from __init__
            dist_x = abs(obj.center[0] - self.frame_center[0]) / self.frame_width
            dist_y = abs(obj.center[1] - self.frame_center[1]) / self.frame_height
            center_score = 1.0 - (dist_x + dist_y) / 2

            # Size score (larger is better, normalized)
            bbox_w = max(0, obj.bbox[2] - obj.bbox[0])
            bbox_h = max(0, obj.bbox[3] - obj.bbox[1])
            bbox_area = bbox_w * bbox_h
            max_area = self.frame_width * self.frame_height  # Always >= 1
            size_score = min(1.0, bbox_area / (max_area * 0.25))

            # Combine scores
            return (
                obj.confidence * 0.4 +
                center_score * 0.3 +
                size_score * 0.3
            )

        best = max(objects.values(), key=score_target)
        return best
