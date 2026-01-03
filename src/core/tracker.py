"""
Target tracking with lock-on capability.

Provides object tracking across frames with support for:
- Centroid-based tracking (simple, fast)
- ByteTrack algorithm (Kalman filter + IoU matching, more robust)
- Automatic target lock-on and re-acquisition
"""

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .detector import Detection

logger = logging.getLogger(__name__)

# Velocity estimation constants
MIN_VELOCITY_DT = 0.01  # Minimum dt to avoid noise
MAX_VELOCITY_DT = 1.0   # Maximum dt before stale
VELOCITY_SMOOTHING = 0.3  # Exponential smoothing factor

# Target re-acquisition
REACQUIRE_RADIUS_MULT = 3  # Search radius multiplier
FRAME_TIME_ESTIMATE = 0.033  # ~30fps assumed
MIN_MATCH_SCORE = 0.5  # Minimum score for reacquisition


class TrackingState(Enum):
    """Target tracking states."""
    SEARCHING = "searching"  # Looking for targets
    ACQUIRING = "acquiring"  # Confirming lock on candidate
    LOCKED = "locked"        # Actively tracking target
    LOST = "lost"            # Target lost, attempting reacquisition


@dataclass
class TrackedObject:
    """Tracked object with position and velocity."""
    object_id: int
    class_name: str
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    frames_visible: int = 1
    frames_missing: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)  # pixels/second
    last_update: float = 0.0

    def predict_position(self, dt: float = 1.0) -> Tuple[int, int]:
        """Predict position after dt seconds."""
        return (
            int(self.center[0] + self.velocity[0] * dt),
            int(self.center[1] + self.velocity[1] * dt),
        )


@dataclass
class TrackerConfig:
    """Tracker configuration."""
    algorithm: str = "centroid"  # "centroid" or "bytetrack"
    max_disappeared: int = 30    # Frames before removing lost object
    max_distance: int = 100      # Max pixels for centroid matching
    min_confidence: float = 0.6  # Min confidence for lock-on
    frames_to_lock: int = 5      # Frames to confirm lock
    frames_to_unlock: int = 15   # Frames before declaring lost

    # ByteTrack parameters
    high_thresh: float = 0.5   # High confidence threshold
    low_thresh: float = 0.1    # Low confidence threshold
    match_thresh: float = 0.8  # IoU threshold for matching

    @classmethod
    def from_dict(cls, config: dict) -> "TrackerConfig":
        tracker = config.get("tracker", {})
        lock_on = tracker.get("lock_on", {})
        bytetrack = tracker.get("bytetrack", {})
        return cls(
            algorithm=tracker.get("algorithm", "centroid"),
            max_disappeared=tracker.get("max_disappeared", 30),
            max_distance=tracker.get("max_distance", 100),
            min_confidence=lock_on.get("min_confidence", 0.6),
            frames_to_lock=lock_on.get("frames_to_lock", 5),
            frames_to_unlock=lock_on.get("frames_to_unlock", 15),
            high_thresh=bytetrack.get("high_thresh", 0.5),
            low_thresh=bytetrack.get("low_thresh", 0.1),
            match_thresh=bytetrack.get("match_thresh", 0.8),
        )


# =============================================================================
# Centroid Tracker
# =============================================================================

class CentroidTracker:
    """
    Simple centroid-based tracker.

    Matches detections to existing tracks by nearest centroid distance.
    Fast but less robust than ByteTrack for occlusions.
    """

    def __init__(self, max_disappeared: int = 30, max_distance: int = 100):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        self._next_id = 0
        self._objects: OrderedDict[int, TrackedObject] = OrderedDict()
        self._disappeared: Dict[int, int] = {}

    @property
    def objects(self) -> Dict[int, TrackedObject]:
        return dict(self._objects)

    def update(self, detections: List[Detection]) -> Dict[int, TrackedObject]:
        """Update tracker with new detections."""
        now = time.time()

        # No detections - increment disappeared count for all
        if not detections:
            to_remove = []
            for oid in self._disappeared:
                self._disappeared[oid] += 1
                if oid in self._objects:
                    self._objects[oid].frames_missing += 1
                if self._disappeared[oid] > self.max_disappeared:
                    to_remove.append(oid)
            for oid in to_remove:
                self._deregister(oid)
            return self.objects

        # No existing objects - register all
        if not self._objects:
            for det in detections:
                self._register(det, now)
            return self.objects

        # Match existing objects to detections using vectorized distance
        object_ids = tuple(self._objects.keys())
        obj_centers = np.array(
            [self._objects[oid].center for oid in object_ids],
            dtype=np.float32
        )
        det_centers = np.array(
            [d.center for d in detections],
            dtype=np.float32
        )

        # Compute squared distances (no sqrt needed for comparison)
        diff = obj_centers[:, np.newaxis, :] - det_centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        max_dist_sq = self.max_distance * self.max_distance

        # Greedy matching by minimum distance
        rows = dist_sq.min(axis=1).argsort()
        cols = dist_sq.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if dist_sq[row, col] > max_dist_sq:
                continue

            oid = object_ids[row]
            det = detections[col]
            self._update_object(oid, det, now)

            used_rows.add(row)
            used_cols.add(col)

        # Mark unmatched objects as disappeared
        for row in set(range(len(object_ids))) - used_rows:
            oid = object_ids[row]
            self._disappeared[oid] += 1
            self._objects[oid].frames_missing += 1
            if self._disappeared[oid] > self.max_disappeared:
                self._deregister(oid)

        # Register unmatched detections
        for col in set(range(len(detections))) - used_cols:
            self._register(detections[col], now)

        return self.objects

    def _register(self, det: Detection, timestamp: float) -> int:
        """Register new object."""
        oid = self._next_id
        self._objects[oid] = TrackedObject(
            object_id=oid,
            class_name=det.class_name,
            center=det.center,
            bbox=det.bbox,
            confidence=det.confidence,
            velocity=(0.0, 0.0),
            last_update=timestamp,
        )
        self._disappeared[oid] = 0
        self._next_id += 1
        return oid

    def _update_object(self, oid: int, det: Detection, timestamp: float) -> None:
        """Update existing object with new detection."""
        old = self._objects[oid]
        dt = timestamp - old.last_update

        # Calculate smoothed velocity
        if MIN_VELOCITY_DT < dt < MAX_VELOCITY_DT:
            raw_vx = (det.center[0] - old.center[0]) / dt
            raw_vy = (det.center[1] - old.center[1]) / dt
            vx = VELOCITY_SMOOTHING * raw_vx + (1 - VELOCITY_SMOOTHING) * old.velocity[0]
            vy = VELOCITY_SMOOTHING * raw_vy + (1 - VELOCITY_SMOOTHING) * old.velocity[1]
        else:
            vx, vy = old.velocity

        self._objects[oid] = TrackedObject(
            object_id=oid,
            class_name=det.class_name,
            center=det.center,
            bbox=det.bbox,
            confidence=det.confidence,
            frames_visible=old.frames_visible + 1,
            frames_missing=0,
            velocity=(vx, vy),
            last_update=timestamp,
        )
        self._disappeared[oid] = 0

    def _deregister(self, oid: int) -> None:
        """Remove object from tracking."""
        del self._objects[oid]
        del self._disappeared[oid]

    def reset(self) -> None:
        """Reset tracker state."""
        self._objects.clear()
        self._disappeared.clear()
        self._next_id = 0


# =============================================================================
# ByteTrack Implementation
# =============================================================================

class KalmanFilter:
    """Kalman filter for bounding box tracking (constant velocity model)."""

    def __init__(self):
        # State: [cx, cy, aspect_ratio, height, vx, vy, va, vh]
        self._motion_mat = np.eye(8, dtype=np.float32)
        for i in range(4):
            self._motion_mat[i, i + 4] = 1.0

        self._update_mat = np.eye(4, 8, dtype=np.float32)
        self._std_weight_pos = 1.0 / 20
        self._std_weight_vel = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize track from measurement [cx, cy, aspect_ratio, height]."""
        mean = np.concatenate([measurement, np.zeros(4, dtype=np.float32)])
        std = [
            2 * self._std_weight_pos * measurement[3],
            2 * self._std_weight_pos * measurement[3],
            1e-2,
            2 * self._std_weight_pos * measurement[3],
            10 * self._std_weight_vel * measurement[3],
            10 * self._std_weight_vel * measurement[3],
            1e-5,
            10 * self._std_weight_vel * measurement[3],
        ]
        covariance = np.diag(np.square(std).astype(np.float32))
        return mean, covariance

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state."""
        std = [
            self._std_weight_pos * mean[3],
            self._std_weight_pos * mean[3],
            1e-2,
            self._std_weight_pos * mean[3],
            self._std_weight_vel * mean[3],
            self._std_weight_vel * mean[3],
            1e-5,
            self._std_weight_vel * mean[3],
        ]
        motion_cov = np.diag(np.square(std).astype(np.float32))
        mean = self._motion_mat @ mean
        cov = self._motion_mat @ cov @ self._motion_mat.T + motion_cov
        return mean, cov

    def update(self, mean: np.ndarray, cov: np.ndarray,
               measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update state with measurement."""
        std = [
            self._std_weight_pos * mean[3],
            self._std_weight_pos * mean[3],
            1e-1,
            self._std_weight_pos * mean[3],
        ]
        innovation_cov = np.diag(np.square(std).astype(np.float32))

        proj_mean = self._update_mat @ mean
        proj_cov = self._update_mat @ cov @ self._update_mat.T + innovation_cov

        chol = np.linalg.cholesky(proj_cov)
        kalman_gain = np.linalg.solve(
            chol.T, np.linalg.solve(chol, (cov @ self._update_mat.T).T)
        ).T

        innovation = measurement - proj_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = cov - kalman_gain @ proj_cov @ kalman_gain.T
        return new_mean, new_cov


class STrack:
    """Single track for ByteTrack."""

    _count = 0
    shared_kalman = KalmanFilter()

    def __init__(self, detection: Detection):
        self.track_id = 0
        self.is_activated = False
        self.state = "new"

        self.class_name = detection.class_name
        self.score = detection.confidence
        self.bbox = detection.bbox

        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None

        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0

    @staticmethod
    def next_id() -> int:
        STrack._count += 1
        return STrack._count

    @staticmethod
    def reset_id() -> None:
        STrack._count = 0

    def activate(self, frame_id: int) -> None:
        """Activate new track."""
        self.track_id = STrack.next_id()
        self.is_activated = True
        self.state = "tracked"
        self.frame_id = frame_id
        self.start_frame = frame_id

        measurement = self._bbox_to_xyah(self.bbox)
        self.mean, self.covariance = self.shared_kalman.initiate(measurement)

    def re_activate(self, det: Detection, frame_id: int, new_id: bool = False) -> None:
        """Re-activate lost track."""
        self.bbox = det.bbox
        self.score = det.confidence
        self.class_name = det.class_name
        self.state = "tracked"
        self.is_activated = True
        self.frame_id = frame_id
        self.tracklet_len = 0

        measurement = self._bbox_to_xyah(self.bbox)
        self.mean, self.covariance = self.shared_kalman.update(
            self.mean, self.covariance, measurement
        )
        if new_id:
            self.track_id = STrack.next_id()

    def predict(self) -> None:
        """Predict next state."""
        if self.mean is not None:
            self.mean, self.covariance = self.shared_kalman.predict(
                self.mean, self.covariance
            )

    def update(self, det: Detection, frame_id: int) -> None:
        """Update with matched detection."""
        self.bbox = det.bbox
        self.score = det.confidence
        self.class_name = det.class_name
        self.state = "tracked"
        self.is_activated = True
        self.frame_id = frame_id
        self.tracklet_len += 1

        measurement = self._bbox_to_xyah(self.bbox)
        self.mean, self.covariance = self.shared_kalman.update(
            self.mean, self.covariance, measurement
        )

    def mark_lost(self) -> None:
        self.state = "lost"

    def mark_removed(self) -> None:
        self.state = "removed"

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def predicted_bbox(self) -> Tuple[int, int, int, int]:
        if self.mean is None:
            return self.bbox
        return self._xyah_to_bbox(self.mean[:4])

    def _bbox_to_xyah(self, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Convert bbox to [cx, cy, aspect_ratio, height]."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, max(1, y2 - y1)
        return np.array([x1 + w / 2, y1 + h / 2, w / h, h], dtype=np.float32)

    def _xyah_to_bbox(self, xyah: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert [cx, cy, aspect_ratio, height] to bbox."""
        cx, cy, a, h = xyah
        w = a * h
        return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


class ByteTracker:
    """
    ByteTrack multi-object tracker.

    Uses Kalman filter for motion prediction and IoU for association.
    Handles low-confidence detections in second association pass.
    """

    def __init__(self, max_disappeared: int = 30, max_distance: int = 100,
                 high_thresh: float = 0.5, low_thresh: float = 0.1,
                 match_thresh: float = 0.8):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh

        self._tracked: List[STrack] = []
        self._lost: List[STrack] = []
        self._frame_id = 0

    @property
    def objects(self) -> Dict[int, TrackedObject]:
        """Get all tracked objects including recently lost."""
        result = {}
        now = time.time()

        for track in self._tracked:
            if track.is_activated:
                result[track.track_id] = self._track_to_object(track, 0, now)

        for track in self._lost:
            if track.track_id not in result:
                frames_lost = self._frame_id - track.frame_id
                result[track.track_id] = self._track_to_object(track, frames_lost, now)

        return result

    def _track_to_object(self, track: STrack, frames_missing: int,
                         timestamp: float) -> TrackedObject:
        """Convert STrack to TrackedObject."""
        bbox = track.predicted_bbox if frames_missing > 0 else track.bbox
        cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2

        # Extract velocity from Kalman state
        vx, vy = 0.0, 0.0
        if track.mean is not None and len(track.mean) >= 6:
            vx, vy = float(track.mean[4]), float(track.mean[5])

        return TrackedObject(
            object_id=track.track_id,
            class_name=track.class_name,
            center=(cx, cy),
            bbox=bbox,
            confidence=track.score * (0.8 if frames_missing > 0 else 1.0),
            frames_visible=track.tracklet_len,
            frames_missing=frames_missing,
            velocity=(vx, vy),
            last_update=timestamp,
        )

    def update(self, detections: List[Detection]) -> Dict[int, TrackedObject]:
        """Update tracker with new detections."""
        self._frame_id += 1

        # Predict all tracks
        for t in self._tracked + self._lost:
            t.predict()

        # Split detections by confidence
        high_dets, low_dets = [], []
        high_tracks, low_tracks = [], []
        for d in detections:
            if d.confidence >= self.high_thresh:
                high_dets.append(d)
                high_tracks.append(STrack(d))
            elif d.confidence >= self.low_thresh:
                low_dets.append(d)
                low_tracks.append(STrack(d))

        # Separate confirmed and unconfirmed tracks
        confirmed = [t for t in self._tracked if t.is_activated]
        unconfirmed = [t for t in self._tracked if not t.is_activated]

        # First association: high-conf detections with tracked + lost
        pool = confirmed + self._lost
        matched, unmatched_t, unmatched_d = self._associate(pool, high_tracks)

        for ti, di in matched:
            track = pool[ti]
            det = high_dets[di]
            if track.state == "tracked":
                track.update(det, self._frame_id)
            else:
                track.re_activate(det, self._frame_id)

        # Second association: low-conf with remaining tracked
        remaining = [pool[i] for i in unmatched_t if pool[i].state == "tracked"]
        matched2, unmatched_t2, _ = self._associate(remaining, low_tracks, thresh=0.5)

        for ti, di in matched2:
            remaining[ti].update(low_dets[di], self._frame_id)

        # Mark unmatched tracks as lost
        matched2_set = {ti for ti, _ in matched2}
        for i in unmatched_t:
            track = pool[i]
            if track.state == "tracked":
                if track in remaining:
                    ri = remaining.index(track)
                    if ri not in matched2_set:
                        track.mark_lost()
                else:
                    track.mark_lost()

        # Match unconfirmed with remaining high-conf detections
        unmatched_high = [high_tracks[i] for i in unmatched_d]
        matched3, unmatched_uc, unmatched_uh = self._associate(unconfirmed, unmatched_high, thresh=0.7)

        for ti, di in matched3:
            unconfirmed[ti].update(high_dets[unmatched_d[di]], self._frame_id)
        for i in unmatched_uc:
            unconfirmed[i].mark_removed()

        # Initialize new tracks from unmatched high-conf
        for i in unmatched_uh:
            det_idx = unmatched_d[i]
            if high_dets[det_idx].confidence >= self.high_thresh:
                new_track = unmatched_high[i]
                new_track.activate(self._frame_id)
                self._tracked.append(new_track)

        # Update track lists
        self._tracked = [t for t in self._tracked if t.state == "tracked"]

        for t in pool:
            if t.state == "lost" and t not in self._lost:
                self._lost.append(t)

        self._lost = [
            t for t in self._lost
            if self._frame_id - t.frame_id <= self.max_disappeared and t.state != "removed"
        ]

        return self.objects

    def _associate(self, tracks: List[STrack], detections: List[STrack],
                   thresh: float = None) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate tracks with detections using IoU."""
        thresh = thresh or self.match_thresh

        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Compute IoU cost matrix
        cost = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, t in enumerate(tracks):
            for j, d in enumerate(detections):
                cost[i, j] = 1.0 - self._iou(t.predicted_bbox, d.bbox)

        # Greedy matching
        matched = []
        unmatched_t = set(range(len(tracks)))
        unmatched_d = set(range(len(detections)))

        indices = np.unravel_index(np.argsort(cost, axis=None), cost.shape)
        for i, j in zip(indices[0], indices[1]):
            if i in unmatched_t and j in unmatched_d and cost[i, j] <= thresh:
                matched.append((i, j))
                unmatched_t.discard(i)
                unmatched_d.discard(j)

        return matched, list(unmatched_t), list(unmatched_d)

    def _iou(self, box1: Tuple, box2: Tuple) -> float:
        """Compute IoU between two bboxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracked.clear()
        self._lost.clear()
        self._frame_id = 0
        STrack.reset_id()


# =============================================================================
# Target Tracker (High-Level Interface)
# =============================================================================

class TargetTracker:
    """
    High-level tracker with lock-on capability.

    Wraps either CentroidTracker or ByteTracker and adds:
    - Automatic target selection and lock-on
    - Target re-identification after loss
    - State machine for tracking workflow
    """

    def __init__(self, config: TrackerConfig, frame_size: Tuple[int, int]):
        self.config = config
        self.frame_width = max(1, frame_size[0])
        self.frame_height = max(1, frame_size[1])
        self.frame_center = (self.frame_width // 2, self.frame_height // 2)

        # Initialize underlying tracker
        if config.algorithm.lower() == "bytetrack":
            logger.info("Using ByteTrack algorithm")
            self._tracker = ByteTracker(
                max_disappeared=config.max_disappeared,
                max_distance=config.max_distance,
                high_thresh=config.high_thresh,
                low_thresh=config.low_thresh,
                match_thresh=config.match_thresh,
            )
        else:
            logger.info("Using Centroid tracker")
            self._tracker = CentroidTracker(
                max_disappeared=config.max_disappeared,
                max_distance=config.max_distance,
            )

        # Lock-on state
        self._state = TrackingState.SEARCHING
        self._locked_id: Optional[int] = None
        self._candidate_id: Optional[int] = None
        self._lock_frames = 0
        self._lost_frames = 0

        # Re-identification state
        self._last_class: Optional[str] = None
        self._last_position: Optional[Tuple[int, int]] = None
        self._last_size: Optional[Tuple[int, int]] = None
        self._last_velocity: Tuple[float, float] = (0.0, 0.0)
        self._reacquire_radius = config.max_distance * REACQUIRE_RADIUS_MULT

    @property
    def state(self) -> TrackingState:
        return self._state

    @property
    def locked_target(self) -> Optional[TrackedObject]:
        if self._locked_id is not None:
            return self._tracker.objects.get(self._locked_id)
        return None

    @property
    def all_targets(self) -> Dict[int, TrackedObject]:
        return self._tracker.objects

    def update(self, detections: List[Detection],
               frame: Optional[np.ndarray] = None) -> Optional[TrackedObject]:
        """Update tracker with new detections."""
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
        """Manually lock onto a target."""
        if target_id in self._tracker.objects:
            self._locked_id = target_id
            self._state = TrackingState.LOCKED
            self._lost_frames = 0
            self._save_target_info(self._tracker.objects[target_id])
            logger.info(f"Locked target {target_id}")
            return True
        return False

    def lock_on_bbox(self, bbox: Tuple[int, int, int, int],
                     frame: Optional[np.ndarray] = None,
                     class_name: str = "target") -> bool:
        """Initialize tracking on a bounding box."""
        det = Detection(
            class_id=0,
            class_name=class_name,
            confidence=1.0,
            bbox=bbox,
            center=((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
        )
        objects = self._tracker.update([det])
        if objects:
            return self.lock_target(list(objects.keys())[0])
        return False

    def unlock(self) -> None:
        """Release lock and return to searching."""
        self._locked_id = None
        self._candidate_id = None
        self._lock_frames = 0
        self._lost_frames = 0
        self._state = TrackingState.SEARCHING
        logger.info("Target unlocked")

    def get_tracking_error(self) -> Optional[Tuple[float, float]]:
        """Get normalized tracking error [-1, 1] for PID controller."""
        target = self.locked_target
        if target is None:
            return None

        error_x = (target.center[0] - self.frame_center[0]) / (self.frame_width / 2)
        error_y = (target.center[1] - self.frame_center[1]) / (self.frame_height / 2)

        return (
            max(-1.0, min(1.0, error_x)),
            max(-1.0, min(1.0, error_y)),
        )

    def reset(self) -> None:
        """Reset to initial state."""
        self._tracker.reset()
        self.unlock()

    def _save_target_info(self, target: TrackedObject) -> None:
        """Save target info for re-identification."""
        self._last_class = target.class_name
        self._last_position = target.center
        self._last_size = (target.bbox[2] - target.bbox[0], target.bbox[3] - target.bbox[1])
        self._last_velocity = target.velocity

    def _handle_searching(self, objects: Dict[int, TrackedObject]) -> None:
        """Look for a target to lock onto."""
        if not objects:
            return

        candidate = self._select_best_target(objects)
        if candidate and candidate.confidence >= self.config.min_confidence:
            self._candidate_id = candidate.object_id
            self._lock_frames = 1
            self._state = TrackingState.ACQUIRING

    def _handle_acquiring(self, objects: Dict[int, TrackedObject]) -> None:
        """Confirm lock on candidate."""
        if self._candidate_id not in objects:
            self._candidate_id = None
            self._lock_frames = 0
            self._state = TrackingState.SEARCHING
            return

        candidate = objects[self._candidate_id]
        if candidate.confidence >= self.config.min_confidence:
            self._lock_frames += 1
            if self._lock_frames >= self.config.frames_to_lock:
                self._locked_id = self._candidate_id
                self._candidate_id = None
                self._state = TrackingState.LOCKED
                self._save_target_info(candidate)
                logger.info(f"Target {self._locked_id} locked")
        else:
            self._lock_frames = max(0, self._lock_frames - 2)
            if self._lock_frames == 0:
                self._candidate_id = None
                self._state = TrackingState.SEARCHING

    def _handle_locked(self, objects: Dict[int, TrackedObject]) -> None:
        """Track locked target."""
        if self._locked_id not in objects:
            self._lost_frames += 1
            if self._lost_frames >= self.config.frames_to_unlock:
                self._state = TrackingState.LOST
                logger.warning(f"Target {self._locked_id} lost")
        else:
            self._save_target_info(objects[self._locked_id])
            self._lost_frames = 0

    def _handle_lost(self, objects: Dict[int, TrackedObject]) -> None:
        """Try to re-acquire lost target."""
        # Check if original ID reappeared
        if self._locked_id in objects:
            self._lost_frames = 0
            self._state = TrackingState.LOCKED
            logger.info(f"Target {self._locked_id} reacquired")
            return

        # Try to find matching target
        match = self._find_matching_target(objects)
        if match:
            old_id = self._locked_id
            self._locked_id = match.object_id
            self._lost_frames = 0
            self._state = TrackingState.LOCKED
            self._save_target_info(match)
            logger.info(f"Target reacquired: {old_id} -> {match.object_id}")
            return

        self._lost_frames += 1

        # Update predicted position
        if self._last_position and self._last_velocity:
            vx, vy = self._last_velocity
            dt = self._lost_frames * FRAME_TIME_ESTIMATE
            px = int(self._last_position[0] + vx * dt)
            py = int(self._last_position[1] + vy * dt)
            self._last_position = (
                max(0, min(self.frame_width, px)),
                max(0, min(self.frame_height, py)),
            )

        # Give up after extended loss
        if self._lost_frames > self.config.frames_to_unlock * 3:
            logger.warning("Target lost, returning to search")
            self.unlock()

    def _find_matching_target(self, objects: Dict[int, TrackedObject]) -> Optional[TrackedObject]:
        """Find target matching lost target's characteristics."""
        if not objects or not self._last_class:
            return None

        radius_sq = self._reacquire_radius ** 2
        best_obj, best_score = None, MIN_MATCH_SCORE

        for obj in objects.values():
            if obj.class_name != self._last_class:
                continue

            # Distance score
            if self._last_position:
                dx = obj.center[0] - self._last_position[0]
                dy = obj.center[1] - self._last_position[1]
                dist_sq = dx * dx + dy * dy
                if dist_sq > radius_sq:
                    continue
                dist_score = 1.0 - (dist_sq / radius_sq)
            else:
                dist_score = 1.0

            # Size score
            size_score = 1.0
            if self._last_size:
                w = max(1, obj.bbox[2] - obj.bbox[0])
                h = max(1, obj.bbox[3] - obj.bbox[1])
                lw, lh = self._last_size
                if lw > 0 and lh > 0:
                    size_score = (min(w, lw) / max(w, lw) + min(h, lh) / max(h, lh)) / 2

            score = dist_score * 0.4 + size_score * 0.3 + obj.confidence * 0.3
            if score > best_score:
                best_score = score
                best_obj = obj

        return best_obj

    def _select_best_target(self, objects: Dict[int, TrackedObject]) -> Optional[TrackedObject]:
        """Select best target: largest, highest confidence, closest to center."""
        if not objects:
            return None

        def score(obj: TrackedObject) -> float:
            # Center proximity
            dx = abs(obj.center[0] - self.frame_center[0]) / self.frame_width
            dy = abs(obj.center[1] - self.frame_center[1]) / self.frame_height
            center_score = 1.0 - (dx + dy) / 2

            # Size
            area = (obj.bbox[2] - obj.bbox[0]) * (obj.bbox[3] - obj.bbox[1])
            max_area = self.frame_width * self.frame_height * 0.25
            size_score = min(1.0, area / max_area)

            return obj.confidence * 0.4 + center_score * 0.3 + size_score * 0.3

        return max(objects.values(), key=score)
