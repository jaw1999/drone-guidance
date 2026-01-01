"""Target tracking module with lock-on capability."""

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .detector import Detection

logger = logging.getLogger(__name__)

# Default paths for NanoTrack models
# NOTE: Must use the official OpenCV-compatible models from SiamTrackers repo
# https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack/models/nanotrackv2
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
DEFAULT_NANOTRACK_BACKBONE = MODELS_DIR / "nanotrack_backbone_sim.onnx"
DEFAULT_NANOTRACK_HEAD = MODELS_DIR / "nanotrack_head_sim.onnx"

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
    algorithm: str = "centroid"  # "centroid" or "bytetrack"
    max_disappeared: int = 30
    max_distance: int = 100
    min_confidence: float = 0.6
    frames_to_lock: int = 5
    frames_to_unlock: int = 15
    # ByteTrack-specific parameters
    high_thresh: float = 0.5  # High confidence threshold for first association
    low_thresh: float = 0.1  # Low confidence threshold for second association
    match_thresh: float = 0.8  # IoU threshold for matching

    @classmethod
    def from_dict(cls, config: dict) -> "TrackerConfig":
        """Create config from dictionary."""
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

            # Vectorized distance calculation using NumPy broadcasting
            # Use squared distances to avoid sqrt
            n_obj = len(object_ids)
            n_det = len(detections)

            # Build center arrays for vectorized computation
            obj_centers = np.array([self._objects[oid].center for oid in object_ids], dtype=np.float32)
            det_centers = np.array([det.center for det in detections], dtype=np.float32)

            # Broadcast: (n_obj, 1, 2) - (1, n_det, 2) = (n_obj, n_det, 2)
            diff = obj_centers[:, np.newaxis, :] - det_centers[np.newaxis, :, :]
            D = np.sum(diff * diff, axis=2)  # Squared distances (n_obj, n_det)

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


# =============================================================================
# ByteTrack Implementation
# =============================================================================

class KalmanFilter:
    """
    Simple Kalman filter for bounding box tracking.

    State vector: [cx, cy, aspect_ratio, height, vx, vy, va, vh]
    Measurement: [cx, cy, aspect_ratio, height]
    """

    def __init__(self):
        # State transition matrix (constant velocity model)
        self._motion_mat = np.eye(8, dtype=np.float32)
        for i in range(4):
            self._motion_mat[i, i + 4] = 1.0

        # Measurement matrix
        self._update_mat = np.eye(4, 8, dtype=np.float32)

        # Process noise (tuned for typical video tracking)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize track from first detection.

        Args:
            measurement: [cx, cy, aspect_ratio, height]

        Returns:
            (mean, covariance) tuple
        """
        mean_pos = measurement
        mean_vel = np.zeros(4, dtype=np.float32)
        mean = np.concatenate([mean_pos, mean_vel])

        # Initial covariance
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std).astype(np.float32))

        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state."""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(std).astype(np.float32))

        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov

        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update state with new measurement."""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std).astype(np.float32))

        # Project state to measurement space
        projected_mean = self._update_mat @ mean
        projected_cov = self._update_mat @ covariance @ self._update_mat.T + innovation_cov

        # Kalman gain
        chol = np.linalg.cholesky(projected_cov)
        kalman_gain = np.linalg.solve(
            chol.T, np.linalg.solve(chol, (covariance @ self._update_mat.T).T)
        ).T

        # Update
        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T

        return new_mean, new_covariance


class STrack:
    """Single object track for ByteTrack."""

    _count = 0
    shared_kalman = KalmanFilter()

    def __init__(self, detection: Detection):
        self.track_id = 0  # Assigned when activated
        self.is_activated = False
        self.state = "new"  # new, tracked, lost, removed

        # Detection info
        self.class_name = detection.class_name
        self.score = detection.confidence
        self.bbox = detection.bbox  # x1, y1, x2, y2

        # Kalman state
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None

        # Frame counters
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
        """Activate a new track."""
        self.track_id = STrack.next_id()
        self.is_activated = True
        self.state = "tracked"
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 0

        # Initialize Kalman filter
        measurement = self._bbox_to_xyah(self.bbox)
        self.mean, self.covariance = self.shared_kalman.initiate(measurement)

    def re_activate(self, detection: Detection, frame_id: int, new_id: bool = False) -> None:
        """Re-activate a lost track."""
        self.bbox = detection.bbox
        self.score = detection.confidence
        self.class_name = detection.class_name
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
        """Predict next state using Kalman filter."""
        if self.mean is not None and self.covariance is not None:
            self.mean, self.covariance = self.shared_kalman.predict(self.mean, self.covariance)

    def update(self, detection: Detection, frame_id: int) -> None:
        """Update track with matched detection."""
        self.bbox = detection.bbox
        self.score = detection.confidence
        self.class_name = detection.class_name
        self.state = "tracked"
        self.is_activated = True
        self.frame_id = frame_id
        self.tracklet_len += 1

        measurement = self._bbox_to_xyah(self.bbox)
        self.mean, self.covariance = self.shared_kalman.update(
            self.mean, self.covariance, measurement
        )

    def mark_lost(self) -> None:
        """Mark track as lost."""
        self.state = "lost"

    def mark_removed(self) -> None:
        """Mark track as removed."""
        self.state = "removed"

    @property
    def center(self) -> Tuple[int, int]:
        """Get bounding box center."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def predicted_bbox(self) -> Tuple[int, int, int, int]:
        """Get predicted bounding box from Kalman state."""
        if self.mean is None:
            return self.bbox
        return self._xyah_to_bbox(self.mean[:4])

    def _bbox_to_xyah(self, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Convert bbox (x1,y1,x2,y2) to (cx, cy, aspect_ratio, height)."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = max(1, y2 - y1)
        cx = x1 + w / 2
        cy = y1 + h / 2
        return np.array([cx, cy, w / h, h], dtype=np.float32)

    def _xyah_to_bbox(self, xyah: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert (cx, cy, aspect_ratio, height) to bbox (x1,y1,x2,y2)."""
        cx, cy, a, h = xyah
        w = a * h
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        return (x1, y1, x2, y2)


class ByteTracker:
    """
    ByteTrack multi-object tracker.

    Key features:
    - Two-stage association: high confidence first, then low confidence
    - Kalman filter for motion prediction
    - Handles occlusions and ID switches better than centroid tracking
    """

    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: int = 100,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
    ):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh

        self._tracked_stracks: List[STrack] = []
        self._lost_stracks: List[STrack] = []
        self._removed_stracks: List[STrack] = []
        self._frame_id = 0

    @property
    def objects(self) -> Dict[int, TrackedObject]:
        """Get all active tracked objects (compatible with CentroidTracker interface)."""
        result = {}
        for track in self._tracked_stracks:
            if track.is_activated:
                result[track.track_id] = TrackedObject(
                    object_id=track.track_id,
                    class_name=track.class_name,
                    center=track.center,
                    bbox=track.bbox,
                    confidence=track.score,
                    frames_visible=track.tracklet_len,
                    frames_missing=0,
                    velocity=self._estimate_velocity(track),
                    last_update=time.time(),
                )
        return result

    def _estimate_velocity(self, track: STrack) -> Tuple[float, float]:
        """Estimate velocity from Kalman state."""
        if track.mean is not None and len(track.mean) >= 6:
            # vx, vy are in state indices 4, 5
            return (float(track.mean[4]), float(track.mean[5]))
        return (0.0, 0.0)

    def update(self, detections: List[Detection]) -> Dict[int, TrackedObject]:
        """Update tracker with new detections."""
        self._frame_id += 1

        # Predict all tracked stracks
        for track in self._tracked_stracks:
            track.predict()
        for track in self._lost_stracks:
            track.predict()

        # Split detections by confidence and create STracks in single pass
        # This avoids 4 separate list comprehensions (2 filters + 2 STrack creates)
        high_dets = []
        low_dets = []
        high_stracks = []
        low_stracks = []
        for d in detections:
            conf = d.confidence
            if conf >= self.high_thresh:
                high_dets.append(d)
                high_stracks.append(STrack(d))
            elif conf >= self.low_thresh:
                low_dets.append(d)
                low_stracks.append(STrack(d))

        # Partition tracked stracks in single pass (avoids 2 list comprehensions)
        unconfirmed = []
        tracked = []
        for t in self._tracked_stracks:
            if t.is_activated:
                tracked.append(t)
            else:
                unconfirmed.append(t)

        # Combine tracked and lost for matching
        strack_pool = tracked + self._lost_stracks

        # Match high confidence detections
        matched, unmatched_tracks, unmatched_dets = self._associate(
            strack_pool, high_stracks, self.match_thresh
        )

        # Update matched tracks
        for track_idx, det_idx in matched:
            track = strack_pool[track_idx]
            det = high_dets[det_idx]
            if track.state == "tracked":
                track.update(Detection(
                    class_id=0, class_name=det.class_name, confidence=det.confidence,
                    bbox=det.bbox, center=det.center
                ), self._frame_id)
            else:
                track.re_activate(Detection(
                    class_id=0, class_name=det.class_name, confidence=det.confidence,
                    bbox=det.bbox, center=det.center
                ), self._frame_id)

        # --- Second association: low confidence with remaining tracked ---
        remaining_tracks = [strack_pool[i] for i in unmatched_tracks if strack_pool[i].state == "tracked"]
        matched2, unmatched_tracks2, _ = self._associate(
            remaining_tracks, low_stracks, 0.5  # Lower threshold for low-conf
        )

        for track_idx, det_idx in matched2:
            track = remaining_tracks[track_idx]
            det = low_dets[det_idx]
            track.update(Detection(
                class_id=0, class_name=det.class_name, confidence=det.confidence,
                bbox=det.bbox, center=det.center
            ), self._frame_id)

        # Mark unmatched tracks as lost
        # Build set of matched track indices from second association for O(1) lookup
        matched2_track_indices = {track_idx for track_idx, _ in matched2}
        for idx in unmatched_tracks:
            track = strack_pool[idx]
            if track.state == "tracked":
                # Check if this track is in remaining_tracks and was matched in second pass
                track_in_remaining = track in remaining_tracks
                if not track_in_remaining:
                    track.mark_lost()
                else:
                    remaining_idx = remaining_tracks.index(track)
                    if remaining_idx not in matched2_track_indices:
                        track.mark_lost()

        # Handle unconfirmed tracks
        # Build list of unmatched high-conf STracks for association with unconfirmed
        unmatched_high_stracks = [high_stracks[i] for i in unmatched_dets]
        matched_unconf, unmatched_unconf, unmatched_from_high = self._associate(
            unconfirmed, unmatched_high_stracks, 0.7
        )
        for track_idx, det_idx in matched_unconf:
            # det_idx is index into unmatched_high_stracks
            original_det_idx = unmatched_dets[det_idx]
            unconfirmed[track_idx].update(
                high_dets[original_det_idx], self._frame_id
            )
        for idx in unmatched_unconf:
            unconfirmed[idx].mark_removed()

        # Initialize new tracks from unmatched high-confidence detections
        for idx in unmatched_from_high:
            # idx is index into unmatched_high_stracks, map back to original
            original_det_idx = unmatched_dets[idx]
            if original_det_idx < len(high_dets) and high_dets[original_det_idx].confidence >= self.high_thresh:
                # Use the STrack from unmatched_high_stracks (same as high_stracks[original_det_idx])
                new_track = unmatched_high_stracks[idx]
                new_track.activate(self._frame_id)
                self._tracked_stracks.append(new_track)

        # Update track lists
        self._tracked_stracks = [t for t in self._tracked_stracks if t.state == "tracked"]

        # Move lost tracks
        for track in strack_pool:
            if track.state == "lost" and track not in self._lost_stracks:
                self._lost_stracks.append(track)

        # Remove tracks lost for too long
        self._lost_stracks = [
            t for t in self._lost_stracks
            if self._frame_id - t.frame_id <= self.max_disappeared and t.state != "removed"
        ]

        return self.objects

    def _associate(
        self,
        tracks: List[STrack],
        detections: List[STrack],
        thresh: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate tracks with detections using IoU."""
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Compute IoU distance matrix
        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou = self._compute_iou(track.predicted_bbox, det.bbox)
                cost_matrix[i, j] = 1.0 - iou

        # Simple greedy matching (could use Hungarian algorithm for optimality)
        matched = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))

        # Sort by cost and greedily assign
        indices = np.unravel_index(np.argsort(cost_matrix, axis=None), cost_matrix.shape)
        for i, j in zip(indices[0], indices[1]):
            if i in unmatched_tracks and j in unmatched_dets:
                if cost_matrix[i, j] <= thresh:
                    matched.append((i, j))
                    unmatched_tracks.discard(i)
                    unmatched_dets.discard(j)

        return matched, list(unmatched_tracks), list(unmatched_dets)

    def _compute_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
    ) -> float:
        """Compute Intersection over Union."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - inter

        if union <= 0:
            return 0.0
        return inter / union

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracked_stracks.clear()
        self._lost_stracks.clear()
        self._removed_stracks.clear()
        self._frame_id = 0
        STrack.reset_id()


# =============================================================================
# NanoTrack Implementation (Siamese Network Tracker)
# =============================================================================

class NanoTracker:
    """
    NanoTrack-based visual object tracker using OpenCV's TrackerNano.

    NanoTrack is a Siamese network tracker that excels at maintaining lock
    on a specific target instance. Unlike detection-based trackers, it learns
    a template of the target and matches it frame-to-frame.

    Key advantages:
    - Very fast (30-60+ FPS on RPi5)
    - Maintains lock through appearance changes, partial occlusions
    - Target-agnostic (tracks whatever you initialize it with)
    - Lightweight (~1.8MB models)

    Use with YOLO: YOLO detects targets, NanoTrack maintains continuous lock.
    """

    def __init__(
        self,
        backbone_path: Optional[Path] = None,
        head_path: Optional[Path] = None,
        max_disappeared: int = 30,
        max_distance: int = 100,
    ):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        # Model paths
        self._backbone_path = backbone_path or DEFAULT_NANOTRACK_BACKBONE
        self._head_path = head_path or DEFAULT_NANOTRACK_HEAD

        # Validate models exist
        if not self._backbone_path.exists():
            raise FileNotFoundError(f"NanoTrack backbone not found: {self._backbone_path}")
        if not self._head_path.exists():
            raise FileNotFoundError(f"NanoTrack head not found: {self._head_path}")

        # Tracker state
        self._cv_tracker: Optional[cv2.TrackerNano] = None
        self._is_initialized = False
        self._track_id = 0
        self._next_id = 1

        # Current tracked object state
        self._bbox: Optional[Tuple[int, int, int, int]] = None
        self._class_name: str = "target"
        self._confidence: float = 1.0
        self._frames_visible: int = 0
        self._frames_missing: int = 0
        self._velocity: Tuple[float, float] = (0.0, 0.0)
        self._last_center: Optional[Tuple[int, int]] = None
        self._last_update: float = 0.0

        # Frame reference for tracking
        self._last_frame: Optional[np.ndarray] = None

        logger.info(f"NanoTracker initialized with models: {self._backbone_path.name}, {self._head_path.name}")

    def _create_tracker(self) -> cv2.TrackerNano:
        """Create a new TrackerNano instance."""
        params = cv2.TrackerNano.Params()
        params.backbone = str(self._backbone_path)
        params.neckhead = str(self._head_path)
        return cv2.TrackerNano.create(params)

    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], class_name: str = "target") -> bool:
        """
        Initialize tracker with a target bounding box.

        Args:
            frame: Current video frame (BGR)
            bbox: Target bounding box (x1, y1, x2, y2)
            class_name: Class name for the target

        Returns:
            True if initialization successful
        """
        # Convert x1,y1,x2,y2 to x,y,w,h for OpenCV
        x1, y1, x2, y2 = bbox
        cv_bbox = (x1, y1, x2 - x1, y2 - y1)

        # Create fresh tracker
        self._cv_tracker = self._create_tracker()

        try:
            logger.debug(f"NanoTrack init: frame {frame.shape}, bbox {cv_bbox}")
            result = self._cv_tracker.init(frame, cv_bbox)
            # NOTE: OpenCV TrackerNano.init() returns None on some builds instead of True
            # We treat None as success since the tracker actually works
            success = result is None or result is True
            if success:
                self._is_initialized = True
                self._track_id = self._next_id
                self._next_id += 1
                self._bbox = bbox
                self._class_name = class_name
                self._confidence = 1.0
                self._frames_visible = 1
                self._frames_missing = 0
                self._last_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                self._last_update = time.time()
                self._last_frame = frame.copy()
                logger.info(f"NanoTrack initialized on target ID {self._track_id}: {class_name}")
                return True
            else:
                logger.warning(f"TrackerNano.init failed with result: {result}")
        except Exception as e:
            logger.error(f"NanoTrack init failed: {e}", exc_info=True)

        self._is_initialized = False
        return False

    def update_frame(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """
        Update tracker with new frame.

        Args:
            frame: Current video frame (BGR)

        Returns:
            (success, bbox) where bbox is (x1, y1, x2, y2) or None
        """
        if not self._is_initialized or self._cv_tracker is None:
            return False, None

        self._last_frame = frame

        try:
            success, cv_bbox = self._cv_tracker.update(frame)

            if success:
                # Convert x,y,w,h back to x1,y1,x2,y2
                x, y, w, h = [int(v) for v in cv_bbox]
                bbox = (x, y, x + w, y + h)

                # Calculate velocity
                current_time = time.time()
                new_center = (x + w // 2, y + h // 2)

                if self._last_center is not None:
                    dt = current_time - self._last_update
                    if MIN_VELOCITY_DT < dt < MAX_VELOCITY_DT:
                        raw_vx = (new_center[0] - self._last_center[0]) / dt
                        raw_vy = (new_center[1] - self._last_center[1]) / dt
                        alpha = VELOCITY_SMOOTHING_ALPHA
                        old_vx, old_vy = self._velocity
                        self._velocity = (
                            alpha * raw_vx + (1 - alpha) * old_vx,
                            alpha * raw_vy + (1 - alpha) * old_vy,
                        )

                self._bbox = bbox
                self._last_center = new_center
                self._last_update = current_time
                self._frames_visible += 1
                self._frames_missing = 0
                self._confidence = 1.0  # NanoTrack doesn't provide confidence, assume good

                return True, bbox
            else:
                self._frames_missing += 1
                self._confidence *= 0.9  # Decay confidence on miss

                if self._frames_missing > self.max_disappeared:
                    logger.warning(f"NanoTrack lost target after {self._frames_missing} frames")
                    self._is_initialized = False

                return False, self._bbox  # Return last known bbox

        except Exception as e:
            logger.error(f"NanoTrack update failed: {e}")
            self._frames_missing += 1
            return False, self._bbox

    def update(self, detections: List[Detection], frame: Optional[np.ndarray] = None) -> Dict[int, TrackedObject]:
        """
        Update tracker - compatible interface with CentroidTracker/ByteTracker.

        For NanoTrack, detections are used to (re)initialize if not tracking.
        The frame parameter is required for NanoTrack to work.

        Args:
            detections: List of Detection objects (used for initialization)
            frame: Current video frame (required for NanoTrack)

        Returns:
            Dictionary of object_id -> TrackedObject
        """
        if frame is None:
            frame = self._last_frame

        if frame is None:
            return {}

        # If not initialized and we have detections, initialize on best detection
        if not self._is_initialized and detections:
            # Pick highest confidence detection
            best = max(detections, key=lambda d: d.confidence)
            self.init(frame, best.bbox, best.class_name)

        # Update with current frame
        success, bbox = self.update_frame(frame)

        if success and bbox is not None:
            return {
                self._track_id: TrackedObject(
                    object_id=self._track_id,
                    class_name=self._class_name,
                    center=self._last_center or ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
                    bbox=bbox,
                    confidence=self._confidence,
                    frames_visible=self._frames_visible,
                    frames_missing=self._frames_missing,
                    velocity=self._velocity,
                    last_update=self._last_update,
                )
            }

        return {}

    @property
    def objects(self) -> Dict[int, TrackedObject]:
        """Get all tracked objects (single object for NanoTrack)."""
        if not self._is_initialized or self._bbox is None:
            return {}

        return {
            self._track_id: TrackedObject(
                object_id=self._track_id,
                class_name=self._class_name,
                center=self._last_center or ((self._bbox[0] + self._bbox[2]) // 2, (self._bbox[1] + self._bbox[3]) // 2),
                bbox=self._bbox,
                confidence=self._confidence,
                frames_visible=self._frames_visible,
                frames_missing=self._frames_missing,
                velocity=self._velocity,
                last_update=self._last_update,
            )
        }

    @property
    def is_tracking(self) -> bool:
        """Check if currently tracking a target."""
        return self._is_initialized and self._frames_missing == 0

    def reset(self) -> None:
        """Reset tracker state."""
        self._cv_tracker = None
        self._is_initialized = False
        self._bbox = None
        self._frames_visible = 0
        self._frames_missing = 0
        self._velocity = (0.0, 0.0)
        self._last_center = None
        self._last_frame = None
        logger.info("NanoTracker reset")


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

        # Initialize tracker based on algorithm selection
        algorithm = config.algorithm.lower()

        if algorithm == "nanotrack":
            logger.info("Using NanoTrack algorithm (Siamese network tracker)")
            self._tracker = NanoTracker(
                max_disappeared=config.max_disappeared,
                max_distance=config.max_distance,
            )
            self._uses_nanotrack = True
        elif algorithm == "bytetrack":
            logger.info("Using ByteTrack algorithm (Kalman + IoU matching)")
            self._tracker = ByteTracker(
                max_disappeared=config.max_disappeared,
                max_distance=config.max_distance,
                high_thresh=config.high_thresh,
                low_thresh=config.low_thresh,
                match_thresh=config.match_thresh,
            )
            self._uses_nanotrack = False
        else:
            logger.info("Using Centroid tracker algorithm")
            self._tracker = CentroidTracker(
                max_disappeared=config.max_disappeared,
                max_distance=config.max_distance,
            )
            self._uses_nanotrack = False

        # Store last frame for NanoTrack
        self._last_frame: Optional[np.ndarray] = None

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

    def update(
        self,
        detections: List[Detection],
        frame: Optional[np.ndarray] = None,
    ) -> Optional[TrackedObject]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detections from detector
            frame: Current video frame (required for NanoTrack)

        Returns:
            Locked target if tracking, None otherwise
        """
        # Store frame for NanoTrack
        if frame is not None:
            self._last_frame = frame

        # Update object tracker
        if self._uses_nanotrack:
            objects = self._tracker.update(detections, frame=self._last_frame)
        else:
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

    def lock_on_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        frame: Optional[np.ndarray] = None,
        class_name: str = "target",
    ) -> bool:
        """
        Initialize tracking on a specific bounding box (for NanoTrack).

        This is the preferred way to start tracking with NanoTrack - provide
        the exact bbox you want to track.

        Args:
            bbox: Target bounding box (x1, y1, x2, y2)
            frame: Video frame containing the target
            class_name: Class name for the target

        Returns:
            True if lock successful
        """
        if frame is None:
            frame = self._last_frame

        if frame is None:
            logger.error("Cannot lock on bbox: no frame available")
            return False

        if self._uses_nanotrack:
            # Initialize NanoTrack directly on the bbox
            success = self._tracker.init(frame, bbox, class_name)
            if success:
                self._locked_target_id = self._tracker._track_id
                self._state = TrackingState.LOCKED
                self._lost_frames = 0
                # Save target info from the tracker
                if self._tracker.objects:
                    target = list(self._tracker.objects.values())[0]
                    self._save_target_info(target)
                logger.info(f"NanoTrack locked on bbox: {bbox}")
                return True
        else:
            # For other trackers, create a detection and update
            from .detector import Detection
            det = Detection(
                class_id=0,
                class_name=class_name,
                confidence=1.0,
                bbox=bbox,
                center=((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
            )
            objects = self._tracker.update([det])
            if objects:
                target_id = list(objects.keys())[0]
                return self.lock_target(target_id)

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
                self._save_target_info(candidate)
                logger.info(f"Target {self._locked_target_id} locked")
        else:
            # Confidence dropped - reduce lock frames but don't reset completely
            # This provides hysteresis to handle fluctuating confidence
            self._lock_frames = max(0, self._lock_frames - 2)
            if self._lock_frames == 0:
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
