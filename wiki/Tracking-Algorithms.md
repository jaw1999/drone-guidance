# Tracking Algorithms

## Overview

Object tracking is implemented in `src/core/tracker.py`. The system provides two tracking algorithms and a high-level `TargetTracker` class that adds lock-on state management.

## Architecture

```
TargetTracker (high-level interface)
    │
    ├── CentroidTracker  (algorithm="centroid")
    │
    └── ByteTracker      (algorithm="bytetrack")
```

## Centroid Tracker

### Algorithm

Centroid tracking matches detections to existing tracks by minimum Euclidean distance between centroids.

### Data Structures

```python
_objects: OrderedDict[int, TrackedObject]  # Active tracks
_disappeared: Dict[int, int]               # Frames since last detection
```

### Update Process

1. **No detections**: Increment `disappeared` count for all tracks. Remove tracks exceeding `max_disappeared`.

2. **No existing tracks**: Register all detections as new tracks.

3. **Match detections to tracks**:
   - Compute pairwise squared distances between track centroids and detection centroids
   - Greedy matching: sort by distance, assign closest pairs first
   - Reject matches exceeding `max_distance`

4. **Handle unmatched**:
   - Unmatched tracks: increment `disappeared`
   - Unmatched detections: register as new tracks

### Distance Calculation

Vectorized using numpy:

```python
# obj_centers: (M, 2), det_centers: (N, 2)
diff = obj_centers[:, np.newaxis, :] - det_centers[np.newaxis, :, :]
dist_sq = np.sum(diff * diff, axis=2)  # (M, N) squared distances
```

### Velocity Estimation

Velocity is computed on each update using exponential smoothing:

```python
if MIN_VELOCITY_DT < dt < MAX_VELOCITY_DT:  # 0.01s < dt < 1.0s
    raw_vx = (new_x - old_x) / dt
    raw_vy = (new_y - old_y) / dt
    vx = 0.3 * raw_vx + 0.7 * old_vx  # VELOCITY_SMOOTHING = 0.3
    vy = 0.3 * raw_vy + 0.7 * old_vy
```

## ByteTrack

ByteTrack is a multi-object tracking algorithm that uses Kalman filtering for motion prediction and IoU (Intersection over Union) for detection-track association.

### Key Features

1. **Two-stage association**: First matches high-confidence detections, then low-confidence detections
2. **Kalman filter**: Predicts object motion using constant velocity model
3. **Re-identification**: Lost tracks can be re-associated with detections

### Kalman Filter

#### State Vector

8-dimensional state: `[cx, cy, aspect_ratio, height, vx, vy, va, vh]`

Where:
- `cx, cy`: bounding box center
- `aspect_ratio`: width / height
- `height`: bounding box height
- `vx, vy, va, vh`: velocities of the above

#### Motion Model

Constant velocity assumption:

```
x_{t+1} = F * x_t + w

F = [I_4  I_4]  (8x8 identity with position-velocity coupling)
    [0    I_4]
```

#### Process Noise

Standard deviations scaled by object height:

```python
std_pos = 1/20 * height   # position noise
std_vel = 1/160 * height  # velocity noise
```

#### Measurement Update

Standard Kalman update equations:

```python
# Innovation
y = z - H * x_predicted

# Kalman gain
K = P_predicted * H^T * (H * P_predicted * H^T + R)^-1

# Updated state
x = x_predicted + K * y
P = (I - K * H) * P_predicted
```

The implementation uses Cholesky decomposition for numerical stability:

```python
chol = np.linalg.cholesky(proj_cov)
kalman_gain = np.linalg.solve(chol.T, np.linalg.solve(chol, (cov @ H.T).T)).T
```

### IoU Calculation

```python
def _iou(box1, box2):
    # Intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)

    # Union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0
```

### Association Process

1. **Predict**: Apply Kalman prediction to all tracks

2. **Split detections**:
   - High-confidence: `conf >= high_thresh` (default 0.5)
   - Low-confidence: `low_thresh <= conf < high_thresh` (default 0.1-0.5)

3. **First association** (high-confidence detections):
   - Pool: confirmed tracks + lost tracks
   - Cost matrix: `1 - IoU(predicted_bbox, detection_bbox)`
   - Greedy matching with threshold `match_thresh` (default 0.8)

4. **Second association** (low-confidence detections):
   - Pool: unmatched confirmed tracks only
   - Lower threshold (0.5)
   - Catches partially occluded objects

5. **Handle unmatched**:
   - Unmatched tracks: mark as "lost"
   - Unmatched high-confidence detections: initialize new tracks

### Track States

| State | Description |
|-------|-------------|
| `new` | Just created, not yet activated |
| `tracked` | Actively tracked |
| `lost` | Not matched for recent frames |
| `removed` | Deleted from tracker |

## Target Tracker

`TargetTracker` wraps either algorithm and adds lock-on state management.

### Tracking States

```python
class TrackingState(Enum):
    SEARCHING = "searching"  # Looking for targets
    ACQUIRING = "acquiring"  # Confirming lock on candidate
    LOCKED = "locked"        # Actively tracking target
    LOST = "lost"            # Target lost, attempting reacquisition
```

### State Machine

```
SEARCHING ──[candidate found]──► ACQUIRING
    ▲                                │
    │                     [confirmed after N frames]
    │                                │
    │                                ▼
    └──[lost for 3x unlock]──── LOCKED
                                     │
                          [missing for unlock_frames]
                                     │
                                     ▼
                                   LOST ──[reacquired]──► LOCKED
```

### Lock-on Process

1. **SEARCHING → ACQUIRING**: When a target with `confidence >= min_confidence` appears, it becomes a candidate.

2. **ACQUIRING → LOCKED**: If the candidate maintains confidence for `frames_to_lock` consecutive frames (default 5), lock is confirmed.

3. **ACQUIRING → SEARCHING**: If candidate drops below threshold for 2 frames per 1 frame of progress, acquisition fails.

4. **LOCKED → LOST**: If locked target is missing for `frames_to_unlock` frames (default 15).

5. **LOST → LOCKED**: If original track ID reappears, or a matching target is found via re-identification.

6. **LOST → SEARCHING**: After `frames_to_unlock * 3` frames without reacquisition.

### Re-identification

When a target is lost, the system attempts to re-identify it using:

1. **Class match**: Must be same class as lost target

2. **Distance constraint**: Must be within `max_distance * 3` pixels of predicted position

3. **Scoring function**:
```python
score = distance_score * 0.4 + size_score * 0.3 + confidence * 0.3
```

Where:
- `distance_score = 1 - (distance² / max_distance²)`
- `size_score = (min(w,lw)/max(w,lw) + min(h,lh)/max(h,lh)) / 2`

4. **Minimum score**: 0.5 required for match

### Target Selection

When no target is locked, `_select_best_target()` scores candidates:

```python
score = confidence * 0.4 + center_proximity * 0.3 + size_score * 0.3
```

Where:
- `center_proximity = 1 - (|dx|/width + |dy|/height) / 2`
- `size_score = min(1, area / (frame_area * 0.25))`

### Tracking Error

`get_tracking_error()` returns normalized error for PID control:

```python
error_x = (target_cx - frame_cx) / (frame_width / 2)  # [-1, 1]
error_y = (target_cy - frame_cy) / (frame_height / 2) # [-1, 1]
```

Positive X = target right of center
Positive Y = target below center

## Configuration

Values from `config/default.yaml`. Code defaults: `algorithm="centroid"`, `max_distance=100`.

```yaml
tracker:
  algorithm: "bytetrack"      # "bytetrack" or "centroid"
  max_disappeared: 30         # Frames before removing lost track
  max_distance: 150           # Max centroid distance for matching (pixels)

  lock_on:
    min_confidence: 0.6       # Minimum confidence for lock-on
    frames_to_lock: 5         # Frames to confirm lock
    frames_to_unlock: 15      # Frames before declaring lost

  bytetrack:
    high_thresh: 0.5          # High confidence threshold
    low_thresh: 0.1           # Low confidence threshold
    match_thresh: 0.8         # IoU threshold for matching
```

## TrackedObject Dataclass

```python
@dataclass
class TrackedObject:
    object_id: int
    class_name: str
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    frames_visible: int = 1
    frames_missing: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)  # pixels/second
    last_update: float = 0.0

    def predict_position(self, dt: float) -> Tuple[int, int]:
        """Predict position after dt seconds."""
```

## Pipeline Integration

Detection runs every N frames (default 3). Between detections, `TrackingInterpolator` in `pipeline.py` predicts positions:

```python
# Time since last detection (clamped to 0.5s max)
dt = min(now - last_detection_time, 0.5)

# Predict new center
new_cx = smoothed_x + velocity_x * dt
new_cy = smoothed_y + velocity_y * dt
```

Position smoothing uses exponential moving average (α = 0.6):
```python
smoothed_x = 0.6 * detected_x + 0.4 * previous_smoothed_x
```
