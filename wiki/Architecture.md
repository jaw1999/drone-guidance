# Architecture

## Overview

The system consists of 7 core modules coordinated by `TerminalGuidance` in `src/app.py`. Each module is designed to run independently with thread-safe interfaces.

## Module Dependency Graph

```
TerminalGuidance (src/app.py)
    │
    ├── CameraCapture (camera.py)
    │
    ├── Pipeline (pipeline.py)
    │       │
    │       ├── ObjectDetector (detector.py)
    │       │
    │       ├── TargetTracker (tracker.py)
    │       │       │
    │       │       └── Detection (detector.py)
    │       │
    │       └── TrackingInterpolator (pipeline.py)
    │
    ├── PIDController (pid.py)
    │
    ├── MAVLinkController (mavlink_controller.py)
    │
    └── UDPStreamer (streamer.py)
```

## Threading Model

The application uses 6 threads:

### 1. Main Thread
- Runs `TerminalGuidance._main_loop()`
- Polls camera for frames at target FPS
- Calls `Pipeline.process_frame()` for each frame
- Executes PID control when target is locked
- Renders overlay and pushes frames to streamer
- Rate-limited to camera FPS (default 30)

### 2. Camera Capture Thread
- Class: `CameraCapture`
- Method: `_capture_loop()`
- Continuously reads frames from cv2.VideoCapture
- Stores latest frame in thread-safe buffer
- Handles reconnection on failure (up to `reconnect_attempts`)
- Calculates actual FPS over 1-second windows

### 3. Detection Worker Thread
- Class: `DetectionWorker`
- Method: `_run()`
- Receives frames via Queue (maxsize=2)
- Runs YOLO inference
- Outputs results via Queue (maxsize=2)
- Queue overflow: drops oldest result

### 4. MAVLink Heartbeat Thread
- Class: `MAVLinkController`
- Method: `_heartbeat_loop()`
- Sends MAV_TYPE_ONBOARD_CONTROLLER heartbeat at `heartbeat_rate` Hz
- Default: 1 Hz

### 5. MAVLink Receive Thread
- Class: `MAVLinkController`
- Method: `_receive_loop()`
- Blocking receive with 1 second timeout
- Parses HEARTBEAT, GLOBAL_POSITION_INT, VFR_HUD, SYS_STATUS, GPS_RAW_INT, HOME_POSITION, COMMAND_LONG
- Updates `VehicleState` dataclass (thread-safe via lock)
- Monitors connection health via heartbeat timeout

### 6. FFmpeg Subprocess
- Managed by: `UDPStreamer`
- FFmpeg runs as subprocess with stdin pipe
- Raw BGR frames written to stdin
- Encoded H.264 output to UDP

## Data Structures

### Detection (detector.py)
```python
@dataclass
class Detection:
    class_id: int           # COCO class index
    class_name: str         # e.g., "person"
    confidence: float       # 0.0 to 1.0
    bbox: tuple             # (x1, y1, x2, y2) in pixels
    center: tuple           # (cx, cy) center point
```

### TrackedObject (tracker.py)
```python
@dataclass
class TrackedObject:
    object_id: int          # Unique track ID
    class_name: str
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    confidence: float
    velocity: Tuple[float, float]  # pixels/sec
    frames_visible: int
    frames_missing: int
    last_update: float      # timestamp
```

### FrameData (pipeline.py)
```python
@dataclass
class FrameData:
    frame: np.ndarray                           # BGR image
    timestamp: float
    frame_id: int
    detections: List[Detection]                 # Raw detections
    tracked_objects: Dict[int, TrackedObject]   # Active tracks
    locked_target: Optional[TrackedObject]
    tracking_state: TrackingState               # SEARCHING/ACQUIRING/LOCKED/LOST
    inference_time_ms: float
```

### VehicleState (mavlink_controller.py)
```python
@dataclass
class VehicleState:
    armed: bool
    mode: str
    latitude: float
    longitude: float
    altitude_msl: float
    altitude_rel: float
    heading: float
    groundspeed: float
    airspeed: float
    battery_percent: float
    battery_voltage: float
    home_distance: float
    home_latitude: float
    home_longitude: float
    home_altitude: float
    gps_fix: int
    satellites: int
    last_update: float
    vehicle_type: str
    throttle_percent: float
```

## Initialization Sequence

```python
# 1. Load and validate config
config = load_config("config/default.yaml")

# 2. Create components
camera = CameraCapture(CameraConfig.from_dict(config))
detector = ObjectDetector(DetectorConfig.from_dict(config))
tracker = TargetTracker(TrackerConfig.from_dict(config), frame_size)
pipeline = Pipeline(detector, tracker, PipelineConfig.from_dict(config))
pid = PIDController(PIDConfig.from_dict(config))
mavlink = MAVLinkController(MAVLinkConfig.from_dict(config), SafetyConfig.from_dict(config))
streamer = UDPStreamer(StreamerConfig.from_dict(config))

# 3. Initialize detector (loads YOLO model)
detector.initialize()  # Loads NCNN model, runs warmup inference

# 4. Start components
camera.start()      # Spawns capture thread
pipeline.start()    # Spawns detection worker thread
mavlink.start()     # Connects, spawns heartbeat + receive threads
streamer.start()    # Spawns FFmpeg subprocess

# 5. Start main loop
main_thread.start()  # Polls camera, runs pipeline, PID, streaming
```

## Shutdown Sequence

```python
# 1. Stop main loop
_running = False
main_thread.join(timeout=3.0)

# 2. Stop components in reverse order
pipeline.stop()     # Stops detection worker
camera.stop()       # Stops capture thread
mavlink.stop()      # Stops heartbeat + receive threads
streamer.stop()     # Terminates FFmpeg subprocess
detector.shutdown() # Releases model
```

## Error Handling

### Main Loop
- Catches all exceptions
- Counts consecutive errors
- Stops after `MAX_CONSECUTIVE_ERRORS` (10) consecutive failures
- Logs with full traceback

### Camera
- Tracks consecutive read failures
- Reconnects after 30 consecutive failures
- Configurable reconnect attempts and delay

### Detector
- Progressive error logging (warning → error → critical)
- Suppresses repeated errors after threshold
- Returns empty detection list on failure

### MAVLink
- Monitors heartbeat timeout (5 seconds)
- Sets `_connected = False` on timeout
- Triggers safety callback on connection loss

## Configuration Hot-Reload

Some settings can be updated without restart:

| Setting | Restart Required |
|---------|------------------|
| PID gains | No |
| Dead zone | No |
| Target classes | Yes (detector) |
| Model/resolution | Yes (detector) |
| Stream destination | Yes (streamer) |
| MAVLink connection | Yes (mavlink) |
| Camera URL | Yes (camera) |

Updated via `TerminalGuidance.update_config()` which calls component-specific restart methods.
