# Configuration

## Overview

Configuration is stored in `config/default.yaml`. The file is loaded at startup by `src/utils/config.py` using PyYAML.

## Complete Reference

### camera

Camera input configuration.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `rtsp_url` | string | `""` | Camera source. Accepts RTSP URL, HTTP MJPEG URL, webcam index ("0"), or file path |
| `resolution.width` | int | 1920 | Frame width (for FOV calculations) |
| `resolution.height` | int | 1080 | Frame height (for FOV calculations) |
| `fov.horizontal` | float | 90.0 | Horizontal field of view (degrees) |
| `fov.vertical` | float | 60.0 | Vertical field of view (degrees) |
| `fps` | int | 30 | Target capture FPS |
| `buffer_size` | int | 1 | OpenCV buffer size (1 = minimal latency) |
| `reconnect_attempts` | int | 5 | Connection retry attempts |
| `reconnect_delay_sec` | float | 2.0 | Delay between retries (seconds) |

**Source formats:**
- RTSP: `rtsp://user:pass@192.168.1.10:554/stream`
- HTTP MJPEG: `http://192.168.1.10:8080/video`
- Webcam: `0` (or `1`, `2`, etc.)
- File: `/path/to/video.mp4`

### detector

Object detection configuration.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | string | `"yolov8n"` | Model name: `"yolov8n"` or `"yolo11n"` |
| `resolution` | string | `"640"` | Input size: `"640"`, `"416"`, or `"320"` |
| `confidence_threshold` | float | 0.5 | Minimum detection confidence (0.0-1.0) |
| `nms_threshold` | float | 0.45 | NMS IoU threshold (0.0-1.0) |
| `target_classes` | list | `[]` | COCO classes to detect. Empty = all |
| `detection_interval` | int | 3 | Run detection every N frames |

**Model paths:**
Models are loaded from `models/ncnn/{model}_{resolution}_ncnn_model/`

### tracker

Object tracking configuration.

| Key | Type | Code Default | Config File | Description |
|-----|------|--------------|-------------|-------------|
| `algorithm` | string | `"centroid"` | `"bytetrack"` | `"bytetrack"` or `"centroid"` |
| `max_disappeared` | int | 30 | 30 | Frames before removing lost track |
| `max_distance` | int | 100 | 150 | Max centroid distance for matching (pixels) |
| `lock_on.min_confidence` | float | 0.6 | 0.6 | Minimum confidence for lock-on |
| `lock_on.frames_to_lock` | int | 5 | 5 | Frames to confirm lock |
| `lock_on.frames_to_unlock` | int | 15 | 15 | Frames before declaring lost |
| `bytetrack.high_thresh` | float | 0.5 | 0.5 | High confidence threshold |
| `bytetrack.low_thresh` | float | 0.1 | 0.1 | Low confidence threshold |
| `bytetrack.match_thresh` | float | 0.8 | 0.8 | IoU threshold for matching |

### pid

PID controller configuration.

Each axis (`yaw`, `pitch`, `throttle`) has:

| Key | Type | Code Default | Config File | Description |
|-----|------|--------------|-------------|-------------|
| `kp` | float | 0.5 | varies | Proportional gain |
| `ki` | float | 0.01 | varies | Integral gain |
| `kd` | float | 0.1 | varies | Derivative gain |
| `max_rate` | float | 30.0 | varies | Maximum output rate |
| `derivative_filter` | float | 0.1 | 0.2 | Derivative filter coefficient (0-1) |
| `slew_rate` | float | 0.0 | varies | Maximum rate change per second |

Top-level PID settings:

| Key | Type | Code Default | Config File | Description |
|-----|------|--------------|-------------|-------------|
| `dead_zone_percent` | float | 5.0 | 10.0 | Error dead zone (% of frame) |
| `update_rate` | float | 20.0 | 20 | Control loop rate (Hz) |

**Config file axis values:**

| Axis | kp | ki | kd | max_rate | derivative_filter | slew_rate |
|------|----|----|----|---------:|------------------:|----------:|
| yaw | 0.3 | 0.005 | 0.15 | 15.0 | 0.2 | 20.0 |
| pitch | 0.25 | 0.005 | 0.12 | 10.0 | 0.2 | 15.0 |
| throttle | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### mavlink

MAVLink communication configuration.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `connection` | string | `"udp:192.168.1.1:14550"` | Connection string |
| `source_system` | int | 255 | MAVLink system ID |
| `source_component` | int | 190 | MAVLink component ID |
| `heartbeat_rate` | float | 1.0 | Heartbeat frequency (Hz) |
| `command_timeout` | float | 5.0 | Command ACK timeout (seconds) |
| `enable_control` | bool | false | Enable sending control commands |
| `vehicle_type` | string | `"plane"` | `"plane"`, `"copter"`, or `"auto"` |
| `intercept.cruise_throttle` | int | 50 | Normal throttle (%) |
| `intercept.tracking_throttle` | int | 70 | Tracking throttle (%) |
| `intercept.max_turn_rate` | float | 45.0 | Max turn rate (deg/sec) |
| `intercept.max_climb_rate` | float | 5.0 | Max climb rate (m/s) |

**Connection string formats:**
- UDP: `udp:192.168.1.1:14550`
- TCP: `tcp:192.168.1.1:5760`
- Serial: `/dev/ttyUSB0` or `/dev/ttyACM0`

### safety

Safety and geofence configuration.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `target_lost_action` | string | `"loiter"` | Action on target loss |
| `search_timeout` | float | 10.0 | Seconds before executing lost action |
| `geofence.enabled` | bool | true | Enable geofence checks |
| `geofence.max_distance_m` | float | 500 | Max distance from home (m) |
| `geofence.max_altitude_m` | float | 120 | Max altitude AGL (m) |
| `geofence.min_altitude_m` | float | 10 | Min altitude AGL (m) |
| `min_battery_percent` | float | 20 | Min battery before RTL (%) |
| `max_tracking_speed` | float | 10.0 | Max tracking speed (m/s) |
| `require_arm_confirmation` | bool | true | Require armed state for control |
| `emergency_stop_enabled` | bool | true | Enable emergency stop |

**Lost target actions:**
- `hover`: Zero velocity (copter only)
- `loiter`: Switch to LOITER mode
- `rtl`: Return to launch
- `land`: Land immediately
- `continue_last`: No action

### output

Video output configuration.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `stream.enabled` | bool | true | Enable UDP streaming |
| `stream.udp_host` | string | `"127.0.0.1"` | Destination IP |
| `stream.udp_port` | int | 5600 | Destination port |
| `resolution.width` | int | 1280 | Output width |
| `resolution.height` | int | 720 | Output height |
| `fps` | int | 30 | Output frame rate |
| `bitrate_kbps` | int | 2000 | H.264 bitrate (kbps) |
| `codec` | string | `"h264"` | Codec (unused, always h264) |
| `hardware_encode` | bool | false | Unused on Pi 5 |
| `overlay.show_detections` | bool | true | Show detection boxes |
| `overlay.show_locked_target` | bool | true | Show locked target |
| `overlay.show_tracking_info` | bool | true | Show tracking state |
| `overlay.show_telemetry` | bool | true | Show telemetry data |
| `overlay.font_scale` | float | 0.6 | Text size |
| `overlay.box_thickness` | int | 2 | Bounding box thickness |

### web

Web UI configuration.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | true | Enable web UI |
| `host` | string | `"0.0.0.0"` | Bind address |
| `port` | int | 5000 | HTTP port |

### logging

Logging configuration.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `level` | string | `"INFO"` | Log level: DEBUG, INFO, WARNING, ERROR |
| `file` | string | `"logs/terminal_guidance.log"` | Log file path |
| `max_size_mb` | int | 10 | Max log file size (MB) |
| `backup_count` | int | 3 | Number of backup files |
| `log_performance` | bool | true | Log performance metrics |
| `performance_interval_sec` | float | 30 | Performance log interval |

### system

System configuration.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_workers` | int | 4 | Thread pool size (unused) |
| `use_gpu` | bool | false | GPU acceleration (unused on Pi) |
| `performance_mode` | string | `"balanced"` | Performance mode (unused) |

## Example Configuration

```yaml
camera:
  rtsp_url: "rtsp://root:12345@192.168.2.10/stream=0"
  resolution:
    width: 1280
    height: 720
  fps: 30

detector:
  model: "yolov8n"
  resolution: "640"
  confidence_threshold: 0.5
  target_classes:
    - "person"
    - "car"
  detection_interval: 3

tracker:
  algorithm: "bytetrack"
  max_disappeared: 30
  lock_on:
    min_confidence: 0.6
    frames_to_lock: 5
    frames_to_unlock: 15

pid:
  yaw:
    kp: 0.3
    ki: 0.005
    kd: 0.15
    max_rate: 15.0
  pitch:
    kp: 0.25
    ki: 0.005
    kd: 0.12
    max_rate: 10.0
  dead_zone_percent: 10.0

mavlink:
  connection: "udp:192.168.1.1:14550"
  enable_control: false
  vehicle_type: "plane"

safety:
  target_lost_action: "loiter"
  geofence:
    enabled: true
    max_distance_m: 500
    max_altitude_m: 120
    min_altitude_m: 10

output:
  stream:
    enabled: true
    udp_host: "192.168.1.129"
    udp_port: 5600
  resolution:
    width: 1280
    height: 720
  fps: 30
  bitrate_kbps: 2000

web:
  enabled: true
  port: 5000
```

## Runtime Updates

Some settings can be updated at runtime via the web API without restart:

| Setting | Restart Required |
|---------|------------------|
| PID gains | No |
| Dead zone | No |
| Target classes | Yes (detector) |
| Model/resolution | Yes (detector) |
| Stream destination | Yes (streamer) |
| MAVLink connection | Yes (mavlink) |
| Camera URL | Yes (camera) |

Use `POST /api/config` to update settings.

## Config Loading

```python
# src/utils/config.py
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
```

Each module has a `Config` dataclass with `from_dict()` classmethod:

```python
@classmethod
def from_dict(cls, config: dict) -> "DetectorConfig":
    det = config.get("detector", {})
    return cls(
        model=det.get("model", "yolov8n"),
        resolution=str(det.get("resolution", "640")),
        # ...
    )
```

Missing keys use default values.
