# Terminal Guidance

Drone companion computer software for Raspberry Pi 5. Performs real-time object detection and tracking, streams video with overlay to QGroundControl, and sends control commands to ArduPilot via MAVLink.

## Features

- YOLO11n detection with NCNN backend (~160ms at 640px, ~100ms at 416px)
- Target tracking with ByteTrack (Kalman filter) or centroid tracker
- PID control for yaw/pitch adjustment to keep target centered
- UDP H.264 video stream with overlay to QGroundControl (port 5600)
- Web UI for configuration and control (port 5000)
- MAVLink integration for telemetry and rate commands
- Systemd service for headless operation

## Requirements

- Raspberry Pi 5 (4GB+ RAM)
- Python 3.11+
- FFmpeg (uses libx264 software encoder)
- Camera (USB, CSI, or RTSP stream)

## Quick Start

```bash
cd terminal_guidance
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m src.app

# Or install as service
./scripts/install-service.sh
```

## Configuration

Edit `config/default.yaml`:

- **camera** - RTSP URL, webcam index, or video file path
- **detector** - Model, confidence threshold, target classes, input size
- **tracker** - Algorithm (bytetrack/centroid), lock-on thresholds, max disappeared frames
- **pid** - Yaw/pitch/throttle gains, dead zone, max rates, slew rate limiting
- **mavlink** - Flight controller connection string, system IDs
- **safety** - Target lost action, geofence, battery limits
- **output** - Resolution, bitrate, UDP destination

### Key Settings

```yaml
detector:
  model: "yolo11n"
  backend: "ncnn"
  weights_path: "yolo11n_ncnn_model"
  input_size: 640              # 192, 416, or 640
  detection_interval: 3        # Run detection every N frames
  confidence_threshold: 0.5
  half_precision: false        # Not supported on Pi 5 CPU

tracker:
  algorithm: "bytetrack"       # "bytetrack" or "centroid"
  max_disappeared: 30
  lock_on:
    frames_to_lock: 5
    frames_to_unlock: 15
  bytetrack:
    high_thresh: 0.5
    low_thresh: 0.1
    match_thresh: 0.8

pid:
  yaw:
    kp: 0.5
    ki: 0.01
    kd: 0.1
    max_rate: 30.0
    derivative_filter: 0.1     # Low-pass filter (0=heavy, 1=none)
    slew_rate: 60.0            # Max rate change per second
  dead_zone_percent: 5.0

output:
  stream:
    udp_host: "192.168.1.129"
    udp_port: 5600
  bitrate_kbps: 2000

mavlink:
  connection: "udp:192.168.1.1:14550"
  enable_control: false

safety:
  target_lost_action: "loiter"  # hover, loiter, rtl, land
  min_battery_percent: 20
  geofence:
    max_distance_m: 500
    max_altitude_m: 120
```

## QGroundControl Setup

1. Open QGC > Application Settings > Video
2. Set Video Source = `UDP h.264 Video Stream`
3. Set Port = `5600`

Video overlay shows:
- Bounding boxes around detected objects
- Lock indicator when tracking
- Telemetry (FPS, inference time, altitude, battery)
- Center crosshair

## Web UI

Access at `http://<pi-ip>:5000`:

- System status (FPS, inference time, targets)
- Flight controller connection config
- PID gain tuning
- Detection parameters
- Lock/unlock targets
- Enable/disable control output
- Emergency stop

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Raspberry Pi 5                              │
│                                                                     │
│  Camera -> Detection (YOLO11n) -> Tracking -> PID Controller        │
│                 |                                  |                │
│           Overlay Renderer                    MAVLink UDP           │
│                 |                                  |                │
│              FFmpeg                                v                │
│                 |                         Flight Controller         │
│                 v                                                   │
│            UDP :5600 ─────────────────────────────────────┐         │
│                                                           │         │
│         Flask API :5000 <─────────────────────────────┐   │         │
└───────────────────────────────────────────────────────│───│─────────┘
                                                        │   │
┌───────────────────────────────────────────────────────│───│─────────┐
│                      Ground Station                   │   │         │
│                                                       v   v         │
│  QGroundControl ─── HTTP API calls ──────────────> Pi:5000          │
│       │                                                             │
│       └── Video Stream <─────────────────────────── Pi:5600         │
│                                                                     │
│  Custom Terminal Guidance tab makes REST API calls to Raspberry Pi  │
└─────────────────────────────────────────────────────────────────────┘
```

**Data Flow:**
1. QGC custom tab sends HTTP requests to Pi's Flask API (port 5000)
2. Pi processes commands and sends MAVLink rate commands to flight controller
3. Pi streams H.264 video with overlay to QGC (port 5600)

### Components

| File | Description |
|------|-------------|
| `src/app.py` | Main application, coordinates all components |
| `src/core/camera.py` | Video capture with auto-reconnection |
| `src/core/detector.py` | YOLO inference wrapper (NCNN/OpenVINO/PyTorch) |
| `src/core/tracker.py` | ByteTrack and centroid tracker with lock-on state machine |
| `src/core/pipeline.py` | Async detection with velocity interpolation |
| `src/core/pid.py` | PID controller with derivative filtering and slew rate limiting |
| `src/core/mavlink_controller.py` | ArduPilot communication |
| `src/core/streamer.py` | FFmpeg UDP streaming with overlay |
| `src/web/app.py` | Flask configuration UI |

### Tracking States

| State | Description |
|-------|-------------|
| SEARCHING | No targets detected |
| ACQUIRING | Target found, confirming lock (frames_to_lock) |
| LOCKED | Target locked, PID control active |
| LOST | Target disappeared, attempting re-identification |

## Performance

Raspberry Pi 5 with NCNN backend (CPU-only):

| Input Size | Inference Time | Detection FPS |
|------------|----------------|---------------|
| 640px | ~160ms | ~6 |
| 416px | ~100ms | ~10 |
| 192px | ~50ms | ~20 |

Notes:
- NCNN backend provides ~4x speedup over PyTorch
- `detection_interval: 3` skips frames; tracker interpolates between detections
- FP16 not supported on Pi 5 CPU
- Pipeline maintains 30 FPS output with velocity interpolation

## HTTP API

The QGC custom plugin controls Terminal Guidance via REST API calls to the Raspberry Pi.

### Tracking Control

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/tracking/lock` | POST | Auto-lock onto best target |
| `/api/tracking/lock/<id>` | POST | Lock specific target by ID |
| `/api/tracking/unlock` | POST | Release current lock |
| `/api/tracking/enable` | POST | Enable PID control output |
| `/api/tracking/disable` | POST | Disable PID control output |

### Status & Config

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | System status (FPS, targets, state) |
| `/api/config` | GET | Current configuration |
| `/api/config` | POST | Update configuration (JSON body) |
| `/api/emergency-stop` | POST | Activate emergency stop |
| `/api/emergency-stop/clear` | POST | Clear emergency stop |

### Service Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/services/status` | GET | Status of all components |
| `/api/restart/detector` | POST | Restart detector |
| `/api/restart/camera` | POST | Restart camera |
| `/api/restart/streamer` | POST | Restart video streamer |
| `/api/restart/mavlink` | POST | Restart MAVLink connection |
| `/api/restart/all` | POST | Restart all services |

### Example Usage

```bash
# Get system status
curl http://192.168.1.100:5000/api/status

# Lock onto a target
curl -X POST http://192.168.1.100:5000/api/tracking/lock

# Update detection confidence
curl -X POST http://192.168.1.100:5000/api/config \
  -H "Content-Type: application/json" \
  -d '{"detector": {"confidence_threshold": 0.6}}'
```

## Systemd Service

```bash
./scripts/install-service.sh

sudo systemctl start terminal-guidance
sudo systemctl stop terminal-guidance
sudo systemctl status terminal-guidance

journalctl -u terminal-guidance -f
```

## Development

```bash
pytest

python -m src.app --log-level DEBUG

python -m src.app --no-web
```

