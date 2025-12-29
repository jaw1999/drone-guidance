# Terminal Guidance

Drone companion computer software for Raspberry Pi 5. Performs real-time object detection and tracking, streams video with overlay to QGroundControl, and sends control commands to ArduPilot via MAVLink.

## Features

- **YOLO11n Detection** - Object detection optimized for Pi 5 CPU (~30ms inference at 320px)
- **Target Tracking** - Lock onto and follow specific objects (person, car, truck, boat)
- **PID Control** - Automatic yaw/pitch adjustment to keep target centered
- **UDP Video Stream** - H.264 video with overlay direct to QGroundControl (port 5600)
- **Web Configuration** - Browser-based UI for tuning parameters (port 5000)
- **MAVLink Integration** - Receives telemetry, sends rate commands to ArduPilot
- **Auto-start** - Systemd service for headless operation

## Requirements

- Raspberry Pi 5 (4GB+ RAM recommended)
- Python 3.11+
- FFmpeg with h264_v4l2m2m encoder
- Camera (USB, CSI, or RTSP stream)

## Quick Start

```bash
# Clone and setup
cd terminal_guidance
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
python -m src.app

# Or install as service (Pi only)
./scripts/install-service.sh
```

## Configuration

Edit `config/default.yaml` to configure:

- **Camera** - RTSP URL, webcam index, or video file path
- **Detection** - Model, confidence threshold, target classes, input size
- **Tracking** - Lock-on thresholds, interpolation, max disappeared frames
- **PID** - Yaw/pitch/throttle gains, dead zone, max rates
- **MAVLink** - Flight controller connection string, system IDs
- **Safety** - Target lost action, geofence, battery limits
- **Stream** - Output resolution, bitrate, UDP destination

### Key Settings

```yaml
detector:
  model: "yolo11n"           # YOLO11 nano - fastest for Pi 5 CPU
  input_size: 320            # Model input (320, 416, or 640)
  half_precision: false      # Keep false for CPU (FP16 hurts CPU performance)
  detection_interval: 2      # Run detection every N frames
  confidence_threshold: 0.5  # Minimum detection confidence

tracker:
  max_disappeared: 30        # Frames before losing track
  lock_on:
    frames_to_lock: 5        # Frames before confirming lock
    frames_to_unlock: 15     # Frames without target before unlocking

output:
  stream:
    udp_host: "127.0.0.1"    # QGC address
    udp_port: 5600           # QGC video port
  hardware_encode: false     # Set true on Pi 5 for h264_v4l2m2m

mavlink:
  connection: "udp:192.168.1.1:14550"
  enable_control: false      # Enable actual flight commands

safety:
  target_lost_action: "loiter"  # hover, loiter, rtl, land
  min_battery_percent: 20
  geofence:
    max_distance_m: 500
    max_altitude_m: 120
```

## QGroundControl Setup

1. Open QGC > Application Settings > Video
2. Set **Video Source** = `UDP h.264 Video Stream`
3. Set **Port** = `5600`

Video feed will show detection overlay with:
- Bounding boxes around detected objects
- Lock indicator when tracking
- Telemetry (FPS, inference time, altitude, battery, etc.)
- Center crosshair for aiming reference

## Web UI

Access at `http://<pi-ip>:5000` to:

- Monitor system status (FPS, inference time, targets)
- Configure flight controller connection
- Tune PID gains in real-time
- Adjust detection parameters
- Lock/unlock targets
- Enable/disable control output
- Emergency stop

## Architecture

```
Camera → Detection (YOLO11n) → Tracking → PID Controller → MAVLink
                ↓
         Overlay Renderer → FFmpeg → UDP/RTP → QGC
                ↓
              Web UI
```

### Components

| File | Description |
|------|-------------|
| `src/app.py` | Main application, coordinates all components |
| `src/core/camera.py` | Video capture with auto-reconnection |
| `src/core/detector.py` | YOLO inference wrapper |
| `src/core/tracker.py` | Centroid tracker with lock-on state machine |
| `src/core/pipeline.py` | Async detection with velocity interpolation |
| `src/core/pid.py` | PID controller for yaw/pitch/throttle |
| `src/core/mavlink_controller.py` | ArduPilot communication |
| `src/core/streamer.py` | FFmpeg UDP streaming with overlay |
| `src/web/app.py` | Flask configuration UI |

### Tracking States

| State | Description |
|-------|-------------|
| SEARCHING | No targets detected, scanning |
| ACQUIRING | Target found, confirming lock (frames_to_lock) |
| LOCKED | Target locked, PID control active |
| LOST | Target disappeared, attempting re-identification |

## Performance

On Raspberry Pi 5 (CPU-only, no GPU):

| Input Size | Detection Interval | Inference Time | Notes |
|------------|-------------------|----------------|-------|
| 320px | 2 | ~30ms | Recommended |
| 416px | 2 | ~50ms | Higher accuracy |
| 320px | 1 | ~30ms | More responsive, higher CPU |

Key optimization notes:
- `half_precision: false` - FP16 hurts CPU performance
- `detection_interval: 2` - Skip frames to reduce CPU load
- `detection_resolution: 320x240` - Downscale before inference
- Velocity interpolation fills gaps between detections

## MAVLink Commands

The system accepts custom commands from QGC via COMMAND_LONG:

| Command | MAV_CMD | Description |
|---------|---------|-------------|
| Auto-lock | 31010 (USER_1) | Lock onto best target |
| Lock target | 31011 (USER_2) | Lock specific target (param1 = target_id) |
| Unlock | 31012 (USER_3) | Release current lock |
| Enable control | 31013 (USER_4) | Enable tracking control |
| Disable control | 31014 (USER_5) | Disable tracking control |

## Systemd Service

```bash
# Install
./scripts/install-service.sh

# Control
sudo systemctl start terminal-guidance
sudo systemctl stop terminal-guidance
sudo systemctl status terminal-guidance

# Logs
journalctl -u terminal-guidance -f
```

## Development

```bash
# Run tests
pytest

# Run with debug logging
python -m src.app --log-level DEBUG

# Disable web UI
python -m src.app --no-web
```

### Testing on macOS

The software can run on macOS for development:
- Uses `h264_videotoolbox` or `libx264` encoder
- Set `half_precision: false` (no GPU)
- Set `hardware_encode: false`
- Use webcam index "0" as camera source

## Dependencies

- `ultralytics` - YOLO11 inference
- `opencv-python-headless` - Video capture and processing
- `pymavlink` - MAVLink communication
- `flask` - Web UI
- `numpy`, `scipy` - Numerical operations

## License

MIT
