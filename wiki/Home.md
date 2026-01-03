# Terminal Guidance

Target tracking system for drone companion computers. Runs on Raspberry Pi 5. Detects objects using YOLO, tracks them across frames, and sends control commands to ArduPilot flight controllers via MAVLink.

## System Components

| Component | File | Purpose |
|-----------|------|---------|
| Camera | `src/core/camera.py` | Threaded frame capture from RTSP/webcam |
| Detector | `src/core/detector.py` | YOLO inference via Ultralytics NCNN backend |
| Pipeline | `src/core/pipeline.py` | Async detection with position interpolation |
| Tracker | `src/core/tracker.py` | Multi-object tracking (ByteTrack/Centroid) |
| PID | `src/core/pid.py` | Control loop for yaw/pitch/throttle |
| MAVLink | `src/core/mavlink_controller.py` | Flight controller communication |
| Streamer | `src/core/streamer.py` | H.264/UDP video output via FFmpeg |
| Web UI | `src/web/app.py` | Flask configuration interface |

## Data Flow

```
Camera (30 fps)
    │
    ▼
Pipeline.process_frame()
    │
    ├──[every N frames]──► DetectionWorker (background thread)
    │                           │
    │                           ▼
    │                      ObjectDetector.detect()
    │                           │
    │                           ▼
    │                      TargetTracker.update()
    │                           │
    │◄──────────────────────────┘
    │
    ▼
TrackingInterpolator.interpolate()
    │
    ▼
PIDController.update() ──► MAVLinkController.send_rate_commands()
    │
    ▼
UDPStreamer.push_frame()
```

## Configuration

All settings are in `config/default.yaml`. The config is validated on load with range clamping. See [[Configuration]] for full reference.

## Documentation

- [[Architecture]] - Component design and threading model
- [[Object-Detection]] - YOLO models, NCNN backend, inference pipeline
- [[Tracking-Algorithms]] - ByteTrack, Centroid, Kalman filter math
- [[PID-Control]] - Control theory, anti-windup, derivative filtering
- [[MAVLink-Integration]] - Protocol, message types, command handling
- [[Video-Streaming]] - FFmpeg pipeline, H.264 encoding, UDP output
- [[Configuration]] - Config file reference
- [[Web-API]] - REST endpoint documentation

## Requirements

- Raspberry Pi 5 (tested), Pi 4 (untested)
- Python 3.9+
- FFmpeg with libx264
- Camera source (RTSP IP camera, USB webcam, or video file)
- ArduPilot flight controller (optional, for control output)

## Installation

```bash
git clone <repo>
cd terminal_guidance
./install.sh
```

## Usage

```bash
source venv/bin/activate
python run.py
```

Web UI available at `http://<pi-ip>:5000`

## Thread Model

| Thread | Purpose |
|--------|---------|
| Main | Frame polling, PID control, overlay rendering |
| CameraCapture | Continuous frame capture from source |
| DetectionWorker | YOLO inference (CPU-bound) |
| MAVLink Heartbeat | 1 Hz heartbeat transmission |
| MAVLink Receive | Message reception and parsing |
| FFmpeg stdin | Frame encoding (subprocess) |

## Measured Performance (Pi 5 @ 3GHz)

| Model | Resolution | Inference Time |
|-------|------------|----------------|
| YOLOv8n | 640x640 | 110-130 ms |
| YOLOv8n | 416x416 | ~44 ms |
| YOLOv8n | 320x320 | ~35 ms |
| YOLO11n | 640x640 | 105-125 ms |

With `detection_interval: 3`, pipeline runs at camera FPS (30) while detection runs at ~10 Hz for 640px models.
