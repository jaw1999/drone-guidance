# Terminal Guidance

Target tracking system for Raspberry Pi 5 companion computers. Detects objects with YOLO, tracks across frames, and sends control commands to ArduPilot via MAVLink.

## Requirements

- Raspberry Pi 5 (4GB+ RAM)
- Python 3.11+
- FFmpeg
- Camera (USB, CSI, or RTSP)

## Quick Start

```bash
./install.sh
source venv/bin/activate
python run.py
```

Web UI: `http://<pi-ip>:5000`

## QGroundControl Video

1. Application Settings → Video
2. Video Source = `UDP h.264 Video Stream`
3. Port = `5600`

## Configuration

Edit `config/default.yaml`. Key settings:

```yaml
camera:
  rtsp_url: "rtsp://user:pass@192.168.1.10:554/stream"

detector:
  model: "yolov8n"        # or "yolo11n"
  resolution: "640"       # "640", "416", "320"
  target_classes: ["person", "car", "boat"]

mavlink:
  connection: "udp:192.168.1.1:14550"
  enable_control: false

output:
  stream:
    udp_host: "192.168.1.129"
    udp_port: 5600
```

## Documentation

See the [Wiki](../../wiki) for detailed documentation:

- [Architecture](../../wiki/Architecture) - Components, threading, data flow
- [Object Detection](../../wiki/Object-Detection) - YOLO models, NCNN backend
- [Tracking Algorithms](../../wiki/Tracking-Algorithms) - ByteTrack, Kalman filter
- [PID Control](../../wiki/PID-Control) - Controller tuning, anti-windup
- [MAVLink Integration](../../wiki/MAVLink-Integration) - ArduPilot communication
- [Video Streaming](../../wiki/Video-Streaming) - FFmpeg pipeline, overlay
- [Configuration](../../wiki/Configuration) - Full config reference
- [Web API](../../wiki/Web-API) - REST endpoints

## Systemd Service

```bash
./scripts/install-service.sh
sudo systemctl start terminal-guidance
```
