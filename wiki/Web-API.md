# Web API

## Overview

The web interface is implemented in `src/web/app.py` using Flask. It provides a REST API for configuration and control, plus a web UI for monitoring.

## Base URL

```
http://<pi-ip>:5000
```

Default port is 5000, configurable via `web.port` in config.

## Endpoints

### Status

#### GET /api/status

Current system status.

**Response:**
```json
{
  "tracking_state": "locked",
  "control_enabled": false,
  "locked_target_id": 5,
  "targets": [
    {"id": 5, "class": "person", "confidence": 0.87, "center": [640, 360]}
  ],
  "fps": 30.2,
  "detection_fps": 9.5,
  "inference_ms": 112.3,
  "altitude": 45.0,
  "speed": 12.3,
  "heading": 270,
  "battery": 85,
  "armed": true,
  "connected": true,
  "camera_connected": true,
  "mavlink_connected": true
}
```

#### GET /api/health

System health metrics.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1704295200.0,
  "system": {
    "cpu_percent": 45.2,
    "cpu_temp_c": 52.0,
    "memory_percent": 38.5,
    "memory_available_mb": 2048,
    "disk_percent": 25.0
  },
  "components": {
    "detector": true,
    "mavlink": true,
    "tracking": false
  },
  "uptime_seconds": 3600.0
}
```

Status values: `healthy`, `degraded`, `unhealthy`

#### GET /api/services/status

Individual service status.

**Response:**
```json
{
  "detector": {"running": true},
  "camera": {"running": true},
  "streamer": {"running": true},
  "mavlink": {"running": false}
}
```

### Configuration

#### GET /api/config

Get current configuration.

**Response:** Full config object (same structure as `config/default.yaml`)

#### POST /api/config

Update configuration.

**Request Body:** Partial config object with values to update

**Example:**
```json
{
  "pid": {
    "yaw": {"kp": 0.4}
  }
}
```

**Response:**
```json
{"status": "ok", "message": "Configuration updated"}
```

**Validation:**
- Config depth limited to 5 levels
- Array length limited to 100 items
- Network parameters validated (IP/hostname, port range)
- Camera URL scheme validated (rtsp, http, https, file)
- MAVLink connection string validated

### Tracking Control

#### POST /api/tracking/lock

Lock onto a target.

**Request Body (optional):**
```json
{"target_id": 5}
```

If `target_id` is omitted, auto-locks onto highest-confidence target.

**Response:**
```json
{"status": "ok", "target_id": 5}
```

**Errors:**
- 404: Target not found / No targets available

#### POST /api/tracking/unlock

Release target lock.

**Response:**
```json
{"status": "ok"}
```

#### POST /api/tracking/enable

Enable PID control output to flight controller.

**Response:**
```json
{"status": "ok"}
```

**Errors:**
- 400: Failed - check MAVLink connection and config

Fails if:
- MAVLink not connected
- `mavlink.enable_control` is false
- Emergency stop is active
- Vehicle not armed (if `safety.require_arm_confirmation` is true)

#### POST /api/tracking/disable

Disable PID control output.

**Response:**
```json
{"status": "ok"}
```

### Emergency

#### POST /api/emergency-stop

Trigger emergency stop.

**Response:**
```json
{"status": "ok"}
```

Sets emergency flag, disables tracking, switches to LOITER mode.

#### POST /api/emergency-stop/clear

Clear emergency stop state.

**Response:**
```json
{"status": "ok"}
```

### Detection Model

#### POST /api/detector/switch

Switch detection model or resolution.

**Request Body:**
```json
{
  "model": "yolov8n",
  "resolution": "640"
}
```

**Valid models:** `yolov8n`, `yolo11n`

**Valid resolutions:** `640`, `416`, `320`

**Response:**
```json
{
  "status": "ok",
  "model": "yolov8n",
  "resolution": "640",
  "message": "Switched to yolov8n @ 640px"
}
```

Restarts detector pipeline with new model.

### Service Management

#### POST /api/restart/{service}

Restart a service.

**Valid services:** `detector`, `camera`, `streamer`, `mavlink`, `all`

**Response:**
```json
{"status": "ok", "message": "detector restarted"}
```

## Web UI

The web UI is served at `GET /` and provides:

- Real-time status display (state, targets, FPS)
- Control buttons (Enable/Disable Control, Lock/Unlock, E-Stop)
- Configuration tabs:
  - **Tracking**: Model, resolution, confidence, target classes, tracker settings
  - **Connection**: Camera URL, MAVLink, video stream destination
  - **PID Tuning**: Yaw/Pitch gains, dead zone
  - **Services**: Restart individual components

### Status Polling

The UI polls `/api/status` every 2 seconds and `/api/services/status` every 5 seconds.

## Input Validation

### IP/Hostname

```python
IP_ADDRESS_PATTERN = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
HOSTNAME_PATTERN = r'^(?=.{1,253}$)(?:(?!-)[A-Za-z0-9-]{1,63}(?<!-)\.)*(?!-)[A-Za-z0-9-]{1,63}(?<!-)$'
```

### Camera URL

- Must be string
- Rejected characters: `;`, `|`, `&`, `$`, `` ` ``, `\n`, `\r`
- Allowed schemes: `rtsp`, `http`, `https`, `file`
- Webcam index (digit string) allowed

### MAVLink Connection

- Must match: `^(udp|tcp|serial):[\w./:@-]+$`
- Rejected characters: `;`, `|`, `&`, `$`, `` ` ``, `\n`, `\r`, space

### Port

- Must be integer 1-65535

### Bitrate

- Must be integer 100-50000 kbps

## Response Format

All API responses use standardized format:

```python
{
  "status": "ok" | "error",
  "message": "...",  # Optional
  # Additional fields depending on endpoint
}
```

HTTP status codes:
- 200: Success
- 400: Bad request / validation error
- 404: Not found
- 500: Server error

## Error Handling

Errors are logged and returned with message:

```json
{"status": "error", "message": "Target not found"}
```

## Security Notes

1. No authentication implemented - should only run on trusted network
2. Input validation prevents command injection in URLs/connection strings
3. Config depth/size limits prevent DoS
4. Flask runs in threaded mode with reloader disabled

## Example Usage

```bash
# Get status
curl http://pi:5000/api/status

# Lock target
curl -X POST http://pi:5000/api/tracking/lock

# Update PID gains
curl -X POST http://pi:5000/api/config \
  -H "Content-Type: application/json" \
  -d '{"pid": {"yaw": {"kp": 0.5}}}'

# Switch model
curl -X POST http://pi:5000/api/detector/switch \
  -H "Content-Type: application/json" \
  -d '{"model": "yolo11n", "resolution": "416"}'

# Emergency stop
curl -X POST http://pi:5000/api/emergency-stop
```
