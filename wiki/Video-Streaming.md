# Video Streaming

## Overview

Video streaming is implemented in `src/core/streamer.py`. The system uses FFmpeg to encode H.264 video and stream it via RTP/UDP to QGroundControl.

## Architecture

```
Main Loop
    │
    ├── render_overlay()  (draw on frame in-place)
    │
    └── push_frame()
            │
            ▼
        Frame Queue (deque, maxsize=2)
            │
            ▼
        Write Thread
            │
            ▼
        FFmpeg stdin (raw BGR24)
            │
            ▼
        H.264 Encoding (libx264 ultrafast)
            │
            ▼
        RTP/UDP Output ──► QGroundControl
```

## FFmpeg Pipeline

### Input

Raw BGR24 frames written to FFmpeg stdin:

```python
frame.tobytes()  # shape (height, width, 3), dtype uint8
```

### Command

```bash
ffmpeg -y \
    -fflags +nobuffer \
    -flags +low_delay \
    -f rawvideo \
    -pix_fmt bgr24 \
    -s 1280x720 \
    -r 30 \
    -i - \
    -c:v libx264 \
    -preset ultrafast \
    -tune zerolatency \
    -b:v 2000k \
    -bufsize 2000k \
    -g 30 \
    -bf 0 \
    -f rtp \
    -sdp_file /tmp/stream.sdp \
    rtp://192.168.1.129:5600
```

### Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `-fflags +nobuffer` | - | Disable input buffering |
| `-flags +low_delay` | - | Low-latency mode |
| `-pix_fmt bgr24` | - | OpenCV native format |
| `-preset ultrafast` | - | Fastest encoding |
| `-tune zerolatency` | - | Minimize latency |
| `-g 30` | = FPS | One keyframe per second |
| `-bf 0` | - | No B-frames (reduces latency) |
| `-bufsize` | = bitrate | Small buffer |

### Encoder Detection

The system checks for available encoders in order:

1. `h264_videotoolbox` (macOS hardware)
2. `libx264` (software, most common)
3. `h264` (fallback)

Pi 5 does not have hardware H.264 encoding, so `libx264` is used.

## Frame Queue

Frames are queued for the write thread:

```python
_frame_queue: Deque[np.ndarray] = deque(maxlen=2)
```

- `maxlen=2` drops old frames if queue is full
- Write thread takes latest frame and discards older ones
- Prevents latency buildup when encoding is slower than capture

## Thread Safety

| Resource | Protection |
|----------|------------|
| Frame queue | `_queue_lock` (threading.Lock) |
| FFmpeg process | `_process_lock` (threading.Lock) |
| Queue signal | `_frame_event` (threading.Event) |

## Write Loop

```python
def _write_loop(self):
    while self._running:
        # Check process health
        if process.poll() is not None:
            break

        # Wait for frame signal (timeout 33ms = 30fps)
        if not self._frame_event.wait(timeout=0.033):
            continue

        # Get latest frame (discard older)
        frame = None
        with self._queue_lock:
            while self._frame_queue:
                frame = self._frame_queue.popleft()

        if frame is not None:
            try:
                self._process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                break
```

## Overlay Rendering

### Colors (BGR)

| Element | Color |
|---------|-------|
| Detection box | Green (0, 255, 0) |
| Locked target | Red (0, 0, 255) |
| Acquiring | Yellow (0, 255, 255) |
| Crosshair | White (255, 255, 255) |
| Text background | Black (0, 0, 0) |
| Text | White (255, 255, 255) |

### Crosshair

Drawn at frame center:
```python
size, gap = 20, 5
# Horizontal lines
cv2.line(frame, (cx - size, cy), (cx - gap, cy), white, 1)
cv2.line(frame, (cx + gap, cy), (cx + size, cy), white, 1)
# Vertical lines
cv2.line(frame, (cx, cy - size), (cx, cy - gap), white, 1)
cv2.line(frame, (cx, cy + gap), (cx, cy + size), white, 1)
# Center dot
cv2.circle(frame, (cx, cy), 2, white, -1)
```

### Detection Boxes

```python
cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
label = f"{class_name} {confidence:.0%}"
_draw_label(frame, label, x1, y1 - 5, color)
```

### Locked Target

Enhanced visibility with:
1. Thicker box (box_thickness + 1)
2. Corner brackets
3. Center cross marker
4. "LOCKED:" prefix on label

```python
# Corner brackets (example: top-left)
cv2.line(frame, (x1, y1), (x1 + bracket, y1), color, thickness)
cv2.line(frame, (x1, y1), (x1, y1 + bracket), color, thickness)

# Center marker
cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 10, 1)
```

### Telemetry Display

Right-aligned in top-right corner:

```
STATE: LOCKED          (color by state)
TARGETS: 3
ERR: X:+5.2% Y:-3.1%

FPS: 30.0
DET: 9.5/s
INF: 110ms
ALT: 45.0m
SPD: 12.3m/s
HDG: 270°
BAT: 85%               (yellow if <30%, red if <15%)
```

### Text Rendering

Text is drawn with black background for visibility:

```python
def _draw_label(frame, text, x, y, color):
    tw, th = cv2.getTextSize(text, font, font_scale, 1)[0]
    # Background
    cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 2, y + 2), black, -1)
    # Text
    cv2.putText(frame, text, (x, y - 2), font, font_scale, color, 1, cv2.LINE_8)
```

Text size is cached via `lru_cache` for performance:

```python
@lru_cache(maxsize=256)
def _text_size_cached(text, font, scale):
    return cv2.getTextSize(text, font, scale, 1)[0]
```

## Input Validation

Host and port are validated before starting FFmpeg to prevent command injection:

```python
# IP address pattern
_IP_PATTERN = r'^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$'

# Hostname pattern
_HOSTNAME_PATTERN = r'^(?=.{1,253}$)(?:(?!-)[A-Za-z0-9-]{1,63}(?<!-)\.)*(?!-)[A-Za-z0-9-]{1,63}(?<!-)$'

def _validate_host(host):
    if any(c in host for c in ';|&$`\n\r"\''):
        return False
    return _IP_PATTERN.match(host) or _HOSTNAME_PATTERN.match(host)

def _validate_port(port):
    return 1 <= int(port) <= 65535
```

## QGroundControl Setup

1. Open QGC > Application Settings > Video
2. Set "Video Source" to "UDP h.264 Video Stream"
3. Set "UDP Port" to 5600 (or configured port)
4. The stream SDP file is saved to `/tmp/stream.sdp` if needed

## Configuration

```yaml
output:
  stream:
    enabled: true
    udp_host: "192.168.1.129"   # QGC machine IP
    udp_port: 5600              # QGC video port

  resolution:
    width: 1280
    height: 720

  fps: 30
  bitrate_kbps: 2000

  overlay:
    show_detections: true
    show_locked_target: true
    show_tracking_info: true
    show_telemetry: true
    font_scale: 0.6
    box_thickness: 2
```

## Error Handling

### FFmpeg Not Found

```python
if not shutil.which("ffmpeg"):
    logger.warning("FFmpeg not found - streaming disabled")
    return True  # Soft fail
```

### Immediate Startup Failure

```python
time.sleep(0.5)
if self._process.poll() is not None:
    stderr = self._process.stderr.read().decode()
    logger.error(f"FFmpeg failed: {stderr[:500]}")
```

### Broken Pipe

Write thread catches `BrokenPipeError` and stops gracefully.

### Cleanup

```python
def stop(self):
    self._running = False
    self._frame_event.set()  # Wake write thread

    # Close stdin
    if self._process.stdin:
        self._process.stdin.close()

    # Wait for exit
    self._process.wait(timeout=2.0)

    # Force kill if needed
    self._process.kill()
```

## Performance Notes

- Frame resize happens if output resolution differs from input
- `frame.tobytes()` creates a copy; unavoidable for FFmpeg stdin
- Overlay rendering modifies frame in-place (no copy needed)
- Text size caching reduces OpenCV calls
