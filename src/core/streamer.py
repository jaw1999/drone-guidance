"""
H.264 UDP video streamer with tracking overlay.

Streams video to QGroundControl via RTP/UDP with:
- Tracking overlay (bounding boxes, crosshair, telemetry)
- Low-latency encoding via FFmpeg (libx264 ultrafast)
- Thread-safe frame queue for async operation
"""

import logging
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

from .tracker import TrackedObject, TrackingState

logger = logging.getLogger(__name__)

# Validation patterns for FFmpeg host/port safety
_IP_PATTERN = re.compile(
    r'^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}'
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$'
)
_HOSTNAME_PATTERN = re.compile(
    r'^(?=.{1,253}$)(?:(?!-)[A-Za-z0-9-]{1,63}(?<!-)\.)*'
    r'(?!-)[A-Za-z0-9-]{1,63}(?<!-)$'
)


def _validate_host(host: str) -> bool:
    """Validate host is safe for FFmpeg (no shell injection)."""
    if not host or not isinstance(host, str):
        return False
    host = host.strip()
    if any(c in host for c in ';|&$`\n\r"\''):
        return False
    return bool(_IP_PATTERN.match(host) or _HOSTNAME_PATTERN.match(host))


def _validate_port(port: int) -> bool:
    """Validate port is in valid range."""
    try:
        return 1 <= int(port) <= 65535
    except (ValueError, TypeError):
        return False


@lru_cache(maxsize=256)
def _text_size_cached(text: str, font: int, scale: float) -> Tuple[int, int]:
    """Cached text size lookup."""
    return cv2.getTextSize(text, font, scale, 1)[0]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OverlayConfig:
    """Overlay rendering options."""
    show_detections: bool = True
    show_locked_target: bool = True
    show_tracking_info: bool = True
    show_telemetry: bool = True
    font_scale: float = 0.6
    box_thickness: int = 2


@dataclass
class StreamerConfig:
    """Streamer configuration."""
    enabled: bool = True
    udp_host: str = "127.0.0.1"
    udp_port: int = 5600
    width: int = 1280
    height: int = 720
    fps: int = 30
    bitrate_kbps: int = 2000
    overlay: OverlayConfig = None

    def __post_init__(self):
        if self.overlay is None:
            self.overlay = OverlayConfig()

    @classmethod
    def from_dict(cls, config: dict) -> "StreamerConfig":
        output = config.get("output", {})
        stream = output.get("stream", {})
        res = output.get("resolution", {})
        overlay_cfg = output.get("overlay", {})

        overlay = OverlayConfig(
            show_detections=overlay_cfg.get("show_detections", True),
            show_locked_target=overlay_cfg.get("show_locked_target", True),
            show_tracking_info=overlay_cfg.get("show_tracking_info", True),
            show_telemetry=overlay_cfg.get("show_telemetry", True),
            font_scale=overlay_cfg.get("font_scale", 0.6),
            box_thickness=overlay_cfg.get("box_thickness", 2),
        )

        return cls(
            enabled=stream.get("enabled", True),
            udp_host=stream.get("udp_host", "127.0.0.1"),
            udp_port=stream.get("udp_port", 5000),
            width=res.get("width", 1280),
            height=res.get("height", 720),
            fps=output.get("fps", 30),
            bitrate_kbps=output.get("bitrate_kbps", 2000),
            overlay=overlay,
        )


# =============================================================================
# Overlay Renderer
# =============================================================================

class OverlayRenderer:
    """Renders tracking overlay on frames."""

    # Colors (BGR)
    COLOR_DETECTION = (0, 255, 0)     # Green
    COLOR_LOCKED = (0, 0, 255)        # Red
    COLOR_ACQUIRING = (0, 255, 255)   # Yellow
    COLOR_CROSSHAIR = (255, 255, 255) # White
    COLOR_TEXT_BG = (0, 0, 0)         # Black
    COLOR_TEXT = (255, 255, 255)      # White

    def __init__(self, config: OverlayConfig):
        self.config = config
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._frame_center: Optional[Tuple[int, int]] = None

    def render(self, frame: np.ndarray, objects: Dict[int, TrackedObject],
               locked_target: Optional[TrackedObject], tracking_state: TrackingState,
               telemetry: Optional[dict] = None) -> np.ndarray:
        """Render overlay on frame (modifies in-place)."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # Crosshair
        self._draw_crosshair(frame, cx, cy)

        # Detection boxes (except locked target)
        if self.config.show_detections:
            for obj_id, obj in objects.items():
                if not locked_target or obj_id != locked_target.object_id:
                    self._draw_detection(frame, obj, self.COLOR_DETECTION)

        # Locked target
        if self.config.show_locked_target and locked_target:
            color = self.COLOR_ACQUIRING if tracking_state == TrackingState.ACQUIRING else self.COLOR_LOCKED
            self._draw_locked_target(frame, locked_target, color)

        # Telemetry
        if self.config.show_telemetry and telemetry:
            self._draw_telemetry(frame, telemetry)

        return frame

    def _draw_crosshair(self, frame: np.ndarray, cx: int, cy: int) -> None:
        """Draw center crosshair."""
        size, gap = 20, 5
        cv2.line(frame, (cx - size, cy), (cx - gap, cy), self.COLOR_CROSSHAIR, 1)
        cv2.line(frame, (cx + gap, cy), (cx + size, cy), self.COLOR_CROSSHAIR, 1)
        cv2.line(frame, (cx, cy - size), (cx, cy - gap), self.COLOR_CROSSHAIR, 1)
        cv2.line(frame, (cx, cy + gap), (cx, cy + size), self.COLOR_CROSSHAIR, 1)
        cv2.circle(frame, (cx, cy), 2, self.COLOR_CROSSHAIR, -1)

    def _draw_detection(self, frame: np.ndarray, obj: TrackedObject,
                        color: Tuple[int, int, int]) -> None:
        """Draw detection bounding box."""
        x1, y1, x2, y2 = obj.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.config.box_thickness)
        label = f"{obj.class_name} {obj.confidence:.0%}"
        self._draw_label(frame, label, x1, y1 - 5, color)

    def _draw_locked_target(self, frame: np.ndarray, target: TrackedObject,
                            color: Tuple[int, int, int]) -> None:
        """Draw locked target with enhanced visibility."""
        x1, y1, x2, y2 = target.bbox
        cx, cy = target.center
        thickness = self.config.box_thickness + 1

        # Main box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Corner brackets
        bracket = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        for (bx, by), (dx, dy) in [
            ((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
            ((x1, y2), (1, -1)), ((x2, y2), (-1, -1)),
        ]:
            cv2.line(frame, (bx, by), (bx + dx * bracket, by), color, thickness)
            cv2.line(frame, (bx, by), (bx, by + dy * bracket), color, thickness)

        # Center marker
        cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 10, 1)

        # Label
        label = f"LOCKED: {target.class_name} {target.confidence:.0%}"
        self._draw_label(frame, label, x1, y1 - 5, color)

    def _draw_telemetry(self, frame: np.ndarray, telemetry: dict) -> None:
        """Draw telemetry in top-right corner."""
        h, w = frame.shape[:2]
        lines, colors = [], []

        # Tracking state
        state = telemetry.get("state", "")
        if state:
            state_colors = {
                "searching": (128, 128, 128),
                "acquiring": (0, 255, 255),
                "locked": (0, 255, 0),
                "lost": (0, 0, 255),
            }
            lines.append(f"STATE: {state.upper()}")
            colors.append(state_colors.get(state.lower(), self.COLOR_TEXT))

        # Targets
        lines.append(f"TARGETS: {telemetry.get('targets', 0)}")
        colors.append(self.COLOR_TEXT)

        # Error
        if "error_x" in telemetry:
            lines.append(f"ERR: X:{telemetry['error_x']:+.1f}% Y:{telemetry['error_y']:+.1f}%")
            colors.append(self.COLOR_TEXT)

        # Separator
        lines.append("")
        colors.append(self.COLOR_TEXT)

        # Performance
        if "fps" in telemetry:
            lines.append(f"FPS: {telemetry['fps']:.1f}")
            colors.append(self.COLOR_TEXT)
        if "detection_fps" in telemetry:
            lines.append(f"DET: {telemetry['detection_fps']:.1f}/s")
            colors.append(self.COLOR_TEXT)
        if "inference_ms" in telemetry:
            lines.append(f"INF: {telemetry['inference_ms']:.0f}ms")
            colors.append(self.COLOR_TEXT)

        # Flight data
        connected = telemetry.get("connected", False)
        if connected:
            lines.append(f"ALT: {telemetry.get('altitude', 0):.1f}m")
            lines.append(f"SPD: {telemetry.get('speed', 0):.1f}m/s")
            lines.append(f"HDG: {telemetry.get('heading', 0):.0f}°")
            bat = telemetry.get("battery", 0)
            bat_color = self.COLOR_TEXT if bat > 30 else (0, 255, 255) if bat > 15 else (0, 0, 255)
            lines.append(f"BAT: {bat:.0f}%")
            colors.extend([self.COLOR_TEXT, self.COLOR_TEXT, self.COLOR_TEXT, bat_color])
        else:
            lines.extend(["ALT: --", "SPD: --", "HDG: --", "BAT: --"])
            colors.extend([(100, 100, 100)] * 4)

        # Draw lines right-aligned
        y = 25
        for line, color in zip(lines, colors):
            if not line:
                y += 8
                continue
            tw, _ = _text_size_cached(line, self._font, self.config.font_scale)
            self._draw_label(frame, line, w - tw - 10, y, color)
            y += 20

    def _draw_label(self, frame: np.ndarray, text: str, x: int, y: int,
                    color: Tuple[int, int, int]) -> None:
        """Draw text with background."""
        tw, th = _text_size_cached(text, self._font, self.config.font_scale)
        cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 2, y + 2), self.COLOR_TEXT_BG, -1)
        cv2.putText(frame, text, (x, y - 2), self._font, self.config.font_scale, color, 1, cv2.LINE_8)


# =============================================================================
# UDP Streamer
# =============================================================================

class UDPStreamer:
    """
    FFmpeg-based H.264/UDP streamer for QGroundControl.

    Streams video via RTP/UDP with low-latency encoding.
    Thread-safe frame queue allows async operation.
    """

    def __init__(self, config: StreamerConfig):
        self.config = config
        self._renderer = OverlayRenderer(config.overlay)

        self._process: Optional[subprocess.Popen] = None
        self._process_lock = threading.Lock()
        self._running = False

        self._write_thread: Optional[threading.Thread] = None
        self._frame_queue: Deque[np.ndarray] = deque(maxlen=2)
        self._queue_lock = threading.Lock()
        self._frame_event = threading.Event()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stream_url(self) -> str:
        return f"udp://{self.config.udp_host}:{self.config.udp_port}"

    def start(self) -> bool:
        """Start the FFmpeg UDP streamer."""
        if not self.config.enabled:
            logger.info("Video streaming disabled")
            return True

        # Validate destination
        if not _validate_host(self.config.udp_host):
            logger.error(f"Invalid stream host: {self.config.udp_host!r}")
            return False
        if not _validate_port(self.config.udp_port):
            logger.error(f"Invalid stream port: {self.config.udp_port}")
            return False

        # Check FFmpeg
        if not shutil.which("ffmpeg"):
            logger.warning("FFmpeg not found - streaming disabled")
            return True

        # Detect encoder and build command
        encoder, encoder_args = self._detect_encoder()
        cmd = self._build_ffmpeg_cmd(encoder, encoder_args)

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            # Check for immediate failure
            time.sleep(0.5)
            if self._process.poll() is not None:
                self._log_ffmpeg_error()
                self._process = None
                return True

        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")
            return True

        self._running = True
        self._write_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._write_thread.start()

        logger.info(f"UDP stream started: {self.stream_url}")
        return True

    def stop(self) -> None:
        """Stop streamer and cleanup."""
        self._running = False
        self._frame_event.set()

        if self._write_thread:
            self._write_thread.join(timeout=2.0)

        with self._process_lock:
            if self._process:
                try:
                    if self._process.stdin:
                        self._process.stdin.close()
                    self._process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=1.0)
                except Exception:
                    self._process.kill()
                finally:
                    if self._process.stderr:
                        try:
                            self._process.stderr.close()
                        except Exception:
                            pass
                    self._process = None

        logger.info("Streamer stopped")

    def render_overlay(self, frame: np.ndarray, objects: Dict[int, TrackedObject],
                       locked_target: Optional[TrackedObject], tracking_state: TrackingState,
                       telemetry: Optional[dict] = None) -> np.ndarray:
        """Render overlay on frame in-place."""
        return self._renderer.render(frame, objects, locked_target, tracking_state, telemetry)

    def push_frame(self, frame: np.ndarray, objects: Dict[int, TrackedObject],
                   locked_target: Optional[TrackedObject], tracking_state: TrackingState,
                   telemetry: Optional[dict] = None, _frame_is_owned: bool = False) -> None:
        """Push frame to stream (overlay should already be rendered)."""
        if not self._running or not self._process:
            return

        # Resize if needed
        if frame.shape[1] != self.config.width or frame.shape[0] != self.config.height:
            output = cv2.resize(frame, (self.config.width, self.config.height))
        elif not _frame_is_owned:
            output = frame.copy()
        else:
            output = frame

        with self._queue_lock:
            self._frame_queue.append(output)
        self._frame_event.set()

    def get_preview_frame(self, frame: np.ndarray, objects: Dict[int, TrackedObject],
                          locked_target: Optional[TrackedObject], tracking_state: TrackingState,
                          telemetry: Optional[dict] = None) -> np.ndarray:
        """Get frame with overlay for web preview."""
        output = frame.copy()
        self._renderer.render(output, objects, locked_target, tracking_state, telemetry)
        return output

    def _detect_encoder(self) -> tuple:
        """Detect best available H.264 encoder."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-encoders"],
                capture_output=True, text=True, timeout=5,
            )
            encoders = result.stdout

            if "h264_videotoolbox" in encoders:
                logger.info("Using VideoToolbox encoder")
                return "h264_videotoolbox", ["-realtime", "true"]

            if "libx264" in encoders:
                logger.info("Using libx264 encoder")
                return "libx264", ["-preset", "ultrafast", "-tune", "zerolatency"]

            return "h264", []

        except Exception:
            return "libx264", ["-preset", "ultrafast", "-tune", "zerolatency"]

    def _build_ffmpeg_cmd(self, encoder: str, encoder_args: list) -> list:
        """Build FFmpeg command."""
        cmd = [
            "ffmpeg", "-y",
            "-fflags", "+nobuffer",
            "-flags", "+low_delay",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.config.width}x{self.config.height}",
            "-r", str(self.config.fps),
            "-i", "-",
            "-c:v", encoder,
        ]
        cmd.extend(encoder_args)
        cmd.extend([
            "-b:v", f"{self.config.bitrate_kbps}k",
            "-bufsize", f"{self.config.bitrate_kbps}k",
            "-g", str(self.config.fps),
            "-bf", "0",
            "-f", "rtp",
            "-sdp_file", "/tmp/stream.sdp",
            f"rtp://{self.config.udp_host}:{self.config.udp_port}",
        ])
        return cmd

    def _write_loop(self) -> None:
        """Background thread to write frames to FFmpeg."""
        while self._running:
            with self._process_lock:
                if not self._process or self._process.poll() is not None:
                    break

            if not self._frame_event.wait(timeout=0.033):
                continue

            self._frame_event.clear()

            if not self._running:
                break

            # Get latest frame
            frame = None
            with self._queue_lock:
                while self._frame_queue:
                    frame = self._frame_queue.popleft()

            if frame is not None:
                with self._process_lock:
                    if self._process and self._process.stdin:
                        try:
                            self._process.stdin.write(frame.tobytes())
                        except BrokenPipeError:
                            logger.warning("FFmpeg pipe broken")
                            self._running = False
                            break
                        except Exception as e:
                            logger.warning(f"Write error: {e}")
                            break

    def _log_ffmpeg_error(self) -> None:
        """Log FFmpeg startup error."""
        try:
            if self._process and self._process.stderr:
                stderr = self._process.stderr.read().decode()
                logger.error(f"FFmpeg failed: {stderr[:500]}")
        except Exception:
            pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
