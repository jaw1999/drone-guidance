"""Video streamer with overlay rendering using FFmpeg to UDP.

This module provides video streaming capabilities for the terminal guidance
system, including overlay rendering for tracking visualization and FFmpeg-based
H.264 encoding for UDP streaming to QGroundControl.

Classes:
    OverlayConfig: Configuration for overlay rendering.
    StreamerConfig: Configuration for the UDP streamer.
    OverlayRenderer: Renders tracking overlays on video frames.
    UDPStreamer: Streams H.264 video over UDP using FFmpeg.
"""

import logging
import shutil
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

from .tracker import TrackedObject, TrackingState

logger = logging.getLogger(__name__)


@dataclass
class OverlayConfig:
    """Overlay rendering configuration."""
    show_detections: bool = True
    show_locked_target: bool = True
    show_tracking_info: bool = True
    show_telemetry: bool = True
    font_scale: float = 0.6
    box_thickness: int = 2


@dataclass
class StreamerConfig:
    """Streamer configuration parameters."""
    enabled: bool = True
    udp_host: str = "127.0.0.1"
    udp_port: int = 5600  # QGC default video port
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
        """Create config from dictionary."""
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


class OverlayRenderer:
    """
    Renders tracking overlay on frames.

    Optimized for Pi 5: caches text sizes, uses simple drawing primitives.
    """

    # Colors (BGR)
    COLOR_DETECTION = (0, 255, 0)      # Green
    COLOR_LOCKED = (0, 0, 255)          # Red
    COLOR_ACQUIRING = (0, 255, 255)     # Yellow
    COLOR_CROSSHAIR = (255, 255, 255)   # White
    COLOR_TEXT_BG = (0, 0, 0)           # Black
    COLOR_TEXT = (255, 255, 255)        # White

    def __init__(self, config: OverlayConfig):
        self.config = config
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._text_cache: Dict[str, Tuple[int, int]] = {}
        self._frame_center: Optional[Tuple[int, int]] = None

    def render(
        self,
        frame: np.ndarray,
        objects: Dict[int, TrackedObject],
        locked_target: Optional[TrackedObject],
        tracking_state: TrackingState,
        telemetry: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Render overlay on frame. Modifies frame in-place for speed.

        Returns the same frame reference, now with overlay.
        """
        h, w = frame.shape[:2]

        # Cache frame center
        if self._frame_center is None or self._frame_center != (w // 2, h // 2):
            self._frame_center = (w // 2, h // 2)

        # Draw crosshair at center
        self._draw_crosshair(frame, self._frame_center[0], self._frame_center[1])

        # Draw detections (skip if locked to save cycles)
        if self.config.show_detections and not locked_target:
            for obj_id, obj in objects.items():
                self._draw_detection(frame, obj, self.COLOR_DETECTION)

        # Draw locked target
        if self.config.show_locked_target and locked_target:
            color = self.COLOR_LOCKED
            if tracking_state == TrackingState.ACQUIRING:
                color = self.COLOR_ACQUIRING
            self._draw_locked_target(frame, locked_target, color)

        # Draw tracking info
        if self.config.show_tracking_info:
            self._draw_tracking_info(frame, tracking_state, locked_target, objects)

        # Draw telemetry
        if self.config.show_telemetry and telemetry:
            self._draw_telemetry(frame, telemetry)

        return frame

    def _draw_crosshair(self, frame: np.ndarray, cx: int, cy: int) -> None:
        """Draw center crosshair."""
        size = 20
        gap = 5
        thickness = 1

        # Horizontal lines
        cv2.line(frame, (cx - size, cy), (cx - gap, cy), self.COLOR_CROSSHAIR, thickness)
        cv2.line(frame, (cx + gap, cy), (cx + size, cy), self.COLOR_CROSSHAIR, thickness)

        # Vertical lines
        cv2.line(frame, (cx, cy - size), (cx, cy - gap), self.COLOR_CROSSHAIR, thickness)
        cv2.line(frame, (cx, cy + gap), (cx, cy + size), self.COLOR_CROSSHAIR, thickness)

        # Center dot
        cv2.circle(frame, (cx, cy), 2, self.COLOR_CROSSHAIR, -1)

    def _draw_detection(
        self, frame: np.ndarray, obj: TrackedObject, color: Tuple[int, int, int]
    ) -> None:
        """Draw detection bounding box."""
        x1, y1, x2, y2 = obj.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.config.box_thickness)

        # Label
        label = f"{obj.class_name} {obj.confidence:.0%}"
        self._draw_label(frame, label, x1, y1 - 5, color)

    def _draw_locked_target(
        self, frame: np.ndarray, target: TrackedObject, color: Tuple[int, int, int]
    ) -> None:
        """Draw locked target with enhanced visibility."""
        x1, y1, x2, y2 = target.bbox
        cx, cy = target.center

        # Main box (thicker)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.config.box_thickness + 1)

        # Corner brackets
        bracket_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        thickness = self.config.box_thickness + 1

        # Top-left
        cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), color, thickness)

        # Top-right
        cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), color, thickness)

        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), color, thickness)

        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), color, thickness)

        # Center marker
        cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 10, 1)

        # Label with "LOCKED"
        label = f"LOCKED: {target.class_name} {target.confidence:.0%}"
        self._draw_label(frame, label, x1, y1 - 5, color)

    def _draw_tracking_info(
        self,
        frame: np.ndarray,
        state: TrackingState,
        locked: Optional[TrackedObject],
        objects: Dict[int, TrackedObject],
    ) -> None:
        """Draw tracking status info in top-right (called by _draw_telemetry)."""
        # Now handled in _draw_telemetry for unified top-right display
        pass

    def _draw_telemetry(self, frame: np.ndarray, telemetry: dict) -> None:
        """Draw all metrics in top-right corner."""
        h, w = frame.shape[:2]
        lines = []
        colors = []

        # Get tracking state from telemetry
        state_str = telemetry.get("state", "")
        if state_str:
            state_colors = {
                "searching": (128, 128, 128),
                "acquiring": (0, 255, 255),
                "locked": (0, 255, 0),
                "lost": (0, 0, 255),
            }
            color = state_colors.get(state_str.lower(), self.COLOR_TEXT)
            lines.append(f"STATE: {state_str.upper()}")
            colors.append(color)

        # Target count
        targets = telemetry.get("targets", 0)
        lines.append(f"TARGETS: {targets}")
        colors.append(self.COLOR_TEXT)

        # Error from center
        if "error_x" in telemetry and "error_y" in telemetry:
            lines.append(f"ERR: X:{telemetry['error_x']:+.1f}% Y:{telemetry['error_y']:+.1f}%")
            colors.append(self.COLOR_TEXT)

        # Separator
        lines.append("")
        colors.append(self.COLOR_TEXT)

        # Performance
        if "fps" in telemetry:
            lines.append(f"FPS: {telemetry['fps']:.1f}")
            colors.append(self.COLOR_TEXT)
        if "inference_ms" in telemetry:
            lines.append(f"INF: {telemetry['inference_ms']:.1f}ms")
            colors.append(self.COLOR_TEXT)

        # Flight data (always show, use -- when no data)
        alt = telemetry.get("altitude", 0)
        spd = telemetry.get("speed", 0)
        hdg = telemetry.get("heading", 0)
        bat = telemetry.get("battery", 0)
        connected = telemetry.get("connected", False)

        if connected:
            lines.append(f"ALT: {alt:.1f}m")
            lines.append(f"SPD: {spd:.1f}m/s")
            lines.append(f"HDG: {hdg:.0f}°")
            bat_color = self.COLOR_TEXT if bat > 30 else (0, 255, 255) if bat > 15 else (0, 0, 255)
            lines.append(f"BAT: {bat:.0f}%")
            colors.extend([self.COLOR_TEXT, self.COLOR_TEXT, self.COLOR_TEXT, bat_color])
        else:
            lines.extend(["ALT: --", "SPD: --", "HDG: --", "BAT: --"])
            colors.extend([(100, 100, 100)] * 4)  # Dim gray for disconnected

        # Draw all lines right-aligned in top-right
        y_offset = 25
        for i, line in enumerate(lines):
            if not line:  # Skip empty separator lines
                y_offset += 8
                continue
            text_w, _ = self._get_text_size(line)  # Use cached size
            x = w - text_w - 10
            self._draw_label(frame, line, x, y_offset, colors[i])
            y_offset += 20

    def _get_text_size(self, text: str) -> Tuple[int, int]:
        """Get text size with LRU-style caching."""
        if text not in self._text_cache:
            size = cv2.getTextSize(
                text, self._font, self.config.font_scale, 1
            )[0]
            # Use larger cache with LRU-style eviction (remove oldest half)
            if len(self._text_cache) > 256:
                # Keep the most recent 128 entries
                keys_to_remove = list(self._text_cache.keys())[:128]
                for key in keys_to_remove:
                    del self._text_cache[key]
            self._text_cache[text] = size
        return self._text_cache[text]

    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw text with background."""
        text_w, text_h = self._get_text_size(text)

        # Background
        cv2.rectangle(
            frame,
            (x - 2, y - text_h - 4),
            (x + text_w + 2, y + 2),
            self.COLOR_TEXT_BG,
            -1,
        )

        # Text (use LINE_8 for faster rendering than LINE_AA)
        cv2.putText(
            frame, text, (x, y - 2),
            self._font, self.config.font_scale, color, 1, cv2.LINE_8
        )


class UDPStreamer:
    """
    Video streamer using FFmpeg to send H.264 over UDP to QGroundControl.

    Uses h264_v4l2m2m hardware encoder on Raspberry Pi, libx264 fallback elsewhere.
    QGC Settings: Video Source = UDP h.264 Video Stream, Port = 5600
    """

    def __init__(self, config: StreamerConfig):
        self.config = config
        self._renderer = OverlayRenderer(config.overlay)
        self._process: Optional[subprocess.Popen] = None
        self._running = False
        self._write_thread: Optional[threading.Thread] = None
        # Use deque with maxlen=1 for O(1) operations and auto-discard
        self._frame_queue: Deque[np.ndarray] = deque(maxlen=2)
        self._queue_lock = threading.Lock()
        self._frame_event = threading.Event()  # Signal new frame available

    @property
    def is_running(self) -> bool:
        """Check if streamer is running."""
        return self._running

    @property
    def stream_url(self) -> str:
        """Get the UDP stream destination."""
        return f"udp://{self.config.udp_host}:{self.config.udp_port}"

    def _detect_encoder(self) -> tuple:
        """
        Detect best available H.264 encoder.

        Returns:
            (encoder_name, extra_args) tuple
        """
        try:
            result = subprocess.run(
                ["ffmpeg", "-encoders"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            encoders = result.stdout

            # Priority order: VideoToolbox (Mac) > libx264 (universal)
            # Note: h264_v4l2m2m on Pi 5 is unreliable, use libx264 instead
            if "h264_videotoolbox" in encoders:
                logger.info("Using macOS VideoToolbox encoder")
                return "h264_videotoolbox", ["-realtime", "true", "-prio_speed", "true"]

            if "libx264" in encoders:
                logger.info("Using software encoder (libx264)")
                return "libx264", ["-preset", "ultrafast", "-tune", "zerolatency"]

            # Last resort: try native encoder
            logger.warning("No known H.264 encoder found, trying native")
            return "h264", []

        except Exception as e:
            logger.warning(f"Encoder detection failed: {e}")
            return "libx264", ["-preset", "ultrafast", "-tune", "zerolatency"]

    def start(self) -> bool:
        """Start the FFmpeg UDP video streamer."""
        if not self.config.enabled:
            logger.info("Video streaming disabled")
            return True

        # Check for ffmpeg
        if not shutil.which("ffmpeg"):
            logger.warning("FFmpeg not found - video streaming disabled")
            logger.warning("Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)")
            return True  # Non-fatal

        # Detect encoder
        encoder, encoder_args = self._detect_encoder()

        # FFmpeg pipeline: raw BGR -> H.264 -> RTP -> UDP
        # Optimized for low latency while maintaining QGC compatibility
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-fflags", "+nobuffer",
            "-flags", "+low_delay",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.config.width}x{self.config.height}",
            "-r", str(self.config.fps),
            "-i", "-",
            "-c:v", encoder,
        ]

        # Add encoder-specific args
        ffmpeg_cmd.extend(encoder_args)

        # Output: RTP over UDP for QGC
        # Key latency settings:
        # - Small GOP (keyframe interval) for faster sync
        # - No B-frames (decode order = display order)
        # - Minimal buffer
        ffmpeg_cmd.extend([
            "-b:v", f"{self.config.bitrate_kbps}k",
            "-bufsize", f"{self.config.bitrate_kbps}k",  # 1 second buffer
            "-g", str(self.config.fps),  # Keyframe every 1 sec
            "-bf", "0",
            "-f", "rtp",
            "-sdp_file", "/tmp/stream.sdp",  # Generate SDP for debugging
            f"rtp://{self.config.udp_host}:{self.config.udp_port}",
        ])

        logger.info(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")

        try:
            self._process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            # Give FFmpeg a moment to start and check for immediate failure
            import time
            time.sleep(0.5)
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                logger.error(f"FFmpeg failed to start: {stderr[:500]}")
                self._process = None
                return True  # Non-fatal

        except FileNotFoundError:
            logger.error("FFmpeg not found in PATH")
            return True
        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")
            return True  # Non-fatal

        self._running = True
        self._write_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._write_thread.start()

        logger.info(f"UDP video stream started: {self.stream_url}")
        logger.info(f"QGC: Video Source = 'UDP h.264 Video Stream', Port = {self.config.udp_port}")
        return True

    def stop(self) -> None:
        """Stop the streamer and clean up resources."""
        self._running = False
        self._frame_event.set()  # Wake up write thread if waiting

        if self._write_thread:
            self._write_thread.join(timeout=2.0)
            self._write_thread = None

        if self._process:
            try:
                if self._process.stdin:
                    self._process.stdin.close()
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                logger.warning("FFmpeg did not exit gracefully, killing")
                self._process.kill()
                self._process.wait(timeout=1.0)  # Wait for kill to complete
            except Exception as e:
                logger.warning(f"Error stopping FFmpeg: {e}")
                self._process.kill()
                try:
                    self._process.wait(timeout=1.0)
                except Exception:
                    pass  # Process may already be dead
            finally:
                # Close stderr to prevent resource leak
                if self._process.stderr:
                    try:
                        self._process.stderr.close()
                    except Exception:
                        pass
                self._process = None

        logger.info("Streamer stopped")

    def _write_loop(self) -> None:
        """Background thread to write frames to FFmpeg."""
        while self._running:
            # Check process state before waiting
            if not self._process or self._process.poll() is not None:
                break

            # Wait for frame signal with timeout
            if not self._frame_event.wait(timeout=0.01):  # Tighter timeout
                continue

            self._frame_event.clear()

            # Check running state again after wait
            if not self._running:
                break

            frame = None
            with self._queue_lock:
                if self._frame_queue:
                    frame = self._frame_queue.popleft()  # O(1) with deque

            if frame is not None and self._process and self._process.stdin:
                try:
                    self._process.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    logger.warning("FFmpeg pipe broken")
                    self._running = False
                    break
                except Exception as e:
                    logger.warning(f"Write error: {e}")
                    break

        # Check if process exited with error and log it
        if self._process and self._process.poll() is not None:
            try:
                if self._process.stderr:
                    stderr = self._process.stderr.read().decode()
                    if stderr:
                        logger.warning(f"FFmpeg exited: {stderr[-300:]}")
            except Exception:
                pass  # Ignore errors reading stderr on shutdown

    def render_overlay(
        self,
        frame: np.ndarray,
        objects: Dict[int, TrackedObject],
        locked_target: Optional[TrackedObject],
        tracking_state: TrackingState,
        telemetry: Optional[dict] = None,
    ) -> np.ndarray:
        """Render overlay on frame in-place. Returns the same frame."""
        self._renderer.render(frame, objects, locked_target, tracking_state, telemetry)
        return frame

    def push_frame(
        self,
        frame: np.ndarray,
        objects: Dict[int, TrackedObject],
        locked_target: Optional[TrackedObject],
        tracking_state: TrackingState,
        telemetry: Optional[dict] = None,
    ) -> None:
        """Push a frame to the UDP stream (frame should already have overlay)."""
        if not self._running or not self._process:
            return

        # Resize if needed for output resolution (avoid copy if same size)
        if frame.shape[1] != self.config.width or frame.shape[0] != self.config.height:
            output = cv2.resize(frame, (self.config.width, self.config.height))
        else:
            # Only copy if not already contiguous
            if frame.flags['C_CONTIGUOUS']:
                output = frame
            else:
                output = np.ascontiguousarray(frame)

        # Add to queue - deque maxlen handles discarding old frames
        with self._queue_lock:
            self._frame_queue.append(output)
        self._frame_event.set()  # Signal write thread

    def get_preview_frame(
        self,
        frame: np.ndarray,
        objects: Dict[int, TrackedObject],
        locked_target: Optional[TrackedObject],
        tracking_state: TrackingState,
        telemetry: Optional[dict] = None,
    ) -> np.ndarray:
        """Get frame with overlay for web preview (without streaming)."""
        output = frame.copy()
        self._renderer.render(output, objects, locked_target, tracking_state, telemetry)
        return output

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
