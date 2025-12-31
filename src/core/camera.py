"""Camera capture module for RTSP stream ingestion."""

import cv2
import threading
import time
import logging
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Camera configuration parameters."""
    rtsp_url: str
    width: int = 1920
    height: int = 1080
    fps: int = 30
    buffer_size: int = 1
    reconnect_attempts: int = 5
    reconnect_delay_sec: float = 2.0
    fov_horizontal: float = 90.0
    fov_vertical: float = 60.0

    @classmethod
    def from_dict(cls, config: dict) -> "CameraConfig":
        """Create config from dictionary."""
        cam = config.get("camera", {})
        res = cam.get("resolution", {})
        fov = cam.get("fov", {})
        return cls(
            rtsp_url=cam.get("rtsp_url", ""),
            width=res.get("width", 1920),
            height=res.get("height", 1080),
            fps=cam.get("fps", 30),
            buffer_size=cam.get("buffer_size", 1),
            reconnect_attempts=cam.get("reconnect_attempts", 5),
            reconnect_delay_sec=cam.get("reconnect_delay_sec", 2.0),
            fov_horizontal=fov.get("horizontal", 90.0),
            fov_vertical=fov.get("vertical", 60.0),
        )


class CameraCapture:
    """
    Handles RTSP stream capture with automatic reconnection.

    Uses a separate thread for capture to minimize latency and
    prevent frame buffering issues.
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self._frame_count = 0
        self._fps_actual = 0.0
        self._last_fps_time = time.time()
        self._fps_frame_count = 0
        self._on_frame_callback: Optional[Callable[[np.ndarray], None]] = None

    @property
    def is_connected(self) -> bool:
        """Check if camera is connected."""
        return self._connected

    @property
    def frame_count(self) -> int:
        """Total frames captured."""
        return self._frame_count

    @property
    def actual_fps(self) -> float:
        """Actual measured FPS."""
        return self._fps_actual

    def set_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Set callback for new frames."""
        self._on_frame_callback = callback

    def _parse_source(self) -> tuple:
        """Parse camera source - returns (source, backend) tuple."""
        url = self.config.rtsp_url

        # Check if it's a webcam index (numeric string)
        if url.isdigit():
            return int(url), cv2.CAP_ANY

        # RTSP or HTTP stream
        if url.startswith(("rtsp://", "http://", "https://")):
            # Set FFmpeg options for low latency (UDP)
            import os
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay|framedrop;1"
            return url, cv2.CAP_FFMPEG

        # File path or other source
        return url, cv2.CAP_ANY

    def connect(self) -> bool:
        """Establish connection to camera source."""
        source, backend = self._parse_source()
        source_type = "webcam" if isinstance(source, int) else "stream"
        logger.info(f"Connecting to {source_type}: {source}")

        for attempt in range(self.config.reconnect_attempts):
            try:
                self._cap = cv2.VideoCapture(source, backend)

                # Configure capture
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
                self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)

                # For webcams, also set resolution
                if isinstance(source, int):
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)

                if self._cap.isOpened():
                    # Read a test frame
                    ret, frame = self._cap.read()
                    if ret and frame is not None:
                        self._connected = True
                        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        logger.info(
                            f"Camera connected: {actual_width}x{actual_height}"
                        )
                        return True

            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")

            if attempt < self.config.reconnect_attempts - 1:
                time.sleep(self.config.reconnect_delay_sec)

        logger.error("Failed to connect to camera")
        self._connected = False
        return False

    def start(self) -> bool:
        """Start capture thread."""
        if not self._connected and not self.connect():
            return False

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Camera capture started")
        return True

    def stop(self) -> None:
        """Stop capture thread and release resources."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        self._connected = False
        logger.info("Camera capture stopped")

    def get_frame(self, copy: bool = True) -> Optional[np.ndarray]:
        """Get the latest frame (thread-safe).

        Args:
            copy: If True, returns a copy (safe but slower).
                  If False, returns a view that may be overwritten by capture thread.
                  WARNING: copy=False is unsafe unless caller processes frame immediately.

        Returns:
            Frame as numpy array, or None if no frame available.
        """
        with self._frame_lock:
            if self._frame is None:
                return None
            # Always copy to prevent race condition where capture thread
            # overwrites frame while caller is processing it
            return self._frame.copy() if copy else np.ascontiguousarray(self._frame)

    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread."""
        consecutive_failures = 0
        max_failures = 30  # ~1 second at 30fps

        while self._running:
            if not self._cap or not self._cap.isOpened():
                if not self._reconnect():
                    time.sleep(self.config.reconnect_delay_sec)
                    continue

            ret, frame = self._cap.read()

            if ret and frame is not None:
                consecutive_failures = 0
                self._frame_count += 1
                self._fps_frame_count += 1

                with self._frame_lock:
                    self._frame = frame

                # Call frame callback if set
                if self._on_frame_callback:
                    try:
                        self._on_frame_callback(frame)
                    except Exception as e:
                        logger.error(f"Frame callback error: {e}")

                # Calculate actual FPS every second
                now = time.time()
                elapsed = now - self._last_fps_time
                if elapsed >= 1.0:
                    self._fps_actual = self._fps_frame_count / elapsed
                    self._fps_frame_count = 0
                    self._last_fps_time = now

            else:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.warning("Too many consecutive failures, reconnecting...")
                    self._reconnect()
                    consecutive_failures = 0

    def _reconnect(self) -> bool:
        """Attempt to reconnect to the camera."""
        logger.info("Attempting to reconnect to camera...")
        self._connected = False

        if self._cap:
            self._cap.release()

        return self.connect()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
