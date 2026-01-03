"""
Terminal Guidance - Main Application.

Coordinates all system components: camera capture, object detection,
target tracking, PID control, MAVLink communication, and video streaming.
"""

# Thread configuration must be set before NumPy/NCNN imports
import os
os.environ["OMP_NUM_THREADS"] = "3"
os.environ["OPENBLAS_NUM_THREADS"] = "3"
os.environ["MKL_NUM_THREADS"] = "3"

import argparse
import logging
import signal
import sys
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from .core.camera import CameraCapture, CameraConfig
from .core.detector import DetectorConfig, ObjectDetector
from .core.mavlink_controller import (
    MAVLinkConfig,
    MAVLinkController,
    SafetyConfig,
    TrackingCommand,
)
from .core.pid import PIDConfig, PIDController
from .core.pipeline import Pipeline, PipelineConfig
from .core.streamer import StreamerConfig, UDPStreamer
from .core.tracker import TargetTracker, TrackerConfig, TrackingState
from .utils.config import load_config

logger = logging.getLogger(__name__)

# Processing constants
MAX_CONSECUTIVE_ERRORS = 10
FRAME_POLL_INTERVAL = 0.001
PERF_LOG_INTERVAL = 100


class TerminalGuidance:
    """
    Main application orchestrating all guidance system components.

    Runs a main processing loop that:
    1. Captures frames from camera
    2. Runs detection through async pipeline
    3. Updates tracking and PID control
    4. Sends commands via MAVLink
    5. Streams video with overlay
    """

    def __init__(self, config_path: str = "config/default.yaml"):
        self.config = load_config(config_path)
        self._config_path = config_path

        # Components (initialized in initialize())
        self._camera: Optional[CameraCapture] = None
        self._detector: Optional[ObjectDetector] = None
        self._tracker: Optional[TargetTracker] = None
        self._pipeline: Optional[Pipeline] = None
        self._pid: Optional[PIDController] = None
        self._mavlink: Optional[MAVLinkController] = None
        self._streamer: Optional[UDPStreamer] = None

        # Runtime state
        self._running = False
        self._main_thread: Optional[threading.Thread] = None
        self._preview_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        # Performance metrics
        self._fps = 0.0
        self._detection_fps = 0.0
        self._inference_ms = 0.0

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def initialize(self) -> bool:
        """Initialize all components. Returns True on success."""
        logger.info("Initializing Terminal Guidance...")

        # Camera
        cam_config = CameraConfig.from_dict(self.config)
        self._camera = CameraCapture(cam_config)

        # Detector
        det_config = DetectorConfig.from_dict(self.config)
        logger.info(f"Using {det_config.model} @ {det_config.resolution}px (NCNN)")
        self._detector = ObjectDetector(det_config)

        if not self._detector.initialize():
            logger.error("Failed to initialize detector")
            return False

        # Tracker
        res = self.config.get("camera", {}).get("resolution", {})
        frame_size = (res.get("width", 1920), res.get("height", 1080))
        tracker_config = TrackerConfig.from_dict(self.config)
        self._tracker = TargetTracker(tracker_config, frame_size)

        # Pipeline (async detection with interpolation)
        pipeline_config = PipelineConfig.from_dict(self.config)
        self._pipeline = Pipeline(self._detector, self._tracker, pipeline_config)

        # PID controller
        pid_config = PIDConfig.from_dict(self.config)
        self._pid = PIDController(pid_config)

        # MAVLink controller
        mav_config = MAVLinkConfig.from_dict(self.config)
        safety_config = SafetyConfig.from_dict(self.config)
        self._mavlink = MAVLinkController(mav_config, safety_config)
        self._mavlink.set_tracking_command_callback(self._handle_tracking_command)

        # Video streamer
        stream_config = StreamerConfig.from_dict(self.config)
        self._streamer = UDPStreamer(stream_config)

        logger.info("Initialization complete")
        return True

    def start(self) -> bool:
        """Start all components and main loop."""
        if not self._camera.start():
            logger.warning("Camera not available - configure via web UI")

        self._pipeline.start()
        self._mavlink.start()
        self._streamer.start()

        self._running = True
        self._main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self._main_thread.start()

        logger.info("Terminal Guidance started")
        return True

    def stop(self) -> None:
        """Stop all components gracefully."""
        logger.info("Stopping Terminal Guidance...")

        self._running = False
        if self._pid:
            self._pid.disable()

        if self._main_thread:
            self._main_thread.join(timeout=3.0)

        if self._pipeline:
            self._pipeline.stop()
        if self._camera:
            self._camera.stop()
        if self._mavlink:
            self._mavlink.stop()
        if self._streamer:
            self._streamer.stop()
        if self._detector:
            self._detector.shutdown()

        logger.info("Terminal Guidance stopped")

    # -------------------------------------------------------------------------
    # Main Processing Loop
    # -------------------------------------------------------------------------

    def _main_loop(self) -> None:
        """Process frames continuously until stopped."""
        lost_target_time: Optional[float] = None
        consecutive_errors = 0

        # Cache config for hot loop
        search_timeout = self.config.get("safety", {}).get("search_timeout", 10.0)
        cam_fps = self.config.get("camera", {}).get("fps", 30)
        target_frame_time = 1.0 / cam_fps

        while self._running:
            loop_start = time.time()

            try:
                frame = self._camera.get_frame(copy=False)
                if frame is None:
                    time.sleep(FRAME_POLL_INTERVAL)
                    continue

                if frame.size == 0 or len(frame.shape) < 2:
                    logger.warning("Invalid frame, skipping")
                    continue

                # Process through detection pipeline
                frame_data = self._pipeline.process_frame(frame)

                # Update metrics
                self._fps = self._pipeline.fps
                self._detection_fps = self._pipeline.detection_fps
                self._inference_ms = self._pipeline.inference_ms

                # Periodic performance logging
                if self._pipeline._frame_count % PERF_LOG_INTERVAL == 0:
                    loop_ms = (time.time() - loop_start) * 1000
                    logger.info(
                        f"[PERF] Cam: {self._camera.actual_fps:.1f}, "
                        f"Pipe: {self._fps:.1f}, Det: {self._detection_fps:.1f}, "
                        f"Inf: {self._inference_ms:.0f}ms, Loop: {loop_ms:.0f}ms"
                    )

                # Handle target lost timeout
                state = frame_data.tracking_state
                if state == TrackingState.LOST:
                    if lost_target_time is None:
                        lost_target_time = time.time()
                    elif time.time() - lost_target_time > search_timeout:
                        logger.warning("Target lost timeout - executing safety action")
                        self._mavlink.execute_lost_target_action()
                        lost_target_time = None
                else:
                    lost_target_time = None

                # PID control when locked
                locked_target = frame_data.locked_target
                if self._pid.enabled and locked_target:
                    self._run_pid_control(frame, locked_target)

                # Build telemetry and render overlay
                telemetry = self._build_telemetry(frame, frame_data)
                self._streamer.render_overlay(
                    frame,
                    frame_data.tracked_objects,
                    locked_target,
                    state,
                    telemetry,
                )

                # Push to UDP stream
                self._streamer.push_frame(
                    frame,
                    frame_data.tracked_objects,
                    locked_target,
                    state,
                    telemetry,
                )

                # Cache for web preview
                with self._frame_lock:
                    self._preview_frame = frame

                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Main loop error: {e}", exc_info=True)

                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.critical(f"Too many errors ({consecutive_errors}), stopping")
                    self._running = False
                    break

                time.sleep(0.1)
                continue

            # Rate limit to target FPS
            elapsed = time.time() - loop_start
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)

    def _run_pid_control(self, frame: np.ndarray, locked_target) -> None:
        """Calculate and send PID control commands."""
        error = self._tracker.get_tracking_error()

        # Calculate target size ratio for throttle
        bbox = locked_target.bbox
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        frame_area = frame.shape[0] * frame.shape[1]
        target_size = area / frame_area if frame_area > 0 else None

        output = self._pid.update(error, target_size)

        if output.is_active:
            self._mavlink.send_rate_commands(
                output.yaw_rate,
                output.pitch_rate,
                output.throttle_rate,
            )

    def _build_telemetry(self, frame: np.ndarray, frame_data) -> Dict[str, Any]:
        """Build telemetry dict for overlay display."""
        vehicle = self._mavlink.vehicle_state
        locked = frame_data.locked_target
        h, w = frame.shape[:2]

        # Calculate tracking error percentage
        error_x, error_y = 0.0, 0.0
        if locked and w > 0 and h > 0:
            cx, cy = locked.center
            error_x = (cx - w / 2) / (w / 2) * 100
            error_y = (cy - h / 2) / (h / 2) * 100

        return {
            "state": frame_data.tracking_state.value,
            "targets": len(frame_data.tracked_objects),
            "error_x": error_x,
            "error_y": error_y,
            "fps": self._fps,
            "detection_fps": self._detection_fps,
            "inference_ms": self._inference_ms,
            "connected": self._mavlink.is_connected if self._mavlink else False,
            "altitude": vehicle.altitude_rel if vehicle else 0,
            "speed": vehicle.groundspeed if vehicle else 0,
            "heading": vehicle.heading if vehicle else 0,
            "battery": vehicle.battery_percent if vehicle else 0,
        }

    # -------------------------------------------------------------------------
    # Target Control API
    # -------------------------------------------------------------------------

    def lock_target(self, target_id: int) -> bool:
        """Lock onto specific target by ID."""
        return self._tracker.lock_target(target_id) if self._tracker else False

    def auto_lock(self) -> bool:
        """Lock onto highest-confidence target."""
        if not self._tracker:
            return False

        targets = self._tracker.all_targets
        if not targets:
            return False

        best = max(targets.values(), key=lambda t: t.confidence)
        return self._tracker.lock_target(best.object_id)

    def unlock_target(self) -> None:
        """Release target lock."""
        if self._tracker:
            self._tracker.unlock()

    def enable_control(self) -> bool:
        """Enable PID tracking control."""
        if self._mavlink and self._mavlink.enable_tracking():
            if self._pid:
                self._pid.enable()
            return True
        return False

    def disable_control(self) -> None:
        """Disable PID tracking control."""
        if self._pid:
            self._pid.disable()
        if self._mavlink:
            self._mavlink.disable_tracking()

    def emergency_stop(self) -> None:
        """Trigger emergency stop."""
        if self._pid:
            self._pid.disable()
        if self._mavlink:
            self._mavlink.emergency_stop()

    def clear_emergency(self) -> None:
        """Clear emergency stop state."""
        if self._mavlink:
            self._mavlink.clear_emergency_stop()

    def _handle_tracking_command(self, cmd: TrackingCommand, param: float) -> bool:
        """Handle tracking commands from QGC via MAVLink."""
        logger.info(f"Handling tracking command: {cmd.name}")

        handlers = {
            TrackingCommand.AUTO_LOCK: lambda: self.auto_lock(),
            TrackingCommand.LOCK_TARGET: lambda: self.lock_target(int(param)),
            TrackingCommand.UNLOCK: lambda: (self.unlock_target(), True)[1],
            TrackingCommand.ENABLE_CONTROL: lambda: self.enable_control(),
            TrackingCommand.DISABLE_CONTROL: lambda: (self.disable_control(), True)[1],
        }

        handler = handlers.get(cmd)
        return handler() if handler else False

    # -------------------------------------------------------------------------
    # Status API
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get current system status for web API."""
        vehicle = self._mavlink.vehicle_state if self._mavlink else None
        locked = self._tracker.locked_target if self._tracker else None

        targets = []
        if self._tracker:
            for obj_id, obj in self._tracker.all_targets.items():
                targets.append({
                    "id": obj_id,
                    "class": obj.class_name,
                    "confidence": obj.confidence,
                    "center": obj.center,
                })

        return {
            "tracking_state": self._tracker.state.value if self._tracker else "unknown",
            "control_enabled": self._pid.enabled if self._pid else False,
            "locked_target_id": locked.object_id if locked else None,
            "targets": targets,
            "fps": self._fps,
            "detection_fps": self._detection_fps,
            "inference_ms": self._inference_ms,
            "altitude": vehicle.altitude_rel if vehicle else 0,
            "speed": vehicle.groundspeed if vehicle else 0,
            "heading": vehicle.heading if vehicle else 0,
            "battery": vehicle.battery_percent if vehicle else 0,
            "armed": vehicle.armed if vehicle else False,
            "connected": self._mavlink.is_connected if self._mavlink else False,
            "camera_connected": self._camera.is_connected if self._camera else False,
            "mavlink_connected": self._mavlink.is_connected if self._mavlink else False,
        }

    def get_preview_frame(self) -> Optional[np.ndarray]:
        """Get cached preview frame for web UI."""
        with self._frame_lock:
            return self._preview_frame

    # -------------------------------------------------------------------------
    # Configuration & Restart
    # -------------------------------------------------------------------------

    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update configuration and restart affected components."""
        try:
            # Determine what needs restart
            restart_streamer = "output" in new_config and "stream" in new_config.get("output", {})
            restart_mavlink = "mavlink" in new_config
            restart_detector = "detector" in new_config and "target_classes" in new_config.get("detector", {})

            # Merge new config
            self._merge_config(self.config, new_config)

            # Restart affected components
            if restart_streamer:
                self.restart_streamer()
            if restart_mavlink:
                self.restart_mavlink()
            if restart_detector:
                self.restart_detector()

            # Update PID in-place (no restart needed)
            if "pid" in new_config and self._pid:
                pid_cfg = new_config["pid"]
                if "yaw" in pid_cfg:
                    self._pid.update_gains("yaw", **pid_cfg["yaw"])
                if "pitch" in pid_cfg:
                    self._pid.update_gains("pitch", **pid_cfg["pitch"])
                if "dead_zone_percent" in pid_cfg:
                    self._pid.dead_zone = pid_cfg["dead_zone_percent"] / 100.0

            return True

        except Exception as e:
            logger.error(f"Config update failed: {e}")
            return False

    def _merge_config(self, base: dict, update: dict) -> None:
        """Recursively merge update into base config."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def restart_camera(self) -> None:
        """Restart camera with current config."""
        logger.info("Restarting camera...")
        if self._camera:
            self._camera.stop()

        cam_config = CameraConfig.from_dict(self.config)
        self._camera = CameraCapture(cam_config)
        self._camera.start()
        logger.info("Camera restarted")

    def restart_detector(self) -> None:
        """Restart detector and pipeline with current config."""
        logger.info("Restarting detector...")
        if self._pipeline:
            self._pipeline.stop()
        if self._detector:
            self._detector.shutdown()

        det_config = DetectorConfig.from_dict(self.config)
        self._detector = ObjectDetector(det_config)
        self._detector.initialize()

        if self._tracker:
            pipeline_config = PipelineConfig.from_dict(self.config)
            self._pipeline = Pipeline(self._detector, self._tracker, pipeline_config)
            self._pipeline.start()

        logger.info("Detector restarted")

    def restart_streamer(self) -> None:
        """Restart UDP streamer with current config."""
        logger.info("Restarting streamer...")
        if self._streamer:
            self._streamer.stop()

        stream_config = StreamerConfig.from_dict(self.config)
        self._streamer = UDPStreamer(stream_config)
        self._streamer.start()
        logger.info("Streamer restarted")

    def restart_mavlink(self) -> None:
        """Restart MAVLink controller with current config."""
        logger.info("Restarting MAVLink...")
        if self._mavlink:
            self._mavlink.stop()

        mav_config = MAVLinkConfig.from_dict(self.config)
        safety_config = SafetyConfig.from_dict(self.config)
        self._mavlink = MAVLinkController(mav_config, safety_config)
        self._mavlink.set_tracking_command_callback(self._handle_tracking_command)
        self._mavlink.start()
        logger.info("MAVLink restarted")

    def restart_all(self) -> None:
        """Restart all services."""
        logger.info("Restarting all services...")
        self.restart_camera()
        self.restart_detector()
        self.restart_streamer()
        self.restart_mavlink()
        logger.info("All services restarted")


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------

def main():
    """Application entry point."""
    parser = argparse.ArgumentParser(
        description="Terminal Guidance - Drone Companion Computer"
    )
    parser.add_argument("-c", "--config", default="config/default.yaml",
                        help="Config file path")
    parser.add_argument("--no-web", action="store_true",
                        help="Disable web UI")
    parser.add_argument("--log-level", default="INFO",
                        help="Logging level")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    app = TerminalGuidance(args.config)

    # Signal handlers
    def shutdown(signum, frame):
        logger.info("Shutdown signal received")
        app.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Initialize and start
    if not app.initialize():
        logger.error("Initialization failed")
        sys.exit(1)

    if not app.start():
        logger.error("Startup failed")
        sys.exit(1)

    # Run web UI if enabled
    web_config = app.config.get("web", {})
    if web_config.get("enabled", True) and not args.no_web:
        from .web.app import create_app, run_web_server

        flask_app = create_app(app)
        host = web_config.get("host", "0.0.0.0")
        port = web_config.get("port", 5000)

        logger.info(f"Web UI: http://{host}:{port}")
        run_web_server(flask_app, host, port)
    else:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    app.stop()


if __name__ == "__main__":
    main()
