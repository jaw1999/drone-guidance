"""Main application for Terminal Guidance."""

import logging
import os
import signal
import sys
import threading
import time

# Enable OpenMP multi-threading for NCNN (4 cores on Pi 5)
os.environ.setdefault("OMP_NUM_THREADS", "4")
from typing import Any, Dict, List, Optional

# Constants
MAX_CONSECUTIVE_ERRORS = 10
FRAME_POLL_INTERVAL = 0.001  # seconds
LOOP_SLEEP_INTERVAL = 0.001  # seconds
PERF_LOG_INTERVAL = 100  # frames

import numpy as np

from .core.camera import CameraCapture, CameraConfig
from .core.detector import Detection, DetectorConfig, ObjectDetector
from .core.tracker import TargetTracker, TrackerConfig, TrackingState
from .core.pid import PIDConfig, PIDController
from .core.mavlink_controller import (
    MAVLinkConfig,
    MAVLinkController,
    SafetyConfig,
    TrackingCommand,
)
from .core.streamer import UDPStreamer, StreamerConfig
from .core.pipeline import Pipeline, PipelineConfig
from .utils.config import load_config

logger = logging.getLogger(__name__)


class TerminalGuidance:
    """Main application coordinating all components."""

    def __init__(self, config_path: str = "config/default.yaml"):
        self.config = load_config(config_path)
        self._config_path = config_path

        # Components
        self._camera: Optional[CameraCapture] = None
        self._detector: Optional[ObjectDetector] = None
        self._tracker: Optional[TargetTracker] = None
        self._pipeline: Optional[Pipeline] = None
        self._pid: Optional[PIDController] = None
        self._mavlink: Optional[MAVLinkController] = None
        self._streamer: Optional[UDPStreamer] = None

        # State
        self._running = False
        self._main_thread: Optional[threading.Thread] = None
        self._preview_frame: Optional[np.ndarray] = None  # Cached preview with overlay
        self._frame_lock = threading.Lock()

        # Performance metrics (from pipeline)
        self._fps = 0.0
        self._inference_ms = 0.0

    def initialize(self) -> bool:
        """Initialize all components."""
        logger.info("Initializing Terminal Guidance...")

        # Camera
        cam_config = CameraConfig.from_dict(self.config)
        self._camera = CameraCapture(cam_config)

        # Detector
        det_config = DetectorConfig.from_dict(self.config)
        logger.info(f"Using {det_config.backend.upper()} backend for detection")
        self._detector = ObjectDetector(det_config)

        if not self._detector.initialize():
            logger.error("Failed to initialize detector")
            return False

        # Get frame size for tracker
        res = self.config.get("camera", {}).get("resolution", {})
        frame_size = (res.get("width", 1920), res.get("height", 1080))

        # Tracker
        tracker_config = TrackerConfig.from_dict(self.config)
        self._tracker = TargetTracker(tracker_config, frame_size)

        # Pipeline (async detection + interpolation)
        pipeline_config = PipelineConfig.from_dict(self.config)
        self._pipeline = Pipeline(self._detector, self._tracker, pipeline_config)

        # PID controller
        pid_config = PIDConfig.from_dict(self.config)
        self._pid = PIDController(pid_config)

        # MAVLink
        mav_config = MAVLinkConfig.from_dict(self.config)
        safety_config = SafetyConfig.from_dict(self.config)
        self._mavlink = MAVLinkController(mav_config, safety_config)
        self._mavlink.set_tracking_command_callback(self._handle_tracking_command)

        # Streamer
        stream_config = StreamerConfig.from_dict(self.config)
        self._streamer = UDPStreamer(stream_config)

        logger.info("Initialization complete")
        return True

    def start(self) -> bool:
        """Start all components and main loop."""
        if not self._camera.start():
            logger.error("Failed to start camera")
            return False

        # Start pipeline
        self._pipeline.start()

        # Start MAVLink (non-blocking, will try to connect)
        self._mavlink.start()

        # Start UDP video stream
        self._streamer.start()

        # Start main processing loop
        self._running = True
        self._main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self._main_thread.start()

        logger.info("Terminal Guidance started")
        return True

    def stop(self) -> None:
        """Stop all components."""
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

    def _main_loop(self) -> None:
        """Main processing loop."""
        lost_target_time: Optional[float] = None
        consecutive_errors = 0

        # Cache config values to avoid repeated dict lookups in hot loop
        search_timeout = self.config.get("safety", {}).get("search_timeout", 10.0)
        cam_fps = self.config.get("camera", {}).get("fps", 30)
        target_frame_time = 1.0 / cam_fps

        while self._running:
            loop_start = time.time()

            try:
                # Get frame from camera (no copy - we process immediately before next get)
                frame = self._camera.get_frame(copy=False)
                if frame is None:
                    time.sleep(FRAME_POLL_INTERVAL)
                    continue

                # Validate frame
                if frame.size == 0 or len(frame.shape) < 2:
                    logger.warning("Invalid frame received, skipping")
                    continue

                # Process through pipeline (async detection + interpolation)
                frame_data = self._pipeline.process_frame(frame)

                # Update metrics
                self._fps = self._pipeline.fps
                self._inference_ms = self._pipeline.inference_ms

                # Debug: log FPS periodically
                if self._pipeline._frame_count % PERF_LOG_INTERVAL == 0:
                    loop_now = time.time()
                    loop_ms = (loop_now - loop_start) * 1000
                    logger.info(f"[PERF] Cam: {self._camera.actual_fps:.1f}, Pipe: {self._fps:.1f}, Inf: {self._inference_ms:.0f}ms, Loop: {loop_ms:.0f}ms, Frames: {self._pipeline._frame_count}")

                # Get tracking state
                locked_target = frame_data.locked_target
                state = frame_data.tracking_state

                # Handle target lost timeout
                if state == TrackingState.LOST:
                    if lost_target_time is None:
                        lost_target_time = time.time()
                    elif time.time() - lost_target_time > search_timeout:
                        logger.warning("Target lost timeout, executing safety action")
                        self._mavlink.execute_lost_target_action()
                        lost_target_time = None
                else:
                    lost_target_time = None

                # PID control
                if self._pid.enabled and locked_target:
                    error = self._tracker.get_tracking_error()

                    # Calculate target size for throttle control
                    target_size = None
                    if locked_target:
                        bbox = locked_target.bbox
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        frame_area = frame.shape[0] * frame.shape[1]
                        if frame_area > 0:
                            target_size = area / frame_area

                    output = self._pid.update(error, target_size)

                    if output.is_active:
                        self._mavlink.send_rate_commands(
                            output.yaw_rate,
                            output.pitch_rate,
                            output.throttle_rate,
                        )

                # Build telemetry for overlay (all metrics in top-right)
                vehicle = self._mavlink.vehicle_state
                h, w = frame.shape[:2]

                # Calculate tracking error (prevent division by zero)
                error_x, error_y = 0.0, 0.0
                if locked_target and w > 0 and h > 0:
                    cx, cy = locked_target.center
                    half_w = w / 2
                    half_h = h / 2
                    error_x = (cx - half_w) / half_w * 100
                    error_y = (cy - half_h) / half_h * 100

                telemetry = {
                    "state": state.value,
                    "targets": len(frame_data.tracked_objects),
                    "error_x": error_x,
                    "error_y": error_y,
                    "fps": self._fps,
                    "inference_ms": self._inference_ms,
                    "connected": self._mavlink.is_connected if self._mavlink else False,
                    "altitude": vehicle.altitude_rel if vehicle else 0,
                    "speed": vehicle.groundspeed if vehicle else 0,
                    "heading": vehicle.heading if vehicle else 0,
                    "battery": vehicle.battery_percent if vehicle else 0,
                }

                # Render overlay on frame (in-place)
                self._streamer.render_overlay(
                    frame,
                    frame_data.tracked_objects,
                    locked_target,
                    state,
                    telemetry,
                )

                # Push to UDP stream first (latency critical path)
                self._streamer.push_frame(
                    frame,
                    frame_data.tracked_objects,
                    locked_target,
                    state,
                    telemetry,
                )

                # Cache for web preview (less critical)
                with self._frame_lock:
                    self._preview_frame = frame

                # Reset error counter on successful iteration
                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Main loop error: {e}", exc_info=True)

                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.critical(f"Too many consecutive errors ({consecutive_errors}), stopping")
                    self._running = False
                    break

                time.sleep(0.1)  # Brief pause before retrying
                continue

            # Rate limiting to target FPS
            loop_time = time.time() - loop_start
            if loop_time < target_frame_time:
                time.sleep(target_frame_time - loop_time)

    def get_status(self) -> Dict[str, Any]:
        """Get current system status for API."""
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
            "inference_ms": self._inference_ms,
            "altitude": vehicle.altitude_rel if vehicle else 0,
            "speed": vehicle.groundspeed if vehicle else 0,
            "heading": vehicle.heading if vehicle else 0,
            "battery": vehicle.battery_percent if vehicle else 0,
            "armed": vehicle.armed if vehicle else False,
            "connected": self._mavlink.is_connected if self._mavlink else False,
        }

    def get_preview_frame(self) -> Optional[np.ndarray]:
        """Get cached preview frame with overlay for web UI."""
        with self._frame_lock:
            return self._preview_frame  # Already rendered in main loop

    def lock_target(self, target_id: int) -> bool:
        """Lock onto specific target."""
        if self._tracker:
            return self._tracker.lock_target(target_id)
        return False

    def auto_lock(self) -> bool:
        """Lock onto best available target."""
        if not self._tracker:
            return False

        targets = self._tracker.all_targets
        if not targets:
            return False

        # Pick highest confidence target
        best = max(targets.values(), key=lambda t: t.confidence)
        return self._tracker.lock_target(best.object_id)

    def unlock_target(self) -> None:
        """Unlock current target."""
        if self._tracker:
            self._tracker.unlock()

    def enable_control(self) -> bool:
        """Enable tracking control."""
        if self._mavlink and self._mavlink.enable_tracking():
            if self._pid:
                self._pid.enable()
            return True
        return False

    def disable_control(self) -> None:
        """Disable tracking control."""
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

    def _handle_tracking_command(self, cmd: TrackingCommand, param: float) -> bool:
        """
        Handle tracking commands from QGC via MAVLink.

        Commands can be sent from QGC using COMMAND_LONG with:
        - MAV_CMD_USER_1 (31010): Auto-lock best target
        - MAV_CMD_USER_2 (31011): Lock target ID (param1 = target_id)
        - MAV_CMD_USER_3 (31012): Unlock target
        - MAV_CMD_USER_4 (31013): Enable tracking control
        - MAV_CMD_USER_5 (31014): Disable tracking control
        """
        logger.info(f"Handling tracking command: {cmd.name}")

        if cmd == TrackingCommand.AUTO_LOCK:
            return self.auto_lock()

        elif cmd == TrackingCommand.LOCK_TARGET:
            target_id = int(param)
            return self.lock_target(target_id)

        elif cmd == TrackingCommand.UNLOCK:
            self.unlock_target()
            return True

        elif cmd == TrackingCommand.ENABLE_CONTROL:
            return self.enable_control()

        elif cmd == TrackingCommand.DISABLE_CONTROL:
            self.disable_control()
            return True

        return False

    def clear_emergency(self) -> None:
        """Clear emergency stop."""
        if self._mavlink:
            self._mavlink.clear_emergency_stop()

    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update configuration and restart affected components."""
        try:
            # Track what changed for component restart
            restart_streamer = "output" in new_config and "stream" in new_config.get("output", {})
            restart_mavlink = "mavlink" in new_config
            restart_detector = "detector" in new_config and "target_classes" in new_config.get("detector", {})

            def merge(base, update):
                for key, value in update.items():
                    if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                        merge(base[key], value)
                    else:
                        base[key] = value

            merge(self.config, new_config)

            # Restart streamer if stream config changed (host/port)
            if restart_streamer and self._streamer:
                logger.info("Restarting streamer with new config...")
                self._streamer.stop()
                stream_config = StreamerConfig.from_dict(self.config)
                self._streamer = UDPStreamer(stream_config)
                self._streamer.start()

            # Restart MAVLink if connection changed
            if restart_mavlink and self._mavlink:
                logger.info("Restarting MAVLink with new config...")
                self._mavlink.stop()
                mav_config = MAVLinkConfig.from_dict(self.config)
                safety_config = SafetyConfig.from_dict(self.config)
                self._mavlink = MAVLinkController(mav_config, safety_config)
                self._mavlink.set_tracking_command_callback(self._handle_tracking_command)
                self._mavlink.start()

            # Restart detector if target_classes changed
            if restart_detector and self._detector:
                logger.info("Restarting detector with new target classes...")
                self.restart_detector()

            # Update PID parameters in-place (no restart needed)
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

    def restart_detector(self) -> None:
        """Restart the detector with current config."""
        logger.info("Restarting detector...")
        if self._pipeline:
            self._pipeline.stop()

        if self._detector:
            self._detector.shutdown()

        det_config = DetectorConfig.from_dict(self.config)
        self._detector = ObjectDetector(det_config)
        self._detector.initialize()

        # Restart pipeline with new detector
        if self._tracker:
            pipeline_config = PipelineConfig.from_dict(self.config)
            self._pipeline = Pipeline(self._detector, self._tracker, pipeline_config)
            self._pipeline.start()

        logger.info("Detector restarted")

    def restart_camera(self) -> None:
        """Restart the camera with current config."""
        logger.info("Restarting camera...")
        if self._camera:
            self._camera.stop()

        cam_config = CameraConfig.from_dict(self.config)
        self._camera = CameraCapture(cam_config)
        self._camera.start()
        logger.info("Camera restarted")

    def restart_streamer(self) -> None:
        """Restart the UDP streamer with current config."""
        logger.info("Restarting streamer...")
        if self._streamer:
            self._streamer.stop()

        stream_config = StreamerConfig.from_dict(self.config)
        self._streamer = UDPStreamer(stream_config)
        self._streamer.start()
        logger.info("Streamer restarted")

    def restart_mavlink(self) -> None:
        """Restart the MAVLink controller with current config."""
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


def main():
    """Application entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Terminal Guidance - Drone Companion Computer")
    parser.add_argument("-c", "--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("--no-web", action="store_true", help="Disable web UI")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Create application
    app = TerminalGuidance(args.config)

    # Handle shutdown signals
    def shutdown(signum, frame):
        logger.info("Shutdown signal received")
        app.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Initialize
    if not app.initialize():
        logger.error("Failed to initialize")
        sys.exit(1)

    # Start
    if not app.start():
        logger.error("Failed to start")
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
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    app.stop()


if __name__ == "__main__":
    main()
