"""MAVLink controller for ArduPilot communication."""

import logging
import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two GPS coordinates using Haversine formula.

    Args:
        lat1, lon1: First coordinate (degrees)
        lat2, lon2: Second coordinate (degrees)

    Returns:
        Distance in meters
    """
    R = 6371000  # Earth radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


class SafetyAction(Enum):
    """Actions to take when target is lost."""
    HOVER = "hover"
    LOITER = "loiter"
    RTL = "rtl"
    CONTINUE_LAST = "continue_last"
    LAND = "land"


class TrackingCommand(Enum):
    """Custom MAVLink commands for tracking control.

    These use MAV_CMD_USER_1 through MAV_CMD_USER_5 (31010-31014).
    Send from QGC via COMMAND_LONG message.
    """
    AUTO_LOCK = 31010       # MAV_CMD_USER_1: Auto-lock best target
    LOCK_TARGET = 31011     # MAV_CMD_USER_2: Lock specific target (param1 = target_id)
    UNLOCK = 31012          # MAV_CMD_USER_3: Unlock current target
    ENABLE_CONTROL = 31013  # MAV_CMD_USER_4: Enable tracking control
    DISABLE_CONTROL = 31014 # MAV_CMD_USER_5: Disable tracking control


@dataclass
class MAVLinkConfig:
    """MAVLink configuration parameters."""
    connection: str = "udp:192.168.1.1:14550"
    source_system: int = 255
    source_component: int = 190
    heartbeat_rate: float = 1.0
    command_timeout: float = 5.0
    enable_control: bool = False

    @classmethod
    def from_dict(cls, config: dict) -> "MAVLinkConfig":
        """Create config from dictionary."""
        mav = config.get("mavlink", {})
        return cls(
            connection=mav.get("connection", "udp:192.168.1.1:14550"),
            source_system=mav.get("source_system", 255),
            source_component=mav.get("source_component", 190),
            heartbeat_rate=mav.get("heartbeat_rate", 1.0),
            command_timeout=mav.get("command_timeout", 5.0),
            enable_control=mav.get("enable_control", False),
        )


@dataclass
class SafetyConfig:
    """Safety configuration parameters."""
    target_lost_action: SafetyAction = SafetyAction.LOITER
    search_timeout: float = 10.0
    max_distance_m: float = 500.0
    max_altitude_m: float = 120.0
    min_altitude_m: float = 10.0
    min_battery_percent: float = 20.0
    max_tracking_speed: float = 10.0
    require_arm_confirmation: bool = True
    geofence_enabled: bool = True
    emergency_stop_enabled: bool = True

    @classmethod
    def from_dict(cls, config: dict) -> "SafetyConfig":
        """Create config from dictionary."""
        safety = config.get("safety", {})
        geofence = safety.get("geofence", {})

        action_str = safety.get("target_lost_action", "loiter")
        try:
            action = SafetyAction(action_str)
        except ValueError:
            action = SafetyAction.LOITER

        return cls(
            target_lost_action=action,
            search_timeout=safety.get("search_timeout", 10.0),
            max_distance_m=geofence.get("max_distance_m", 500.0),
            max_altitude_m=geofence.get("max_altitude_m", 120.0),
            min_altitude_m=geofence.get("min_altitude_m", 10.0),
            min_battery_percent=safety.get("min_battery_percent", 20.0),
            max_tracking_speed=safety.get("max_tracking_speed", 10.0),
            require_arm_confirmation=safety.get("require_arm_confirmation", True),
            geofence_enabled=geofence.get("enabled", True),
            emergency_stop_enabled=safety.get("emergency_stop_enabled", True),
        )


@dataclass
class VehicleState:
    """Current vehicle state from telemetry."""
    armed: bool = False
    mode: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    altitude_msl: float = 0.0
    altitude_rel: float = 0.0
    heading: float = 0.0
    groundspeed: float = 0.0
    airspeed: float = 0.0
    battery_percent: float = 100.0
    battery_voltage: float = 0.0
    home_distance: float = 0.0
    home_latitude: float = 0.0
    home_longitude: float = 0.0
    home_altitude: float = 0.0
    gps_fix: int = 0
    satellites: int = 0
    last_update: float = 0.0


class MAVLinkController:
    """
    MAVLink interface for ArduPilot flight controller.

    Handles:
    - Connection management
    - Heartbeat sending
    - Telemetry reception
    - Control command transmission
    - Safety monitoring
    """

    def __init__(self, mav_config: MAVLinkConfig, safety_config: SafetyConfig):
        self.mav_config = mav_config
        self.safety_config = safety_config

        self._connection = None
        self._connected = False
        self._running = False

        self._vehicle_state = VehicleState()
        self._state_lock = threading.Lock()

        self._heartbeat_thread: Optional[threading.Thread] = None
        self._receive_thread: Optional[threading.Thread] = None

        self._emergency_stop = False
        self._tracking_enabled = False

        # Connection health tracking
        self._last_heartbeat_time: float = 0.0
        self._heartbeat_timeout: float = 5.0  # Consider disconnected after 5s without heartbeat

        # Callbacks
        self._on_state_update: Optional[Callable[[VehicleState], None]] = None
        self._on_safety_triggered: Optional[Callable[[str], None]] = None
        self._on_tracking_command: Optional[Callable[[TrackingCommand, float], bool]] = None

    @property
    def is_connected(self) -> bool:
        """Check if connected to flight controller (thread-safe)."""
        with self._state_lock:
            return self._connected

    @property
    def vehicle_state(self) -> VehicleState:
        """Get current vehicle state (thread-safe copy)."""
        with self._state_lock:
            return VehicleState(**self._vehicle_state.__dict__)

    @property
    def is_armed(self) -> bool:
        """Check if vehicle is armed (thread-safe)."""
        with self._state_lock:
            return self._vehicle_state.armed

    @property
    def tracking_enabled(self) -> bool:
        """Check if tracking control is enabled (thread-safe)."""
        with self._state_lock:
            return self._tracking_enabled and not self._emergency_stop and self._connected

    def set_state_callback(self, callback: Callable[[VehicleState], None]) -> None:
        """Set callback for state updates."""
        self._on_state_update = callback

    def set_tracking_command_callback(
        self, callback: Callable[[TrackingCommand, float], bool]
    ) -> None:
        """
        Set callback for tracking commands from QGC.

        Callback receives (command, param) and returns True if handled.
        """
        self._on_tracking_command = callback

    def set_safety_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for safety events."""
        self._on_safety_triggered = callback

    def connect(self) -> bool:
        """Establish MAVLink connection."""
        try:
            from pymavlink import mavutil

            logger.info(f"Connecting to: {self.mav_config.connection}")

            self._connection = mavutil.mavlink_connection(
                self.mav_config.connection,
                source_system=self.mav_config.source_system,
                source_component=self.mav_config.source_component,
            )

            # Wait for heartbeat
            logger.info("Waiting for heartbeat...")
            msg = self._connection.wait_heartbeat(timeout=10)

            if msg:
                with self._state_lock:
                    self._connected = True
                logger.info(
                    f"Connected to system {self._connection.target_system}, "
                    f"component {self._connection.target_component}"
                )
                return True
            else:
                logger.error("No heartbeat received")
                return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def start(self) -> bool:
        """Start communication threads."""
        if not self._connected:
            if not self.connect():
                return False

        self._running = True

        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

        # Start receive thread
        self._receive_thread = threading.Thread(
            target=self._receive_loop, daemon=True
        )
        self._receive_thread.start()

        logger.info("MAVLink controller started")
        return True

    def stop(self) -> None:
        """Stop communication and close connection."""
        self._running = False
        self._tracking_enabled = False

        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2.0)
        if self._receive_thread:
            self._receive_thread.join(timeout=2.0)

        if self._connection:
            self._connection.close()

        with self._state_lock:
            self._connected = False
        logger.info("MAVLink controller stopped")

    def enable_tracking(self) -> bool:
        """Enable tracking control commands."""
        with self._state_lock:
            if not self._connected:
                logger.warning("Cannot enable tracking: not connected")
                return False

            if not self.mav_config.enable_control:
                logger.warning("Control disabled in configuration")
                return False

            if self._emergency_stop:
                logger.warning("Cannot enable tracking: emergency stop active")
                return False

            if self.safety_config.require_arm_confirmation and not self._vehicle_state.armed:
                logger.warning("Cannot enable tracking: vehicle not armed")
                return False

            self._tracking_enabled = True
            logger.info("Tracking control enabled")
            return True

    def disable_tracking(self) -> None:
        """Disable tracking control."""
        with self._state_lock:
            self._tracking_enabled = False
        logger.info("Tracking control disabled")

    def emergency_stop(self) -> None:
        """Trigger emergency stop."""
        with self._state_lock:
            self._emergency_stop = True
            self._tracking_enabled = False
        logger.warning("EMERGENCY STOP ACTIVATED")

        if self._on_safety_triggered:
            self._on_safety_triggered("emergency_stop")

        # Execute safety action
        self._execute_safety_action(SafetyAction.LOITER)

    def clear_emergency_stop(self) -> None:
        """Clear emergency stop state."""
        with self._state_lock:
            self._emergency_stop = False
        logger.info("Emergency stop cleared")

    def send_rate_commands(
        self,
        yaw_rate: float,
        pitch_rate: float,
        throttle_rate: float,
    ) -> bool:
        """
        Send rate-based control commands.

        Args:
            yaw_rate: Yaw rate in deg/sec (positive = clockwise)
            pitch_rate: Pitch rate in deg/sec (positive = nose up)
            throttle_rate: Vertical rate in m/sec (positive = up)

        Returns:
            True if command sent successfully
        """
        if not self.tracking_enabled:
            return False

        # Check safety limits
        if not self._check_safety():
            return False

        # Clamp rates
        max_speed = self.safety_config.max_tracking_speed
        yaw_rate = max(-90, min(90, yaw_rate))
        pitch_rate = max(-45, min(45, pitch_rate))
        throttle_rate = max(-max_speed, min(max_speed, throttle_rate))

        try:
            # Send SET_ATTITUDE_TARGET or velocity commands
            # Using velocity control for ArduPilot
            from pymavlink import mavutil

            # Convert pitch rate to forward velocity (simplified)
            # Positive pitch = nose up = forward motion in tracking context
            vx = pitch_rate * 0.5  # Scale factor
            vy = 0  # No lateral movement for now
            vz = -throttle_rate  # NED frame: negative Z is up

            # Send velocity command in body frame
            self._connection.mav.set_position_target_local_ned_send(
                0,  # time_boot_ms
                self._connection.target_system,
                self._connection.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                0b0000011111000111,  # Velocity only
                0, 0, 0,  # Position (ignored)
                vx, vy, vz,  # Velocity
                0, 0, 0,  # Acceleration (ignored)
                0,  # Yaw (ignored)
                yaw_rate * 0.0174533,  # Yaw rate in rad/s
            )

            return True

        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False

    def execute_lost_target_action(self) -> None:
        """Execute configured action when target is lost."""
        action = self.safety_config.target_lost_action
        logger.info(f"Target lost, executing: {action.value}")
        self._execute_safety_action(action)

    def _execute_safety_action(self, action: SafetyAction) -> None:
        """Execute a safety action."""
        if not self._connected or not self._connection:
            return

        try:
            from pymavlink import mavutil

            if action == SafetyAction.HOVER:
                # Send zero velocity
                self._connection.mav.set_position_target_local_ned_send(
                    0, self._connection.target_system,
                    self._connection.target_component,
                    mavutil.mavlink.MAV_FRAME_BODY_NED,
                    0b0000011111000111,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                )

            elif action == SafetyAction.LOITER:
                self._set_mode("LOITER")

            elif action == SafetyAction.RTL:
                self._set_mode("RTL")

            elif action == SafetyAction.LAND:
                self._set_mode("LAND")

            elif action == SafetyAction.CONTINUE_LAST:
                pass  # Continue current trajectory

        except Exception as e:
            logger.error(f"Failed to execute safety action: {e}")

    def _set_mode(self, mode_name: str) -> bool:
        """Set flight mode."""
        if not self._connection:
            return False

        try:
            from pymavlink import mavutil

            mode_id = self._connection.mode_mapping().get(mode_name)
            if mode_id is None:
                logger.error(f"Unknown mode: {mode_name}")
                return False

            self._connection.mav.set_mode_send(
                self._connection.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id,
            )
            logger.info(f"Mode change requested: {mode_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to set mode: {e}")
            return False

    def _check_safety(self) -> bool:
        """Check safety constraints."""
        state = self.vehicle_state

        # Battery check
        if state.battery_percent < self.safety_config.min_battery_percent:
            logger.warning(f"Low battery: {state.battery_percent}%")
            self.disable_tracking()  # Disable tracking on safety trigger
            if self._on_safety_triggered:
                self._on_safety_triggered("low_battery")
            self._execute_safety_action(SafetyAction.RTL)
            return False

        # Geofence check
        if self.safety_config.geofence_enabled:
            if state.home_distance > self.safety_config.max_distance_m:
                logger.warning(f"Geofence breach: distance {state.home_distance}m")
                self.disable_tracking()
                if self._on_safety_triggered:
                    self._on_safety_triggered("geofence_distance")
                self._execute_safety_action(SafetyAction.RTL)
                return False

            if state.altitude_rel > self.safety_config.max_altitude_m:
                logger.warning(f"Max altitude breach: {state.altitude_rel}m")
                self.disable_tracking()
                if self._on_safety_triggered:
                    self._on_safety_triggered("geofence_altitude_max")
                self._execute_safety_action(SafetyAction.LOITER)
                return False

            if state.altitude_rel < self.safety_config.min_altitude_m:
                logger.warning(f"Min altitude breach: {state.altitude_rel}m")
                self.disable_tracking()
                if self._on_safety_triggered:
                    self._on_safety_triggered("geofence_altitude_min")
                self._execute_safety_action(SafetyAction.LOITER)
                return False

        return True

    def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        from pymavlink import mavutil

        interval = 1.0 / self.mav_config.heartbeat_rate

        while self._running:
            try:
                if self._connection:
                    self._connection.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                        0, 0, 0,
                    )
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

            time.sleep(interval)

    def _receive_loop(self) -> None:
        """Receive and process MAVLink messages."""
        while self._running:
            try:
                if not self._connection:
                    time.sleep(0.1)
                    continue

                msg = self._connection.recv_match(blocking=True, timeout=1.0)
                if msg:
                    self._process_message(msg)

                # Check for connection timeout (no heartbeat received)
                if self._last_heartbeat_time > 0:
                    time_since_heartbeat = time.time() - self._last_heartbeat_time
                    if time_since_heartbeat > self._heartbeat_timeout:
                        with self._state_lock:
                            if self._connected:
                                logger.warning(f"Connection lost: no heartbeat for {time_since_heartbeat:.1f}s")
                                self._connected = False
                                self._tracking_enabled = False
                        if self._on_safety_triggered:
                            self._on_safety_triggered("connection_lost")

            except Exception as e:
                logger.error(f"Receive error: {e}")
                time.sleep(0.1)

    def _process_message(self, msg) -> None:
        """Process received MAVLink message."""
        msg_type = msg.get_type()

        with self._state_lock:
            if msg_type == "HEARTBEAT":
                from pymavlink import mavutil
                self._vehicle_state.armed = (
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                ) != 0
                # Update heartbeat time for connection monitoring
                self._last_heartbeat_time = time.time()
                # Restore connection if it was lost
                if not self._connected:
                    logger.info("Connection restored")
                    self._connected = True

            elif msg_type == "GLOBAL_POSITION_INT":
                self._vehicle_state.latitude = msg.lat / 1e7
                self._vehicle_state.longitude = msg.lon / 1e7
                self._vehicle_state.altitude_msl = msg.alt / 1000.0
                self._vehicle_state.altitude_rel = msg.relative_alt / 1000.0
                self._vehicle_state.heading = msg.hdg / 100.0

                # Update distance from home on every position update (not just HOME_POSITION)
                if self._vehicle_state.home_latitude != 0 and self._vehicle_state.latitude != 0:
                    self._vehicle_state.home_distance = haversine_distance(
                        self._vehicle_state.latitude,
                        self._vehicle_state.longitude,
                        self._vehicle_state.home_latitude,
                        self._vehicle_state.home_longitude,
                    )

            elif msg_type == "VFR_HUD":
                self._vehicle_state.groundspeed = msg.groundspeed
                self._vehicle_state.airspeed = msg.airspeed
                self._vehicle_state.heading = msg.heading

            elif msg_type == "SYS_STATUS":
                if msg.battery_remaining >= 0:
                    self._vehicle_state.battery_percent = msg.battery_remaining
                self._vehicle_state.battery_voltage = msg.voltage_battery / 1000.0

            elif msg_type == "GPS_RAW_INT":
                self._vehicle_state.gps_fix = msg.fix_type
                self._vehicle_state.satellites = msg.satellites_visible

            elif msg_type == "HOME_POSITION":
                # Store home position
                self._vehicle_state.home_latitude = msg.latitude / 1e7
                self._vehicle_state.home_longitude = msg.longitude / 1e7
                self._vehicle_state.home_altitude = msg.altitude / 1000.0

                # Calculate distance from home
                if self._vehicle_state.latitude != 0 and self._vehicle_state.home_latitude != 0:
                    self._vehicle_state.home_distance = haversine_distance(
                        self._vehicle_state.latitude,
                        self._vehicle_state.longitude,
                        self._vehicle_state.home_latitude,
                        self._vehicle_state.home_longitude,
                    )

            self._vehicle_state.last_update = time.time()

        # Handle commands outside state lock
        if msg_type == "COMMAND_LONG":
            self._handle_command(msg)

        if self._on_state_update:
            self._on_state_update(self.vehicle_state)

    def _handle_command(self, msg) -> None:
        """Handle COMMAND_LONG messages for tracking control."""
        cmd_id = msg.command

        # Check if it's one of our tracking commands
        try:
            tracking_cmd = TrackingCommand(cmd_id)
        except ValueError:
            return  # Not a tracking command

        logger.info(f"Received tracking command: {tracking_cmd.name} (param1={msg.param1})")

        # Send to callback if registered
        result = False
        if self._on_tracking_command:
            result = self._on_tracking_command(tracking_cmd, msg.param1)

        # Send ACK back to QGC
        self._send_command_ack(cmd_id, result)

    def _send_command_ack(self, command: int, success: bool) -> None:
        """Send COMMAND_ACK response."""
        if not self._connection:
            return

        try:
            from pymavlink import mavutil

            result = (
                mavutil.mavlink.MAV_RESULT_ACCEPTED
                if success
                else mavutil.mavlink.MAV_RESULT_FAILED
            )

            self._connection.mav.command_ack_send(command, result)
        except Exception as e:
            logger.error(f"Failed to send command ACK: {e}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
