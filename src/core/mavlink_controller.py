"""
MAVLink Controller for ArduPilot Communication.

Provides bidirectional communication with ArduPilot flight controllers
for telemetry, control commands, and safety management.
"""

import logging
import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two GPS coordinates in meters."""
    R = 6371000  # Earth radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------

class SafetyAction(Enum):
    """Actions to take when target is lost."""
    HOVER = "hover"
    LOITER = "loiter"
    RTL = "rtl"
    CONTINUE_LAST = "continue_last"
    LAND = "land"


class VehicleType(Enum):
    """Vehicle type for control mode selection."""
    COPTER = "copter"
    PLANE = "plane"
    AUTO = "auto"


class TrackingCommand(Enum):
    """
    Custom MAVLink commands for tracking control.

    Uses MAV_CMD_USER_1 through MAV_CMD_USER_5 (31010-31014).
    Send from QGC via COMMAND_LONG message.
    """
    AUTO_LOCK = 31010       # Auto-lock best target
    LOCK_TARGET = 31011     # Lock specific target (param1 = target_id)
    UNLOCK = 31012          # Unlock current target
    ENABLE_CONTROL = 31013  # Enable tracking control
    DISABLE_CONTROL = 31014 # Disable tracking control


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class MAVLinkConfig:
    """MAVLink connection and control configuration."""
    connection: str = "udp:192.168.1.1:14550"
    source_system: int = 255
    source_component: int = 190
    heartbeat_rate: float = 1.0
    command_timeout: float = 5.0
    enable_control: bool = False
    vehicle_type: VehicleType = VehicleType.PLANE
    cruise_throttle: int = 50
    tracking_throttle: int = 70
    max_turn_rate: float = 45.0
    max_climb_rate: float = 5.0

    @classmethod
    def from_dict(cls, config: dict) -> "MAVLinkConfig":
        mav = config.get("mavlink", {})
        intercept = mav.get("intercept", {})

        try:
            vehicle_type = VehicleType(mav.get("vehicle_type", "plane"))
        except ValueError:
            vehicle_type = VehicleType.PLANE

        return cls(
            connection=mav.get("connection", "udp:192.168.1.1:14550"),
            source_system=mav.get("source_system", 255),
            source_component=mav.get("source_component", 190),
            heartbeat_rate=mav.get("heartbeat_rate", 1.0),
            command_timeout=mav.get("command_timeout", 5.0),
            enable_control=mav.get("enable_control", False),
            vehicle_type=vehicle_type,
            cruise_throttle=intercept.get("cruise_throttle", 50),
            tracking_throttle=intercept.get("tracking_throttle", 70),
            max_turn_rate=intercept.get("max_turn_rate", 45.0),
            max_climb_rate=intercept.get("max_climb_rate", 5.0),
        )


@dataclass
class SafetyConfig:
    """Safety limits and geofence configuration."""
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
        safety = config.get("safety", {})
        geofence = safety.get("geofence", {})

        try:
            action = SafetyAction(safety.get("target_lost_action", "loiter"))
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
    """Current vehicle telemetry state."""
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
    vehicle_type: str = "unknown"
    throttle_percent: float = 0.0


# -----------------------------------------------------------------------------
# Controller
# -----------------------------------------------------------------------------

class MAVLinkController:
    """
    MAVLink interface for ArduPilot flight controller.

    Handles:
    - Connection management with heartbeat monitoring
    - Telemetry reception and state tracking
    - Control commands for tracking (heading, altitude, throttle)
    - Safety checks (geofence, battery, altitude limits)
    - Custom tracking commands from QGC
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

        # Connection health
        self._last_heartbeat_time: float = 0.0
        self._heartbeat_timeout: float = 5.0

        # Callbacks
        self._on_state_update: Optional[Callable[[VehicleState], None]] = None
        self._on_safety_triggered: Optional[Callable[[str], None]] = None
        self._on_tracking_command: Optional[Callable[[TrackingCommand, float], bool]] = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        with self._state_lock:
            return self._connected

    @property
    def vehicle_state(self) -> VehicleState:
        """Get thread-safe copy of vehicle state."""
        with self._state_lock:
            return VehicleState(**self._vehicle_state.__dict__)

    @property
    def is_armed(self) -> bool:
        with self._state_lock:
            return self._vehicle_state.armed

    @property
    def tracking_enabled(self) -> bool:
        with self._state_lock:
            return (self._tracking_enabled and
                    not self._emergency_stop and
                    self._connected)

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def set_state_callback(self, callback: Callable[[VehicleState], None]) -> None:
        self._on_state_update = callback

    def set_tracking_command_callback(
        self, callback: Callable[[TrackingCommand, float], bool]
    ) -> None:
        """Set callback for tracking commands from QGC."""
        self._on_tracking_command = callback

    def set_safety_callback(self, callback: Callable[[str], None]) -> None:
        self._on_safety_triggered = callback

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

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

            logger.error("No heartbeat received")
            return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def start(self) -> bool:
        """Start communication threads."""
        if not self._connected and not self.connect():
            return False

        self._running = True

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

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

    # -------------------------------------------------------------------------
    # Tracking Control
    # -------------------------------------------------------------------------

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
            if (self.safety_config.require_arm_confirmation and
                    not self._vehicle_state.armed):
                logger.warning("Cannot enable tracking: vehicle not armed")
                return False

            self._tracking_enabled = True

        self._set_mode("GUIDED")
        self.set_throttle_boost(True)
        logger.info("Tracking enabled - GUIDED mode, throttle boosted")
        return True

    def disable_tracking(self) -> None:
        """Disable tracking control."""
        with self._state_lock:
            was_enabled = self._tracking_enabled
            self._tracking_enabled = False

        if was_enabled:
            self.set_throttle_boost(False)
            self._set_mode("LOITER")
        logger.info("Tracking disabled")

    def emergency_stop(self) -> None:
        """Trigger emergency stop."""
        with self._state_lock:
            self._emergency_stop = True
            self._tracking_enabled = False
        logger.warning("EMERGENCY STOP ACTIVATED")

        if self._on_safety_triggered:
            self._on_safety_triggered("emergency_stop")

        self._execute_safety_action(SafetyAction.LOITER)

    def clear_emergency_stop(self) -> None:
        """Clear emergency stop state."""
        with self._state_lock:
            self._emergency_stop = False
        logger.info("Emergency stop cleared")

    # -------------------------------------------------------------------------
    # Control Commands
    # -------------------------------------------------------------------------

    def send_rate_commands(self, yaw_rate: float, pitch_rate: float,
                           throttle_rate: float) -> bool:
        """
        Send control commands for fixed-wing intercept.

        Args:
            yaw_rate: Turn rate (deg/sec) to center target horizontally
            pitch_rate: Climb/dive rate to center target vertically
            throttle_rate: Ignored (throttle boosted when tracking enabled)

        Returns:
            True if command sent successfully
        """
        if not self.tracking_enabled:
            return False
        if not self._check_safety():
            return False

        # Clamp rates
        max_turn = self.mav_config.max_turn_rate
        yaw_rate = max(-max_turn, min(max_turn, yaw_rate))
        pitch_rate = max(-20, min(20, pitch_rate))

        try:
            from pymavlink import mavutil

            # Calculate new heading
            current_heading = self._vehicle_state.heading
            heading_delta = yaw_rate * 0.05  # 50ms update rate
            new_heading = (current_heading + heading_delta) % 360

            # Calculate new altitude
            current_alt = self._vehicle_state.altitude_rel
            alt_delta = -pitch_rate * 0.1  # Scale to meters
            new_alt = current_alt + alt_delta

            # Clamp to safety limits
            new_alt = max(self.safety_config.min_altitude_m,
                         min(self.safety_config.max_altitude_m, new_alt))

            # Send GUIDED mode target
            self._connection.mav.set_position_target_global_int_send(
                0,  # time_boot_ms
                self._connection.target_system,
                self._connection.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                0b0000111111111000,  # Use altitude and yaw only
                0, 0,  # lat/lon (ignored)
                new_alt,
                0, 0, 0,  # velocity (ignored)
                0, 0, 0,  # acceleration (ignored)
                math.radians(new_heading),
                0,  # yaw_rate (ignored)
            )

            return True

        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False

    def set_throttle(self, throttle_percent: int) -> bool:
        """Set throttle via DO_CHANGE_SPEED command."""
        if not self._connected or not self._connection:
            return False

        try:
            from pymavlink import mavutil

            throttle_percent = max(0, min(100, throttle_percent))

            self._connection.mav.command_long_send(
                self._connection.target_system,
                self._connection.target_component,
                mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
                0,  # confirmation
                0,  # speed type
                -1,  # speed (no change)
                throttle_percent,
                0, 0, 0, 0,
            )

            logger.info(f"Throttle set to {throttle_percent}%")
            return True

        except Exception as e:
            logger.error(f"Failed to set throttle: {e}")
            return False

    def set_throttle_boost(self, enabled: bool) -> bool:
        """Enable/disable throttle boost for intercept."""
        throttle = (self.mav_config.tracking_throttle if enabled
                   else self.mav_config.cruise_throttle)
        return self.set_throttle(throttle)

    def execute_lost_target_action(self) -> None:
        """Execute configured action when target is lost."""
        action = self.safety_config.target_lost_action
        logger.info(f"Target lost, executing: {action.value}")
        self._execute_safety_action(action)

    # -------------------------------------------------------------------------
    # Safety
    # -------------------------------------------------------------------------

    def _check_safety(self) -> bool:
        """Check safety constraints. Returns False if violated."""
        state = self.vehicle_state

        # Battery check
        if state.battery_percent < self.safety_config.min_battery_percent:
            logger.warning(f"Low battery: {state.battery_percent}%")
            self.disable_tracking()
            if self._on_safety_triggered:
                self._on_safety_triggered("low_battery")
            self._execute_safety_action(SafetyAction.RTL)
            return False

        # Geofence checks
        if self.safety_config.geofence_enabled:
            if state.home_distance > self.safety_config.max_distance_m:
                logger.warning(f"Geofence breach: {state.home_distance}m")
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

    def _execute_safety_action(self, action: SafetyAction) -> None:
        """Execute a safety action."""
        if not self._connected or not self._connection:
            return

        try:
            from pymavlink import mavutil

            if action == SafetyAction.HOVER:
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
            # CONTINUE_LAST: do nothing

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

    # -------------------------------------------------------------------------
    # Communication Threads
    # -------------------------------------------------------------------------

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

                # Check connection timeout
                if self._last_heartbeat_time > 0:
                    elapsed = time.time() - self._last_heartbeat_time
                    if elapsed > self._heartbeat_timeout:
                        with self._state_lock:
                            if self._connected:
                                logger.warning(f"Connection lost: no heartbeat for {elapsed:.1f}s")
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
                self._last_heartbeat_time = time.time()
                if not self._connected:
                    logger.info("Connection restored")
                    self._connected = True

            elif msg_type == "GLOBAL_POSITION_INT":
                self._vehicle_state.latitude = msg.lat / 1e7
                self._vehicle_state.longitude = msg.lon / 1e7
                self._vehicle_state.altitude_msl = msg.alt / 1000.0
                self._vehicle_state.altitude_rel = msg.relative_alt / 1000.0
                self._vehicle_state.heading = msg.hdg / 100.0

                # Update home distance
                if (self._vehicle_state.home_latitude != 0 and
                        self._vehicle_state.latitude != 0):
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
                self._vehicle_state.home_latitude = msg.latitude / 1e7
                self._vehicle_state.home_longitude = msg.longitude / 1e7
                self._vehicle_state.home_altitude = msg.altitude / 1000.0

                if (self._vehicle_state.latitude != 0 and
                        self._vehicle_state.home_latitude != 0):
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
        try:
            tracking_cmd = TrackingCommand(msg.command)
        except ValueError:
            return  # Not a tracking command

        logger.info(f"Received tracking command: {tracking_cmd.name} (param1={msg.param1})")

        result = False
        if self._on_tracking_command:
            result = self._on_tracking_command(tracking_cmd, msg.param1)

        self._send_command_ack(msg.command, result)

    def _send_command_ack(self, command: int, success: bool) -> None:
        """Send COMMAND_ACK response."""
        if not self._connection:
            return

        try:
            from pymavlink import mavutil

            result = (mavutil.mavlink.MAV_RESULT_ACCEPTED if success
                     else mavutil.mavlink.MAV_RESULT_FAILED)
            self._connection.mav.command_ack_send(command, result)

        except Exception as e:
            logger.error(f"Failed to send command ACK: {e}")

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
