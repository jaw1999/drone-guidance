"""PID controller for target tracking."""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PIDGains:
    """PID controller gains."""
    kp: float = 0.5
    ki: float = 0.01
    kd: float = 0.1
    max_output: float = 30.0
    derivative_filter: float = 0.1  # Low-pass filter coefficient (0-1, lower = more filtering)
    slew_rate: float = 0.0  # Max output change per second (0 = disabled)

    @classmethod
    def from_dict(cls, config: dict) -> "PIDGains":
        """Create gains from dictionary."""
        return cls(
            kp=config.get("kp", 0.5),
            ki=config.get("ki", 0.01),
            kd=config.get("kd", 0.1),
            max_output=config.get("max_rate", 30.0),
            derivative_filter=config.get("derivative_filter", 0.1),
            slew_rate=config.get("slew_rate", 0.0),
        )


@dataclass
class PIDConfig:
    """PID controller configuration."""
    yaw: PIDGains
    pitch: PIDGains
    throttle: PIDGains
    dead_zone_percent: float = 5.0
    update_rate: float = 20.0

    @classmethod
    def from_dict(cls, config: dict) -> "PIDConfig":
        """Create config from dictionary."""
        pid = config.get("pid", {})
        return cls(
            yaw=PIDGains.from_dict(pid.get("yaw", {})),
            pitch=PIDGains.from_dict(pid.get("pitch", {})),
            throttle=PIDGains.from_dict(pid.get("throttle", {})),
            dead_zone_percent=pid.get("dead_zone_percent", 5.0),
            update_rate=pid.get("update_rate", 20.0),
        )


class PIDAxis:
    """Single-axis PID with anti-windup and derivative filtering."""

    MIN_KI_FOR_LIMIT = 0.001  # Minimum ki to calculate meaningful integral limit

    def __init__(self, gains: PIDGains):
        self.gains = gains
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_measurement = 0.0  # For derivative-on-measurement
        self._prev_time: Optional[float] = None
        self._first_update = True
        self._filtered_derivative = 0.0  # Low-pass filtered derivative
        self._prev_output = 0.0  # For slew rate limiting

        # Anti-windup limit (cap at reasonable value for small ki)
        self._integral_limit = self._calculate_integral_limit(gains.ki, gains.max_output)

    def _calculate_integral_limit(self, ki: float, max_output: float) -> float:
        """Calculate integral limit, capped for small ki values."""
        if ki >= self.MIN_KI_FOR_LIMIT:
            return min(max_output / ki, max_output * 100)
        return max_output * 100

    def update(self, error: float, dt: Optional[float] = None, measurement: float = 0.0) -> float:
        """
        Calculate PID output for given error.

        Args:
            error: Current error value (normalized -1 to 1)
            dt: Time delta in seconds (calculated automatically if None)
            measurement: Process variable for derivative-on-measurement (optional)

        Returns:
            Control output (clamped to max_output)
        """
        current_time = time.time()

        if dt is None:
            if self._prev_time is None:
                dt = 0.05  # Default 20Hz
            else:
                dt = current_time - self._prev_time
                # Guard against negative dt from clock issues
                if dt <= 0:
                    dt = 0.05
        self._prev_time = current_time

        # Prevent division by zero
        dt = max(dt, 0.001)

        # Proportional term
        p_term = self.gains.kp * error

        # Integral term with anti-windup (skip if error is zero to avoid windup in dead zone)
        if error != 0.0:
            self._integral += error * dt
            self._integral = max(-self._integral_limit,
                                min(self._integral_limit, self._integral))
        i_term = self.gains.ki * self._integral

        # Derivative term on measurement (avoids derivative kick on setpoint change)
        # Falls back to error derivative if no measurement provided
        if self._first_update:
            raw_derivative = 0.0
            self._first_update = False
        elif measurement != 0.0:
            # Derivative on measurement (negative because we want to oppose change)
            raw_derivative = -(measurement - self._prev_measurement) / dt
        else:
            raw_derivative = (error - self._prev_error) / dt

        # Low-pass filter on derivative to reduce noise sensitivity
        # filter_coeff: 0 = heavy filtering (slow response), 1 = no filtering (noisy)
        alpha = max(0.0, min(1.0, self.gains.derivative_filter))
        self._filtered_derivative = alpha * raw_derivative + (1 - alpha) * self._filtered_derivative
        d_term = self.gains.kd * self._filtered_derivative

        self._prev_error = error
        self._prev_measurement = measurement

        # Sum and clamp output
        output = p_term + i_term + d_term
        output = max(-self.gains.max_output, min(self.gains.max_output, output))

        # Slew rate limiting (max change per second)
        if self.gains.slew_rate > 0:
            max_change = self.gains.slew_rate * dt
            delta = output - self._prev_output
            if abs(delta) > max_change:
                output = self._prev_output + max_change * (1 if delta > 0 else -1)

        self._prev_output = output
        return output

    def reset(self) -> None:
        """Reset controller state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_measurement = 0.0
        self._prev_time = None
        self._first_update = True
        self._filtered_derivative = 0.0
        self._prev_output = 0.0


@dataclass
class ControlOutput:
    """Control output for flight controller."""
    yaw_rate: float = 0.0      # deg/sec, positive = turn right
    pitch_rate: float = 0.0    # deg/sec, positive = pitch up (climb)
    throttle_rate: float = 0.0  # m/sec, positive = increase throttle
    is_active: bool = False     # Whether control is active


class PIDController:
    """Multi-axis PID controller for target tracking."""

    def __init__(self, config: PIDConfig):
        self.config = config
        self._yaw = PIDAxis(config.yaw)
        self._pitch = PIDAxis(config.pitch)
        self._throttle = PIDAxis(config.throttle)
        self._enabled = False
        self._dead_zone = config.dead_zone_percent / 100.0

    @property
    def enabled(self) -> bool:
        """Check if controller is enabled."""
        return self._enabled

    @property
    def dead_zone(self) -> float:
        """Get dead zone as fraction."""
        return self._dead_zone

    @dead_zone.setter
    def dead_zone(self, value: float) -> None:
        """Set dead zone as fraction."""
        self._dead_zone = value

    def update_gains(self, axis: str, kp: float = None, ki: float = None,
                     kd: float = None, max_rate: float = None,
                     derivative_filter: float = None, slew_rate: float = None,
                     reset_integral: bool = True) -> None:
        """Update gains for a specific axis.

        Args:
            axis: The axis to update ('yaw', 'pitch', or 'throttle')
            kp: New proportional gain (optional)
            ki: New integral gain (optional)
            kd: New derivative gain (optional)
            max_rate: New maximum output rate (optional)
            derivative_filter: Low-pass filter coefficient for derivative (0-1, optional)
            slew_rate: Max output change per second (optional, 0 = disabled)
            reset_integral: Whether to reset integral term (default True)
        """
        pid_axis = {"yaw": self._yaw, "pitch": self._pitch, "throttle": self._throttle}.get(axis)
        if not pid_axis:
            logger.warning(f"Unknown PID axis: {axis}")
            return

        # Validate gains (prevent negative values that cause instability)
        if kp is not None:
            if kp < 0:
                logger.warning(f"Ignoring negative kp={kp} for {axis}")
            else:
                pid_axis.gains.kp = kp
        if ki is not None:
            if ki < 0:
                logger.warning(f"Ignoring negative ki={ki} for {axis}")
            else:
                pid_axis.gains.ki = ki
                pid_axis._integral_limit = pid_axis._calculate_integral_limit(ki, pid_axis.gains.max_output)
        if kd is not None:
            if kd < 0:
                logger.warning(f"Ignoring negative kd={kd} for {axis}")
            else:
                pid_axis.gains.kd = kd
        if max_rate is not None:
            if max_rate <= 0:
                logger.warning(f"Ignoring invalid max_rate={max_rate} for {axis}")
            else:
                pid_axis.gains.max_output = max_rate
                pid_axis._integral_limit = pid_axis._calculate_integral_limit(pid_axis.gains.ki, max_rate)
        if derivative_filter is not None:
            if derivative_filter < 0 or derivative_filter > 1:
                logger.warning(f"Ignoring invalid derivative_filter={derivative_filter} for {axis} (must be 0-1)")
            else:
                pid_axis.gains.derivative_filter = derivative_filter
        if slew_rate is not None:
            if slew_rate < 0:
                logger.warning(f"Ignoring negative slew_rate={slew_rate} for {axis}")
            else:
                pid_axis.gains.slew_rate = slew_rate

        # Reset integral to prevent windup issues from old accumulated error
        if reset_integral:
            pid_axis.reset()
            logger.info(f"Updated {axis} PID gains and reset integral: kp={pid_axis.gains.kp}, ki={pid_axis.gains.ki}, kd={pid_axis.gains.kd}")
        else:
            logger.info(f"Updated {axis} PID gains: kp={pid_axis.gains.kp}, ki={pid_axis.gains.ki}, kd={pid_axis.gains.kd}")

    def enable(self) -> None:
        """Enable control output."""
        self._enabled = True
        logger.info("PID controller enabled")

    def disable(self) -> None:
        """Disable control output and reset."""
        self._enabled = False
        self.reset()
        logger.info("PID controller disabled")

    def reset(self) -> None:
        """Reset all controller states."""
        self._yaw.reset()
        self._pitch.reset()
        self._throttle.reset()

    def update(
        self,
        error: Optional[Tuple[float, float]],
        target_size: Optional[float] = None
    ) -> ControlOutput:
        """
        Calculate control output from tracking error.

        Args:
            error: (x_error, y_error) normalized to [-1, 1]
                   None if no target
            target_size: Optional relative target size for throttle control
                        (larger = closer = slow down)

        Returns:
            ControlOutput with rate commands
        """
        if not self._enabled or error is None:
            return ControlOutput(is_active=False)

        x_error, y_error = error

        # Apply dead zone
        if abs(x_error) < self._dead_zone:
            x_error = 0.0
        if abs(y_error) < self._dead_zone:
            y_error = 0.0

        # Calculate control outputs
        # Positive X error (target right of center) -> positive yaw (turn right)
        yaw_rate = self._yaw.update(x_error)

        # Positive Y error (target below center) -> negative pitch (nose down)
        # Invert because in image coords, Y increases downward
        pitch_rate = -self._pitch.update(y_error)

        # Throttle based on target size (optional)
        throttle_rate = 0.0
        if target_size is not None:
            # Target size error: desired ~0.2 of frame, adjust throttle to maintain
            size_error = 0.2 - target_size
            throttle_rate = self._throttle.update(size_error)
        else:
            # Reset throttle PID when no target to prevent stale state
            self._throttle.reset()

        return ControlOutput(
            yaw_rate=yaw_rate,
            pitch_rate=pitch_rate,
            throttle_rate=throttle_rate,
            is_active=True,
        )

    def get_status(self) -> dict:
        """Get controller status for telemetry."""
        return {
            "enabled": self._enabled,
            "yaw_integral": self._yaw._integral,
            "pitch_integral": self._pitch._integral,
            "throttle_integral": self._throttle._integral,
        }
