"""
PID Controller for Target Tracking.

Provides multi-axis PID control with anti-windup, derivative filtering,
and slew rate limiting for smooth vehicle control.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PIDGains:
    """PID controller gains for a single axis."""
    kp: float = 0.5
    ki: float = 0.01
    kd: float = 0.1
    max_output: float = 30.0
    derivative_filter: float = 0.1  # Low-pass filter (0-1, lower = smoother)
    slew_rate: float = 0.0  # Max output change per second (0 = disabled)

    @classmethod
    def from_dict(cls, config: dict) -> "PIDGains":
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
    """Configuration for multi-axis PID controller."""
    yaw: PIDGains
    pitch: PIDGains
    throttle: PIDGains
    dead_zone_percent: float = 5.0
    update_rate: float = 20.0

    @classmethod
    def from_dict(cls, config: dict) -> "PIDConfig":
        pid = config.get("pid", {})
        return cls(
            yaw=PIDGains.from_dict(pid.get("yaw", {})),
            pitch=PIDGains.from_dict(pid.get("pitch", {})),
            throttle=PIDGains.from_dict(pid.get("throttle", {})),
            dead_zone_percent=pid.get("dead_zone_percent", 5.0),
            update_rate=pid.get("update_rate", 20.0),
        )


@dataclass
class ControlOutput:
    """Control output for flight controller."""
    yaw_rate: float = 0.0       # deg/sec, positive = turn right
    pitch_rate: float = 0.0     # deg/sec, positive = pitch up
    throttle_rate: float = 0.0  # m/sec, positive = increase throttle
    is_active: bool = False


class PIDAxis:
    """
    Single-axis PID controller with advanced features.

    Features:
    - Anti-windup via integral clamping
    - Low-pass filtering on derivative term
    - Slew rate limiting for smooth output
    """

    MIN_KI_FOR_LIMIT = 0.001

    def __init__(self, gains: PIDGains):
        self.gains = gains

        # State
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time: Optional[float] = None
        self._first_update = True
        self._filtered_derivative = 0.0
        self._prev_output = 0.0

        # Anti-windup limit
        self._integral_limit = self._calculate_integral_limit(
            gains.ki, gains.max_output
        )

    def _calculate_integral_limit(self, ki: float, max_output: float) -> float:
        """Calculate integral limit with cap for small ki values."""
        if ki >= self.MIN_KI_FOR_LIMIT:
            return min(max_output / ki, max_output * 100)
        return max_output * 100

    def update(self, error: float, dt: Optional[float] = None) -> float:
        """
        Calculate PID output for given error.

        Args:
            error: Current error (normalized -1 to 1)
            dt: Time delta in seconds (auto-calculated if None)

        Returns:
            Control output clamped to max_output
        """
        now = time.time()

        # Calculate dt
        if dt is None:
            if self._prev_time is None:
                dt = 0.05  # Default 20Hz
            else:
                dt = max(0.001, now - self._prev_time)
        self._prev_time = now
        dt = max(dt, 0.001)

        # Proportional
        p_term = self.gains.kp * error

        # Integral with anti-windup
        if error != 0.0:
            self._integral += error * dt
            self._integral = max(-self._integral_limit,
                                min(self._integral_limit, self._integral))
        i_term = self.gains.ki * self._integral

        # Derivative with filtering
        if self._first_update:
            raw_derivative = 0.0
            self._first_update = False
        else:
            raw_derivative = (error - self._prev_error) / dt

        # Low-pass filter on derivative
        alpha = max(0.0, min(1.0, self.gains.derivative_filter))
        self._filtered_derivative = (
            alpha * raw_derivative + (1 - alpha) * self._filtered_derivative
        )
        d_term = self.gains.kd * self._filtered_derivative

        self._prev_error = error

        # Sum and clamp
        output = p_term + i_term + d_term
        output = max(-self.gains.max_output, min(self.gains.max_output, output))

        # Slew rate limiting
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
        self._prev_time = None
        self._first_update = True
        self._filtered_derivative = 0.0
        self._prev_output = 0.0


class PIDController:
    """
    Multi-axis PID controller for target tracking.

    Controls yaw (horizontal), pitch (vertical), and throttle (distance)
    based on tracking error from the center of frame.
    """

    def __init__(self, config: PIDConfig):
        self.config = config
        self._yaw = PIDAxis(config.yaw)
        self._pitch = PIDAxis(config.pitch)
        self._throttle = PIDAxis(config.throttle)
        self._enabled = False
        self._dead_zone = config.dead_zone_percent / 100.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def dead_zone(self) -> float:
        return self._dead_zone

    @dead_zone.setter
    def dead_zone(self, value: float) -> None:
        self._dead_zone = value

    def enable(self) -> None:
        """Enable control output."""
        self._enabled = True
        logger.info("PID controller enabled")

    def disable(self) -> None:
        """Disable control and reset state."""
        self._enabled = False
        self.reset()
        logger.info("PID controller disabled")

    def reset(self) -> None:
        """Reset all axis controllers."""
        self._yaw.reset()
        self._pitch.reset()
        self._throttle.reset()

    def update(self, error: Optional[Tuple[float, float]],
               target_size: Optional[float] = None) -> ControlOutput:
        """
        Calculate control output from tracking error.

        Args:
            error: (x_error, y_error) normalized to [-1, 1], None if no target
            target_size: Relative target size for throttle (larger = closer)

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

        # Yaw: positive X error -> positive yaw (turn right)
        yaw_rate = self._yaw.update(x_error)

        # Pitch: positive Y error (below center) -> negative pitch (nose down)
        pitch_rate = -self._pitch.update(y_error)

        # Throttle: based on target size (maintain ~20% of frame)
        throttle_rate = 0.0
        if target_size is not None:
            size_error = 0.2 - target_size
            throttle_rate = self._throttle.update(size_error)
        else:
            self._throttle.reset()

        return ControlOutput(
            yaw_rate=yaw_rate,
            pitch_rate=pitch_rate,
            throttle_rate=throttle_rate,
            is_active=True,
        )

    def update_gains(self, axis: str, kp: float = None, ki: float = None,
                     kd: float = None, max_rate: float = None,
                     derivative_filter: float = None, slew_rate: float = None,
                     reset_integral: bool = True) -> None:
        """
        Update gains for a specific axis.

        Args:
            axis: 'yaw', 'pitch', or 'throttle'
            kp, ki, kd: New gains (optional)
            max_rate: Maximum output rate (optional)
            derivative_filter: Low-pass coefficient 0-1 (optional)
            slew_rate: Max change per second (optional)
            reset_integral: Whether to reset integral term
        """
        pid_axis = {
            "yaw": self._yaw,
            "pitch": self._pitch,
            "throttle": self._throttle,
        }.get(axis)

        if not pid_axis:
            logger.warning(f"Unknown PID axis: {axis}")
            return

        gains = pid_axis.gains

        # Validate and apply gains
        if kp is not None and kp >= 0:
            gains.kp = kp
        if ki is not None and ki >= 0:
            gains.ki = ki
            pid_axis._integral_limit = pid_axis._calculate_integral_limit(
                ki, gains.max_output
            )
        if kd is not None and kd >= 0:
            gains.kd = kd
        if max_rate is not None and max_rate > 0:
            gains.max_output = max_rate
            pid_axis._integral_limit = pid_axis._calculate_integral_limit(
                gains.ki, max_rate
            )
        if derivative_filter is not None and 0 <= derivative_filter <= 1:
            gains.derivative_filter = derivative_filter
        if slew_rate is not None and slew_rate >= 0:
            gains.slew_rate = slew_rate

        if reset_integral:
            pid_axis.reset()

        logger.info(f"Updated {axis} PID: kp={gains.kp}, ki={gains.ki}, kd={gains.kd}")

    def get_status(self) -> dict:
        """Get controller status for telemetry."""
        return {
            "enabled": self._enabled,
            "yaw_integral": self._yaw._integral,
            "pitch_integral": self._pitch._integral,
            "throttle_integral": self._throttle._integral,
        }
