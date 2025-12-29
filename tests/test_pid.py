"""Tests for PID controller module."""

import pytest
import time

from src.core.pid import PIDAxis, PIDGains, PIDConfig, PIDController, ControlOutput


class TestPIDGains:
    """Tests for PIDGains."""

    def test_from_dict(self):
        """Gains load from dict."""
        data = {"kp": 1.0, "ki": 0.5, "kd": 0.2, "max_rate": 50.0}
        gains = PIDGains.from_dict(data)

        assert gains.kp == 1.0
        assert gains.ki == 0.5
        assert gains.kd == 0.2
        assert gains.max_output == 50.0

    def test_defaults(self):
        """Gains have sensible defaults."""
        gains = PIDGains.from_dict({})

        assert gains.kp == 0.5
        assert gains.ki == 0.01
        assert gains.kd == 0.1


class TestPIDAxis:
    """Tests for single-axis PID controller."""

    def test_proportional_response(self):
        """P term responds to error."""
        gains = PIDGains(kp=1.0, ki=0.0, kd=0.0, max_output=100.0)
        pid = PIDAxis(gains)

        output = pid.update(0.5, dt=0.1)

        assert output == pytest.approx(0.5, abs=0.01)

    def test_integral_accumulates(self):
        """I term accumulates over time."""
        gains = PIDGains(kp=0.0, ki=1.0, kd=0.0, max_output=100.0)
        pid = PIDAxis(gains)

        # Apply constant error
        pid.update(1.0, dt=0.1)
        output = pid.update(1.0, dt=0.1)

        assert output > 0.1  # should have accumulated

    def test_derivative_responds_to_change(self):
        """D term responds to error change."""
        gains = PIDGains(kp=0.0, ki=0.0, kd=1.0, max_output=100.0)
        pid = PIDAxis(gains)

        pid.update(0.0, dt=0.1)
        output = pid.update(1.0, dt=0.1)

        assert output > 0  # error increased, D should respond

    def test_output_clamped(self):
        """Output clamped to max_output."""
        gains = PIDGains(kp=100.0, ki=0.0, kd=0.0, max_output=10.0)
        pid = PIDAxis(gains)

        output = pid.update(1.0, dt=0.1)

        assert output == 10.0

    def test_negative_output_clamped(self):
        """Negative output clamped to -max_output."""
        gains = PIDGains(kp=100.0, ki=0.0, kd=0.0, max_output=10.0)
        pid = PIDAxis(gains)

        output = pid.update(-1.0, dt=0.1)

        assert output == -10.0

    def test_reset_clears_state(self):
        """Reset clears integral and derivative state."""
        gains = PIDGains(kp=0.0, ki=1.0, kd=0.0, max_output=100.0)
        pid = PIDAxis(gains)

        # Build up integral
        for _ in range(10):
            pid.update(1.0, dt=0.1)

        pid.reset()

        # After reset, integral should be zero
        assert pid._integral == 0.0
        assert pid._prev_error == 0.0

    def test_anti_windup(self):
        """Integral term has anti-windup limiting."""
        gains = PIDGains(kp=0.0, ki=1.0, kd=0.0, max_output=10.0)
        pid = PIDAxis(gains)

        # Try to wind up integral excessively
        for _ in range(1000):
            pid.update(1.0, dt=0.1)

        # Integral should be clamped
        assert abs(pid._integral) <= pid._integral_limit


class TestPIDConfig:
    """Tests for PIDConfig."""

    def test_from_dict(self, sample_config):
        """Config loads all axes from dict."""
        config = PIDConfig.from_dict(sample_config)

        assert config.yaw.kp == 0.5
        assert config.pitch.kp == 0.4
        assert config.throttle.kp == 0.3
        assert config.dead_zone_percent == 5.0


class TestPIDController:
    """Tests for multi-axis PID controller."""

    def test_disabled_by_default(self, sample_config):
        """Controller starts disabled."""
        config = PIDConfig.from_dict(sample_config)
        pid = PIDController(config)

        assert not pid.enabled

    def test_enable_disable(self, sample_config):
        """Controller can be enabled and disabled."""
        config = PIDConfig.from_dict(sample_config)
        pid = PIDController(config)

        pid.enable()
        assert pid.enabled

        pid.disable()
        assert not pid.enabled

    def test_no_output_when_disabled(self, sample_config):
        """Returns inactive output when disabled."""
        config = PIDConfig.from_dict(sample_config)
        pid = PIDController(config)

        output = pid.update((0.5, 0.5))

        assert not output.is_active
        assert output.yaw_rate == 0.0

    def test_no_output_when_no_error(self, sample_config):
        """Returns inactive output when error is None."""
        config = PIDConfig.from_dict(sample_config)
        pid = PIDController(config)
        pid.enable()

        output = pid.update(None)

        assert not output.is_active

    def test_yaw_responds_to_x_error(self, sample_config):
        """Positive X error produces positive yaw rate."""
        config = PIDConfig.from_dict(sample_config)
        pid = PIDController(config)
        pid.enable()

        output = pid.update((0.5, 0.0))

        assert output.is_active
        assert output.yaw_rate > 0

    def test_pitch_responds_to_y_error(self, sample_config):
        """Positive Y error (below center) produces negative pitch."""
        config = PIDConfig.from_dict(sample_config)
        pid = PIDController(config)
        pid.enable()

        output = pid.update((0.0, 0.5))

        assert output.is_active
        assert output.pitch_rate < 0  # nose down to track target below

    def test_dead_zone(self, sample_config):
        """Small errors within dead zone produce no output."""
        config = PIDConfig.from_dict(sample_config)
        config.dead_zone_percent = 10.0  # 10% dead zone
        pid = PIDController(config)
        pid.enable()

        # Error within dead zone (0.05 < 0.1)
        output = pid.update((0.05, 0.05))

        assert output.yaw_rate == 0.0
        assert output.pitch_rate == 0.0

    def test_throttle_with_target_size(self, sample_config):
        """Throttle responds to target size error."""
        config = PIDConfig.from_dict(sample_config)
        pid = PIDController(config)
        pid.enable()

        # Small target (far away) - should increase throttle
        output = pid.update((0.0, 0.0), target_size=0.05)

        assert output.throttle_rate > 0

    def test_reset_clears_all_axes(self, sample_config):
        """Reset clears state on all axes."""
        config = PIDConfig.from_dict(sample_config)
        pid = PIDController(config)
        pid.enable()

        # Build up state
        for _ in range(10):
            pid.update((0.5, 0.5))

        pid.reset()

        assert pid._yaw._integral == 0.0
        assert pid._pitch._integral == 0.0
        assert pid._throttle._integral == 0.0

    def test_get_status(self, sample_config):
        """Status returns controller state."""
        config = PIDConfig.from_dict(sample_config)
        pid = PIDController(config)
        pid.enable()

        status = pid.get_status()

        assert "enabled" in status
        assert status["enabled"] is True
