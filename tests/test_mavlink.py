"""Tests for MAVLink controller module."""

import pytest

from src.core.mavlink_controller import (
    MAVLinkConfig,
    MAVLinkController,
    SafetyAction,
    SafetyConfig,
    VehicleState,
)


class TestSafetyAction:
    """Tests for SafetyAction enum."""

    def test_all_actions_defined(self):
        """All expected safety actions exist."""
        assert SafetyAction.HOVER.value == "hover"
        assert SafetyAction.LOITER.value == "loiter"
        assert SafetyAction.RTL.value == "rtl"
        assert SafetyAction.CONTINUE_LAST.value == "continue_last"
        assert SafetyAction.LAND.value == "land"


class TestMAVLinkConfig:
    """Tests for MAVLinkConfig."""

    def test_from_dict(self, sample_config):
        """Config loads from dict."""
        config = MAVLinkConfig.from_dict(sample_config)

        assert config.connection == "udp:127.0.0.1:14550"
        assert config.enable_control is False

    def test_defaults(self):
        """Config has sensible defaults."""
        config = MAVLinkConfig.from_dict({})

        assert config.source_system == 255
        assert config.source_component == 190
        assert config.heartbeat_rate == 1.0


class TestSafetyConfig:
    """Tests for SafetyConfig."""

    def test_from_dict(self, sample_config):
        """Config loads from dict."""
        config = SafetyConfig.from_dict(sample_config)

        assert config.target_lost_action == SafetyAction.LOITER
        assert config.search_timeout == 10.0
        assert config.max_distance_m == 500
        assert config.max_altitude_m == 120
        assert config.min_altitude_m == 10

    def test_invalid_action_defaults_to_loiter(self):
        """Invalid action string defaults to LOITER."""
        config = SafetyConfig.from_dict({
            "safety": {"target_lost_action": "invalid_action"}
        })

        assert config.target_lost_action == SafetyAction.LOITER

    def test_geofence_settings(self, sample_config):
        """Geofence settings load correctly."""
        config = SafetyConfig.from_dict(sample_config)

        assert config.geofence_enabled is True
        assert config.max_distance_m == 500
        assert config.max_altitude_m == 120


class TestVehicleState:
    """Tests for VehicleState dataclass."""

    def test_default_values(self):
        """VehicleState has safe defaults."""
        state = VehicleState()

        assert state.armed is False
        assert state.battery_percent == 100.0
        assert state.altitude_rel == 0.0


class TestMAVLinkController:
    """Tests for MAVLinkController."""

    def test_initial_state(self, sample_config):
        """Controller starts disconnected."""
        mav_config = MAVLinkConfig.from_dict(sample_config)
        safety_config = SafetyConfig.from_dict(sample_config)
        controller = MAVLinkController(mav_config, safety_config)

        assert not controller.is_connected
        assert not controller.tracking_enabled

    def test_vehicle_state_default(self, sample_config):
        """Vehicle state has defaults before connection."""
        mav_config = MAVLinkConfig.from_dict(sample_config)
        safety_config = SafetyConfig.from_dict(sample_config)
        controller = MAVLinkController(mav_config, safety_config)

        state = controller.vehicle_state

        assert state.armed is False
        assert state.battery_percent == 100.0

    def test_tracking_requires_connection(self, sample_config):
        """Cannot enable tracking without connection."""
        mav_config = MAVLinkConfig.from_dict(sample_config)
        safety_config = SafetyConfig.from_dict(sample_config)
        controller = MAVLinkController(mav_config, safety_config)

        result = controller.enable_tracking()

        assert result is False
        assert not controller.tracking_enabled

    def test_emergency_stop(self, sample_config):
        """Emergency stop disables tracking."""
        mav_config = MAVLinkConfig.from_dict(sample_config)
        safety_config = SafetyConfig.from_dict(sample_config)
        controller = MAVLinkController(mav_config, safety_config)

        controller._tracking_enabled = True
        controller.emergency_stop()

        assert not controller.tracking_enabled
        assert controller._emergency_stop is True

    def test_clear_emergency(self, sample_config):
        """Emergency stop can be cleared."""
        mav_config = MAVLinkConfig.from_dict(sample_config)
        safety_config = SafetyConfig.from_dict(sample_config)
        controller = MAVLinkController(mav_config, safety_config)

        controller.emergency_stop()
        controller.clear_emergency_stop()

        assert controller._emergency_stop is False

    def test_safety_callback(self, sample_config):
        """Safety callback is invoked on emergency."""
        mav_config = MAVLinkConfig.from_dict(sample_config)
        safety_config = SafetyConfig.from_dict(sample_config)
        controller = MAVLinkController(mav_config, safety_config)

        callback_data = []

        def callback(reason):
            callback_data.append(reason)

        controller.set_safety_callback(callback)
        controller.emergency_stop()

        assert "emergency_stop" in callback_data

    def test_send_commands_requires_tracking(self, sample_config):
        """Commands rejected when tracking disabled."""
        mav_config = MAVLinkConfig.from_dict(sample_config)
        safety_config = SafetyConfig.from_dict(sample_config)
        controller = MAVLinkController(mav_config, safety_config)

        result = controller.send_rate_commands(10.0, 5.0, 1.0)

        assert result is False

    def test_is_armed_property(self, sample_config):
        """is_armed reflects vehicle state."""
        mav_config = MAVLinkConfig.from_dict(sample_config)
        safety_config = SafetyConfig.from_dict(sample_config)
        controller = MAVLinkController(mav_config, safety_config)

        assert controller.is_armed is False

        controller._vehicle_state.armed = True
        assert controller.is_armed is True
