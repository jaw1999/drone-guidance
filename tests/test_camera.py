"""Tests for camera capture module."""

import pytest
import numpy as np

from src.core.camera import CameraConfig, CameraCapture


class TestCameraConfig:
    """Tests for CameraConfig."""

    def test_from_dict_with_full_config(self, sample_config):
        """Config loads all values from dict."""
        config = CameraConfig.from_dict(sample_config)

        assert config.rtsp_url == "rtsp://test:554/stream"
        assert config.width == 1280
        assert config.height == 720
        assert config.fps == 30
        assert config.fov_horizontal == 90.0
        assert config.fov_vertical == 60.0

    def test_from_dict_with_empty_config(self):
        """Config uses defaults for missing values."""
        config = CameraConfig.from_dict({})

        assert config.rtsp_url == ""
        assert config.width == 1920
        assert config.height == 1080
        assert config.fps == 30

    def test_from_dict_partial_config(self):
        """Config merges partial values with defaults."""
        partial = {"camera": {"rtsp_url": "rtsp://custom:554/live"}}
        config = CameraConfig.from_dict(partial)

        assert config.rtsp_url == "rtsp://custom:554/live"
        assert config.width == 1920  # default


class TestCameraCapture:
    """Tests for CameraCapture."""

    def test_initial_state(self, sample_config):
        """Camera starts disconnected."""
        config = CameraConfig.from_dict(sample_config)
        camera = CameraCapture(config)

        assert not camera.is_connected
        assert camera.frame_count == 0
        assert camera.actual_fps == 0.0

    def test_get_frame_when_not_started(self, sample_config):
        """Returns None when no frames captured."""
        config = CameraConfig.from_dict(sample_config)
        camera = CameraCapture(config)

        frame = camera.get_frame()
        assert frame is None

    def test_frame_callback_can_be_set(self, sample_config):
        """Frame callback is stored."""
        config = CameraConfig.from_dict(sample_config)
        camera = CameraCapture(config)

        callback_called = []

        def callback(frame):
            callback_called.append(True)

        camera.set_frame_callback(callback)
        assert camera._on_frame_callback is not None
