"""Tests for UDP streamer module."""

import pytest
import numpy as np

from src.core.streamer import (
    OverlayConfig,
    OverlayRenderer,
    StreamerConfig,
    UDPStreamer,
)
from src.core.tracker import TrackedObject, TrackingState


class TestOverlayConfig:
    """Tests for OverlayConfig."""

    def test_defaults(self):
        """Config has sensible defaults."""
        config = OverlayConfig()

        assert config.show_detections is True
        assert config.show_locked_target is True
        assert config.font_scale == 0.6
        assert config.box_thickness == 2


class TestStreamerConfig:
    """Tests for StreamerConfig."""

    def test_from_dict(self):
        """Config loads from dict."""
        data = {
            "output": {
                "stream": {"enabled": True, "udp_host": "192.168.1.100", "udp_port": 5600},
                "resolution": {"width": 1280, "height": 720},
                "fps": 30,
                "bitrate_kbps": 2000,
            }
        }
        config = StreamerConfig.from_dict(data)

        assert config.enabled is True
        assert config.udp_host == "192.168.1.100"
        assert config.udp_port == 5600
        assert config.width == 1280
        assert config.height == 720
        assert config.fps == 30

    def test_stream_url(self):
        """Stream URL formatted correctly."""
        config = StreamerConfig(udp_host="192.168.1.50", udp_port=5600)
        streamer = UDPStreamer(config)

        assert streamer.stream_url == "udp://192.168.1.50:5600"


class TestOverlayRenderer:
    """Tests for OverlayRenderer."""

    @pytest.fixture
    def renderer(self):
        """Create renderer with default config."""
        return OverlayRenderer(OverlayConfig())

    @pytest.fixture
    def tracked_object(self):
        """Create a sample tracked object."""
        return TrackedObject(
            object_id=1,
            class_name="person",
            center=(640, 360),
            bbox=(590, 310, 690, 410),
            confidence=0.9,
            frames_visible=10,
        )

    def test_render_returns_frame(self, renderer, sample_frame):
        """Render returns a frame."""
        result = renderer.render(
            sample_frame,
            objects={},
            locked_target=None,
            tracking_state=TrackingState.SEARCHING,
        )

        assert result is not None
        assert result.shape == sample_frame.shape

    def test_render_with_objects(self, renderer, sample_frame, tracked_object):
        """Render draws detection boxes."""
        objects = {1: tracked_object}

        result = renderer.render(
            sample_frame,
            objects=objects,
            locked_target=None,
            tracking_state=TrackingState.SEARCHING,
        )

        # Frame should be modified (overlay drawn)
        assert not np.array_equal(result, np.zeros_like(sample_frame))

    def test_render_with_locked_target(self, renderer, sample_frame, tracked_object):
        """Render draws locked target differently."""
        objects = {1: tracked_object}

        result = renderer.render(
            sample_frame,
            objects=objects,
            locked_target=tracked_object,
            tracking_state=TrackingState.LOCKED,
        )

        assert result is not None

    def test_render_with_telemetry(self, renderer, sample_frame):
        """Render draws telemetry data."""
        telemetry = {
            "fps": 30.0,
            "inference_ms": 15.5,
            "altitude": 50.0,
            "speed": 5.0,
            "battery": 80.0,
        }

        result = renderer.render(
            sample_frame,
            objects={},
            locked_target=None,
            tracking_state=TrackingState.SEARCHING,
            telemetry=telemetry,
        )

        assert result is not None

    def test_crosshair_at_center(self, renderer, sample_frame):
        """Crosshair drawn at frame center."""
        result = renderer.render(
            sample_frame,
            objects={},
            locked_target=None,
            tracking_state=TrackingState.SEARCHING,
        )

        # Check center pixel area is modified (crosshair drawn)
        h, w = sample_frame.shape[:2]
        center_region = result[h//2-5:h//2+5, w//2-5:w//2+5]

        # Should have some non-zero pixels from crosshair
        assert np.any(center_region > 0)
