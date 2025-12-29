"""Tests for async processing pipeline."""

import pytest
import time
import numpy as np

from src.core.pipeline import (
    PipelineConfig,
    FrameData,
    TrackingInterpolator,
)
from src.core.tracker import TrackedObject, TrackingState


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_from_dict(self, sample_config):
        """Config loads from dict."""
        # Add pipeline-specific config
        sample_config["detector"]["detection_interval"] = 3
        sample_config["detector"]["detection_resolution"] = {
            "width": 640, "height": 480
        }
        sample_config["detector"]["roi_detection"] = {
            "enabled": True,
            "padding_percent": 50.0,
        }

        config = PipelineConfig.from_dict(sample_config)

        assert config.detection_interval == 3
        assert config.detection_width == 640
        assert config.detection_height == 480
        assert config.roi_enabled is True
        assert config.roi_padding_percent == 50.0

    def test_defaults(self):
        """Config has sensible defaults."""
        config = PipelineConfig.from_dict({})

        assert config.detection_interval == 3
        assert config.detection_width == 640
        assert config.detection_height == 480


class TestFrameData:
    """Tests for FrameData container."""

    def test_default_values(self, sample_frame):
        """FrameData has correct defaults."""
        data = FrameData(
            frame=sample_frame,
            timestamp=time.time(),
            frame_id=1,
        )

        assert len(data.detections) == 0
        assert len(data.tracked_objects) == 0
        assert data.locked_target is None
        assert data.tracking_state == TrackingState.SEARCHING


class TestTrackingInterpolator:
    """Tests for TrackingInterpolator."""

    @pytest.fixture
    def interpolator(self):
        return TrackingInterpolator()

    @pytest.fixture
    def sample_tracked_objects(self):
        """Create tracked objects with velocity."""
        return {
            1: TrackedObject(
                object_id=1,
                class_name="person",
                center=(100, 100),
                bbox=(50, 50, 150, 150),
                confidence=0.9,
                velocity=(10.0, 5.0),  # 10px/sec right, 5px/sec down
                last_update=time.time(),
            ),
            2: TrackedObject(
                object_id=2,
                class_name="car",
                center=(500, 300),
                bbox=(450, 250, 550, 350),
                confidence=0.85,
                velocity=(-5.0, 0.0),  # 5px/sec left
                last_update=time.time(),
            ),
        }

    def test_interpolate_empty(self, interpolator):
        """Interpolate returns empty when no objects."""
        result = interpolator.interpolate(time.time())
        assert result == {}

    def test_interpolate_after_update(self, interpolator, sample_tracked_objects):
        """Interpolation predicts positions based on velocity."""
        now = time.time()
        interpolator.update_from_detection(sample_tracked_objects, now)

        # Interpolate 0.5 seconds later
        result = interpolator.interpolate(now + 0.5)

        # Object 1: moved right 5px (10*0.5), down 2.5px (5*0.5)
        assert result[1].center[0] == 105  # 100 + 10*0.5
        assert result[1].center[1] == 102  # 100 + 5*0.5 (int)

        # Object 2: moved left 2.5px
        assert result[2].center[0] == 497  # 500 - 5*0.5 (int)

    def test_interpolate_no_extrapolate_too_far(self, interpolator, sample_tracked_objects):
        """Don't extrapolate more than 1 second."""
        now = time.time()
        interpolator.update_from_detection(sample_tracked_objects, now)

        # Try to interpolate 2 seconds later
        result = interpolator.interpolate(now + 2.0)

        # Should return original positions (no extrapolation)
        assert result[1].center == sample_tracked_objects[1].center

    def test_bbox_moves_with_center(self, interpolator, sample_tracked_objects):
        """Bbox is updated to match new center position."""
        now = time.time()
        interpolator.update_from_detection(sample_tracked_objects, now)

        result = interpolator.interpolate(now + 0.5)

        # Check bbox moved with center
        obj = result[1]
        bbox_center_x = (obj.bbox[0] + obj.bbox[2]) // 2
        bbox_center_y = (obj.bbox[1] + obj.bbox[3]) // 2

        assert bbox_center_x == obj.center[0]
        assert bbox_center_y == obj.center[1]
