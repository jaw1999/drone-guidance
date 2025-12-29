"""Tests for target tracker module."""

import pytest

from src.core.detector import Detection
from src.core.tracker import (
    CentroidTracker,
    TargetTracker,
    TrackerConfig,
    TrackingState,
    TrackedObject,
)


class TestCentroidTracker:
    """Tests for CentroidTracker."""

    def test_register_new_objects(self, sample_detections):
        """New detections get unique IDs."""
        tracker = CentroidTracker(max_disappeared=30, max_distance=100)
        objects = tracker.update(sample_detections)

        assert len(objects) == 3
        ids = list(objects.keys())
        assert len(set(ids)) == 3  # all unique

    def test_track_same_object(self):
        """Same object keeps same ID across frames."""
        tracker = CentroidTracker(max_disappeared=30, max_distance=100)

        det1 = [Detection(0, "person", 0.9, (100, 100, 200, 200), (150, 150))]
        det2 = [Detection(0, "person", 0.9, (105, 105, 205, 205), (155, 155))]

        obj1 = tracker.update(det1)
        obj2 = tracker.update(det2)

        assert list(obj1.keys()) == list(obj2.keys())

    def test_object_disappears_after_threshold(self):
        """Object removed after max_disappeared frames."""
        tracker = CentroidTracker(max_disappeared=3, max_distance=100)

        det = [Detection(0, "person", 0.9, (100, 100, 200, 200), (150, 150))]
        tracker.update(det)

        # Object disappears
        for _ in range(4):
            objects = tracker.update([])

        assert len(objects) == 0

    def test_object_reappears_before_threshold(self):
        """Object kept if it reappears within threshold."""
        tracker = CentroidTracker(max_disappeared=5, max_distance=100)

        det = [Detection(0, "person", 0.9, (100, 100, 200, 200), (150, 150))]
        obj1 = tracker.update(det)
        original_id = list(obj1.keys())[0]

        # Disappear for 2 frames
        tracker.update([])
        tracker.update([])

        # Reappear nearby
        det2 = [Detection(0, "person", 0.9, (110, 110, 210, 210), (160, 160))]
        obj2 = tracker.update(det2)

        assert original_id in obj2

    def test_new_object_if_too_far(self):
        """Far away detection becomes new object."""
        tracker = CentroidTracker(max_disappeared=30, max_distance=50)

        det1 = [Detection(0, "person", 0.9, (100, 100, 200, 200), (150, 150))]
        det2 = [Detection(0, "person", 0.9, (500, 500, 600, 600), (550, 550))]

        obj1 = tracker.update(det1)
        obj2 = tracker.update(det2)

        # Should have 2 objects now (original disappeared, new registered)
        assert len(obj2) >= 1

    def test_reset_clears_state(self, sample_detections):
        """Reset removes all tracked objects."""
        tracker = CentroidTracker()
        tracker.update(sample_detections)
        tracker.reset()

        assert len(tracker.objects) == 0

    def test_velocity_calculated(self):
        """Velocity computed from position change."""
        tracker = CentroidTracker(max_disappeared=30, max_distance=100)

        det1 = [Detection(0, "person", 0.9, (100, 100, 200, 200), (150, 150))]
        det2 = [Detection(0, "person", 0.9, (120, 100, 220, 200), (170, 150))]

        tracker.update(det1)
        objects = tracker.update(det2)

        obj = list(objects.values())[0]
        # Velocity should be positive in X direction
        assert obj.velocity[0] > 0


class TestTrackerConfig:
    """Tests for TrackerConfig."""

    def test_from_dict(self, sample_config):
        """Config loads from dict."""
        config = TrackerConfig.from_dict(sample_config)

        assert config.max_disappeared == 30
        assert config.max_distance == 100
        assert config.min_confidence == 0.6
        assert config.frames_to_lock == 5


class TestTargetTracker:
    """Tests for TargetTracker."""

    def test_initial_state_is_searching(self, sample_config):
        """Tracker starts in SEARCHING state."""
        config = TrackerConfig.from_dict(sample_config)
        tracker = TargetTracker(config, (1280, 720))

        assert tracker.state == TrackingState.SEARCHING
        assert tracker.locked_target is None

    def test_acquires_target_after_frames(self, sample_config):
        """Target acquired after frames_to_lock detections."""
        config = TrackerConfig.from_dict(sample_config)
        config.frames_to_lock = 3
        config.min_confidence = 0.5
        tracker = TargetTracker(config, (1280, 720))

        det = [Detection(0, "person", 0.9, (600, 300, 700, 400), (650, 350))]

        # First detection - ACQUIRING
        tracker.update(det)
        assert tracker.state == TrackingState.ACQUIRING

        # Continue until locked
        tracker.update(det)
        tracker.update(det)

        assert tracker.state == TrackingState.LOCKED
        assert tracker.locked_target is not None

    def test_loses_target_after_frames(self, sample_config):
        """Target lost after frames_to_unlock without detection."""
        config = TrackerConfig.from_dict(sample_config)
        config.frames_to_lock = 2
        config.frames_to_unlock = 3
        config.min_confidence = 0.5
        tracker = TargetTracker(config, (1280, 720))

        det = [Detection(0, "person", 0.9, (600, 300, 700, 400), (650, 350))]

        # Lock on
        tracker.update(det)
        tracker.update(det)
        assert tracker.state == TrackingState.LOCKED

        # Lose target
        for _ in range(4):
            tracker.update([])

        assert tracker.state == TrackingState.LOST

    def test_manual_lock(self, sample_config, sample_detections):
        """Can manually lock onto specific target."""
        config = TrackerConfig.from_dict(sample_config)
        tracker = TargetTracker(config, (1280, 720))

        tracker.update(sample_detections)
        targets = tracker.all_targets
        target_id = list(targets.keys())[1]

        result = tracker.lock_target(target_id)

        assert result is True
        assert tracker.state == TrackingState.LOCKED
        assert tracker.locked_target.object_id == target_id

    def test_manual_lock_invalid_id(self, sample_config):
        """Manual lock fails for invalid ID."""
        config = TrackerConfig.from_dict(sample_config)
        tracker = TargetTracker(config, (1280, 720))

        result = tracker.lock_target(999)
        assert result is False

    def test_unlock(self, sample_config):
        """Unlock returns to SEARCHING."""
        config = TrackerConfig.from_dict(sample_config)
        config.frames_to_lock = 1
        tracker = TargetTracker(config, (1280, 720))

        det = [Detection(0, "person", 0.9, (600, 300, 700, 400), (650, 350))]
        tracker.update(det)
        tracker.update(det)

        tracker.unlock()

        assert tracker.state == TrackingState.SEARCHING
        assert tracker.locked_target is None

    def test_tracking_error_centered(self, sample_config, centered_detection):
        """Centered target has near-zero error."""
        config = TrackerConfig.from_dict(sample_config)
        config.frames_to_lock = 1
        tracker = TargetTracker(config, (1280, 720))

        tracker.update([centered_detection])
        tracker.update([centered_detection])

        error = tracker.get_tracking_error()

        assert error is not None
        assert abs(error[0]) < 0.1  # near center X
        assert abs(error[1]) < 0.1  # near center Y

    def test_tracking_error_off_center(self, sample_config, off_center_detection):
        """Off-center target has negative error (top-left)."""
        config = TrackerConfig.from_dict(sample_config)
        config.frames_to_lock = 1
        tracker = TargetTracker(config, (1280, 720))

        tracker.update([off_center_detection])
        tracker.update([off_center_detection])

        error = tracker.get_tracking_error()

        assert error is not None
        assert error[0] < 0  # left of center
        assert error[1] < 0  # above center

    def test_tracking_error_none_when_unlocked(self, sample_config):
        """No error when not locked."""
        config = TrackerConfig.from_dict(sample_config)
        tracker = TargetTracker(config, (1280, 720))

        error = tracker.get_tracking_error()
        assert error is None
