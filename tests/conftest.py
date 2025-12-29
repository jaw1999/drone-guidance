"""Pytest fixtures for Terminal Guidance tests."""

import pytest
import numpy as np

from src.core.detector import Detection
from src.core.tracker import TrackedObject


@pytest.fixture
def sample_frame():
    """Generate a blank 720p test frame."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_1080p():
    """Generate a blank 1080p test frame."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def sample_detection():
    """Create a sample detection at frame center."""
    return Detection(
        class_id=0,
        class_name="person",
        confidence=0.85,
        bbox=(600, 340, 680, 420),
        center=(640, 380),
    )


@pytest.fixture
def sample_detections():
    """Create multiple sample detections."""
    return [
        Detection(
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=(100, 100, 200, 300),
            center=(150, 200),
        ),
        Detection(
            class_id=0,
            class_name="person",
            confidence=0.75,
            bbox=(500, 200, 600, 400),
            center=(550, 300),
        ),
        Detection(
            class_id=2,
            class_name="car",
            confidence=0.82,
            bbox=(800, 400, 1000, 550),
            center=(900, 475),
        ),
    ]


@pytest.fixture
def centered_detection():
    """Detection perfectly centered in 1280x720 frame."""
    return Detection(
        class_id=0,
        class_name="person",
        confidence=0.95,
        bbox=(590, 310, 690, 410),
        center=(640, 360),
    )


@pytest.fixture
def off_center_detection():
    """Detection in top-left quadrant."""
    return Detection(
        class_id=0,
        class_name="person",
        confidence=0.88,
        bbox=(100, 100, 200, 200),
        center=(150, 150),
    )


@pytest.fixture
def sample_config():
    """Sample configuration dictionary."""
    return {
        "camera": {
            "rtsp_url": "rtsp://test:554/stream",
            "resolution": {"width": 1280, "height": 720},
            "fps": 30,
            "buffer_size": 1,
            "fov": {"horizontal": 90.0, "vertical": 60.0},
        },
        "detector": {
            "model": "yolov8n",
            "confidence_threshold": 0.5,
            "target_classes": ["person", "car"],
        },
        "tracker": {
            "max_disappeared": 30,
            "max_distance": 100,
            "lock_on": {
                "min_confidence": 0.6,
                "frames_to_lock": 5,
                "frames_to_unlock": 15,
            },
        },
        "pid": {
            "yaw": {"kp": 0.5, "ki": 0.01, "kd": 0.1, "max_rate": 30.0},
            "pitch": {"kp": 0.4, "ki": 0.01, "kd": 0.08, "max_rate": 20.0},
            "throttle": {"kp": 0.3, "ki": 0.005, "kd": 0.05, "max_rate": 2.0},
            "dead_zone_percent": 5.0,
        },
        "mavlink": {
            "connection": "udp:127.0.0.1:14550",
            "enable_control": False,
        },
        "safety": {
            "target_lost_action": "loiter",
            "search_timeout": 10.0,
            "geofence": {
                "enabled": True,
                "max_distance_m": 500,
                "max_altitude_m": 120,
                "min_altitude_m": 10,
            },
        },
    }
