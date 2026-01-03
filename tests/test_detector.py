"""Tests for object detector module."""

import pytest
import numpy as np

from src.core.detector import Detection, DetectorConfig


class TestDetection:
    """Tests for Detection dataclass."""

    def test_detection_properties(self):
        """Detection calculates width, height, area."""
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=(100, 100, 200, 300),
            center=(150, 200),
        )

        assert det.width == 100
        assert det.height == 200
        assert det.area == 20000

    def test_detection_center(self, sample_detection):
        """Detection stores center correctly."""
        assert sample_detection.center == (640, 380)


class TestDetectorConfig:
    """Tests for DetectorConfig."""

    def test_from_dict_full_config(self, sample_config):
        """Config loads from complete dict."""
        config = DetectorConfig.from_dict(sample_config)

        assert config.model == "yolov8n"
        assert config.confidence_threshold == 0.5
        assert "person" in config.target_classes
        assert "car" in config.target_classes

    def test_from_dict_empty(self):
        """Config uses defaults for empty dict."""
        config = DetectorConfig.from_dict({})

        assert config.model == "yolov8n"
        assert config.confidence_threshold == 0.5
        assert config.input_size == 640
        assert config.resolution == "640"

    def test_target_classes_empty_means_all(self):
        """Empty target_classes list means detect all."""
        config = DetectorConfig.from_dict({"detector": {"target_classes": []}})
        assert config.target_classes == []
