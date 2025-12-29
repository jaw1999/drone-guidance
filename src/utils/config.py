"""Configuration loading utilities with validation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# Validation rules: (min, max, default)
VALIDATION_RULES: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "detector": {
        "confidence_threshold": (0.1, 1.0, 0.5),
        "nms_threshold": (0.1, 1.0, 0.45),
        "input_size": (160, 1280, 416),
        "detection_interval": (1, 10, 2),
    },
    "tracker": {
        "max_disappeared": (1, 300, 30),
        "max_distance": (10, 1000, 150),
    },
    "tracker.lock_on": {
        "min_confidence": (0.1, 1.0, 0.6),
        "frames_to_lock": (1, 60, 5),
        "frames_to_unlock": (1, 120, 15),
    },
    "pid.yaw": {
        "kp": (0.0, 5.0, 0.5),
        "ki": (0.0, 1.0, 0.01),
        "kd": (0.0, 2.0, 0.1),
        "max_rate": (1.0, 180.0, 30.0),
    },
    "pid.pitch": {
        "kp": (0.0, 5.0, 0.4),
        "ki": (0.0, 1.0, 0.01),
        "kd": (0.0, 2.0, 0.08),
        "max_rate": (1.0, 90.0, 20.0),
    },
    "pid.throttle": {
        "kp": (0.0, 5.0, 0.3),
        "ki": (0.0, 1.0, 0.005),
        "kd": (0.0, 2.0, 0.05),
        "max_rate": (0.1, 10.0, 2.0),
    },
    "pid": {
        "dead_zone_percent": (0.0, 50.0, 5.0),
        "update_rate": (1, 100, 20),
    },
    "safety": {
        "search_timeout": (1.0, 120.0, 10.0),
        "min_battery_percent": (5, 50, 20),
        "max_tracking_speed": (1.0, 50.0, 10.0),
    },
    "safety.geofence": {
        "max_distance_m": (10, 10000, 500),
        "max_altitude_m": (10, 500, 120),
        "min_altitude_m": (0, 100, 10),
    },
    "output": {
        "fps": (1, 60, 30),
        "bitrate_kbps": (500, 10000, 2000),
    },
    "camera": {
        "fps": (1, 120, 30),
        "buffer_size": (1, 10, 1),
        "reconnect_attempts": (1, 20, 5),
        "reconnect_delay_sec": (0.5, 30.0, 2.0),
    },
    "camera.fov": {
        "horizontal": (10.0, 180.0, 90.0),
        "vertical": (10.0, 180.0, 60.0),
    },
}


def _get_nested(config: Dict, path: str) -> Optional[Dict]:
    """Get nested config section by dot-separated path."""
    parts = path.split(".")
    current = config
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current if isinstance(current, dict) else None


def _set_nested(config: Dict, path: str, key: str, value: Any) -> None:
    """Set a value in nested config by dot-separated path.

    Creates intermediate dicts as needed. If an intermediate value
    exists but is not a dict, it is replaced with a dict.
    """
    parts = path.split(".")
    current = config
    for part in parts:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[key] = value


def validate_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate configuration values and apply defaults for invalid/missing values.

    Returns:
        Tuple of (validated_config, list_of_warnings)
    """
    warnings: List[str] = []

    for section_path, rules in VALIDATION_RULES.items():
        section = _get_nested(config, section_path)

        for key, (min_val, max_val, default) in rules.items():
            if section is None or key not in section:
                _set_nested(config, section_path, key, default)
                continue

            value = section[key]

            if not isinstance(value, (int, float)):
                warnings.append(f"{section_path}.{key}: invalid type, using default {default}")
                _set_nested(config, section_path, key, default)
                continue

            if value < min_val or value > max_val:
                clamped = max(min_val, min(max_val, value))
                warnings.append(
                    f"{section_path}.{key}: {value} out of range [{min_val}, {max_val}], clamped to {clamped}"
                )
                _set_nested(config, section_path, key, clamped)

    # Validate specific constraints
    safety = config.get("safety", {}).get("geofence", {})
    if safety.get("min_altitude_m", 0) >= safety.get("max_altitude_m", 120):
        warnings.append("safety.geofence: min_altitude must be less than max_altitude, fixing")
        config["safety"]["geofence"]["min_altitude_m"] = 0

    return config, warnings


def load_config(config_path: str = "config/default.yaml") -> Dict[str, Any]:
    """Load and validate configuration from YAML file."""
    path = Path(config_path)

    if not path.exists():
        logger.warning(f"Config not found: {config_path}, using defaults")
        config, _ = validate_config({})
        return config

    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config {config_path}: {e}")
        logger.warning("Using default configuration")
        config, _ = validate_config({})
        return config
    except Exception as e:
        logger.error(f"Failed to read config {config_path}: {e}")
        config, _ = validate_config({})
        return config

    if config is None:
        config = {}

    config, warnings = validate_config(config)
    for warning in warnings:
        logger.warning(f"Config validation: {warning}")

    logger.info(f"Loaded config from {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """Save configuration to YAML file."""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved config to {config_path}")
    return True
