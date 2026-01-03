"""Flask configuration UI for Terminal Guidance."""

import logging
import re
import time
from typing import TYPE_CHECKING, Any, Optional, Tuple

import psutil
from flask import Flask, Response, jsonify, render_template_string, request

if TYPE_CHECKING:
    from ..app import TerminalGuidance

logger = logging.getLogger(__name__)

# Input validation patterns and limits
IP_ADDRESS_PATTERN = re.compile(
    r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
    r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
)
HOSTNAME_PATTERN = re.compile(
    r'^(?=.{1,253}$)(?:(?!-)[A-Za-z0-9-]{1,63}(?<!-)\.)*'
    r'(?!-)[A-Za-z0-9-]{1,63}(?<!-)$'
)
MAVLINK_CONNECTION_PATTERN = re.compile(
    r'^(udp|tcp|serial):[\w./:@-]+$'
)
# Allowed schemes for camera URLs
CAMERA_URL_SCHEMES = {'rtsp', 'http', 'https', 'file'}
# Port range
MIN_PORT = 1
MAX_PORT = 65535
# Bitrate limits (kbps)
MIN_BITRATE = 100
MAX_BITRATE = 50000


def validate_ip_or_hostname(value: str) -> bool:
    """Validate that a string is a valid IP address or hostname."""
    if not value or not isinstance(value, str):
        return False
    value = value.strip()
    if not value:
        return False
    # Check for valid IP address
    if IP_ADDRESS_PATTERN.match(value):
        return True
    # Check for valid hostname
    if HOSTNAME_PATTERN.match(value):
        return True
    return False


def validate_port(value: Any) -> bool:
    """Validate that a value is a valid port number."""
    try:
        port = int(value)
        return MIN_PORT <= port <= MAX_PORT
    except (ValueError, TypeError):
        return False


def validate_camera_url(url: str) -> Tuple[bool, Optional[str]]:
    """Validate camera URL for safety.

    Returns:
        (is_valid, error_message) tuple
    """
    if not url or not isinstance(url, str):
        return False, "Camera URL is required"

    url = url.strip()

    # Check for shell injection characters
    dangerous_chars = [';', '|', '&', '$', '`', '\n', '\r']
    for char in dangerous_chars:
        if char in url:
            return False, f"Invalid character in URL: {repr(char)}"

    # Check URL scheme
    if '://' in url:
        scheme = url.split('://')[0].lower()
        if scheme not in CAMERA_URL_SCHEMES:
            return False, f"Unsupported URL scheme: {scheme}"
    elif url.isdigit():
        # Webcam index (0, 1, 2, etc.)
        return True, None
    else:
        return False, "URL must include scheme (rtsp://, http://, etc.) or be a webcam index"

    return True, None


def validate_mavlink_connection(conn: str) -> Tuple[bool, Optional[str]]:
    """Validate MAVLink connection string.

    Returns:
        (is_valid, error_message) tuple
    """
    if not conn or not isinstance(conn, str):
        return False, "MAVLink connection string is required"

    conn = conn.strip()

    # Check for shell injection characters
    dangerous_chars = [';', '|', '&', '$', '`', '\n', '\r', ' ']
    for char in dangerous_chars:
        if char in conn:
            return False, f"Invalid character in connection string: {repr(char)}"

    # Basic pattern validation
    if not MAVLINK_CONNECTION_PATTERN.match(conn):
        return False, "Connection must be format: udp:host:port, tcp:host:port, or serial:/dev/..."

    return True, None


def validate_network_config(config: dict) -> Tuple[bool, Optional[str]]:
    """Validate network-related configuration values.

    Returns:
        (is_valid, error_message) tuple
    """
    # Validate camera URL if present
    camera = config.get('camera', {})
    if 'rtsp_url' in camera and camera['rtsp_url']:
        valid, err = validate_camera_url(camera['rtsp_url'])
        if not valid:
            return False, f"Camera URL: {err}"

    # Validate MAVLink connection if present
    mavlink = config.get('mavlink', {})
    if 'connection' in mavlink and mavlink['connection']:
        valid, err = validate_mavlink_connection(mavlink['connection'])
        if not valid:
            return False, f"MAVLink: {err}"

    # Validate stream settings if present
    output = config.get('output', {})
    stream = output.get('stream', {})

    if 'udp_host' in stream and stream['udp_host']:
        if not validate_ip_or_hostname(stream['udp_host']):
            return False, "Stream host must be a valid IP address or hostname"

    if 'udp_port' in stream:
        if not validate_port(stream['udp_port']):
            return False, f"Stream port must be between {MIN_PORT} and {MAX_PORT}"

    if 'bitrate_kbps' in output:
        try:
            bitrate = int(output['bitrate_kbps'])
            if not (MIN_BITRATE <= bitrate <= MAX_BITRATE):
                return False, f"Bitrate must be between {MIN_BITRATE} and {MAX_BITRATE} kbps"
        except (ValueError, TypeError):
            return False, "Bitrate must be a number"

    return True, None

CONFIG_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Terminal Guidance</title>
    <style>
        :root {
            --bg: #0a0e14;
            --card: #12171f;
            --card-hover: #1a2130;
            --border: #252d3a;
            --text: #e1e4e8;
            --text-dim: #6e7a8a;
            --accent: #4d9fff;
            --success: #2dd4a0;
            --warning: #f0b429;
            --danger: #ff5c5c;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
            padding: 20px;
            max-width: 840px;
            margin: 0 auto;
        }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
        }
        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .logo svg { opacity: 0.9; }
        h1 { font-size: 18px; font-weight: 600; letter-spacing: -0.3px; }
        .status-badge {
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .status-badge::before {
            content: '';
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: currentColor;
        }
        .status-badge.ok { background: rgba(45, 212, 160, 0.15); color: var(--success); }
        .status-badge.warn { background: rgba(240, 180, 41, 0.15); color: var(--warning); }
        .status-badge.error { background: rgba(255, 92, 92, 0.15); color: var(--danger); }

        .status-bar {
            display: flex;
            gap: 16px;
            align-items: center;
            margin-bottom: 24px;
            padding: 16px 20px;
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 10px;
        }
        .status-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
        }
        .status-item .label { color: var(--text-dim); }
        .status-item .value { font-weight: 600; font-variant-numeric: tabular-nums; }
        .status-item .value.ok { color: var(--success); }
        .status-item .value.warn { color: var(--warning); }
        .status-item .value.error { color: var(--danger); }
        .status-sep { width: 1px; height: 20px; background: var(--border); }

        .control-bar {
            display: flex;
            gap: 10px;
            margin-bottom: 24px;
            flex-wrap: wrap;
        }

        .tabs {
            display: flex;
            gap: 4px;
            margin-bottom: 16px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0;
        }
        .tab {
            padding: 10px 18px;
            font-size: 13px;
            font-weight: 500;
            color: var(--text-dim);
            background: none;
            border: none;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            margin-bottom: -1px;
            transition: color 0.15s;
        }
        .tab:hover { color: var(--text); }
        .tab.active {
            color: var(--accent);
            border-bottom-color: var(--accent);
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 10px;
            margin-bottom: 16px;
        }
        .card-header {
            padding: 14px 18px;
            border-bottom: 1px solid var(--border);
            font-weight: 600;
            font-size: 13px;
            color: var(--text);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .card-body { padding: 18px; }

        .form-row {
            display: flex;
            align-items: center;
            margin-bottom: 14px;
        }
        .form-row:last-child { margin-bottom: 0; }
        .form-label {
            flex: 1;
            font-size: 13px;
            color: var(--text-dim);
        }
        .form-input {
            width: 140px;
            padding: 9px 12px;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text);
            font-size: 13px;
            transition: border-color 0.15s, box-shadow 0.15s;
        }
        .form-input:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(77, 159, 255, 0.12);
        }
        .form-input.wide { width: 180px; }

        .btn-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 20px; }
        .btn {
            padding: 11px 18px;
            border: none;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
        }
        .btn:hover { filter: brightness(1.1); transform: translateY(-1px); }
        .btn:active { transform: translateY(0) scale(0.98); }
        .btn-primary { background: var(--accent); color: #fff; }
        .btn-success { background: var(--success); color: #0a0e14; }
        .btn-danger { background: var(--danger); color: #fff; }
        .btn-outline {
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text);
        }
        .btn-outline:hover { background: var(--card-hover); border-color: var(--text-dim); }

        .toast {
            position: fixed;
            bottom: 24px;
            right: 24px;
            padding: 14px 22px;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 500;
            opacity: 0;
            transform: translateY(20px);
            transition: all 0.3s;
            z-index: 1000;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .toast.show { opacity: 1; transform: translateY(0); }
        .toast.success { background: var(--success); color: #0a0e14; }
        .toast.error { background: var(--danger); color: #fff; }

        .section-title {
            font-size: 10px;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 14px;
            font-weight: 600;
        }
        .divider { height: 1px; background: var(--border); margin: 18px 0; }

        .emergency-btn {
            width: 100%;
            padding: 18px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 2px;
            border-radius: 10px;
            box-shadow: 0 0 0 0 rgba(255, 92, 92, 0.4);
            animation: pulse-danger 2s infinite;
        }
        @keyframes pulse-danger {
            0%, 100% { box-shadow: 0 0 0 0 rgba(255, 92, 92, 0.4); }
            50% { box-shadow: 0 0 0 6px rgba(255, 92, 92, 0); }
        }
        .emergency-btn:hover { animation: none; }

        .collapsible {
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .collapsible::after {
            content: '';
            border: solid var(--text-dim);
            border-width: 0 2px 2px 0;
            padding: 3px;
            transform: rotate(45deg);
            transition: transform 0.2s;
        }
        .collapsible.collapsed::after { transform: rotate(-45deg); }
        .collapse-content {
            max-height: 400px;
            overflow-y: auto;
            transition: max-height 0.3s ease-out;
        }
        .collapse-content.collapsed { max-height: 0; overflow: hidden; }

        .class-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 8px;
        }
        .class-checkbox {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 10px;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.15s;
        }
        .class-checkbox:hover { border-color: var(--accent); }
        .class-checkbox input { accent-color: var(--accent); }
        .class-checkbox.selected { border-color: var(--accent); background: rgba(77, 159, 255, 0.1); }
        .class-checkbox span { font-size: 12px; }

        .service-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
            gap: 10px;
        }
        .service-btn {
            padding: 12px 16px;
            font-size: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
            justify-content: center;
        }
        .service-btn svg { width: 14px; height: 14px; }
        .service-status {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            color: var(--text-dim);
            margin-top: 6px;
        }
        .service-status .dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--success);
        }
        .service-status .dot.off { background: var(--text-dim); }

        @media (max-width: 500px) {
            .form-input { width: 110px; }
            .form-input.wide { width: 140px; }
            .btn { padding: 10px 14px; font-size: 12px; }
        }
    </style>
</head>
<body>
    <header>
        <div class="logo">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <circle cx="12" cy="12" r="3"/>
                <path d="M12 2v4M12 18v4M2 12h4M18 12h4"/>
                <path d="M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
            </svg>
            <h1>Terminal Guidance</h1>
        </div>
        <span class="status-badge" id="conn-badge">--</span>
    </header>

    <div class="status-bar">
        <div class="status-item">
            <span class="label">State:</span>
            <span class="value" id="st-state">--</span>
        </div>
        <div class="status-sep"></div>
        <div class="status-item">
            <span class="label">Targets:</span>
            <span class="value" id="st-targets">0</span>
        </div>
        <div class="status-sep"></div>
        <div class="status-item">
            <span class="label">FPS:</span>
            <span class="value" id="st-fps">--</span>
        </div>
    </div>

    <div class="control-bar">
        <button class="btn btn-success" id="btn-control" onclick="toggleControl()">Enable Control</button>
        <button class="btn btn-outline" onclick="lockTarget()">Lock Target</button>
        <button class="btn btn-outline" onclick="unlockTarget()">Unlock</button>
        <button class="btn btn-danger" id="btn-estop" onclick="emergencyStop()">E-Stop</button>
    </div>

    <div class="tabs">
        <button class="tab active" onclick="switchTab('tracking')">Tracking</button>
        <button class="tab" onclick="switchTab('connection')">Connection</button>
        <button class="tab" onclick="switchTab('pid')">PID Tuning</button>
        <button class="tab" onclick="switchTab('services')">Services</button>
    </div>

    <div id="tab-tracking" class="tab-content active">
        <div class="card">
            <div class="card-header">Detection Model</div>
            <div class="card-body">
                <div class="form-row">
                    <span class="form-label">Model</span>
                    <select class="form-input wide" id="det-model">
                        <option value="yolov8n">YOLOv8n (6.2 MB)</option>
                        <option value="yolo11n">YOLO11n (5.4 MB)</option>
                    </select>
                </div>
                <div class="form-row">
                    <span class="form-label">Resolution</span>
                    <select class="form-input wide" id="det-resolution">
                        <option value="640">640px (best range, ~90ms)</option>
                        <option value="416">416px (balanced, ~44ms)</option>
                        <option value="320">320px (fastest, ~35ms)</option>
                    </select>
                </div>
                <div class="form-row">
                    <span class="form-label">Confidence</span>
                    <input type="number" step="0.05" min="0.1" max="1.0" class="form-input" id="det-conf">
                </div>
                <div class="form-row">
                    <span class="form-label">Interval (frames)</span>
                    <input type="number" step="1" min="1" max="10" class="form-input" id="det-interval">
                </div>
                <div class="btn-row" style="margin-top: 14px;">
                    <button class="btn btn-primary" onclick="switchModel()">Apply Model Change</button>
                    <span id="model-status" style="font-size: 12px; color: var(--text-dim); margin-left: 10px;"></span>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header collapsible collapsed" onclick="toggleCollapse(this)">
                Detection Classes <span id="class-count" style="font-weight: normal; font-size: 11px; opacity: 0.7;">(0 selected)</span>
            </div>
            <div class="card-body collapse-content collapsed">
                <div class="section-title">Select classes to detect (empty = all)</div>
                <div class="class-grid" id="class-grid">
                    <!-- Populated by JS -->
                </div>
                <div class="btn-row" style="margin-top: 14px;">
                    <button class="btn btn-outline" onclick="selectAllClasses()">Select All</button>
                    <button class="btn btn-outline" onclick="clearAllClasses()">Clear All</button>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">Tracker</div>
            <div class="card-body">
                <div class="form-row">
                    <span class="form-label">Algorithm</span>
                    <select class="form-input wide" id="tracker-algorithm">
                        <option value="bytetrack">ByteTrack (Kalman)</option>
                        <option value="centroid">Centroid (Simple)</option>
                    </select>
                </div>
                <div class="form-row">
                    <span class="form-label">Lock after</span>
                    <input type="number" step="1" min="1" max="30" class="form-input" id="tracker-lock-frames"> frames
                </div>
                <div class="form-row">
                    <span class="form-label">Unlock after</span>
                    <input type="number" step="1" min="1" max="60" class="form-input" id="tracker-unlock-frames"> frames
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">Safety</div>
            <div class="card-body">
                <div class="form-row">
                    <span class="form-label">On target lost</span>
                    <select class="form-input wide" id="safety-lost-action">
                        <option value="loiter">Loiter</option>
                        <option value="hover">Hover</option>
                        <option value="rtl">Return to Launch</option>
                        <option value="land">Land</option>
                    </select>
                </div>
                <div class="form-row">
                    <span class="form-label">Search timeout</span>
                    <input type="number" step="1" class="form-input" id="safety-timeout"> sec
                </div>
                <div class="form-row">
                    <span class="form-label">Max distance</span>
                    <input type="number" step="10" class="form-input" id="safety-distance"> m
                </div>
            </div>
        </div>
    </div>

    <div id="tab-connection" class="tab-content">
        <div class="card">
            <div class="card-header">Camera</div>
            <div class="card-body">
                <div class="form-row">
                    <span class="form-label">Source URL</span>
                    <input type="text" class="form-input" style="width: 100%; max-width: 350px;" id="camera-url" placeholder="rtsp://user:pass@192.168.1.10/stream">
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">Flight Controller</div>
            <div class="card-body">
                <div class="form-row">
                    <span class="form-label">MAVLink</span>
                    <input type="text" class="form-input wide" id="fc-connection" placeholder="udp:192.168.1.1:14550">
                </div>
                <div class="form-row">
                    <span class="form-label">Send Commands</span>
                    <select class="form-input" id="mav-control">
                        <option value="false">Disabled</option>
                        <option value="true">Enabled</option>
                    </select>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">Video Stream</div>
            <div class="card-body">
                <div class="form-row">
                    <span class="form-label">Destination IP</span>
                    <input type="text" class="form-input wide" id="stream-host" placeholder="192.168.1.100">
                </div>
                <div class="form-row">
                    <span class="form-label">Port</span>
                    <input type="number" class="form-input" id="stream-port" placeholder="5600">
                </div>
                <div class="form-row">
                    <span class="form-label">Bitrate</span>
                    <input type="number" class="form-input" id="stream-bitrate" placeholder="2000"> kbps
                </div>
            </div>
        </div>
    </div>

    <div id="tab-pid" class="tab-content">
        <div class="card">
            <div class="card-header">Yaw (Horizontal)</div>
            <div class="card-body">
                <div class="form-row">
                    <span class="form-label">Kp</span>
                    <input type="number" step="0.01" class="form-input" id="yaw-kp">
                </div>
                <div class="form-row">
                    <span class="form-label">Ki</span>
                    <input type="number" step="0.001" class="form-input" id="yaw-ki">
                </div>
                <div class="form-row">
                    <span class="form-label">Kd</span>
                    <input type="number" step="0.01" class="form-input" id="yaw-kd">
                </div>
                <div class="form-row">
                    <span class="form-label">Max Rate</span>
                    <input type="number" step="1" class="form-input" id="yaw-max"> deg/s
                </div>
                <div class="form-row">
                    <span class="form-label">D Filter</span>
                    <input type="number" step="0.05" min="0" max="1" class="form-input" id="yaw-dfilter">
                </div>
                <div class="form-row">
                    <span class="form-label">Slew Rate</span>
                    <input type="number" step="5" min="0" class="form-input" id="yaw-slew">
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">Pitch (Vertical)</div>
            <div class="card-body">
                <div class="form-row">
                    <span class="form-label">Kp</span>
                    <input type="number" step="0.01" class="form-input" id="pitch-kp">
                </div>
                <div class="form-row">
                    <span class="form-label">Ki</span>
                    <input type="number" step="0.001" class="form-input" id="pitch-ki">
                </div>
                <div class="form-row">
                    <span class="form-label">Kd</span>
                    <input type="number" step="0.01" class="form-input" id="pitch-kd">
                </div>
                <div class="form-row">
                    <span class="form-label">Max Rate</span>
                    <input type="number" step="1" class="form-input" id="pitch-max"> deg/s
                </div>
                <div class="form-row">
                    <span class="form-label">D Filter</span>
                    <input type="number" step="0.05" min="0" max="1" class="form-input" id="pitch-dfilter">
                </div>
                <div class="form-row">
                    <span class="form-label">Slew Rate</span>
                    <input type="number" step="5" min="0" class="form-input" id="pitch-slew">
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">General</div>
            <div class="card-body">
                <div class="form-row">
                    <span class="form-label">Dead Zone</span>
                    <input type="number" step="0.5" class="form-input" id="pid-deadzone"> %
                </div>
            </div>
        </div>
    </div>

    <div id="tab-services" class="tab-content">
        <div class="card">
            <div class="card-header">Component Control</div>
            <div class="card-body">
                <div class="section-title">Restart individual components after config changes</div>
                <div class="service-grid">
                    <div>
                        <button class="btn btn-outline service-btn" onclick="restartService('detector')">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M1 4v6h6M23 20v-6h-6"/>
                                <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15"/>
                            </svg>
                            Detector
                        </button>
                        <div class="service-status">
                            <span class="dot" id="svc-detector-dot"></span>
                            <span id="svc-detector-status">--</span>
                        </div>
                    </div>
                    <div>
                        <button class="btn btn-outline service-btn" onclick="restartService('camera')">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M1 4v6h6M23 20v-6h-6"/>
                                <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15"/>
                            </svg>
                            Camera
                        </button>
                        <div class="service-status">
                            <span class="dot" id="svc-camera-dot"></span>
                            <span id="svc-camera-status">--</span>
                        </div>
                    </div>
                    <div>
                        <button class="btn btn-outline service-btn" onclick="restartService('streamer')">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M1 4v6h6M23 20v-6h-6"/>
                                <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15"/>
                            </svg>
                            Streamer
                        </button>
                        <div class="service-status">
                            <span class="dot" id="svc-streamer-dot"></span>
                            <span id="svc-streamer-status">--</span>
                        </div>
                    </div>
                    <div>
                        <button class="btn btn-outline service-btn" onclick="restartService('mavlink')">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M1 4v6h6M23 20v-6h-6"/>
                                <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15"/>
                            </svg>
                            MAVLink
                        </button>
                        <div class="service-status">
                            <span class="dot" id="svc-mavlink-dot"></span>
                            <span id="svc-mavlink-status">--</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">Full System</div>
            <div class="card-body">
                <div class="section-title">Restart all components</div>
                <div class="btn-row">
                    <button class="btn btn-primary" onclick="restartService('all')">Restart All Services</button>
                </div>
            </div>
        </div>
    </div>

    <div class="btn-row">
        <button class="btn btn-primary" onclick="saveConfig()">Save Config</button>
    </div>

    <div class="toast" id="toast"></div>

    <script>
        let controlEnabled = false;
        let config = {};
        let selectedClasses = [];

        // COCO 80 classes (YOLO11n, YOLOv8, YOLOv5 all use these)
        const COCO_CLASSES = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ];

        function switchTab(name) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelector(`.tab[onclick*="${name}"]`).classList.add('active');
            document.getElementById('tab-' + name).classList.add('active');
        }

        function toggleCollapse(header) {
            header.classList.toggle('collapsed');
            header.nextElementSibling.classList.toggle('collapsed');
        }

        function updateClassCount() {
            const count = selectedClasses.length;
            document.getElementById('class-count').textContent = count === 0 ? '(all classes)' : `(${count} selected)`;
        }

        function toast(msg, isError = false) {
            const t = document.getElementById('toast');
            t.textContent = msg;
            t.className = 'toast show ' + (isError ? 'error' : 'success');
            setTimeout(() => t.className = 'toast', 3000);
        }

        function setVal(id, val) {
            const el = document.getElementById(id);
            if (el) el.value = val ?? '';
        }

        function getVal(id, isNum = true) {
            const el = document.getElementById(id);
            if (!el) return null;
            return isNum ? parseFloat(el.value) : el.value;
        }

        async function loadStatus() {
            try {
                const r = await fetch('/api/status');
                const d = await r.json();

                const stateEl = document.getElementById('st-state');
                const state = d.tracking_state?.toUpperCase() || '--';
                stateEl.textContent = state;
                stateEl.className = 'value ' + (d.tracking_state === 'locked' ? 'ok' : d.tracking_state === 'lost' ? 'error' : '');

                document.getElementById('st-fps').textContent = (d.fps || 0).toFixed(1);
                document.getElementById('st-targets').textContent = (d.targets || []).length;

                const badge = document.getElementById('conn-badge');
                badge.textContent = d.connected ? 'Connected' : 'Disconnected';
                badge.className = 'status-badge ' + (d.connected ? 'ok' : 'error');

                controlEnabled = d.control_enabled;
                const btn = document.getElementById('btn-control');
                btn.textContent = controlEnabled ? 'Disable' : 'Enable Control';
                btn.className = 'btn ' + (controlEnabled ? 'btn-danger' : 'btn-success');
            } catch (e) { console.error(e); }
        }

        function buildClassGrid() {
            const grid = document.getElementById('class-grid');
            grid.innerHTML = '';
            COCO_CLASSES.forEach(cls => {
                const label = document.createElement('label');
                label.className = 'class-checkbox' + (selectedClasses.includes(cls) ? ' selected' : '');
                label.innerHTML = `<input type="checkbox" value="${cls}" ${selectedClasses.includes(cls) ? 'checked' : ''}><span>${cls}</span>`;
                label.querySelector('input').addEventListener('change', (e) => {
                    if (e.target.checked) {
                        if (!selectedClasses.includes(cls)) selectedClasses.push(cls);
                        label.classList.add('selected');
                    } else {
                        selectedClasses = selectedClasses.filter(c => c !== cls);
                        label.classList.remove('selected');
                    }
                    updateClassCount();
                });
                grid.appendChild(label);
            });
            updateClassCount();
        }

        function selectAllClasses() {
            selectedClasses = [...COCO_CLASSES];
            buildClassGrid();
        }

        function clearAllClasses() {
            selectedClasses = [];
            buildClassGrid();
        }

        async function loadConfig() {
            try {
                const r = await fetch('/api/config');
                config = await r.json();

                // Load target classes
                selectedClasses = config.detector?.target_classes || [];
                buildClassGrid();

                // Camera
                setVal('camera-url', config.camera?.rtsp_url);

                // Flight controller - full connection string
                setVal('fc-connection', config.mavlink?.connection);
                document.getElementById('mav-control').value = config.mavlink?.enable_control ? 'true' : 'false';

                setVal('stream-host', config.output?.stream?.udp_host);
                setVal('stream-port', config.output?.stream?.udp_port);
                setVal('stream-bitrate', config.output?.bitrate_kbps);

                setVal('yaw-kp', config.pid?.yaw?.kp);
                setVal('yaw-ki', config.pid?.yaw?.ki);
                setVal('yaw-kd', config.pid?.yaw?.kd);
                setVal('yaw-max', config.pid?.yaw?.max_rate);
                setVal('yaw-dfilter', config.pid?.yaw?.derivative_filter);
                setVal('yaw-slew', config.pid?.yaw?.slew_rate);

                setVal('pitch-kp', config.pid?.pitch?.kp);
                setVal('pitch-ki', config.pid?.pitch?.ki);
                setVal('pitch-kd', config.pid?.pitch?.kd);
                setVal('pitch-max', config.pid?.pitch?.max_rate);
                setVal('pitch-dfilter', config.pid?.pitch?.derivative_filter);
                setVal('pitch-slew', config.pid?.pitch?.slew_rate);

                setVal('pid-deadzone', config.pid?.dead_zone_percent);

                // Tracker settings
                document.getElementById('tracker-algorithm').value = config.tracker?.algorithm || 'bytetrack';
                setVal('tracker-lock-frames', config.tracker?.lock_on?.frames_to_lock);
                setVal('tracker-unlock-frames', config.tracker?.lock_on?.frames_to_unlock);

                // Detection model settings
                document.getElementById('det-model').value = config.detector?.model || 'yolov8n';
                document.getElementById('det-resolution').value = config.detector?.resolution || '640';
                setVal('det-conf', config.detector?.confidence_threshold);
                setVal('det-interval', config.detector?.detection_interval);

                document.getElementById('safety-lost-action').value = config.safety?.target_lost_action || 'loiter';
                setVal('safety-timeout', config.safety?.search_timeout);
                setVal('safety-distance', config.safety?.geofence?.max_distance_m);
            } catch (e) {
                console.error(e);
                toast('Failed to load config', true);
            }
        }

        async function saveConfig() {
            const cfg = {
                camera: {
                    rtsp_url: getVal('camera-url', false),
                },
                mavlink: {
                    connection: getVal('fc-connection', false),
                    enable_control: document.getElementById('mav-control').value === 'true',
                },
                output: {
                    stream: {
                        udp_host: getVal('stream-host', false),
                        udp_port: getVal('stream-port'),
                    },
                    bitrate_kbps: getVal('stream-bitrate'),
                },
                pid: {
                    yaw: {
                        kp: getVal('yaw-kp'),
                        ki: getVal('yaw-ki'),
                        kd: getVal('yaw-kd'),
                        max_rate: getVal('yaw-max'),
                        derivative_filter: getVal('yaw-dfilter'),
                        slew_rate: getVal('yaw-slew'),
                    },
                    pitch: {
                        kp: getVal('pitch-kp'),
                        ki: getVal('pitch-ki'),
                        kd: getVal('pitch-kd'),
                        max_rate: getVal('pitch-max'),
                        derivative_filter: getVal('pitch-dfilter'),
                        slew_rate: getVal('pitch-slew'),
                    },
                    dead_zone_percent: getVal('pid-deadzone'),
                },
                tracker: {
                    algorithm: document.getElementById('tracker-algorithm').value,
                    lock_on: {
                        frames_to_lock: getVal('tracker-lock-frames'),
                        frames_to_unlock: getVal('tracker-unlock-frames'),
                    },
                },
                detector: {
                    model: document.getElementById('det-model').value,
                    resolution: document.getElementById('det-resolution').value,
                    confidence_threshold: getVal('det-conf'),
                    detection_interval: getVal('det-interval'),
                    target_classes: selectedClasses,
                },
                safety: {
                    target_lost_action: document.getElementById('safety-lost-action').value,
                    search_timeout: getVal('safety-timeout'),
                    geofence: { max_distance_m: getVal('safety-distance') },
                },
            };

            try {
                const r = await fetch('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(cfg)
                });
                const d = await r.json();
                toast(d.status === 'ok' ? 'Configuration saved' : (d.message || 'Failed'), d.status !== 'ok');
            } catch (e) {
                toast('Failed to save: ' + e, true);
            }
        }

        async function toggleControl() {
            const endpoint = controlEnabled ? '/api/tracking/disable' : '/api/tracking/enable';
            try {
                const r = await fetch(endpoint, { method: 'POST' });
                const d = await r.json();
                if (d.status !== 'ok') toast(d.message || 'Failed', true);
                loadStatus();
            } catch (e) { toast('Failed: ' + e, true); }
        }

        async function lockTarget() {
            try {
                const r = await fetch('/api/tracking/lock', { method: 'POST' });
                const d = await r.json();
                toast(d.status === 'ok' ? 'Target locked' : (d.message || 'No targets'), d.status !== 'ok');
            } catch (e) { toast('Failed: ' + e, true); }
        }

        async function unlockTarget() {
            try {
                await fetch('/api/tracking/unlock', { method: 'POST' });
                toast('Target unlocked');
            } catch (e) { toast('Failed: ' + e, true); }
        }

        async function emergencyStop() {
            if (!confirm('Activate EMERGENCY STOP?')) return;
            try {
                await fetch('/api/emergency-stop', { method: 'POST' });
                toast('Emergency stop activated');
                loadStatus();
            } catch (e) { toast('Failed: ' + e, true); }
        }

        async function restartService(service) {
            try {
                const r = await fetch(`/api/restart/${service}`, { method: 'POST' });
                const d = await r.json();
                toast(d.status === 'ok' ? `${service} restarted` : (d.message || 'Failed'), d.status !== 'ok');
                loadServiceStatus();
            } catch (e) { toast('Failed: ' + e, true); }
        }

        async function switchModel() {
            const model = document.getElementById('det-model').value;
            const resolution = document.getElementById('det-resolution').value;
            const statusEl = document.getElementById('model-status');

            statusEl.textContent = 'Switching model...';
            statusEl.style.color = 'var(--warning)';

            try {
                const r = await fetch('/api/detector/switch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model, resolution })
                });
                const d = await r.json();

                if (d.status === 'ok') {
                    statusEl.textContent = `Loaded: ${model} @ ${resolution}px`;
                    statusEl.style.color = 'var(--success)';
                    toast(`Switched to ${model} @ ${resolution}px`);
                } else {
                    statusEl.textContent = d.message || 'Failed';
                    statusEl.style.color = 'var(--danger)';
                    toast(d.message || 'Failed to switch model', true);
                }
            } catch (e) {
                statusEl.textContent = 'Error: ' + e;
                statusEl.style.color = 'var(--danger)';
                toast('Failed: ' + e, true);
            }
        }

        async function loadServiceStatus() {
            try {
                const r = await fetch('/api/services/status');
                const d = await r.json();
                ['detector', 'camera', 'streamer', 'mavlink'].forEach(svc => {
                    const dot = document.getElementById(`svc-${svc}-dot`);
                    const status = document.getElementById(`svc-${svc}-status`);
                    if (dot && status) {
                        const isRunning = d[svc]?.running ?? false;
                        dot.className = 'dot' + (isRunning ? '' : ' off');
                        status.textContent = isRunning ? 'Running' : 'Stopped';
                    }
                });
            } catch (e) { console.error(e); }
        }

        loadConfig();
        loadStatus();
        loadServiceStatus();
        setInterval(loadStatus, 2000);
        setInterval(loadServiceStatus, 5000);
    </script>
</body>
</html>
"""


def api_response(
    success: bool,
    data: Any = None,
    message: str = None,
    status: int = 200
) -> Tuple[Response, int]:
    """Create standardized API response."""
    response = {"status": "ok" if success else "error"}
    if data is not None:
        if isinstance(data, dict):
            response.update(data)
        else:
            response["data"] = data
    if message:
        response["message"] = message
    return jsonify(response), status


def create_app(guidance: "TerminalGuidance") -> Flask:
    """Create Flask application with API routes."""
    app = Flask(__name__)
    app.guidance = guidance

    @app.route("/")
    def index() -> str:
        """Serve the main configuration page."""
        return render_template_string(CONFIG_PAGE)

    @app.route("/api/status")
    def status() -> Response:
        """Get current system status."""
        return jsonify(guidance.get_status())

    @app.route("/api/config", methods=["GET"])
    def get_config() -> Response:
        """Get current configuration."""
        return jsonify(guidance.config)

    @app.route("/api/config", methods=["POST"])
    def update_config() -> Tuple[Response, int]:
        """Update configuration with provided values."""
        try:
            new_config = request.get_json()
        except Exception:
            return api_response(False, message="Invalid JSON", status=400)

        if not new_config:
            return api_response(False, message="No config provided", status=400)

        if not isinstance(new_config, dict):
            return api_response(False, message="Config must be a JSON object", status=400)

        # Validate and sanitize nested config (limit depth to prevent DoS)
        def validate_config(obj, depth=0):
            if depth > 5:
                raise ValueError("Config too deeply nested")
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if not isinstance(k, str):
                        raise ValueError("Config keys must be strings")
                    validate_config(v, depth + 1)
            elif isinstance(obj, list):
                if len(obj) > 100:
                    raise ValueError("Config arrays too large")
                for item in obj:
                    validate_config(item, depth + 1)
            elif not isinstance(obj, (str, int, float, bool, type(None))):
                raise ValueError(f"Invalid config value type: {type(obj)}")

        try:
            validate_config(new_config)
        except ValueError as e:
            return api_response(False, message=str(e), status=400)

        # Validate network parameters to prevent injection attacks
        valid, err = validate_network_config(new_config)
        if not valid:
            return api_response(False, message=err, status=400)

        if guidance.update_config(new_config):
            return api_response(True, message="Configuration updated")
        return api_response(False, message="Failed to update config", status=400)

    @app.route("/api/tracking/lock", methods=["POST"])
    def lock_target() -> Tuple[Response, int]:
        """Lock onto a target by ID or auto-lock best target."""
        data = request.get_json() or {}
        target_id = data.get("target_id")
        if target_id is not None:
            # Validate target_id is a valid integer
            try:
                target_id_int = int(target_id)
                if target_id_int < 0:
                    return api_response(False, message="Invalid target ID", status=400)
            except (ValueError, TypeError):
                return api_response(False, message="Invalid target ID format", status=400)

            if guidance.lock_target(target_id_int):
                return api_response(True, {"target_id": target_id_int})
            return api_response(False, message="Target not found", status=404)
        if guidance.auto_lock():
            return api_response(True, message="Auto-locked")
        return api_response(False, message="No targets available", status=404)

    @app.route("/api/tracking/unlock", methods=["POST"])
    def unlock_target() -> Tuple[Response, int]:
        """Release the current target lock."""
        guidance.unlock_target()
        return api_response(True)

    @app.route("/api/tracking/enable", methods=["POST"])
    def enable_tracking() -> Tuple[Response, int]:
        """Enable tracking control output to flight controller."""
        if guidance.enable_control():
            return api_response(True)
        return api_response(
            False,
            message="Failed - check MAVLink connection and config",
            status=400
        )

    @app.route("/api/tracking/disable", methods=["POST"])
    def disable_tracking() -> Tuple[Response, int]:
        """Disable tracking control output."""
        guidance.disable_control()
        return api_response(True)

    @app.route("/api/emergency-stop", methods=["POST"])
    def emergency_stop() -> Tuple[Response, int]:
        """Trigger emergency stop."""
        guidance.emergency_stop()
        logger.warning("Emergency stop via web UI")
        return api_response(True)

    @app.route("/api/emergency-stop/clear", methods=["POST"])
    def clear_emergency() -> Tuple[Response, int]:
        """Clear emergency stop state."""
        guidance.clear_emergency()
        return api_response(True)

    @app.route("/api/services/status")
    def services_status() -> Response:
        """Get status of all services."""
        return jsonify({
            "detector": {
                "running": guidance._detector is not None and guidance._detector.is_initialized,
            },
            "camera": {
                "running": guidance._camera is not None and guidance._camera.is_running,
            },
            "streamer": {
                "running": guidance._streamer is not None and guidance._streamer.is_running,
            },
            "mavlink": {
                "running": guidance._mavlink is not None and guidance._mavlink.is_connected,
            },
        })

    @app.route("/api/detector/switch", methods=["POST"])
    def switch_detector() -> Tuple[Response, int]:
        """Switch the detection model and/or resolution."""
        try:
            data = request.get_json()
        except Exception:
            return api_response(False, message="Invalid JSON", status=400)

        if not data:
            return api_response(False, message="No data provided", status=400)

        model = data.get("model", "yolov8n")
        resolution = data.get("resolution", "640")

        # Validate model name
        valid_models = {"yolov8n", "yolo11n"}
        if model not in valid_models:
            return api_response(False, message=f"Invalid model: {model}", status=400)

        # Validate resolution
        valid_resolutions = {"640", "416", "320"}
        if resolution not in valid_resolutions:
            return api_response(False, message=f"Invalid resolution: {resolution}", status=400)

        try:
            # Update config
            guidance.config["detector"]["model"] = model
            guidance.config["detector"]["resolution"] = resolution

            # Restart detector with new model
            guidance.restart_detector()

            return api_response(True, {
                "model": model,
                "resolution": resolution,
            }, message=f"Switched to {model} @ {resolution}px")
        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            return api_response(False, message=str(e), status=500)

    @app.route("/api/restart/<service>", methods=["POST"])
    def restart_service(service: str) -> Tuple[Response, int]:
        """Restart a specific service or all services."""
        valid_services = {"detector", "camera", "streamer", "mavlink", "all"}
        if service not in valid_services:
            return api_response(False, message=f"Invalid service: {service}", status=400)

        try:
            if service == "all":
                guidance.restart_all()
            elif service == "detector":
                guidance.restart_detector()
            elif service == "camera":
                guidance.restart_camera()
            elif service == "streamer":
                guidance.restart_streamer()
            elif service == "mavlink":
                guidance.restart_mavlink()

            return api_response(True, message=f"{service} restarted")
        except Exception as e:
            logger.error(f"Failed to restart {service}: {e}")
            return api_response(False, message=str(e), status=500)

    @app.route("/api/health")
    def health_check() -> Response:
        """Health check endpoint with system metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Try to get CPU temperature (Pi-specific)
            cpu_temp = None
            try:
                temps = psutil.sensors_temperatures()
                if "cpu_thermal" in temps:
                    cpu_temp = temps["cpu_thermal"][0].current
                elif "coretemp" in temps:
                    cpu_temp = temps["coretemp"][0].current
            except Exception:
                pass

            # Component health from guidance status
            status = guidance.get_status()

            health = {
                "status": "healthy",
                "timestamp": time.time(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "cpu_temp_c": cpu_temp,
                    "memory_percent": memory.percent,
                    "memory_available_mb": memory.available // (1024 * 1024),
                    "disk_percent": disk.percent,
                },
                "components": {
                    "detector": status.get("detector", {}).get("initialized", False),
                    "mavlink": status.get("mavlink", {}).get("connected", False),
                    "tracking": status.get("tracking", {}).get("locked", False),
                },
                "uptime_seconds": time.time() - getattr(app, "_start_time", time.time()),
            }

            # Determine overall health status
            if cpu_percent > 90 or memory.percent > 90:
                health["status"] = "degraded"
            if cpu_temp and cpu_temp > 80:
                health["status"] = "degraded"

            return jsonify(health)
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return jsonify({"status": "unhealthy", "error": str(e)}), 500

    # Store app start time for uptime calculation
    app._start_time = time.time()

    return app


def run_web_server(app: Flask, host: str = "0.0.0.0", port: int = 5000) -> None:
    """Run the Flask development server."""
    app.run(host=host, port=port, threaded=True, use_reloader=False)
