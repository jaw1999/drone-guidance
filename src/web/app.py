"""Flask configuration UI for Terminal Guidance.

This module provides a lightweight web interface for configuring and
monitoring the terminal guidance system. The UI is designed for minimal
resource usage on Raspberry Pi 5.

Video streaming is handled separately via UDP to QGroundControl.

Functions:
    api_response: Create standardized API response.
    create_app: Create Flask application with all routes.
    run_web_server: Run the Flask development server.
"""

import logging
import time
from typing import TYPE_CHECKING, Any, Tuple

import psutil
from flask import Flask, Response, jsonify, render_template_string, request

if TYPE_CHECKING:
    from ..app import TerminalGuidance

logger = logging.getLogger(__name__)

CONFIG_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Terminal Guidance</title>
    <style>
        :root {
            --bg: #0d1117;
            --card: #161b22;
            --border: #30363d;
            --text: #c9d1d9;
            --text-dim: #8b949e;
            --accent: #58a6ff;
            --success: #3fb950;
            --warning: #d29922;
            --danger: #f85149;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
            padding: 16px;
            max-width: 800px;
            margin: 0 auto;
        }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
        }
        h1 { font-size: 20px; font-weight: 600; }
        .status-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
        }
        .status-badge.ok { background: rgba(63, 185, 80, 0.2); color: var(--success); }
        .status-badge.warn { background: rgba(210, 153, 34, 0.2); color: var(--warning); }
        .status-badge.error { background: rgba(248, 81, 73, 0.2); color: var(--danger); }

        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }
        .stat-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
        }
        .stat-label { font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }
        .stat-value { font-size: 24px; font-weight: 600; margin-top: 4px; }
        .stat-value.ok { color: var(--success); }
        .stat-value.warn { color: var(--warning); }
        .stat-value.error { color: var(--danger); }

        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 16px;
        }
        .card-header {
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            font-weight: 600;
            font-size: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .card-body { padding: 16px; }

        .form-row {
            display: flex;
            align-items: center;
            margin-bottom: 12px;
        }
        .form-row:last-child { margin-bottom: 0; }
        .form-label {
            flex: 1;
            font-size: 13px;
            color: var(--text-dim);
        }
        .form-input {
            width: 140px;
            padding: 8px 12px;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text);
            font-size: 14px;
        }
        .form-input:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.15);
        }
        .form-input.wide { width: 200px; }

        .btn-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 16px; }
        .btn {
            padding: 10px 16px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s;
        }
        .btn:hover { filter: brightness(1.1); }
        .btn:active { transform: scale(0.98); }
        .btn-primary { background: var(--accent); color: #fff; }
        .btn-success { background: var(--success); color: #fff; }
        .btn-danger { background: var(--danger); color: #fff; }
        .btn-outline {
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text);
        }
        .btn-outline:hover { background: var(--border); }

        .toast {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 12px 20px;
            border-radius: 8px;
            font-size: 14px;
            opacity: 0;
            transform: translateY(20px);
            transition: all 0.3s;
            z-index: 1000;
        }
        .toast.show { opacity: 1; transform: translateY(0); }
        .toast.success { background: var(--success); color: #fff; }
        .toast.error { background: var(--danger); color: #fff; }

        .section-title {
            font-size: 11px;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }
        .divider { height: 1px; background: var(--border); margin: 16px 0; }

        .emergency-btn {
            width: 100%;
            padding: 16px;
            font-size: 16px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        @media (max-width: 500px) {
            .form-input { width: 100px; }
            .form-input.wide { width: 140px; }
        }
    </style>
</head>
<body>
    <header>
        <h1>Terminal Guidance</h1>
        <span class="status-badge" id="conn-badge">--</span>
    </header>

    <div class="grid">
        <div class="stat-card">
            <div class="stat-label">State</div>
            <div class="stat-value" id="st-state">--</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">FPS</div>
            <div class="stat-value" id="st-fps">--</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Targets</div>
            <div class="stat-value" id="st-targets">0</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Battery</div>
            <div class="stat-value" id="st-battery">--</div>
        </div>
    </div>

    <div class="card">
        <div class="card-header">
            <span>Flight Controller (UDPCI)</span>
        </div>
        <div class="card-body">
            <div class="form-row">
                <span class="form-label">FC IP Address</span>
                <input type="text" class="form-input wide" id="fc-ip" placeholder="192.168.1.1">
            </div>
            <div class="form-row">
                <span class="form-label">FC Port</span>
                <input type="number" class="form-input" id="fc-port" placeholder="14550">
            </div>
            <div class="form-row">
                <span class="form-label">Enable Control</span>
                <select class="form-input" id="mav-control">
                    <option value="false">Disabled</option>
                    <option value="true">Enabled</option>
                </select>
            </div>
        </div>
    </div>

    <div class="card">
        <div class="card-header">
            <span>Video Stream</span>
        </div>
        <div class="card-body">
            <div class="form-row">
                <span class="form-label">UDP Destination IP</span>
                <input type="text" class="form-input wide" id="stream-host" placeholder="192.168.1.100">
            </div>
            <div class="form-row">
                <span class="form-label">UDP Port</span>
                <input type="number" class="form-input" id="stream-port" placeholder="5600">
            </div>
            <div class="form-row">
                <span class="form-label">Bitrate (kbps)</span>
                <input type="number" class="form-input" id="stream-bitrate" placeholder="2000">
            </div>
        </div>
    </div>

    <div class="card">
        <div class="card-header">
            <span>PID Controller</span>
        </div>
        <div class="card-body">
            <div class="section-title">Yaw (Horizontal)</div>
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
                <span class="form-label">Max Rate (deg/s)</span>
                <input type="number" step="1" class="form-input" id="yaw-max">
            </div>

            <div class="divider"></div>

            <div class="section-title">Pitch (Vertical)</div>
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
                <span class="form-label">Max Rate (deg/s)</span>
                <input type="number" step="1" class="form-input" id="pitch-max">
            </div>

            <div class="divider"></div>

            <div class="section-title">General</div>
            <div class="form-row">
                <span class="form-label">Dead Zone (%)</span>
                <input type="number" step="0.5" class="form-input" id="pid-deadzone">
            </div>
        </div>
    </div>

    <div class="card">
        <div class="card-header">
            <span>Detection</span>
        </div>
        <div class="card-body">
            <div class="form-row">
                <span class="form-label">Confidence Threshold</span>
                <input type="number" step="0.05" min="0.1" max="1.0" class="form-input" id="det-conf">
            </div>
            <div class="form-row">
                <span class="form-label">Detection Interval (frames)</span>
                <input type="number" step="1" min="1" max="10" class="form-input" id="det-interval">
            </div>
            <div class="form-row">
                <span class="form-label">Input Size (px)</span>
                <input type="number" step="32" min="160" max="640" class="form-input" id="det-size">
            </div>
        </div>
    </div>

    <div class="card">
        <div class="card-header">
            <span>Safety</span>
        </div>
        <div class="card-body">
            <div class="form-row">
                <span class="form-label">Target Lost Action</span>
                <select class="form-input wide" id="safety-lost-action">
                    <option value="loiter">Loiter</option>
                    <option value="hover">Hover</option>
                    <option value="rtl">Return to Launch</option>
                    <option value="land">Land</option>
                </select>
            </div>
            <div class="form-row">
                <span class="form-label">Search Timeout (sec)</span>
                <input type="number" step="1" class="form-input" id="safety-timeout">
            </div>
            <div class="form-row">
                <span class="form-label">Min Battery (%)</span>
                <input type="number" step="1" class="form-input" id="safety-battery">
            </div>
            <div class="form-row">
                <span class="form-label">Max Distance (m)</span>
                <input type="number" step="10" class="form-input" id="safety-distance">
            </div>
        </div>
    </div>

    <div class="btn-row">
        <button class="btn btn-primary" onclick="saveConfig()">Save Configuration</button>
        <button class="btn btn-success" id="btn-control" onclick="toggleControl()">Enable Control</button>
        <button class="btn btn-outline" onclick="lockTarget()">Auto-Lock Target</button>
        <button class="btn btn-outline" onclick="unlockTarget()">Unlock</button>
    </div>

    <div style="margin-top: 24px;">
        <button class="btn btn-danger emergency-btn" onclick="emergencyStop()">EMERGENCY STOP</button>
    </div>

    <div class="toast" id="toast"></div>

    <script>
        let controlEnabled = false;
        let config = {};

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

                document.getElementById('st-state').textContent = d.tracking_state?.toUpperCase() || '--';
                document.getElementById('st-state').className = 'stat-value ' +
                    (d.tracking_state === 'locked' ? 'ok' : d.tracking_state === 'lost' ? 'error' : '');

                document.getElementById('st-fps').textContent = (d.fps || 0).toFixed(1);
                document.getElementById('st-targets').textContent = (d.targets || []).length;

                const bat = d.battery || 0;
                document.getElementById('st-battery').textContent = bat > 0 ? bat.toFixed(0) + '%' : '--';
                document.getElementById('st-battery').className = 'stat-value ' +
                    (bat > 30 ? 'ok' : bat > 15 ? 'warn' : bat > 0 ? 'error' : '');

                const badge = document.getElementById('conn-badge');
                badge.textContent = d.connected ? 'Connected' : 'Disconnected';
                badge.className = 'status-badge ' + (d.connected ? 'ok' : 'error');

                controlEnabled = d.control_enabled;
                const btn = document.getElementById('btn-control');
                btn.textContent = controlEnabled ? 'Disable Control' : 'Enable Control';
                btn.className = 'btn ' + (controlEnabled ? 'btn-danger' : 'btn-success');
            } catch (e) { console.error(e); }
        }

        function parseConnection(conn) {
            // Parse "udp:IP:PORT" or "udpci:IP:PORT" format
            if (!conn) return { ip: '', port: 14550 };
            const match = conn.match(/^(?:udp(?:ci)?:)?([^:]+):(\\d+)$/i);
            if (match) return { ip: match[1], port: parseInt(match[2]) };
            return { ip: conn, port: 14550 };
        }

        async function loadConfig() {
            try {
                const r = await fetch('/api/config');
                config = await r.json();

                // Parse FC connection into IP and port
                const fc = parseConnection(config.mavlink?.connection);
                setVal('fc-ip', fc.ip);
                setVal('fc-port', fc.port);
                document.getElementById('mav-control').value = config.mavlink?.enable_control ? 'true' : 'false';

                setVal('stream-host', config.output?.stream?.udp_host);
                setVal('stream-port', config.output?.stream?.udp_port);
                setVal('stream-bitrate', config.output?.bitrate_kbps);

                setVal('yaw-kp', config.pid?.yaw?.kp);
                setVal('yaw-ki', config.pid?.yaw?.ki);
                setVal('yaw-kd', config.pid?.yaw?.kd);
                setVal('yaw-max', config.pid?.yaw?.max_rate);

                setVal('pitch-kp', config.pid?.pitch?.kp);
                setVal('pitch-ki', config.pid?.pitch?.ki);
                setVal('pitch-kd', config.pid?.pitch?.kd);
                setVal('pitch-max', config.pid?.pitch?.max_rate);

                setVal('pid-deadzone', config.pid?.dead_zone_percent);

                setVal('det-conf', config.detector?.confidence_threshold);
                setVal('det-interval', config.detector?.detection_interval);
                setVal('det-size', config.detector?.input_size);

                document.getElementById('safety-lost-action').value = config.safety?.target_lost_action || 'loiter';
                setVal('safety-timeout', config.safety?.search_timeout);
                setVal('safety-battery', config.safety?.min_battery_percent);
                setVal('safety-distance', config.safety?.geofence?.max_distance_m);
            } catch (e) {
                console.error(e);
                toast('Failed to load config', true);
            }
        }

        async function saveConfig() {
            // Build connection string from IP and port
            const fcIp = getVal('fc-ip', false) || '192.168.1.1';
            const fcPort = getVal('fc-port') || 14550;

            const cfg = {
                mavlink: {
                    connection: `udp:${fcIp}:${fcPort}`,
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
                    yaw: { kp: getVal('yaw-kp'), ki: getVal('yaw-ki'), kd: getVal('yaw-kd'), max_rate: getVal('yaw-max') },
                    pitch: { kp: getVal('pitch-kp'), ki: getVal('pitch-ki'), kd: getVal('pitch-kd'), max_rate: getVal('pitch-max') },
                    dead_zone_percent: getVal('pid-deadzone'),
                },
                detector: {
                    confidence_threshold: getVal('det-conf'),
                    detection_interval: getVal('det-interval'),
                    input_size: getVal('det-size'),
                },
                safety: {
                    target_lost_action: document.getElementById('safety-lost-action').value,
                    search_timeout: getVal('safety-timeout'),
                    min_battery_percent: getVal('safety-battery'),
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

        loadConfig();
        loadStatus();
        setInterval(loadStatus, 2000);
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
    """Create standardized API response.

    Args:
        success: Whether the operation succeeded.
        data: Optional response data (dict or other).
        message: Optional message string.
        status: HTTP status code.

    Returns:
        Tuple of (Response, status_code) for Flask.
    """
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
    """Create Flask application with API routes.

    Args:
        guidance: The TerminalGuidance instance to control.

    Returns:
        Configured Flask application.
    """
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

    @app.route("/api/health")
    def health_check() -> Response:
        """Health check endpoint for monitoring and load balancers.

        Returns system health metrics including:
        - CPU usage and temperature
        - Memory usage
        - Disk usage
        - Component status (detector, MAVLink, tracking)
        """
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
    """Run the Flask development server.

    Args:
        app: Flask application instance.
        host: Host address to bind to.
        port: Port number to listen on.
    """
    app.run(host=host, port=port, threaded=True, use_reloader=False)
