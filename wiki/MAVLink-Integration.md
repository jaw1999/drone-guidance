# MAVLink Integration

## Overview

MAVLink communication is handled by `MAVLinkController` in `src/core/mavlink_controller.py`. It uses pymavlink to communicate with ArduPilot flight controllers.

## Connection

### Connection String Format

```
protocol:host:port
```

Examples:
- `udp:192.168.1.1:14550` - UDP to flight controller
- `tcp:192.168.1.1:5760` - TCP connection
- `/dev/ttyUSB0` - Serial connection (baud rate set separately)

### Initialization

```python
from pymavlink import mavutil

connection = mavutil.mavlink_connection(
    "udp:192.168.1.1:14550",
    source_system=255,      # Companion computer system ID
    source_component=190,   # Component ID
)

# Wait for heartbeat to establish connection
msg = connection.wait_heartbeat(timeout=10)
```

### System/Component IDs

| ID | Value | Description |
|----|-------|-------------|
| source_system | 255 | Companion computer (default) |
| source_component | 190 | Custom component ID |
| target_system | Auto-detected from heartbeat |
| target_component | Auto-detected from heartbeat |

## Messages Received

### HEARTBEAT

Monitors connection health and armed state.

```python
armed = (msg.base_mode & MAV_MODE_FLAG_SAFETY_ARMED) != 0
```

Connection is considered lost if no heartbeat received for 5 seconds.

### GLOBAL_POSITION_INT

GPS position and altitude.

| Field | Type | Description |
|-------|------|-------------|
| lat | int32 | Latitude (degE7) |
| lon | int32 | Longitude (degE7) |
| alt | int32 | Altitude MSL (mm) |
| relative_alt | int32 | Altitude AGL (mm) |
| hdg | uint16 | Heading (cdeg) |

Conversion:
```python
latitude = msg.lat / 1e7
longitude = msg.lon / 1e7
altitude_msl = msg.alt / 1000.0
altitude_rel = msg.relative_alt / 1000.0
heading = msg.hdg / 100.0
```

### VFR_HUD

Flight instruments.

| Field | Type | Description |
|-------|------|-------------|
| groundspeed | float | Ground speed (m/s) |
| airspeed | float | Airspeed (m/s) |
| heading | int16 | Heading (deg) |

### SYS_STATUS

System status including battery.

| Field | Type | Description |
|-------|------|-------------|
| voltage_battery | uint16 | Voltage (mV) |
| battery_remaining | int8 | Remaining capacity (%) |

### GPS_RAW_INT

Raw GPS data.

| Field | Type | Description |
|-------|------|-------------|
| fix_type | uint8 | 0=no fix, 3=3D fix |
| satellites_visible | uint8 | Number of satellites |

### HOME_POSITION

Home location for distance calculations.

| Field | Type | Description |
|-------|------|-------------|
| latitude | int32 | Latitude (degE7) |
| longitude | int32 | Longitude (degE7) |
| altitude | int32 | Altitude (mm) |

### COMMAND_LONG

Receives custom tracking commands from QGC.

## Messages Sent

### Heartbeat

Sent at `heartbeat_rate` Hz (default 1 Hz):

```python
connection.mav.heartbeat_send(
    mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
    0, 0, 0,
)
```

### SET_POSITION_TARGET_GLOBAL_INT

Sends heading and altitude targets in GUIDED mode:

```python
connection.mav.set_position_target_global_int_send(
    0,                                              # time_boot_ms
    target_system,
    target_component,
    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
    0b0000111111111000,                            # type_mask (use yaw + alt only)
    0, 0,                                          # lat_int, lon_int (ignored)
    altitude,                                      # alt (m, relative)
    0, 0, 0,                                       # vx, vy, vz (ignored)
    0, 0, 0,                                       # afx, afy, afz (ignored)
    math.radians(heading),                         # yaw (rad)
    0,                                             # yaw_rate (ignored)
)
```

### SET_MODE

Changes flight mode:

```python
mode_id = connection.mode_mapping().get(mode_name)  # e.g., "GUIDED", "LOITER"
connection.mav.set_mode_send(
    target_system,
    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
    mode_id,
)
```

### DO_CHANGE_SPEED

Sets throttle percentage:

```python
connection.mav.command_long_send(
    target_system,
    target_component,
    mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
    0,              # confirmation
    0,              # speed type (0 = airspeed)
    -1,             # speed (no change)
    throttle_pct,   # throttle percentage
    0, 0, 0, 0,
)
```

### COMMAND_ACK

Acknowledges received commands:

```python
connection.mav.command_ack_send(
    command_id,
    mavutil.mavlink.MAV_RESULT_ACCEPTED,  # or MAV_RESULT_FAILED
)
```

## Custom Tracking Commands

Commands are sent from QGC via COMMAND_LONG using custom command IDs:

| Command | ID | Description |
|---------|----|-------------|
| AUTO_LOCK | 31010 | Lock onto highest-confidence target |
| LOCK_TARGET | 31011 | Lock specific target (param1 = target_id) |
| UNLOCK | 31012 | Release target lock |
| ENABLE_CONTROL | 31013 | Enable PID control output |
| DISABLE_CONTROL | 31014 | Disable PID control output |

These use `MAV_CMD_USER_1` through `MAV_CMD_USER_5` (31010-31014).

## Control Flow

### Tracking Enable

```python
def enable_tracking(self) -> bool:
    # Precondition checks
    if not connected: return False
    if not enable_control in config: return False
    if emergency_stop: return False
    if require_arm_confirmation and not armed: return False

    # Enable
    _tracking_enabled = True
    _set_mode("GUIDED")
    set_throttle(tracking_throttle)  # e.g., 70%
    return True
```

### Tracking Disable

```python
def disable_tracking(self):
    _tracking_enabled = False
    set_throttle(cruise_throttle)  # e.g., 50%
    _set_mode("LOITER")
```

### Rate Commands

```python
def send_rate_commands(self, yaw_rate, pitch_rate, throttle_rate) -> bool:
    if not tracking_enabled: return False
    if not check_safety(): return False

    # Clamp rates
    yaw_rate = clamp(yaw_rate, -max_turn_rate, max_turn_rate)
    pitch_rate = clamp(pitch_rate, -20, 20)

    # Calculate new heading
    heading_delta = yaw_rate * 0.05  # 50ms update rate
    new_heading = (current_heading + heading_delta) % 360

    # Calculate new altitude
    alt_delta = -pitch_rate * 0.1  # Scale to meters
    new_alt = clamp(current_alt + alt_delta, min_altitude, max_altitude)

    # Send command
    send_position_target(new_heading, new_alt)
    return True
```

## Safety System

### Checks Performed

Each control command triggers safety checks:

1. **Battery**: If `battery_percent < min_battery_percent`, disable tracking and RTL
2. **Distance**: If `home_distance > max_distance_m`, disable tracking and RTL
3. **Max Altitude**: If `altitude_rel > max_altitude_m`, disable tracking and LOITER
4. **Min Altitude**: If `altitude_rel < min_altitude_m`, disable tracking and LOITER

### Safety Actions

| Action | Behavior |
|--------|----------|
| HOVER | Zero velocity (copter only) |
| LOITER | Switch to LOITER mode |
| RTL | Switch to RTL mode |
| LAND | Switch to LAND mode |
| CONTINUE_LAST | No action |

### Emergency Stop

```python
def emergency_stop(self):
    _emergency_stop = True
    _tracking_enabled = False
    execute_safety_action(LOITER)
```

Cleared with `clear_emergency_stop()`.

## Distance Calculation

Haversine formula for GPS distance:

```python
def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    R = 6371000  # Earth radius in meters

    phi1 = radians(lat1)
    phi2 = radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)

    a = sin(delta_phi/2)**2 + cos(phi1) * cos(phi2) * sin(delta_lambda/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c
```

## Configuration

```yaml
mavlink:
  connection: "udp:192.168.1.1:14550"
  source_system: 255
  source_component: 190
  heartbeat_rate: 1.0
  command_timeout: 5.0
  enable_control: false       # Must be true for actual control
  vehicle_type: "plane"       # "plane", "copter", or "auto"

  intercept:
    cruise_throttle: 50       # Normal throttle %
    tracking_throttle: 70     # Tracking throttle %
    max_turn_rate: 45.0       # deg/sec
    max_climb_rate: 5.0       # m/s

safety:
  target_lost_action: "loiter"
  search_timeout: 10.0

  geofence:
    enabled: true
    max_distance_m: 500
    max_altitude_m: 120
    min_altitude_m: 10

  min_battery_percent: 20
  require_arm_confirmation: true
  emergency_stop_enabled: true
```

## VehicleState Dataclass

```python
@dataclass
class VehicleState:
    armed: bool = False
    mode: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    altitude_msl: float = 0.0
    altitude_rel: float = 0.0
    heading: float = 0.0
    groundspeed: float = 0.0
    airspeed: float = 0.0
    battery_percent: float = 100.0
    battery_voltage: float = 0.0
    home_distance: float = 0.0
    home_latitude: float = 0.0
    home_longitude: float = 0.0
    home_altitude: float = 0.0
    gps_fix: int = 0
    satellites: int = 0
    last_update: float = 0.0
    vehicle_type: str = "unknown"
    throttle_percent: float = 0.0
```

Thread-safe access via property with lock:

```python
@property
def vehicle_state(self) -> VehicleState:
    with self._state_lock:
        return VehicleState(**self._vehicle_state.__dict__)
```

## Thread Safety

- `VehicleState` access protected by `_state_lock`
- All message processing happens in receive thread
- Control commands called from main thread
- Heartbeat sending in separate thread
