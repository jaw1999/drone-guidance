# PID Control

## Overview

PID control is implemented in `src/core/pid.py`. The system provides three-axis control (yaw, pitch, throttle) with anti-windup, derivative filtering, and slew rate limiting.

## PID Algorithm

### Standard Form

```
u(t) = Kp * e(t) + Ki * ∫e(τ)dτ + Kd * de(t)/dt
```

Where:
- `u(t)` = control output
- `e(t)` = error (target - current)
- `Kp` = proportional gain
- `Ki` = integral gain
- `Kd` = derivative gain

### Discrete Implementation

```python
# Proportional
p_term = kp * error

# Integral (with anti-windup)
integral += error * dt
integral = clamp(integral, -integral_limit, integral_limit)
i_term = ki * integral

# Derivative (filtered)
raw_derivative = (error - prev_error) / dt
filtered_derivative = alpha * raw_derivative + (1 - alpha) * prev_filtered
d_term = kd * filtered_derivative

output = p_term + i_term + d_term
output = clamp(output, -max_output, max_output)
```

## Anti-Windup

Integral windup occurs when the output saturates but the integral term continues growing. This causes overshoot when the error changes sign.

### Implementation

The integral term is clamped to prevent runaway:

```python
if ki >= 0.001:
    integral_limit = min(max_output / ki, max_output * 100)
else:
    integral_limit = max_output * 100
```

This ensures the integral term alone cannot exceed the maximum output.

## Derivative Filtering

Raw derivative is noisy. A first-order low-pass filter is applied:

```python
filtered = alpha * raw + (1 - alpha) * prev_filtered
```

Where `alpha` = `derivative_filter` config (code default 0.1, config file sets 0.2). Lower values = smoother but more lag.

## Slew Rate Limiting

Output change rate can be limited for smooth control:

```python
if slew_rate > 0:
    max_change = slew_rate * dt
    delta = output - prev_output
    if abs(delta) > max_change:
        output = prev_output + sign(delta) * max_change
```

This prevents jerky movements from sudden error changes.

## Dead Zone

Small errors near zero produce no output:

```python
if abs(error_x) < dead_zone:
    error_x = 0.0
if abs(error_y) < dead_zone:
    error_y = 0.0
```

Default `dead_zone_percent` is 5.0 in code, 10.0 in config file. Errors within this percentage of frame center produce no correction.

## Axis Configuration

Values below are from `config/default.yaml`. Code defaults differ (see Class Reference).

### Yaw (Horizontal Centering)

Controls heading to center target horizontally.

| Parameter | Config Value | Description |
|-----------|--------------|-------------|
| kp | 0.3 | Proportional gain |
| ki | 0.005 | Integral gain |
| kd | 0.15 | Derivative gain |
| max_rate | 15.0 | Maximum turn rate (deg/sec) |
| derivative_filter | 0.2 | Filter coefficient |
| slew_rate | 20.0 | Max rate change (deg/sec²) |

### Pitch (Vertical Centering)

Controls climb/dive to center target vertically.

| Parameter | Config Value | Description |
|-----------|--------------|-------------|
| kp | 0.25 | Proportional gain |
| ki | 0.005 | Integral gain |
| kd | 0.12 | Derivative gain |
| max_rate | 10.0 | Maximum pitch rate (deg/sec) |
| derivative_filter | 0.2 | Filter coefficient |
| slew_rate | 15.0 | Max rate change (deg/sec²) |

### Throttle (Distance Control)

Controls throttle to maintain target size. Currently disabled for fixed-wing (throttle set directly).

| Parameter | Config Value | Description |
|-----------|--------------|-------------|
| kp | 0.0 | Disabled |
| ki | 0.0 | Disabled |
| kd | 0.0 | Disabled |
| max_rate | 0.0 | Disabled |

## Input/Output Mapping

### Input: Tracking Error

From `TargetTracker.get_tracking_error()`:

```python
error_x = (target_cx - frame_cx) / (frame_width / 2)   # [-1, 1]
error_y = (target_cy - frame_cy) / (frame_height / 2)  # [-1, 1]
```

- Positive X error = target is right of center
- Positive Y error = target is below center

### Output: Rate Commands

```python
@dataclass
class ControlOutput:
    yaw_rate: float = 0.0       # deg/sec, positive = turn right
    pitch_rate: float = 0.0     # deg/sec, positive = pitch up
    throttle_rate: float = 0.0  # m/sec, positive = increase
    is_active: bool = False
```

### Axis Mapping

```python
# Yaw: positive X error -> positive yaw (turn right to center)
yaw_rate = yaw_pid.update(error_x)

# Pitch: positive Y error (below center) -> negative pitch (nose down)
pitch_rate = -pitch_pid.update(error_y)

# Throttle: based on target size (not tracking error)
size_error = 0.2 - target_size  # Maintain 20% frame coverage
throttle_rate = throttle_pid.update(size_error)
```

## Target Size for Throttle

Target size ratio is computed in the main loop:

```python
bbox = locked_target.bbox
area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
frame_area = frame_height * frame_width
target_size = area / frame_area  # 0.0 to 1.0
```

The throttle PID tries to maintain `target_size = 0.2` (20% of frame).

## State Management

### Enable/Disable

```python
controller.enable()   # Start outputting commands
controller.disable()  # Stop and reset state
```

Disabling resets all integral terms and filtered derivatives.

### Gain Updates

Gains can be updated at runtime without restart:

```python
controller.update_gains(
    axis="yaw",
    kp=0.5,
    ki=0.01,
    kd=0.2,
    max_rate=20.0,
    reset_integral=True  # Optional
)
```

## Configuration

```yaml
pid:
  yaw:
    kp: 0.3
    ki: 0.005
    kd: 0.15
    max_rate: 15.0
    derivative_filter: 0.2
    slew_rate: 20.0

  pitch:
    kp: 0.25
    ki: 0.005
    kd: 0.12
    max_rate: 10.0
    derivative_filter: 0.2
    slew_rate: 15.0

  throttle:
    kp: 0.0
    ki: 0.0
    kd: 0.0
    max_rate: 0.0
    derivative_filter: 0.0
    slew_rate: 0.0

  dead_zone_percent: 10.0
  update_rate: 20
```

## Tuning Guidelines

### Proportional (Kp)

- Start low, increase until system responds to errors
- Too high = oscillation
- Too low = slow response

### Integral (Ki)

- Eliminates steady-state error
- Start very low (0.001-0.01)
- Too high = overshoot, oscillation
- Too low = steady-state error persists

### Derivative (Kd)

- Dampens oscillation
- Start around 0.1-0.2 * Kp
- Too high = noise amplification
- Too low = underdamped response

### Typical Tuning Process

1. Set Ki = Kd = 0
2. Increase Kp until steady oscillation
3. Set Kp to ~60% of oscillation value
4. Increase Kd to dampen oscillation
5. Add small Ki to eliminate steady-state error

### Fixed-Wing Specific

For fixed-wing aircraft, aggressive corrections cause issues:
- Low Kp (0.2-0.4) for gentle corrections
- High slew_rate limiting to prevent jerky inputs
- Larger dead zone (10%+) to avoid constant corrections
- Lower max_rate to respect aircraft turn limits

## Class Reference

### PIDGains

Code defaults (overridden by config file):

```python
@dataclass
class PIDGains:
    kp: float = 0.5
    ki: float = 0.01
    kd: float = 0.1
    max_output: float = 30.0
    derivative_filter: float = 0.1
    slew_rate: float = 0.0
```

### PIDAxis

Single-axis PID controller.

```python
class PIDAxis:
    def __init__(self, gains: PIDGains)
    def update(self, error: float, dt: float = None) -> float
    def reset(self) -> None
```

### PIDController

Multi-axis controller with enable/disable.

```python
class PIDController:
    def __init__(self, config: PIDConfig)
    def enable(self) -> None
    def disable(self) -> None
    def reset(self) -> None
    def update(self, error: Tuple[float, float], target_size: float = None) -> ControlOutput
    def update_gains(self, axis: str, **kwargs) -> None
    def get_status(self) -> dict
```
