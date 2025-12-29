# QGroundControl Custom Build for Terminal Guidance

This is a custom QGroundControl build that adds a Terminal Guidance tracking control panel to the Fly View. The panel communicates with the Terminal Guidance server running on the companion computer via REST API.

## Features

- **Terminal Guidance Control Panel** - Toggle panel in Fly View for tracking control
- **Target Lock Controls** - Auto-lock, manual lock by ID, unlock
- **Flight Control** - Enable/disable tracking-based flight control
- **Emergency Stop** - Immediate stop button
- **Status Display** - Real-time tracking state, target info, connection status
- **Configurable Server URL** - Set the companion computer IP address

## Prerequisites

### macOS
```bash
brew install cmake ninja qt@6
```

### Ubuntu 22.04+
```bash
sudo apt update
sudo apt install -y \
    build-essential cmake ninja-build \
    qt6-base-dev qt6-declarative-dev qt6-positioning-dev \
    qt6-serialport-dev qt6-svg-dev qt6-charts-dev \
    qt6-multimedia-dev qt6-shadertools-dev qt6-tools-dev \
    libqt6opengl6-dev qt6-l10n-tools \
    libsdl2-dev libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-good \
    libspeechd-dev flite1-dev libspeechd2
```

### Raspberry Pi (for building on Pi itself)
```bash
# Same as Ubuntu, but will be slower to compile
# Consider cross-compiling from a faster machine
```

## Building

### 1. Clone QGroundControl (if not already done)
```bash
cd qgc
git clone --recursive https://github.com/mavlink/qgroundcontrol.git
```

### 2. Copy Custom Build Files
```bash
# Copy our custom directory into the QGC source
cp -r custom qgroundcontrol/custom
```

### 3. Build QGC with Custom Plugin
```bash
cd qgroundcontrol
mkdir build && cd build

# Configure with custom build enabled
cmake .. -G Ninja -DQGC_CUSTOM_BUILD=ON

# Build
ninja
```

### 4. Run
```bash
# macOS
open ./QGroundControl.app

# Linux
./QGroundControl
```

## Using the Terminal Guidance Panel

1. **Start Terminal Guidance** on your companion computer:
   ```bash
   cd /path/to/terminal_guidance
   python -m src.app -c config/default.yaml
   ```

2. **Launch QGC** custom build

3. **Configure Server URL**:
   - Click the "Terminal Guidance" button in Fly View
   - Enter the companion computer IP in the Server field (e.g., `http://192.168.1.100:5000`)

4. **Control Tracking**:
   - **Auto** - Lock onto the highest-confidence detected target
   - **Unlock** - Release current target lock
   - **Enable** - Enable flight control (drone follows target)
   - **Disable** - Disable flight control (manual control)
   - **EMERGENCY STOP** - Immediately stop all tracking control

## Panel Status Indicators

| Indicator | Meaning |
|-----------|---------|
| Green button | Connected to Terminal Guidance server |
| Red/Gray button | Not connected |
| State: Locked | Target is locked and being tracked |
| State: Tracking | Target visible but not locked |
| State: Lost | Target was locked but is now lost |
| State: Searching | Looking for targets |

## API Endpoints Used

The panel communicates with these Terminal Guidance API endpoints:

| Action | Endpoint | Method |
|--------|----------|--------|
| Get Status | `/api/status` | GET |
| Auto Lock | `/api/tracking/lock` | POST |
| Lock ID | `/api/tracking/lock` | POST `{"target_id": N}` |
| Unlock | `/api/tracking/unlock` | POST |
| Enable Control | `/api/tracking/enable` | POST |
| Disable Control | `/api/tracking/disable` | POST |
| Emergency Stop | `/api/emergency-stop` | POST |

## File Structure

```
custom/
├── CMakeLists.txt              # Build configuration
├── custom.qrc                  # Qt resource file
├── src/
│   ├── CustomPlugin.cc/h       # Plugin entry point
│   ├── FlyViewCustomLayer.qml  # Custom fly view overlay
│   ├── TerminalGuidancePanel.qml # Tracking control panel
│   ├── AutoPilotPlugin/        # Custom autopilot plugin
│   └── FirmwarePlugin/         # Custom firmware plugin
└── res/
    ├── Custom/Widgets/         # Custom QML widgets
    └── Images/                 # Custom icons
```

## Customization

### Changing Default Server URL
Edit `src/TerminalGuidancePanel.qml`:
```qml
property string serverUrl: "http://192.168.1.100:5000"
```

### Branding
- Replace images in `res/Images/`
- Modify colors in `src/CustomPlugin.cc` `paletteOverride()`

## Troubleshooting

### "Connection failed"
- Verify Terminal Guidance server is running
- Check IP address and port
- Ensure firewall allows connections on port 5000

### Build errors
- Ensure Qt6 is properly installed
- Clean build directory and rebuild:
  ```bash
  rm -rf build && mkdir build && cd build
  cmake .. -G Ninja -DQGC_CUSTOM_BUILD=ON && ninja
  ```

### Panel not showing
- Ensure custom build was used (`-DQGC_CUSTOM_BUILD=ON`)
- Check QGC console for QML errors
