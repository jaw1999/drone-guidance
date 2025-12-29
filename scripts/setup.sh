#!/bin/bash
# Terminal Guidance - Full setup script
# Supports: Raspberry Pi (Debian/Ubuntu), macOS, Linux

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Terminal Guidance Setup ==="
echo "Project directory: $PROJECT_DIR"

# Detect OS and architecture
detect_platform() {
    OS="$(uname -s)"
    ARCH="$(uname -m)"

    case "$OS" in
        Linux)
            if [[ -f /etc/os-release ]]; then
                . /etc/os-release
                DISTRO="$ID"
            fi
            if [[ "$ARCH" == "aarch64" ]]; then
                IS_PI=true
                echo "Detected: Raspberry Pi / ARM64 Linux"
            else
                IS_PI=false
                echo "Detected: Linux ($ARCH)"
            fi
            ;;
        Darwin)
            IS_PI=false
            DISTRO="macos"
            echo "Detected: macOS ($ARCH)"
            ;;
        *)
            echo "Unsupported OS: $OS"
            exit 1
            ;;
    esac
}

# Install system dependencies
install_system_deps() {
    echo ""
    echo "=== Installing system dependencies ==="

    case "$DISTRO" in
        macos)
            if ! command -v brew &> /dev/null; then
                echo "Homebrew not found. Install from https://brew.sh"
                exit 1
            fi
            brew install python@3.11 ffmpeg opencv || true
            ;;
        ubuntu|debian|raspbian)
            sudo apt update
            sudo apt install -y \
                python3-venv \
                python3-pip \
                python3-dev \
                ffmpeg \
                v4l-utils \
                libopencv-dev \
                libatlas-base-dev
            ;;
        fedora)
            sudo dnf install -y \
                python3-devel \
                python3-pip \
                ffmpeg \
                opencv-devel
            ;;
        arch)
            sudo pacman -S --noconfirm \
                python \
                python-pip \
                ffmpeg \
                opencv
            ;;
        *)
            echo "Unknown distro: $DISTRO"
            echo "Please install manually: python3, pip, ffmpeg, opencv"
            ;;
    esac
}

# Create virtual environment
setup_venv() {
    echo ""
    echo "=== Creating Python virtual environment ==="
    cd "$PROJECT_DIR"

    if [[ -d "venv" ]]; then
        echo "venv already exists, skipping creation"
    else
        python3 -m venv venv
        echo "Created venv"
    fi

    # Activate venv
    source venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip
}

# Install Python dependencies
install_python_deps() {
    echo ""
    echo "=== Installing Python dependencies ==="

    # On Pi, use piwheels for faster installs
    if [[ "$IS_PI" == true ]]; then
        pip install --extra-index-url https://www.piwheels.org/simple -r requirements.txt
    else
        pip install -r requirements.txt
    fi
}


# Download YOLO model
download_model() {
    echo ""
    echo "=== Downloading YOLOv8n model ==="
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 2>/dev/null || {
        echo "Model will download on first run"
    }
}

# Main
detect_platform
install_system_deps
setup_venv
install_python_deps

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"

download_model

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run Terminal Guidance:"
echo "  cd $PROJECT_DIR"
echo "  source venv/bin/activate"
echo "  python run.py"
echo ""
echo "Edit config/default.yaml to configure camera and flight controller."
