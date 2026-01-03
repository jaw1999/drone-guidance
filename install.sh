#!/bin/bash
# Terminal Guidance - Installation Script
# For Raspberry Pi 5 with NCNN backend

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo ""
echo "==================================="
echo "  Terminal Guidance Installer"
echo "==================================="
echo ""

# Get script directory (works even if script is sourced)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if running as root (we need sudo for apt, but not for pip)
if [[ $EUID -eq 0 ]]; then
    log_error "Do not run this script as root. It will use sudo when needed."
    exit 1
fi

# Check if running on Pi (optional)
if [[ -f /proc/device-tree/model ]]; then
    MODEL=$(tr -d '\0' < /proc/device-tree/model)
    log_info "Detected: $MODEL"
    if [[ ! "$MODEL" =~ "Raspberry Pi" ]]; then
        log_warn "This script is optimized for Raspberry Pi"
    fi
else
    log_warn "Could not detect hardware model - this script is designed for Raspberry Pi 5"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for required commands
for cmd in python3 sudo; do
    if ! command -v "$cmd" &> /dev/null; then
        log_error "Required command not found: $cmd"
        exit 1
    fi
done

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 9 ]]; then
    log_error "Python 3.9+ required, found Python $PYTHON_VERSION"
    exit 1
fi
log_info "Python version: $PYTHON_VERSION"

echo ""
echo "[1/6] Updating package lists..."
sudo apt update

echo ""
echo "[2/6] Installing system dependencies..."
sudo apt install -y \
    python3-venv \
    python3-dev \
    python3-pip \
    python3-numpy \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1

# Try to install python3-ncnn (may not be available on all distros)
if apt-cache show python3-ncnn &> /dev/null; then
    log_info "Installing python3-ncnn from apt..."
    sudo apt install -y python3-ncnn
else
    log_warn "python3-ncnn not available via apt - will install via pip"
fi

echo ""
echo "[3/6] Creating Python virtual environment..."
if [[ -d "venv" ]]; then
    log_info "Virtual environment already exists"
    read -p "Recreate it? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Removing old virtual environment..."
        rm -rf venv
        python3 -m venv venv --system-site-packages
        log_info "Virtual environment recreated"
    fi
else
    python3 -m venv venv --system-site-packages
    log_info "Virtual environment created"
fi

echo ""
echo "[4/6] Installing Python packages..."
# Activate venv
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip wheel setuptools

# Install requirements
if [[ -f "requirements.txt" ]]; then
    pip install -r requirements.txt
else
    log_error "requirements.txt not found in $SCRIPT_DIR"
    exit 1
fi

echo ""
echo "[5/6] Creating config directory..."
mkdir -p config
if [[ ! -f "config/default.yaml" ]]; then
    log_warn "No config/default.yaml found - you may need to create one"
fi

echo ""
echo "[6/6] Setting up YOLO models..."
mkdir -p models

# Check if model already exists
if [[ -d "models/yolov8n_ncnn_model" ]]; then
    log_info "NCNN model already exists at models/yolov8n_ncnn_model"
    read -p "Re-download and export? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Skipping model download"
        SKIP_MODEL=1
    fi
fi

if [[ -z "${SKIP_MODEL:-}" ]]; then
    log_info "Downloading and exporting YOLOv8n to NCNN format..."
    python3 << 'PYTHON_SCRIPT'
import os
import sys
import shutil

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed properly")
    sys.exit(1)

models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

print("Downloading YOLOv8n...")
model = YOLO('yolov8n.pt')

print("Exporting to NCNN format (this may take a few minutes)...")
try:
    model.export(format='ncnn', imgsz=640)
except Exception as e:
    print(f"ERROR: NCNN export failed: {e}")
    print("You may need to install ncnn manually or use a different backend")
    sys.exit(1)

# Move exported files to models directory
ncnn_dir = 'yolov8n_ncnn_model'
if os.path.exists(ncnn_dir):
    dest = os.path.join(models_dir, ncnn_dir)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.move(ncnn_dir, dest)
    print(f"Moved {ncnn_dir} to {dest}")
else:
    print(f"WARNING: Expected {ncnn_dir} not found after export")

# Move .pt file to models directory
pt_file = 'yolov8n.pt'
if os.path.exists(pt_file):
    shutil.move(pt_file, os.path.join(models_dir, pt_file))
    print(f"Moved {pt_file} to models/")

print("Model setup complete!")
PYTHON_SCRIPT
fi

# Verify installation
echo ""
log_info "Verifying installation..."
python3 -c "
import sys
errors = []

try:
    import numpy
    print(f'  numpy: {numpy.__version__}')
except ImportError as e:
    errors.append(f'numpy: {e}')

try:
    import cv2
    print(f'  opencv: {cv2.__version__}')
except ImportError as e:
    errors.append(f'opencv: {e}')

try:
    from ultralytics import YOLO
    print(f'  ultralytics: OK')
except ImportError as e:
    errors.append(f'ultralytics: {e}')

try:
    import flask
    print(f'  flask: {flask.__version__}')
except ImportError as e:
    errors.append(f'flask: {e}')

try:
    import pymavlink
    print(f'  pymavlink: OK')
except ImportError as e:
    errors.append(f'pymavlink: {e}')

if errors:
    print()
    print('ERRORS:')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)
"

if [[ $? -ne 0 ]]; then
    log_error "Installation verification failed"
    exit 1
fi

# Get IP address
IP_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}')
if [[ -z "$IP_ADDR" ]]; then
    IP_ADDR="<your-pi-ip>"
fi

echo ""
echo "==================================="
echo -e "  ${GREEN}Installation Complete!${NC}"
echo "==================================="
echo ""
echo "To run Terminal Guidance:"
echo ""
echo "  cd $SCRIPT_DIR"
echo "  source venv/bin/activate"
echo "  python -m src.app"
echo ""
echo "Web UI will be available at: http://${IP_ADDR}:5000"
echo ""
echo "To run on boot, consider creating a systemd service."
echo ""
