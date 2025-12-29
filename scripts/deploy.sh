#!/bin/bash
#
# Deploy Terminal Guidance to Raspberry Pi
#
# This script copies the project to a Pi, installs dependencies,
# and optionally sets up the systemd service.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[*]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[X]${NC} $1"
}

# Get script directory (where this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "  Terminal Guidance - Pi Deployment"
echo "========================================"
echo ""

# Get Pi connection details
read -p "Raspberry Pi IP address: " PI_IP
if [[ -z "$PI_IP" ]]; then
    print_error "IP address is required"
    exit 1
fi

read -p "Username [pi]: " PI_USER
PI_USER="${PI_USER:-pi}"

read -p "Remote install path [~/terminal_guidance]: " REMOTE_PATH
REMOTE_PATH="${REMOTE_PATH:-~/terminal_guidance}"

read -p "Install as systemd service? [y/N]: " INSTALL_SERVICE
INSTALL_SERVICE="${INSTALL_SERVICE:-n}"

read -sp "Password: " PI_PASSWORD
echo ""

# Build SSH/SCP options
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"

# Check for sshpass
if ! command -v sshpass &> /dev/null; then
    print_error "sshpass is required for password authentication"
    print_error "Install with: brew install sshpass"
    exit 1
fi

PI_TARGET="${PI_USER}@${PI_IP}"

# Helper function to run SSH commands with password
run_ssh() {
    sshpass -p "$PI_PASSWORD" ssh $SSH_OPTS "$PI_TARGET" "$@"
}

# Helper function to run rsync with password
run_rsync() {
    sshpass -p "$PI_PASSWORD" rsync "$@"
}

# Test connection
print_status "Testing connection to ${PI_TARGET}..."
if ! run_ssh "echo 'Connection successful'" 2>/dev/null; then
    print_error "Failed to connect to Pi"
    print_error "Check IP address, username, and password"
    exit 1
fi

print_status "Connection verified"

# Check Pi architecture
print_status "Checking Pi architecture..."
PI_ARCH=$(run_ssh "uname -m")
print_status "Pi architecture: $PI_ARCH"

if [[ "$PI_ARCH" != "aarch64" && "$PI_ARCH" != "armv7l" ]]; then
    print_warning "Unexpected architecture: $PI_ARCH"
    read -p "Continue anyway? [y/N]: " CONTINUE
    if [[ "$(echo "$CONTINUE" | tr '[:upper:]' '[:lower:]')" != "y" ]]; then
        exit 1
    fi
fi

# Check Python version
print_status "Checking Python version on Pi..."
PI_PYTHON=$(run_ssh "python3 --version 2>/dev/null || echo 'not found'")
print_status "Pi Python: $PI_PYTHON"

if [[ "$PI_PYTHON" == "not found" ]]; then
    print_error "Python 3 not found on Pi"
    print_error "Install with: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

# Check for python3-venv (need version-specific package on Ubuntu)
print_status "Checking python3-venv on Pi..."
PY_VERSION=$(run_ssh "python3 -c 'import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")'")
if ! run_ssh "python3 -m ensurepip --version" >/dev/null 2>&1; then
    print_warning "python3-venv not installed"
    print_status "Installing python${PY_VERSION}-venv..."
    run_ssh "echo '$PI_PASSWORD' | sudo -S apt update && echo '$PI_PASSWORD' | sudo -S apt install -y python${PY_VERSION}-venv"
fi

# Check for FFmpeg
print_status "Checking FFmpeg on Pi..."
if run_ssh "which ffmpeg" >/dev/null 2>&1; then
    print_status "FFmpeg found"
else
    print_warning "FFmpeg not found on Pi"
    print_warning "Video streaming will not work without FFmpeg"
    read -p "Install FFmpeg on Pi? [Y/n]: " INSTALL_FFMPEG
    if [[ "$(echo "$INSTALL_FFMPEG" | tr '[:upper:]' '[:lower:]')" != "n" ]]; then
        print_status "Installing FFmpeg on Pi..."
        run_ssh "echo '$PI_PASSWORD' | sudo -S apt update && echo '$PI_PASSWORD' | sudo -S apt install -y ffmpeg"
    fi
fi

# Create remote directory
print_status "Creating remote directory..."
run_ssh "mkdir -p $REMOTE_PATH"

# Sync project files (excluding unnecessary files)
print_status "Syncing project files to Pi..."
run_rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.pytest_cache' \
    --exclude 'venv' \
    --exclude '.venv' \
    --exclude '*.egg-info' \
    --exclude '.DS_Store' \
    --exclude '*.pt' \
    --exclude '*.onnx' \
    --exclude 'runs/' \
    --exclude 'qgc/' \
    --exclude '.claude' \
    --exclude 'logs/' \
    -e "ssh $SSH_OPTS" \
    "$PROJECT_DIR/" "${PI_TARGET}:${REMOTE_PATH}/"

print_status "Files synced successfully"

# Setup virtual environment and install dependencies
print_status "Setting up Python virtual environment on Pi..."
run_ssh << EOF
    cd $REMOTE_PATH

    # Create venv if it doesn't exist or is broken
    if [ ! -f "venv/bin/activate" ]; then
        echo "Creating virtual environment..."
        rm -rf venv
        python3 -m venv venv
    fi

    # Activate and install dependencies
    echo "Installing dependencies..."
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

    echo "Dependencies installed"
EOF

print_status "Python environment ready"

# Update config for Pi defaults
print_status "Checking configuration..."
run_ssh << EOF
    cd $REMOTE_PATH

    # Create config directory if needed
    mkdir -p config

    # Check if default config exists
    if [ ! -f "config/default.yaml" ]; then
        echo "Warning: No config/default.yaml found"
        echo "You may need to create one before running"
    else
        echo "Config file found"
    fi
EOF

# Install systemd service if requested
if [[ "$(echo "$INSTALL_SERVICE" | tr '[:upper:]' '[:lower:]')" == "y" ]]; then
    print_status "Installing systemd service..."

    # Create service file
    SERVICE_FILE=$(cat << EOF
[Unit]
Description=Terminal Guidance - Drone Tracking System
After=network.target

[Service]
Type=simple
User=$PI_USER
WorkingDirectory=$REMOTE_PATH
ExecStart=$REMOTE_PATH/venv/bin/python -m src.app
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
)

    # Write and install service
    echo "$SERVICE_FILE" | run_ssh "echo '$PI_PASSWORD' | sudo -S tee /etc/systemd/system/terminal-guidance.service > /dev/null"

    run_ssh << EOF
        echo '$PI_PASSWORD' | sudo -S systemctl daemon-reload
        echo '$PI_PASSWORD' | sudo -S systemctl enable terminal-guidance
        echo "Service installed and enabled"
        echo ""
        echo "Service commands:"
        echo "  sudo systemctl start terminal-guidance"
        echo "  sudo systemctl stop terminal-guidance"
        echo "  sudo systemctl status terminal-guidance"
        echo "  journalctl -u terminal-guidance -f"
EOF

    read -p "Start service now? [y/N]: " START_NOW
    if [[ "$(echo "$START_NOW" | tr '[:upper:]' '[:lower:]')" == "y" ]]; then
        run_ssh "echo '$PI_PASSWORD' | sudo -S systemctl start terminal-guidance"
        print_status "Service started"
    fi
fi

echo ""
echo "========================================"
print_status "Deployment complete!"
echo "========================================"
echo ""
echo "To run manually on Pi:"
echo "  ssh ${PI_TARGET}"
echo "  cd ${REMOTE_PATH}"
echo "  source venv/bin/activate"
echo "  python -m src.app"
echo ""
echo "Web UI will be available at:"
echo "  http://${PI_IP}:5000"
echo ""

if [[ "$(echo "$INSTALL_SERVICE" | tr '[:upper:]' '[:lower:]')" == "y" ]]; then
    echo "Systemd service commands:"
    echo "  sudo systemctl start terminal-guidance"
    echo "  sudo systemctl stop terminal-guidance"
    echo "  sudo systemctl restart terminal-guidance"
    echo "  journalctl -u terminal-guidance -f"
    echo ""
fi
