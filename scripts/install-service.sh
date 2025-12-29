#!/bin/bash
# Install Terminal Guidance as a systemd service (Pi only)
# This makes it start automatically on boot

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Systemd services only available on Linux"
    exit 1
fi

echo "Installing Terminal Guidance systemd service..."

# Create service file
sudo tee /etc/systemd/system/terminal-guidance.service > /dev/null <<EOF
[Unit]
Description=Terminal Guidance - Drone Companion Computer
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=${PROJECT_DIR}
ExecStart=${PROJECT_DIR}/venv/bin/python -m src.app
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

# Resource limits for Pi
Nice=10
CPUQuota=80%
MemoryMax=1G

[Install]
WantedBy=multi-user.target
EOF

# Reload and enable
sudo systemctl daemon-reload
sudo systemctl enable terminal-guidance

echo ""
echo "Service installed. Commands:"
echo "  sudo systemctl start terminal-guidance   # Start now"
echo "  sudo systemctl stop terminal-guidance    # Stop"
echo "  sudo systemctl status terminal-guidance  # Check status"
echo "  journalctl -u terminal-guidance -f       # View logs"
echo ""
echo "The service will start automatically on boot."
