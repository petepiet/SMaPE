#!/bin/bash
# Piano Fingering GUI Launcher
# Double-click or run from terminal to start the GUI without manual venv activation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
source .venv/bin/activate

# Run the GUI
python3 gui.py
