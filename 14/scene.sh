#!/bin/bash
set -eu

if [ "$(uname)" == "Darwin" ]; then
    echo "Your platform is MacOS"
    python scene_on_mac.py
else
    echo "Your platform is Linux Server"
    echo "[Warning] before running the script, you need to run VNC server"

    export DISPLAY=:1

    python scene.py
fi
