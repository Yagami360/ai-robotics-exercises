#!/bin/bash
set -eu

# check os
if [ "$(uname)" == "Darwin" ]; then
    echo "Your platform is MacOS"
    python scene_on_mac.py
else
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    export DISPLAY=:1

    echo "Your platform is Linux"
    python scene.py
fi
