#!/bin/bash
set -eu

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export DISPLAY=:1

python scene.py
