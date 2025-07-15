#! /bin/bash
set -eu
PROJECT_DIR=$(cd $(dirname $0)/..; pwd)

export DISPLAY=:1
# export MUJOCO_GL=egl

python scene.py
