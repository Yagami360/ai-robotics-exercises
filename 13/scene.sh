#!/bin/bash
set -eu

if [ "$(uname)" == "Darwin" ]; then
    echo "Your platform is MacOS"
    python scene_on_mac.py
else
    echo "Your platform is Linux Server"
    echo "[Warning] before running the script, you need to run VNC server"

    export DISPLAY=:1
    # export MUJOCO_GL=egl
    # export PYOPENGL_PLATFORM=egl

    # # ライブラリのバージョン競合を解決
    # export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

    # # Anaconda環境のlibstdc++を無効化
    # export CONDA_OVERRIDE_CUDA=11.8
    # export CONDA_OVERRIDE_CUDNN=8.9.2.26

    # # システムのlibstdc++を優先
    # export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

    python scene.py
    # vglrun -d :1 python scene.py
fi
