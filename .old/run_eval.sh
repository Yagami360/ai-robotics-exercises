#!/bin/bash
set -eu

# 環境変数設定
export DISPLAY=:1
export CUDA_VISIBLE_DEVICES=0
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330

# CUDA関連の設定
export CUDA_LAUNCH_BLOCKING=1

# Isaac Sim関連の設定
export OMNI_KIT_ACCEPT_EULA=YES

# メモリ関連の設定
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "環境変数設定完了"
echo "DISPLAY: $DISPLAY"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# シミュレーションの実行
python eval_isaaclab.py \
    --dataset_path "../Isaac-GR00T/demo_data/robot_sim.PickNPlace" \
    --model_path "nvidia/GR00T-N1-2B" \
    --seed 42 \
    --gpu_id 0 \
    --num_steps 300
