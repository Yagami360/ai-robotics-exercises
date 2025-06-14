#! /bin/bash
set -eu
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=1

# ------------------------------------------------------------
# ACT
# ------------------------------------------------------------
STEP=001000
# STEP=020000
# STEP=040000
# STEP=060000
# STEP=080000
# STEP=100000

# rm -rf outputs/eval/act-xarm-**-step${STEP}

# python eval_xarm.py \
#     --model_type act \
#     --load_checkpoint_dir ../checkpoints/act-xarm-wo-dataaug-20250614-debug/checkpoints/${STEP}/pretrained_model \
#     --output_dir outputs/eval/act-xarm-wo-dataaug-20250614-debug-step${STEP}

python eval_xarm.py \
    --model_type act \
    --load_checkpoint_dir ../checkpoints/act-xarm-wo-dataaug-20250614-debug/checkpoints/${STEP}/pretrained_model \
    --output_dir outputs/eval/act-xarm-wo-dataaug-20250614-debug-occlusion-step${STEP} \
    --occlusion --occlusion_shuffle
