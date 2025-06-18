#! /bin/bash
set -eu
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=1

# ------------------------------------------------------------
# ACT
# ------------------------------------------------------------
# STEP=010000
# STEP=020000
# STEP=030000
# STEP=040000
# STEP=060000
# STEP=080000
STEP=100000

rm -rf outputs/eval/act-xarm-**-step${STEP}

python eval_xarm.py \
    --model_type act \
    --load_checkpoint_dir ../checkpoints/act-xarm-wo-dataaug-20250614/checkpoints/${STEP}/pretrained_model \
    --output_dir outputs/eval/act-xarm-wo-dataaug-20250614-step${STEP} \
    --normalize_img

python eval_xarm.py \
    --model_type act \
    --load_checkpoint_dir ../checkpoints/act-xarm-wo-dataaug-20250614/checkpoints/${STEP}/pretrained_model \
    --output_dir outputs/eval/act-xarm-wo-dataaug-20250614-occlusion-step${STEP} \
    --normalize_img --occlusion --occlusion_shuffle
