#! /bin/bash
set -eu

# ------------------------------------------------------------
# ACT
# ------------------------------------------------------------
STEP=020000
# STEP=040000
# STEP=060000
# STEP=080000
# STEP=100000

# rm -rf outputs/eval/act-xarm-**-step${STEP}

python eval_xarm.py \
    --model_type act \
    --load_checkpoint_dir ../checkpoints/act-aloha-wo-dataaug-20250613/checkpoints/${STEP}/pretrained_model \
    --output_dir outputs/eval/act-xarm-wo-dataaug-20250613-step${STEP}

# ------------------------------------------------------------
# PI0
# ------------------------------------------------------------
# STEP=040000

# rm -rf outputs/eval/pi0-aloha-*

# python eval_aloha.py \
#     --model_type pi0 \
#     --load_checkpoint_dir ../checkpoints/pi0-aloha-random-erasing-20250613/checkpoints/${STEP}/pretrained_model \
#     --output_dir outputs/eval/pi0-aloha-random-erasing-20250613-step${STEP}


# python eval_aloha.py \
#     --model_type pi0 \
#     --load_checkpoint_dir ../checkpoints/pi0-aloha-random-erasing-20250613/checkpoints/${STEP}/pretrained_model \
#     --output_dir outputs/eval/pi0-aloha-random-erasing-20250613-occlusion-step${STEP} \
#     --occlusion --occlusion_shuffle
