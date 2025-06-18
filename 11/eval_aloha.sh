#! /bin/bash
set -eu

# ------------------------------------------------------------
# ACT
# ------------------------------------------------------------
# STEPS=(020000 040000 060000 080000 100000)
STEPS=(020000)

for STEP in ${STEPS[@]}; do
    rm -rf outputs/eval/act-aloha-**-step${STEP}

    python eval_aloha.py \
        --model_type act \
        --load_checkpoint_dir ../checkpoints/act-aloha-wo-dataaug-20250614/checkpoints/${STEP}/pretrained_model \
        --output_dir outputs/eval/act-aloha-wo-dataaug-20250614-step${STEP} \
        --num_episodes 100 --normalize_img

    python eval_aloha.py \
        --model_type act \
        --load_checkpoint_dir ../checkpoints/act-aloha-wo-dataaug-20250614/checkpoints/${STEP}/pretrained_model \
        --output_dir outputs/eval/act-aloha-wo-dataaug-20250614-occlusion-step${STEP} \
        --num_episodes 100 --normalize_img --occlusion

    python eval_aloha.py \
        --model_type act \
        --load_checkpoint_dir ../checkpoints/act-aloha-random-erasing-20250614/checkpoints/${STEP}/pretrained_model \
        --output_dir outputs/eval/act-aloha-random-erasing-20250614-step${STEP} \
        --num_episodes 100 --normalize_img

    python eval_aloha.py \
        --model_type act \
        --load_checkpoint_dir ../checkpoints/act-aloha-random-erasing-20250614/checkpoints/${STEP}/pretrained_model \
        --output_dir outputs/eval/act-aloha-random-erasing-20250614-occlusion-step${STEP} \
        --num_episodes 100 --normalize_img --occlusion
done

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
