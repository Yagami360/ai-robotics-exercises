#! /bin/bash
set -eu

NUM_EPISODES=50
# NUM_EPISODES=100

# ------------------------------------------------------------
# ACT
# ------------------------------------------------------------
STEPS=(020000)
# STEPS=(020000 040000 060000 080000 100000)

for STEP in ${STEPS[@]}; do
    # rm -rf outputs/eval/act-aloha-**-step${STEP}

    python eval_aloha.py \
        --model_type act \
        --load_checkpoint_dir ../checkpoints/act-aloha-wo-dataaug-20250614/checkpoints/${STEP}/pretrained_model \
        --output_dir outputs/eval/act-aloha-wo-dataaug-20250614-step${STEP} \
        --num_episodes ${NUM_EPISODES} --normalize_img

    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-wo-dataaug-20250614/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-wo-dataaug-20250614-occlusion-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img --occlusion

    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-wo-dataaug-20250614/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-wo-dataaug-20250614-blur-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img --blur

    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-random-erasing-20250614/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-random-erasing-20250614-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img

    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-random-erasing-20250614/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-random-erasing-20250614-occlusion-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img --occlusion
done

# ------------------------------------------------------------
# PI0
# ------------------------------------------------------------
# STEPS=(100000)
# # STEPS=(020000 040000 060000 080000 100000)

# for STEP in ${STEPS[@]}; do
#     # rm -rf outputs/eval/pi0-aloha-**-step${STEP}

#     python eval_aloha.py \
#         --model_type pi0 \
#         --load_checkpoint_dir ../checkpoints/06-28-03_aloha_pi0/checkpoints/${STEP}/pretrained_model \
#         --output_dir outputs/eval/pi0-aloha-wo-dataaug-20250618-step${STEP} \
#         --num_episodes ${NUM_EPISODES} --normalize_img

#     python eval_aloha.py \
#         --model_type pi0 \
#         --load_checkpoint_dir ../checkpoints/06-28-03_aloha_pi0/checkpoints/${STEP}/pretrained_model \
#         --output_dir outputs/eval/pi0-aloha-wo-dataaug-20250618-occlusion-step${STEP} \
#         --num_episodes ${NUM_EPISODES} --normalize_img \
#         --occlusion

#     python eval_aloha.py \
#         --model_type pi0 \
#         --load_checkpoint_dir ../checkpoints/06-28-03_aloha_pi0/checkpoints/${STEP}/pretrained_model \
#         --output_dir outputs/eval/pi0-aloha-wo-dataaug-20250618-blur-step${STEP} \
#         --num_episodes ${NUM_EPISODES} --normalize_img \
#         --blur
# done
