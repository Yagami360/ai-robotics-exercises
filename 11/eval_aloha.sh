#! /bin/bash
set -eu
PROJECT_DIR=$(cd $(dirname $0)/..; pwd)

# cd ${PROJECT_DIR}/lerobot && pip install -e .
# cd ${PROJECT_DIR}/11

NUM_EPISODES=50
# NUM_EPISODES=100

# ------------------------------------------------------------
# ACT
# ------------------------------------------------------------
STEPS=(020000)
# STEPS=(020000 040000 060000 080000 100000)

for STEP in ${STEPS[@]}; do
    # rm -rf outputs/eval/act-aloha-**-step${STEP}

    # original ACT
    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-wo-dataaug-20250614/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-wo-dataaug-20250614-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img

    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-wo-dataaug-20250614/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-wo-dataaug-20250614-blur-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img --blur

    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-wo-dataaug-20250614/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-wo-dataaug-20250614-occlusion-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img --occlusion

    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-wo-dataaug-20250614/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-wo-dataaug-20250614-blur-occlusion-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img --blur --occlusion

    # improved ACT (blur)
    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-blur-20250618/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-blur-20250618-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img

    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-blur-20250618/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-blur-20250618-blur-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img --blur

    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-blur-20250618/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-blur-20250618-occlusion-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img --occlusion

    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-blur-20250618/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-blur-20250618-blur-occlusion-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img --blur --occlusion

    # improved ACT (random erasing)
    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-random-erasing-20250614/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-random-erasing-20250614-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img

    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-random-erasing-20250614/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-random-erasing-20250614-blur-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img --blur

    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-random-erasing-20250614/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-random-erasing-20250614-occlusion-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img --occlusion

    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-random-erasing-20250614/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-random-erasing-20250614-blur-occlusion-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img --blur --occlusion

    # improved ACT (blur + occlusion)
    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-random-erasing-blur-20250619/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-random-erasing-blur-20250619-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img

    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-random-erasing-blur-20250619/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-random-erasing-blur-20250619-blur-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img --blur

    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-random-erasing-blur-20250619/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-random-erasing-blur-20250619-occlusion-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img --occlusion

    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-random-erasing-blur-20250619/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-random-erasing-blur-20250619-blur-occlusion-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img --blur --occlusion

    # improved ACT (depth map)
    # rm -rf outputs/eval/act-aloha-depth-**-step${STEP}

    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-depth-20250619/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-depth-20250619-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img --depth_model_checkpoint_path ../checkpoints/depth_anything_v2/depth_anything_v2_vitb.pth

    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-depth-20250619/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-depth-20250619-blur-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img --depth_model_checkpoint_path ../checkpoints/depth_anything_v2/depth_anything_v2_vitb.pth --blur

    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-depth-20250619/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-depth-20250619-occlusion-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img --depth_model_checkpoint_path ../checkpoints/depth_anything_v2/depth_anything_v2_vitb.pth --occlusion

    # python eval_aloha.py \
    #     --model_type act \
    #     --load_checkpoint_dir ../checkpoints/act-aloha-depth-20250619/checkpoints/${STEP}/pretrained_model \
    #     --output_dir outputs/eval/act-aloha-depth-20250619-blur-occlusion-step${STEP} \
    #     --num_episodes ${NUM_EPISODES} --normalize_img --depth_model_checkpoint_path ../checkpoints/depth_anything_v2/depth_anything_v2_vitb.pth --blur --occlusion

    # improved ACT (depth map + blur + occlusion)
    # rm -rf outputs/eval/act-aloha-depth-blur-erasing-**-step${STEP}

    python eval_aloha.py \
        --model_type act \
        --load_checkpoint_dir ../checkpoints/act-aloha-depth-blur-erasing-20250619/checkpoints/${STEP}/pretrained_model \
        --output_dir outputs/eval/act-aloha-depth-blur-erasing-20250619-step${STEP} \
        --num_episodes ${NUM_EPISODES} --normalize_img --depth_model_checkpoint_path ../checkpoints/depth_anything_v2/depth_anything_v2_vitb.pth

    python eval_aloha.py \
        --model_type act \
        --load_checkpoint_dir ../checkpoints/act-aloha-depth-blur-erasing-20250619/checkpoints/${STEP}/pretrained_model \
        --output_dir outputs/eval/act-aloha-depth-blur-erasing-20250619-blur-step${STEP} \
        --num_episodes ${NUM_EPISODES} --normalize_img --depth_model_checkpoint_path ../checkpoints/depth_anything_v2/depth_anything_v2_vitb.pth --blur

    python eval_aloha.py \
        --model_type act \
        --load_checkpoint_dir ../checkpoints/act-aloha-depth-blur-erasing-20250619/checkpoints/${STEP}/pretrained_model \
        --output_dir outputs/eval/act-aloha-depth-blur-erasing-20250619-occlusion-step${STEP} \
        --num_episodes ${NUM_EPISODES} --normalize_img --depth_model_checkpoint_path ../checkpoints/depth_anything_v2/depth_anything_v2_vitb.pth --occlusion

    python eval_aloha.py \
        --model_type act \
        --load_checkpoint_dir ../checkpoints/act-aloha-depth-blur-erasing-20250619/checkpoints/${STEP}/pretrained_model \
        --output_dir outputs/eval/act-aloha-depth-blur-erasing-20250619-blur-occlusion-step${STEP} \
        --num_episodes ${NUM_EPISODES} --normalize_img --depth_model_checkpoint_path ../checkpoints/depth_anything_v2/depth_anything_v2_vitb.pth --blur --occlusion

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
