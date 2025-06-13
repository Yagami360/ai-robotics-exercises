#! /bin/bash
set -eu

# ------------------------------------------------------------
# ACT
# ------------------------------------------------------------
# python eval_aloha.py \
#     --model_type act \
#     --load_checkpoint_dir ../checkpoints/act-aloha-wo-dataaug-20250613/checkpoints/100000/pretrained_model \
#     --output_dir outputs/eval/act-aloha-wo-dataaug-20250613-no-occlusion-step100000

# python eval_aloha.py \
#     --model_type act \
#     --load_checkpoint_dir ../checkpoints/act-aloha-wo-dataaug-20250613/checkpoints/100000/pretrained_model \
#     --output_dir outputs/eval/act-aloha-wo-dataaug-20250613-occlusion-step100000 \
#     --occlusion \
#     --blur

python eval_aloha.py \
    --model_type act \
    --load_checkpoint_dir ../checkpoints/act-aloha-random-erasing-20250613/checkpoints/040000/pretrained_model \
    --output_dir outputs/eval/act-aloha-random-erasing-20250613-no-occlusion-step40000

python eval_aloha.py \
    --model_type act \
    --load_checkpoint_dir ../checkpoints/act-aloha-random-erasing-20250613/checkpoints/040000/pretrained_model \
    --output_dir outputs/eval/act-aloha-random-erasing-20250613-occlusion-step40000 \
    --occlusion

# ------------------------------------------------------------
# PI0
# ------------------------------------------------------------
# python eval_aloha.py \
#     --model_type pi0 \
#     --load_checkpoint_dir ../checkpoints/pi0-aloha-own-random-erasing-20250612-nohup/20000 \
#     --output_dir outputs/eval/pi0-aloha-own-random-erasing-20250612-no-occlusion-step20000

# python eval_aloha.py \
#     --model_type pi0 \
#     --load_checkpoint_dir ../checkpoints/pi0-aloha-own-random-erasing-20250612-nohup/20000 \
#     --output_dir outputs/eval/pi0-aloha-own-random-erasing-20250612-occlusion-step20000 \
#     --occlusion