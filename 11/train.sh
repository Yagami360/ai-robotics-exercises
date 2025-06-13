#!/bin/bash
set -eu
PROJECT_DIR=$(cd $(dirname $0)/..; pwd)
# export CUDA_VISIBLE_DEVICES=0

cd ${PROJECT_DIR}/lerobot

# aloha シミュレーター環境用にファインチューニング
# aloha のシミュレーター環境の場合、docker 内で動かすと "mujoco.FatalError: gladLoadGL error" が出る
# このエラーは推論時にシミュレーターを render しているために発生する。そのため、eval_freq を 200000 にして、エラーが出ないようにする

rm -rf ${PROJECT_DIR}/checkpoints/act-aloha-random-erasing-20250613
python lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human_image \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=6 \
    --output_dir=${PROJECT_DIR}/checkpoints/act-aloha-random-erasing-20250613 \
    --env.type=aloha \
    --batch_size=4 \
    --num_workers=1 \
    --steps=100000 \
    --eval_freq 200000 \
    --policy.device=cuda

# rm -rf ${PROJECT_DIR}/checkpoints/pi0-aloha-random-erasing-20250613
# python lerobot/scripts/train.py \
#     --policy.path=lerobot/pi0 \
#     --dataset.repo_id=lerobot/aloha_sim_insertion_human_image \
#     --dataset.image_transforms.enable=true \
#     --dataset.image_transforms.max_num_transforms=6 \
#     --output_dir=${PROJECT_DIR}/checkpoints/pi0-aloha-random-erasing-20250613 \
#     --env.type=aloha \
#     --batch_size=2 \
#     --num_workers=1 \
#     --steps=100000 \
#     --eval_freq 200000 \
#     --policy.device=cuda
