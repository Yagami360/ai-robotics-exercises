#!/bin/bash
set -eu
PROJECT_DIR=$(cd $(dirname $0)/..; pwd)
# export CUDA_VISIBLE_DEVICES=0

cd ${PROJECT_DIR}/lerobot

# aloha シミュレーター環境用にファインチューニング
# aloha のシミュレーター環境の場合、docker 内で動かすと "mujoco.FatalError: gladLoadGL error" が出る
# このエラーは推論時にシミュレーターを render しているために発生する。そのため、eval_freq を 200000 にして、エラーが出ないようにする
python lerobot/scripts/train.py \
    --policy.path=lerobot/pi0 \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human_image \
    --env.type=aloha \
    --batch_size=2 \
    --num_workers=4 \
    --steps=100000 \
    --log_freq 1 \
    --eval_freq 200000 \
    --policy.device=cuda
