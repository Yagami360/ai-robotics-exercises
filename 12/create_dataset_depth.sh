#! /bin/bash
set -eu
PROJECT_DIR=$(cd $(dirname $0)/..; pwd)

rm -rf ${HOME}/.cache/huggingface/lerobot/Yagami360/aloha_sim_insertion_human_with_depth_images_20250619

python create_dataset_depth.py

ls -l ${HOME}/.cache/huggingface/lerobot/Yagami360/aloha_sim_insertion_human_with_depth_images_20250619
ls -l ${HOME}/.cache/huggingface/lerobot/Yagami360/aloha_sim_insertion_human_with_depth_images_20250619/data/chunk-000

mkdir -p ${PROJECT_DIR}/huggingface/lerobot/Yagami360/aloha_sim_insertion_human_with_depth_images_20250619
cp -r ${HOME}/.cache/huggingface/lerobot/Yagami360/aloha_sim_insertion_human_with_depth_images_20250619 ${PROJECT_DIR}/huggingface/lerobot/Yagami360/aloha_sim_insertion_human_with_depth_images_20250619

# huggingface-cli upload Yagami360/aloha_sim_insertion_human_with_depth_images aloha_sim_insertion_human_with_depth_images_20250619 --repo-type=dataset
