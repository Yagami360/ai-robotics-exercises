#! /bin/bash
set -eu

rm -rf ${HOME}/.cache/huggingface/lerobot/Yagami360/aloha_sim_insertion_human_with_segument_images_20250619

python create_dataset_mask.py
# python create_dataset_ma.py --episodes 0,1

ls -l ${HOME}/.cache/huggingface/lerobot/Yagami360/aloha_sim_insertion_human_with_segument_images_20250619
ls -l ${HOME}/.cache/huggingface/lerobot/Yagami360/aloha_sim_insertion_human_with_segument_images_20250619/data/chunk-000
