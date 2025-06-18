#! /bin/bash
set -eu

rm -rf ${HOME}/.cache/huggingface/lerobot/Yagami360/aloha_sim_insertion_human_with_segument_images

python create_dataset.py

ls -l ${HOME}/.cache/huggingface/lerobot/Yagami360/aloha_sim_insertion_human_with_segument_images
ls -l ${HOME}/.cache/huggingface/lerobot/Yagami360/aloha_sim_insertion_human_with_segument_images/data/chunk-000
