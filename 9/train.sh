#!/bin/bash
# set -eu
PROJECT_DIR=$(cd $(dirname $0)/..; pwd)

source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate isaac-labs

cd $PROJECT_DIR/Isaac-GR00T
python scripts/gr00t_finetune.py \
    --dataset-path ./demo_data/robot_sim.PickNPlace \
    --output-dir ${PROJECT_DIR}/checkpoints/gr00t.robot_sim.PickNPlace \
    --data_config gr1_arms_only \
    --num-gpus 1

poweroff
