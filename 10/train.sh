#!/bin/bash
#set -eu
PROJECT_DIR=$(cd $(dirname $0)/..; pwd)

DATASET_NAME="single_panda_gripper.OpenDrawer"
# DATASET_NAME="single_panda_gripper.OpenSingleDoor"
# DATASET_NAME="bimanual_panda_gripper.Transport"
# DATASET_NAME="bimanual_panda_gripper.Threading"
# DATASET_NAME="bimanual_panda_hand.LiftTray"

source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate isaac-labs

cd ${PROJECT_DIR}
python Isaac-GR00T/scripts/gr00t_finetune.py \
    --dataset-path ${PROJECT_DIR}/datasets/${DATASET_NAME} \
    --output-dir checkpoints/gr00t.${DATASET_NAME} \
    --data_config single_panda_gripper \
    --batch-size 4 \
    --num-gpus 1

# poweroff
