#!/bin/bash
set -eu
PROJECT_DIR=$(cd $(dirname $0)/..; pwd)
DATASET_NAME="single_panda_gripper.OpenDrawer"
# DATASET_NAME="single_panda_gripper.OpenSingleDoor"
# DATASET_NAME="bimanual_panda_gripper.Transport"
# DATASET_NAME="bimanual_panda_hand.LiftTray"
# DATASET_NAME="bimanual_panda_gripper.Threading"

source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate isaac-labs

mkdir -p ${PROJECT_DIR}/datasets

# Download dataset
# See: https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
huggingface-cli download nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
    --repo-type dataset \
    --include "${DATASET_NAME}/**" \
    --local-dir ${PROJECT_DIR}/datasets

python load_dataset.py --dataset_path ${PROJECT_DIR}/datasets/${DATASET_NAME}
