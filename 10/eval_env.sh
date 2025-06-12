#!/bin/bash
PROJECT_DIR=$(cd $(dirname $0)/..; pwd)
export DISPLAY=:1

source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate isaac-labs

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/anaconda3/envs/isaac-labs/lib/python3.10/site-packages/torch/lib

vglrun -d :1 python eval_env.py \
    --data_config single_panda_gripper \
    --model_path ${PROJECT_DIR}/checkpoints/gr00t.single_panda_gripper.OpenDrawer/checkpoint-3000
