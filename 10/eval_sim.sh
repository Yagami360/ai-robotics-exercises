#!/bin/bash
PROJECT_DIR=$(cd $(dirname $0)/..; pwd)
export DISPLAY=:1

source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate isaac-labs

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/anaconda3/envs/isaac-labs/lib/python3.10/site-packages/torch/lib

# python eval_sim.py
vglrun -d :1 python eval_sim.py \
    --model_path ${PROJECT_DIR}/checkpoints/gr00t.bimanual_panda_gripper.Transport/checkpoint-10000
