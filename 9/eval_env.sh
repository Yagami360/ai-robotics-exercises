#!/bin/bash
export DISPLAY=:1

source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate isaac-labs

python eval_env.py --model_path ../checkpoints/gr00t/checkpoint-1000
