#!/bin/bash
export DISPLAY=:1

source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate isaac-labs

# python eval_isaaclab_scene_gr1t1.py
python eval_isaaclab_scene_gr1t2.py --model_path ../checkpoints/gr00t/checkpoint-3000
# python eval_isaaclab_env_gr1t2.py
# python check_camera_rot_isaaclab_scene_gr1t2.py

# vglrun -d :1 python eval_isaaclab_scene_gr1t1.py
# vglrun -d :1 python eval_isaaclab_scene_gr1t2.py
