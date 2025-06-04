#!/bin/bash
export DISPLAY=:1

# python eval_isaaclab_scene_gr1t1.py
python eval_isaaclab_scene_gr1t2.py --model_path ../checkpoints/gr00t/checkpoint-1000
# python eval_isaaclab_env_gr1t2.py

# vglrun -d :1 python eval_isaaclab_scene_gr1t1.py
# vglrun -d :1 python eval_isaaclab_scene_gr1t2.py
