#!/bin/bash
export DISPLAY=:1

# pkill -f "python eval_isaaclab_gr1t2_2.py"

# python eval_isaaclab_scene_gr1t1.py
# python eval_isaaclab_scene_gr1t2.py
python eval_isaaclab_env_gr1t2.py

# vglrun -d :1 python eval_isaaclab_scene_gr1t1.py
# vglrun -d :1 python eval_isaaclab_scene_gr1t2.py
