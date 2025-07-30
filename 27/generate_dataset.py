import os
import nest_asyncio
nest_asyncio.apply()

from argparse import ArgumentParser, Namespace
from isaaclab.app import AppLauncher

parser = ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([])
args_cli.enable_cameras = True
args_cli.kit_args = "--enable omni.videoencoding"

config = {
    "task": "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0",
    "num_envs": 1,
    "generation_num_trials": 100,
    "input_file": "../datasets/teleop_franka_demo/annotated_dataset.hdf5",
    "output_file": "../datasets/teleop_franka_demo/generated_dataset.hdf5", 
    "pause_subtask": False,
    "enable": "omni.kit.renderer.capture",
    # "headless": True,
}

# Update the default configuration
args_dict = vars(args_cli)
args_dict.update(config)
args_cli = Namespace(**args_dict)

# Now launch the simulator with the final configuration
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import asyncio
import gymnasium as gym
import numpy as np
import random
import torch

import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.datagen.generation import env_loop, setup_env_config, setup_async_generation
from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths, interactive_update_randomizable_params, reset_env
from isaaclab.managers import ObservationTermCfg as ObsTerm

import isaaclab_tasks  # noqa: F401
num_envs = args_cli.num_envs

# Setup output paths and get env name
output_dir, output_file_name = setup_output_paths(args_cli.output_file)
env_name = args_cli.task or get_env_name_from_dataset(args_cli.input_file)

# Configure environment
env_cfg, success_term = setup_env_config(
    env_name=env_name,
    output_dir=output_dir,
    output_file_name=output_file_name,
    num_envs=num_envs,
    device=args_cli.device,
    generation_num_trials=args_cli.generation_num_trials,
)

# Set observation output directory
OUTPUT_DIR = "datasets/generated_dataset"
for obs in vars(env_cfg.observations.rgb_camera).values():
    if not isinstance(obs, ObsTerm):
        continue
    obs.params["image_path"] = os.path.join(OUTPUT_DIR, obs.params["image_path"])
env_cfg.observations

# create environment
env = gym.make(env_name, cfg=env_cfg).unwrapped

# set seed for generation
random.seed(env.cfg.datagen_config.seed)
np.random.seed(env.cfg.datagen_config.seed)
torch.manual_seed(env.cfg.datagen_config.seed)

# reset before starting
reset_env(env, 100)

# set randomizable parameters to generate more diverse dataset
randomizable_params = {
    "randomize_franka_joint_state": {
        "mean": (0.0, 0.5, 0.01),
        "std": (0.0, 0.1, 0.01),
    },
    "randomize_cube_positions": {
        "pose_range": {
                "x": (0.3, 0.9, 0.01),
                "y": (-0.3, 0.3, 0.01),
            },
        "min_separation": (0.0, 0.5, 0.01),
    }
}

for i in range(len(env.unwrapped.event_manager._mode_term_cfgs["reset"])):
    event_term = env.unwrapped.event_manager._mode_term_cfgs["reset"][i]
    name = env.unwrapped.event_manager.active_terms["reset"][i]
    print(f"Updating parameters for event: {event_term.func.__name__}")
    interactive_update_randomizable_params(event_term, name, randomizable_params[name], env=env)

# data generation
import sys

# Create a new output capture
class OutputCapture:
    def __init__(self):
        self._buffer = ""
    
    def write(self, text):
        if text.strip():  # Only process non-empty strings
            # print(text.rstrip())
            pass

    def flush(self):
        if self._buffer:
            print(self._buffer)
            self._buffer = ""

# Move stdout redirection before setup_async_generation
old_stdout = sys.stdout
sys.stdout = OutputCapture()

try:
    # Setup and run async data generation
    async_components = setup_async_generation(
        env=env,
        num_envs=args_cli.num_envs,
        input_file=args_cli.input_file,
        success_term=success_term,
        pause_subtask=args_cli.pause_subtask
    )

    future = asyncio.ensure_future(asyncio.gather(*async_components['tasks']))
    env_loop(env, async_components['action_queue'], async_components['info_pool'], async_components['event_loop'])
except asyncio.CancelledError:
    print("Tasks were cancelled.")
except AttributeError as e:
    if "'FrankaCubeStackIKRelMimicEnv' object has no attribute 'scene'" in str(e):
        print("Environment was closed during execution. This is expected behavior.")
except Exception as e:
    print(f"Error occurred: {str(e)}")
finally:
    # Restore original stdout first
    sys.stdout = old_stdout

    # Cancel the future and ignore any AttributeErrors from pending tasks
    if 'future' in locals():
        future.cancel()
        try:
            async_components['event_loop'].run_until_complete(future)
        except (asyncio.CancelledError, AttributeError) as e:
            if isinstance(e, AttributeError) and "'FrankaCubeStackIKRelMimicEnv' object has no attribute 'scene'" in str(e):
                print("Environment was closed during execution. This is expected behavior!")
            elif isinstance(e, asyncio.CancelledError):
                print("Tasks were properly cancelled during cleanup.")
            else:
                print(f"Unexpected cleanup error: {str(e)}")
