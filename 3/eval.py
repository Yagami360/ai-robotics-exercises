import os

import torch
import numpy
import imageio

import gym_pusht  # noqa: F401
import gymnasium as gym

import lerobot
from lerobot.common.constants import OBS_ROBOT
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

checkpoint_dir = "../checkpoints/pi0"
os.makedirs(checkpoint_dir, exist_ok=True)
output_dir = "outputs/eval/pi0"
os.makedirs(output_dir, exist_ok=True)

# Select your device
device = "cuda"
# device = "cpu"

# Define the policy
policy = PI0Policy.from_pretrained("lerobot/pi0")
print("Policy config:", vars(policy.config))

# Save checkpoints
# policy.save_pretrained(checkpoint_dir)

# Initialize evaluation environment to render two observation types:
# an image of the scene and state/position of the agent. The environment
# also automatically stops running after 300 interactions/steps.
env = gym.make(
    "gym_pusht/PushT-v0",
    obs_type="pixels_agent_pos",
    max_episode_steps=300,
)

# We can verify that the shapes of the features expected by the policy match the ones from the observations
# produced by the environment
print("policy.config.input_features:", policy.config.input_features)
print("env.observation_space:", env.observation_space)

# Similarly, we can check that the actions produced by the policy will match the actions expected by the
# environment
print("policy.config.output_features:", policy.config.output_features)
print("env.action_space:", env.action_space)

# Reset the policy and environments to prepare for rollout
policy.reset()
numpy_observation, info = env.reset(seed=42)

# Prepare to collect every rewards and all the frames of the episode,
# from initial state to final state.
rewards = []
frames = []

# Render frame of the initial state
frames.append(env.render())

step = 0
done = False
while not done:
    # Prepare observation for the policy running in Pytorch
    state = torch.from_numpy(numpy_observation["agent_pos"])
    image = torch.from_numpy(numpy_observation["pixels"])

    # Convert to float32 with image from channel first in [0,255]
    # to channel last in [0,1]
    state = state.to(torch.float32)
    image = image.to(torch.float32) / 255
    image = image.permute(2, 0, 1)

    # Send data tensors from CPU to GPU
    state = state.to(device, non_blocking=True)
    image = image.to(device, non_blocking=True)

    # Add extra (empty) batch dimension, required to forward the policy
    state = state.unsqueeze(0)
    image = image.unsqueeze(0)

    # Define the observation
    # observation = {
    #     "observation.state": state,
    #     "observation.image": image,
    # }

    observation = {
        OBS_ROBOT: state,                           # ロボットの状態: OBS_ROBOT定数を使用
        "image": image,                             # ロボットのカメラ画像: 画像特徴量
        "task": "Push the object to the target",    # ロボットへの制御指示テキスト
    }

    # Predict the next action with respect to the current observation
    with torch.inference_mode():
        action = policy.select_action(observation)

    # Prepare the action for the environment
    numpy_action = action.squeeze(0).to("cpu").numpy()

    # Step through the environment and receive a new observation
    numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
    print(f"{step=} {reward=} {terminated=}")

    # Keep track of all the rewards and frames
    rewards.append(reward)
    frames.append(env.render())

    # The rollout is considered done when the success state is reached (i.e. terminated is True),
    # or the maximum number of iterations is reached (i.e. truncated is True)
    done = terminated | truncated | done
    step += 1

if terminated:
    print("Success!")
else:
    print("Failure!")

# Get the speed of environment (i.e. its number of frames per second).
fps = env.metadata["render_fps"]

# Encode all frames into a mp4 video.
video_path = os.path.join(output_dir, "rollout.mp4")
imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

print(f"Video of the evaluation is available in '{video_path}'.")
