import os

import torch
import numpy
import imageio

import gymnasium as gym
import gym_aloha  # noqa: F401

import lerobot
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

os.environ["CUDA_VISIBLE_DEVICES"] = "15"

load_checkpoint_dir = "../checkpoints/03-05-39_aloha_pi0/checkpoints/last/pretrained_model"
output_dir = "outputs/eval/pi0_aloha"
os.makedirs(output_dir, exist_ok=True)

# Select your device
device = "cuda"
# device = "cpu"

# Initialize simulation environment
# os.environ["MUJOCO_GL"] = "osmesa"
# os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_GL"] = "glfw"

env = gym.make(
    "gym_aloha/AlohaInsertion-v0",
    max_episode_steps=300,
)

# Define the policy
policy = PI0Policy.from_pretrained(
    load_checkpoint_dir,
    strict=False,
)
print("Policy config:", vars(policy.config))

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
observation_np, info = env.reset(seed=42)

# Prepare to collect every rewards and all the frames of the episode,
# from initial state to final state.
rewards = []
frames = []

# Render frame of the initial state
frames.append(env.render())

# step = 0
# done = False
# while not done:
#     # ロボットの x, y 位置
#     state = torch.from_numpy(observation_np["agent_pos"]).to(device)
#     state = state.to(torch.float32)
#     state = state.unsqueeze(0)

#     # 環境の画像
#     image = torch.from_numpy(observation_np["pixels"]).to(device)
#     image = image.to(torch.float32) / 255
#     image = image.permute(2, 0, 1)
#     image = image.unsqueeze(0)

#     # π0 モデルのポリシー（行動方策）が期待する形式に合わせて観測データ（observation）を構成
#     observation = {
#         # ロボットの状態
#         "observation.state": state,
#         # 環境の画像
#         "observation.image": image,
#         # ロボットへの制御指示テキスト
#         # `lerobot/pusht` の学習用データセットと同じ内容のテキストにする
#         "task": ["Push the T-shaped block onto the T-shaped target."]
#     }

#     # π0 モデルの行動方策に基づき、次の行動を推論
#     with torch.inference_mode():
#         action = policy.select_action(observation)

#     # Prepare the action for the environment
#     action_np = action.squeeze(0).to("cpu").numpy()

#     # Step through the environment and receive a new observation
#     observation_np, reward, terminated, truncated, info = env.step(action_np)
#     print(f"{step=} {reward=} {terminated=}")

#     # Keep track of all the rewards and frames
#     rewards.append(reward)
#     frames.append(env.render())

#     # The rollout is considered done when the success state is reached (i.e. terminated is True),
#     # or the maximum number of iterations is reached (i.e. truncated is True)
#     done = terminated | truncated | done
#     step += 1

# if terminated:
#     print("Success!")
# else:
#     print("Failure!")

# # Get the speed of environment (i.e. its number of frames per second).
# fps = env.metadata["render_fps"]

# # Encode all frames into a mp4 video.
# video_path = os.path.join(output_dir, f"eval_frames.mp4")
# imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

# print(f"Video of the evaluation is available in '{video_path}'.")
