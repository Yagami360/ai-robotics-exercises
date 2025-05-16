import argparse
import os

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import numpy as np
import torch

import lerobot
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/eval/pi0_pusht")
    parser.add_argument(
        "--load_checkpoint_dir",
        type=str,
        default="../checkpoints/09-50-05_pusht_pi0/checkpoints/last/pretrained_model",
    )
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.gpu_id < 0:
        device = "cpu"
    else:
        device = "cuda"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Define simulation environment with PushT
    env = gym.make(
        "gym_pusht/PushT-v0",
        # 観測データ（observation） は、ロボットの x,y位置（pos） + 環境の画像（pixels）
        obs_type="pixels_agent_pos",
        max_episode_steps=args.max_episode_steps,
    )
    observation_np, info = env.reset(seed=args.seed)
    print("env.observation_space:", env.observation_space)
    print("env.action_space:", env.action_space)

    # Load pre-trained p0 model (policy)
    # This pre-trained model needs to finetune with the PushT environment input-output feature shapes.
    policy = PI0Policy.from_pretrained(
        args.load_checkpoint_dir,
        strict=False,
    )
    policy.reset()
    print("Policy config:", vars(policy.config))
    print("policy.config.input_features:", policy.config.input_features)
    print("policy.config.output_features:", policy.config.output_features)

    # -----------------------------------------------
    # Infer p0 policy with simulation environment
    # -----------------------------------------------
    rewards = []
    frames = []
    step = 0
    done = False

    # Render initial frame
    frames.append(env.render())

    while not done:
        # pusht environment has x-y position of the agent as the observation
        state = torch.from_numpy(observation_np["agent_pos"]).to(device)
        state = state.to(torch.float32)
        state = state.unsqueeze(0)

        # pusht environment has RGB image of the environment as the observation
        image = torch.from_numpy(observation_np["pixels"]).to(device)
        image = image.to(torch.float32) / 255
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)

        # set observation for the pretrained p0-policy with pusht dataset
        observation = {
            # agent's x-y position
            "observation.state": state,
            # environment's RGB image
            "observation.image": image,
            # agent's control instruction text
            # same as the text in the training dataset of `lerobot/pusht`
            "task": ["Push the T-shaped block onto the T-shaped target."],
        }

        # infer the next action based on the p0-policy
        with torch.inference_mode():
            action = policy.select_action(observation)

        # step through the simulation environment and receive a new observation
        action_np = action.squeeze(0).to("cpu").numpy()
        observation_np, reward, terminated, truncated, info = env.step(action_np)
        print(f"{step=} {reward=} {terminated=}")

        # render the environment
        frames.append(env.render())

        # keep track of all the rewards
        rewards.append(reward)

        # finish inference when the success state is reached (i.e. terminated is True),
        # or the maximum number of iterations is reached (i.e. truncated is True)
        done = terminated | truncated | done
        step += 1

    if terminated:
        print("Success!")
    else:
        print("Failure!")

    # Get fps of simulation environment
    fps = env.metadata["render_fps"]

    # save the simulation frames as a video
    video_path = os.path.join(args.output_dir, f"eval_frames.mp4")
    imageio.mimsave(str(video_path), np.stack(frames), fps=fps)

    print(f"Video of the evaluation is available in '{video_path}'.")
