import argparse
import os

import cv2
import gym_aloha  # noqa: F401
import gymnasium as gym
import imageio
import numpy as np
import torch

import lerobot
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy


def add_occlusion(image, start_x, start_y, occlusion_height, occlusion_width, alpha=0.8):
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (start_x, start_y),
        (start_x + occlusion_width, start_y + occlusion_height),
        (0, 0, 0),
        -1,
    )
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/eval/pi0_aloha")
    parser.add_argument(
        "--load_checkpoint_dir",
        type=str,
        default="../checkpoints/08-12-47_aloha_pi0/checkpoints/last/pretrained_model",
    )
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--occlusion", type=bool, default=True)
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

    # Define simulation environment with AlohaInsertion-v0
    os.environ["MUJOCO_GL"] = "egl"
    # os.environ["MUJOCO_GL"] = "osmesa"

    env = gym.make(
        "gym_aloha/AlohaInsertion-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=args.max_episode_steps,
    )
    observation_np, info = env.reset(seed=args.seed)
    print("env.observation_space:", env.observation_space)
    print("env.action_space:", env.action_space)

    # Load pre-trained p0 model (policy)
    # This pre-trained model needs to finetune with the AlohaInsertion environment input-output feature shapes.
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
    frame = env.render()
    if args.occlusion:
        OCCLUSION_X = 270
        OCCLUSION_Y = 180
        OCCLUSION_W = 100
        OCCLUSION_H = 100
        frame = add_occlusion(frame, OCCLUSION_X, OCCLUSION_Y, OCCLUSION_H, OCCLUSION_W)
    frames.append(frame)

    while not done:
        # aloha environment has x-y position of the agent as the observation
        state = torch.from_numpy(observation_np["agent_pos"]).to(device)
        state = state.to(torch.float32)
        state = state.unsqueeze(0)

        # aloha environment has RGB image of the environment as the observation
        image = torch.from_numpy(observation_np["pixels"]["top"]).to(device)
        image = image.to(torch.float32) / 255
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)

        image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)

        # add some occlusion mask to the env image
        if args.occlusion:
            image_np = add_occlusion(image_np, OCCLUSION_X, OCCLUSION_Y, OCCLUSION_H, OCCLUSION_W)
            image = torch.from_numpy(image_np).to(device)
            image = image.to(torch.float32) / 255
            image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)

        # cv2.imwrite(f"{args.output_dir}/env_image.png", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        # p0-policy expects the following observation format
        observation = {
            # agent's x-y position
            "observation.state": state,
            # environment's RGB image
            "observation.images.top": image,
            # agent's control instruction text
            "task": ["Insert the peg into the socket"],
        }

        # infer the next action based on the p0-policy
        with torch.inference_mode():
            action = policy.select_action(observation)

        # step through the simulation environment and receive a new observation
        action_np = action.squeeze(0).to("cpu").numpy()
        observation_np, reward, terminated, truncated, info = env.step(action_np)
        print(f"{step=} {reward=} {terminated=}")

        # render the environment
        frame = env.render()
        if args.occlusion:
            frame = add_occlusion(frame, OCCLUSION_X, OCCLUSION_Y, OCCLUSION_H, OCCLUSION_W)

        # cv2.imwrite(f"{args.output_dir}/env_frame.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frames.append(frame)

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
