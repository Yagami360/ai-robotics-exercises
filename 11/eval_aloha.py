import argparse
import os

import cv2
import gym_aloha  # noqa: F401
import gymnasium as gym
import imageio
import lerobot
import numpy as np
import torch
from lerobot.common.policies.act.modeling_act import ACTConfig, ACTPolicy
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy


def add_occlusion(
    image, start_x, start_y, occlusion_height, occlusion_width, alpha=1.0
):
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (start_x, start_y),
        (start_x + occlusion_width, start_y + occlusion_height),
        (0, 0, 0),
        -1,
    )
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def rotate_state(state, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    xy = state[..., :2]  # (バッチ, 2)
    xy_rot = np.dot(xy, R.T)
    state_rot = state.copy()
    state_rot[..., :2] = xy_rot
    return state_rot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="pi0", choices=["act", "pi0"])
    parser.add_argument("--output_dir", type=str, default="outputs/eval/pi0_aloha")
    parser.add_argument(
        "--load_checkpoint_dir",
        type=str,
        default="../checkpoints/08-12-47_aloha_pi0/checkpoints/last/pretrained_model",
    )
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--occlusion", action="store_true")
    parser.add_argument("--occlusion_x", type=int, default=250)
    parser.add_argument("--occlusion_y", type=int, default=250)
    parser.add_argument("--occlusion_w", type=int, default=75)
    parser.add_argument("--occlusion_h", type=int, default=75)
    parser.add_argument("--occlusion_alpha", type=float, default=1.0)
    parser.add_argument("--blur", action="store_true")
    parser.add_argument("--blur_kernel_size", type=int, default=15)
    # parser.add_argument("--rotation", action='store_true')
    # parser.add_argument("--rotation_angle", type=int, default=15)
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

    # Load model (policy)
    if args.model_type == "act":
        policy = ACTPolicy.from_pretrained(
            args.load_checkpoint_dir,
            strict=False,
        )
    elif args.model_type == "pi0":
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
        frame = add_occlusion(
            frame,
            args.occlusion_x,
            args.occlusion_y,
            args.occlusion_h,
            args.occlusion_w,
            args.occlusion_alpha,
        )
    if args.blur:
        frame = cv2.GaussianBlur(
            frame, (args.blur_kernel_size, args.blur_kernel_size), 0
        )
    # if args.rotation:
    #     frame = rotate_image(frame, args.rotation_angle)

    frames.append(frame)

    while not done:
        # aloha environment has x-y position of the agent as the observation
        state = torch.from_numpy(observation_np["agent_pos"]).to(device)
        state = state.to(torch.float32)
        state = state.unsqueeze(0)
        # print(f"[state] shape={state.shape}, min={state.min()}, max={state.max()}, dtype={state.dtype}")
        # if args.rotation:
        #     state_np = state.cpu().numpy()
        #     state_np = rotate_state(state_np, args.rotation_angle)
        #     state = torch.from_numpy(state_np).to(device).to(torch.float32)

        # aloha environment has RGB image of the environment as the observation
        image = torch.from_numpy(observation_np["pixels"]["top"]).to(device)
        image = image.to(torch.float32) / 255
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)

        image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)

        # add some occlusion mask to the env image
        if args.occlusion:
            image_np = add_occlusion(
                image_np,
                args.occlusion_x,
                args.occlusion_y,
                args.occlusion_h,
                args.occlusion_w,
                args.occlusion_alpha,
            )
        if args.blur:
            image_np = cv2.GaussianBlur(
                image_np, (args.blur_kernel_size, args.blur_kernel_size), 0
            )
        # if args.rotation:
        #     image_np = rotate_image(image_np, args.rotation_angle)

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
            frame = add_occlusion(
                frame,
                args.occlusion_x,
                args.occlusion_y,
                args.occlusion_h,
                args.occlusion_w,
                args.occlusion_alpha,
            )
        if args.blur:
            frame = cv2.GaussianBlur(
                frame, (args.blur_kernel_size, args.blur_kernel_size), 0
            )
        # if args.rotation:
        #     frame = rotate_image(frame, args.rotation_angle)

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
