import argparse
import importlib.util
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

try:
    # NOTE: git clone https://github.com/DepthAnything/Depth-Anything-V2 and fix path your appropriate directory.
    depth_anything_path = os.path.join(
        os.path.dirname(__file__), "..", "Depth-Anything-V2"
    )
    import sys

    sys.path.append(depth_anything_path)
    from depth_anything_v2.dpt import DepthAnythingV2
except:
    print("If you want to use depth map, please install Depth-Anything-V2.")


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="pi0", choices=["act", "pi0"])
    parser.add_argument("--output_dir", type=str, default="outputs/eval/pi0_aloha")
    parser.add_argument(
        "--load_checkpoint_dir",
        type=str,
        default="../checkpoints/08-12-47_aloha_pi0/checkpoints/last/pretrained_model",
    )
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--fix_seed", action="store_true")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--normalize_img", action="store_true", default=True)
    parser.add_argument("--depth_model_checkpoint_path", type=str, default=None)
    parser.add_argument("--occlusion", action="store_true")
    parser.add_argument("--occlusion_shuffle", action="store_true")
    parser.add_argument("--occlusion_x", type=int, default=250)
    parser.add_argument("--occlusion_y", type=int, default=230)
    parser.add_argument("--occlusion_w", type=int, default=75)
    parser.add_argument("--occlusion_h", type=int, default=75)
    parser.add_argument("--occlusion_alpha", type=float, default=1.0)
    parser.add_argument("--blur", action="store_true")
    parser.add_argument("--blur_kernel_size", type=int, default=15)
    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.gpu_id < 0:
        device = "cpu"
    else:
        device = "cuda"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.fix_seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Define simulation environment with AlohaInsertion-v0
    os.environ["MUJOCO_GL"] = "egl"

    env = gym.make(
        "gym_aloha/AlohaInsertion-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=args.max_episode_steps,
    )
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

    # Load depth map preprocessing model
    if (
        args.depth_model_checkpoint_path is not None
        and args.depth_model_checkpoint_path != ""
    ):
        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }
        encoder = "vitb"

        depth_model = DepthAnythingV2(**model_configs[encoder])
        depth_model.load_state_dict(
            torch.load(args.depth_model_checkpoint_path, map_location="cpu")
        )
        depth_model = depth_model.to(device).eval()

    # -----------------------------------------------
    # Infer policy with simulation environment
    # -----------------------------------------------
    num_success = 0
    num_failure = 0
    occlusion_x = args.occlusion_x
    occlusion_y = args.occlusion_y
    occlusion_h = args.occlusion_h
    occlusion_w = args.occlusion_w
    occlusion_alpha = args.occlusion_alpha

    for episode in range(args.num_episodes):
        policy.reset()
        if args.fix_seed:
            observation_np, info = env.reset(seed=args.seed)
        else:
            observation_np, info = env.reset()

        rewards = []
        frames = []
        frames_depth = []
        step = 0
        done = False

        # Render initial frame
        frame = env.render()
        if args.occlusion:
            frame = add_occlusion(
                frame,
                occlusion_x,
                occlusion_y,
                occlusion_h,
                occlusion_w,
                occlusion_alpha,
            )
        if args.blur:
            frame = cv2.GaussianBlur(
                frame, (args.blur_kernel_size, args.blur_kernel_size), 0
            )

        frames.append(frame)

        while not done:
            # aloha environment has x-y position of the agent as the observation
            state = torch.from_numpy(observation_np["agent_pos"]).to(device)
            state = state.to(torch.float32)
            state = state.unsqueeze(0)

            # aloha environment has RGB image of the environment as the observation
            image = torch.from_numpy(observation_np["pixels"]["top"]).to(device)
            if args.normalize_img:
                image = image.to(torch.float32) / 255
            image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)

            image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if args.normalize_img:
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

            image = torch.from_numpy(image_np).to(device)
            if args.normalize_img:
                image = image.to(torch.float32) / 255
            image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)
            # cv2.imwrite(f"{args.output_dir}/env_image.png", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            # aloha insert task expects the following observation format
            observation = {
                # agent's x-y position
                "observation.state": state,
                # environment's RGB image
                "observation.images.top": image,
                # agent's control instruction text
                "task": ["Insert the peg into the socket"],
            }

            # add depth map to the observation
            if (
                args.depth_model_checkpoint_path is not None
                and args.depth_model_checkpoint_path != ""
            ):
                depth_map = depth_model.infer_image(image_np)
                depth_map_vis = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
                depth_map_vis = np.stack([depth_map_vis, depth_map_vis, depth_map_vis], axis=-1)
                frames_depth.append(depth_map_vis)
                # cv2.imwrite(f"{args.output_dir}/env_depth_map.png", cv2.cvtColor(depth_map_vis, cv2.COLOR_RGB2BGR))

                if args.normalize_img:
                    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

                depth_map = np.stack([depth_map, depth_map, depth_map], axis=0)
                depth_map = torch.from_numpy(depth_map).unsqueeze(0).to(device)
                observation["observation.depths.top"] = depth_map

            if episode == 0 and step == 0:
                for key in observation:
                    if isinstance(observation[key], torch.Tensor) or isinstance(
                        observation[key], np.ndarray
                    ):
                        print(
                            f"[observation.{key}] shape={observation[key].shape}, min={observation[key].min()}, max={observation[key].max()}, dtype={observation[key].dtype}"
                        )

            # infer the next action based on the policy
            with torch.inference_mode():
                action = policy.select_action(observation)
                if episode == 0 and step == 0:
                    print(
                        f"[action] shape={action.shape}, min={action.min()}, max={action.max()}, dtype={action.dtype}"
                    )

            # step through the simulation environment and receive a new observation
            action_np = action.squeeze(0).to("cpu").numpy()
            if episode == 0 and step == 0:
                print(
                    f"[action_np] shape={action_np.shape}, min={action_np.min()}, max={action_np.max()}, dtype={action_np.dtype}"
                )

            observation_np, reward, terminated, truncated, info = env.step(action_np)
            print(f"{step=} {reward=} {terminated=}")

            # render the environment
            frame = env.render()
            if args.occlusion:
                frame = add_occlusion(
                    frame,
                    occlusion_x,
                    occlusion_y,
                    occlusion_h,
                    occlusion_w,
                    occlusion_alpha,
                )
            if args.blur:
                frame = cv2.GaussianBlur(
                    frame, (args.blur_kernel_size, args.blur_kernel_size), 0
                )
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
            num_success += 1
        else:
            print("Failure!")
            num_failure += 1

        # save the simulation frames as a video
        if terminated:
            video_name = f"eval_frames_ep{episode}_ok.mp4"
        else:
            video_name = f"eval_frames_ep{episode}_ng.mp4"
        video_path = os.path.join(args.output_dir, video_name)
        imageio.mimsave(str(video_path), np.stack(frames), fps=env.metadata["render_fps"])
        print(f"Video of the evaluation is available in '{video_path}'.")

        if (
            args.depth_model_checkpoint_path is not None
            and args.depth_model_checkpoint_path != ""
        ):
            if terminated:
                video_name = f"eval_frames_depth_ep{episode}_ok.mp4"
            else:
                video_name = f"eval_frames_depth_ep{episode}_ng.mp4"
            video_path = os.path.join(args.output_dir, video_name)
            imageio.mimsave(str(video_path), np.stack(frames_depth), fps=env.metadata["render_fps"])

        # shuffle occlusion position and size for next episode
        if args.occlusion_shuffle:
            x_range = np.arange(-100, 101, 10)
            y_range = np.arange(-100, 101, 10)
            h_range = np.arange(-10, 51, 10)
            w_range = np.arange(-10, 51, 10)

            occlusion_x = args.occlusion_x + np.random.choice(x_range)
            occlusion_y = args.occlusion_y + np.random.choice(y_range)
            occlusion_h = args.occlusion_h + np.random.choice(h_range)
            occlusion_w = args.occlusion_w + np.random.choice(w_range)

    print(f"Success rate: {num_success / args.num_episodes * 100:.2f}%")
    print(f"Failure rate: {num_failure / args.num_episodes * 100:.2f}%")

    # write evaluation results to a file
    with open(os.path.join(args.output_dir, "eval_results.txt"), "w") as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.write(f"Success rate: {num_success / args.num_episodes * 100:.2f}%\n")
        f.write(f"Failure rate: {num_failure / args.num_episodes * 100:.2f}%\n")
