import argparse
import os

import cv2
import gym_aloha  # noqa: F401
import gym_xarm
import gymnasium as gym
import imageio
import numpy as np
import torch

import lerobot
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="pi0", choices=["act", "pi0"])
    parser.add_argument("--output_dir", type=str, default="outputs/eval/pi0_aloha")
    parser.add_argument(
        "--load_checkpoint_dir",
        type=str,
        default="../checkpoints/08-12-47_aloha_pi0/checkpoints/last/pretrained_model",
    )
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--fix_seed", action="store_true")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--normalize_img", action="store_true", default=True)
    parser.add_argument("--occlusion", action="store_true")
    parser.add_argument("--occlusion_shuffle", action="store_true")
    parser.add_argument("--occlusion_x", type=int, default=350)
    parser.add_argument("--occlusion_y", type=int, default=350)
    parser.add_argument("--occlusion_w", type=int, default=50)
    parser.add_argument("--occlusion_h", type=int, default=50)
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
    # os.environ["MUJOCO_GL"] = "osmesa"

    # print(gym.envs.registry)

    env = gym.make(
        "gym_xarm/XarmLift-v0",
        obs_type="pixels_agent_pos",
        # render_mode="human",
        render_mode="rgb_array",
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

    # policy.reset()
    print("Policy config:", vars(policy.config))
    print("policy.config.input_features:", policy.config.input_features)
    print("policy.config.output_features:", policy.config.output_features)

    # -----------------------------------------------
    # Infer p0 policy with simulation environment
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
            observation_env, info = env.reset(seed=args.seed)
        else:
            observation_env, info = env.reset()

        for k, v in observation_env.items():
            if isinstance(v, np.ndarray):
                print(f"[observation_env] {k}: {v.shape}")
            else:
                print(f"[observation_env] {k}: {v}")

        rewards = []
        frames = []
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
            state = torch.from_numpy(observation_env["agent_pos"]).to(device)
            state = state.to(torch.float32)
            state = state.unsqueeze(0)
            # print(f"[state] shape={state.shape}, min={state.min()}, max={state.max()}, dtype={state.dtype}")

            # aloha environment has RGB image of the environment as the observation
            # NOTE: 学習用データセットの画像サイズに合わせてリサイズ
            image_np = observation_env["pixels"]
            image_np = cv2.resize(image_np, (84, 84))
            image = torch.from_numpy(image_np).to(device)
            if args.normalize_img:
                image = image.to(torch.float32) / 255
            image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)
            # print(f"[image] shape={image.shape}, min={image.min()}, max={image.max()}, mean={image.mean()}, dtype={image.dtype}")

            image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if args.normalize_img:
                image_np = (image_np * 255).astype(np.uint8)
            # print(f"[image_np] shape={image_np.shape}, min={image_np.min()}, max={image_np.max()}, dtype={image_np.dtype}")

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
            # print(f"[image] shape={image.shape}, min={image.min()}, max={image.max()}, mean={image.mean()}, dtype={image.dtype}")

            # p0-policy expects the following observation format
            observation = {
                # agent's x-y position
                "observation.state": state,
                # environment's RGB image
                "observation.image": image,
                # agent's control instruction text
                "task": ["Pick up the cube and lift it."],
            }
            # for k, v in observation.items():
            #     if isinstance(v, torch.Tensor):
            #         print(f"[observation] {k}: shape={v.shape}, min={v.min()}, max={v.max()}, mean={v.mean()}, dtype={v.dtype}")
            #     else:
            #         print(f"[observation] {k}: {v}")

            # infer the next action
            with torch.inference_mode():
                action = policy.select_action(observation)
                # print(f"[action] shape={action.shape}, min={action.min()}, max={action.max()}, dtype={action.dtype}")
                # 0,1,2: [x, y, z] represent the position of the end effector
                # 3: [w] represents the gripper control

            # step through the simulation environment and receive a new observation
            action_np = action.squeeze(0).to("cpu").numpy()

            # グリッパー開閉状態の変換
            # if action_np[3] >= 0.5:
            #     action_np[3] = 1.0
            # elif action_np[3] <= -0.5:
            #     action_np[3] = -1.0
            # else:
            #     action_np[3] = 0.0
            # action_np[3] = (action_np[3] + 1.0) / 2.0
            # action_np[3] = max(0.0, action_np[3])

            # clip the action to the range of [-1.0, 1.0] to avoid the out-of-range error in environment
            action_np = np.clip(action_np, -1.0, 1.0)

            observation_env, reward, terminated, truncated, info = env.step(action_np)
            reward = float(reward)
            terminated = bool(terminated)
            truncated = bool(truncated)
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

        # Get fps of simulation environment
        fps = env.metadata["render_fps"]

        # save the simulation frames as a video
        if terminated:
            video_name = f"eval_frames_ep{episode}_ok.mp4"
        else:
            video_name = f"eval_frames_ep{episode}_ng.mp4"
        video_path = os.path.join(args.output_dir, video_name)
        imageio.mimsave(str(video_path), np.stack(frames), fps=fps)

        if args.occlusion_shuffle:
            # 10間隔でランダム値を生成
            x_range = np.arange(-100, 101, 10)
            y_range = np.arange(-100, 101, 10)
            h_range = np.arange(0, 101, 10)
            w_range = np.arange(0, 101, 10)

            occlusion_x = args.occlusion_x + np.random.choice(x_range)
            occlusion_y = args.occlusion_y + np.random.choice(y_range)
            occlusion_h = args.occlusion_h + np.random.choice(h_range)
            occlusion_w = args.occlusion_w + np.random.choice(w_range)
            # occlusion_alpha = args.occlusion_alpha + np.random.randint(-0.1, 0.1)

    print(f"Success rate: {num_success / args.num_episodes * 100:.2f}%")
    print(f"Failure rate: {num_failure / args.num_episodes * 100:.2f}%")

    # write evaluation results to a file
    with open(os.path.join(args.output_dir, "eval_results.txt"), "w") as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.write(f"Success rate: {num_success / args.num_episodes * 100:.2f}%\n")
        f.write(f"Failure rate: {num_failure / args.num_episodes * 100:.2f}%\n")
