import argparse
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import lerobot
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lerobot.common.datasets.lerobot_dataset import (LeRobotDataset,
                                                     LeRobotDatasetMetadata)
from matplotlib import cm
from tqdm import tqdm

# NOTE: git clone https://github.com/DepthAnything/Depth-Anything-V2 and fix path your appropriate directory.
depth_anything_path = os.path.join(os.path.dirname(__file__), "..", "Depth-Anything-V2")
import sys

sys.path.append(depth_anything_path)
from depth_anything_v2.dpt import DepthAnythingV2


def create_dataset(
    model: Any,
    original_dataset_id: str,
    custom_dataset_id: str,
    fps: int = 50,
    robot_type: str = "aloha",
    push_to_hub: bool = False,
):
   # get metadata from original dataset
    original_metadata = LeRobotDatasetMetadata(original_dataset_id)
    features = original_metadata.features.copy()
    print("original features:", features)

    # add new feature
    features["observation.depths.top"] = {
        "dtype": "image",
        "shape": features["observation.images.top"]["shape"],
        "names": features["observation.images.top"]["names"],
    }
    print("new features:", features)

    # create new dataset
    custom_dataset = LeRobotDataset.create(
        repo_id=custom_dataset_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=True,
        image_writer_processes=4,
        image_writer_threads=8,
    )

    # load original dataset
    original_dataset = LeRobotDataset(original_dataset_id)

    # copy all data
    num_episodes = len(original_dataset.episode_data_index["from"])
    # num_episodes = 1

    for episode_idx in tqdm(range(num_episodes), desc="Creating dataset with episodes"):
        global_frame_start = original_dataset.episode_data_index["from"][episode_idx].item()
        global_frame_end = original_dataset.episode_data_index["to"][episode_idx].item()
        for frame_idx in tqdm(range(global_frame_start, global_frame_end), desc="Creating dataset with frames"):
            frame = original_dataset[frame_idx]
            new_frame = {}

            # copy original frame data
            for k, v in frame.items():
                if k in ['task_index', 'episode_index', 'index', 'frame_index', 'timestamp']:
                    continue
                if k == "observation.images.top" and isinstance(v, torch.Tensor) and v.shape[0] == 3:
                    v = v.permute(1, 2, 0)
                if k in ["next.done"]:
                    v = v.unsqueeze(0)
                new_frame[k] = v.clone() if hasattr(v, "clone") else v

            img = frame["observation.images.top"]
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).cpu().numpy()
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)

            # add depth image as new feature data
            depth_map = model.infer_image(img)
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            depth_map = np.stack([depth_map, depth_map, depth_map], axis=-1)
            new_frame["observation.depths.top"] = depth_map

            # add 1 step data (frame)
            custom_dataset.add_frame(new_frame)

        # save episode
        custom_dataset.save_episode()

    # push to huggingface
    if push_to_hub:
        custom_dataset.push_to_hub()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_dataset_id", type=str, default="lerobot/aloha_sim_insertion_human_image"
    )
    parser.add_argument(
        "--custom_dataset_id",
        type=str,
        default="Yagami360/aloha_sim_insertion_human_with_depth_images_20250619",
    )
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--robot_type", type=str, default="aloha")
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    parser.add_argument("--model_checkpoint_path", type=str, default="../checkpoints/depth_anything_v2/depth_anything_v2_vitb.pth")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    if args.model_checkpoint_path.endswith("depth_anything_v2_vits.pth"):
        encoder = 'vits'
    elif args.model_checkpoint_path.endswith("depth_anything_v2_vitb.pth"):
        encoder = 'vitb'
    elif args.model_checkpoint_path.endswith("depth_anything_v2_vitl.pth"):
        encoder = 'vitl'
    else:
        raise ValueError(f"Invalid model checkpoint path: {args.model_checkpoint_path}")

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(args.model_checkpoint_path, map_location='cpu'))
    model = model.to(args.device).eval()

    custom_dataset = create_dataset(
        model=model,
        original_dataset_id=args.original_dataset_id,
        custom_dataset_id=args.custom_dataset_id,
        fps=args.fps,
        robot_type=args.robot_type,
        push_to_hub=args.push_to_hub,
    )
