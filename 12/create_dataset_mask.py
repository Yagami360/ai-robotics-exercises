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
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata
)
from matplotlib import cm
from segment_anything import (
    SamAutomaticMaskGenerator,
    SamPredictor,
    sam_model_registry
)
from tqdm import tqdm


def label_to_rgb(segmentation, num_labels=None):
    """
    segmentation: numpy array (H, W), uint8 (ラベル値: 0, 1, 2, 3...)
    num_labels: ラベル数（Noneなら自動検出）
    return: numpy array (H, W, 3), uint8 (RGB値)
    """
    if num_labels is None:
        num_labels = segmentation.max() + 1

    # カラーマップを生成（0は背景色、1以降は異なる色）
    colors = cm.tab20(np.linspace(0, 1, num_labels))  # (num_labels, 4) RGBA
    colors = (colors[:, :3] * 255).astype(np.uint8)  # RGBに変換

    # 背景（ラベル0）は黒にする
    colors[0] = [0, 0, 0]

    # ラベル値からRGB値を取得
    rgb = colors[segmentation]
    return rgb


def create_dataset(
    sam_model: Any,
    original_dataset_id: str,
    custom_dataset_id: str,
    fps: int = 50,
    robot_type: str = "aloha",
    push_to_hub: bool = False,
    max_masks: int = 100,
):
    # get metadata from original dataset
    original_metadata = LeRobotDatasetMetadata(original_dataset_id)
    features = original_metadata.features.copy()
    print("original features:", features)

    # add new feature
    features["observation.segmentations.top"] = {
        "dtype": "image",
        "shape": features["observation.images.top"]["shape"],
        "names": features["observation.images.top"]["names"],
    }
    features["observation.segmentations.vis.top"] = {
        "dtype": "image",
        "shape": features["observation.images.top"]["shape"],
        "names": features["observation.images.top"]["names"],
    }
    # features["observation.bboxs.top"] = {
    #     "dtype": "float32",
    #     "shape": (max_masks, 4),
    #     "names": ["num_masks", "bbox_coords"],
    # }
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
    # num_episodes = len(original_dataset.episode_data_index["from"])
    num_episodes = 1

    for episode_idx in tqdm(range(num_episodes), desc="Creating dataset with episodes"):
        global_frame_start = original_dataset.episode_data_index["from"][
            episode_idx
        ].item()
        global_frame_end = original_dataset.episode_data_index["to"][episode_idx].item()
        # print(f"episode_idx: {episode_idx}, global_frame_start: {global_frame_start}, global_frame_end: {global_frame_end}")
        for frame_idx in tqdm(
            range(global_frame_start, global_frame_end),
            desc="Creating dataset with frames",
        ):
            frame = original_dataset[frame_idx]
            new_frame = {}

            # copy original frame data
            for k, v in frame.items():
                # print(f"[frame] k: {k}, v.shape: {v.shape if isinstance(v, torch.Tensor) else v}")
                if k in [
                    "task_index",
                    "episode_index",
                    "index",
                    "frame_index",
                    "timestamp",
                ]:
                    continue
                if (
                    k == "observation.images.top"
                    and isinstance(v, torch.Tensor)
                    and v.shape[0] == 3
                ):
                    v = v.permute(1, 2, 0)
                if k in ["next.done"]:
                    v = v.unsqueeze(0)
                new_frame[k] = v.clone() if hasattr(v, "clone") else v

            img = frame["observation.images.top"]
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).cpu().numpy()
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)

            # add segmentation image as new feature data
            mask_generator = SamAutomaticMaskGenerator(sam_model)
            masks = mask_generator.generate(img)

            # NOTE: sort by area value to fix each object's label value
            masks = sorted(masks, key=lambda x: x["area"], reverse=True)

            segmentation = np.zeros(img.shape[:2], dtype=np.uint8)
            for i, mask in enumerate(masks):
                segmentation[mask["segmentation"]] = i + 1
            new_frame["observation.segmentations.top"] = torch.from_numpy(
                np.stack([segmentation] * 3, axis=-1)
            )

            segmentation_rgb = label_to_rgb(segmentation)
            new_frame["observation.segmentations.vis.top"] = torch.from_numpy(
                segmentation_rgb
            )

            # add BBOX as new feature data
            # bboxes = np.array([mask["bbox"] for mask in masks], dtype=np.float32)
            # if len(bboxes) > max_masks:
            #     bboxes = bboxes[:max_masks]
            # elif len(bboxes) < max_masks:
            #     padding = np.zeros((max_masks - len(bboxes), 4), dtype=np.float32)
            #     bboxes = np.vstack([bboxes, padding])
            # new_frame["observation.bboxs.top"] = torch.from_numpy(bboxes)

            # 面積情報
            # areas = np.array([mask["area"] for mask in masks], dtype=np.int64)
            # new_frame["observation.area.top"] = torch.from_numpy(areas)

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
        "--original_dataset_id",
        type=str,
        default="lerobot/aloha_sim_insertion_human_image",
    )
    parser.add_argument(
        "--custom_dataset_id",
        type=str,
        default="Yagami360/aloha_sim_insertion_human_with_segument_images_20250619",
    )
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--robot_type", type=str, default="aloha")
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    parser.add_argument(
        "--sam_model_type",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
    )
    parser.add_argument(
        "--sam_checkpoint_path",
        type=str,
        default="../checkpoints/sam/sam_vit_h_4b8939.pth",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    sam_model = sam_model_registry[args.sam_model_type](
        checkpoint=args.sam_checkpoint_path
    )
    sam_model.to(args.device)

    custom_dataset = create_dataset(
        sam_model=sam_model,
        original_dataset_id=args.original_dataset_id,
        custom_dataset_id=args.custom_dataset_id,
        fps=args.fps,
        robot_type=args.robot_type,
        push_to_hub=args.push_to_hub,
    )
