import argparse
import os
from pathlib import Path

import torch
from torchvision.transforms import ToPILImage, v2

from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata
)
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.configs.types import FeatureType

# os.environ["CUDA_VISIBLE_DEVICES"] = "13"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/train/pi0_pusht")
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--optimizer_lr", type=float, default=2.5e-5)
    parser.add_argument("--optimizer_beta_1", type=float, default=0.9)
    parser.add_argument("--optimizer_beta_2", type=float, default=0.95)
    parser.add_argument("--optimizer_eps", type=float, default=1e-8)
    parser.add_argument("--optimizer_weight_decay", type=float, default=1e-10)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--display_freq", type=int, default=100)
    parser.add_argument("--save_checkpoint_freq", type=int, default=10000)
    parser.add_argument("--use_sampler", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Select your device
    device = torch.device(args.device)

    # Get input-output features in dataset to fine-tuning properly input-output shapes of p0 model
    dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
    # print("dataset_metadata: ", dataset_metadata)
    features = dataset_to_policy_features(dataset_metadata.features)
    # print("features: ", features)
    output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {
        key: ft for key, ft in features.items() if key not in output_features
    }
    print("output_features: ", output_features)
    print("input_features: ", input_features)

    # Define p0 model (policy)
    # class PI0Policy fine-tuninig input-output liner layer shapes by input_features and output_features.
    train_cfg = PI0Config(
        input_features=input_features, output_features=output_features
    )
    policy = PI0Policy(train_cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    # Define Transform for data-augmentation
    # randam Brightness
    # randam Contrast
    # randam Saturation
    # randam Hue
    # random affine transform is not used,
    # because In reinforcement learning, the performance of the policy is often lowered when the spatial relationship changes.
    # image_transforms = (
    #     ImageTransforms(train_cfg.dataset.image_transforms) if train_cfg.dataset.image_transforms.enable else None
    # )
    image_transforms = v2.Compose(
        [
            v2.ColorJitter(brightness=(0.5, 1.5)),
            v2.ColorJitter(saturation=(0.5, 1.5)),
            v2.ColorJitter(contrast=(0.5, 1.5)),
            v2.ColorJitter(hue=(-0.1, 0.1)),
            v2.RandomAdjustSharpness(sharpness_factor=2, p=1),
        ]
    )

    # Load dataset
    dataset = LeRobotDataset(
        "lerobot/pusht",
        image_transforms=image_transforms,
        # need specific time period input-output data when train
        # In p0 model, only current timestep [0.0] is used for input-output data
        delta_timestamps={
            # pusht dataset and environment has image, state for input
            "observation.image": (
                [i / dataset_metadata.fps for i in train_cfg.observation_delta_indices]
                if train_cfg.observation_delta_indices
                else [0.0]
            ),
            "observation.state": (
                [i / dataset_metadata.fps for i in train_cfg.observation_delta_indices]
                if train_cfg.observation_delta_indices
                else [0.0]
            ),
            # pusht dataset and environment has action for output
            "action": (
                [i / dataset_metadata.fps for i in train_cfg.action_delta_indices]
                if train_cfg.action_delta_indices
                else [0.0]
            ),
            # example
            # Load the previous image and state at -0.1 seconds before current frame, then load current image and state corresponding to 0.0 second.
            # "observation.image": [-0.1, 0.0],
            # "observation.state": [-0.1, 0.0],
            # Load the previous action (-0.1), the next action to be executed (0.0), and 14 future actions with a 0.1 seconds spacing.
            # All these actions will be used to supervise the policy.
            # "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
        },
    )

    # Define dataloader
    # EpisodeAwareSampler is used to load data from dataset with episode boundary information.
    if args.use_sampler:
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=train_cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # define optimizer
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=args.optimizer_lr,
        betas=(args.optimizer_beta_1, args.optimizer_beta_2),
        eps=args.optimizer_eps,
        weight_decay=args.optimizer_weight_decay,
    )

    # Run training loop.
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            # set input tensor
            # In p0 model, use "task" as input text, so add "task" to batch
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            batch["task"] = [
                "Push the block to the target position\n"
            ] * args.batch_size
            if "observation.state" in batch and batch["observation.state"].ndim == 3:
                # convert [batch_size, 1, state_dim] -> [batch_size, state_dim] shapes
                batch["observation.state"] = batch["observation.state"].squeeze(1)

            if step == 0:
                print("batch.keys: ", batch.keys())
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        print(f"{k} shape: {v.shape}")
                    elif isinstance(v, list) and k == "task":
                        print(f"{k}: {v[0]}")

            # send input tensor to p0 model and calculate loss
            loss, _ = policy.forward(batch)

            # calucurate loss gradient with back propagation
            loss.backward()

            # update model network weights
            optimizer.step()

            # reset gradient
            optimizer.zero_grad()

            if step % args.display_freq == 0:
                print(f"step: {step} loss: {loss.item():.5f}")
            if step % args.save_checkpoint_freq == 0:
                output_dir = os.path.join(args.output_dir, f"{step}")
                os.makedirs(output_dir, exist_ok=True)
                policy.save_pretrained(output_dir)

            step += 1
            if step >= args.n_steps:
                done = True
                break

    # Save a policy checkpoint.
    output_dir = os.path.join(args.output_dir, f"last")
    os.makedirs(output_dir, exist_ok=True)
    policy.save_pretrained(output_dir)
