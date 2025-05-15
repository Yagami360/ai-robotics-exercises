# LeRobot の π0 モデルを gymnasium のシミュレーター環境でファインチューニングする（自身で実装した学習スクリプトを使用する場合）

## 使用方法

### 事前準備

1. CPU メモリ 16GB 程度, GPU メモリ 32GB 以上（V100, A100など）の GPU インスタンスを用意する

1. LeRobot の π0 をインストールする<br>
    ```sh
    cd ${PROJECT_ROOT}/lerobot
    pip install -e ".[pi0]"
    ```

1. Hugging Face にログインする
    ```sh
    huggingface-cli login ${TOKEN}
    ```

    > 推論時に、π0モデルが内部で `google/paligemma-3b-pt-224` というプライベートリポジトリにアクセスするために必要

1. Hugging Face 上の Paligemma のリポジトリのアクセス権限をリクエストする<br>
    "https://huggingface.co/google/paligemma-3b-pt-224" に移動して、アクセス権限をリクエストする

### pusht シミュレーター用にファインチューニングする場合

[`gym-pusht`](https://github.com/huggingface/gym-pusht) のシミュレーター環境用にファインチューニングする場合、以下の手順を実行する


1. LeRobot の π0 モデルを使用した学習スクリプトを実装する

    LeRobot の π0 モデルを `pusht` 用のデータセットでファインチューニングするための学習スクリプト[train_pusht.py](./train_pusht.py) を実装する

    ```python
    ```

    ポイントは、以下の通り

    - xxx

1. π0 モデルを pusht タスク用にファインチューニングする
    ```sh
    python train_pusht.py
    ```

### aloha シミュレーター環境用にファインチューニングする場合

[`gym-aloha`](https://github.com/huggingface/gym-aloha) のシミュレーター環境用にファインチューニングする場合、以下の手順を実行する

1. LeRobot の π0 モデルを使用した学習スクリプトを実装する

    LeRobot の π0 モデルを `aloha` 用のデータセットでファインチューニングするための学習スクリプト[train_aloha.py](./train_aloha.py) を実装する

    ```python
    import os
    import argparse
    from pathlib import Path

    import torch

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.common.datasets.utils import dataset_to_policy_features
    from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.configs.types import FeatureType


    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--output_dir", default="outputs/train/pi0_pusht")
        parser.add_argument("--n_steps", default=10000)
        parser.add_argument("--batch_size", default=4)
        parser.add_argument("--lr", default=1e-4)
        parser.add_argument("--n_workers", default=4)
        parser.add_argument("--display_freq", default=100)
        parser.add_argument("--device", default="cuda")
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
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if key not in output_features}
        print("output_features: ", output_features)
        print("input_features: ", input_features)

        # Define p0 model (policy)
        # class PI0Policy fine-tuninig input-output liner layer shapes by input_features and output_features.
        train_cfg = PI0Config(input_features=input_features, output_features=output_features)
        policy = PI0Policy(train_cfg, dataset_stats=dataset_metadata.stats)
        policy.train()
        policy.to(device)

        # load dataset
        dataset = LeRobotDataset(
            "lerobot/pusht",
            # need specific time period input-output data when train
            # In p0 model, only current timestep [0.0] is used for input-output data
            delta_timestamps={
                # pusht dataset and environment has image, state for input
                "observation.image": [i / dataset_metadata.fps for i in train_cfg.observation_delta_indices] if train_cfg.observation_delta_indices else [0.0],
                "observation.state": [i / dataset_metadata.fps for i in train_cfg.observation_delta_indices] if train_cfg.observation_delta_indices else [0.0],
                # pusht dataset and environment has action for output
                "action": [i / dataset_metadata.fps for i in train_cfg.action_delta_indices] if train_cfg.action_delta_indices else [0.0],

                # example
                # Load the previous image and state at -0.1 seconds before current frame, then load current image and state corresponding to 0.0 second.
                # "observation.image": [-0.1, 0.0],
                # "observation.state": [-0.1, 0.0],
                # Load the previous action (-0.1), the next action to be executed (0.0), and 14 future actions with a 0.1 seconds spacing. All these actions will be used to supervise the policy.
                # "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
            }
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            shuffle=True,
            pin_memory=device.type != "cpu",
            drop_last=True,
        )

        # define optimizer
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

        # Run training loop.
        step = 0
        done = False
        while not done:
            for batch in dataloader:
                # set input tensor
                # In p0 model, use "task" as input text, so add "task" to batch
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                batch["task"] = ["Push the block to the target position\n"] * args.batch_size
                if 'observation.state' in batch and batch['observation.state'].ndim == 3:
                    # convert [batch_size, 1, state_dim] -> [batch_size, state_dim] shapes
                    batch['observation.state'] = batch['observation.state'].squeeze(1)

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
                    print(f"step: {step} loss: {loss.item():.3f}")

                step += 1
                if step >= args.n_steps:
                    done = True
                    break

        # Save a policy checkpoint.
        policy.save_pretrained(args.output_dir)
    ```

    ポイントは、以下の通り

    - xxx

1. π0 モデルを aloha タスク用にファインチューニングする
    ```sh
    python train_aloha.py
    ```
