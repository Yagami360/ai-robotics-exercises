# LeRobot のチュートリアルを実行する

## 使用方法

### LeRobot のインストール（conda を使用する場合）

1. LeRobot インストール用の conda 環境を作成

    ```sh
    conda create -y -n lerobot python=3.10
    conda activate lerobot
    conda install ffmpeg -c conda-forge
    ```

1. LeRobot をインストールする

    ```sh
    cd ${PROJECT_ROOT}
    git clone https://github.com/huggingface/lerobot.git
    cd lerobot
    pip install -e .
    ```

    LeRobot では [gymnasium](https://github.com/Farama-Foundation/Gymnasium) でのシミュレーターがデフォルトでインストールされる

    > Gymnasium: Python で強化学習のシミュレーションを行うための OSS のライブラリ。以前の OpenAI Gym を継承・発展させたもので、強化学習のための標準的な環境インターフェースを提供している。

1. [gym_pusht](https://github.com/huggingface/gym-pusht) シミュレーターをインストールする

    チュートリアルのコードは、`gym-pusht` のシミュレーターを使用しているため、`gym-pusht` のシミュレーターをインストールする

    ```sh
    pip install -e ".[pusht]"
    ```

### LeRobot のチュートリアルを実行する

lerobot のレポジトリの構成は、以下のようになっている

```
/lerobot
├── examples             # contains demonstration examples, start here to learn about LeRobot
|   └── advanced         # contains even more examples for those who have mastered the basics
├── lerobot
|   ├── configs          # contains config classes with all options that you can override in the command line
|   ├── common           # contains classes and utilities
|   |   ├── datasets       # various datasets of human demonstrations: aloha, pusht, xarm
|   |   ├── envs           # various sim environments: aloha, pusht, xarm
|   |   ├── policies       # various policies: act, diffusion, tdmpc
|   |   ├── robot_devices  # various real devices: dynamixel motors, opencv cameras, koch robots
|   |   └── utils          # various utilities
|   └── scripts          # contains functions to execute via command line
|       ├── eval.py                 # load policy and evaluate it on an environment
|       ├── train.py                # train a policy via imitation learning and/or reinforcement learning
|       ├── control_robot.py        # teleoperate a real robot, record data, run a policy
|       ├── push_dataset_to_hub.py  # convert your dataset into LeRobot dataset format and upload it to the Hugging Face hub
|       └── visualize_dataset.py    # load a dataset and render its demonstrations
├── outputs               # contains results of scripts execution: logs, videos, model checkpoints
└── tests                 # contains pytest utilities for continuous integration
```

1. LeRobot のデータセットを可視化する

    - Ubuntsu サーバーなどの GUI がない環境の場合

        ```sh
        mkdir -p outputs
        python lerobot/scripts/visualize_dataset.py \
            --repo-id lerobot/xarm_lift_medium \
            --root outputs \
            --episode-index 0 \
            --save 1 \
            --output-dir outputs/datasets/
        ```

        ```sh
        rerun --web-viewer outputs/datasets/lerobot_aloha_static_coffee_episode_0.rrd
        ```

        rerun の GUI が起動し、データセットを可視化できる

        <img width="734" alt="Image" src="https://github.com/user-attachments/assets/4342bedf-b65c-4822-b506-805583ab1659" />

    上記スクリプト（`1_load_lerobot_dataset.py`）のように、LeRobot では、`LeRobotDataset()` メソッドを使用して `dataset = LeRobotDataset("lerobot/aloha_static_coffee")` のような形式で、LeRobot のデータセットを簡単に読み込むことができるようになっている


1. 事前学習済みモデルで推論する<br>

    Hugging Face hub からダウンロードした事前学習済み強化学習モデル（最適な行動方策 policy を推論するモデル）を使用して推論を行う。

    ```sh
    cd ${PROJECT_ROOT}/lerobot
    mkdir -p outputs
    python lerobot/scripts/eval.py \
        --policy.path=lerobot/diffusion_pusht \
        --output_dir=outputs/eval/diffusion_pusht/175000 \
        --env.type=pusht \
        --eval.batch_size=10 \
        --eval.n_episodes=10 \
        --policy.use_amp=false \
        --policy.device=cuda
    ```
    - `lerobot/diffusion_pusht` : gym-pusht のシミュレーション環境で事前学習された強化学習モデル（最適な行動方策 policy を推論するモデル）
    -  cpu の場合も `--policy.device=cuda` で動作するようになっている（但しかなり遅い）
    - `env.type` : ロボットのタスク
        - `pusht`: ロボットが平面上のオブジェクトをT字型のターゲット位置に押し込むタスク
        - `aloha`: 両腕を使った複雑な操作タスク（物体の把持、移動、操作など）
        - `xarm`: UFactory社のxArmロボットアームを使用する環境で、単腕ロボットによる様々なマニピュレーションタスク

    出力ファイルは、以下のようになり、うまく目的の場所にオブジェクトを移動できていることが見て取れる

https://github.com/user-attachments/assets/ea0be1bc-9ec8-4038-9ca5-8e5a1231fc86

https://github.com/user-attachments/assets/9adc5b59-30d7-4acf-a479-8e84e7af8d83


- `eval_info.json`
    ```json
    {
    "per_episode": [
        {
        "episode_ix": 0,
        "sum_reward": 45.856420459035675,
        "max_reward": 1.0,
        "success": true,
        "seed": 1000
        },
        {
        "episode_ix": 1,
        "sum_reward": 23.790821169876416,
        "max_reward": 1.0,
        "success": true,
        "seed": 1001
        },
        ...,
        {
        "episode_ix": 9,
        "sum_reward": 59.05523120088187,
        "max_reward": 0.9103616945756469,
        "success": false,
        "seed": 1009
        }
    ],
    "aggregated": {
        "avg_sum_reward": 114.6479469723378,
        "avg_max_reward": 0.9879925237535374,
        "pc_success": 60.0,
        "eval_s": 123.24566745758057,
        "eval_ep_s": 12.324567008018494
    },
    "video_paths": [
        "outputs/eval/diffusion_pusht/175000/videos/eval_episode_0.mp4",
        "outputs/eval/diffusion_pusht/175000/videos/eval_episode_1.mp4",
        "outputs/eval/diffusion_pusht/175000/videos/eval_episode_2.mp4",
        "outputs/eval/diffusion_pusht/175000/videos/eval_episode_3.mp4",
        "outputs/eval/diffusion_pusht/175000/videos/eval_episode_4.mp4",
        "outputs/eval/diffusion_pusht/175000/videos/eval_episode_5.mp4",
        "outputs/eval/diffusion_pusht/175000/videos/eval_episode_6.mp4",
        "outputs/eval/diffusion_pusht/175000/videos/eval_episode_7.mp4",
        "outputs/eval/diffusion_pusht/175000/videos/eval_episode_8.mp4",
        "outputs/eval/diffusion_pusht/175000/videos/eval_episode_9.mp4"
    ]
    }
    ```

3. 【オプション】学習状況を可視化するための wandb の設定を行なう

    1. wandb にログインする
        ```sh
        wandb login
        ```

    1. wandb のプロジェクト（`ai-robotics-exercises`）を作成する

4. モデルを学習する

    ```sh
    cd ${PROJECT_ROOT}/lerobot
    python lerobot/scripts/train.py \
        --wandb.project=ai-robotics-exercises \
        --wandb.enable=true \
        --output_dir=outputs/train/diffusion_pusht \
        --env.type=pusht \
        --dataset.repo_id=lerobot/pusht \
        --policy.type=diffusion \
        --policy.device=cuda
    ```

    `lerobot/pusht` のデータセットで追加学習
