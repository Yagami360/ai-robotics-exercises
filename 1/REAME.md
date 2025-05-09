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
            --repo-id lerobot/aloha_static_coffee \
            --root outputs \
            --episode-index 0 \
            --save 1 \
            --output-dir outputs/visualizations
        ```

        ```sh
        rerun --web-viewer outputs/visualizations/lerobot_aloha_static_coffee_episode_0.rrd
        ```

        rerun の GUI が起動し、データセットを可視化できる

        <img width="734" alt="Image" src="https://github.com/user-attachments/assets/4342bedf-b65c-4822-b506-805583ab1659" />

    上記スクリプト（`1_load_lerobot_dataset.py`）のように、LeRobot では、`LeRobotDataset()` メソッドを使用して `dataset = LeRobotDataset("lerobot/aloha_static_coffee")` のような形式で、LeRobot のデータセットを簡単に読み込むことができるようになっている


1. 事前学習済みモデルで推論する<br>

    Hugging Face hub からダウンロードした事前学習済み強化学習モデル（最適な行動方策 policy を推論するモデル）を使用して推論を行う。

    ```sh
    cd ${PROJECT_ROOT}/lerobot
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

    ```sh
    /home/sakai/miniconda3/envs/lerobot/lib/python3.10/site-packages/gymnasium/core.py:311: UserWarning: WARN: env.task to get variables from other wrappers is deprecated and will be removed in v1.0, to get this variable you can do `env.unwrapped.task` for environment variables or `env.get_wrapper_attr('task')` that will search the reminding wrappers.
    logger.warn(
    Stepping through eval batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [31:09<00:00, 1869.08s/it, running_success_rate=50.0%]
    {'avg_sum_reward': 107.82850712331738, 'avg_max_reward': 0.9986999737089051, 'pc_success': 50.0, 'eval_s': 1870.085877418518, 'eval_ep_s': 935.0429430007935}                                             
    INFO 2025-05-09 06:39:58 pts/eval.py:501 End of eval
    ```

1. モデルを学習する

    ```sh
    cd ${PROJECT_ROOT}/lerobot
    python lerobot/scripts/train.py \
        --output_dir=outputs/train/diffusion_pusht \
        --env.type=pusht \
        --policy.type=diffusion \
        --policy.device=cuda
    ```
