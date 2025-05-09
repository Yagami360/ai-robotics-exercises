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
        mkdir -p ${OUTPUT_DIR}
        python lerobot/scripts/visualize_dataset.py \
            --repo-id lerobot/aloha_static_coffee \
            --root ${OUTPUT_DIR} \
            --episode-index 0 \
            --save 1 \
            --output-dir ${OUTPUT_DIR}/visualizations
        ```

        ```sh
        rerun --web-viewer ${OUTPUT_DIR}/visualizations/lerobot_aloha_static_coffee_episode_0.rrd
        ```

        rerun の GUI が起動し、データセットを可視化できる

        <img width="734" alt="Image" src="https://github.com/user-attachments/assets/4342bedf-b65c-4822-b506-805583ab1659" />

    上記スクリプト（`1_load_lerobot_dataset.py`）のように、LeRobot では、`LeRobotDataset()` メソッドを使用して `dataset = LeRobotDataset("lerobot/aloha_static_coffee")` のような形式で、LeRobot のデータセットを簡単に読み込むことができるようになっている


1. 事前学習済みモデルで推論する<br>

    Hugging Face hub からダウンロードした事前学習済みモデルを使用して、推論を行う
    `lerobot/diffusion_pusht` : gym-pusht のシミュレーション環境で事前学習されたモデル

    - GPU を使用する場合

        ```sh
        cd ${PROJECT_ROOT}/lerobot
        python lerobot/scripts/eval.py \
            --policy.path=lerobot/diffusion_pusht \
            --output_dir=${OUTPUT_DIR}/eval/diffusion_pusht/175000 \
            --env.type=pusht \
            --eval.batch_size=10 \
            --eval.n_episodes=10 \
            --policy.use_amp=false \
            --policy.device=cuda
        ```

    - CPU を使用する場合

        ```sh
        cd ${PROJECT_ROOT}/lerobot
        python lerobot/scripts/eval.py \
            --policy.path=lerobot/diffusion_pusht \
            --output_dir=${OUTPUT_DIR}/eval/diffusion_pusht/175000 \
            --env.type=pusht \
            --eval.batch_size=4 \
            --eval.n_episodes=2 \
            --policy.use_amp=false \
            --policy.device=cpu
        ```


1. モデルを学習する