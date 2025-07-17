# Isaac Lab Mimic を使用して、シミュレーター環境上でヒューマノイドロボット（GR1）用の学習用データセットを作成し、Robomimic を使用してモデルを学習＆推論する

1. ヒューマノイドロボット（GR1）のキーボードでの遠隔操作で少数の学習用データセットを作成する

    - キーボードで操作する場合

        ```bash
        ./isaaclab.sh -p scripts/tools/record_demos.py \
            --device cuda \
            --task Isaac-PickPlace-GR1T2-Abs-v0 \
            --teleop_device keyboard \
            --dataset_file ../datasets/teleop_gr1/dataset_gr1.hdf5 \
            --num_demos 10 --enable_pinocchio
        ```

1. 学習用データセットの内容に従ってロボットを動かす（リプレイする）

    - キーボードで操作する場合

        ```bash
        ./isaaclab.sh -p scripts/tools/replay_demos.py \
        --device cuda \
        --task Isaac-PickPlace-GR1T2-Abs-v0 \
        --dataset_file ../datasets/teleop_gr1/dataset_gr1.hdf5 \
        --enable_pinocchio
        ```

1. Isaac Lab Mimic を使用して、自動データ生成のためのアノテーションを付与する

    ```bash
    ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --device cuda \
    --task Isaac-PickPlace-GR1T2-Abs-Mimic-v0 \
    --input_file ../datasets/teleop_gr1/dataset_gr1.hdf5 \
    --output_file ../datasets/teleop_gr1/dataset_annotated_gr1.hdf5 \
    --enable_pinocchio
    ```

1. Isaac Lab Mimic を使用して、遠隔操作で作成した少数の学習用データセットから大量の学習用データセットを作成する

    ```bash
    ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
        --device cuda \
        --headless \
        --num_envs 10 --generation_num_trials 1000 \
        --enable_pinocchio \
        --input_file ../datasets/teleop_gr1/dataset_annotated_gr1.hdf5 \
        --output_file ../datasets/teleop_gr1/generated_dataset_gr1.hdf5
    ```

1. Robomimic を使用して、作成した学習用データセットでモデルを模倣学習で学習する

    Robomimic は、スタンフォード大学が開発した模倣学習（Imitation Learning）のための OSS のフレームワーク。

    Robomimic では Isaac Lab Mimic で作成した学習用データセット（HDF5形式）を用いて、モデルの学習や推論が行えるようになっている

    ```bash
    ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
    --task Isaac-PickPlace-GR1T2-Abs-v0 \
    --algo bc \
    --normalize_training_actions \
    --dataset ../datasets/teleop_gr1/generated_dataset_gr1.hdf5
    ```

    - `--algo`
        - `bc`: BC (Behavioral Cloning): 基本的な模倣学習

        - `bc-rnn`: リカレントニューラルネットワークを使用したBC

        - xxx

        > Robomimic は、従来のCNN/Transformerベースの模倣学習フレームワークで、LLM ベースの VLA モデルなどはサポートされていない点に注意

    `IssacLab/logs/robomimic` 以下に学習済みチェックポイントが保存される

1. 学習済みモデルで推論させながらロボットを動かす

    ```bash
    ./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
    --device cuda \
    --enable_pinocchio \
    --task Isaac-PickPlace-GR1T2-Abs-v0 \
    --num_rollouts 50 \
    --norm_factor_min <NORM_FACTOR_MIN> \
    --norm_factor_max <NORM_FACTOR_MAX> \
    --checkpoint IssacLab/logs/robomimic/desired_model_checkpoint.pth
    ```


## 参考サイト

- https://isaac-sim.github.io/IsaacLab/main/source/overview/teleop_imitation.html#demo-data-generation-and-policy-training-for-a-humanoid-robot
