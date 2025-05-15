# LeRobot の π0 モデルを gymnasium のシミュレーター環境で推論する（Lerobot の推論コード `lerobot/scripts/eval.py` を使用する場合）

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

### pusht のシミュレーター環境を使用する場合

1. π0 モデルを LeRobot 用にファインチューニングする

    PI0モデルは通常、事前に計算された統計情報（平均と標準偏差）を使って入力データを正規化する。
    オリジナルのπ0 モデルの実装（https://github.com/Physical-Intelligence/openpi）は、JAX で実装されている。
    一方で、LeRobot の π0 モデルは PyTorch で実装されている。

    LeRobot の π0 モデルは、JAXからPyTorchへの変換過程でこの統計情報（平均と標準偏差）が失われたため、モデルは無限大の値を持つ統計情報で初期化される

    そのため、ファインチューニングを行っていない π0 モデルで推論すると以下のエラーが発生する

    ```sh
    cd lerobot
    python lerobot/scripts/eval.py \
        --policy.path=lerobot/pi0 \
        --output_dir=outputs/eval/pi0 \
        --env.type=pusht \
        --eval.batch_size=10 \
        --eval.n_episodes=10 \
        --policy.use_amp=false \
        --policy.device=cuda
    ```
    ```sh
    AssertionError: `mean` is infinity. You should either initialize with `stats` as an argument, or use a pretrained model.
    ```
    Issue: https://github.com/huggingface/lerobot/issues/694

    そのため、まずは LeRobot の π0 モデルをファインチューニングして、推論時の入力データ正規化用の統計情報を計算できるようにしておく必要がある

    LeRobot の学習スクリプトをそのまま利用する場合、π0 モデルのファインチューニングは以下コマンドで行える

    - openpi の π0 モデル（`pi0_base`）のチェックポイントからファインチューニングする場合
        ```sh
        cd lerobot
        python lerobot/scripts/train.py \
            --policy.path=lerobot/pi0 \
            --dataset.repo_id=lerobot/pusht \
            --env.type=pusht \
            --batch_size=2 \
            --num_workers=2 \
            --steps=100000 \
            --policy.device=cuda
        ```
        - `batch_size`, `num_workers`, `steps` は、GPU 環境に応じて要調整
        - 環境に応じて、`--policy.use_amp=true`, `--policy.device=cpu` を指定する
        - `env.type` : ロボットのタスク
            - `pusht`: ロボットが平面上のオブジェクトをT字型のターゲット位置に押し込むタスク
            - `aloha`: 両腕を使った複雑な操作タスク（物体の把持、移動、操作など）
            - `xarm`: UFactory社のxArmロボットアームを使用する環境で、単腕ロボットによる様々なマニピュレーションタスク

    - VLM のデフォルトパラメーターで事前学習された π0 モデルをファインチューニングする場合
        ```sh
        python lerobot/scripts/train.py \
            --policy.type=pi0 \
            --dataset.repo_id=lerobot/pusht \
            --env.type=pusht \
            --batch_size=2 \
            --num_workers=2 \
            --steps=10000 \
            --policy.device=cuda
        ```

    学習用データセットを pusht 用のデータセットとし、シミュレーター環境も pusht 用に設定し、π0 モデルを pusht タスク用にファインチューニングする

    - `lerobot/pusht` データセットの中身

        https://huggingface.co/datasets/lerobot/pusht

        ```yml
        {
            # 環境の画像
            'observation.image': tensor([[[1.0000, 0.9725, 0.9725,  ..., 0.9725, 0.9725, 1.0000],
                [0.9725, 0.9098, 0.9098,  ..., 0.9098, 0.9098, 0.9725],
                [0.9725, 0.9098, 0.9725,  ..., 1.0000, 0.9098, 0.9725],
                ...,
                [1.0000, 0.9725, 0.9725,  ..., 0.9725, 0.9725, 1.0000]]]),
            # ロボットの x, y 位置
            'observation.state': tensor([222.,  97.]),
            # ロボットの次の行動
            'action': tensor([233.,  71.]),
            # エピソードの値
            'episode_index': tensor(0),
            # フレーム（時間ステップ）
            'frame_index': tensor(0),
            # 時間情報
            'timestamp': tensor(0.),
            # エピソードが終了したかどうかのフラグ
            'next.done': tensor(False),
            # エピソードが成功したかどうかのフラグ
            'next.success': tensor(False),
            # データセット内の絶対的なインデックス
            'index': tensor(0),
            # タスクの種類を示すインデックス
            'task_index': tensor(0),
            # ロボットへの制御指示テキスト
            'task': 'Push the T-shaped block onto the T-shaped target.'
        }
        ```

1. gymnasium のシミュレーターを使用して π0 モデルの推論を実行する

    ```sh
    python lerobot/scripts/eval.py \
        --policy.path=outputs/train/2025-05-13/09-50-05_pusht_pi0/checkpoints/last/pretrained_model \
        --output_dir=outputs/eval/pi0_pusht \
        --env.type=pusht \
        --eval.batch_size=10 \
        --eval.n_episodes=10 \
        --policy.device=cuda
    ```

    - `policy.path`: ファインチューニングした π0 モデルのパスを指定。ファインチューニングしてないπ0モデルのパスや `lerobot/pi0` を指定した場合は、以下のエラーが発生する
        ```sh
        AssertionError: `mean` is infinity. You should either initialize with `stats` as an argument, or use a pretrained model.
        ```
    - `env.type` : ロボットのタスク
        - `pusht`: ロボットが平面上のオブジェクトをT字型のターゲット位置に押し込むタスク
        - `aloha`: 両腕を使った複雑な操作タスク（物体の把持、移動、操作など）
        - `xarm`: UFactory社のxArmロボットアームを使用する環境で、単腕ロボットによる様々なマニピュレーションタスク

    以下のような出力が得られる

- 学習ステップ数: 1000 のファインチューニングモデルで推論した場合<br>

https://github.com/user-attachments/assets/6dcd7573-4933-46ff-aa3d-9d2e8a0641b7

- 学習ステップ数: 10000 のファインチューニングモデルで推論した場合<br>

https://github.com/user-attachments/assets/86659dc5-50fa-4d6d-880f-8b3645aa1687

- 学習ステップ数: 60000 のファインチューニングモデルで推論した場合<br>

https://github.com/user-attachments/assets/f9b0a59e-e2c3-44c3-ab0d-903d73a73293

### aloha のシミュレーター環境を使用する場合

[`gym-aloha`](https://github.com/huggingface/gym-aloha) のシミュレーター環境を使用する場合は、以下の手順を実行する

1. π0 モデルを aloha タスク用にファインチューニングする

    ```sh
    cd lerobot
    python lerobot/scripts/train.py \
        --policy.path=lerobot/pi0 \
        --dataset.repo_id=lerobot/aloha_sim_insertion_human_image \
        --env.type=aloha \
        --batch_size=2 \
        --num_workers=2 \
        --steps=100000 \
        --policy.device=cuda
    ```
    - `batch_size`, `num_workers`, `steps` は、インスタンス環境に応じて要調整
    - 環境に応じて、`--policy.use_amp=true`, `--policy.device=cpu` を指定する
    - `env.type` : ロボットのタスク
        - `pusht`: ロボットが平面上のオブジェクトをT字型のターゲット位置に押し込むタスク
        - `aloha`: 両腕を使った複雑な操作タスク（物体の把持、移動、操作など）
        - `xarm`: UFactory社のxArmロボットアームを使用する環境で、単腕ロボットによる様々なマニピュレーションタスク

    学習用データセットを aloha 用のデータセットとし、シミュレーター環境も aloha 用に設定し、π0 モデルを aloha タスク用にファインチューニングする

    - `lerobot/aloha_sim_insertion_human_image` データセットの中身

        https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human_image

        ```yml
        ```

    loss 値の変化は、以下のようになるので、収束するまで十分なステップ数で学習する（学習率や Optimizerはデフォルト値で設定）

    > loss 値のグラフ追加

1. gymnasium のシミュレーターを使用して π0 モデルの推論を実行する

    ```sh
    python lerobot/scripts/eval.py \
        --policy.path=outputs/train/2025-05-13/07-32-44_aloha_pi0/checkpoints/last/pretrained_model \
        --output_dir=outputs/eval/pi0_aloha \
        --env.type=aloha \
        --eval.batch_size=10 \
        --eval.n_episodes=10 \
        --policy.device=cuda
    ```

    - `policy.path`: ファインチューニングした π0 モデルのパスを指定。ファインチューニングしてないπ0モデルのパスや `lerobot/pi0` を指定した場合は、以下のエラーが発生する
        ```sh
        AssertionError: `mean` is infinity. You should either initialize with `stats` as an argument, or use a pretrained model.
        ```
    - `env.type` : ロボットのタスク
        - `pusht`: ロボットが平面上のオブジェクトをT字型のターゲット位置に押し込むタスク
        - `aloha`: 両腕を使った複雑な操作タスク（物体の把持、移動、操作など）
        - `xarm`: UFactory社のxArmロボットアームを使用する環境で、単腕ロボットによる様々なマニピュレーションタスク
    - `eval_freq`: 推論を行なうステップ間隔
        - aloha のシミュレーター環境の場合、docker 内で動かすと "mujoco.FatalError: gladLoadGL error" というグラフィックデバイス関連のエラーが発生する。このエラーは推論時にシミュレーター関連を OpenGL を使用して render しているために発生する。そのため、docker 環境で学習を行なう場合は、`--eval_freq` の値を `--steps` の値よりも大きくすることを推奨。

    以下のような出力が得られる

- 学習ステップ数: 1000 のファインチューニングモデルで推論した場合<br>

https://github.com/user-attachments/assets/f6b3e0c6-f0e0-45d3-b1b2-72c68113e632

- 学習ステップ数: 20000 のファインチューニングモデルで推論した場合<br>

https://github.com/user-attachments/assets/378a2f95-aa01-4cab-8f20-30f2bb184ab8

https://github.com/user-attachments/assets/c07495af-b202-4557-b47d-e7e29118450f

- 学習ステップ数: 100000 のファインチューニングモデルで推論した場合<br>
