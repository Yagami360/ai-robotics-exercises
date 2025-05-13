# LeRobot の π0 モデルを使用して gymnasium のシミュレーター環境で推論する

## 使用方法

1. CPU メモリ 16GB 程度, GPU メモリ 40GB 程度（A100など）の GPU インスタンスを用意する

1. LeRobot の π0 をインストールする<br>
    ```sh
    cd ${PROJECT_ROOT}/lerobot
    pip install -e ".[pi0]"
    ```

1. Hugging Face にログインする
    ```sh
    huggingface-cli login
    ```

    > 推論時に、π0モデルが内部で `google/paligemma-3b-pt-224` というプライベートリポジトリにアクセスするために必要

1. Hugging Face 上の Paligemma のリポジトリのアクセス権限をリクエストする<br>
    "https://huggingface.co/google/paligemma-3b-pt-224" に移動して、アクセス権限をリクエストする


1. π0 モデルを LeRobot 用にファインチューニングする

    PI0モデルは通常、事前に計算された統計情報（平均と標準偏差）を使って入力データを正規化する。
    オリジナルのπ0 モデルの実装（https://github.com/Physical-Intelligence/openpi）は、JAX で実装されている。
    一方で、LeRobot の π0 モデルは PyTorch で実装されている。

    LeRobot の π0 モデルは、JAXからPyTorchへの変換過程でこの統計情報（平均と標準偏差）が失われたため、モデルは無限大の値を持つ統計情報で初期化される

    そのため、ファインチューニングを行っていない π0 モデルで推論すると以下のエラーが発生する

    ```sh
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
    https://github.com/huggingface/lerobot/issues/694

    そのため、まずは LeRobot の π0 モデルをファインチューニングして、推論時の入力データ正規化用の統計情報を計算できるようにしておく必要がある

    LeRobot の学習スクリプトをそのまま利用する場合、π0 モデルのファインチューニングは以下コマンドで行える

    ```sh
    python lerobot/scripts/train.py \
        --policy.path=lerobot/pi0 \
        --dataset.repo_id=danaaubakirova/koch_test \
        --env.type=pusht \
        --batch_size=4 \
        --num_workers=4 \
        --steps=1000
    ```
    - `batch_size`, `num_workers`, `steps` は、GPU 環境に応じて要調整
    - 環境に応じて、`--policy.use_amp=true`, `--policy.device=cpu` を指定する
    - `env.type` : ロボットのタスク
        - `pusht`: ロボットが平面上のオブジェクトをT字型のターゲット位置に押し込むタスク
        - `aloha`: 両腕を使った複雑な操作タスク（物体の把持、移動、操作など）
        - `xarm`: UFactory社のxArmロボットアームを使用する環境で、単腕ロボットによる様々なマニピュレーションタスク

1. gymnasium のシミュレーターを使用して π0 モデルの推論を実行する

    - lerobot の推論コードを使用する場合

        ```sh
        python lerobot/scripts/eval.py \
            --policy.path=outputs/train/2025-05-13/02-13-01_pusht_pi0/checkpoints/last/pretrained_model \
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

    - 独自の推論スクリプトを作成する場合

        ```sh
        python eval.py
        ```
