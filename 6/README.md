# LeRobot の事前学習済み Isaac-GR00T モデルに対してデモ用データセットで推論を行なう

## 使用方法

1. GPU インスタンス環境を準備する

    cuda 12.4 の GPU インスタンス環境を用意する。

    Isaac-GR00T は、flash-attention がサポートされている GPU（A100など）しか使えない点に注意。サポート外の GPU（T4, V100など）では以下のエラーが発生する
    ```bash
    RuntimeError: FlashAttention only supports Ampere GPUs or newer.
    ```

    > FlashAttention: Transformer モデルにおけるアテンション機構の計算を高速化し、メモリ効率を改善するアルゴリズム

1. Isaac-GR00T をインストールする

    - conda を使用する場合

        ```bash
        conda create -n gr00t python=3.10
        conda activate gr00t

        pip install --upgrade setuptools
        git clone https://github.com/NVIDIA/Isaac-GR00T
        cd Isaac-GR00T
        pip install -e .
        pip install --no-build-isolation flash-attn==2.7.1.post4
        ```

1. （オプション）flash-attention を無効にする

    A100 などの flash-attention がサポートされていない GPU（T4, V100など）の場合は、以下のコンフィグファイルを変更することで、flash-attention を無効にすることができる。

    - `Isaac-GR00T/gr00t/model/backbone/eagle2_hg_model/config.json`

        ```json
        {
            // ... 既存のコード ...
            "llm_config": {
                // ... 既存のコード ...
                "attn_implementation": "eager",  // "flash_attention_2" から "eager" に変更
                // ... 既存のコード ...
            },
            // ... 既存のコード ...
            "vision_config": {
                // ... 既存のコード ...
                "_attn_implementation": "eager"  // "flash_attention_2" から "eager" に変更
            }
        }
        ```

1. Isaac-GR00T の推論コードを実装する

    - [eval.py](./eval.py)

        ```python
        import argparse
        import os

        import gr00t
        import numpy as np
        import torch
        from gr00t.data.dataset import LeRobotSingleDataset
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.experiment.data_config import DATA_CONFIG_MAP
        from gr00t.model.policy import Gr00tPolicy

        if __name__ == "__main__":
            parser = argparse.ArgumentParser()
            parser.add_argument(
                "--dataset_path",
                type=str,
                default="../Isaac-GR00T/demo_data/robot_sim.PickNPlace",
            )
            parser.add_argument("--model_path", type=str, default="nvidia/GR00T-N1-2B")
            parser.add_argument("--seed", type=int, default=42)
            parser.add_argument("--gpu_id", type=int, default=0)
            args = parser.parse_args()
            for arg in vars(args):
                print(f"{arg}: {getattr(args, arg)}")

            if args.gpu_id < 0:
                device = "cpu"
            else:
                device = "cuda"
                os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

            # get configs
            data_config = DATA_CONFIG_MAP["gr1_arms_only"]
            print("data_config.modality_config():", data_config.modality_config())

            # Load dataset from local
            dataset = LeRobotSingleDataset(
                dataset_path=args.dataset_path,
                modality_configs=data_config.modality_config(),
                transforms=None,
                embodiment_tag=EmbodimentTag.GR1,
            )
            print("dataset[0]:", dataset[0])

            # Load pre-trained Isaac-GR00T model (policy) from HuggingFace LeRobot
            policy = Gr00tPolicy(
                model_path=args.model_path,
                modality_config=data_config.modality_config(),
                modality_transform=data_config.transform(),
                embodiment_tag=EmbodimentTag.GR1,
                device=device,
            )
            # print(policy.model)

            # inference
            with torch.inference_mode():
                # デモ用データの最初のデータで推論
                # NOTE: 恐らく、学習用データセットとして使用したデータと同じようなデータなので、当然ながら推論精度は高くなる
                action_chunk = policy.get_action(dataset[0])
                print("action_chunk:", action_chunk)
        ```

        ポイントは、以下の通り

        - flash-attention がサポートされている GPU（A100など）しか使えない

            flash-attention サポート外の GPU（T4, V100など）では以下のエラーが発生する
            ```bash
            RuntimeError: FlashAttention only supports Ampere GPUs or newer.
            ```

        - CPU はサポート外

        - デモ用のデータセット（`Isaac-GR00T/demo_data/robot_sim.PickNPlace`）を使用して推論を行う

            データセットの中身は、以下のような形式になっている

            ```python
            {
                'video.ego_view': array([[[[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        ...,
                        [0, 0, 0]]]], dtype=uint8),
                'state.left_arm': array([[-0.01147083,  0.12207967,  0.04229397, -2.1       , -0.01441445,
                        -0.03013532, -0.00384387]]),
                'state.right_arm': array([[ 6.74682520e-03, -9.05242648e-02,  8.14010333e-03,
                        -2.10000000e+00, -2.22802847e-02,  1.40373494e-02,
                        1.04679727e-03]]),
                'state.left_hand': array([[0.00067388, 0.00076318, 0.00084776, 0.00069391, 0.00075118,
                    0.00485883]]),
                'state.right_hand': array([[0.00187319, 0.00216486, 0.002383  , 0.00182536, 0.02100907,
                    0.01359221]]),
                'action.left_arm': array([[-1.07706680e-02,  1.04934344e-01,  4.15936080e-02,
                    -2.09762367e+00, -1.78164827e-02, -3.33876667e-02,
                    -9.85330611e-03],
                    ...,
                    [ 2.07126401e-02,  4.32274177e-02, -1.72182433e-02,
                    -2.07203352e+00,  1.92976598e-02, -2.92817179e-02,
                    -1.06464981e-02]]),
                'action.right_arm': array([[ 2.21627088e-03, -9.37728608e-02, -3.97849900e-02,
                    -2.06814219e+00, -8.56739215e-02,  5.79283621e-02,
                    -5.77253327e-02],
                    ...,
                    [ 5.03588613e-03, -6.39530345e-01, -8.39274535e-02,
                    -1.92642545e+00,  2.47624030e-01,  4.06113278e-01,
                    2.24670301e-01]]),
                'action.left_hand': array([[-1.5, -1.5, -1.5, -1.5, -3. ,  3. ],
                    ...,
                    [-1.5, -1.5, -1.5, -1.5, -3. ,  3. ]]),
                'action.right_hand': array([[-1.5, -1.5, -1.5, -1.5, -3. ,  3. ],
                    ...,
                    [-1.5, -1.5, -1.5, -1.5, -3. ,  3. ]]),
                'annotation.human.action.task_description': ['pick the pear from the counter and place it in the plate']
            }
            ```

            https://github.com/user-attachments/assets/70b908b3-36ce-43e7-9df4-cc5cff9559e0

        - デモ用データセットは、学習用データセットと同じようなデータセットになっていると思われるため、当然ながらこのデータで推論すると高い精度が出る

1. 推論を実行する

    ```bash
    python eval.py
    ```

    以下な次回時間ステップにおけるロボットの最適行動ベクトルが出力される。
    今回のコード例では、シミュレーターを使用していないので、１ステップにおける行動ベクトルのみ出力するコードになっている点に注意

    ```python
    action_chunk: {
        'action.left_arm': array([[ 4.81812954e-02,  2.75385708e-01,  6.35790825e-02,
            -1.90729749e+00,  5.06091118e-03,  8.64368677e-02,
            1.70541644e-01],
            ...,
            [-2.77497768e-02,  3.04946452e-01, -5.54144382e-02,
            -1.79736257e+00,  1.15072966e-01,  2.42852449e-01,
            1.73459411e-01]]),
        'action.right_arm': array([[ 4.48875427e-02, -1.52109146e-01,  4.42804098e-02,
            -2.06703615e+00,  2.35692501e-01,  5.76206446e-02,
            5.29288054e-02],
            ...,
            [ 3.28695774e-02, -4.15798664e-01, -9.25958157e-03,
            -1.77494442e+00,  2.32261896e-01,  1.49469614e-01,
            -9.03737545e-03]]),
        'action.left_hand': array([[-0.03844249, -0.05444551, -0.03831387, -0.03725791, -0.06225586,
            0.05859375],
            ...,
            [-0.02465153, -0.04116249, -0.03471994, -0.0216043 , -0.02069092,
            0.02929688]]),
        'action.right_hand': array([[-1.48800468, -1.48742819, -1.48929691, -1.47856259, -2.953125  ,
            2.86523438],
            ...,
            [-1.5       , -1.5       , -1.5       , -1.4892813 , -2.98828125,
            2.95898438]])
    }
    ```
