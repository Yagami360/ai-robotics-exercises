# Cosmos Transfer を使用してデータ拡張を行なう

## 方法

1. Ampere 世代以降（RTX3000番台以降 or A100 など）の　GPU サーバーを準備する

    > NVIDIA Cosmos Transfer の世界基盤モデルは、内部で flash-attention を使用しているので、Ampere 世代以降の GPU でしか動かないことに注意

    > GPUメモリ40GB, CPU メモリ40GB程度必要

1. レポジトリを clone する

    ```bash
    git clone https://github.com/nvidia-cosmos/cosmos-transfer1
    cd cosmos-transfer1
    ```

1. インストールする

    - docker を使用する場合

        ```bash
        docker build -f Dockerfile . -t nvcr.io/$USER/cosmos-transfer1:latest
        ```

1. コンテナに接続する<br>

    ```bash
    docker run -it \
        -v $(PWD)/cosmos-transfer1:/workspace \
        -v $(PWD)/cosmos-transfer1/checkpoints:/workspace/checkpoints \
        --gpus all \
        nvcr.io/$USER/cosmos-transfer1:latest
    ```

1. Hugging Face にログインする<br>

    コンテナ内で Hugging Face にログインする

    ```bash
    huggingface-cli login
    ```

1. 以下のサイトから各 Hugging Face 上モデルのアクセス権限を付与する<br>

    - https://huggingface.co/meta-llama/Llama-Guard-3-8B
    - https://huggingface.co/nvidia/Cosmos-Transfer1-7B
    - https://huggingface.co/nvidia/Cosmos-Tokenize1-CV8x8x8-720p

1. 学習済みモデルをダウンロードする<br>

    コンテナ内で以下のコマンドを実行する

    ```bash
    PYTHONPATH=$(pwd) python scripts/download_checkpoints.py --output_dir checkpoints/
    ```

    > 全てのモデルで合計300GB程度になるので、ディスクの空き容量に注意が必要

1. セグメンテーション画像リストを作成する

    - `segmentation` ディレクトリに各動画フレームのセグメンテーション画像リストを配置する

        - フレーム１

            <img width="500" alt="Image" src="https://github.com/user-attachments/assets/a06236b6-1b04-4c54-aab1-cf2c3764e691" />

        - フレーム２

            <img width="500" alt="Image" src="https://github.com/user-attachments/assets/682d10cb-fff9-4ed3-91c9-5b8d25bb27af" />

        ...

    - `segmentation_label` ディレクトリ以下に各動画フレームのセグメンテーションラベル定義の json ファイルを配置する

        - フレーム１

            ```json
            {
                "(29, 0, 0, 255)": {
                    "class": "gripper0_right_r_palm_vis"
                },
                "(31, 0, 0, 255)": {
                    "class": "gripper0_right_R_thumb_proximal_base_link_vis"
                },
                "(33, 0, 0, 255)": {
                    "class": "gripper0_right_R_thumb_proximal_link_vis"
                },
                ...
            }
            ```

        - フレーム２

            ```json
            {
                "(29, 0, 0, 255)": {
                    "class": "gripper0_right_r_palm_vis"
                },
                "(31, 0, 0, 255)": {
                    "class": "gripper0_right_R_thumb_proximal_base_link_vis"
                },
                "(33, 0, 0, 255)": {
                    "class": "gripper0_right_R_thumb_proximal_link_vis"
                },
                ...
            }
            ```

        ...

1. セグメンテーション画像リストからロボットの前景と背景のマスク画像（ロボットとロボット以外のマスク画像）リストを取得する

    コンテナ内で以下のコマンドを実行する

    ```bash
    PYTHONPATH=$(pwd) python cosmos_transfer1/auxiliary/robot_augmentation/spatial_temporal_weight.py \
        --setting fg_vis_edge_bg_seg \
        --robot-keywords world_robot gripper robot \
        --input-dir assets/robot_augmentation_example \
        --output-dir outputs/robot_augmentation_example
    ```
    - `--setting` : 
        - `fg_vis_edge_bg_seg`: RBGとエッジの特徴でロボットを強調（ビズ：1.0前景、エッジ：1.0前景、セグ：1.0背景）
        - `fg_edge_bg_seg`: 

        > 公式README記載の `setting1`, `` から名前が変わっているので注意


    - 出力データ

        - フレーム１

            <img width="500" alt="Image" src="https://github.com/user-attachments/assets/ed5f6a12-f512-4e72-9efe-b5db70c2e7e9" />

        - フレーム２

            <img width="500" alt="Image" src="https://github.com/user-attachments/assets/f4284495-de99-46f1-a60b-8c9c39988d5a" />

        ...

1. Cosmos Transfer モデルで推論し、入力RGB動画のデータ拡張を行なう

    コンテナ内で以下のコマンドを実行する

    - 単一GPUで推論する場合<br>
        ```bash
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
        export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
        export NUM_GPU="${NUM_GPU:=1}"

        PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 \
        cosmos_transfer1/diffusion/inference/transfer.py \
            --checkpoint_dir $CHECKPOINT_DIR \
            --video_save_folder outputs/robot_example_spatial_temporal_setting1 \
            --controlnet_specs assets/robot_augmentation_example/example1/inference_cosmos_transfer1_robot_spatiotemporal_weights.json \
            --offload_text_encoder_model \
            --offload_guardrail_models \
            --num_gpus $NUM_GPU
        ```

    - 複数GPUで推論する場合<br>
        ```bash
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
        export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
        export NUM_GPU="${NUM_GPU:=4}"

        PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
            --checkpoint_dir $CHECKPOINT_DIR \
            --video_save_folder outputs/robot_example_spatial_temporal_setting1 \
            --controlnet_specs assets/robot_augmentation_example/example1/inference_cosmos_transfer1_robot_spatiotemporal_weights.json \
            --offload_text_encoder_model \
            --offload_guardrail_models \
            --num_gpus $NUM_GPU
        ```

    - 設定ファイル: `assets/inference_cosmos_transfer1_robot_spatiotemporal_weights.json`

        ```json
        {
            "prompt": "a robotic grasp an apple from the table and move it to another place.",
            "input_video_path" : "assets/robot_augmentation_example/example1/input_video.mp4",
            "vis": {
                "control_weight": "outputs/robot_augmentation_example/example1/vis_weights.pt"
            },
            "edge": {
                "control_weight": "outputs/robot_augmentation_example/example1/edge_weights.pt"
            },
            "depth": {
                "control_weight": "outputs/robot_augmentation_example/example1/depth_weights.pt"
            },
            "seg": {
                "input_control": "assets/robot_augmentation_example/example1/segmentation.mp4",
                "control_weight": "outputs/robot_augmentation_example/example1/seg_weights.pt"
            }
        }
        ```

    - 入力データ

        - プロンプト

            ```bash
            a robotic grasp an apple from the table and move it to another place.
            ```

        - RGB動画

            https://github.com/user-attachments/assets/d7d0637a-8d89-4b1d-a3cb-5112e1045361

        - セグメンテーション動画

            https://github.com/user-attachments/assets/32623730-354a-4ded-baad-cec5a8889f4b

    - 出力データ

        - 出力動画


            テーブルなどのオブジェクトの色や質感が変化した様々な出力動画が出力され、入力RGB動画のデータ拡張できている
