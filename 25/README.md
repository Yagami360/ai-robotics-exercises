# Cosmos Transfer を使用して（物理法則が考慮されたフォトリアリスティックな）動画生成を行なう

## 方法

1. Ampere 世代以降（RTX3000番台以降 or A100 など）の　GPU サーバーを準備する

    > NVIDIA Cosmos Transfer の世界基盤モデルは、内部で flash-attention を使用しているので、Ampere 世代以降の GPU でしか動かないことに注意

<!--
    > GPUメモリ50GB, CPU メモリ40GB程度必要
-->

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


1. Cosmos Transfer モデルで推論し、エッジ動画からの動画生成する<br>
    コンテナ内で以下のコマンドを実行する

    - 単一GPUで推論する場合<br>
        ```bash
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
        export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
        export NUM_GPU="${NUM_GPU:=1}"
        PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
            --checkpoint_dir $CHECKPOINT_DIR \
            --video_save_folder outputs/example1_single_control_edge \
            --controlnet_specs assets/inference_cosmos_transfer1_single_control_edge.json \
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
            --video_save_folder outputs/example1_single_control_edge \
            --controlnet_specs assets/inference_cosmos_transfer1_single_control_edge.json \
            --offload_text_encoder_model \
            --offload_guardrail_models \
            --num_gpus $NUM_GPU
        ```

    - 設定ファイル: `assets/inference_cosmos_transfer1_single_control_edge.json`
        ```json
        {
            "prompt": "The video is set in a modern, well-lit office environment with a sleek, minimalist design. The background features several people working at desks, indicating a busy workplace atmosphere. The main focus is on a robotic interaction at a counter. Two robotic arms, equipped with black gloves, are seen handling a red and white patterned coffee cup with a black lid. The arms are positioned in front of a woman who is standing on the opposite side of the counter. She is wearing a dark vest over a gray long-sleeve shirt and has long dark hair. The robotic arms are articulated and move with precision, suggesting advanced technology. \n\nAt the beginning, the robotic arms hold the coffee cup securely. As the video progresses, the woman reaches out with her right hand to take the cup. The interaction is smooth, with the robotic arms adjusting their grip to facilitate the handover. The woman's hand approaches the cup, and she grasps it confidently, lifting it from the robotic grip. The camera remains static throughout, focusing on the exchange between the robotic arms and the woman. The setting includes a white countertop with a container holding stir sticks and a potted plant, adding to the modern aesthetic. The video highlights the seamless integration of robotics in everyday tasks, emphasizing efficiency and precision in a contemporary office setting.",
            "input_video_path" : "assets/example1_input_video.mp4",
            "edge": {
                "control_weight": 1.0
            }
        }
        ```

    | Prompt |Input video | Output video |
    |-------------|-------------|--------------|
    |The video is set in a modern, well-lit office environment with a sleek, minimalist design. The background features ...| <video width="512" src="https://github.com/user-attachments/assets/d7955c4a-4676-42eb-8400-b3da8f653df4"></video> | <video width="512" src="https://github.com/user-attachments/assets/6b83e970-b5cb-4fc9-a183-3661268e5f9a"></video> |

1. Cosmos Transfer モデルで推論し、エッジ動画からの動画生成する深度マップ動画からの動画生成<br>
    コンテナ内で以下のコマンドを実行する

    ```bash
    export CUDA_VISIBLE_DEVICES=0
    export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
    PYTHONPATH=$(pwd) python cosmos_transfer1/diffusion/inference/transfer.py \
        --checkpoint_dir $CHECKPOINT_DIR \
        --video_save_folder outputs/example1_single_control_depth \
        --controlnet_specs assets/inference_cosmos_transfer1_single_control_depth.json \
        --offload_text_encoder_model
    ```

    - 入力動画

    - 出力動画


1. Cosmos Transfer モデルで推論し、セグメンテーション動画から動画生成する<br>
    コンテナ内で以下のコマンドを実行する

    ```bash
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
    export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
    export NUM_GPU="${NUM_GPU:=1}"
    PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
        --checkpoint_dir $CHECKPOINT_DIR \
        --video_save_folder outputs/example1_single_control_seg \
        --controlnet_specs assets/inference_cosmos_transfer1_single_control_seg.json \
        --offload_text_encoder_model \
        --offload_guardrail_models \
        --num_gpus $NUM_GPU
    ```

1. Cosmos Transfer モデルで推論し、超解像度での動画生成する<br>
    コンテナ内で以下のコマンドを実行する

    ```bash
    ```

1. Cosmos Transfer モデルで推論し、マルチモーダルでの動画生成する<br>
    コンテナ内で以下のコマンドを実行する

    マルチモーダル（深度マップ動画・セグメンテーション動画）

    ```bash
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
    export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
    export NUM_GPU="${NUM_GPU:=1}"
    PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
        --checkpoint_dir $CHECKPOINT_DIR \
        --video_save_folder outputs/example2_uniform_weights \
        --controlnet_specs assets/inference_cosmos_transfer1_uniform_weights.json \
        --offload_text_encoder_model \
        --offload_guardrail_models \
        --num_gpus $NUM_GPU
    ```

    - 設定ファイル: `assets/inference_cosmos_transfer1_uniform_weights.json`

        ```json
        {
            "prompt": "The video is set in a modern, well-lit office environment with a sleek, minimalist design. The background features several people working at desks, indicating a busy workplace atmosphere. The main focus is on a robotic interaction at a counter. Two robotic arms, equipped with black gloves, are seen handling a red and white patterned coffee cup with a black lid. The arms are positioned in front of a woman who is standing on the opposite side of the counter. She is wearing a dark vest over a gray long-sleeve shirt and has long dark hair. The robotic arms are articulated and move with precision, suggesting advanced technology. \n\nAt the beginning, the robotic arms hold the coffee cup securely. As the video progresses, the woman reaches out with her right hand to take the cup. The interaction is smooth, with the robotic arms adjusting their grip to facilitate the handover. The woman's hand approaches the cup, and she grasps it confidently, lifting it from the robotic grip. The camera remains static throughout, focusing on the exchange between the robotic arms and the woman. The setting includes a white countertop with a container holding stir sticks and a potted plant, adding to the modern aesthetic. The video highlights the seamless integration of robotics in everyday tasks, emphasizing efficiency and precision in a contemporary office setting.",
            "input_video_path" : "assets/example1_input_video.mp4",
            "vis": {
                "control_weight": 0.25
            },
            "edge": {
                "control_weight": 0.25
            },
            "depth": {
                "input_control": "assets/example1_depth.mp4",
                "control_weight": 0.25
            },
            "seg": {
                "input_control": "assets/example1_seg.mp4",
                "control_weight": 0.25
            }
        }
        ```