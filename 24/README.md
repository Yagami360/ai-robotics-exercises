# Cosmos Predict を使用して（物理法則が考慮されたフォトリアリスティックな）動画生成を行なう

## 方法

1. Ampere 世代以降（RTX3000番台以降 or A100 など）の　GPU サーバーを準備する

    > NVIDIA Cosmos の世界基盤モデルは、内部で flash-attention を使用しているので、Ampere 世代以降の GPU でしか動かないことに注意

    > GPUメモリ50GB, CPU メモリ40GB程度必要

1. NVIDIA の NGC にログインして SetUp 画面がら API キーを作成する

    https://ngc.nvidia.com/

1. NVIDIA NGC Container Registr にログインする

    ```bash
    docker login nvcr.io
    ```
    - ユーザー名: $oauthtoken
    - パスワード: 上記作成した API トークン

1. Cosmos Predict 2 のレポジトリを clone する

    ```bash
    git clone https://github.com/nvidia-cosmos/cosmos-predict2
    cd cosmos-predict2
    ```

1. Cosmos Predict 2 をインストールする

    - Docker でインストールする場合
        ```bash
        # 事前に NVIDIA NGC Container Registr にログインしておく必要あり
        docker pull nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.1

        # run container
        docker run --gpus all -it --rm \
        -v ${PWD}:/workspace \
        -v ${PWD}/datasets:/workspace/datasets \
        -v ${PWD}/checkpoints:/workspace/checkpoints \
        nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.1
        ```

        コンテナ接続後、以下のコマンドで動作テストする

        ```bash
        python /workspace/scripts/test_environment.py
        ```

1. Hugging Face にログインする

    コンテナ内で Hugging Face にログインする

    ```bash
    huggingface-cli login
    ```

1. 以下のサイトから各 Hugging Face 上モデルのアクセス権限を付与する

    - https://huggingface.co/meta-llama/Llama-Guard-3-8B
    - https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image
    - https://huggingface.co/nvidia/Cosmos-Guardrail1

1. 世界基盤モデル（WFM）の学習済みモデルをダウンロードする

    - text-to-image 系のモデル
        - `Cosmos-Predict2-2B-Text2Image`<br>
            ```bash
            python -m scripts.download_checkpoints --model_types text2image --model_sizes 2B
            ```

    - video-to-world & text-to-world 系のモデル

        - `Cosmos-Predict2-2B-Video2World`<br>
            ```bash
            python -m scripts.download_checkpoints --model_types video2world --model_sizes 2B --resolution 480 --fps 10
            ```

        - `Cosmos-Predict2-14B-Video2World`<br>
            ```bash
            python -m scripts.download_checkpoints --model_types video2world --model_sizes 14B --resolution 480 --fps 10
            ```

    - video-to-world のヒューマノイドロボット（GR1）特化のモデル

        - `Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1`<br>
            ```bash
            python -m scripts.download_checkpoints --model_types sample_gr00t_dreams_gr1
            ```

            何故か `nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1` のモデルがダウンロードされない
            ```bash
            root@e8f733c501a9:/workspace# python -m scripts.download_checkpoints --model_types sample_gr00t_dreams_gr1
            Files for google-t5/t5-11b already exist (MD5 not verified).
            ---------------------
            Files for nvidia/Cosmos-Guardrail1 already exist (MD5 not verified).
            ---------------------
            Files for meta-llama/Llama-Guard-3-8B already exist (MD5 not verified).
            ---------------------
            Checkpoint downloading done.
            ```

    - その他？

        - `Cosmos-Predict2-2B-Sample-Action-Conditioned`<br>
            ```bash
            python -m scripts.download_checkpoints --model_types sample_action_conditioned
            ```

1. text-to-image の推論を行なう

    ```bash
    PROMPT="A well-worn broom sweeps across a dusty wooden floor, its bristles gathering crumbs and flecks of debris in swift, rhythmic strokes. Dust motes dance in the sunbeams filtering through the window, glowing momentarily before settling. The quiet swish of straw brushing wood is interrupted only by the occasional creak of old floorboards. With each pass, the floor grows cleaner, restoring a sense of quiet order to the humble room."

    # Run text2image generation
    python -m examples.text2image \
        --prompt "${PROMPT}" \
        --model_size 2B \
        --offload_guardrail \
        --save_path output/text2image_2b.jpg
    ```

    以下のような入力テキストの内容に沿った画像が出力される

    ![Image](https://github.com/user-attachments/assets/5d5f7142-06e0-453a-9a21-c971403c71ca)

1. video-to-world の推論を行なう

    - 1GPUで推論する場合

        ```bash
        # Set the input prompt
        PROMPT="A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene."

        # Run video2world generation
        python -m examples.video2world \
            --model_size 2B \
            --resolution 480 \
            --fps 10 \
            --offload_guardrail \
            --offload_prompt_refiner \
            --input_path assets/video2world/input0.jpg \
            --num_conditional_frames 1 \
            --prompt "${PROMPT}" \
            --save_path output/video2world_2b.mp4
        ```

        > `--input_path` に画像ではなく動画ファイルを指定することで、video-to-world も可能？

    - 複数GPUで推論する場合

        ```bash
        # Set the input prompt
        PROMPT="A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene."

        # Run video2world generation with multi-gpus
        # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        torchrun --nproc_per_node=2 --master_port=12341 -m examples.video2world \
            --model_size 2B \
            --resolution 480 \
            --fps 10 \
            --num_gpus 2 \
            --offload_guardrail \
            --offload_prompt_refiner \
            --input_path assets/video2world/input0.jpg \
            --num_conditional_frames 1 \
            --prompt "${PROMPT}" \
            --save_path output/video2world_2b.mp4
        ```

    ｛テキスト・画像｝を入力として、以下のような動画が出力される

    | Input image | Output video |
    |-------------|--------------|
    | <image width="512" src="https://github.com/user-attachments/assets/fea3c2e3-8389-4bc9-9a84-b5bfebd7c953"></image> | <video width="512" src="https://github.com/user-attachments/assets/1925caef-2534-448d-b391-ca3098f79d02"></video> |

1. text-to-world の推論を行なう

    - 1GPUで推論する場合

        ```bash
        # Set the input prompt
        PROMPT="An autonomous welding robot arm operating inside a modern automotive factory, sparks flying as it welds a car frame with precision under bright overhead lights."

        # Run text2world generation
        python -m examples.text2world \
            --model_size 2B \
            --resolution 480 \
            --fps 10 \
            --prompt "${PROMPT}" \
            --offload_guardrail \
            --offload_prompt_refiner \
            --save_path output/text2world_2b.mp4
        ```

    - 複数GPUで推論する場合

        xxx

    以下のような動画が出力される

    xxx


1. ヒューマノイドロボット（GR1）特化モデルを使用して video-to-world での推論を行なう

    ```bash
    PROMPT="Use the right hand to pick up rubik\'s cube from from the bottom of the three-tiered wooden shelf to to the top of the three-tiered wooden shelf."

    python -m examples.video2world_gr00t \
        --model_size 14B \
        --disable_guardrail \
        --gr00t_variant gr1 \
        --prompt "${PROMPT}" \
        --input_path assets/sample_gr00t_dreams_gr1/sample.png \
        --prompt_prefix "" \
        --save_path output/generated_video_gr1.mp4
    ```


> VLAなどの模倣学習のロボティクスモデルでは、動画データを入力するのではなく、ロボットの状態やカメラ画像や観測データを入力するので、学習用データセットとしてロボットの状態やカメラ画像や観測データなどが必要。Cosmos では動画データを合成データとして生成するだけ？動画データだけでは VLA の学習用データセットとして利用できない
