# Cosmos Predict を使用して（物理法則が考慮されたフォトリアリスティックな）合成データ生成を行なう

## 方法

1. Ampere 世代以降（RTX3000番台以降 or A100 など）の　GPU サーバーを準備する

    > NVIDIA Cosmos の世界基盤モデルは、内部で flash-attention を使用しているので、Ampere 世代以降の GPU でしか動かないことに注意

1. Hugging Face にログインする

    ```bash
    huggingface-cli login
    ```

1. 以下のサイトから Llama Guard を許可する

    https://huggingface.co/meta-llama/Llama-Guard-3-8B

1. Cosmos Predict 2 のレポジトリを clone する

    ```bash
    git clone https://github.com/nvidia-cosmos/cosmos-predict2
    cd cosmos-predict2
    ```

1. Cosmos Predict 2 をインストールする

    - Docker でインストールする場合
        ```bash
        # build docer image
        docker build -t ai-robotics-exercises-cosmos-predict2 -f Dockerfile .

        # run container
        docker run --gpus all -it --rm \
        -v ${PWD}:/workspace \
        -v ../datasets:/workspace/datasets \
        -v ../checkpoints:/workspace/checkpoints \
        ai-robotics-exercises-cosmos-predict2
        ```

        コンテナ接続後、以下のコマンドで動作テストする

        ```bash
        python /workspace/scripts/test_environment.py
        ```


1. 世界基盤モデル（WFM）の学習済みモデルをダウンロードする

    - text-to-image 系のモデル
        - `Cosmos-Predict2-2B-Text2Image`<br>
            ```bash
            python -m scripts.download_checkpoints --model_types text2image --model_sizes 2B
            ```

    - video-to-world 系のモデル

        - `Cosmos-Predict2-2B-Video2World`<br>
            ```bash
            python -m scripts.download_checkpoints --model_types video2world --model_sizes 2B
            ```

    - その他？

        - `Cosmos-Predict2-2B-Sample-Action-Conditioned`<br>
            ```bash
            python -m scripts.download_checkpoints --model_types sample_action_conditioned
            ```

        - `Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1`<br>
            ```bash
            python -m scripts.download_checkpoints --model_types sample_gr00t_dreams_gr1
            ```

1. text-to-image の推論を行なう

    ```bash
    PROMPT="A well-worn broom sweeps across a dusty wooden floor, its bristles gathering crumbs and flecks of debris in swift, rhythmic strokes. Dust motes dance in the sunbeams filtering through the window, glowing momentarily before settling. The quiet swish of straw brushing wood is interrupted only by the occasional creak of old floorboards. With each pass, the floor grows cleaner, restoring a sense of quiet order to the humble room."
    # Run text2image generation
    python -m examples.text2image \
        --prompt "${PROMPT}" \
        --model_size 2B \
        --save_path output/text2image_2b.jpg
    ```

1. video-to-world の推論を行なう

    ```bash
    ```

1. text-to-world の推論を行なう

    ```bash
    ```

