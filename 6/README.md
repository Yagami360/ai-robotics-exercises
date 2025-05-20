# LeRobot の Isaac-GR00T モデルを gymnasium のシミュレーター環境上で動かす

## 使用方法

1. GPU インスタンス環境を準備する

    cuda 12.4 の GPU インスタンス環境を用意する。

    Isaac-GR00T は、flash-attention がサポートされている GPU（A100など）しか使えない点に注意。サポート外の GPU（T4, V100など）では以下のエラーが発生する
    ```bash
    RuntimeError: FlashAttention only supports Ampere GPUs or newer.
    ```

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

1. Isaac-GR00T の推論コードを実装する

    [eval.py](./eval.py) を実装する。

    ポイントは、以下の通り

    - xxx

1. 推論を実行する

    ```bash
    python eval.py
    ```
