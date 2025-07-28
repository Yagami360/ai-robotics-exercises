# Isaac GR00T blueprint （Isaac Lab Mimic + Cosmos Transfer）を使用して学習用データセット作成とデータ拡張を行なう


## 方法

1. X11サーバーへのローカル接続を許可する

    ローカル環境（Mac）で以下のコマンドを実行し、

    ```bash
    xhost +local:
    ```

    > 既に実行済みの場合は、XQuart を終了して再度実行

1. Ampire 世代以降の GPU 環境を準備する

    Cosmos での推論を行うため、A100などのAmpire 世代以降の GPU 環境を準備する

    GPU メモリは 50GB 以上必要

1. ポートを開放する

    - VNC サーバーのための `5901` ポート開放

    - notebook のための `8888` ポートを開放

1. Isaac GR00T blueprint（synthetic-manipulation-motion-generation）のレポジトリを clone する

    ```bash
    git clone https://github.com/NVIDIA-Omniverse-blueprints/synthetic-manipulation-motion-generation

    cd synthetic-manipulation-motion-generation
    ```

1. VNC サーバを起動する

    ```bash
    vncserver ${DISPLAY} \
        -geometry 1280x720 \
        -depth 16 \
        -localhost no \
        -SecurityTypes VncAuth \
        -SendCutText=0 \
        -AcceptCutText=0 \
        -AcceptPointerEvents=1 \
        -AcceptKeyEvents=1
    ```

1. ディスプレイ設定

    ```bash
    export DISPLAY=:1
    ```

1. Isaac GR00T blueprint のコンテナを起動する

    ```bash
    docker compose -f docker-compose.yml up -d
    ```

1. notebook をブラウザで開く

    ```bash
    open http://localhost:8888/lab/tree/generate_dataset.ipynb
    ```

    VMインスタンス上からではなく、ブラウザ画面に直接 URL を貼り付けてアクセス

1. ブラウザ上から notebook を実行する

    処理の流れは、以下のようになる

    1. xxx
