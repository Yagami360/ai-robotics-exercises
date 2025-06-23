# Genesis のシミュレーション環境を起動する

## 使用方法

1. PyTorch をインストールする
    ```bash
    # コマンド例
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
    ```

1. Genesis をインストールする
    ```bash
    pip install genesis-world
    ```

1. （オプション）libigl を再インストールする

    シミュレーターのコード内にて `ValueError: too many values to unpack (expected 3)` のエラーが発生する場合は、libigl を再インストールする

    ```bash
    pip uninstall libigl
    pip install "libigl==2.5.1"
    ```

    https://github.com/Genesis-Embodied-AI/Genesis/issues/1225

1. （オプション）VNC サーバーを起動する

    1. （VMインスタンスの場合）VNC 用のファイアウォールルールを作成し 5901 ポートを開放する

        ```bash
        gcloud compute firewall-rules create allow-vnc \
            --allow tcp:5901 \
            --source-ranges ${LOCAL_IP_ADDRESS}/32
        ```

        その後、作成したファイアウォールのネットワークタグを VM インスタンスに付与する

    1. Ubutn サーバー側に VNCサーバー環境パッケージをインストールする

        ```bash
        sudo apt update
        sudo apt install -y tigervnc-standalone-server
        sudo apt install -y xfce4 xfce4-goodies
        ```

    1. VNC 設定ファイルを作成する

        ```bash
        cat > ~/.vnc/xstartup << 'EOF'
        #!/bin/bash
        unset SESSION_MANAGER
        unset DBUS_SESSION_BUS_ADDRESS
        export XKL_XMODMAP_DISABLE=1

        # マウスとキーボードの設定
        xsetroot -solid grey
        xrdb $HOME/.Xresources 2>/dev/null || true

        # VNC設定
        vncconfig -iconic &

        # デスクトップ環境を起動
        if command -v startxfce4 >/dev/null 2>&1; then
            exec startxfce4
        else
            exec xterm -geometry 80x24+10+10 -ls -title "$VNCDESKTOP Desktop"
        fi
        EOF

        chmod +x ~/.vnc/xstartup
        ```

    1. VNC サーバーを起動する

        ```bash
        export DISPLAY=:1
        vncserver -kill :1
        vncserver :1 -geometry 1024x768 -depth 24 -localhost no -SecurityTypes VncAuth -SendCutText=0 -AcceptCutText=0 -AcceptPointerEvents=1 -AcceptKeyEvents=1
        ```
        View-onlyパスワードは「n」で拒否する。画面を見ることはできるが、マウスやキーボードで操作できない制限付きアクセス用のパスワードのため

    1. （Macの場合）画面共有アプリを起動し `${VM_INSTANCE_EXTERNAL_IP}:5901` を設定する

        <img width="300" alt="Image" src="https://github.com/user-attachments/assets/43050e48-505b-48f1-afa2-4a185fa17265" />

1. xxx


## 参考サイト

- https://genesis-world.readthedocs.io/en/latest/user_guide/index.html