# MuJoCo のシミュレーション環境を起動する

MuJoCo は、Googleの Deepmind 社の３Dシミュレーション環境（物理シミュレーション含む）。

Apache License 2.0で公開されており、商用・非商用で利用可能

## VNC を使用して Ubuntu サーバーなどの非GUI環境で動かす場合

1. VNC サーバーを起動する

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


1. MuJoCo のインストール

    - conda でインストールする場合
        ```bash
        conda create -n mujoco python=3.11
        conda activate mujoco

        conda install -c conda-forge mujoco
        conda install -c conda-forge gymnasium[mujoco]
        ```

1. mujoco で公開されているロボットのオブジェクトを参照するために mojoco のレポジトリを clone する

    ```bash
    git clone https://github.com/google-deepmind/mujoco
    ```

1. MuJoCo のシミュレーター起動コードを作成する

    - `scene.py`

        ```python
        import mujoco
        from mujoco.viewer import launch

        # load object
        model = mujoco.MjModel.from_xml_path('../mujoco/model/humanoid/humanoid.xml')
        data = mujoco.MjData(model)

        # Run simulater
        launch(model, data)
        ```

1. MuJoCo のシミュレーターを起動する

    ```bash
    export DISPLAY=:1

    conda activate mujoco
    python scene.py
    ```

    起動に成功すると、以下のようなシミュレーターが表示される

    <img width="1000" height="766" alt="Image" src="https://github.com/user-attachments/assets/3a3dd0fd-aa42-4196-9255-ff78b2dac357" />