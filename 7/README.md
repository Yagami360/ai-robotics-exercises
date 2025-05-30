# Issac Sim & Issac Labs の空シミュレーション環境を起動する

## X11フォワーディングを使用して Ubuntu サーバーなどの非GUI環境で動かす場合

1. GPU ありの Ubuntu 22.04 インスタンス環境を準備する

    GPU ありの Ubuntu 22.04 インスタンス環境を準備する

1. ローカル環境上で X11フォワーディングを有効にする

    - Mac の場合

        1. XQuartz をインストールする

            ```bash
            brew install --cask xquartz
            ```
            > XQuartz: macOS上でX Window Systemを実現するソフトウェア

        2. Mac を再起動する

        3. XQuartz を起動する

            ```bash
            open -a XQuartz
            ```
    
        4. （オプション）ローカル環境上のディスプレイ環境変数が設定されているか確認する

            ```bash
            yusukesakai@YusukenoMacBook-Pro ~ % echo $DISPLAY
            /private/tmp/com.apple.launchd.Elrhpzr2Yd/org.xquartz:0
            ```

1. X11フォワーディングあり（`-X` オプション）で サーバーに ssh 接続する
    ```bash
    ssh -X ${USER_NAME}@${SERVER_IP}
    ```
    - `-X`: X11 フォワーディングを許可

1. サーバー上のディスプレイ環境変数が設定されているか確認する

    ```bash
    echo $DISPLAY
    localhost:10.0
    ```

    うまく X11 フォワーディングができていない場合は、`DISPLAY` 環境変数に何も設定されない挙動になる

1. （オプション）X11 フォワーディングの動作テストする
    ```bash
    # x11-apps をインストール
    sudo apt update
    sudo apt install -y x11-apps

    # X11 フォワーディングができているかの動作テスト
    xclock
    ```

    コマンド実行後に、以下のような GUI がローカル環境上に表示されれば X11 フォワーディングが正常に動作している

    <img width="500" alt="Image" src="https://github.com/user-attachments/assets/3881646f-6d4e-4a8f-a0d0-5424855188dc" />

1. Issac Labs のレポジトリをクローンする

    ```bash
    git clone https://github.com/isaac-sim/IsaacLab
    cd IsaacLab
    ```

1. Issac Labs のコンテナを起動する

    ```bash
    # Issac Labs のコンテナを起動する
    ./docker/container.py start
    [INFO] Using container profile: base
    [INFO] X11 forwarding from the Isaac Lab container is disabled by default.
    [INFO] It will fail if there is no display, or this script is being run via ssh without proper configuration.
    Would you like to enable it? (y/N) y
    ```

    Ubuntu サーバーなどの GUI がない環境では、上記コマンド実行時に `y` を入力して、X11 フォワーディングを有効にする必要がある

    一度上記コマンドを実行した後は、`docker/.container.cfg` ファイルが自動的に作成されるので、後で X11 フォワーディングの有効無効を変更したい場合は、コンフィグファイル（`docker/.container.cfg` ファイル）の `x11_forwarding_enabled` を直接変更すれば良い

    - IsaacLab/docker/.container.cfg
        ```
        [X11]
        x11_forwarding_enabled = 1
        ```

1. Issac Labs のコンテナに接続する
    ```bash
    # Enter the container
    # We pass 'base' explicitly, but if we hadn't it would default to 'base'
    ./docker/container.py enter base
    ```

    コンテナ接続後に、`isaaclab` コマンド等が利用できる

    ```bash
    (base) sakai@sakai-gpu-dev:~/personal-repositories/ai-robotics-exercises/IsaacLab$ ./docker/container.py enter base
    [INFO] Using container profile: base
    [INFO] X11 Forwarding is disabled from the settings in '.container.cfg'
    [INFO] X11 forwarding is disabled. No action taken.
    [INFO] Entering the existing 'isaac-lab-base' container in a bash session...

    root@sakai-gpu-dev:/workspace/isaaclab# 
    ```
    ```bash
    root@sakai-gpu-dev:/workspace/isaaclab# isaaclab
    [Error] No arguments provided.                                                                             

    usage: isaaclab.sh [-h] [-i] [-f] [-p] [-s] [-t] [-o] [-v] [-d] [-n] [-c] -- Utility to manage Isaac Lab.

    optional arguments:
        -h, --help           Display the help content.
        -i, --install [LIB]  Install the extensions inside Isaac Lab and learning frameworks as extra dependencies. Default is 'all'.
        -f, --format         Run pre-commit to format the code and check lints.
        -p, --python         Run the python executable provided by Isaac Sim or virtual environment (if active).
        -s, --sim            Run the simulator executable (isaac-sim.sh) provided by Isaac Sim.
        -t, --test           Run all python unittest tests.
        -o, --docker         Run the docker container helper script (docker/container.sh).
        -v, --vscode         Generate the VSCode settings file from template.
        -d, --docs           Build the documentation from source using sphinx.
        -n, --new            Create a new external project or internal task from template.
        -c, --conda [NAME]   Create the conda environment for Isaac Lab. Default name is 'env_isaaclab'.
    ```

1. チュートリアルのサンプルコードを実行する
    Issac Labs のコンテナ内で、以下のコマンドを実行する

    ```bash
    python scripts/tutorials/00_sim/create_empty.py
    ```

    - `create_empty.py` の中身
        ```python
        # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
        # All rights reserved.
        #
        # SPDX-License-Identifier: BSD-3-Clause

        """This script demonstrates how to create a simple stage in Isaac Sim.

        .. code-block:: bash

            # Usage
            ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

        """

        """Launch Isaac Sim Simulator first."""


        import argparse

        from isaaclab.app import AppLauncher

        # create argparser
        parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
        # append AppLauncher cli args
        AppLauncher.add_app_launcher_args(parser)
        # parse the arguments
        args_cli = parser.parse_args()
        # launch omniverse app
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

        """Rest everything follows."""

        from isaaclab.sim import SimulationCfg, SimulationContext


        def main():
            """Main function."""

            # Initialize the simulation context
            sim_cfg = SimulationCfg(dt=0.01)
            sim = SimulationContext(sim_cfg)
            # Set main camera
            sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

            # Play the simulator
            sim.reset()
            # Now we are ready!
            print("[INFO]: Setup complete...")

            # Simulate physics
            while simulation_app.is_running():
                # perform step
                sim.step()


        if __name__ == "__main__":
            # run the main function
            main()
            # close sim app
            simulation_app.close()
        ```

    以下のような GUI が X11 フォワーディング経由でローカル環境上に表示されれば成功

    <img width="1200" alt="Image" src="https://github.com/user-attachments/assets/a68e47fd-311b-4fd0-ab0c-c3271666738b" />

1. Issac Labs のコンテナを停止する
    ```bash
    # stop the container
    ./docker/container.py stop
    ```

<!--
## docker を使用しない場合

1. Ubuntu 22.04 + CUDA 12.4 の GPU インスタンス環境を準備する

1. glibc のバージョンが `2.34` 以上になっているか確認する

    以下コマンドで glibc のバージョンを確認する
    ```bash
    ldd --version
    ```

    > glibc: GNUシステム用の標準Cライブラリ

    > Ubuntu 20.04 や Debian 11 の場合は、glibc バージョンはデフォルトで `2.31` になっているので注意

    バージョンが `2.34` 未満の場合は、Ubuntu や Debian のバージョンを更新する

1. Issac Sim をインストールする

    - conda を使用する場合

        https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html

        ```bash
        conda create -n isaac-labs python=3.10
        conda activate isaac-labs

        pip install --upgrade pip
        pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
        pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
        ```

        > glibc のバージョンが `2.34` 未満の場合は、`pip install isaacsim` の部分でエラーが発生するので注意
        > ```bash
        > RuntimeError: Didn't find wheel for isaacsim 4.5.0.0
        > ```

1. Issac Sim がインストールできたことを確認する

    ```bash
    isaacsim
    ```

1. Issac Labs のレポジトリをクローンする

    ```bash
    git clone https://github.com/isaac-sim/IsaacLab
    cd IsaacLab
    ```

1. Issac Labs をインストールする

    ```bash
    sudo apt install cmake build-essential
    ./isaaclab.sh --install
    ```

1. チュートリアルのサンプルコードを実行する
    Issac Labs のコンテナ内で、以下のコマンドを実行する

    ```bash
    python scripts/tutorials/00_sim/create_empty.py
    ```

    以下のような GUI が起動されれば成功

    <img width="771" alt="Image" src="https://github.com/user-attachments/assets/178dd060-677a-4d05-9720-f309d0de6de0" />

    > Ubuntu サーバーなどの GUI がない環境では、表示されないので注意

-->

## 参考サイト

- https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
- https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html#container-installation
