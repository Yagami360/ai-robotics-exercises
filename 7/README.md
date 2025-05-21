# Issac Labs をインストールして空のシミュレーション環境を起動する

## docker を使用する場合

1. GPU インスタンス環境を準備する

    GPU インスタンス環境を準備する

<!--
1. Issac Sim をインストールする

    https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html#container-installation

    ```bash
    docker pull nvcr.io/nvidia/isaac-sim:4.5.0
    ```
    ```bash
    docker run --name isaac-sim --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
        -e "PRIVACY_CONSENT=Y" \
        -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
        -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
        -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
        -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
        -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
        -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
        -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
        -v ~/docker/isaac-sim/documents:/root/Documents:rw \
        nvcr.io/nvidia/isaac-sim:4.5.0
    ```
-->

1. Issac Labs のレポジトリをクローンする

    ```bash
    git clone https://github.com/isaac-sim/IsaacLab
    cd IsaacLab
    ```

1. （オプション）非 GUI 環境上から動かす場合
    1. x11 フォワーディングを有効にする

        docker コンテナ内から GUI を表示できるようコンフィグファイル（`docker/.container.cfg` ファイル）の `x11_forwarding_enabled` を `0` -> `1` に変更する

        - IsaacLab/docker/.container.cfg
            ```
            [X11]
            x11_forwarding_enabled = 1
            ```

    1. x11 関連パッケージをインストールする

        `x11_forwarding_enabled = 1` に変更した場合は、インストールする必要がある

        ```bash
        sudo apt-get update
        sudo apt-get install -y x11-apps xauth
        ```

    1. Xサーバー（X Window System）を起動する

        ```bash
        export DISPLAY=0
        ```
        ```bash

        ```

1. Issac Labs をインストールする

    ```bash
    # Launch the container in detached mode
    # We don't pass an image extension arg, so it defaults to 'base'
    ./docker/container.py start

    # If we want to add .env or .yaml files to customize our compose config,
    # we can simply specify them in the same manner as the compose cli
    # ./docker/container.py start --file my-compose.yaml --env-file .env.my-vars
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
    cd scripts/tutorials/00_sim
    python create_empty.py
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

1. Issac Labs のコンテナを停止する
    ```bash
    # stop the container
    ./docker/container.py stop
    ```

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

<!--
    - バイナリからインストールする場合

        https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html

        ```bash
        cd ~

        # Download Isaac Sim 4.5.0 binary
        curl -O https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone%404.5.0-rc.36%2Brelease.19112.f59b3005.gl.linux-x86_64.release.zip

        # Unzip the binary
        mkdir -p isaacsim
        unzip isaac-sim-standalone%404.5.0-rc.36%2Brelease.19112.f59b3005.gl.linux-x86_64.release.zip -d isaacsim

        # Install Isaac Sim
        cd isaacsim
        ./post_install.sh
        ./isaac-sim.selector.sh
        ```

        > Ubuntu サーバーなどの GUI がない環境ではインストールできない？
-->

1. Issac Sim がインストールできたことを確認する

    ```bash
    isaacsim
    ```

1. Issac Labs をインストールする

    ```bash
    git clone git@github.com:isaac-sim/IsaacLab.git

    sudo apt install cmake build-essential
    ./isaaclab.sh --install
    ```

## 参考サイト

- https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
- https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html#container-installation
