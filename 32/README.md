# Isaac Sim & Lab のシミュレーター環境上で遠隔操作（Teleoperation）により、｛LeRobot SO-ARMS ロボット x 特定タスク｝の少数学習用データセットを作成する


Isaac Lab で提供している Teleoperation のスクリプト（`scripts/environments/teleoperation/teleop_se3_agent.py`）は、SO-ARMS ロボットに対応していないので、このスクリプトを参考にして自作する必要がある

> 具体的には、`--task` 引数に自作した環境（`LeRobot-SO101-StackCube-v0`）を設定し、環境へのパスを正しく設定して実行しても、ロボット関節点の次元数が合わずエラーが出る
> ```bash
> cd IsaacLab
> ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
>     --task LeRobot-SO101-StackCube-IK-Rel-v0 --num_envs 1 \
>     --teleop_device keyboard \
>     --enable_cameras
> ```
> ```bash
> ```

## 方法

### 公式スクリプトを参考にして自作

1. 強化学習＆模倣学習用の環境を作成する

    Isaac Lab を使用した `ManagerBasedRLEnv` での環境のコードを作成

    [lerobot_so101/env_rl_so101.py](lerobot_so101/env_rl_so101.py)

1. 作成した環境を Gymnasium 環境として登録

    環境を Gymnasium に登録して `gym.make()` で作成できるようにする

    [lerobot_so101/\_\_init\_\_.py](lerobot_so101/__init__.py)

1. 手動遠隔操作（Teleoperation）のスクリプトを作成する

    [teleop_se3_so101.py](teleop_se3_so101.py)

    > TODO: HDFデータセットへの書き込み処理を追加

1. 自作した手動遠隔操作（Teleoperation）のスクリプトを実行し、シミュレーターを起動する

    ```bash
    # VNCサーバーを使用する場合
    export DISPLAY=:1

    python teleop_se3_so101.py
    ```

    以下のキーボード操作でロボットを遠隔操作できる

    ```bash
    ============================================================
    キーボード操作ガイド (SE3 → 関節角度制御)
    ============================================================
    位置制御 (手先位置):
        W/S: 前後移動 → 肘関節 (elbow_flex)
        A/D: 左右移動 → 肩パン関節 (shoulder_pan)
        Q/E: 上下移動 → 肩リフト関節 (shoulder_lift)

    回転制御 (手先姿勢):
        Z/X: X軸回転 → 手首フレックス (wrist_flex)
        T/G: Y軸回転 → (未マップ)
        C/V: Z軸回転 → 手首ロール (wrist_roll)

    グリッパー制御:
        K: グリッパー開閉トグル (True=閉じる, False=開く)

    システム制御:
        R: 環境リセット
        L: Teleoperation有効/無効切り替え
        ESC: 終了
    ```

### leisaac を利用して自作する場合

1. leisaac をインストールする
    ```bash
    conda create -n leisaac python=3.10
    conda activate leisaac

    # Install cuda-toolkit
    conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

    # Install PyTorch
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

    # Install IsaacSim
    pip install --upgrade pip
    pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

    # Install IsaacLab
    git clone git@github.com:isaac-sim/IsaacLab.git
    sudo apt install cmake build-essential

    cd IsaacLab
    # fix isaaclab version for isaacsim4.5
    git checkout v2.1.0
    ./isaaclab.sh --install
    ```

1. 強化学習＆模倣学習用の環境を作成する

    Isaac Lab を使用した `ManagerBasedRLEnv` での環境のコードを作成

    [lerobot_so101/env_rl_so101.py](lerobot_so101/env_rl_so101.py)

1. 作成した環境を Gymnasium 環境として登録

    環境を Gymnasium に登録して `gym.make()` で作成できるようにする

    [lerobot_so101/\_\_init\_\_.py](lerobot_so101/__init__.py)

1. leisaac を利用した手動遠隔操作（Teleoperation）のスクリプトを作成する

    [teleop_se3_so101_leisaac.py](teleop_se3_so101_leisaac.py)

1. 自作した手動遠隔操作（Teleoperation）のスクリプトを実行し、シミュレーターを起動する

    ```bash
    # VNCサーバーを使用する場合
    export DISPLAY=:1

    python teleop_se3_so101_leisaac.py
    ```

    以下のキーボード操作でロボットを遠隔操作できる

    ```bash
    Keyboard Controller for SE(3).
        Keyboard name: Isaac Sim 4.5.0
        ----------------------------------------------
        Joint 1 (shoulder_pan):  Q/U
        Joint 2 (shoulder_lift): W/I
        Joint 3 (elbow_flex):    E/O
        Joint 4 (wrist_flex):    A/J
        Joint 5 (wrist_roll):    S/K
        Joint 6 (gripper):       D/L
        ----------------------------------------------
        Start Control: B
        Task Failed and Reset: R
        Task Success and Reset: N
        Control+C: quit

    [INFO]: メインループを開始します
    [INFO]: Bキーを押して制御を開始してください
    ```
