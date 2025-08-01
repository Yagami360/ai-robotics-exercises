# Isaac Sim & Lab のシミュレーター環境上で遠隔操作（Teleoperation）により、｛LeRobot SO-ARMS ロボット x 特定タスク｝の少数学習用データセットを作成する

## 方法

1. 強化学習＆模倣学習用の IK 対応版の環境を作成する

    キーボード経由での IK 操作ができるように、IK に対応した強化学習＆模倣学習用の環境を作成する

    [lerobot_so101/env_rl_so101.py](lerobot_so101/env_rl_so101_ik.py)

1. 作成した環境を Gymnasium 環境として登録

    環境を Gymnasium に登録して `gym.make()` で作成できるようにする

    [lerobot_so101/\_\_init\_\_.py](lerobot_so101/__init__.py)

1. 手動遠隔操作（Teleoperation）のスクリプトを作成する

    [teleop_so101.py](teleop_so101.py)

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

1. 自作した手動遠隔操作（Teleoperation）のスクリプトを実行し、シミュレーターを起動する

    ```bash
    # VNCサーバーを使用する場合
    export DISPLAY=:1

    python teleop_so101.py
    ```

    以下のキーボード操作でロボットを遠隔操作できる

    ```bash
    ```
