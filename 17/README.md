# Issac Sim & Labs のシミュレーター環境上で遠隔操作（Teleoperation）により片腕ロボット（Franka）を操作する

1. VNC サーバを起動する

1. ディスプレイ設定
    ```bash
    export DISPLAY=:1
    ```

1. 片腕ロボット（Franka）のキーボードでの遠隔操作用のスクリプトを実行する<br>

    - キーボードで操作する場合
        ```bash
        ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
            --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --num_envs 1 \
            --teleop_device keyboard
        ```

        <img width="1000" height="745" alt="Image" src="https://github.com/user-attachments/assets/a1f125a9-469a-4c20-8bda-5b11a7ddcd2e" />

        シミュレーター起動後、以下のキーボード入力でロボット操作できる
        ```bash
        Keyboard Controller for SE(3): Se3Keyboard
            Reset all commands: R
            Toggle gripper (open/close): K
            Move arm along x-axis: W/S
            Move arm along y-axis: A/D
            Move arm along z-axis: Q/E
            Rotate arm along x-axis: Z/X
            Rotate arm along y-axis: T/G
            Rotate arm along z-axis: C/V
        ```

    - SpaceMouse というデバイスで操作する場合<br>
        xxx

    - XR（拡張現実）デバイスで操作する場合<br>
        xxx

## 参考サイト

- https://isaac-sim.github.io/IsaacLab/main/source/overview/teleop_imitation.html
