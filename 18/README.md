# Isaac Sim & Lab のシミュレーター環境上で遠隔操作（Teleoperation）により片腕マニピュレーターロボット（Franka）を操作しながら学習用データセットを作成する

1. VNC サーバを起動する

1. ディスプレイ設定
    ```bash
    export DISPLAY=:1
    ```

1. 片腕ロボット（Franka）のキーボードでの遠隔操作と学習用データセット作成用のスクリプトを実行する<br>

    - キーボードで操作する場合
        ```bash
        ./isaaclab.sh -p scripts/tools/record_demos.py \
            --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
            --teleop_device keyboard \
            --dataset_file ../datasets/teleop_franka/dataset.hdf5 \
            --num_demos 10
        ```

        <img width="1000" height="745" alt="Image" src="https://github.com/user-attachments/assets/a1f125a9-469a-4c20-8bda-5b11a7ddcd2e" />

        シミュレーター起動後、以下のキーボード入力でロボット操作できるので、`--num_demos` の回数分の成功タスク（今回のサンプリの場合は、物を掴むタスク）をデモンストレーションする

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

        デモ中にミスした場合は、`R` キーでキャンセルする。

        `--dataset_file` で指定したファイルに、HDF5 フォーマットで学習用データセットが保存される。（ `--num_demos` の回数分の成功タスクのデモを行わないと空データになることに注意）

        なおこの方法で作成した学習用データセットは、https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Mimic/dataset.hdf5 からダウンロード可能


    - SpaceMouse というデバイスで操作する場合<br>
        xxx

    - XR（拡張現実）デバイスで操作する場合<br>
        xxx

1. 学習用データセットの内容に従ってロボットを動かす（リプレイする）

    - キーボードで操作する場合

        ```bash
        ./isaaclab.sh -p scripts/tools/replay_demos.py \
            --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
            --dataset_file ../datasets/teleop_franka/dataset.hdf5
        ```

        https://github.com/user-attachments/assets/c8198242-a9d2-4b88-a52f-39a25c654358

## 参考サイト

- https://isaac-sim.github.io/IsaacLab/main/source/overview/teleop_imitation.html
