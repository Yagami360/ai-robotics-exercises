# LeRobot の事前学習済み Isaac-GR00T モデルを Issac Labs のシミュレーター環境で推論する

1. Isaac-GR00T をインストールする

1. pinocchio をインストールする
    `[Isaac-PickPlace-GR1T2-Abs-v0](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place/pickplace_gr1t2_env_cfg.py)` の環境では、pinocchio を使用しているので、インストールする

    ```bash
    conda install -c conda-forge pinocchio -y
    ```
    > pinocchio: ロボットの運動学と動力学を計算するためのライブラリ

    pinocchio インストール後に Issac Lab の numpy バージョンと不整合が発生した場合は、以下のコマンドも実行する

    ```bash
    conda install numpy=1.25.0 -y
    ```

1. 利用可能な環境のリストを確認する

    ```bash
    python scripts/environments/list_envs.py
    ```

1. xxx