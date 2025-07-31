# Isaac Sim & Lab のシミュレーター上で｛LeRobot SO-ARMS ロボット x 特定タスク｝の環境を自作する

## 方法

1. Isaac Lab を使用した `ManagerBasedRLEnv` での環境のコードを作成する

    [env_rl_so101.py](env_rl_so101.py)

1. シミュレーターを起動する

    ```python
    # VNCサーバーを使用する場合
    export DISPLAY=:1

    python env_rl_so101.py
    ```

    シミュレーション起動後に、以下のような ｛LeRobot SO-ARMS ロボット x 特定タスク｝のための環境が表示される



## 参考サイト

- https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_rl_env.html
