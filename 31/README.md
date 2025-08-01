# Isaac Sim & Lab のシミュレーター上で｛LeRobot SO-ARMS ロボット x 特定タスク｝の Gymnasium 環境を自作する

## 方法

1. 強化学習＆模倣学習用の環境の作成する
    Isaac Lab を使用した `ManagerBasedRLEnv` での環境のコードを作成する

    [lerobot_so101/env_rl_so101.py](lerobot_so101/env_rl_so101.py)

1. 作成した環境を Gymnasium 環境として登録する

    環境を Gymnasium に登録して `gym.make()` で作成できるようにする

    [lerobot_so101/\_\_init\_\_.py](lerobot_so101/__init__.py)

1. 登録された環境を使用するスクリプトを作成する

    [run_registered_env.py](run_registered_env.py)

    - 登録された環境の使用例<br>
        ```python
        import gymnasium as gym
        import sys
        import os

        # 環境登録モジュールをインポート
        sys.path.append('path/to/31')
        import lerobot_so101  # 環境を登録

        # 環境設定を作成
        from lerobot_so101.env_rl_so101 import LeRobotSO101StackCubeRLEnvCfg
        env_cfg = LeRobotSO101StackCubeRLEnvCfg()

        # 登録された環境を作成
        env = gym.make("LeRobot-SO101-StackCube-v0", cfg=env_cfg)

        # 環境を使用
        observations, info = env.reset()
        action = env.action_space.sample()
        observations, rewards, terminated, truncated, info = env.step(action)
        ```


1. シミュレーターを起動する

    ```bash
    # VNCサーバーを使用する場合
    export DISPLAY=:1

    python run_registered_env.py
    ```

## 参考サイト

- https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_rl_env.html
- https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/register_rl_env_gym.html
