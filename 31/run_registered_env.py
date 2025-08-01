import argparse
import torch
import gymnasium as gym

# ------------------------------------------------------------
# シミュレーターアプリ作成
# ------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd_path", type=str, default="../assets/so101_new_calib_fix_articulation_root.usd", help="SO101ロボットのUSDファイルパス")
parser.add_argument("--num_envs", type=int, default=1, help="環境数")
parser.add_argument("--episode_length_s", type=float, default=10.0, help="エピソード長（秒）")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# カメラを有効化
args_cli.enable_cameras = True

# シミュレーター起動
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------------------------
# 強化学習環境定義
# NOTE: "ModuleNotFoundError: No module named 'isaacsim.core'" のエラーがでないように、
# IsaacSim 関連の import 文は AppLauncher の後に記載する必要がある
# ------------------------------------------------------------
import isaaclab_tasks  # noqa: F401 （全ての環境を登録するため）
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import lerobot_so101  # lerobot_so101パッケージをインポートして環境を登録


def update_env_cfg_from_args():
    """コマンドライン引数から環境設定を作成"""
    from lerobot_so101.env_rl_so101 import LeRobotSO101StackCubeRLEnvCfg

    # デフォルト設定を取得
    env_cfg = LeRobotSO101StackCubeRLEnvCfg()

    # コマンドライン引数で設定を上書き
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # ロボットUSDファイルパスを設定
    env_cfg.robot_usd_path = args_cli.usd_path

    # エピソード長を設定
    env_cfg.episode_length_s = args_cli.episode_length_s

    return env_cfg


def main():
    """メイン関数"""
    print(f"[INFO]: 登録された環境 'LeRobot-SO101-StackCube-v0' を使用してシミュレーションを開始")

    # 環境設定を引数で上書き
    env_cfg = update_env_cfg_from_args()

    # 登録された環境を作成
    env = gym.make("LeRobot-SO101-StackCube-v0", cfg=env_cfg)

    # 環境情報を出力
    print(f"[INFO]: 観測空間: {env.observation_space}")
    print(f"[INFO]: アクション空間: {env.action_space}")
    print(f"[INFO]: 環境数: {env_cfg.scene.num_envs}")

    # 環境をリセット
    observations, info = env.reset()
    print(f"[INFO]: 初期観測の形状: {observations['policy'].shape}")

    # シミュレーションループ
    step_count = 0
    max_steps = int(args_cli.episode_length_s * 60)

    while simulation_app.is_running() and step_count < max_steps:
        # ランダムアクションを生成
        actions = torch.tensor(env.action_space.sample(), dtype=torch.float32)

        # ステップを実行
        observations, rewards, terminated, truncated, info = env.step(actions)

        # 情報出力
        if step_count % 60 == 0:
            print(f"[INFO]: ステップ {step_count}")
            print(f"[INFO]: 報酬: {rewards[0].item():.3f}")
            print(f"[INFO]: 終了フラグ: terminated={terminated[0].item()}, truncated={truncated[0].item()}")
            if 'policy' in observations and observations['policy'].shape[-1] >= 6:
                joint_positions = observations['policy'][0, :6]
                print(f"[INFO]: 関節角度 [rad]: {[f'{pos:.3f}' for pos in joint_positions.cpu().tolist()]}")

        # エピソード終了時にリセット
        if terminated.any() or truncated.any():
            print(f"[INFO]: エピソード終了、環境をリセット")
            observations, info = env.reset()

        step_count += 1

    # 環境を閉じる
    env.close()
    print(f"[INFO]: シミュレーション終了")


if __name__ == "__main__":
    main()
    simulation_app.close()
