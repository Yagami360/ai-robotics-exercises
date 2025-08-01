"""IK対応SO101環境でteleoperation機能をテストするスクリプト"""

import argparse
import torch
import gymnasium as gym

# シミュレーターアプリ作成
from isaaclab.app import AppLauncher

# 引数解析
parser = argparse.ArgumentParser(description="IK対応SO101環境でteleoperation機能をテスト")
parser.add_argument("--num_envs", type=int, default=1, help="環境数")
parser.add_argument("--teleop_device", type=str, default="keyboard", choices=["keyboard", "spacemouse"], help="Teleoperation デバイス")
parser.add_argument("--usd_path", type=str, default="../assets/so101_new_calib_fix_articulation_root.usd", help="SO101ロボットのUSDファイルパス")

# AppLauncher の引数を追加
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# カメラを有効化
args_cli.enable_cameras = True

# シミュレーター起動
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------------------------
# 強化学習環境定義
# ------------------------------------------------------------
import isaaclab_tasks  # noqa: F401
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import lerobot_so101  # lerobot_so101パッケージをインポートして環境を登録

def test_ik_teleop_compatibility():
    """IK対応環境のTeleoperation互換性をテスト"""
    from lerobot_so101.env_rl_so101_ik import LeRobotSO101StackCubeIKRelEnvCfg

    print(f"[INFO]: IK対応Teleoperation互換性テストを開始")
    print(f"[INFO]: デバイス: {args_cli.teleop_device}")
    print(f"[INFO]: 環境: LeRobot-SO101-StackCube-IK-Rel-v0")

    try:
        # IK環境設定を作成
        env_cfg = LeRobotSO101StackCubeIKRelEnvCfg()
        env_cfg.robot_usd_path = args_cli.usd_path

        # teleoperation設定を適用
        env_cfg.use_teleop_device(args_cli.teleop_device)

        print(f"[INFO]: ✅ use_teleop_device メソッドが正常に動作")

        # IK対応環境を作成
        env = gym.make("LeRobot-SO101-StackCube-IK-Rel-v0", cfg=env_cfg)

        print(f"[INFO]: ✅ IK環境作成成功")
        print(f"[INFO]: 観測空間: {env.observation_space}")
        print(f"[INFO]: アクション空間: {env.action_space}")

        # アクション空間のチェック
        action_shape = env.action_space.shape
        print(f"[INFO]: アクション次元: {action_shape}")

        if action_shape[-1] == 7:  # SE(3) pose (6) + gripper (1)
            print(f"[INFO]: ✅ SE(3)+グリッパー制御（7次元）- Teleoperation対応")
        elif action_shape[-1] == 6:  # SE(3) pose only
            print(f"[INFO]: ✅ SE(3)制御（6次元）- Teleoperation対応")
        else:
            print(f"[WARNING]: ⚠️ アクション次元が想定と異なります: {action_shape[-1]}")

        # 基本的な動作テスト
        print(f"[INFO]: 基本動作テストを実行...")
        observations, info = env.reset()
        print(f"[INFO]: ✅ 環境リセット成功")

        if 'policy' in observations:
            print(f"[INFO]: 初期観測の形状: {observations['policy'].shape}")

        # IK制御用のテストアクション（SE(3) pose + gripper）
        if action_shape[-1] == 7:
            # [dx, dy, dz, droll, dpitch, dyaw, gripper]形式
            test_action = torch.tensor([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        elif action_shape[-1] == 6:
            # [dx, dy, dz, droll, dpitch, dyaw]形式
            test_action = torch.tensor([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        else:
            # フォールバック: ランダムアクション
            test_action = torch.tensor(env.action_space.sample(), dtype=torch.float32)

        print(f"[INFO]: テストアクション: {test_action}")

        observations, rewards, terminated, truncated, info = env.step(test_action)
        print(f"[INFO]: ✅ IKステップ実行成功")
        print(f"[INFO]: 報酬: {rewards[0].item():.3f}")

        env.close()
        print(f"[INFO]: ✅ 環境クローズ成功")

        print(f"\n[SUCCESS]: 🎉 SO101 IK環境はteleoperation機能と互換性があります！")
        print(f"[INFO]: Isaac Lab標準teloperationスクリプトで使用可能:")
        print(f"[INFO]: ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \\")
        print(f"[INFO]:     --task LeRobot-SO101-StackCube-IK-Rel-v0 --num_envs 1 \\")
        print(f"[INFO]:     --teleop_device keyboard")

    except Exception as e:
        print(f"[ERROR]: ❌ テスト中にエラーが発生: {str(e)}")
        print(f"[INFO]: 考えられる原因:")
        print(f"[INFO]: 1. SO101ロボットの関節名が正しく設定されていない")
        print(f"[INFO]: 2. エンドエフェクターのリンク名が間違っている")
        print(f"[INFO]: 3. USDファイルの構造とコード設定の不一致")
        import traceback
        traceback.print_exc()

    finally:
        print(f"[INFO]: テスト完了")

def main():
    """メイン関数"""
    test_ik_teleop_compatibility()

if __name__ == "__main__":
    main()
    simulation_app.close()
