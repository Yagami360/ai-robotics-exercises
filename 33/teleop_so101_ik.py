"""SO101ロボット用の高度なteloperationスクリプト
Isaac Labの標準teleoperation deviceを使用し、SE(3)コマンドを関節角度に変換します。
"""

import argparse
import torch
import numpy as np
import gymnasium as gym
from isaaclab.app import AppLauncher

# argparseセットアップ  
parser = argparse.ArgumentParser(description="SO101 Advanced Teleoperation")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--task", type=str, default="LeRobot-SO101-StackCube-IK-Rel-v0", help="Environment name")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Teleoperation device")
parser.add_argument("--usd_path", type=str, default="../assets/so101_new_calib_fix_articulation_root.usd", help="Robot USD path")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor")

# AppLauncher設定
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# Simulatorを起動
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab デバイスとライブラリのインポート
import lerobot_so101  # カスタム環境の登録
from isaaclab.devices import Se3Keyboard, Se3SpaceMouse


class SO101TeleopJointMapper:
    """SE(3)コマンドを関節角度制御に変換するマッパー"""
    
    def __init__(self, scaling_factor: float = 0.5):
        self.scaling_factor = scaling_factor
        self.prev_joint_targets = torch.zeros(6)
        
        # 簡易的なSE(3) → 関節角度マッピング
        # 実際にはIKソルバーを使用すべきですが、ここでは簡易マッピングを使用
        self.se3_to_joint_mapping = {
            # [dx, dy, dz, drx, dry, drz] -> [j0, j1, j2, j3, j4, gripper]
            # これは簡易的なマッピングで、実際の用途では適切なIKが必要
        }
        
        print("SO101 Teleoperation Joint Mapper")
        print("================================")
        print("SE(3)コマンドを関節角度制御に変換します")
        print("注意: これは簡易マッピングです。正確な制御にはIKソルバーが必要です")
        
    def se3_to_joint_deltas(self, delta_pose: np.ndarray, gripper_command: bool) -> torch.Tensor:
        """SE(3)デルタ姿勢を関節角度デルタに変換
        
        Args:
            delta_pose: [dx, dy, dz, drx, dry, drz] のSE(3)デルタ
            gripper_command: グリッパーコマンド (True=開く, False=閉じる)
        
        Returns:
            関節角度デルタ [j0, j1, j2, j3, j4, gripper] 
        """
        dx, dy, dz, drx, dry, drz = delta_pose
        
        # 簡易的なマッピング（実際のロボットキネマティクスに基づいて調整が必要）
        joint_deltas = torch.zeros(6)
        
        # 平行移動をベース関節に大まかにマッピング
        joint_deltas[0] = dy * self.scaling_factor  # shoulder_pan (Y軸移動)
        joint_deltas[1] = -dz * self.scaling_factor  # shoulder_lift (Z軸移動、反転)
        joint_deltas[2] = dx * self.scaling_factor   # elbow_flex (X軸移動)
        
        # 回転を手首関節にマッピング
        joint_deltas[3] = drx * self.scaling_factor  # wrist_flex
        joint_deltas[4] = drz * self.scaling_factor  # wrist_roll
        
        # グリッパー
        if gripper_command:
            joint_deltas[5] = 0.01  # 開く
        else:
            joint_deltas[5] = -0.01  # 閉じる
            
        return joint_deltas
    
    def update_joint_targets(self, delta_pose: np.ndarray, gripper_command: bool) -> torch.Tensor:
        """関節目標値を更新"""
        joint_deltas = self.se3_to_joint_deltas(delta_pose, gripper_command)
        self.prev_joint_targets += joint_deltas
        
        # 関節制限を適用
        joint_limits = [
            [-3.14, 3.14],   # shoulder_pan
            [-1.57, 1.57],   # shoulder_lift
            [-2.62, 2.62],   # elbow_flex
            [-1.92, 1.92],   # wrist_flex
            [-3.14, 3.14],   # wrist_roll
            [0.0, 0.04],     # gripper
        ]
        
        for i, (min_val, max_val) in enumerate(joint_limits):
            self.prev_joint_targets[i] = torch.clamp(
                self.prev_joint_targets[i], min_val, max_val
            )
        
        return self.prev_joint_targets


def create_teleop_interface():
    """Teleoperation インターフェースを作成"""
    if args_cli.teleop_device.lower() == "keyboard":
        return Se3Keyboard(pos_sensitivity=args_cli.sensitivity, rot_sensitivity=args_cli.sensitivity)
    elif args_cli.teleop_device.lower() == "spacemouse":
        return Se3SpaceMouse(pos_sensitivity=args_cli.sensitivity, rot_sensitivity=args_cli.sensitivity)
    else:
        raise ValueError(f"Unsupported teleop device: {args_cli.teleop_device}")


def create_env():
    """環境を作成"""
    from lerobot_so101.env_rl_so101 import LeRobotSO101StackCubeRLEnvCfg
    
    # 環境設定を作成
    env_cfg = LeRobotSO101StackCubeRLEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.robot_usd_path = args_cli.usd_path
    env_cfg.use_teleop_device(args_cli.teleop_device)
    
    # 環境を作成
    env = gym.make(args_cli.task, cfg=env_cfg)
    return env


def main():
    """メイン関数"""
    print(f"[INFO]: SO101高度teleoperation環境を起動中...")
    
    # 環境作成
    env = create_env()
    
    # Teleoperation インターフェース作成
    teleop_interface = create_teleop_interface()
    
    # SE(3) → 関節角度マッパー作成
    joint_mapper = SO101TeleopJointMapper(scaling_factor=0.1)
    
    # 環境をリセット
    observations, _ = env.reset()
    teleop_interface.reset()
    
    print(f"[INFO]: 環境が正常に起動しました")
    print(f"[INFO]: アクション次元: {env.action_space.shape}")
    print(f"[INFO]: Teleoperation device: {args_cli.teleop_device}")
    
    # フラグ
    should_reset = False
    teleoperation_active = True
    
    # コールバック関数
    def reset_env():
        nonlocal should_reset
        should_reset = True
        print("環境リセットが要求されました")
    
    def toggle_teleop():
        nonlocal teleoperation_active
        teleoperation_active = not teleoperation_active
        print(f"Teleoperation: {'有効' if teleoperation_active else '無効'}")
    
    # コールバックを登録
    teleop_interface.add_callback("R", reset_env)
    teleop_interface.add_callback("L", toggle_teleop)
    
    # メインループ
    step_count = 0
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                # Teleoperation データを取得
                teleop_data = teleop_interface.advance()
                
                if teleoperation_active and teleop_data is not None:
                    delta_pose, gripper_command = teleop_data
                    
                    # SE(3)コマンドを関節角度に変換
                    joint_targets = joint_mapper.update_joint_targets(delta_pose, gripper_command)
                    
                    # アクションをテンソルに変換
                    actions = joint_targets.unsqueeze(0).to(args_cli.device)
                    
                    # 環境ステップ実行
                    observations, rewards, terminated, truncated, info = env.step(actions)
                    
                else:
                    # Teleoperation非活性時はレンダリングのみ
                    env.sim.render()
                
                # 情報表示（50ステップごと）
                if step_count % 50 == 0 and step_count > 0:
                    current_joint_pos = observations['policy'][0, :6]  # 最初の6次元が関節位置
                    print(f"Step {step_count}: 関節位置 = {current_joint_pos.cpu().numpy().round(3)}")
                
                step_count += 1
                
                # リセット処理
                if should_reset or terminated.any() or truncated.any():
                    if should_reset:
                        print("手動リセット実行中...")
                    else:
                        print("エピソード終了 - 自動リセット中...")
                    
                    observations, _ = env.reset()
                    joint_mapper.prev_joint_targets = torch.zeros(6)  # 関節目標値もリセット
                    should_reset = False
                    step_count = 0
    
    except KeyboardInterrupt:
        print("\nKeyboard interrupt - 終了中...")
    
    finally:
        env.close()
        print("環境をクローズしました")


if __name__ == "__main__":
    main()