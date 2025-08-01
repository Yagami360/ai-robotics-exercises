"""SO101ロボット用の高度なteloperationスクリプト
Isaac Labの標準teleoperation deviceを使用し、SE(3)コマンドを関節角度に変換します。
"""
import os
import sys
import numpy as np
import time

import argparse
import torch
import numpy as np
import gymnasium as gym
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SO101 Advanced Teleoperation")
parser.add_argument("--usd_path", type=str, default="../assets/so101_new_calib_fix_articulation_root.usd", help="Robot USD path")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--task", type=str, default="LeRobot-SO101-StackCube-v0", help="Environment name")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Teleoperation device (only 'keyboard' supported)")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor")
parser.add_argument("--record", action="store_true", default=True, help="whether to enable record function")
parser.add_argument("--dataset_file", type=str, default="../datasets/teleop_so101/dataset.hdf5", help="File path to export recorded demos.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--num_demos", type=int, default=10, help="Number of demonstrations to record. Set to 0 for infinite.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# Simulatorを起動
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab 関連のインポート
import lerobot_so101  # カスタム環境の登録
from isaaclab.envs.mdp.recorders import ActionStateRecorderManagerCfg
from isaaclab.managers import TerminationTermCfg

# leisaac 関連のインポート
# Isaac Lab 公式の Se3Keyboard ではなく、leisaac の Se3Keyboard を使用する
from leisaac.enhance.managers import StreamingRecorderManager
from leisaac.devices import Se3Keyboard

class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


class SO101TeleopJointMapper:
    """SE(3)コマンドを関節角度制御に変換するマッパー"""

    def __init__(self, scaling_factor: float = 0.5, gripper_close_value: float = 0.00, gripper_open_value: float = 0.10):
        self.scaling_factor = scaling_factor
        self.gripper_close_value = gripper_close_value
        self.gripper_open_value = gripper_open_value
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

    def se3_to_joint_deltas(self, delta_pose: np.ndarray, gripper_command: bool) -> torch.Tensor:
        """SE(3)デルタ姿勢を関節角度デルタに変換
        Args:
            delta_pose: [dx, dy, dz, drx, dry, drz] のSE(3)デルタ
            gripper_command: グリッパーコマンド (True=閉じる, False=開く)

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

        # グリッパー制御は絶対位置で設定（デルタではなく）
        # Isaac LabのSe3Keyboardでは gripper_command=True が「閉じる」を意味する
        joint_deltas[5] = 0.0  # デルタは0にして、update_joint_targetsで絶対値を設定

        return joint_deltas

    def update_joint_targets(self, delta_pose: np.ndarray, gripper_command: bool) -> torch.Tensor:
        """関節目標値を更新"""
        joint_deltas = self.se3_to_joint_deltas(delta_pose, gripper_command)

        # アーム関節（0-4）のデルタ制御
        self.prev_joint_targets[:5] += joint_deltas[:5]

        # グリッパー（5）の絶対位置制御
        if gripper_command:
            self.prev_joint_targets[5] = self.gripper_close_value
        else:
            self.prev_joint_targets[5] = self.gripper_open_value

        # 関節制限を適用
        joint_limits = [
            [-3.14, 3.14],   # shoulder_pan
            [-1.57, 1.57],   # shoulder_lift
            [-2.62, 2.62],   # elbow_flex
            [-1.92, 1.92],   # wrist_flex
            [-3.14, 3.14],   # wrist_roll
            [-1.0, 1.0],     # gripper
        ]

        for i, (min_val, max_val) in enumerate(joint_limits):
            self.prev_joint_targets[i] = torch.clamp(
                self.prev_joint_targets[i], min_val, max_val
            )

        return self.prev_joint_targets


def create_teleop_interface(env):
    """Teleoperation インターフェースを作成"""
    if args_cli.teleop_device.lower() == "keyboard":
        # leisaac版Se3Keyboardは環境オブジェクトを必要とする
        teleop_interface = Se3Keyboard(env, sensitivity=args_cli.sensitivity)

        # 必要なコールバックを登録
        def reset_failed_callback():
            """タスク失敗時のリセットコールバック"""
            print("[RESET] タスク失敗によるリセット")

        def reset_success_callback():
            """タスク成功時のリセットコールバック"""
            print("[RESET] タスク成功によるリセット")

        teleop_interface._additional_callbacks["R"] = reset_failed_callback
        teleop_interface._additional_callbacks["N"] = reset_success_callback

        return teleop_interface
    else:
        raise ValueError(f"Unsupported teleop device: {args_cli.teleop_device}")


def create_env():
    """環境を作成"""
    from lerobot_so101.env_rl_so101 import LeRobotSO101StackCubeRLEnvCfg

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 環境設定を作成
    env_cfg = LeRobotSO101StackCubeRLEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.robot_usd_path = args_cli.usd_path
    env_cfg.use_teleop_device(args_cli.teleop_device)

    # 設定の修正 (leisaac標準方式)
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    if args_cli.record:
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        if not hasattr(env_cfg.terminations, "success"):
            setattr(env_cfg.terminations, "success", None)
        env_cfg.terminations.success = TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    else:
        env_cfg.recorders = None

    # 環境を作成 (.unwrappedで直接ManagerBasedRLEnvを取得)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # StreamingRecorderManager でレコーダーを構成
    if args_cli.record:
        del env.recorder_manager
        env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
        env.recorder_manager.flush_steps = 100
        env.recorder_manager.compression = 'lzf'

    return env


def main():
    """メイン関数"""
    print(f"[INFO]: SO101高度teleoperation環境を起動中...")

    # 環境作成
    env = create_env()

    # Teleoperation インターフェース作成
    teleop_interface = create_teleop_interface(env)

    # SE(3) → 関節角度マッパー作成（グリッパーをより強く閉じる設定）
    joint_mapper = SO101TeleopJointMapper(
        scaling_factor=0.1,
        gripper_close_value=0.00,
        gripper_open_value=1.00
    )

    # RateLimiterの初期化
    rate_limiter = RateLimiter(args_cli.step_hz)

    # レコーダー状況の表示
    if args_cli.record:
        print(f"[INFO]: StreamingRecorderManager が有効です")
        print(f"[INFO]: データ出力先: {args_cli.dataset_file}")
        print(f"[INFO]: フラッシュステップ: {env.unwrapped.recorder_manager.flush_steps}")
        print(f"[INFO]: 圧縮形式: {env.unwrapped.recorder_manager.compression}")
    else:
        print(f"[INFO]: データセット記録は無効です（--recordで有効化）")

    # キーボード操作説明を表示 (leisaac Se3Keyboard用)
    if args_cli.teleop_device.lower() == "keyboard":
        print("\n" + "="*60)
        print("キーボード操作ガイド (Leisaac 関節制御)")
        print("="*60)
        print("関節制御:")
        print("  Q/U: Joint 1 (shoulder_pan)")
        print("  W/I: Joint 2 (shoulder_lift)")  
        print("  E/O: Joint 3 (elbow_flex)")
        print("  A/J: Joint 4 (wrist_flex)")
        print("  S/K: Joint 5 (wrist_roll)")
        print("  D/L: Joint 6 (gripper)")
        print("")
        print("システム制御:")
        print("  B: 制御開始")
        print("  R: 失敗時リセット")
        print("  N: 成功時リセット")
        print("  Ctrl+C: 終了")
        print("="*60)
        print("注意: Leisaac Se3Keyboard による直接関節制御")
        print("      Bキーで制御開始してください")
        if args_cli.record:
            print("📊 データセット記録: 有効 (StreamingRecorderManager)")
            if args_cli.num_demos > 0:
                print(f"🎯 目標デモ数: {args_cli.num_demos}")
            print(f"⏱️  実行レート: {args_cli.step_hz} Hz")
        else:
            print("📊 データセット記録: 無効 (--recordで有効化)")
        print("="*60 + "\n")

    # 環境をリセット
    observations, _ = env.reset()
    teleop_interface.reset()
    print(f"[INFO]: 環境が正常に起動しました")
    print(f"[INFO]: アクション次元: {env.action_space.shape}")
    print(f"[INFO]: Teleoperation device: {args_cli.teleop_device}")    
    print(f"[DEBUG]: Isaac Lab Se3Keyboard デバイス情報:")
    print(teleop_interface)

    # フラグ
    teleoperation_active = True

    # メインループ
    step_count = 0
    current_recorded_demo_count = 0
    print("[INFO]: メインループを開始します")
    print("[INFO]: Bキーを押して制御を開始してください")

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                # Teleoperation データを取得 (leisaac style)
                teleop_data = teleop_interface.input2action()

                # 基本的なレンダリングを常に実行
                env.sim.render()

                if teleop_data.get('started', False) and not teleop_data.get('reset', False):
                    # leisaac Se3Keyboardからの関節状態を取得
                    joint_deltas = teleop_data['joint_state']  # 6次元の関節デルタ

                    # 関節デルタ表示（ゼロでない場合）
                    if np.any(np.abs(joint_deltas) > 0.001):
                        print(f"[TELEOP] 関節デルタ: [{joint_deltas[0]:.3f}, {joint_deltas[1]:.3f}, {joint_deltas[2]:.3f}, {joint_deltas[3]:.3f}, {joint_deltas[4]:.3f}, {joint_deltas[5]:.3f}]")

                    # 関節目標値を更新
                    joint_mapper.prev_joint_targets += torch.from_numpy(joint_deltas).float()

                    # 関節制限を適用
                    joint_limits = [
                        [-3.14, 3.14],   # shoulder_pan
                        [-1.57, 1.57],   # shoulder_lift
                        [-2.62, 2.62],   # elbow_flex
                        [-1.92, 1.92],   # wrist_flex
                        [-3.14, 3.14],   # wrist_roll
                        [-1.0, 1.0],     # gripper
                    ]
                    for i, (min_val, max_val) in enumerate(joint_limits):
                        joint_mapper.prev_joint_targets[i] = torch.clamp(
                            joint_mapper.prev_joint_targets[i], min_val, max_val
                        )

                    # アクションをテンソルに変換
                    actions = joint_mapper.prev_joint_targets.unsqueeze(0).to(args_cli.device)

                    # 環境ステップ実行
                    observations, rewards, terminated, truncated, info = env.step(actions)

                    # レコーダー統計確認 (leisaac style)
                    if args_cli.record and hasattr(env, 'recorder_manager'):
                        if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                            current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                            print(f"Recorded {current_recorded_demo_count} successful demonstrations.")
                        if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                            print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                            break

                # リセット処理 (leisaac style)
                elif teleop_data.get('reset', False):
                    observations, _ = env.reset()
                    joint_mapper.prev_joint_targets = torch.zeros(6)
                    print("環境をリセットしました")

                # 情報表示（100ステップごと）
                if step_count % 100 == 0 and step_count > 0 and teleop_data.get('started', False):
                    try:
                        current_joint_pos = observations['policy'][0, :6]  # 最初の6次元が関節位置
                        target_joint_pos = joint_mapper.prev_joint_targets
                        print(f"[INFO] Step {step_count}:")
                        print(f"  現在関節位置: {current_joint_pos.cpu().numpy().round(3)}")
                        print(f"  目標関節位置: {target_joint_pos.cpu().numpy().round(3)}")
                        print(f"  関節名: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]")
                    except:
                        pass

                step_count += 1

                # RateLimiter適用 (leisaac style)
                rate_limiter.sleep(env)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt - 終了中...")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("シミュレーションを終了しています...")
        try:
            env.close()
        except:
            pass
        try:
            simulation_app.close()
        except:
            pass
        print("環境をクローズしました")


if __name__ == "__main__":
    main()
