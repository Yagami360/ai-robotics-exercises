import argparse
import os
import math
import torch
import numpy as np
import cv2

# ------------------------------------------------------------
# シミュレーターアプリ作成
# ------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd_path", type=str, default="../assets/so101_new_calib_fix_articulation_root.usd")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ロボットからのカメラを配置しているので有効化
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------------------------
# 環境定義
# NOTE: "ModuleNotFoundError: No module named 'isaacsim.core'" のエラーがでないように、
# IsaacSim 関連の import 文は AppLauncher の後に記載する必要がある
# ------------------------------------------------------------
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ActionTermCfg, ObservationTermCfg, ObservationGroupCfg, SceneEntityCfg
from isaaclab.managers import EventTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils import configclass


@configclass
class LeRobotSO101ActionCfg:
    """｛LeRobot の SO-ARM101 ロボット x 特定タスク｝環境の action 定義"""
    joint_position = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
        scale=1.0
    )


@configclass  
class LeRobotSO101ObservationCfg:
    """｛LeRobot の SO-ARM101 ロボット x 特定タスク｝環境の observation（観測）定義"""

    @configclass
    class JointPolicyCfg(ObservationGroupCfg):
        """関節データの observation グループ"""
        # 関節位置の観測
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        # 関節速度の観測
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

    @configclass
    class CameraPolicyCfg(ObservationGroupCfg):
        """カメラ画像の observation グループ"""
        # RGB画像の観測
        rgb_image = ObservationTermCfg(
            func=lambda env: env.scene["robot_camera"].data.output["rgb"],
            params={},
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False  # カメラ画像は連結しない

    # NOTE: Policy という名前が入っているが、行動方策（ポリシー）ではなく観測データ（observation）のグループのこと
    # 関節データとカメラ画像で次元数が異なるため、別の観測グループに分離する必要あり
    # 関節データ用の観測グループ
    joint: JointPolicyCfg = JointPolicyCfg()
    # カメラデータ用の観測グループ
    camera: CameraPolicyCfg = CameraPolicyCfg()


@configclass
class LeRobotSO101EventCfg:
    """
    ｛LeRobot の SO-ARM101 ロボット x 特定タスク｝環境のイベント設定
    イベント：シミュレーション状態の変化に対応するイベント。例えば、シーンのリセット、物理特性のランダム化など
    mode 引数で、イベントの実行タイミングを指定できる。
    - "startup" - 環境の起動時に一度だけ実行されるイベント。
    - "reset" - 環境の終了とリセット時に発生するイベント。
    - "interval" - 指定された間隔で実行されるイベント、つまり一定のステップ数の後に定期的に実行される。
    """

    # リセット時の関節位置ランダム化
    reset_joint_positions = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.05, 0.05),
        },
    )


@configclass
class LeRobotSO101EnvCfg(ManagerBasedEnvCfg):
    """｛LeRobot の SO-ARM101 ロボット x 特定タスク｝環境の設定クラス"""
    # シーン設定
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    # observation 設定
    observations: LeRobotSO101ObservationCfg = LeRobotSO101ObservationCfg()
    # action 設定
    actions: LeRobotSO101ActionCfg = LeRobotSO101ActionCfg()
    # event 設定
    events: LeRobotSO101EventCfg = LeRobotSO101EventCfg()

    def __post_init__(self):
        """初期化後の設定"""
        # 基本設定
        self.decimation = 4
        self.episode_length_s = 10.0

        # シーン設定を更新
        self.scene.num_envs = args_cli.num_envs
        self.scene.env_spacing = 2.0

        # 地面を配置（Franka環境と同じ高さに調整）
        self.scene.ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        )

        # 照明を配置
        self.scene.dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        )

        # テーブルを配置（
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.6, 0.0, 0.0),
                rot=(0.70711, 0.0, 0.0, 0.70711)
            ),
        )

        # LeRobot の SO-ARM101 ロボットを配置
        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/SO101_Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=args_cli.usd_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    # URDF ファイルのジョイント名
                    "shoulder_pan": 0.0,
                    "shoulder_lift": 0.0,
                    "elbow_flex": 0.0,
                    "wrist_flex": 0.0,
                    "wrist_roll": 0.0,
                    "gripper": 0.0,
                },
                pos=(0.0, 0.0, 0.0),
            ),
            actuators={
                "arm_actuator": ImplicitActuatorCfg(
                    joint_names_expr=[
                        "shoulder_pan",
                        "shoulder_lift",
                        "elbow_flex",
                        "wrist_flex",
                        "wrist_roll",
                    ],
                    effort_limit=100.0,
                    velocity_limit=10.0,
                    stiffness=100.0,
                    damping=10.0,
                ),
                "gripper_actuator": ImplicitActuatorCfg(
                    joint_names_expr=[
                        "gripper",
                    ],
                    effort_limit=50.0,
                    velocity_limit=5.0,
                    stiffness=50.0,
                    damping=5.0,
                ),
            },
        )

        # カメラセンサー（シーン全体を見下ろす位置に配置）
        self.scene.robot_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0.1,  # 10Hzで更新
            height=512,
            width=512,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 20.0)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.6, 0.6, 1.0),
                rot=(0.7071, -0.1830, 0.1830, 0.6830),
                convention="world"
            ),
                )

        # Cubeオブジェクトの共通物理プロパティ
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # Cube 1 (青色)
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.25, 0.0, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_1")],
            ),
        )

        # Cube 2 (赤色)
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.30, 0.10, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_2")],
            ),
        )

        # Cube 3 (緑色)
        self.scene.cube_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.30, -0.10, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_3")],
            ),
        )

        # 視点カメラ設定
        self.viewer.eye = [1.0, 1.0, 1.2]
        self.viewer.lookat = [0.2, 0.0, 0.2]


class LeRobotSO101Env(ManagerBasedEnv):
    """｛LeRobot の SO-ARM101 ロボット x 特定タスク｝環境クラス"""

    cfg: LeRobotSO101EnvCfg

    def __init__(self, cfg: LeRobotSO101EnvCfg, **kwargs):
        """環境の初期化"""
        super().__init__(cfg, **kwargs)

        # 時間カウンタ
        self._sim_time = 0.0
        self._step_count = 0

        # ロボットの参照
        self._robot = self.scene["robot"]

        # 関節名のリスト
        self._joint_names = [
            "shoulder_pan",
            "shoulder_lift", 
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        print(f"[INFO]: 制御可能な関節: {self._joint_names}")
        print(f"[INFO]: ロボットの関節数: {self._robot.num_joints}")

        # カメラの参照
        self._robot_camera = self.scene["robot_camera"]
        print(f"[INFO]: カメラが追加されました: {self._robot_camera}")
        
        # Cubeオブジェクトの参照
        self._cube_1 = self.scene["cube_1"]
        self._cube_2 = self.scene["cube_2"]
        self._cube_3 = self.scene["cube_3"]
        print(f"[INFO]: Cubeオブジェクトが追加されました:")
        print(f"  - Cube 1 (青): {self._cube_1}")
        print(f"  - Cube 2 (赤): {self._cube_2}")
        print(f"  - Cube 3 (緑): {self._cube_3}")

        print(f"[INFO]: 環境が初期化されました")

    def _setup_scene(self):
        """シーンのセットアップ"""
        super()._setup_scene()

    def get_camera_data(self):
        """カメラデータを取得"""
        return {
            "rgb": self._robot_camera.data.output["rgb"]
        }
    
    def get_cube_positions(self):
        """Cubeオブジェクトの位置を取得"""
        return {
            "cube_1": self._cube_1.data.root_pos_w[0].cpu().numpy(),
            "cube_2": self._cube_2.data.root_pos_w[0].cpu().numpy(),
            "cube_3": self._cube_3.data.root_pos_w[0].cpu().numpy(),
        }


def get_actions(env: LeRobotSO101Env):
    """デモ用のロボット動作（正弦波動作）"""
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    sim_time = env._sim_time

    # 各関節に正弦波の動きを与える
    # shoulder_pan: ±30度、周期2秒
    actions[:, 0] = 0.5 * math.sin(2 * math.pi * sim_time / 2.0)
    # shoulder_lift: ±20度、周期3秒
    actions[:, 1] = 0.35 * math.sin(2 * math.pi * sim_time / 3.0)
    # elbow_flex: ±40度、周期2.5秒
    actions[:, 2] = 0.7 * math.sin(2 * math.pi * sim_time / 2.5)
    # wrist_flex: ±25度、周期1.5秒
    actions[:, 3] = 0.45 * math.sin(2 * math.pi * sim_time / 1.5)
    # wrist_roll: ±15度、周期4秒
    actions[:, 4] = 0.25 * math.sin(2 * math.pi * sim_time / 4.0)
    # gripper: 固定
    actions[:, 5] = 0.0
    return actions


def run_environment():
    """環境を使用したシミュレーションの実行"""
    print("[INFO]: ManagerBasedEnv を使用したシミュレーション開始...")

    # 環境を作成
    env_cfg = LeRobotSO101EnvCfg()
    env_cfg.__post_init__()
    env = LeRobotSO101Env(cfg=env_cfg)

    # 環境をリセット
    obs, _ = env.reset()
    print(f"[INFO]: 関節データの観測値の形状: {obs['joint'].shape}")
    print(f"[INFO]: カメラデータの観測値の形状: {obs['camera']['rgb_image'].shape}")

    step_count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            # リセットのタイミング
            if step_count % 300 == 0:
                step_count = 0
                env._sim_time = 0.0
                obs, _ = env.reset()
                print("-" * 80)
                print("[INFO]: 環境をリセットしました")

            # デモ用のロボット動作でアクション取得
            actions = get_actions(env)

            # アクションを実行
            obs, info = env.step(actions)
            # if step_count % 100 == 0:
            #     for key, value in obs.items():
            #         print(f"[INFO]: observation: {key}: {value.shape}")
            #     for key, value in info.items():
            #         print(f"[INFO]: info: {key}: {value}")

            # 時間を更新
            env._sim_time += env.step_dt
            step_count += 1

            # 定期的に情報を出力
            if step_count % 100 == 0:
                print(f"[INFO]: ステップ数: {step_count}, シミュレーション時間: {env._sim_time:.2f}s")
                # 現在の関節角度を表示
                current_positions = env._robot.data.joint_pos[0, :6]
                print(f"[INFO]: 関節角度 [rad]: {[f'{pos:.3f}' for pos in current_positions.cpu().tolist()]}")

                # カメラデータの情報を表示
                camera_data = env.get_camera_data()
                if camera_data:
                    rgb_shape = camera_data["rgb"].shape
                    print(f"[INFO]: カメラRGB画像の形状: {rgb_shape}")
                    camera_image = camera_data["rgb"].cpu().numpy()
                    cv2.imwrite(
                        "robot_camera.png", cv2.cvtColor(camera_image[0], cv2.COLOR_RGB2BGR)
                    )
                
                # Cubeの位置情報を表示
                cube_positions = env.get_cube_positions()
                if cube_positions:
                    print(f"[INFO]: Cube位置:")
                    for name, pos in cube_positions.items():
                        print(f"  - {name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # 環境を閉じる
    env.close()


def main():
    """メイン関数"""
    # 引数の出力
    for arg in vars(args_cli):
        print(f"{arg}: {getattr(args_cli, arg)}")

    # USDファイルが存在するかチェック
    if not os.path.exists(args_cli.usd_path):
        print(f"エラー: USDファイルが見つかりません: {args_cli.usd_path}")
        return
    else:
        print(f"USDファイルを確認しました: {args_cli.usd_path}")

    # 環境ベースのシミュレーションを実行
    run_environment()

    # シミュレーションを終了
    simulation_app.close()


if __name__ == "__main__":
    main()
