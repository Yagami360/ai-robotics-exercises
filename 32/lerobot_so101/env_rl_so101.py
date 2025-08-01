"""LeRobot SO-101 ロボットを使用したキューブスタッキング環境の設定
このモジュールは Isaac Lab の ManagerBasedRLEnv を使用した強化学習＆模倣学習用の環境の設定クラスを定義します。
"""
import os
import math
import torch
import numpy as np
import cv2

# ------------------------------------------------------------
# 強化学習環境定義
# ------------------------------------------------------------
import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg, ObservationTermCfg, ObservationGroupCfg, SceneEntityCfg
from isaaclab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils import configclass


@configclass
class LeRobotSO101StackCubeActionCfg:
    """｛LeRobot の SO-ARM101 ロボット x キューブ積み重ね｝環境の action 定義"""
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
class LeRobotSO101StackCubeObservationCfg:
    """｛LeRobot の SO-ARM101 ロボット x キューブ積み重ね｝環境の observation（観測）定義"""

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
    policy: JointPolicyCfg = JointPolicyCfg()
    # カメラデータ用の観測グループ
    camera: CameraPolicyCfg = CameraPolicyCfg()


@configclass
class LeRobotSO101StackCubeRewardCfg:
    """｛LeRobot の SO-ARM101 ロボット x キューブ積み重ね｝環境の報酬定義
    強化学習ではなく模倣学習用の環境の場合は不要
    """

    # 主要報酬1: cube_2をcube_1の上に正確に積む
    cube_1_2_xy_alignment = RewardTermCfg(
        func=lambda env: -torch.norm(
            env.scene["cube_2"].data.root_pos_w[:, :2] - env.scene["cube_1"].data.root_pos_w[:, :2], dim=1
        ),
        weight=5.0,  # XY位置合わせは重要
        params={},
    )

    cube_1_2_height_alignment = RewardTermCfg(
        func=lambda env: -torch.abs(
            torch.norm(env.scene["cube_1"].data.root_pos_w[:, 2:] - env.scene["cube_2"].data.root_pos_w[:, 2:], dim=1) - 0.0468
        ),
        weight=10.0,  # 高さ合わせは非常に重要
        params={},
    )

    # 主要報酬2: cube_3をcube_2の上に正確に積む
    cube_2_3_xy_alignment = RewardTermCfg(
        func=lambda env: -torch.norm(
            env.scene["cube_3"].data.root_pos_w[:, :2] - env.scene["cube_2"].data.root_pos_w[:, :2], dim=1
        ),
        weight=7.0,  # 最上段なので高い重み
        params={},
    )

    cube_2_3_height_alignment = RewardTermCfg(
        func=lambda env: -torch.abs(
            torch.norm(env.scene["cube_2"].data.root_pos_w[:, 2:] - env.scene["cube_3"].data.root_pos_w[:, 2:], dim=1) - 0.0468
        ),
        weight=15.0,  # 最上段の高さ合わせは最も重要
        params={},
    )

    # 段階的成功ボーナス（Isaac Lab cubes_stacked基準）
    cube_1_2_stacked_bonus = RewardTermCfg(
        func=lambda env: (
            (torch.norm(env.scene["cube_2"].data.root_pos_w[:, :2] - env.scene["cube_1"].data.root_pos_w[:, :2], dim=1) < 0.05) &
            (torch.abs(torch.norm(env.scene["cube_1"].data.root_pos_w[:, 2:] - env.scene["cube_2"].data.root_pos_w[:, 2:], dim=1) - 0.0468) < 0.005)
        ).float() * 50.0,  # 段階的成功時の大きなボーナス
        weight=1.0,
        params={},
    )

    cube_2_3_stacked_bonus = RewardTermCfg(
        func=lambda env: (
            (torch.norm(env.scene["cube_3"].data.root_pos_w[:, :2] - env.scene["cube_2"].data.root_pos_w[:, :2], dim=1) < 0.05) &
            (torch.abs(torch.norm(env.scene["cube_2"].data.root_pos_w[:, 2:] - env.scene["cube_3"].data.root_pos_w[:, 2:], dim=1) - 0.0468) < 0.005)
        ).float() * 75.0,  # 最終段階の更に大きなボーナス
        weight=1.0,
        params={},
    )

    # 完全成功ボーナス（全て正確に積み重なった状態）
    all_cubes_stacked_bonus = RewardTermCfg(
        func=lambda env: (
            # cube_1とcube_2の条件
            (torch.norm(env.scene["cube_2"].data.root_pos_w[:, :2] - env.scene["cube_1"].data.root_pos_w[:, :2], dim=1) < 0.05) &
            (torch.abs(torch.norm(env.scene["cube_1"].data.root_pos_w[:, 2:] - env.scene["cube_2"].data.root_pos_w[:, 2:], dim=1) - 0.0468) < 0.005) &
            # cube_2とcube_3の条件
            (torch.norm(env.scene["cube_3"].data.root_pos_w[:, :2] - env.scene["cube_2"].data.root_pos_w[:, :2], dim=1) < 0.05) &
            (torch.abs(torch.norm(env.scene["cube_2"].data.root_pos_w[:, 2:] - env.scene["cube_3"].data.root_pos_w[:, 2:], dim=1) - 0.0468) < 0.005)
        ).float() * 200.0,  # タスク完了の巨大ボーナス
        weight=1.0,
        params={},
    )

    # エンドエフェクタとキューブの距離報酬（操作誘導）
    ee_to_cubes_distance = RewardTermCfg(
        func=lambda env: -torch.min(torch.stack([
            torch.norm(env.scene["robot"].data.body_state_w[:, env.scene["robot"].find_bodies("gripper_link")[0][0], :3] - 
                      env.scene["cube_1"].data.root_pos_w[:, :3], dim=1),
            torch.norm(env.scene["robot"].data.body_state_w[:, env.scene["robot"].find_bodies("gripper_link")[0][0], :3] - 
                      env.scene["cube_2"].data.root_pos_w[:, :3], dim=1),
            torch.norm(env.scene["robot"].data.body_state_w[:, env.scene["robot"].find_bodies("gripper_link")[0][0], :3] - 
                      env.scene["cube_3"].data.root_pos_w[:, :3], dim=1),
        ], dim=1), dim=1)[0],
        weight=1.0,
        params={},
    )

    # 制約: 動作の滑らかさ
    action_rate = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    # 制約: 関節速度を抑制
    joint_vel = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class LeRobotSO101StackCubeTerminationCfg:
    """｛LeRobot の SO-ARM101 ロボット x キューブ積み重ね｝環境の終了条件定義
    Isaac Lab標準のcubes_stacked関数と同じロジックを採用
    """

    # 時間切れ
    time_out = TerminationTermCfg(
        func=mdp.time_out,
        time_out=True
    )

    # キューブが地面に落ちた場合の終了条件（Isaac Lab標準）
    cube_1_dropping = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_1")}
    )

    cube_2_dropping = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_2")}
    )

    cube_3_dropping = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_3")}
    )

    # 成功条件: Isaac Lab標準のcubes_stackedと同じロジック
    stacking_success = TerminationTermCfg(
        func=lambda env: (
            # XY平面での位置チェック（5cm以内）
            (torch.norm(env.scene["cube_1"].data.root_pos_w[:, :2] - env.scene["cube_2"].data.root_pos_w[:, :2], dim=1) < 0.05) &
            (torch.norm(env.scene["cube_2"].data.root_pos_w[:, :2] - env.scene["cube_3"].data.root_pos_w[:, :2], dim=1) < 0.05) &
            # 高さ差チェック（0.0468m ± 0.005m）
            (torch.abs(torch.norm(env.scene["cube_1"].data.root_pos_w[:, 2:] - env.scene["cube_2"].data.root_pos_w[:, 2:], dim=1) - 0.0468) < 0.005) &
            (torch.abs(torch.norm(env.scene["cube_2"].data.root_pos_w[:, 2:] - env.scene["cube_3"].data.root_pos_w[:, 2:], dim=1) - 0.0468) < 0.005)
        ),
        params={},
    )


@configclass
class LeRobotSO101StackCubeCommandCfg:
    """｛LeRobot の SO-ARM101 ロボット x キューブ積み重ね｝環境のコマンド定義"""

    # キューブ1の位置コマンド
    cube_1_position = mdp.UniformPoseCommandCfg(
        asset_name="cube_1",
        body_name="Cube",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.2, 0.3),
            pos_y=(-0.1, 0.1),
            pos_z=(0.02, 0.05),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )

    # キューブ2の位置コマンド
    cube_2_position = mdp.UniformPoseCommandCfg(
        asset_name="cube_2",
        body_name="Cube",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.25, 0.35),
            pos_y=(-0.15, 0.15),
            pos_z=(0.02, 0.05),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )

    # キューブ3の位置コマンド
    cube_3_position = mdp.UniformPoseCommandCfg(
        asset_name="cube_3",
        body_name="Cube",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.22, 0.32),
            pos_y=(-0.12, 0.12),
            pos_z=(0.02, 0.05),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class LeRobotSO101StackCubeEventCfg:
    """
    ｛LeRobot の SO-ARM101 ロボット x キューブ積み重ね｝環境のイベント設定
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

    # キューブ1の位置をランダム化
    reset_cube_1_positions = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.05, 0.06), "y": (-0.05, 0.05), "z": (0.02, 0.05)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_1"),
        },
    )

    # キューブ2の位置をランダム化
    reset_cube_2_positions = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.1, 0.15), "y": (-0.05, 0.05), "z": (0.02, 0.05)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_2"),
        },
    )

    # キューブ3の位置をランダム化
    reset_cube_3_positions = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.09, 0.06), "y": (-0.05, 0.05), "z": (0.02, 0.05)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_3"),
        },
    )


@configclass
class LeRobotSO101StackCubeRLEnvCfg(ManagerBasedRLEnvCfg):
    """｛LeRobot の SO-ARM101 ロボット x キューブ積み重ね｝強化学習環境の設定クラス"""
    # ロボットUSDファイルパス
    robot_usd_path: str = "../assets/so101_new_calib_fix_articulation_root.usd"

    # シーン設定
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    # observation 設定
    observations: LeRobotSO101StackCubeObservationCfg = LeRobotSO101StackCubeObservationCfg()
    # action 設定
    actions: LeRobotSO101StackCubeActionCfg = LeRobotSO101StackCubeActionCfg()
    # event 設定
    events: LeRobotSO101StackCubeEventCfg = LeRobotSO101StackCubeEventCfg()
    # ManagerBasedEnv -> ManagerBasedRLEnvCfg で追加される設定
    # 報酬設定
    rewards: LeRobotSO101StackCubeRewardCfg = LeRobotSO101StackCubeRewardCfg()
    # 終了条件設定
    terminations: LeRobotSO101StackCubeTerminationCfg = LeRobotSO101StackCubeTerminationCfg()
    # コマンド設定
    commands: LeRobotSO101StackCubeCommandCfg = LeRobotSO101StackCubeCommandCfg()

    def use_teleop_device(self, device: str):
        """Teleoperation デバイス設定を適用"""
        if device.lower() == "keyboard":
            # キーボード用設定
            self.scene.num_envs = 1
            self.episode_length_s = 60.0
            # タイムアウト終了を無効化（teleoperation時は無制限に実行）
            self.terminations.time_out = None
            
            # 注意: 関節角度制御環境では7次元アクション（SE3+gripper）は使用できません
            print(f"[WARNING]: この環境は関節角度制御（6次元）です")
            print(f"[WARNING]: 標準teloperationスクリプト（7次元SE3+gripper）とは互換性がありません")
            print(f"[INFO]: IK対応環境 LeRobot-SO101-StackCube-IK-Rel-v0 の使用を推奨します")
            
        elif device.lower() == "spacemouse":
            # SpaceMouse用設定  
            self.scene.num_envs = 1
            self.episode_length_s = 60.0
            self.terminations.time_out = None
            
            # 同様の警告
            print(f"[WARNING]: この環境は関節角度制御（6次元）です")
            print(f"[WARNING]: 標準teloperationスクリプト（7次元SE3+gripper）とは互換性がありません")
            print(f"[INFO]: IK対応環境 LeRobot-SO101-StackCube-IK-Rel-v0 の使用を推奨します")

    def __post_init__(self):
        """初期化後の設定"""
        # 基本設定
        self.decimation = 4
        self.episode_length_s = 60.0

        # シーン設定を更新
        self.scene.num_envs = 1
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
                usd_path=self.robot_usd_path,
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
                pos=(-0.25, 0.20, 0.5),
                rot=(0.8660, -0.3536, 0.3536, 0.0),
                convention="world"
            ),
        )

        # 視点カメラ設定
        self.viewer.eye = [1.2, 1.2, 1.5]
        self.viewer.lookat = [0.2, 0.0, 0.3]

        # シミュレーション設定
        self.sim.dt = 0.01  # 100Hz
